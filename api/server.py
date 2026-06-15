"""
Backend API integration layer for Humanists AI frontend.

Run:
    uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import base64
import os
import tempfile
import traceback
import uuid
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from database_manager import DatabaseManager
from interview_memory_manager import InterviewMemoryManager
from interview_orchestrator import (
    choose_categories_for_round,
    choose_follow_up_mode,
    generate_resume_specific_question,
    generate_skill_specific_question,
    initialize_session_state,
    personalize_question,
    update_session_memory,
)
from multimodal_evaluator import fuse_multimodal_evaluation
from question_retriever import QuestionRetriever, canonical_role, resolve_allowed_roles
from recommend_jobs import (
    EMBED_MODEL_NAME,
    JOB_EMBEDDINGS_FILE,
    JOBS_METADATA_FILE,
    build_output,
    extract_text_from_pdf,
    load_job_index,
    recommend_jobs_for_profile,
)
from resume_agent import extract_resume_data
from resume_registry import ResumeRegistry
from session_manager import MAX_INTERVIEWS_PER_SESSION, SessionManager
from answer_evaluator import evaluate_answer
from adaptive_difficulty import adjust_difficulty, normalize_difficulty

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import noisereduce as nr
except Exception:
    nr = None

try:
    from CV.cv.pipeline import VisionPipeline
except Exception:
    VisionPipeline = None


app = FastAPI(title="Humanists AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


db = DatabaseManager()
resume_registry = ResumeRegistry(db=db)
session_manager = SessionManager(db=db)
memory_manager = InterviewMemoryManager(db=db)
retriever = QuestionRetriever()
job_embeddings, jobs_metadata = load_job_index(JOB_EMBEDDINGS_FILE, JOBS_METADATA_FILE)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
whisper_model = WhisperModel("small", compute_type="int8") if WhisperModel else None
vision_pipeline = None
if VisionPipeline and cv2 is not None:
    # Same fallback behavior used during CV streamlit testing:
    # if env vars are missing, auto-pick local .task files from CV/models.
    face_model_env = os.getenv("MEDIAPIPE_FACE_MODEL")
    pose_model_env = os.getenv("MEDIAPIPE_POSE_MODEL")
    face_model_fallback = os.path.abspath("CV/models/face_landmarker.task")
    pose_model_fallback = os.path.abspath("CV/models/pose_landmarker.task")
    face_model_path = face_model_env or (face_model_fallback if os.path.exists(face_model_fallback) else None)
    pose_model_path = pose_model_env or (pose_model_fallback if os.path.exists(pose_model_fallback) else None)

    print("FACE MODEL:", face_model_path)
    print("POSE MODEL:", pose_model_path)
    try:
        vision_pipeline = VisionPipeline(
            face_model_path=face_model_path,
            pose_model_path=pose_model_path,
        )
        print("vision_pipeline =", vision_pipeline)
    except Exception as e:
        print("\n=== VISION PIPELINE INIT FAILED ===")
        traceback.print_exc()
        print("===================================\n")
        vision_pipeline = None

print("cv2:", cv2)
print("np:", np)
print("VisionPipeline:", VisionPipeline)
print("whisper_model:", whisper_model)
print("vision_pipeline:", vision_pipeline)


def _audio_conf_from_segments(segs: List[Any]) -> float:
    logs = [getattr(s, "avg_logprob", None) for s in segs]
    logs = [x for x in logs if x is not None]
    if not logs:
        return 0.0
    avg_logprob = float(sum(logs) / len(logs))
    return max(0.0, min(100.0, (avg_logprob + 2.5) / 2.5 * 100.0))


def _audio_no_speech_prob(segs: List[Any]) -> float:
    vals = [getattr(s, "no_speech_prob", None) for s in segs]
    vals = [x for x in vals if x is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _is_poor_transcript(text: str, conf: float, no_speech_prob: float) -> bool:
    words = text.split()
    if len(words) < 3:
        return True
    if conf < 38:
        return True
    if no_speech_prob > 0.75:
        return True
    if words:
        uniq_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
        if len(words) >= 8 and uniq_ratio < 0.35:
            return True
    return False


def _trim_trailing_silence_for_pause_metric(
    audio: np.ndarray,
    sample_rate: int,
    energy_threshold: float = 0.008,
    max_trim_seconds: float = 4.0,
) -> np.ndarray:
    win_samples = int(0.2 * sample_rate)
    if audio.size == 0 or win_samples <= 0:
        return audio
    max_trim_windows = max(1, int((max_trim_seconds * sample_rate) / win_samples))
    rms_vals = []
    for i in range(0, len(audio) - win_samples + 1, win_samples):
        seg = audio[i:i + win_samples]
        rms_vals.append(float(np.sqrt(np.mean(np.square(seg)))))
    if not rms_vals:
        return audio
    trailing_quiet = 0
    for v in reversed(rms_vals):
        if v < energy_threshold and trailing_quiet < max_trim_windows:
            trailing_quiet += 1
        else:
            break
    if trailing_quiet == 0:
        return audio
    trim_samples = trailing_quiet * win_samples
    if trim_samples >= len(audio):
        return audio
    return audio[:-trim_samples]


def _estimate_audio_features(audio: np.ndarray, sample_rate: int, text: str) -> Dict[str, float]:
    duration_sec = max(1e-6, float(len(audio)) / float(sample_rate))
    words = text.split()
    wpm = (len(words) / duration_sec) * 60.0

    audio_for_pause = _trim_trailing_silence_for_pause_metric(audio, sample_rate=sample_rate)
    win = int(0.2 * sample_rate)
    if win <= 0 or len(audio_for_pause) < win:
        pause_ratio = 0.0
    else:
        vals = []
        for i in range(0, len(audio_for_pause) - win + 1, win):
            seg = audio_for_pause[i:i + win]
            vals.append(float(np.sqrt(np.mean(np.square(seg)))))
        quiet = sum(1 for v in vals if v < 0.008)
        pause_ratio = quiet / max(1, len(vals))

    return {
        "pause_ratio": round(float(pause_ratio), 4),
        "wpm": round(float(wpm), 2),
        "filler_density": 0.0,
        "pitch_variation": 0.0,
        "speech_consistency": 0.0,
    }


def _communication_score(transcription_confidence: float, features: Dict[str, float]) -> float:
    pace = float(features.get("wpm", 0.0))
    pace_score = 1.0 - min(1.0, abs(pace - 135.0) / 135.0) if pace > 0 else 0.0
    pause_score = 1.0 - min(1.0, abs(float(features.get("pause_ratio", 0.0)) - 0.22))
    comm = (0.55 * (float(transcription_confidence) / 100.0)) + (0.25 * pace_score) + (0.20 * pause_score)
    return max(0.0, min(100.0, comm * 100.0))


# in-memory active interview runs (frontend session runtime)
ACTIVE_INTERVIEWS: Dict[str, Dict[str, Any]] = {}


class SessionSelectRequest(BaseModel):
    resume_id: int
    mode: str  # "new" | "continue"
    session_id: Optional[int] = None
    session_name: Optional[str] = None


class StartInterviewRequest(BaseModel):
    session_id: int
    selected_role: str
    questions: int = 5
    coaching_mode: bool = False


class SubmitAnswerRequest(BaseModel):
    interview_id: str
    answer_text: str
    transcript: Optional[str] = ""
    communication_score: Optional[float] = None
    transcription_confidence: Optional[float] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    cv_analysis: Optional[Dict[str, Any]] = None


class EndInterviewRequest(BaseModel):
    interview_id: str
    reason: Optional[str] = "candidate_ended"


class DeleteSessionRequest(BaseModel):
    session_id: int


def _pick_next_question(runtime: Dict[str, Any]) -> Dict[str, Any]:
    if not runtime.get("intro_done", False):
        intro_q = (
            "Hi my name is H.I.A, I will be your interview training partner for today. "
            "Lets start the interview by getting a brief introduction about yourself."
        )
        runtime["current_question"] = {
            "question_type": "intro",
            "question_item": {
                "question": intro_q,
                "ideal_answer": "Brief background, skills, and goals.",
                "category": "Introduction",
                "role": "",
                "keywords": ["introduction"],
                "difficulty": runtime["current_difficulty"],
                "source_type": "System-Intro",
            },
            "asked_question": intro_q,
            "allowed_roles": [canonical_role(runtime["selected_role"])],
            "question_embedding": None,
            "difficulty": runtime["current_difficulty"],
            "follow_up_mode": "none",
        }
        return runtime["current_question"]

    # current_round includes intro round, so content plan index is shifted by 1
    q_idx = max(0, int(runtime["current_round"]) - 1)
    total_questions = int(runtime["total_questions"])
    profile = runtime["structured_profile"]
    selected_role = runtime["selected_role"]
    session_state = runtime["session_state"]
    history = runtime["history"]
    current_difficulty = runtime["current_difficulty"]
    question_plan = runtime["question_plan"]

    question_type = question_plan[q_idx] if q_idx < len(question_plan) else "strict_role"
    follow_up_mode = choose_follow_up_mode(history)
    preferred_categories = choose_categories_for_round(
        session_state=session_state,
        profile=profile,
        round_idx=q_idx,
        follow_up_mode=follow_up_mode,
    )

    context_tags = [
        f"round_{q_idx+1}",
        f"mode_{follow_up_mode}",
        f"prev_score_{history[-1]['evaluation']['score']}" if history else "first_question",
    ]
    question_embedding = None

    if question_type in {"strict_role", "cross_role"}:
        if question_type == "strict_role":
            allowed_roles = [canonical_role(selected_role)]
            role_match_mode = "exact"
        else:
            allowed_roles = resolve_allowed_roles(selected_role)
            role_match_mode = "fuzzy"

        retrieved = retriever.retrieve_questions(
            target_role=selected_role,
            difficulty=current_difficulty,
            candidate_skills=profile.get("skills", []),
            candidate_weak_areas=(profile.get("weak_areas", []) + session_state.get("prior_weak_areas", [])),
            context_tags=context_tags,
            categories=preferred_categories,
            allowed_roles=allowed_roles,
            session_state=session_state,
            top_k=5,
            duplicate_similarity_threshold=0.78,
            role_match_mode=role_match_mode,
        )
        if not retrieved:
            raise RuntimeError("No question retrieved for current state.")
        question_item = retrieved[0]
        question_embedding = retriever.embeddings[question_item["index"]]
        asked_question = personalize_question(question_item, profile, history)
    elif question_type == "resume_specific":
        allowed_roles = [canonical_role(selected_role)]
        question_item = generate_resume_specific_question(
            profile=profile,
            history=history,
            difficulty=current_difficulty,
            session_state=session_state,
        )
        asked_question = question_item.get("question", "")
    else:
        allowed_roles = [canonical_role(selected_role)]
        question_item = generate_skill_specific_question(
            profile=profile,
            selected_role=selected_role,
            difficulty=current_difficulty,
            history=history,
        )
        asked_question = question_item.get("question", "")

    runtime["current_question"] = {
        "question_type": question_type,
        "question_item": question_item,
        "asked_question": asked_question,
        "allowed_roles": allowed_roles,
        "question_embedding": question_embedding,
        "difficulty": current_difficulty,
        "follow_up_mode": follow_up_mode,
    }
    return runtime["current_question"]


def _finalize_interview_runtime(runtime: Dict[str, Any]) -> Dict[str, Any]:
    avg_score = 0.0
    if runtime["history"]:
        avg_score = sum(t["evaluation"].get("score", 0) for t in runtime["history"]) / len(runtime["history"])

    summary = {
        "selected_role": runtime["selected_role"],
        "total_questions": runtime["total_questions"],
        "completed_questions": len(runtime["history"]),
        "average_score": round(avg_score, 2),
        "final_difficulty": runtime["current_difficulty"],
        "used_categories": runtime["session_state"].get("used_categories", []),
        "interview_history": runtime["history"],
    }
    row = session_manager.add_interview_record(
        session_id=runtime["session_id"],
        target_role=runtime["selected_role"],
        interview_result=summary,
    )
    mem = memory_manager.build_compressed_memory(summary)
    memory_manager.upsert_memory(runtime["session_id"], mem)
    runtime["summary"] = summary
    runtime["final_interview_id"] = row["interview_id"]
    return {"summary": summary, "interview_db_id": row["interview_id"]}


@app.post("/resume/upload")
async def upload_resume(
    resume: UploadFile = File(...),
    top_k: int = Form(5),
):
    suffix = os.path.splitext(resume.filename or "resume.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await resume.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        resume_text = extract_text_from_pdf(tmp_path)
        structured_profile = extract_resume_data(resume_text)
        if "error" in structured_profile:
            raise HTTPException(status_code=500, detail=f"Resume extraction failed: {structured_profile['error']}")

        rec = recommend_jobs_for_profile(
            profile=structured_profile,
            model=embed_model,
            job_embeddings=job_embeddings,
            jobs_metadata=jobs_metadata,
            top_k=top_k,
        )
        recommendation_output = build_output(structured_profile, rec)

        reg = resume_registry.register_or_get_resume(tmp_path, structured_profile)
        resume_row = reg["resume"]
        sessions = session_manager.list_sessions(int(resume_row["resume_id"]))
        active_sessions = [s for s in sessions if int(s.get("total_interviews", 0)) < MAX_INTERVIEWS_PER_SESSION]

        return {
            "resume_id": resume_row["resume_id"],
            "is_new_resume": reg["is_new"],
            "sessions": active_sessions,
            **recommendation_output,
            "structured_profile": structured_profile,
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/upload_resume")
async def upload_resume_alias(
    resume: UploadFile = File(...),
    top_k: int = Form(5),
):
    return await upload_resume(resume=resume, top_k=top_k)


@app.post("/analyze_resume")
def analyze_resume(payload: Dict[str, Any]):
    resume_id = int(payload.get("resume_id", 0))
    if not resume_id:
        raise HTTPException(status_code=400, detail="resume_id is required")
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM resumes WHERE resume_id = ?", (resume_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Resume not found")
    profile = db.loads(row["semantic_profile_json"], {})
    return {"resume_id": resume_id, "structured_profile": profile}


@app.get("/recommended_roles")
def recommended_roles(resume_id: int):
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM resumes WHERE resume_id = ?", (resume_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Resume not found")
    profile = db.loads(row["semantic_profile_json"], {})
    rec = recommend_jobs_for_profile(
        profile=profile,
        model=embed_model,
        job_embeddings=job_embeddings,
        jobs_metadata=jobs_metadata,
        top_k=5,
    )
    return {"recommended_jobs": rec}


@app.post("/transcribe_audio")
async def transcribe_audio(audio: UploadFile = File(...)):
    if whisper_model is None or sf is None or np is None:
        raise HTTPException(status_code=503, detail="Whisper model is unavailable")

    suffix = os.path.splitext(audio.filename or "answer.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    try:
        # Transcribe directly from uploaded file path first (works for browser webm).
        seg1, _ = whisper_model.transcribe(audio_path, beam_size=5, language="en", vad_filter=False)
        s1 = list(seg1)
        txt1 = " ".join((s.text or "").strip() for s in s1).strip()
        c1 = _audio_conf_from_segments(s1)
        n1 = _audio_no_speech_prob(s1)

        chosen = {
            "text": txt1,
            "conf": c1,
            "nsp": n1,
            "variant": "raw_no_vad",
        }

        if _is_poor_transcript(txt1, c1, n1):
            seg2, _ = whisper_model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            s2 = list(seg2)
            txt2 = " ".join((s.text or "").strip() for s in s2).strip()
            c2 = _audio_conf_from_segments(s2)
            n2 = _audio_no_speech_prob(s2)
            if not _is_poor_transcript(txt2, c2, n2):
                chosen = {"text": txt2, "conf": c2, "nsp": n2, "variant": "raw_vad"}

        # Optional waveform metrics: only if libsndfile can decode this container.
        features = {
            "pause_ratio": 0.0,
            "wpm": 0.0,
            "filler_density": 0.0,
            "pitch_variation": 0.0,
            "speech_consistency": 0.0,
        }
        try:
            raw_audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if isinstance(raw_audio, np.ndarray) and raw_audio.ndim > 1:
                raw_audio = np.mean(raw_audio, axis=1)
            raw_audio = np.asarray(raw_audio, dtype=np.float32)

            target_sr = 16000
            if int(sr) != target_sr and raw_audio.size > 0:
                old_idx = np.linspace(0.0, 1.0, num=len(raw_audio), endpoint=True)
                new_len = int(len(raw_audio) * (target_sr / float(sr)))
                new_idx = np.linspace(0.0, 1.0, num=max(1, new_len), endpoint=True)
                raw_audio = np.interp(new_idx, old_idx, raw_audio).astype(np.float32)
                sr = target_sr

            # Keep cleaned-feature path when decode works.
            analysis_audio = raw_audio
            if nr is not None:
                try:
                    analysis_audio = nr.reduce_noise(y=analysis_audio, sr=sr, prop_decrease=0.4).astype(np.float32)
                except Exception:
                    analysis_audio = raw_audio
            features = _estimate_audio_features(analysis_audio, sample_rate=sr, text=chosen["text"])
        except Exception as decode_exc:
            print("[TRANSCRIBE] waveform decode unavailable, using transcript-only metrics:", decode_exc)

        communication_score = _communication_score(chosen["conf"], features)

        return {
            "transcript": chosen["text"],
            "answer_text": chosen["text"],
            "transcription_confidence": round(float(chosen["conf"]), 2),
            "no_speech_prob": round(float(chosen["nsp"]), 4),
            "communication_score": round(float(communication_score), 2),
            "audio_analysis": {
                "transcription_confidence": round(float(chosen["conf"]), 2),
                "no_speech_prob": round(float(chosen["nsp"]), 4),
                "transcription_variant": chosen["variant"],
                "pause_ratio": features["pause_ratio"],
                "filler_density": features["filler_density"],
                "speech_consistency": features["speech_consistency"],
                "wpm": features["wpm"],
                "pitch_variation": features["pitch_variation"],
                "communication_score": round(float(communication_score), 2),
            },
        }
    except Exception as exc:
        print("\n========== TRANSCRIBE ERROR ==========")
        traceback.print_exc()
        print("======================================\n")
        # Return a safe payload instead of 500 so interview flow can continue.
        return {
            "transcript": "",
            "answer_text": "",
            "transcription_confidence": 0.0,
            "no_speech_prob": 1.0,
            "communication_score": 0.0,
            "audio_analysis": {
                "transcription_confidence": 0.0,
                "no_speech_prob": 1.0,
                "transcription_variant": "error_fallback",
                "pause_ratio": 0.0,
                "filler_density": 0.0,
                "speech_consistency": 0.0,
                "wpm": 0.0,
                "pitch_variation": 0.0,
                "communication_score": 0.0,
            },
            "error": str(exc),
        }
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass


@app.post("/analyze_video_frame")
def analyze_video_frame(payload: Dict[str, Any]):
    try:
        print("\n========== CV REQUEST ==========")
        print("vision_pipeline:", vision_pipeline)

        if vision_pipeline is None or cv2 is None or np is None:
            print("VISION PIPELINE IS NONE OR DEPENDENCY MISSING")
            return {
                "available": False,
                "face_detected": False,
                "cv_analysis": {},
                "error": "vision_pipeline_none",
            }

        image_data = str(payload.get("image", ""))
        if not image_data:
            print("NO IMAGE DATA")
            return {
                "available": True,
                "face_detected": False,
                "cv_analysis": {},
                "error": "missing_image",
            }

        print("IMAGE LENGTH:", len(image_data))
        timestamp = float(payload.get("timestamp", 0.0) or 0.0)
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        raw = base64.b64decode(image_data)
        print("BYTES LENGTH:", len(raw))
        arr = np.frombuffer(raw, np.uint8)
        print("NP ARRAY:", getattr(arr, "shape", None))
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("FRAME DECODE FAILED")
            return {"available": True, "face_detected": False, "cv_analysis": {}, "error": "frame_decode_failed"}

        # Debug signal for incoming frames
        print("FRAME RECEIVED:", getattr(frame, "shape", None))

        metrics = vision_pipeline.analyze_frame(frame, timestamp)
        if metrics is None:
            print("NO FACE DETECTED (metrics=None)")
            return {"available": True, "face_detected": False, "cv_analysis": {}}

        print("CV METRICS:", metrics)
        face_detected = bool(getattr(metrics, "raw", {}).get("face_detected", True))

        eye_contact = float(metrics.eye_contact_score or 0.0)
        engagement = float(metrics.engagement_score or 0.0)
        attention = float(metrics.head_pose_score or 0.0)
        blink_rate = float(metrics.blink_rate or 0.0)
        perclos = float(metrics.perclos or 0.0)
        distraction = float(max(0.0, 1.0 - eye_contact))

        return {
            "available": True,
            "face_detected": face_detected,
            "cv_analysis": {
                "engagement_score": round(engagement * 10.0, 2),
                "eye_contact_score": round(eye_contact * 10.0, 2),
                "attention_score": round(attention * 10.0, 2),
                "blink_rate": round(blink_rate, 2),
                "perclos": round(perclos, 4),
                "distraction_score": round(distraction * 10.0, 2),
                "face_detected": face_detected,
            },
        }
    except Exception as exc:
        print("\n========== CV ERROR ==========")
        traceback.print_exc()
        print("================================\n")
        return {"available": True, "face_detected": False, "cv_analysis": {}, "error": str(exc)}


@app.post("/session/select")
def select_session(req: SessionSelectRequest):
    if req.mode == "new":
        created = session_manager.create_session(req.resume_id, session_name=req.session_name)
        return {"session": created, "created": True}

    if req.mode == "continue":
        if not req.session_id:
            raise HTTPException(status_code=400, detail="session_id required for continuation mode")
        existing = session_manager.get_session(req.session_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")
        if not session_manager.can_add_interview(req.session_id):
            raise HTTPException(status_code=400, detail="Session already has 3 interviews. Start a new session.")
        return {"session": existing, "created": False}

    raise HTTPException(status_code=400, detail="mode must be 'new' or 'continue'")


@app.post("/start_session")
def start_session_alias(req: SessionSelectRequest):
    return select_session(req)


@app.post("/interview/start")
def start_interview(req: StartInterviewRequest):
    session = session_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session_manager.can_add_interview(req.session_id):
        raise HTTPException(status_code=400, detail="Session already has 3 interviews. Start new session.")

    with db.connect() as conn:
        resume_row = conn.execute(
            """
            SELECT r.* FROM resumes r
            JOIN sessions s ON s.resume_id = r.resume_id
            WHERE s.session_id = ?
            """,
            (req.session_id,),
        ).fetchone()
    if not resume_row:
        raise HTTPException(status_code=404, detail="Resume not found for session")

    structured_profile = db.loads(resume_row["semantic_profile_json"], {})
    prior_memory = memory_manager.get_latest_memory(req.session_id) or {}
    session_state = initialize_session_state(prior_memory=prior_memory)
    for q in prior_memory.get("asked_questions", []):
        if q:
            session_state["already_asked_questions"].append(q)
    for emb in prior_memory.get("asked_question_embeddings", []):
        if isinstance(emb, list) and emb:
            session_state["already_asked_embeddings"].append(emb)

    from interview_orchestrator import build_question_type_plan

    runtime = {
        "session_id": req.session_id,
        "structured_profile": structured_profile,
        "selected_role": req.selected_role,
        "total_questions": max(1, int(req.questions)) + 1,  # +1 guaranteed intro
        "coaching_mode": req.coaching_mode,
        "current_round": 0,
        "current_difficulty": normalize_difficulty(structured_profile.get("recommended_question_difficulty", "Medium")),
        "question_plan": build_question_type_plan(max(1, int(req.questions))),
        "intro_done": False,
        "session_state": session_state,
        "history": [],
    }
    q = _pick_next_question(runtime)
    interview_id = str(uuid.uuid4())
    ACTIVE_INTERVIEWS[interview_id] = runtime

    return {
        "interview_id": interview_id,
        "session_id": req.session_id,
        "round": 1,
        "total_questions": runtime["total_questions"],
        "question": q["asked_question"],
        "question_type": q["question_type"],
        "difficulty": q["difficulty"],
    }


@app.post("/start_interview")
def start_interview_alias(req: StartInterviewRequest):
    return start_interview(req)


@app.post("/next_question")
def next_question(payload: Dict[str, Any]):
    interview_id = payload.get("interview_id", "")
    runtime = ACTIVE_INTERVIEWS.get(interview_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Interview not found")
    cq = runtime.get("current_question")
    if not cq:
        cq = _pick_next_question(runtime)
    return {
        "round": runtime["current_round"] + 1,
        "question": cq["asked_question"],
        "question_type": cq["question_type"],
        "difficulty": cq["difficulty"],
    }


@app.post("/interview/answer")
def submit_answer(req: SubmitAnswerRequest):
    runtime = ACTIVE_INTERVIEWS.get(req.interview_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Interview not found")

    cq = runtime["current_question"]
    question_item = cq["question_item"]
    asked_question = cq["asked_question"]
    difficulty = cq["difficulty"]
    question_embedding = cq["question_embedding"]

    technical_eval = evaluate_answer(question_item, req.answer_text)
    audio_analysis = req.audio_analysis or {
        "transcription_confidence": float(req.transcription_confidence or 0.0),
        "pause_ratio": 0.0,
        "filler_density": 0.0,
        "speech_consistency": 0.0,
        "wpm": 0.0,
        "pitch_variation": 0.0,
    }
    multimodal_eval = fuse_multimodal_evaluation(
        question=asked_question,
        candidate_answer=req.answer_text,
        audio_analysis=audio_analysis,
        technical_evaluation=technical_eval,
        conversation_history=runtime["history"],
        cv_analysis=req.cv_analysis or {},
    )

    evaluation = {
        "score": multimodal_eval["overall_score"],
        "communication_score": multimodal_eval["communication_score"],
        "technical_score": multimodal_eval["technical_score"],
        "confidence_estimate": multimodal_eval["confidence_score"],
        "feedback": (
            multimodal_eval["coaching_feedback"]["technical_feedback"] + " "
            + multimodal_eval["coaching_feedback"]["communication_feedback"] + " "
            + multimodal_eval["coaching_feedback"]["behavioral_feedback"]
        ).strip(),
        "recommended_next_difficulty": multimodal_eval["recommended_next_difficulty"],
        "behavioral_signals": multimodal_eval["behavioral_signals"],
        "coaching_feedback": multimodal_eval["coaching_feedback"],
        "warnings": multimodal_eval.get("warnings", []) if runtime.get("coaching_mode") else [],
    }
    next_difficulty = adjust_difficulty(runtime["current_difficulty"], evaluation)

    update_session_memory(
        session_state=runtime["session_state"],
        asked_question=asked_question,
        question_embedding=question_embedding,
        question_item=question_item,
        difficulty=difficulty,
        candidate_answer=req.answer_text,
        evaluation=evaluation,
    )

    runtime["history"].append(
        {
            "round": runtime["current_round"] + 1,
            "retrieved_question": question_item.get("question", ""),
            "asked_question": asked_question,
            "category": question_item.get("category", ""),
            "difficulty": difficulty,
            "candidate_answer": req.answer_text,
            "candidate_response": {
                "answer_text": req.answer_text,
                "transcript": req.transcript or req.answer_text,
                "audio_analysis": audio_analysis,
            },
            "evaluation": evaluation,
            "next_difficulty": next_difficulty,
            "follow_up_mode": cq["follow_up_mode"],
            "allowed_roles": cq["allowed_roles"],
            "question_type": cq["question_type"],
            "question_keywords": question_item.get("keywords", []),
        }
    )
    if cq["question_type"] == "intro":
        runtime["intro_done"] = True

    runtime["current_difficulty"] = next_difficulty
    runtime["current_round"] += 1

    done = runtime["current_round"] >= runtime["total_questions"]
    if done:
        finalized = _finalize_interview_runtime(runtime)
        return {"done": True, **finalized}

    q = _pick_next_question(runtime)
    return {
        "done": False,
        "round": runtime["current_round"] + 1,
        "question": q["asked_question"],
        "question_type": q["question_type"],
        "difficulty": q["difficulty"],
        "evaluation": evaluation,
    }


@app.post("/submit_answer")
def submit_answer_alias(req: SubmitAnswerRequest):
    return submit_answer(req)


@app.post("/interview/end")
def end_interview(req: EndInterviewRequest):
    runtime = ACTIVE_INTERVIEWS.get(req.interview_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Interview not found")
    finalized = _finalize_interview_runtime(runtime)
    return {
        "done": True,
        "ended_early": True,
        "reason": req.reason or "candidate_ended",
        **finalized,
    }


@app.post("/end_interview")
def end_interview_alias(req: EndInterviewRequest):
    return end_interview(req)


@app.get("/interview/summary/{interview_id}")
def get_summary(interview_id: str):
    runtime = ACTIVE_INTERVIEWS.get(interview_id)
    if runtime and runtime.get("summary"):
        return runtime["summary"]
    raise HTTPException(status_code=404, detail="Summary not available yet")


@app.get("/interview_summary")
def interview_summary_alias(interview_id: str):
    return get_summary(interview_id)


@app.get("/sessions")
def list_sessions(resume_id: int):
    return {"sessions": session_manager.list_sessions(resume_id)}


@app.get("/session_interviews")
def list_session_interviews(session_id: int):
    return {"interviews": session_manager.list_interviews(session_id)}


@app.get("/interview_record/{interview_id}")
def interview_record_summary(interview_id: int):
    summary = session_manager.get_interview_summary(interview_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Interview summary not found")
    return {"summary": summary}


@app.delete("/session/{session_id}")
def delete_session(session_id: int):
    result = session_manager.delete_session_cascade(session_id)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.post("/delete_session")
def delete_session_alias(req: DeleteSessionRequest):
    return delete_session(req.session_id)


@app.get("/resume_list")
def list_resumes():
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT resume_id, original_resume_name, candidate_name, upload_timestamp, experience_level
            FROM resumes ORDER BY upload_timestamp DESC
            """
        ).fetchall()
    return {"resumes": [dict(r) for r in rows]}


@app.get("/resume/{resume_id}")
def get_resume(resume_id: int):
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM resumes WHERE resume_id = ?", (resume_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Resume not found")
    r = dict(row)
    r["semantic_profile_json"] = db.loads(r.get("semantic_profile_json"), {})
    return r
