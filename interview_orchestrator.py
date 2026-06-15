# =========================================================
# interview_orchestrator.py
# =========================================================
# PURPOSE:
# Main adaptive interview runtime engine.
# Includes session memory, category rotation, follow-up behavior,
# and multimodal (audio + technical) evaluation.
# =========================================================

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from typing import Dict, List

import pdfplumber
from openai import OpenAI

from adaptive_difficulty import adjust_difficulty, normalize_difficulty
from answer_evaluator import evaluate_answer
from audio_adapter import AudioResponseAdapter
from audio_live_adapter import LiveAudioResponseAdapter
from cv_adapter import CVResponseAdapter
from multimodal_evaluator import fuse_multimodal_evaluation
from question_retriever import QuestionRetriever, resolve_allowed_roles, canonical_role
from resume_agent import extract_resume_data

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "deepseek/deepseek-chat"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]


def extract_text_from_pdf(pdf_path: str) -> str:
    page_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = clean_text(page.extract_text() or "")
            if page_text:
                page_chunks.append(page_text)

    full_text = clean_text("\n".join(page_chunks))
    if not full_text:
        raise ValueError("No readable text found in PDF.")
    return full_text


def maybe_speak_question(question_text: str, enable_tts: bool = False, tts_voice: str = "Samantha"):
    if not enable_tts:
        return
    import sys
    try:
        if sys.platform == "darwin":
            # macOS native say command
            subprocess.run(["say", "-v", tts_voice, question_text], check=False)
        elif sys.platform == "win32":
            # Windows native SpeechSynthesizer via PowerShell
            escaped_text = question_text.replace('"', '\\"')
            cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{escaped_text}")'
            subprocess.run(["powershell", "-Command", cmd], check=False)
        else:
            # Linux fallback via espeak
            subprocess.run(["espeak", question_text], check=False)
    except Exception:
        pass


def personalize_question(question_item: Dict, profile: Dict, history: List[Dict]) -> str:
    base_question = clean_text(question_item.get("question", ""))
    role = clean_text(question_item.get("role", ""))
    difficulty = clean_text(question_item.get("difficulty", "Medium"))
    category = clean_text(question_item.get("category", ""))

    profile_skills = ", ".join(safe_list(profile.get("skills", []))[:8])
    weak_areas = ", ".join(safe_list(profile.get("weak_areas", []))[:6])
    summary = clean_text(profile.get("resume_summary", ""))[:500]

    recent_context = []
    for turn in history[-2:]:
        recent_context.append(
            f"Q: {clean_text(turn.get('asked_question',''))} | "
            f"Score: {turn.get('evaluation', {}).get('score', 'NA')}"
        )

    prompt = f"""
You are an adaptive AI interviewer.
Personalize the retrieved question for the candidate WITHOUT changing its intent.

Return only one question sentence/paragraph.

Grounding Data:
- Base Retrieved Question: {base_question}
- Role: {role}
- Category: {category}
- Difficulty: {difficulty}
- Candidate Skills: {profile_skills}
- Candidate Weak Areas: {weak_areas}
- Resume Summary: {summary}
- Recent Interview Context: {' || '.join(recent_context)}

Rules:
- Keep question concise.
- Keep role relevance.
- If possible, reference candidate projects/skills.
- Do not invent certifications or companies not present in profile.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a focused interview question personalizer."},
                {"role": "user", "content": prompt},
            ],
        )
        personalized = clean_text(response.choices[0].message.content)
        personalized = personalized.strip().strip('"').strip("'")
        personalized = re.sub(r"^\*+|\*+$", "", personalized).strip()
        personalized = re.sub(r"\s*\(.*?keeps intent.*?\)\s*$", "", personalized, flags=re.IGNORECASE)
        return personalized or base_question
    except Exception:
        return base_question


def get_candidate_response(
    prompt_text: str,
    input_mode: str = "terminal",
    round_index: int = 0,
    audio_adapter: AudioResponseAdapter = None,
    live_audio_adapter: LiveAudioResponseAdapter = None,
    recording_seconds: int = 90,
    enable_tts: bool = False,
    tts_voice: str = "Samantha",
) -> Dict:
    print("\nINTERVIEWER:")
    print(prompt_text)

    if input_mode == "audio_json":
        if audio_adapter is None:
            raise ValueError("audio_json mode requires an AudioResponseAdapter instance")
        response = audio_adapter.get_response(round_index)
        print("\nCANDIDATE (from audio_json transcript)>", response.get("answer_text", ""))
        print("[Audio] transcription_confidence:", response.get("audio_analysis", {}).get("transcription_confidence", 0.0))
        return response

    if input_mode == "audio_live":
        if live_audio_adapter is None:
            raise ValueError("audio_live mode requires a LiveAudioResponseAdapter instance")
        # In live audio mode, always dictate first, then turn mic on.
        # If caller did not pass --enable_tts, we still dictate by default for this mode.
        maybe_speak_question(prompt_text, enable_tts=True if not enable_tts else enable_tts, tts_voice=tts_voice)
        response = live_audio_adapter.get_response(round_index, duration_sec=recording_seconds, silence_seconds=5.0)
        print("\nCANDIDATE (live transcript)>", response.get("answer_text", ""))
        print("[Audio] transcription_confidence:", response.get("transcription_confidence", 0.0))
        return response

    answer = input("\nCANDIDATE> ").strip()
    return {
        "answer_text": answer,
        "transcript": answer,
        "communication_score": None,
        "confidence_score": None,
        "audio_features": {},
        "audio_analysis": {},
    }


def initialize_session_state(prior_memory: Dict = None) -> Dict:
    prior_memory = prior_memory or {}
    return {
        "already_asked_questions": [],
        "already_asked_embeddings": [],
        "used_categories": [],
        "difficulty_history": [],
        "conversation_context": [],
        "used_project_indices": [],
        "prior_covered_topics": safe_list(prior_memory.get("covered_topics", [])),
        "prior_weak_areas": safe_list(prior_memory.get("weak_areas", [])),
        "prior_strong_areas": safe_list(prior_memory.get("strong_areas", [])),
    }


def choose_categories_for_round(session_state: Dict, profile: Dict, round_idx: int, follow_up_mode: str) -> List[str]:
    used = [clean_text(c) for c in safe_list(session_state.get("used_categories", [])) if clean_text(c)]
    counts = Counter(c.lower() for c in used)

    if follow_up_mode != "none" and session_state.get("conversation_context"):
        last = session_state["conversation_context"][-1]
        last_category = clean_text(last.get("category", ""))
        return [last_category] if last_category else []

    weak_areas = [clean_text(w).lower() for w in safe_list(profile.get("weak_areas", [])) if clean_text(w)]
    weak_hint = []
    if any("communication" in w or "behavior" in w for w in weak_areas):
        weak_hint.append("Behavioral")
    if any("system design" in w or "architecture" in w for w in weak_areas):
        weak_hint.append("System Design")
    if any("problem" in w or "analysis" in w for w in weak_areas):
        weak_hint.append("Problem Solving")

    pool = ["Technical", "Behavioral", "Problem Solving", "Leadership", "System Design", "Career Goals", "Adaptability"]
    ranked = sorted(pool, key=lambda c: counts[c.lower()])
    ordered = weak_hint + [c for c in ranked if c not in weak_hint]
    return ordered[:2]


def choose_follow_up_mode(history: List[Dict]) -> str:
    if not history:
        return "none"

    prev_score = float(history[-1].get("evaluation", {}).get("score", 5))
    if prev_score >= 7:
        return "deepen"
    if prev_score <= 4:
        return "simplify"
    return "none"


def build_question_type_plan(total_questions: int) -> List[str]:
    ratio = {
        "cross_role": 0.15,
        "strict_role": 0.20,
        "resume_specific": 0.30,
        "skill_specific": 0.35,
    }
    counts = {k: int(total_questions * v) for k, v in ratio.items()}
    while sum(counts.values()) < total_questions:
        for k in ["skill_specific", "resume_specific", "strict_role", "cross_role"]:
            if sum(counts.values()) < total_questions:
                counts[k] += 1

    plan = []
    while len(plan) < total_questions:
        # Required progression order:
        # skill_specific -> strict_role -> resume_specific -> cross_role
        for k in ["skill_specific", "strict_role", "resume_specific", "cross_role"]:
            if counts[k] > 0 and len(plan) < total_questions:
                plan.append(k)
                counts[k] -= 1
    return plan


def generate_resume_specific_question(profile: Dict, history: List[Dict], difficulty: str, session_state: Dict = None) -> Dict:
    projects = safe_list(profile.get("projects", []))[:3]
    work_exp = safe_list(profile.get("work_experience", []))[:3]
    summary = clean_text(profile.get("resume_summary", ""))[:500]
    last_context = history[-1]["asked_question"] if history else ""
    prompt = f"""
You are an interviewer. Generate ONE resume-specific interview question.
Focus on projects/work experience from the candidate profile.
Difficulty: {difficulty}
Prior question context: {last_context}
Resume summary: {summary}
Projects: {json.dumps(projects, ensure_ascii=False)}
Work Experience: {json.dumps(work_exp, ensure_ascii=False)}

Return only JSON:
{{
  "question": "...",
  "ideal_answer": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You generate grounded resume-specific interview questions."},
                {"role": "user", "content": prompt},
            ],
        )
        text = clean_text(response.choices[0].message.content).replace("```json", "").replace("```", "")
        parsed = json.loads(text)
        question = clean_text(parsed.get("question", ""))
        if not question:
            raise ValueError("Empty question from model")
        return {
            "question": question,
            "ideal_answer": clean_text(parsed.get("ideal_answer", "")),
            "category": "Resume Deep Dive",
            "role": "",
            "keywords": [],
            "difficulty": difficulty,
            "source_type": "Generated-Resume",
        }
    except Exception:
        asked_lower = {
            clean_text(h.get("asked_question", "")).lower()
            for h in history
            if clean_text(h.get("asked_question", ""))
        }
        fallback_candidates = [
            "Walk me through one project from your resume and explain your exact contribution, technical choices, and impact.",
            "Pick one project from your resume and explain the hardest technical tradeoff you made and why.",
            "Choose one resume project and describe one failure you faced, how you debugged it, and the final outcome.",
            "From one project on your resume, explain how you measured success and what metrics improved.",
        ]
        fallback_q = next((q for q in fallback_candidates if q.lower() not in asked_lower), fallback_candidates[-1])
        return {
            "question": fallback_q,
            "ideal_answer": "Structured project explanation with ownership, decisions, metrics, and learning.",
            "category": "Resume Deep Dive",
            "role": "",
            "keywords": [],
            "difficulty": difficulty,
            "source_type": "Generated-Resume",
        }


def generate_skill_specific_question(profile: Dict, selected_role: str, difficulty: str, history: List[Dict] = None) -> Dict:
    skills = safe_list(profile.get("skills", []))[:10]
    prompt = f"""
Generate ONE role-relevant skill question for: {selected_role}
Candidate skills: {', '.join(skills)}
Difficulty: {difficulty}

Return only JSON:
{{
  "question": "...",
  "ideal_answer": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You generate concise skill-specific interview questions."},
                {"role": "user", "content": prompt},
            ],
        )
        text = clean_text(response.choices[0].message.content).replace("```json", "").replace("```", "")
        parsed = json.loads(text)
        question = clean_text(parsed.get("question", ""))
        if not question:
            raise ValueError("Empty question from model")
        return {
            "question": question,
            "ideal_answer": clean_text(parsed.get("ideal_answer", "")),
            "category": "Skill Specific",
            "role": selected_role,
            "keywords": skills[:5],
            "difficulty": difficulty,
            "source_type": "Generated-Skill",
        }
    except Exception:
        asked_lower = {
            clean_text(h.get("asked_question", "")).lower()
            for h in (history or [])
            if clean_text(h.get("asked_question", ""))
        }
        fallback_candidates = [
            f"Pick one skill from your resume and explain how you used it in a real {selected_role} project.",
            f"Choose one {selected_role} skill from your resume and walk through a debugging scenario where you applied it.",
            f"From your resume skills, explain one tool or framework you used for {selected_role}, the tradeoff, and impact.",
        ]
        fallback_q = next((q for q in fallback_candidates if q.lower() not in asked_lower), fallback_candidates[-1])
        return {
            "question": fallback_q,
            "ideal_answer": "Concrete skill usage, architecture/approach, tradeoffs, and impact.",
            "category": "Skill Specific",
            "role": selected_role,
            "keywords": skills[:5],
            "difficulty": difficulty,
            "source_type": "Generated-Skill",
        }


def update_session_memory(session_state: Dict, asked_question: str, question_embedding, question_item: Dict, difficulty: str, candidate_answer: str, evaluation: Dict):
    session_state["already_asked_questions"].append(clean_text(question_item.get("question", "")))
    if question_embedding is not None:
        session_state["already_asked_embeddings"].append(question_embedding.tolist())
    session_state["used_categories"].append(clean_text(question_item.get("category", "")))
    session_state["difficulty_history"].append(difficulty)
    session_state["conversation_context"].append(
        {
            "question": asked_question,
            "category": clean_text(question_item.get("category", "")),
            "topic_keywords": safe_list(question_item.get("keywords", []))[:5],
            "answer": clean_text(candidate_answer),
            "score": evaluation.get("score", 0),
        }
    )


def run_interview_session(
    structured_profile: Dict,
    selected_role: str,
    total_questions: int = 5,
    input_mode: str = "terminal",
    audio_json_path: str = "audio_pipeline_results.json",
    recording_seconds: int = 90,
    enable_tts: bool = False,
    tts_voice: str = "Samantha",
    cv_mode: str = "off",
    cv_jsonl_path: str = "CV/cv_metrics.jsonl",
    prior_memory: Dict = None,
    coaching_mode: bool = False,
) -> Dict:
    retriever = QuestionRetriever()

    current_difficulty = normalize_difficulty(structured_profile.get("recommended_question_difficulty", "Medium"))

    audio_adapter = None
    live_audio_adapter = None
    cv_adapter = None
    if input_mode == "audio_json":
        audio_adapter = AudioResponseAdapter(audio_json_path)
    elif input_mode == "audio_live":
        live_audio_adapter = LiveAudioResponseAdapter()
        # Keep calibration optional; forced calibration degraded transcripts on some setups.
    if cv_mode == "jsonl":
        cv_adapter = CVResponseAdapter(cv_jsonl_path)

    session_state = initialize_session_state(prior_memory=prior_memory)
    for q in safe_list((prior_memory or {}).get("asked_questions", [])):
        q = clean_text(q)
        if q:
            session_state["already_asked_questions"].append(q)
    for emb in safe_list((prior_memory or {}).get("asked_question_embeddings", [])):
        if isinstance(emb, list) and emb:
            session_state["already_asked_embeddings"].append(emb)
    history = []
    question_plan = build_question_type_plan(total_questions)

    for q_idx in range(total_questions):
        question_type = question_plan[q_idx] if q_idx < len(question_plan) else "strict_role"
        follow_up_mode = choose_follow_up_mode(history)
        preferred_categories = choose_categories_for_round(
            session_state=session_state,
            profile=structured_profile,
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
                candidate_skills=safe_list(structured_profile.get("skills", [])),
                candidate_weak_areas=(
                    safe_list(structured_profile.get("weak_areas", []))
                    + safe_list(session_state.get("prior_weak_areas", []))
                ),
                context_tags=context_tags,
                categories=preferred_categories,
                allowed_roles=allowed_roles,
                session_state=session_state,
                top_k=5,
                duplicate_similarity_threshold=0.78,
                role_match_mode=role_match_mode,
            )

            if not retrieved:
                continue

            question_item = retrieved[0]
            question_embedding = retriever.embeddings[question_item["index"]]
            asked_question = personalize_question(question_item, structured_profile, history)
        elif question_type == "resume_specific":
            allowed_roles = [canonical_role(selected_role)]
            question_item = generate_resume_specific_question(
                profile=structured_profile,
                history=history,
                difficulty=current_difficulty,
                session_state=session_state,
            )
            asked_question = clean_text(question_item.get("question", ""))
        else:
            allowed_roles = [canonical_role(selected_role)]
            question_item = generate_skill_specific_question(
                profile=structured_profile,
                selected_role=selected_role,
                difficulty=current_difficulty,
                history=history,
            )
            asked_question = clean_text(question_item.get("question", ""))

        # For non-live modes, optional TTS here. In live mode, TTS is handled inside
        # get_candidate_response to strictly enforce speak-then-record sequencing.
        if input_mode != "audio_live":
            maybe_speak_question(asked_question, enable_tts=enable_tts, tts_voice=tts_voice)

        candidate_response = get_candidate_response(
            asked_question,
            input_mode=input_mode,
            round_index=q_idx,
            audio_adapter=audio_adapter,
            live_audio_adapter=live_audio_adapter,
            recording_seconds=recording_seconds,
            enable_tts=enable_tts,
            tts_voice=tts_voice,
        )
        candidate_answer = clean_text(candidate_response.get("answer_text", ""))

        # ASR failure handling for audio modes
        if input_mode in {"audio_json", "audio_live"}:
            asr_conf = float(candidate_response.get("audio_analysis", {}).get("transcription_confidence", 0.0) or 0.0)
            no_speech_prob = float(candidate_response.get("audio_analysis", {}).get("no_speech_prob", 0.0) or 0.0)
            if (not candidate_answer) or asr_conf < 25 or no_speech_prob > 0.85:
                print("[Audio] Low-confidence or empty transcript detected. Please repeat once.")
                candidate_response = get_candidate_response(
                    asked_question,
                    input_mode=input_mode,
                    round_index=q_idx,
                    audio_adapter=audio_adapter,
                    live_audio_adapter=live_audio_adapter,
                    recording_seconds=recording_seconds,
                )
                candidate_answer = clean_text(candidate_response.get("answer_text", ""))

        technical_eval = evaluate_answer(question_item, candidate_answer)
        cv_analysis = cv_adapter.get_cv_analysis(q_idx, total_questions) if cv_adapter else {}
        multimodal_eval = fuse_multimodal_evaluation(
            question=asked_question,
            candidate_answer=candidate_answer,
            audio_analysis=candidate_response.get("audio_analysis", {}),
            technical_evaluation=technical_eval,
            conversation_history=history,
            cv_analysis=cv_analysis,
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
            "warnings": multimodal_eval.get("warnings", []) if coaching_mode else [],
        }

        next_difficulty = adjust_difficulty(current_difficulty, evaluation)

        update_session_memory(
            session_state=session_state,
            asked_question=asked_question,
            question_embedding=question_embedding,
            question_item=question_item,
            difficulty=current_difficulty,
            candidate_answer=candidate_answer,
            evaluation=evaluation,
        )

        history.append(
            {
                "round": q_idx + 1,
                "retrieved_question": question_item.get("question", ""),
                "asked_question": asked_question,
                "category": question_item.get("category", ""),
                "difficulty": current_difficulty,
                "candidate_answer": candidate_answer,
                "candidate_response": candidate_response,
                "technical_evaluation": technical_eval,
                "cv_analysis": cv_analysis,
                "multimodal_evaluation": multimodal_eval,
                "evaluation": evaluation,
                "next_difficulty": next_difficulty,
                "follow_up_mode": follow_up_mode,
                "allowed_roles": allowed_roles,
                "question_type": question_type,
                "question_keywords": safe_list(question_item.get("keywords", [])),
            }
        )

        current_difficulty = next_difficulty

        print("\nEVALUATION:")
        print(json.dumps(evaluation, indent=2))

    avg_score = 0.0
    if history:
        avg_score = sum(turn["evaluation"].get("score", 0) for turn in history) / len(history)

    return {
        "selected_role": selected_role,
        "total_questions": total_questions,
        "completed_questions": len(history),
        "average_score": round(avg_score, 2),
        "final_difficulty": current_difficulty,
        "used_categories": session_state.get("used_categories", []),
        "interview_history": history,
    }


def start_interview_session(
    structured_profile: Dict,
    selected_role: str,
    total_questions: int = 5,
    input_mode: str = "terminal",
    audio_json_path: str = "audio_pipeline_results.json",
    recording_seconds: int = 90,
    enable_tts: bool = False,
    tts_voice: str = "Samantha",
    cv_mode: str = "off",
    cv_jsonl_path: str = "CV/cv_metrics.jsonl",
    prior_memory: Dict = None,
    coaching_mode: bool = False,
) -> Dict:
    return run_interview_session(
        structured_profile=structured_profile,
        selected_role=selected_role,
        total_questions=total_questions,
        input_mode=input_mode,
        audio_json_path=audio_json_path,
        recording_seconds=recording_seconds,
        enable_tts=enable_tts,
        tts_voice=tts_voice,
        cv_mode=cv_mode,
        cv_jsonl_path=cv_jsonl_path,
        prior_memory=prior_memory,
        coaching_mode=coaching_mode,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive interview session")
    parser.add_argument("--profile_json", default="", help="Path to pre-generated structured profile JSON")
    parser.add_argument("--resume_pdf", default="", help="Path to uploaded resume PDF")
    parser.add_argument("--role", required=True, help="Selected target interview role")
    parser.add_argument("--questions", type=int, default=5, help="Number of interview rounds")
    parser.add_argument("--input_mode", default="terminal", choices=["terminal", "audio_json", "audio_live"], help="Candidate response mode")
    parser.add_argument("--audio_json_path", default="audio_pipeline_results.json", help="Path to audio pipeline output JSON (used in audio_json mode)")
    parser.add_argument("--recording_seconds", type=int, default=90, help="Max audio capture window per answer in audio_live mode")
    parser.add_argument("--enable_tts", action="store_true", help="Dictate interview questions with TTS")
    parser.add_argument("--tts_voice", default="Samantha", help="macOS voice name for TTS")
    parser.add_argument("--cv_mode", default="off", choices=["off", "jsonl"], help="Computer vision signal source")
    parser.add_argument("--cv_jsonl_path", default="CV/cv_metrics.jsonl", help="Path to CV metrics JSONL")
    parser.add_argument("--coaching_mode", action="store_true", help="Enable coaching warnings/alerts in output")
    parser.add_argument("--save_json", default="", help="Optional path to save session output JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.profile_json and not args.resume_pdf:
        raise ValueError("Provide either --profile_json or --resume_pdf")

    if args.profile_json:
        with open(args.profile_json, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if "structured_profile" in profile:
            profile = profile["structured_profile"]
    else:
        if not os.path.exists(args.resume_pdf):
            raise FileNotFoundError(f"Resume PDF not found: {args.resume_pdf}")
        resume_text = extract_text_from_pdf(args.resume_pdf)
        profile = extract_resume_data(resume_text)
        if "error" in profile:
            raise RuntimeError(f"Resume agent failed: {profile['error']}")

    result = run_interview_session(
        structured_profile=profile,
        selected_role=args.role,
        total_questions=args.questions,
        input_mode=args.input_mode,
        audio_json_path=args.audio_json_path,
        recording_seconds=args.recording_seconds,
        enable_tts=args.enable_tts,
        tts_voice=args.tts_voice,
        cv_mode=args.cv_mode,
        cv_jsonl_path=args.cv_jsonl_path,
        coaching_mode=args.coaching_mode,
    )

    print("\n" + "=" * 60)
    print("INTERVIEW SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved interview session: {args.save_json}")


if __name__ == "__main__":
    main()
