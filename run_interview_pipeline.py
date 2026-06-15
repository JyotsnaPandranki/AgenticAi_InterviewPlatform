# =========================================================
# run_interview_pipeline.py
# =========================================================
# PURPOSE:
# Unified demo runtime flow:
# resume PDF -> recommendations -> role selection -> adaptive interview
# =========================================================

import argparse
import json
import os
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from database_manager import DatabaseManager
from interview_memory_manager import InterviewMemoryManager
from interview_orchestrator import start_interview_session
from recommend_jobs import (
    EMBED_MODEL_NAME,
    JOB_EMBEDDINGS_FILE,
    JOBS_METADATA_FILE,
    build_output,
    extract_text_from_pdf,
    load_job_index,
    recommend_jobs_for_profile,
    save_json_output,
)
from resume_agent import extract_resume_data
from resume_registry import ResumeRegistry
from session_manager import MAX_INTERVIEWS_PER_SESSION, SessionManager


def print_top_recommendations(recommended_jobs: List[Dict]) -> None:
    print("\n" + "=" * 49)
    print("TOP RECOMMENDED ROLES")
    print("=" * 49)

    for i, job in enumerate(recommended_jobs, start=1):
        title = job.get("title", "")
        score = job.get("match_score", 0)
        print(f"{i}. {title:<30} ({score}%)")


def select_recommended_role(recommended_jobs: List[Dict]) -> Dict:
    """
    Selection abstraction.
    Current: terminal input.
    Future: replace with frontend selection event.
    """
    if not recommended_jobs:
        raise ValueError("No recommended jobs available for selection.")

    while True:
        raw = input("\nSelect role number (1-{}): ".format(len(recommended_jobs))).strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(recommended_jobs):
                return recommended_jobs[idx - 1]
        except ValueError:
            pass

        print("Invalid selection. Please enter a valid number.")


def choose_session_for_resume(session_manager: SessionManager, resume_row: Dict) -> Dict:
    """
    Session selection abstraction.
    Current: terminal input.
    Future: replace with frontend session selector.
    """
    resume_id = int(resume_row["resume_id"])
    sessions = session_manager.list_sessions(resume_id)

    if not sessions:
        print("\nNo existing sessions for this resume. Creating a new session...")
        return session_manager.create_session(resume_id, session_name="Session 1")

    print("\nResume already exists.")
    print("Choose:")
    print("1. Start New Session")
    print("2. Continue Existing Session")
    choice = input("Select option (1-2): ").strip()

    if choice == "1":
        return session_manager.create_session(
            resume_id,
            session_name=f"Session {len(sessions) + 1}",
        )

    active_sessions = [s for s in sessions if int(s.get("total_interviews", 0)) < MAX_INTERVIEWS_PER_SESSION]
    if not active_sessions:
        print("All existing sessions reached max interviews (3). Creating a new session...")
        return session_manager.create_session(
            resume_id,
            session_name=f"Session {len(sessions) + 1}",
        )

    print("\nExisting Sessions")
    for i, s in enumerate(active_sessions, start=1):
        print(
            f"{i}. {s.get('session_name')} | interviews: {s.get('total_interviews', 0)}/3 | "
            f"latest_role: {s.get('latest_role', '')} | latest_score: {s.get('latest_score', 0)}"
        )

    while True:
        raw = input(f"Select session number (1-{len(active_sessions)}): ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(active_sessions):
                return active_sessions[idx - 1]
        except ValueError:
            pass
        print("Invalid selection.")


def process_resume_and_recommend(
    resume_pdf: str,
    top_k: int,
) -> Dict:
    """
    Build in-memory profile and recommendations for one uploaded resume.
    """
    if not os.path.exists(resume_pdf):
        raise FileNotFoundError(f"Resume PDF not found: {resume_pdf}")

    print("=" * 60)
    print("Loading job index and embedding model...")
    print("=" * 60)
    job_embeddings, jobs_metadata = load_job_index(JOB_EMBEDDINGS_FILE, JOBS_METADATA_FILE)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("=" * 60)
    print("Extracting resume text and understanding candidate profile...")
    print("=" * 60)
    resume_text = extract_text_from_pdf(resume_pdf)
    structured_profile = extract_resume_data(resume_text)

    if "error" in structured_profile:
        raise RuntimeError(f"Resume agent failed: {structured_profile['error']}")

    print("=" * 60)
    print("Generating top role recommendations...")
    print("=" * 60)
    recommended_jobs = recommend_jobs_for_profile(
        profile=structured_profile,
        model=model,
        job_embeddings=job_embeddings,
        jobs_metadata=jobs_metadata,
        top_k=top_k,
    )

    return {
        "structured_profile": structured_profile,
        "recommendations": build_output(structured_profile, recommended_jobs),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Unified resume-to-interview pipeline")
    parser.add_argument("resume_pdf", help="Path to uploaded resume PDF")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommended roles to show")
    parser.add_argument("--questions", type=int, default=5, help="Number of interview questions")
    parser.add_argument(
        "--input_mode",
        default="terminal",
        choices=["terminal", "audio_json", "audio_live"],
        help="Candidate response mode",
    )
    parser.add_argument(
        "--audio_json_path",
        default="audio_pipeline_results.json",
        help="Path to audio pipeline output JSON (used in audio_json mode)",
    )
    parser.add_argument(
        "--recording_seconds",
        type=int,
        default=90,
        help="Audio recording duration per question in audio_live mode",
    )
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        help="Dictate interview questions with TTS",
    )
    parser.add_argument(
        "--tts_voice",
        default="Samantha",
        help="macOS voice name for TTS",
    )
    parser.add_argument(
        "--cv_mode",
        default="off",
        choices=["off", "jsonl"],
        help="Computer vision signal source",
    )
    parser.add_argument(
        "--cv_jsonl_path",
        default="CV/cv_metrics.jsonl",
        help="Path to CV metrics JSONL",
    )
    parser.add_argument(
        "--coaching_mode",
        action="store_true",
        help="Enable coaching warnings/alerts in interview output",
    )
    parser.add_argument(
        "--save_recommendations_json",
        default="",
        help="Optional path to save recommendation output JSON",
    )
    parser.add_argument(
        "--save_interview_json",
        default="",
        help="Optional path to save interview session output JSON",
    )
    parser.add_argument(
        "--db_path",
        default="data/processed_data/interview_sessions.db",
        help="SQLite path for resume/session/interview memory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = DatabaseManager(db_path=args.db_path)
    resume_registry = ResumeRegistry(db=db)
    session_manager = SessionManager(db=db)
    memory_manager = InterviewMemoryManager(db=db)

    pipeline_output = process_resume_and_recommend(
        resume_pdf=args.resume_pdf,
        top_k=args.top_k,
    )
    structured_profile = pipeline_output["structured_profile"]
    recommendation_output = pipeline_output["recommendations"]

    reg = resume_registry.register_or_get_resume(args.resume_pdf, structured_profile)
    resume_row = reg["resume"]
    selected_session = choose_session_for_resume(session_manager, resume_row)
    session_id = int(selected_session["session_id"])
    if not session_manager.can_add_interview(session_id):
        raise RuntimeError("Selected session already has 3 interviews. Start a new session.")

    prior_memory = memory_manager.get_latest_memory(session_id=session_id) or {}

    recommended_jobs = recommendation_output.get("recommended_jobs", [])
    print_top_recommendations(recommended_jobs)

    if args.save_recommendations_json:
        save_json_output(args.save_recommendations_json, recommendation_output)
        print(f"\nSaved recommendations: {args.save_recommendations_json}")

    selected_job = select_recommended_role(recommended_jobs)
    selected_role = selected_job.get("title", "")

    print("\n" + "=" * 60)
    print(f"Starting adaptive interview for role: {selected_role}")
    print("=" * 60)

    interview_result = start_interview_session(
        structured_profile=structured_profile,
        selected_role=selected_role,
        total_questions=args.questions,
        input_mode=args.input_mode,
        audio_json_path=args.audio_json_path,
        recording_seconds=args.recording_seconds,
        enable_tts=args.enable_tts,
        tts_voice=args.tts_voice,
        cv_mode=args.cv_mode,
        cv_jsonl_path=args.cv_jsonl_path,
        coaching_mode=args.coaching_mode,
        prior_memory=prior_memory,
    )

    interview_row = session_manager.add_interview_record(
        session_id=session_id,
        target_role=selected_role,
        interview_result=interview_result,
    )
    compressed_memory = memory_manager.build_compressed_memory(interview_result)
    memory_manager.upsert_memory(session_id=session_id, memory=compressed_memory)

    print("\n" + "=" * 60)
    print("INTERVIEW SUMMARY")
    print("=" * 60)
    print(json.dumps(interview_result, indent=2, ensure_ascii=False))
    print(f"\n[Session] session_id={session_id} | interview_id={interview_row.get('interview_id')}")

    if args.save_interview_json:
        save_json_output(args.save_interview_json, interview_result)
        print(f"\nSaved interview session: {args.save_interview_json}")


if __name__ == "__main__":
    main()
