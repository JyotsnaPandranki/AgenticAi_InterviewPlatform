# =========================================================
# answer_evaluator.py
# =========================================================
# PURPOSE:
# Evaluate candidate answer with LLM grounded by retrieved question
# and ideal answer from dataset.
# =========================================================

import json
import re

import os
from openai import OpenAI

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


def _fallback_evaluation(candidate_answer: str):
    word_count = len(clean_text(candidate_answer).split())
    base = 4 if word_count < 20 else 6 if word_count < 80 else 7
    return {
        "score": base,
        "communication_score": max(1, min(10, base)),
        "technical_score": max(1, min(10, base)),
        "confidence_estimate": max(1, min(10, base - 1)),
        "feedback": "Answer captured. Add more structure, examples, and role-specific depth.",
        "recommended_next_difficulty": "Medium",
    }


def evaluate_answer(question_item: dict, candidate_response) -> dict:
    """
    Input question_item expects:
    - question
    - ideal_answer
    - difficulty
    - role
    """
    question = clean_text(question_item.get("question", ""))
    ideal_answer = clean_text(question_item.get("ideal_answer", ""))
    role = clean_text(question_item.get("role", ""))
    difficulty = clean_text(question_item.get("difficulty", "Medium"))

    if isinstance(candidate_response, dict):
        candidate_answer = clean_text(candidate_response.get("answer_text", ""))
        ext_communication_score = candidate_response.get("communication_score", None)
        ext_audio_features = candidate_response.get("audio_features", {}) or {}
    else:
        candidate_answer = clean_text(candidate_response)
        ext_communication_score = None
        ext_audio_features = {}

    if not clean_text(candidate_answer):
        return {
            "score": 1,
            "communication_score": 1,
            "technical_score": 1,
            "confidence_estimate": 1,
            "feedback": "No answer provided.",
            "recommended_next_difficulty": "Easy",
        }

    prompt = f"""
You are an interview evaluator.
Evaluate the candidate answer based on the retrieved interview question and ideal answer.

Return ONLY valid JSON with this exact schema:
{{
  "score": 0,
  "communication_score": 0,
  "technical_score": 0,
  "confidence_estimate": 0,
  "feedback": "",
  "recommended_next_difficulty": "Easy"
}}

Rules:
- Scores must be integers from 0 to 10.
- Keep feedback concise and actionable (1-3 sentences).
- recommended_next_difficulty must be one of Easy, Medium, Hard.

Question: {question}
Role: {role}
Current Difficulty: {difficulty}
Ideal Answer Reference: {ideal_answer}
Candidate Answer: {clean_text(candidate_answer)}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a strict, fair interview answer evaluator."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)

        for key in ["score", "communication_score", "technical_score", "confidence_estimate"]:
            parsed[key] = int(max(0, min(10, int(parsed.get(key, 0)))))

        next_diff = str(parsed.get("recommended_next_difficulty", "Medium")).strip().lower()
        if next_diff == "easy":
            parsed["recommended_next_difficulty"] = "Easy"
        elif next_diff == "hard":
            parsed["recommended_next_difficulty"] = "Hard"
        else:
            parsed["recommended_next_difficulty"] = "Medium"

        parsed["feedback"] = clean_text(parsed.get("feedback", ""))

        # Blend semantic score + communication quality.
        semantic_score = float(parsed.get("score", 0))
        if ext_communication_score is not None:
            # External communication score is expected in 0..100 scale.
            comm_score_10 = max(0.0, min(10.0, float(ext_communication_score) / 10.0))
            parsed["communication_score"] = int(round(comm_score_10))
        else:
            comm_score_10 = float(parsed.get("communication_score", 0))

        blended_score = (0.7 * semantic_score) + (0.3 * comm_score_10)
        parsed["score"] = int(round(max(0.0, min(10.0, blended_score))))

        # Add audio-specific coaching signals when available.
        extra_feedback = []
        filler_density = float(ext_audio_features.get("filler_density", 0.0) or 0.0)
        wpm = float(ext_audio_features.get("wpm", 0.0) or 0.0)
        pause_ratio = float(ext_audio_features.get("pause_ratio", 0.0) or 0.0)

        if filler_density > 0.10:
            extra_feedback.append("Reduce filler words for stronger clarity.")
        if wpm > 0 and wpm < 95:
            extra_feedback.append("Increase speaking pace slightly to sound more confident.")
        elif wpm > 175:
            extra_feedback.append("Slow down a bit to improve comprehension.")
        if pause_ratio > 0.45:
            extra_feedback.append("Try smoother sentence flow with fewer long pauses.")

        if extra_feedback:
            parsed["feedback"] = clean_text(parsed["feedback"] + " " + " ".join(extra_feedback))

        return parsed

    except Exception:
        return _fallback_evaluation(candidate_answer)
