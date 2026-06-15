# =========================================================
# multimodal_evaluator.py
# =========================================================
# PURPOSE:
# Fuse technical evaluation with audio behavioral signals.
# =========================================================

from typing import Dict, List


def _clip10(v: float) -> float:
    return max(0.0, min(10.0, float(v)))


def _speech_clarity_label(transcription_confidence: float) -> str:
    if transcription_confidence >= 80:
        return "high"
    if transcription_confidence >= 60:
        return "moderate"
    return "low"


def _pace_label(wpm: float) -> str:
    if wpm <= 0:
        return "unknown"
    if wpm < 95:
        return "slow"
    if wpm > 175:
        return "fast"
    return "natural"


def _confidence_level(score_10: float) -> str:
    if score_10 >= 7.5:
        return "high"
    if score_10 >= 5.0:
        return "moderate"
    return "low"


def _pause_interpretation(pause_ratio: float) -> str:
    if pause_ratio <= 0:
        return "Pause behavior could not be measured reliably."
    if pause_ratio < 0.18:
        return "Your delivery was very fluent with minimal dead air."
    if pause_ratio < 0.28:
        return "Your pause rhythm sounded natural and controlled."
    if pause_ratio < 0.38:
        return "You had a few noticeable pauses; tightening transitions would improve flow."
    if pause_ratio < 0.50:
        return "Your answer included frequent pauses; try outlining points before speaking."
    return "Long pauses reduced answer momentum; use concise opening structure and examples."


def _wpm_interpretation(wpm: float) -> str:
    if wpm <= 0:
        return "Speaking pace could not be estimated."
    if wpm < 105:
        return "Your pace was calm but slightly slow for interview clarity."
    if wpm < 130:
        return "Your pace was measured and professional."
    if wpm < 155:
        return "Your pace was conversational and confident."
    if wpm < 180:
        return "Your pace was slightly fast in places; add short emphasis pauses."
    return "Your pace was too fast for clear delivery; slow down to improve precision."


def _filler_interpretation(filler_density: float) -> str:
    if filler_density <= 0.02:
        return "Filler usage was very low, which improved clarity."
    if filler_density <= 0.06:
        return "Filler usage was controlled and generally acceptable."
    if filler_density <= 0.10:
        return "Some filler words appeared; replacing them with short pauses would sound sharper."
    return "High filler usage affected clarity; use shorter sentences and deliberate pauses."


def _trend_interpretation(
    pause_ratio: float,
    wpm: float,
    prev_pause: float,
    prev_wpm: float,
) -> List[str]:
    bits: List[str] = []
    if prev_pause > 0:
        delta_pause = pause_ratio - prev_pause
        if delta_pause <= -0.08:
            bits.append("Compared to the previous round, your fluency improved with fewer pauses.")
        elif delta_pause >= 0.10:
            bits.append("Compared to the previous round, pauses increased; aim for tighter structure.")

    if prev_wpm > 0 and wpm > 0:
        delta_wpm = wpm - prev_wpm
        if abs(delta_wpm) > 18:
            if delta_wpm > 0:
                bits.append("Your pace increased this round; keep it controlled for clarity.")
            else:
                bits.append("Your pace dropped this round; maintain steady momentum.")
    return bits


def _build_distraction_warnings(
    audio_analysis: Dict,
    cv_analysis: Dict,
    conversation_history: List[Dict],
) -> List[Dict]:
    warnings = []

    pause_ratio = float(audio_analysis.get("pause_ratio", 0.0) or 0.0)
    filler_density = float(audio_analysis.get("filler_density", 0.0) or 0.0)
    wpm = float(audio_analysis.get("wpm", 0.0) or 0.0)
    asr_conf = float(audio_analysis.get("transcription_confidence", 0.0) or 0.0)

    if pause_ratio >= 0.50:
        warnings.append({
            "source": "voice",
            "metric": "pause_ratio",
            "level": "high",
            "message": "Frequent long pauses detected; candidate may be losing flow.",
        })
    elif pause_ratio >= 0.42:
        warnings.append({
            "source": "voice",
            "metric": "pause_ratio",
            "level": "medium",
            "message": "Noticeable pauses detected; consider a slower, structured response opening.",
        })

    if filler_density >= 0.12:
        warnings.append({
            "source": "voice",
            "metric": "filler_density",
            "level": "medium",
            "message": "High filler usage detected; potential confidence/distraction issue.",
        })

    if asr_conf < 55:
        warnings.append({
            "source": "voice",
            "metric": "transcription_confidence",
            "level": "medium",
            "message": "Low speech clarity/confidence detected in transcription signal.",
        })

    if wpm > 0 and (wpm < 90 or wpm > 185):
        warnings.append({
            "source": "voice",
            "metric": "wpm",
            "level": "low",
            "message": "Pacing is outside natural range; delivery may feel rushed or hesitant.",
        })

    # CV distraction-style signals (non-scoring alerts only)
    eye = float(cv_analysis.get("eye_contact_score", 0.0) or 0.0)
    fidget = float(cv_analysis.get("fidget_score", 0.0) or 0.0)
    blink = float(cv_analysis.get("blink_rate", 0.0) or 0.0)

    if eye > 0 and eye < 3.5:
        warnings.append({
            "source": "cv",
            "metric": "eye_contact_score",
            "level": "medium",
            "message": "Low eye-contact signal detected; possible visual distraction.",
        })
    if fidget > 6.5:
        warnings.append({
            "source": "cv",
            "metric": "fidget_score",
            "level": "medium",
            "message": "Elevated fidget signal detected; candidate may be restless/distracted.",
        })
    if blink > 55:
        warnings.append({
            "source": "cv",
            "metric": "blink_rate",
            "level": "low",
            "message": "High blink rate observed; may indicate strain or distraction.",
        })

    # Change-based warnings vs previous round
    if conversation_history:
        prev = conversation_history[-1]
        prev_audio = (prev.get("candidate_response") or {}).get("audio_analysis", {})
        prev_cv = prev.get("cv_analysis") or {}

        prev_pause = float(prev_audio.get("pause_ratio", 0.0) or 0.0)
        prev_wpm = float(prev_audio.get("wpm", 0.0) or 0.0)
        prev_eye = float(prev_cv.get("eye_contact_score", 0.0) or 0.0)
        prev_fidget = float(prev_cv.get("fidget_score", 0.0) or 0.0)

        if pause_ratio - prev_pause > 0.15:
            warnings.append({
                "source": "voice",
                "metric": "pause_ratio_change",
                "level": "medium",
                "message": "Pause ratio increased sharply from previous round.",
            })
        if prev_wpm > 0 and abs(wpm - prev_wpm) > 45:
            warnings.append({
                "source": "voice",
                "metric": "wpm_change",
                "level": "low",
                "message": "Speaking pace changed significantly from previous round.",
            })
        if prev_eye > 0 and (prev_eye - eye) > 2.0:
            warnings.append({
                "source": "cv",
                "metric": "eye_contact_drop",
                "level": "medium",
                "message": "Eye-contact signal dropped versus previous round.",
            })
        if fidget - prev_fidget > 2.0:
            warnings.append({
                "source": "cv",
                "metric": "fidget_increase",
                "level": "medium",
                "message": "Fidget signal increased versus previous round.",
            })

    return warnings


def fuse_multimodal_evaluation(
    question: str,
    candidate_answer: str,
    audio_analysis: Dict,
    technical_evaluation: Dict,
    conversation_history: List[Dict],
    cv_analysis: Dict | None = None,
) -> Dict:
    """
    Returns unified multimodal scoring + coaching output.
    """
    tech = float(technical_evaluation.get("technical_score", technical_evaluation.get("score", 0)))

    tc = float(audio_analysis.get("transcription_confidence", 0.0)) / 10.0  # 0..10
    pause_ratio = float(audio_analysis.get("pause_ratio", 0.0))
    filler_density = float(audio_analysis.get("filler_density", 0.0))
    speech_consistency = float(audio_analysis.get("speech_consistency", 0.0)) * 10.0
    wpm = float(audio_analysis.get("wpm", 0.0))
    pitch_variation = float(audio_analysis.get("pitch_variation", 0.0))
    cv_analysis = cv_analysis or {}
    cv_engagement = float(cv_analysis.get("engagement_score", 0.0))  # expected 0..10
    cv_eye = float(cv_analysis.get("eye_contact_score", 0.0))        # expected 0..10

    # convert to 0..10 signals
    pause_score = _clip10((1.0 - min(1.0, abs(pause_ratio - 0.22))) * 10.0)
    filler_score = _clip10((1.0 - min(1.0, filler_density)) * 10.0)
    if wpm <= 0:
        wpm_score = 5.0
    else:
        wpm_score = _clip10((1.0 - min(1.0, abs(wpm - 135.0) / 135.0)) * 10.0)
    pitch_score = _clip10(min(10.0, pitch_variation * 2.0))

    # weighted fusion
    overall = (
        0.45 * _clip10(tech)
        + 0.15 * _clip10(tc)
        + 0.10 * pause_score
        + 0.10 * filler_score
        + 0.10 * _clip10(speech_consistency)
        + 0.05 * wpm_score
        + 0.05 * pitch_score
    )
    # Optional CV contribution (lightweight) when available.
    # NOTE:
    # - posture_score is intentionally excluded from scoring (unstable signal).
    # - stress_score and emotion_confidence are also excluded from scoring.
    if cv_analysis:
        cv_comp = (0.6 * _clip10(cv_engagement) + 0.4 * _clip10(cv_eye))
        overall = 0.9 * overall + 0.1 * cv_comp

    communication_score = _clip10(
        (0.35 * _clip10(tc)) + (0.20 * pause_score) + (0.20 * filler_score) + (0.15 * wpm_score) + (0.10 * _clip10(speech_consistency))
    )

    confidence_score = _clip10(
        (0.35 * _clip10(tc)) + (0.25 * _clip10(speech_consistency)) + (0.20 * filler_score) + (0.20 * pause_score)
    )

    hesitation_detected = pause_ratio > 0.45 or filler_density > 0.12
    warnings = _build_distraction_warnings(
        audio_analysis=audio_analysis,
        cv_analysis=cv_analysis,
        conversation_history=conversation_history,
    )

    tech_feedback = technical_evaluation.get("feedback", "")
    comm_feedback_bits = []
    behav_feedback_bits = []

    pace = _pace_label(wpm)
    clarity = _speech_clarity_label(float(audio_analysis.get("transcription_confidence", 0.0)))
    conf_level = _confidence_level(confidence_score)

    # Range-based communication coaching (always informative, not binary-only).
    comm_feedback_bits.append(_pause_interpretation(pause_ratio))
    comm_feedback_bits.append(_wpm_interpretation(wpm))
    comm_feedback_bits.append(_filler_interpretation(filler_density))

    prev_pause = 0.0
    prev_wpm = 0.0
    if conversation_history:
        prev = conversation_history[-1]
        prev_audio = (prev.get("candidate_response") or {}).get("audio_analysis", {})
        prev_pause = float(prev_audio.get("pause_ratio", 0.0) or 0.0)
        prev_wpm = float(prev_audio.get("wpm", 0.0) or 0.0)
    comm_feedback_bits.extend(_trend_interpretation(pause_ratio, wpm, prev_pause, prev_wpm))

    if hesitation_detected:
        behav_feedback_bits.append("Hesitation detected; take a 2-second structure pause then answer confidently.")
    if conf_level == "low":
        behav_feedback_bits.append("Confidence appears low; focus on clear opening statements and concrete examples.")
    elif conf_level == "high":
        behav_feedback_bits.append("Confidence came through well; keep combining clarity with concrete impact examples.")
    else:
        behav_feedback_bits.append("Confidence is moderate; stronger opening statements can make answers more convincing.")

    if cv_analysis:
        eye = float(cv_analysis.get("eye_contact_score", 0.0) or 0.0)
        attention = float(cv_analysis.get("attention_score", 0.0) or 0.0)
        if eye > 0 and eye < 3.5:
            behav_feedback_bits.append("Eye-contact signal was low; keep your gaze more camera-aligned while explaining.")
        elif eye >= 4.5:
            behav_feedback_bits.append("Eye-contact signal was strong, which helped interview presence.")
        if attention > 0 and attention < 3.0:
            behav_feedback_bits.append("Attention signal dipped at times; reduce visual distractions between points.")

    coaching_feedback = {
        "technical_feedback": tech_feedback,
        "communication_feedback": " ".join(dict.fromkeys([b.strip() for b in comm_feedback_bits if b and b.strip()])),
        "behavioral_feedback": " ".join(dict.fromkeys([b.strip() for b in behav_feedback_bits if b and b.strip()])),
    }

    recommended_next_difficulty = technical_evaluation.get("recommended_next_difficulty", "Medium")
    if _clip10(tech) >= 7 and confidence_score >= 5:
        recommended_next_difficulty = "Hard"
    elif _clip10(tech) <= 4:
        recommended_next_difficulty = "Easy"

    return {
        "technical_score": int(round(_clip10(tech))),
        "communication_score": int(round(communication_score)),
        "confidence_score": int(round(confidence_score)),
        "overall_score": int(round(_clip10(overall))),
        "behavioral_signals": {
            "hesitation_detected": bool(hesitation_detected),
            "speech_clarity": clarity,
            "pace": pace,
            "confidence_level": conf_level,
        },
        "coaching_feedback": coaching_feedback,
        "recommended_next_difficulty": recommended_next_difficulty,
        "warnings": warnings,
    }
