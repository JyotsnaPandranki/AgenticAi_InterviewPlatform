# =========================================================
# audio_adapter.py
# =========================================================
# PURPOSE:
# Adapter between audio pipeline output JSON and interview system.
# Produces a standardized candidate response object.
# =========================================================

import json
from typing import Dict, List


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


class AudioResponseAdapter:
    """
    Reads audio_pipeline_results.json and exposes standardized responses.
    """

    def __init__(self, json_path: str = "audio_pipeline_results.json"):
        self.json_path = json_path
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Audio results JSON must be a list")
        return data

    def _to_standard(self, row: Dict) -> Dict:
        transcript = (row or {}).get("transcript", "")
        features = (row or {}).get("features", {}) or {}
        scores = (row or {}).get("scores", {}) or {}

        communication_score = _safe_float(scores.get("communication_score", 0.0))

        # Confidence proxy combines speech consistency + low fillers + moderate pauses.
        speech_consistency = _safe_float(features.get("speech_consistency", 0.0))  # usually 0..1
        filler_density = _safe_float(features.get("filler_density", 0.0))          # lower is better
        pause_ratio = _safe_float(features.get("pause_ratio", 0.0))                # moderate is better

        filler_penalty = max(0.0, min(1.0, filler_density))
        pause_quality = max(0.0, 1.0 - abs(pause_ratio - 0.22))
        confidence_score = (
            100.0 * (
                0.5 * max(0.0, min(1.0, speech_consistency))
                + 0.3 * pause_quality
                + 0.2 * (1.0 - filler_penalty)
            )
        )

        return {
            "answer_text": str(transcript or "").strip(),
            "transcript": str(transcript or "").strip(),
            "communication_score": round(communication_score, 2),
            "confidence_score": round(confidence_score, 2),
            "audio_features": {
                "pause_ratio": _safe_float(features.get("pause_ratio", 0.0)),
                "wpm": _safe_float(features.get("wpm", 0.0)),
                "filler_density": _safe_float(features.get("filler_density", 0.0)),
                "pitch_variation": _safe_float(features.get("pitch_variation", 0.0)),
                "speech_consistency": _safe_float(features.get("speech_consistency", 0.0)),
            },
            "audio_analysis": {
                "transcription_confidence": _safe_float((row or {}).get("transcription_confidence", confidence_score)),
                "pause_ratio": _safe_float(features.get("pause_ratio", 0.0)),
                "wpm": _safe_float(features.get("wpm", 0.0)),
                "filler_density": _safe_float(features.get("filler_density", 0.0)),
                "pitch_variation": _safe_float(features.get("pitch_variation", 0.0)),
                "speech_consistency": _safe_float(features.get("speech_consistency", 0.0)),
                "communication_score": round(communication_score, 2),
            },
        }

    def get_response(self, round_index: int) -> Dict:
        if not self.rows:
            return {
                "answer_text": "",
                "communication_score": 0.0,
                "confidence_score": 0.0,
                "audio_features": {},
            }

        if round_index < len(self.rows):
            row = self.rows[round_index]
        else:
            # Reuse last row if interview rounds exceed audio samples.
            row = self.rows[-1]

        return self._to_standard(row)
