# =========================================================
# cv_adapter.py
# =========================================================
# PURPOSE:
# Adapter for CV metrics JSONL output into standardized interview signals.
# =========================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class CVResponseAdapter:
    def __init__(self, jsonl_path: str = "CV/cv_metrics.jsonl"):
        self.path = Path(jsonl_path)
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _slice_for_round(self, round_index: int, total_rounds: int) -> List[Dict]:
        if not self.rows:
            return []
        total_rounds = max(1, int(total_rounds))
        n = len(self.rows)
        start = int((round_index / total_rounds) * n)
        end = int(((round_index + 1) / total_rounds) * n)
        start = max(0, min(start, n - 1))
        end = max(start + 1, min(end, n))
        return self.rows[start:end]

    def get_cv_analysis(self, round_index: int, total_rounds: int) -> Dict:
        rows = self._slice_for_round(round_index, total_rounds)
        if not rows:
            return {
                "engagement_score": 0.0,
                "stress_score": 0.0,
                "eye_contact_score": 0.0,
                "posture_score": 0.0,
                "blink_rate": 0.0,
                "fidget_score": 0.0,
                "emotion_confidence": 0.0,
                "non_scoring_metrics": ["stress_score", "posture_score", "emotion_confidence"],
            }

        return {
            "engagement_score": _mean([r.get("engagement_score", 0.0) for r in rows]) * 10.0,
            "stress_score": _mean([r.get("stress_score", 0.0) for r in rows]) * 10.0,
            "eye_contact_score": _mean([r.get("eye_contact_score", 0.0) for r in rows]) * 10.0,
            "posture_score": _mean([r.get("posture_score", 0.0) for r in rows]) * 10.0,
            "blink_rate": _mean([r.get("blink_rate", 0.0) for r in rows]),
            "fidget_score": _mean([r.get("fidget_score", 0.0) for r in rows]) * 10.0,
            "emotion_confidence": _mean([r.get("emotion_confidence", 0.0) for r in rows]) * 10.0,
            "non_scoring_metrics": ["stress_score", "posture_score", "emotion_confidence"],
        }
