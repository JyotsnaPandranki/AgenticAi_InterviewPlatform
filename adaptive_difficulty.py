# =========================================================
# adaptive_difficulty.py
# =========================================================
# PURPOSE:
# Reusable difficulty progression logic for adaptive interviews.
# =========================================================

from typing import Dict
import random

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]


def normalize_difficulty(level: str) -> str:
    if not level:
        return "Medium"
    lv = str(level).strip().lower()
    if lv == "easy":
        return "Easy"
    if lv == "hard":
        return "Hard"
    return "Medium"


def next_harder(level: str) -> str:
    level = normalize_difficulty(level)
    idx = DIFFICULTY_LEVELS.index(level)
    return DIFFICULTY_LEVELS[min(idx + 1, len(DIFFICULTY_LEVELS) - 1)]


def next_easier(level: str) -> str:
    level = normalize_difficulty(level)
    idx = DIFFICULTY_LEVELS.index(level)
    return DIFFICULTY_LEVELS[max(idx - 1, 0)]


def adjust_difficulty(
    current_difficulty: str,
    eval_result: Dict,
    up_threshold: float = 7.5,
    down_threshold: float = 4.5,
    epsilon: float = 0.10,
) -> str:
    """
    Default policy:
    - strong answer => harder
    - weak answer => easier
    - otherwise hold

    Uses overall score if present, otherwise technical score.
    """
    current = normalize_difficulty(current_difficulty)
    score = eval_result.get("score", eval_result.get("technical_score", 5))

    try:
        score = float(score)
    except Exception:
        score = 5.0

    if score >= up_threshold:
        target = next_harder(current)
    elif score <= down_threshold:
        target = next_easier(current)
    else:
        target = None

    if target is None:
        model_suggested = eval_result.get("recommended_next_difficulty", "")
        if model_suggested:
            target = normalize_difficulty(model_suggested)
        else:
            target = current

    # Epsilon-greedy exploration around chosen direction.
    # 90% follow target direction, 10% explore any difficulty.
    if random.random() < max(0.0, min(1.0, epsilon)):
        return random.choice(DIFFICULTY_LEVELS)
    return target
