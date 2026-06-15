from __future__ import annotations

from dataclasses import dataclass

from .features import normalize_score


@dataclass
class EngagementInputs:
    eye_contact: float
    posture: float
    emotion: float


@dataclass
class StressInputs:
    blink_rate: float
    gaze_aversion: float
    posture_instability: float
    perclos: float
    fidget: float


def engagement_score(inputs: EngagementInputs, weights: tuple[float, float, float]) -> float:
    e_w, p_w, m_w = weights
    total = e_w + p_w + m_w
    if total == 0:
        return 0.0
    return (inputs.eye_contact * e_w + inputs.posture * p_w + inputs.emotion * m_w) / total


def stress_score(inputs: StressInputs, weights: tuple[float, float, float, float, float]) -> float:
    b_w, g_w, p_w, c_w, f_w = weights
    total = b_w + g_w + p_w + c_w + f_w
    if total == 0:
        return 0.0
    return (
        inputs.blink_rate * b_w
        + inputs.gaze_aversion * g_w
        + inputs.posture_instability * p_w
        + inputs.perclos * c_w
        + inputs.fidget * f_w
    ) / total


def blink_rate_score(blinks_per_minute: float) -> float:
    return normalize_score(blinks_per_minute, 6.0, 30.0)


def perclos_score(perclos: float, low: float, high: float) -> float:
    return normalize_score(perclos, low, high)
