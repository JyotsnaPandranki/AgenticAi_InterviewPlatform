from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FrameMetrics:
    timestamp: float
    eye_contact_score: float
    gaze_score: float
    head_pose_score: float
    head_pose_yaw: float
    head_pose_pitch: float
    head_pose_roll: float
    engagement_score: float
    stress_score: float
    posture_score: float
    posture_instability: float
    blink_rate: float
    perclos: float
    fidget_score: float
    emotion: Optional[str]
    emotion_confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)
