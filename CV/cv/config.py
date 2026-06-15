from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EyeConfig:
    ear_blink_threshold: float = 0.21
    gaze_center_threshold: float = 0.18
    blink_min_interval_s: float = 0.15
    perclos_low: float = 0.05
    perclos_high: float = 0.35


@dataclass
class PostureConfig:
    max_shoulder_slope_deg: float = 10.0
    min_shoulder_hip_angle_deg: float = 70.0
    max_shoulder_center_std: float = 0.03


@dataclass
class HeadPoseConfig:
    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 15.0


@dataclass
class EngagementConfig:
    eye_contact_weight: float = 0.45
    posture_weight: float = 0.35
    emotion_weight: float = 0.20


@dataclass
class StressConfig:
    blink_rate_weight: float = 0.35
    gaze_aversion_weight: float = 0.25
    posture_weight: float = 0.15
    perclos_weight: float = 0.15
    fidget_weight: float = 0.10


@dataclass
class VisionConfig:
    eye: EyeConfig = field(default_factory=EyeConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    calibration_seconds: float = 8.0
    rolling_window_seconds: float = 30.0
