from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from .config import VisionConfig
from .emotion_model import OnnxEmotionModel
from .features import center_distance_ratio, eye_aspect_ratio, normalize_score, shoulder_slope
from .landmarks import (
    FACE_CHIN,
    FACE_LEFT_EYE,
    FACE_LEFT_EYE_OUTER,
    FACE_LEFT_IRIS,
    FACE_LEFT_MOUTH,
    FACE_NOSE_TIP,
    FACE_RIGHT_EYE,
    FACE_RIGHT_EYE_OUTER,
    FACE_RIGHT_IRIS,
    FACE_RIGHT_MOUTH,
    POSE_LEFT_HIP,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_HIP,
    POSE_RIGHT_SHOULDER,
    LandmarkExtractor,
)
from .metrics import (
    EngagementInputs,
    StressInputs,
    blink_rate_score,
    engagement_score,
    perclos_score,
    stress_score,
)
from .types import FrameMetrics


class VisionPipeline:
    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        emotion_model: Optional[OnnxEmotionModel] = None,
        face_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
    ) -> None:
        self.config = config or VisionConfig()
        self.extractor = LandmarkExtractor(face_model_path=face_model_path, pose_model_path=pose_model_path)
        self.emotion_model = emotion_model
        self._ear_values: Deque[tuple[float, float]] = deque()
        self._blink_times: Deque[float] = deque()
        self._shoulder_centers: Deque[tuple[float, tuple[float, float]]] = deque()

    def analyze_frame(self, frame_bgr: np.ndarray, timestamp_s: float) -> Optional[FrameMetrics]:
        packet = self.extractor.process(frame_bgr)
        if not packet.face:
            return FrameMetrics(
                timestamp=timestamp_s,
                eye_contact_score=0.0,
                gaze_score=0.0,
                head_pose_score=0.0,
                head_pose_yaw=0.0,
                head_pose_pitch=0.0,
                head_pose_roll=0.0,
                engagement_score=0.0,
                stress_score=0.0,
                posture_score=0.0,
                posture_instability=0.0,
                blink_rate=0.0,
                perclos=0.0,
                fidget_score=0.0,
                emotion=None,
                emotion_confidence=0.0,
                raw={"face_detected": False},
            )

        gaze_score = self._gaze_score(packet.face)
        head_pose_yaw, head_pose_pitch, head_pose_roll, head_pose_score = self._head_pose(frame_bgr, packet.face)
        eye_contact = self._combine_attention(gaze_score, head_pose_score)
        posture_score = self._posture_score(packet.pose) if packet.pose else 0.0
        posture_instability = self._posture_instability(packet.pose, timestamp_s) if packet.pose else 0.0
        blink_rate = self._blink_rate(packet.face, timestamp_s)
        perclos = self._perclos()
        emotion_label, emotion_conf = self._emotion_score(frame_bgr, packet.face)

        engagement = engagement_score(
            EngagementInputs(eye_contact=eye_contact, posture=posture_score, emotion=emotion_conf),
            (
                self.config.engagement.eye_contact_weight,
                self.config.engagement.posture_weight,
                self.config.engagement.emotion_weight,
            ),
        )
        stress = stress_score(
            StressInputs(
                blink_rate=blink_rate_score(blink_rate),
                gaze_aversion=1.0 - eye_contact,
                posture_instability=1.0 - posture_score,
                perclos=perclos_score(perclos, self.config.eye.perclos_low, self.config.eye.perclos_high),
                fidget=posture_instability,
            ),
            (
                self.config.stress.blink_rate_weight,
                self.config.stress.gaze_aversion_weight,
                self.config.stress.posture_weight,
                self.config.stress.perclos_weight,
                self.config.stress.fidget_weight,
            ),
        )

        raw = {
            "eye_contact": eye_contact,
            "gaze_score": gaze_score,
            "head_pose_score": head_pose_score,
            "head_pose_yaw": head_pose_yaw,
            "head_pose_pitch": head_pose_pitch,
            "head_pose_roll": head_pose_roll,
            "posture": posture_score,
            "posture_instability": posture_instability,
            "blink_rate": blink_rate,
            "perclos": perclos,
            "fidget_score": posture_instability,
            "emotion": emotion_label,
            "emotion_confidence": emotion_conf,
        }

        return FrameMetrics(
            timestamp=timestamp_s,
            eye_contact_score=eye_contact,
            gaze_score=gaze_score,
            head_pose_score=head_pose_score,
            head_pose_yaw=head_pose_yaw,
            head_pose_pitch=head_pose_pitch,
            head_pose_roll=head_pose_roll,
            engagement_score=engagement,
            stress_score=stress,
            posture_score=posture_score,
            posture_instability=posture_instability,
            blink_rate=blink_rate,
            perclos=perclos,
            fidget_score=posture_instability,
            emotion=emotion_label,
            emotion_confidence=emotion_conf,
            raw=raw,
        )

    def _gaze_score(self, face_landmarks: list[tuple[float, float, float]]) -> float:
        if len(face_landmarks) <= max(FACE_RIGHT_IRIS):
            return 0.5
        left_eye = [self._to_2d(face_landmarks[i]) for i in FACE_LEFT_EYE]
        right_eye = [self._to_2d(face_landmarks[i]) for i in FACE_RIGHT_EYE]
        left_iris = [self._to_2d(face_landmarks[i]) for i in FACE_LEFT_IRIS]
        right_iris = [self._to_2d(face_landmarks[i]) for i in FACE_RIGHT_IRIS]

        left_center = self._center(left_eye)
        right_center = self._center(right_eye)
        left_gaze = center_distance_ratio(left_iris, left_center)
        right_gaze = center_distance_ratio(right_iris, right_center)
        gaze = (left_gaze + right_gaze) / 2.0

        return 1.0 - normalize_score(gaze, 0.0, self.config.eye.gaze_center_threshold)

    def _head_pose(
        self, frame_bgr: np.ndarray, face_landmarks: list[tuple[float, float, float]]
    ) -> tuple[float, float, float, float]:
        height, width = frame_bgr.shape[:2]
        image_points = np.array(
            [
                self._to_pixel(face_landmarks[FACE_NOSE_TIP], width, height),
                self._to_pixel(face_landmarks[FACE_CHIN], width, height),
                self._to_pixel(face_landmarks[FACE_LEFT_EYE_OUTER], width, height),
                self._to_pixel(face_landmarks[FACE_RIGHT_EYE_OUTER], width, height),
                self._to_pixel(face_landmarks[FACE_LEFT_MOUTH], width, height),
                self._to_pixel(face_landmarks[FACE_RIGHT_MOUTH], width, height),
            ],
            dtype=np.float64,
        )

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float64,
        )

        focal_length = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0, 0.5

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_to_euler(rotation_matrix)

        yaw_score = 1.0 - normalize_score(abs(yaw), 0.0, self.config.head_pose.max_yaw_deg)
        pitch_score = 1.0 - normalize_score(abs(pitch), 0.0, self.config.head_pose.max_pitch_deg)
        head_pose_score = max(0.0, min(1.0, 0.5 * yaw_score + 0.5 * pitch_score))
        return yaw, pitch, roll, head_pose_score

    def _combine_attention(self, gaze_score: float, head_pose_score: float) -> float:
        return max(0.0, min(1.0, 0.65 * gaze_score + 0.35 * head_pose_score))

    def _posture_score(self, pose_landmarks: list[tuple[float, float, float]]) -> float:
        left_shoulder = self._to_2d(pose_landmarks[POSE_LEFT_SHOULDER])
        right_shoulder = self._to_2d(pose_landmarks[POSE_RIGHT_SHOULDER])
        left_hip = self._to_2d(pose_landmarks[POSE_LEFT_HIP])
        right_hip = self._to_2d(pose_landmarks[POSE_RIGHT_HIP])

        slope = shoulder_slope(left_shoulder, right_shoulder)
        slope_score = 1.0 - normalize_score(slope, 0.0, self.config.posture.max_shoulder_slope_deg)

        shoulder_center = self._center([left_shoulder, right_shoulder])
        hip_center = self._center([left_hip, right_hip])
        vertical_angle = 90.0 - abs(shoulder_center[0] - hip_center[0]) * 90.0
        vertical_score = normalize_score(vertical_angle, self.config.posture.min_shoulder_hip_angle_deg, 90.0)

        return max(0.0, min(1.0, 0.6 * slope_score + 0.4 * vertical_score))

    def _blink_rate(self, face_landmarks: list[tuple[float, float, float]], timestamp_s: float) -> float:
        left_eye = [self._to_2d(face_landmarks[i]) for i in FACE_LEFT_EYE]
        right_eye = [self._to_2d(face_landmarks[i]) for i in FACE_RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        self._ear_values.append((timestamp_s, ear))
        while self._ear_values and timestamp_s - self._ear_values[0][0] > self.config.rolling_window_seconds:
            self._ear_values.popleft()

        is_blink = ear < self.config.eye.ear_blink_threshold
        if is_blink:
            if not self._blink_times or (timestamp_s - self._blink_times[-1]) > self.config.eye.blink_min_interval_s:
                self._blink_times.append(timestamp_s)

        while self._blink_times and timestamp_s - self._blink_times[0] > self.config.rolling_window_seconds:
            self._blink_times.popleft()

        window = min(self.config.rolling_window_seconds, max(1.0, timestamp_s - (self._ear_values[0][0])))
        blinks_per_minute = len(self._blink_times) / window * 60.0
        return blinks_per_minute

    def _perclos(self) -> float:
        if not self._ear_values:
            return 0.0
        closed = sum(1 for _, ear in self._ear_values if ear < self.config.eye.ear_blink_threshold)
        return closed / len(self._ear_values)

    def _posture_instability(
        self, pose_landmarks: list[tuple[float, float, float]], timestamp_s: float
    ) -> float:
        left_shoulder = self._to_2d(pose_landmarks[POSE_LEFT_SHOULDER])
        right_shoulder = self._to_2d(pose_landmarks[POSE_RIGHT_SHOULDER])
        center = self._center([left_shoulder, right_shoulder])

        self._shoulder_centers.append((timestamp_s, center))
        while (
            self._shoulder_centers
            and timestamp_s - self._shoulder_centers[0][0] > self.config.rolling_window_seconds
        ):
            self._shoulder_centers.popleft()

        centers = [c for _, c in self._shoulder_centers]
        if len(centers) < 2:
            return 0.0
        xs = np.array([c[0] for c in centers], dtype=np.float32)
        ys = np.array([c[1] for c in centers], dtype=np.float32)
        std = float(np.sqrt(np.var(xs) + np.var(ys)))
        return normalize_score(std, 0.0, self.config.posture.max_shoulder_center_std)

    def _emotion_score(
        self, frame_bgr: np.ndarray, face_landmarks: Optional[list[tuple[float, float, float]]]
    ) -> tuple[Optional[str], float]:
        if not self.emotion_model or not face_landmarks:
            return None, 0.0

        face_crop = self._crop_face(frame_bgr, face_landmarks)
        if face_crop is None:
            return None, 0.0

        result = self.emotion_model.predict(face_crop)
        return result.label, result.confidence

    @staticmethod
    def _crop_face(
        frame_bgr: np.ndarray, face_landmarks: list[tuple[float, float, float]], padding: float = 0.15
    ) -> Optional[np.ndarray]:
        height, width = frame_bgr.shape[:2]
        xs = [lm[0] for lm in face_landmarks]
        ys = [lm[1] for lm in face_landmarks]
        min_x, max_x = max(min(xs) - padding, 0.0), min(max(xs) + padding, 1.0)
        min_y, max_y = max(min(ys) - padding, 0.0), min(max(ys) + padding, 1.0)

        x1, x2 = int(min_x * width), int(max_x * width)
        y1, y2 = int(min_y * height), int(max_y * height)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    @staticmethod
    def _to_2d(point: tuple[float, float, float]) -> tuple[float, float]:
        return point[0], point[1]

    @staticmethod
    def _to_pixel(point: tuple[float, float, float], width: int, height: int) -> tuple[float, float]:
        return point[0] * width, point[1] * height

    @staticmethod
    def _center(points: list[tuple[float, float]]) -> tuple[float, float]:
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

    @staticmethod
    def _rotation_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0.0
        return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))
