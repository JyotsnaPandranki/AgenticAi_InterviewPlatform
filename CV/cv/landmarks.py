from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("mediapipe is required for landmark extraction") from exc

_HAS_SOLUTIONS = hasattr(mp, "solutions")

Point3D = Tuple[float, float, float]

FACE_LEFT_EYE = [33, 160, 158, 133, 153, 144]
FACE_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_LEFT_IRIS = [474, 475, 476, 477]
FACE_RIGHT_IRIS = [469, 470, 471, 472]
FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_LEFT_EYE_OUTER = 33
FACE_RIGHT_EYE_OUTER = 263
FACE_LEFT_MOUTH = 61
FACE_RIGHT_MOUTH = 291

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24


@dataclass
class LandmarkPacket:
    face: Optional[List[Point3D]]
    pose: Optional[List[Point3D]]


class LandmarkExtractor:
    def __init__(self, face_model_path: Optional[str] = None, pose_model_path: Optional[str] = None) -> None:
        self._backend = "solutions" if _HAS_SOLUTIONS else "tasks"
        if self._backend == "solutions":
            self._face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        face_model_path = face_model_path or os.environ.get("MEDIAPIPE_FACE_MODEL")
        pose_model_path = pose_model_path or os.environ.get("MEDIAPIPE_POSE_MODEL")
        if not face_model_path or not pose_model_path:
            raise RuntimeError(
                "MediaPipe solutions are unavailable. Set MEDIAPIPE_FACE_MODEL and "
                "MEDIAPIPE_POSE_MODEL to .task model paths, or install a MediaPipe "
                "version with solutions (e.g. mediapipe==0.10.14)."
            )

        self._image_cls = Image
        self._image_format = ImageFormat.SRGB

        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._face = vision.FaceLandmarker.create_from_options(face_options)
        self._pose = vision.PoseLandmarker.create_from_options(pose_options)

    def process(self, frame_bgr: np.ndarray) -> LandmarkPacket:
        frame_rgb = frame_bgr[:, :, ::-1]
        if self._backend == "solutions":
            face_results = self._face.process(frame_rgb)
            pose_results = self._pose.process(frame_rgb)

            face_landmarks = None
            if face_results.multi_face_landmarks:
                face_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in face_results.multi_face_landmarks[0].landmark
                ]

            pose_landmarks = None
            if pose_results.pose_landmarks:
                pose_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark
                ]

            return LandmarkPacket(face=face_landmarks, pose=pose_landmarks)

        mp_image = self._image_cls(image_format=self._image_format, data=frame_rgb)
        face_results = self._face.detect(mp_image)
        pose_results = self._pose.detect(mp_image)

        face_landmarks = None
        if face_results.face_landmarks:
            face_landmarks = [(lm.x, lm.y, lm.z) for lm in face_results.face_landmarks[0]]

        pose_landmarks = None
        if pose_results.pose_landmarks:
            pose_landmarks = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks[0]]

        return LandmarkPacket(face=face_landmarks, pose=pose_landmarks)
