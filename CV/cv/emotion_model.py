from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class EmotionResult:
    label: Optional[str]
    confidence: float


class OnnxEmotionModel:
    def __init__(
        self,
        model_path: str,
        labels: Optional[list[str]] = None,
        use_clahe: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("onnxruntime is required for OnnxEmotionModel") from exc

        self.labels = labels or [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        self._output_name = self._session.get_outputs()[0].name
        self._layout = _infer_layout(input_meta.shape)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    def predict(self, face_bgr: np.ndarray) -> EmotionResult:
        if face_bgr.size == 0:
            return EmotionResult(label=None, confidence=0.0)

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        if self._clahe is not None:
            resized = self._clahe.apply(resized)
        tensor = resized.astype(np.float32) / 255.0
        if self._layout == "NHWC":
            tensor = tensor.reshape(1, 64, 64, 1)
        else:
            tensor = tensor.reshape(1, 1, 64, 64)

        logits = self._session.run([self._output_name], {self._input_name: tensor})[0]
        probs = _softmax(logits[0])
        idx = int(np.argmax(probs))
        label = self.labels[idx] if self.labels else None
        return EmotionResult(label=label, confidence=float(probs[idx]))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _infer_layout(shape: list | tuple) -> str:
    if len(shape) == 4:
        if shape[1] in (1, None) and shape[3] not in (1, None):
            return "NCHW"
        if shape[3] in (1, None):
            return "NHWC"
    return "NCHW"
