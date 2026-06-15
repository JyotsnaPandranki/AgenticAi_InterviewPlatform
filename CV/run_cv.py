from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

import cv2

from cv.emotion_model import OnnxEmotionModel
from cv.pipeline import VisionPipeline

# Update these values for your run
VIDEO_SOURCE: Union[int, Path] = 0  # Use 0 for webcam, or Path("C:/path/to/video.mp4")
OUTPUT_PATH = Path("cv_metrics.jsonl")

# Needed only if MediaPipe solutions are unavailable.
FACE_MODEL_PATH: str | None = None
POSE_MODEL_PATH: str | None = None

# Optional emotion model (FER+ ONNX)
EMOTION_MODEL_PATH = "emotion-ferplus-8.onnx"  # Set to None to disable emotion recognition
EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def iter_frames(video_source: Union[int, Path]) -> Iterable[tuple[float, any]]:
    if isinstance(video_source, Path):
        cap = cv2.VideoCapture(str(video_source))
    else:
        cap = cv2.VideoCapture(int(video_source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = idx / fps
        idx += 1
        yield timestamp, frame
    cap.release()


def main() -> None:
    emotion_model = None
    if EMOTION_MODEL_PATH:
        model_path = Path(EMOTION_MODEL_PATH)
        if model_path.exists():
            emotion_model = OnnxEmotionModel(str(model_path), labels=EMOTION_LABELS)
        else:
            print(f"Emotion model not found at {model_path}. Emotion will be disabled.")
    pipeline = VisionPipeline(
        emotion_model=emotion_model,
        face_model_path=FACE_MODEL_PATH,
        pose_model_path=POSE_MODEL_PATH,
    )
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ts, frame in iter_frames(VIDEO_SOURCE):
            metrics = pipeline.analyze_frame(frame, ts)
            if not metrics:
                continue
            f.write(json.dumps(metrics.__dict__) + "\n")

    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
