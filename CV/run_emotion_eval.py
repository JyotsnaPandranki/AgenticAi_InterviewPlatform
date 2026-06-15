from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from cv.emotion_model import OnnxEmotionModel

# Update these paths for your run
MODEL_PATH = "emotion-ferplus-8.onnx"
DATASET_DIR = Path(r"C:\path\to\emotion_dataset")  # expects subfolders named by label

LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def iter_images(root: Path) -> Iterable[Tuple[np.ndarray, str]]:
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for image_path in label_dir.glob("**/*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            yield image, label


def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise SystemExit("MODEL_PATH does not exist")
    if not DATASET_DIR.exists():
        raise SystemExit("DATASET_DIR does not exist")

    model = OnnxEmotionModel(MODEL_PATH, labels=LABELS)

    total = 0
    correct = 0
    confidences = []

    for image, label in iter_images(DATASET_DIR):
        result = model.predict(image)
        if result.label is None:
            continue
        total += 1
        confidences.append(result.confidence)
        if result.label == label:
            correct += 1

    if total == 0:
        raise SystemExit("No images found in DATASET_DIR")

    accuracy = correct / total
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    print(f"samples={total}")
    print(f"accuracy={accuracy:.3f}")
    print(f"avg_confidence={avg_conf:.3f}")


if __name__ == "__main__":
    main()
