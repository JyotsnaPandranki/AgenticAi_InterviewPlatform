from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2

from .pipeline import VisionPipeline


def iter_frames(video_path: Path) -> Iterable[tuple[float, any]]:
    cap = cv2.VideoCapture(str(video_path))
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
    parser = argparse.ArgumentParser(description="Run CV pipeline on a video file")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--out", type=str, default="cv_metrics.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)

    pipeline = VisionPipeline()
    with out_path.open("w", encoding="utf-8") as f:
        for ts, frame in iter_frames(video_path):
            metrics = pipeline.analyze_frame(frame, ts)
            if not metrics:
                continue
            f.write(json.dumps(metrics.__dict__) + "\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
