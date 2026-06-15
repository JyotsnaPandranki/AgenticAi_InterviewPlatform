from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# Update this path to your JSONL output
METRICS_PATH = Path("cv_metrics.jsonl")


def iter_metrics(path: Path) -> Iterable[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    if not METRICS_PATH.exists():
        raise SystemExit("METRICS_PATH does not exist")

    eye = []
    gaze = []
    head_pose = []
    head_yaw = []
    head_pitch = []
    engagement = []
    stress = []
    posture = []
    posture_instability = []
    blink = []
    perclos = []
    fidget = []
    emotion_conf = []
    emotion_present = 0

    total = 0
    for item in iter_metrics(METRICS_PATH):
        total += 1
        eye.append(item.get("eye_contact_score", 0.0))
        gaze.append(item.get("gaze_score", 0.0))
        head_pose.append(item.get("head_pose_score", 0.0))
        head_yaw.append(item.get("head_pose_yaw", 0.0))
        head_pitch.append(item.get("head_pose_pitch", 0.0))
        engagement.append(item.get("engagement_score", 0.0))
        stress.append(item.get("stress_score", 0.0))
        posture.append(item.get("posture_score", 0.0))
        posture_instability.append(item.get("posture_instability", 0.0))
        blink.append(item.get("blink_rate", 0.0))
        perclos.append(item.get("perclos", 0.0))
        fidget.append(item.get("fidget_score", 0.0))
        conf = item.get("emotion_confidence", 0.0)
        if item.get("emotion"):
            emotion_present += 1
            emotion_conf.append(conf)

    if total == 0:
        raise SystemExit("No metrics found in METRICS_PATH")

    def summarize(values: list[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    eye_mean, eye_std = summarize(eye)
    gaze_mean, gaze_std = summarize(gaze)
    head_mean, head_std = summarize(head_pose)
    yaw_mean, yaw_std = summarize(head_yaw)
    pitch_mean, pitch_std = summarize(head_pitch)
    eng_mean, eng_std = summarize(engagement)
    stress_mean, stress_std = summarize(stress)
    posture_mean, posture_std = summarize(posture)
    inst_mean, inst_std = summarize(posture_instability)
    blink_mean, blink_std = summarize(blink)
    perclos_mean, perclos_std = summarize(perclos)
    fidget_mean, fidget_std = summarize(fidget)

    print(f"frames={total}")
    print(f"eye_contact_mean={eye_mean:.3f} std={eye_std:.3f}")
    print(f"gaze_mean={gaze_mean:.3f} std={gaze_std:.3f}")
    print(f"head_pose_mean={head_mean:.3f} std={head_std:.3f}")
    print(f"head_yaw_mean={yaw_mean:.2f} std={yaw_std:.2f}")
    print(f"head_pitch_mean={pitch_mean:.2f} std={pitch_std:.2f}")
    print(f"engagement_mean={eng_mean:.3f} std={eng_std:.3f}")
    print(f"stress_mean={stress_mean:.3f} std={stress_std:.3f}")
    print(f"posture_mean={posture_mean:.3f} std={posture_std:.3f}")
    print(f"posture_instability_mean={inst_mean:.3f} std={inst_std:.3f}")
    print(f"blink_rate_mean={blink_mean:.2f} std={blink_std:.2f}")
    print(f"perclos_mean={perclos_mean:.3f} std={perclos_std:.3f}")
    print(f"fidget_mean={fidget_mean:.3f} std={fidget_std:.3f}")
    print(f"emotion_frames={emotion_present}/{total}")
    if emotion_conf:
        print(f"emotion_conf_mean={float(np.mean(emotion_conf)):.3f}")


if __name__ == "__main__":
    main()
