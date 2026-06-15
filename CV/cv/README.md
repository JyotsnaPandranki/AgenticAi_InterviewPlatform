# Computer Vision Module

This module provides on-device visual intelligence for the mock interview agent.

## Outputs
- Engagement score
- Stress indicator
- Eye contact and posture signals
- Head pose, PERCLOS (eye closure), and fidget stability

## Quick Start
1. Install dependencies from requirements-cv.txt.
2. Run on a video file:

```
python -m cv.cli path/to/video.mp4 --out cv_metrics.jsonl
```

## Notes
- Uses MediaPipe for face/pose landmarks.
- Emotion model is optional and supports ONNX FER+ weights.

## MediaPipe Versions
- If your MediaPipe build includes `solutions`, the default pipeline works.
- If not, set `MEDIAPIPE_FACE_MODEL` and `MEDIAPIPE_POSE_MODEL` to .task model files
	or set `FACE_MODEL_PATH`/`POSE_MODEL_PATH` in run_cv.py.

## Pretrained Emotion Model (FER+)
1. Download the FER+ ONNX model (emotion-ferplus-8.onnx) from the ONNX Model Zoo.
2. Set `EMOTION_MODEL_PATH` in run_cv.py to the downloaded file.
3. Run `run_emotion_eval.py` to get basic accuracy and confidence stats on a labeled folder.
