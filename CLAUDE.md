# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Run with default webcam (index 0)
python main.py

# Run with a specific webcam index
python main.py --cam 1

# Run with a video file
python main.py --video /path/to/video.mp4
```

**Runtime controls** (in the OpenCV window):
- `c` — calibrate the neutral head position (center)
- `q` or `ESC` — quit

To find available camera indices on the machine:
```bash
python testa_cameras.py
```

## Installing Dependencies

```bash
pip install -r requirements.txt
git lfs pull  # downloads the ONNX model files in assets/
```

Note: `requirements.txt` pins `onnxruntime-gpu`. If no GPU is available, replace it with `onnxruntime`. Both `CUDAExecutionProvider` and `CPUExecutionProvider` are listed in the session providers, so it will fall back to CPU automatically if CUDA is unavailable.

`pyautogui` is used in `main.py` but is not listed in `requirements.txt` — install it separately if needed:
```bash
pip install pyautogui
```

## Architecture

The pipeline has three sequential stages, each in its own module:

1. **[face_detection.py](face_detection.py)** — `FaceDetector` wraps an ONNX SCRFD model (`assets/face_detector.onnx`). Given a full camera frame, it returns bounding boxes and optional keypoints. Detection is expensive and is only re-run when the tracked face is lost or after 30 frames without re-detection (see `main.py:67`).

2. **[mark_detection.py](mark_detection.py)** — `MarkDetector` wraps an ONNX CNN model (`assets/face_landmarks.onnx`) that takes a cropped face patch and returns 68 facial landmark points (x, y) normalized to [0, 1] relative to the patch.

3. **[pose_estimation.py](pose_estimation.py)** — `PoseEstimator` uses OpenCV's `solvePnP` (with temporal tracking via `useExtrinsicGuess=True`) to compute a 6-DoF head pose from the 68 landmarks against a 3D face model loaded from `assets/model.txt`. Returns a rotation vector and translation vector.

**[main.py](main.py)** orchestrates the pipeline and adds mouse control:
- Runs the full detector only when tracking is lost; otherwise reuses and updates the bounding box each frame using the landmark geometry (prevents the "black hole" drift effect).
- Converts Euler angles (yaw/pitch from `cv2.RQDecomp3x3`) into mouse movement via `pyautogui`, with a deadzone, acceleration curve, and dynamic smoothing factor.
- Calibration sets the current head angles as the neutral reference point (`neutral_pitch`, `neutral_yaw`).

**[utils.py](utils.py)** — single helper `refine()` that squarifies and shifts face bounding boxes before passing them to the landmark detector.

## Key Parameters (main.py)

| Variable | Default | Effect |
|---|---|---|
| `deadzone` | `0.8` deg | Minimum angle to trigger mouse movement |
| `max_yaw_deg` | `18.0` | Full horizontal screen range |
| `max_pitch_deg` | `12.0` | Full vertical screen range |
| `acceleration_curve` | `1.4` | Non-linear mapping exponent |
| `history_pitch/yaw` | `deque(maxlen=5)` | Temporal smoothing window |
