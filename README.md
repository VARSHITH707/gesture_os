# GestureOS

Real-time hand-gesture control for Windows. A webcam replaces your mouse: move the cursor, click, drag, and scroll with bare-hand gestures at a stable 30 FPS.

## How it works

- **MediaPipe hand landmarks** (21-point model) feed a **finite-state-machine gesture classifier** (`gesture_os/input/gesture_fsm.py`)
- **One Euro filtering** smooths cursor motion for low-latency, jitter-free tracking
- Gesture thresholds are **normalized by hand size** (wrist-to-middle-MCP distance), so sensitivity stays consistent whether your hand is near or far from the camera
- Scroll uses hysteresis + throttled dispatch to prevent misfires and event flooding

## Gestures

| Gesture | Action |
|---|---|
| Index finger point | Move cursor |
| Thumb-index pinch | Left click (double-tap for right click) |
| Pinch and hold | Drag |
| Index + middle extended | Scroll |

## Run it

```
pip install -r requirements.txt   # or: pip install mediapipe opencv-python pynput
python main.py
```

## Tests

130 tests cover the FSM, filters, and gesture classification — all headless, no camera needed:

```
python -m pytest tests -q
```

Built with Python, MediaPipe, OpenCV, pynput.
