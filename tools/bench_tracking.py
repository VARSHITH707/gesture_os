"""
Phase 2 — Hand-tracking latency benchmark.

Opens camera at the best mode (640x480@30), feeds real frames through
MediaPipe HandLandmarker VIDEO mode, and reports timing for each stage.
No gesture FSM. No cursor. No display window.

Usage:
    python tools/bench_tracking.py
"""
import os, sys, time
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE   = 0
WIDTH    = 640
HEIGHT   = 480
REQ_FPS  = 30
DURATION = 10.0   # seconds to sample
MODEL    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "hand_landmarker.task")


def open_cap():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          REQ_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    for _ in range(5):
        cap.read()
    return cap


def make_detector():
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opts)


def run():
    print("Opening camera...")
    cap = open_cap()
    print("Loading MediaPipe model...")
    det = make_detector()

    cam_times   = []   # how long cap.read() blocks (ms)
    mp_times    = []   # MediaPipe detect_for_video (ms)
    total_times = []   # cam + mp together (ms)
    hand_count  = 0
    frame_n     = 0
    prev_t      = None
    intervals   = []

    print(f"Sampling {DURATION:.0f}s — keep hand in frame for realistic numbers...\n")
    deadline = time.perf_counter() + DURATION

    while time.perf_counter() < deadline:
        # --- camera read ---
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            continue

        cam_ms = (t1 - t0) * 1000.0
        cam_times.append(cam_ms)

        # frame interval
        if prev_t is not None:
            intervals.append((t0 - prev_t) * 1000.0)
        prev_t = t0

        # --- MediaPipe ---
        from mediapipe import Image, ImageFormat
        frame_rgb = frame[:, :, ::-1].copy()
        mp_img    = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        ts_ms     = int(t0 * 1000)

        t2 = time.perf_counter()
        result = det.detect_for_video(mp_img, ts_ms)
        t3 = time.perf_counter()

        mp_ms = (t3 - t2) * 1000.0
        mp_times.append(mp_ms)
        total_times.append(cam_ms + mp_ms)

        if result.hand_landmarks:
            hand_count += 1
        frame_n += 1

    cap.release()
    det.close()

    if frame_n < 2:
        print("Not enough frames.")
        return

    def stats(lst, name):
        avg = sum(lst) / len(lst)
        mx  = max(lst)
        mn  = min(lst)
        p95 = sorted(lst)[int(len(lst) * 0.95)]
        print(f"  {name:<28} avg={avg:6.1f}ms  min={mn:5.1f}ms  max={mx:6.1f}ms  p95={p95:6.1f}ms")

    print("=" * 60)
    print(f"Frames processed : {frame_n}")
    print(f"Hands detected   : {hand_count}  ({100*hand_count/frame_n:.0f}%)")
    if intervals:
        delivered_fps = 1000.0 / (sum(intervals)/len(intervals))
        print(f"Delivered FPS    : {delivered_fps:.1f}")
    print()
    stats(cam_times,   "Camera cap.read()")
    stats(mp_times,    "MediaPipe inference")
    stats(total_times, "Total per-frame (cam+mp)")
    print()
    avg_cam = sum(cam_times)   / len(cam_times)
    avg_mp  = sum(mp_times)    / len(mp_times)
    print(f"  Bottleneck: camera={avg_cam:.1f}ms  mediapipe={avg_mp:.1f}ms  "
          f"=> {'CAMERA' if avg_cam > avg_mp else 'MEDIAPIPE'}")


if __name__ == "__main__":
    run()
