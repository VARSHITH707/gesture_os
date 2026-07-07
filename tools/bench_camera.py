"""
Phase 1 — Camera benchmark.

Tests every requested mode against the actual webcam and reports
what the hardware really delivers. Does not touch any gesture code.

Usage:
    python tools/bench_camera.py
"""
import time
import cv2

DEVICE_INDEX = 0
MEASURE_SECONDS = 5.0   # how long to sample each mode

MODES = [
    {"label": "640x480 @ 30 FPS",  "w": 640,  "h": 480, "fps": 30},
    {"label": "640x480 @ 60 FPS",  "w": 640,  "h": 480, "fps": 60},
    {"label": "1280x720 @ 30 FPS", "w": 1280, "h": 720, "fps": 30},
    {"label": "1280x720 @ 60 FPS", "w": 1280, "h": 720, "fps": 60},
]


def _open(w, h, fps):
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


def _drain(cap, n=5):
    for _ in range(n):
        cap.read()


def _actual_resolution(cap):
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return aw, ah


def _reported_fps(cap):
    return cap.get(cv2.CAP_PROP_FPS)


def benchmark_mode(w, h, fps, label):
    print(f"\n--- {label} ---")
    cap = _open(w, h, fps)

    if not cap.isOpened():
        print("  FAILED to open camera")
        return None

    aw, ah = _actual_resolution(cap)
    reported = _reported_fps(cap)
    print(f"  Requested : {w}x{h} @ {fps} FPS")
    print(f"  Actual res: {aw}x{ah}")
    print(f"  Reported  : {reported:.1f} FPS  (cap.get CAP_PROP_FPS)")

    _drain(cap)

    # --- Synchronous read timing ---
    intervals = []
    read_times = []
    prev_t = None
    deadline = time.perf_counter() + MEASURE_SECONDS
    count = 0

    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        ret, _ = cap.read()
        t1 = time.perf_counter()

        if not ret:
            continue

        read_times.append((t1 - t0) * 1000)   # ms
        if prev_t is not None:
            intervals.append((t0 - prev_t) * 1000)  # ms between successive reads
        prev_t = t0
        count += 1

    cap.release()

    if count < 2:
        print("  Not enough frames captured.")
        return None

    avg_interval  = sum(intervals) / len(intervals)
    avg_read      = sum(read_times) / len(read_times)
    delivered_fps = 1000.0 / avg_interval if avg_interval > 0 else 0

    # Stale frame detection: frames with read_time < 1 ms were served from buffer
    stale = sum(1 for t in read_times if t < 1.0)
    max_interval = max(intervals)

    print(f"  Frames    : {count} in {MEASURE_SECONDS:.0f}s")
    print(f"  Delivered : {delivered_fps:.1f} FPS  (1000 / avg_interval)")
    print(f"  Avg interval  : {avg_interval:.1f} ms")
    print(f"  Max interval  : {max_interval:.1f} ms  (worst frame gap)")
    print(f"  Avg read time : {avg_read:.1f} ms  (how long cap.read() blocks)")
    print(f"  Stale frames  : {stale}/{count}  (read < 1 ms = buffer hit)")

    return {
        "label": label,
        "w": aw, "h": ah,
        "requested_fps": fps,
        "reported_fps": reported,
        "delivered_fps": delivered_fps,
        "avg_interval_ms": avg_interval,
        "max_interval_ms": max_interval,
        "avg_read_ms": avg_read,
        "stale_pct": 100.0 * stale / count,
        "frames": count,
    }


def pick_best(results):
    """Choose the mode with highest delivered FPS, breaking ties by resolution."""
    valid = [r for r in results if r is not None]
    if not valid:
        return None
    valid.sort(key=lambda r: (r["delivered_fps"], r["w"] * r["h"]), reverse=True)
    return valid[0]


def main():
    print("=" * 60)
    print("GestureOS — Phase 1 Camera Benchmark")
    print("=" * 60)

    results = []
    for mode in MODES:
        r = benchmark_mode(mode["w"], mode["h"], mode["fps"], mode["label"])
        results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<25} {'Req':>5} {'Report':>8} {'Actual':>8} {'Interval':>10} {'ReadT':>8} {'Stale':>7}")
    print("-" * 75)
    for r in results:
        if r is None:
            continue
        print(
            f"{r['label']:<25} "
            f"{r['requested_fps']:>5} "
            f"{r['reported_fps']:>8.1f} "
            f"{r['delivered_fps']:>8.1f} "
            f"{r['avg_interval_ms']:>9.1f}ms "
            f"{r['avg_read_ms']:>7.1f}ms "
            f"{r['stale_pct']:>6.1f}%"
        )

    best = pick_best(results)
    if best:
        print(f"\nBEST MODE: {best['label']}")
        print(f"  {best['w']}x{best['h']}  delivered {best['delivered_fps']:.1f} FPS")
        print(f"  avg frame interval {best['avg_interval_ms']:.1f} ms")
        print(f"  cap.read() blocks  {best['avg_read_ms']:.1f} ms on average")


if __name__ == "__main__":
    main()
