"""
GestureOS — CURSOR + LEFT CLICK TEST MODE

What works:
  - Move index finger -> cursor follows
  - Index-thumb pinch and release -> left click
  - Hand leaves frame -> cursor freezes, any held state released
  - Hand returns -> no jump

All disabled:
  - Right click, drag, scroll, swipe, Wispr, fist, pause

Click logic (no FSM import):
  - Pinch detected when thumb-index distance < PINCH_ENTER threshold
  - Must stay pinched for DEBOUNCE_MS before it counts
  - Click fires on release (pinch_open after debounce)
  - SETTLE_MS window after release before next click can start
    (prevents accidental double-fire from a single tap)

Debug overlay shows pinch distance and click state.

Press Q in the debug window to quit.
"""
import os, sys, time, logging
os.environ["GLOG_minloglevel"]    = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

# --- project path ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gesture_os.vision.camera import ThreadedCamera, CameraConfig
from gesture_os.input.cursor import RelativeCursorController, CursorConfig
from gesture_os.input.mouse import MouseController, build_monitor_layout
from gesture_os.vision.filters import OneEuroState2D, filter_point_2d

# ================================================================
# TUNABLE CONSTANTS — adjust these to find the sweet spot
# ================================================================
FILTER_MIN_CUTOFF = 0.4    # LOWER = more smoothing when hand is still (0.1–3.0)
FILTER_BETA       = 0.06   # higher = less lag when moving intentionally (0.01–0.5)
FILTER_D_CUTOFF   = 1.0    # derivative smoothing (keep 0.5–2.0)

CURSOR_SENSITIVITY = 2.5   # pixels-per-normalized-unit-per-pixel-screen
CURSOR_DEAD_ZONE   = 0.008 # LARGER = more movement ignored as jitter (0.002–0.02)
CURSOR_MAX_DELTA   = 300   # px; clamp single-frame movement

SHOW_DEBUG_WINDOW  = True  # set False to run headless (for latency-only test)

CAMERA_WIDTH       = 640
CAMERA_HEIGHT      = 480
CAMERA_FPS         = 30

# --- Pinch / tap tuning ---
PINCH_ENTER     = 0.06   # normalized dist; below = pinch detected
PINCH_EXIT      = 0.10   # above = pinch released (hysteresis)
DEBOUNCE_MS     = 120.0  # hold pinch this long to confirm it (raised to filter tremor)
DOUBLE_TAP_MS   = 320.0  # second pinch within this window = right click
DRAG_MS         = 500.0  # hold confirmed pinch this long = drag
SETTLE_MS       = 350.0  # cooldown after any click/drag before next action (raised to prevent re-fire)
PINCH_FRAMES    = 2      # consecutive frames thumb must be close before entering debounce

# --- Scroll tuning ---
SCROLL_SPEED        = 80.0   # scale: normalized-delta * SCROLL_SPEED = ticks
SCROLL_MIN_VELOCITY = 0.25   # normalized units/sec; slower = repositioning (ignored)
SCROLL_RATE_MS      = 150.0  # minimum ms between scroll events (prevents snap-physics fight)
SCROLL_EMA_ALPHA    = 0.3    # EMA smoothing on scroll Y (0.2=smooth, 0.5=responsive)
# ================================================================


logging.basicConfig(
    level=logging.WARNING,   # suppress routine INFO during cursor test
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)


def make_detector():
    model = os.path.join(_ROOT, "hand_landmarker.task")
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opts)


def extract_index_tip(result):
    """Return (x, y) of index tip (landmark 8) normalized, x-flipped for mirror."""
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    x = 1.0 - lm[8].x
    y = lm[8].y
    return x, y


def thumb_index_dist(result):
    """Return normalized 2D distance between thumb tip (4) and index tip (8)."""
    if not result.hand_landmarks:
        return 1.0
    lm = result.hand_landmarks[0]
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    return (dx * dx + dy * dy) ** 0.5


def scroll_pose_y(result):
    """Return (is_scroll_pose, avg_y_of_index_and_middle_tips).

    Scroll pose: index + middle extended (tip.y < pip.y),
                 ring + pinky curled (tip.y > pip.y), no pinch.
    """
    if not result.hand_landmarks:
        return False, 0.0
    lm = result.hand_landmarks[0]
    idx_ext = lm[8].y  < lm[6].y   # index extended
    mid_ext = lm[12].y < lm[10].y  # middle extended
    rng_cur = lm[16].y > lm[14].y  # ring curled
    pky_cur = lm[20].y > lm[18].y  # pinky curled
    # Scroll pose: index+middle extended, ring+pinky curled.
    # Do NOT gate on thumb-index distance — scroll takes explicit priority over
    # pinch in tick(), so the thumb being near the index during a peace-sign
    # must NOT suppress scroll detection (that's exactly the bug it was causing).
    is_scroll = idx_ext and mid_ext and rng_cur and pky_cur
    avg_y = (lm[8].y + lm[12].y) * 0.5
    return is_scroll, avg_y


# ------------------------------------------------------------------ #
# Gesture detector
#
# States:
#   OPEN       — hand open, cursor moves freely
#   DEBOUNCE   — pinch just started; waiting to confirm
#   HELD       — confirmed pinch; waiting for drag timeout or release
#   DRAG       — pinch held long enough; mouse button is pressed
#   TAP_SETTLE — pinch released; waiting for possible double-tap
#   COOLING    — after any click/drag; short cooldown
#   SCROLL     — two-finger scroll pose active
#
# Cursor moves in:  OPEN, TAP_SETTLE, COOLING, DRAG
# Cursor frozen in: DEBOUNCE, HELD, SCROLL
# ------------------------------------------------------------------ #
_G_OPEN       = "open"
_G_DEBOUNCE   = "debounce"
_G_HELD       = "held"
_G_DRAG       = "drag"
_G_TAP_SETTLE = "tap_settle"
_G_COOLING    = "cooling"
_G_SCROLL     = "scroll"

from dataclasses import dataclass as _dc, field as _field_dc

@_dc
class GestureOutput:
    left_click:    bool  = False
    right_click:   bool  = False
    mouse_press:   bool  = False   # drag start
    mouse_release: bool  = False   # drag end
    scroll_dy:     float = 0.0     # non-zero when scrolling
    cursor_active: bool  = True    # False = freeze cursor


class GestureDetector:
    def __init__(self):
        self.state        = _G_OPEN
        self._entered     = 0.0
        self._scroll_ref  = 0.0   # y at last frame inside SCROLL
        self._scroll_t    = 0.0   # timestamp of last scroll frame
        self._pinch_frames  = 0    # consecutive frames thumb-index below PINCH_ENTER
        self._last_scroll_t = 0.0  # wall-clock time of last emitted scroll event
        self._scroll_ema    = 0.0  # EMA-smoothed scroll Y position

    def _ms(self, now):
        return (now - self._entered) * 1000.0

    def tick(self, dist: float, is_scroll: bool, scroll_y: float, now: float) -> GestureOutput:
        out      = GestureOutput()
        pinching = dist < PINCH_ENTER
        released = dist > PINCH_EXIT

        # Scroll overrides pinch in any non-committed state.
        # _G_DEBOUNCE is included so a partial pinch that forms into a scroll
        # pose aborts cleanly instead of eventually firing a left click.
        if self.state in (_G_OPEN, _G_DEBOUNCE, _G_COOLING, _G_TAP_SETTLE):
            if is_scroll:
                self._pinch_frames = 0    # clear any partial pinch accumulation
                self.state        = _G_SCROLL
                self._entered     = now
                self._scroll_ref  = scroll_y
                self._scroll_t    = now
                out.cursor_active = False
                return out

        if self.state == _G_OPEN:
            out.cursor_active = True
            if pinching:
                self._pinch_frames += 1
                if self._pinch_frames >= PINCH_FRAMES:
                    self.state    = _G_DEBOUNCE
                    self._entered = now
            else:
                self._pinch_frames = 0

        elif self.state == _G_DEBOUNCE:
            out.cursor_active = False
            if released:
                # too fast — noise, ignore
                self.state    = _G_COOLING
                self._entered = now
            elif self._ms(now) >= DEBOUNCE_MS:
                self.state = _G_HELD

        elif self.state == _G_HELD:
            out.cursor_active = False
            if released:
                # released before drag timeout → potential tap
                self.state    = _G_TAP_SETTLE
                self._entered = now
            elif self._ms(now) >= DRAG_MS:
                # held long enough → start drag
                self.state    = _G_DRAG
                self._entered = now
                out.mouse_press = True

        elif self.state == _G_DRAG:
            out.cursor_active = True   # cursor moves while dragging
            if released:
                out.mouse_release = True
                self.state        = _G_COOLING
                self._entered     = now

        elif self.state == _G_TAP_SETTLE:
            out.cursor_active = True
            if pinching:
                # second pinch → right click
                out.right_click = True
                self.state      = _G_COOLING
                self._entered   = now
            elif self._ms(now) >= DOUBLE_TAP_MS:
                # window expired → left click
                out.left_click = True
                self.state     = _G_COOLING
                self._entered  = now

        elif self.state == _G_COOLING:
            out.cursor_active = True
            if self._ms(now) >= SETTLE_MS:
                self.state = _G_OPEN

        elif self.state == _G_SCROLL:
            out.cursor_active = False
            if not is_scroll:
                self.state = _G_OPEN
            else:
                # EMA-smooth the raw Y to kill single-frame landmark jitter
                # before computing velocity.  First frame: seed EMA to current Y.
                if self._scroll_t == 0.0:
                    self._scroll_ema = scroll_y
                else:
                    self._scroll_ema = (SCROLL_EMA_ALPHA * scroll_y +
                                        (1.0 - SCROLL_EMA_ALPHA) * self._scroll_ema)

                dt = max(now - self._scroll_t, 1e-4)
                dy = self._scroll_ema - self._scroll_ref
                velocity = abs(dy) / dt
                self._scroll_ref = self._scroll_ema
                self._scroll_t   = now

                # Gate 1: velocity — ignore slow repositioning strokes.
                # Gate 2: rate limiter — max 1 scroll event per SCROLL_RATE_MS
                #         so snap-physics apps (Instagram) don't receive 15 events/s.
                rate_ok = (now - self._last_scroll_t) * 1000.0 >= SCROLL_RATE_MS
                if velocity > SCROLL_MIN_VELOCITY and rate_ok:
                    out.scroll_dy       = dy * SCROLL_SPEED
                    self._last_scroll_t = now

        return out

    def reset(self):
        self.state          = _G_OPEN
        self._entered       = 0.0
        self._scroll_ref    = 0.0
        self._scroll_t      = 0.0
        self._pinch_frames  = 0
        self._last_scroll_t = 0.0
        self._scroll_ema    = 0.0


_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_TIP_COLORS = {4:(0,200,255), 8:(0,80,255), 12:(255,80,0), 16:(255,0,180), 20:(0,255,120)}

def draw_landmarks(frame, result, pinching=False):
    """Draw hand skeleton lightly — kept fast (one pass, no outlines)."""
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    lc = (60, 60, 60) if pinching else (160, 160, 160)
    for hand in result.hand_landmarks:
        pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in hand]
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], lc, 1)
        for i, pt in enumerate(pts):
            color = _TIP_COLORS.get(i, (0, 200, 0))
            r = 7 if i in _TIP_COLORS else 3
            cv2.circle(frame, pt, r, color, -1)


def overlay_text(frame, lines):
    """Draw lines of text in top-left of frame."""
    y = 20
    for line in lines:
        cv2.putText(frame, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
        y += 20


_STATE_LABELS = {
    _G_OPEN:       ("MOVING",           (0, 220, 0)),
    _G_DEBOUNCE:   ("PINCH DETECTED",   (0, 200, 255)),
    _G_HELD:       ("HOLD FOR DRAG...", (0, 160, 255)),
    _G_DRAG:       ("DRAGGING",         (0, 80, 255)),
    _G_TAP_SETTLE: ("RELEASE - TAP?",   (0, 200, 255)),
    _G_COOLING:    ("COOLING",          (180, 180, 0)),
    _G_SCROLL:     ("SCROLLING",        (255, 160, 0)),
}


def main():
    print("GestureOS CURSOR-ONLY TEST MODE")
    print(f"  Filter: min_cutoff={FILTER_MIN_CUTOFF}  beta={FILTER_BETA}")
    print(f"  Sensitivity: {CURSOR_SENSITIVITY}  dead_zone: {CURSOR_DEAD_ZONE}")
    print("  Press Q in the preview window to quit.")
    print()

    import threading

    cam_cfg = CameraConfig(
        width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
        fps=CAMERA_FPS, buffer_size=1,
    )
    print("Opening camera...")
    cam = ThreadedCamera(cam_cfg)

    # Load MediaPipe on a background thread so the camera preview is visible
    # immediately. This hides the ~4-second cold-start lag behind a live feed.
    _det_box   = [None]
    _det_err   = [None]
    _det_ready = threading.Event()

    def _bg_load():
        try:
            _det_box[0] = make_detector()
        except Exception as exc:
            _det_err[0] = exc
        finally:
            _det_ready.set()

    threading.Thread(target=_bg_load, daemon=True).start()

    print("MediaPipe loading in background — camera preview active...")

    # Show a live camera feed with loading overlay until the model is ready.
    from mediapipe import Image, ImageFormat
    while not _det_ready.is_set():
        ok, frame, _ = cam.read_blocking(timeout=0.08)
        if ok and frame is not None and SHOW_DEBUG_WINDOW:
            display = cv2.flip(frame, 1)
            h_d, w_d = display.shape[:2]
            cv2.putText(display, "Initializing gesture detector...",
                        (20, h_d // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "Please wait (first run ~5 seconds)",
                        (20, h_d // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("GestureOS Cursor Test", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            cam.release()
            cv2.destroyAllWindows()
            return

    if _det_err[0]:
        print(f"ERROR loading MediaPipe detector: {_det_err[0]}")
        cam.release()
        return

    det = _det_box[0]
    print("Detector ready.")

    layout = build_monitor_layout()
    mouse  = MouseController()

    cur_cfg = CursorConfig(
        sensitivity=CURSOR_SENSITIVITY,
        dead_zone=CURSOR_DEAD_ZONE,
        max_delta_px=CURSOR_MAX_DELTA,
    )
    cursor  = RelativeCursorController(cur_cfg, layout.total_width, layout.total_height)

    # Build filter with tuned params
    from gesture_os.vision.filters import OneEuroState, LowPassState
    from dataclasses import field as _f

    def _make_state():
        s = OneEuroState()
        s.min_cutoff = FILTER_MIN_CUTOFF
        s.beta       = FILTER_BETA
        s.d_cutoff   = FILTER_D_CUTOFF
        return s

    filt = OneEuroState2D(x_state=_make_state(), y_state=_make_state())

    tap  = GestureDetector()

    # Perf tracking — fixed-size deques avoid per-frame list allocation
    from collections import deque
    cam_times   = deque(maxlen=30)
    mp_times    = deque(maxlen=30)
    total_times = deque(maxlen=30)
    intervals   = deque(maxlen=30)
    prev_frame_t = -1.0
    fps_display  = 0.0
    fps_count     = 0
    fps_t         = time.perf_counter()
    hand_visible  = False
    prev_pinching = False

    print("Running.\n")
    print("  Open hand        = cursor moves")
    print("  Pinch + release  = left click")
    print("  Double pinch     = right click")
    print("  Pinch + hold 0.5s + move = drag")
    print("  Index+middle up, ring+pinky curled = scroll\n")

    while True:
        t0 = time.perf_counter()

        ok, frame, frame_t = cam.read_blocking(timeout=0.08)
        t1 = time.perf_counter()

        if not ok or frame is None:
            continue

        # Skip duplicate frames
        if frame_t <= prev_frame_t:
            continue
        if prev_frame_t > 0:
            intervals.append((frame_t - prev_frame_t) * 1000.0)
        prev_frame_t = frame_t

        cam_ms = (t1 - t0) * 1000.0
        cam_times.append(cam_ms)

        # --- MediaPipe ---
        frame_rgb = frame[:, :, ::-1]  # no copy — view only; MediaPipe reads it
        mp_img    = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        ts_ms     = int(frame_t * 1000)

        t2 = time.perf_counter()
        result = det.detect_for_video(mp_img, ts_ms)
        t3 = time.perf_counter()

        mp_ms = (t3 - t2) * 1000.0
        mp_times.append(mp_ms)

        now = time.perf_counter()
        total_times.append((now - t0) * 1000.0)

        # --- Inputs ---
        tip              = extract_index_tip(result)
        dist             = thumb_index_dist(result)
        is_scroll, scr_y = scroll_pose_y(result)

        if tip is not None:
            hand_visible = True
        else:
            if hand_visible:
                cursor.on_hand_lost()
                tap.reset()
            hand_visible  = False
            prev_pinching = False

        # --- Gesture tick ---
        g = GestureOutput()
        if hand_visible:
            g = tap.tick(dist, is_scroll, scr_y, now)

        # --- Apply outputs ---
        if g.left_click:
            mouse.click("left")
            print(f"  LEFT CLICK")
        if g.right_click:
            mouse.click("right")
            print(f"  RIGHT CLICK")
        if g.mouse_press:
            mouse.press("left")
            print(f"  DRAG START")
        if g.mouse_release:
            mouse.release("left")
            print(f"  DRAG END")
        if g.scroll_dy:
            mouse.scroll(g.scroll_dy)

        # --- Cursor ---
        pinching = not g.cursor_active
        if hand_visible and tip is not None:
            if pinching:
                if not prev_pinching:
                    cursor.on_hand_lost()   # clear ref so no jump on release
            else:
                fx, fy = filter_point_2d(filt, tip, now)
                px, py = cursor.update(fx, fy)
                mouse.move_to(px, py)

        prev_pinching = pinching

        fps_count += 1
        elapsed = now - fps_t
        if elapsed >= 1.5:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_t       = now

        # --- Debug window ---
        if SHOW_DEBUG_WINDOW:
            # Flip horizontally so the preview is a natural mirror view.
            # Landmark x coords are already drawn as (1-lm.x)*w, which
            # aligns correctly with the flipped frame.
            display = cv2.flip(frame, 1)
            draw_landmarks(display, result, pinching=pinching)

            n = max(len(cam_times), 1)
            avg_cam = sum(cam_times) / n
            avg_mp  = sum(mp_times)  / n
            avg_tot = sum(total_times) / n
            avg_int = sum(intervals) / max(len(intervals), 1) if intervals else 0.0
            del_fps = 1000.0 / avg_int if avg_int > 0 else 0.0

            hand_str = "HAND" if hand_visible else "NO HAND"
            lines = [
                f"FPS:{fps_display:.0f} cam:{avg_cam:.0f}ms mp:{avg_mp:.0f}ms",
                f"{hand_str} pinch:{dist:.3f} {tap.state}",
                "Q=quit",
            ]
            overlay_text(display, lines)

            h_d, w_d = display.shape[:2]
            banner, color = _STATE_LABELS.get(tap.state, ("NO HAND", (0, 0, 200))) \
                            if hand_visible else ("NO HAND", (0, 0, 200))
            cv2.putText(display, banner, (w_d // 2 - 160, h_d - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),  3, cv2.LINE_AA)
            cv2.putText(display, banner, (w_d // 2 - 160, h_d - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,       1, cv2.LINE_AA)

            cv2.imshow("GestureOS Cursor Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cam.release()
    det.close()
    if SHOW_DEBUG_WINDOW:
        cv2.destroyAllWindows()

    # --- Final report ---
    def _stats(lst, name):
        if not lst:
            return
        avg = sum(lst) / len(lst)
        p95 = sorted(lst)[int(len(lst) * 0.95)]
        mx  = max(lst)
        print(f"  {name:<28} avg={avg:6.1f}ms  p95={p95:6.1f}ms  max={mx:6.1f}ms")

    print("\n--- Session latency ---")
    _stats(cam_times,   "Camera wait")
    _stats(mp_times,    "MediaPipe inference")
    _stats(total_times, "Total per-frame")
    if intervals:
        avg_i = sum(intervals) / len(intervals)
        print(f"  {'Delivered FPS':<28} {1000/avg_i:.1f} FPS  (avg interval {avg_i:.1f}ms)")


if __name__ == "__main__":
    main()
