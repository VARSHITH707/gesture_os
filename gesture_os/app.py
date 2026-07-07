"""GestureOS main application orchestrator.

Owns all components, runs the camera+gesture loop, applies FSM output,
and exposes pause/resume to the system tray.
"""
import logging
import threading
import time

import cv2
import numpy as np

from gesture_os.config import AppConfig
from gesture_os.input.cursor import RelativeCursorController
from gesture_os.input.gesture_fsm import (
    GestureFSM,
    GestureMode,
    extract_features,
    reset_fsm,
    tick_fsm,
)
from gesture_os.input.keyboard import (
    release_all_modifiers,
    send_alt_tab_backward,
    send_alt_tab_forward,
    send_ctrl_alt,
    send_win_m,
    send_win_shift_m,
)
from gesture_os.input.mouse import MouseController, build_monitor_layout
from gesture_os.tray import SystemTray
from gesture_os.vision.camera import CameraConfig, ThreadedCamera
from gesture_os.vision.filters import OneEuroState2D, filter_point_2d
from gesture_os.vision.landmarks import create_hands_detector, extract_landmarks

logger = logging.getLogger(__name__)


class GestureApp:
    """Complete GestureOS application. Call run() to start."""

    # Minimum gap between physical wheel events. Throttles dispatch so we
    # don't send a scroll tick every camera frame and fight apps (e.g.
    # Instagram) that apply their own snap-physics on rapid wheel input.
    _SCROLL_DISPATCH_INTERVAL_S = 0.045

    def __init__(self, config: AppConfig | None = None) -> None:
        self._cfg = config or AppConfig()
        self._lock = threading.Lock()

        self._active = True
        self._running = False

        # Precompute screen dimensions
        self._layout = build_monitor_layout()
        sw = self._layout.total_width
        sh = self._layout.total_height

        # All components initialised here, not inside the frame loop
        self._mouse = MouseController()
        self._cursor = RelativeCursorController(self._cfg.cursor, sw, sh)
        self._fsm = GestureFSM(config=self._cfg.gesture)
        self._filter = OneEuroState2D()

        # Performance tracking (ring buffer, 300 samples)
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._latencies: list[float] = []
        self._cam_fps: float = 0.0

        # Scroll dispatch throttling state
        self._scroll_accum: int = 0
        self._last_scroll_t: float = 0.0

        self._tray: SystemTray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pause(self) -> None:
        with self._lock:
            if self._active:
                self._active = False
                self._safe_release_all()
                reset_fsm(self._fsm)
                self._cursor.reset()
                logger.info("GestureOS paused.")

    def resume(self) -> None:
        with self._lock:
            if not self._active:
                self._active = True
                reset_fsm(self._fsm)
                self._cursor.reset()
                logger.info("GestureOS resumed.")

    def is_active(self) -> bool:
        return self._active

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self._start_tray()

        lm_cfg = self._cfg.landmark
        cam = ThreadedCamera(self._cfg.camera)
        detector = create_hands_detector(lm_cfg)

        # Measure real delivered FPS
        self._cam_fps = cam.measure_fps(1.5)
        logger.info(
            "GestureOS started — camera %.1f FPS  screen %dx%d",
            self._cam_fps,
            self._layout.total_width,
            self._layout.total_height,
        )

        try:
            self._loop(cam, detector)
        finally:
            self._safe_release_all()
            release_all_modifiers()
            cam.release()
            detector.close()
            if self._cfg.show_debug_window:
                cv2.destroyAllWindows()
            if self._tray:
                self._tray.stop()
            if self._latencies:
                avg = sum(self._latencies) / len(self._latencies) * 1000
                logger.info(
                    "Shutdown — frames=%d  FPS=%.1f  avg_lat=%.1fms",
                    self._frame_count, self._fps, avg,
                )

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _loop(self, cam: ThreadedCamera, detector) -> None:
        cfg = self._cfg
        prev_hand = False
        fps_count = 0
        fps_t = time.perf_counter()
        last_frame_t = -1.0

        while self._running:
            t0 = time.perf_counter()

            # Use blocking read so we don't spin-waste CPU between frames
            ok, frame, frame_t = cam.read_blocking(timeout=0.05)
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            # Skip duplicate frames — same _frame_t means camera hasn't
            # delivered a new frame yet; passing duplicate ts_ms to MediaPipe
            # would violate its monotonic-timestamp requirement.
            if frame_t <= last_frame_t:
                continue
            last_frame_t = frame_t

            now = time.perf_counter()
            ts_ms = int(frame_t * 1000)

            hands = extract_landmarks(frame, detector, ts_ms)

            with self._lock:
                active = self._active

            if not hands:
                if prev_hand:
                    self._cursor.on_hand_lost()
                    reset_fsm(self._fsm)
                prev_hand = False
                self._maybe_debug(frame, None, None)
                continue

            prev_hand = True
            hand = hands[0]
            feat = extract_features(hand.landmarks, cfg.gesture, now)

            # Smooth index tip
            sx, sy = filter_point_2d(
                self._filter, (feat.index_tip_x, feat.index_tip_y), now
            )

            if not active:
                self._cursor.on_hand_lost()
                self._maybe_debug(frame, feat, None)
                continue

            out = tick_fsm(self._fsm, feat, now)

            # Pause toggle from 2-second fist
            if out.toggle_pause:
                self.pause()
                if self._tray:
                    self._tray.update_icon()
                continue

            # Cursor
            if out.cursor_active:
                px, py = self._cursor.update(sx, sy)
                self._mouse.move_to(px, py)
            else:
                self._cursor.on_hand_lost()

            # Mouse press/release (drag)
            if out.mouse_press:
                self._mouse.press("left")
                logger.info("DRAG START")
            if out.mouse_release:
                self._mouse.release("left")
                logger.info("DRAG END")

            # Click actions
            if out.left_click:
                self._mouse.click("left")
                logger.info("LEFT CLICK")
            if out.right_click:
                self._mouse.click("right")
                logger.info("RIGHT CLICK (double tap)")

            # Scroll — accumulate every frame, dispatch at a throttled
            # interval (or immediately once the scroll gesture ends) so we
            # never lose ticks while still capping wheel-event frequency.
            if out.scroll_dy:
                self._scroll_accum += out.scroll_dy
            if self._scroll_accum and (
                (now - self._last_scroll_t) >= self._SCROLL_DISPATCH_INTERVAL_S
                or out.mode != GestureMode.SCROLL
            ):
                self._mouse.scroll(self._scroll_accum)
                logger.info("SCROLL dy=%d", self._scroll_accum)
                self._scroll_accum = 0
                self._last_scroll_t = now

            # Swipe
            if out.swipe:
                self._handle_swipe(out.swipe)

            # Wispr
            if out.wispr_shortcut:
                send_ctrl_alt()
                logger.info("Wispr shortcut — local toggle=%s", self._fsm._wispr_active)

            # Perf tracking
            lat = time.perf_counter() - t0
            self._latencies.append(lat)
            if len(self._latencies) > 300:
                self._latencies = self._latencies[-300:]
            self._frame_count += 1
            fps_count += 1

            elapsed = now - fps_t
            if elapsed >= 2.0:
                self._fps = fps_count / elapsed
                fps_count = 0
                fps_t = now

            self._maybe_debug(frame, feat, out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_swipe(self, direction: str) -> None:
        if direction == "RIGHT":
            send_alt_tab_forward()
            logger.info("Swipe RIGHT → next tab")
        elif direction == "LEFT":
            send_alt_tab_backward()
            logger.info("Swipe LEFT → prev tab")
        elif direction == "DOWN":
            send_win_m()
            logger.info("Swipe DOWN → Win+M")
        elif direction == "UP":
            send_win_shift_m()
            logger.info("Swipe UP → Win+Shift+M")

    def _safe_release_all(self) -> None:
        try:
            self._mouse.release_all()
        except Exception:
            pass
        try:
            release_all_modifiers()
        except Exception:
            pass

    def _start_tray(self) -> None:
        self._tray = SystemTray(
            on_pause=self.pause,
            on_resume=self.resume,
            on_exit=self.stop,
            is_active_fn=self.is_active,
        )
        self._tray.start()

    def _maybe_debug(self, frame, feat, out) -> None:
        if not self._cfg.show_debug_window:
            return
        label = f"Mode:{self._fsm.mode.value}  FPS:{self._fps:.1f}"
        if feat:
            label += f"  pinch:{feat.thumb_index_dist:.3f}"
        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("GestureOS Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()

    def get_stats(self) -> dict:
        avg_lat = (
            sum(self._latencies) / len(self._latencies) * 1000
            if self._latencies else 0.0
        )
        p95_lat = 0.0
        if self._latencies:
            sl = sorted(self._latencies)
            p95_lat = sl[int(len(sl) * 0.95)] * 1000
        return {
            "cam_fps": round(self._cam_fps, 1),
            "pipeline_fps": round(self._fps, 1),
            "frame_count": self._frame_count,
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "active": self._active,
            "mode": self._fsm.mode.value,
        }
