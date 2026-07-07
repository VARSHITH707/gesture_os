"""Camera capture with threaded reader for minimum-latency frame delivery.

The ThreadedCamera runs a background thread that continuously grabs frames,
so the main pipeline never blocks waiting for the camera. This reduces
end-to-end latency significantly versus synchronous reads.
"""
import cv2
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30          # laptop webcams typically top out at 30 FPS
    buffer_size: int = 1   # keep internal OS buffer minimal to reduce staleness


def open_camera(config: CameraConfig) -> cv2.VideoCapture:
    """Open and configure a camera capture device."""
    try:
        cap = cv2.VideoCapture(config.device_index, cv2.CAP_DSHOW)  # DirectShow on Windows
        if not cap.isOpened():
            # Fallback without backend hint
            cap = cv2.VideoCapture(config.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {config.device_index} failed to open.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        cap.set(cv2.CAP_PROP_FPS, config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
        # Drain any stale buffered frames at startup
        for _ in range(3):
            cap.read()
        return cap
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error opening camera: {e}") from e


def release_camera(cap: cv2.VideoCapture) -> None:
    """Release the camera capture device safely."""
    try:
        cap.release()
    except Exception as e:
        logger.warning("Error releasing camera: %s", e)


def read_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """Read one frame from an open capture device."""
    try:
        success, frame = cap.read()
        if not success or frame is None:
            return False, None
        return True, frame
    except Exception as e:
        logger.warning("read_frame exception: %s", e)
        return False, None


class ThreadedCamera:
    """Background-thread camera that always serves the freshest available frame.

    The grab loop runs continuously in a daemon thread. The main pipeline
    calls read() and immediately gets whatever the camera most recently
    delivered — no blocking wait for the next exposure.

    This removes the ~25 ms camera-block from the main loop, allowing
    MediaPipe inference to run at its own speed (~10 ms) and only wait on
    available frames when the camera is genuinely slower.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._cap = open_camera(config)
        self._frame: np.ndarray | None = None
        self._frame_t: float = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._new_frame = threading.Event()
        self._t = threading.Thread(target=self._grab_loop, daemon=True, name="CameraGrab")
        self._t.start()
        # Wait for the first frame before returning
        self._new_frame.wait(timeout=3.0)

    def _grab_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                t = time.perf_counter()
                with self._lock:
                    self._frame = frame
                    self._frame_t = t
                self._new_frame.set()
            else:
                time.sleep(0.005)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return (success, frame, capture_timestamp). Non-blocking."""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._frame_t

    def read_blocking(self, timeout: float = 0.1) -> tuple[bool, np.ndarray | None, float]:
        """Block until a new frame arrives or timeout elapses."""
        self._new_frame.clear()
        self._new_frame.wait(timeout=timeout)
        return self.read()

    def release(self) -> None:
        """Stop the grab thread and release the camera."""
        self._running = False
        self._t.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass

    def measure_fps(self, duration_s: float = 2.0) -> float:
        """Measure real delivered FPS over duration_s seconds."""
        count = 0
        last_t = self._frame_t
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            self._new_frame.clear()
            self._new_frame.wait(timeout=0.1)
            with self._lock:
                if self._frame_t != last_t:
                    last_t = self._frame_t
                    count += 1
        return count / duration_s
