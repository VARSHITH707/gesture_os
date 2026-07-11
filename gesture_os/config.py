"""Central configuration for GestureOS.

All tunable thresholds and settings live here. Modify these values to
calibrate gesture sensitivity without touching algorithm code.
"""
import os
from dataclasses import dataclass, field

# Absolute path to MediaPipe model, resolved relative to this package.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_PACKAGE_ROOT, "hand_landmarker.task")


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class LandmarkConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.60
    model_asset_path: str = ""  # resolved to DEFAULT_MODEL_PATH at startup

    def __post_init__(self) -> None:
        if not self.model_asset_path:
            self.model_asset_path = DEFAULT_MODEL_PATH


@dataclass
class FilterConfig:
    """One Euro Filter parameters for cursor smoothing.

    Values match the hand-tuned constants validated in run_cursor_test.py —
    the production defaults here previously lagged behind that tuning pass.
    """
    min_cutoff: float = 0.4
    beta: float = 0.06
    d_cutoff: float = 1.0


@dataclass
class CursorConfig:
    """Relative trackpad-style cursor configuration.

    Values match the hand-tuned constants validated in run_cursor_test.py —
    the production defaults here previously lagged behind that tuning pass.
    """
    sensitivity: float = 2.5        # multiplier: finger delta * sensitivity = cursor delta
    dead_zone: float = 0.008        # normalized dead zone; suppresses micro-tremor
    max_delta_px: float = 300.0     # per-frame cursor movement clamp in pixels


@dataclass
class GestureConfig:
    """All gesture recognition thresholds and timing values."""

    # --- Pinch distances (ratio of hand size — wrist-to-middle-MCP distance,
    # not a fixed frame-space distance) ---
    #
    # Ported from AirBench's hands.js (js/hands.js PINCH_ON/PINCH_OFF), which
    # uses the same MediaPipe 21-point landmark model and the same wrist-to-
    # middle-MCP hand-size reference, so these ratios carry over directly:
    # a pinch means the same thing whether the hand is close to the camera
    # or far from it, instead of only working at one specific hand-to-camera
    # distance the old absolute 0.06/0.10 values happened to be tuned for.
    pinch_enter: float = 0.32       # thumb-index distance to start pinch recognition
    pinch_exit: float = 0.48        # thumb-index distance to cancel pinch
    wispr_enter: float = 0.32       # thumb-middle distance to start Wispr gesture
    wispr_exit: float = 0.48        # thumb-middle distance to cancel Wispr gesture
    three_finger_enter: float = 0.37  # both thumb-index and thumb-middle close (same 1.17x proportion to pinch_enter as before)

    # --- Finger curl/extension (ratio of hand size — tip-to-MCP distance) ---
    # A real geometric curl measurement per finger, robust to hand rotation/
    # tilt — replaces a `tip.y < pip.y` screen-space heuristic that only
    # worked with the hand held upright facing the camera. Value ported from
    # AirBench's hands.js OPEN_ON; aggregation stays count-based (see the
    # comment in extract_features for why the average-based version doesn't
    # fit this app's gesture set).
    finger_extend_ratio: float = 0.85  # per-finger tip-to-MCP ratio above this = that finger counts as extended
    open_palm_min_fingers: int = 3     # fingers (of 4) extended to count as open palm

    # --- Timing (milliseconds unless stated) ---
    debounce_ms: float = 80.0       # minimum pinch duration to confirm
    long_press_ms: float = 500.0    # pinch held this long → drag
    double_tap_ms: float = 320.0    # two pinches within this window → double tap
    fist_pause_ms: float = 2000.0   # fist held this long → pause toggle
    wispr_debounce_ms: float = 80.0 # minimum middle-thumb pinch duration

    # --- Scroll pose ---
    scroll_enter_ms: float = 60.0
    scroll_dead_zone: float = 0.004    # normalized delta to start scrolling
    scroll_speed_scale: float = 100.0  # maps 0.01 normalized delta → ~1 tick
    max_scroll_speed: int = 30         # max scroll ticks per frame

    # --- Swipe (normalized 0–1 coords) ---
    swipe_min_displacement: float = 0.18   # minimum finger travel
    swipe_min_velocity: float = 0.55       # normalized units/sec
    swipe_direction_tolerance_deg: float = 28.0
    swipe_cooldown_s: float = 0.6


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    show_debug_window: bool = False
    autostart_name: str = "GestureOS"
    log_level: str = "INFO"
