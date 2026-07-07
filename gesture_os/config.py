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
    """One Euro Filter parameters for cursor smoothing."""
    min_cutoff: float = 0.35
    beta: float = 0.035
    d_cutoff: float = 0.8


@dataclass
class CursorConfig:
    """Relative trackpad-style cursor configuration."""
    sensitivity: float = 2.8        # multiplier: finger delta * sensitivity = cursor delta
    dead_zone: float = 0.003        # normalized dead zone; suppresses micro-tremor
    max_delta_px: float = 80.0      # per-frame cursor movement clamp in pixels


@dataclass
class GestureConfig:
    """All gesture recognition thresholds and timing values."""

    # --- Pinch distances (normalized 0–1 space) ---
    pinch_enter: float = 0.06       # thumb-index distance to start pinch recognition
    pinch_exit: float = 0.10        # thumb-index distance to cancel pinch
    wispr_enter: float = 0.06       # thumb-middle distance to start Wispr gesture
    wispr_exit: float = 0.10        # thumb-middle distance to cancel Wispr gesture
    three_finger_enter: float = 0.07  # both thumb-index and thumb-middle close

    # --- Timing (milliseconds unless stated) ---
    debounce_ms: float = 80.0       # minimum pinch duration to confirm
    long_press_ms: float = 500.0    # pinch held this long → drag
    double_tap_ms: float = 320.0    # two pinches within this window → double tap
    fist_pause_ms: float = 2000.0   # fist held this long → pause toggle
    wispr_debounce_ms: float = 80.0 # minimum middle-thumb pinch duration

    # --- Fist detection ---
    # Number of fingers (out of 4: index/middle/ring/pinky) whose tips must be
    # below their PIP joint for a fist to be recognized.
    fist_finger_count: int = 4

    # --- Scroll pose ---
    scroll_enter_ms: float = 60.0
    scroll_dead_zone: float = 0.004    # normalized delta to start scrolling
    scroll_speed_scale: float = 100.0  # maps 0.01 normalized delta → ~1 tick
    max_scroll_speed: int = 30         # max scroll ticks per frame

    # --- Open palm / swipe (normalized 0–1 coords) ---
    open_palm_min_fingers: int = 3
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
