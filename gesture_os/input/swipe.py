from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwipeConfig:
    """Thresholds for swipe gesture detection."""
    velocity_threshold_px: float = 1000.0
    direction_tolerance_deg: float = 25.0
    min_hold_ms: float = 80.0
    min_displacement_px: float = 150.0


@dataclass
class SwipeState:
    """Mutable state for swipe tracking."""
    last_x: float = 0.0
    last_y: float = 0.0
    last_time: float = field(default_factory=time.perf_counter)


def detect_swipe(
    state: SwipeState,
    x: float,
    y: float,
    now: float,
    config: SwipeConfig,
) -> str | None:
    """Return swipe direction string or None. Mutates state."""
    try:
        dt = max(now - state.last_time, 1e-6)
        dx = x - state.last_x
        dy = y - state.last_y
        dist = (dx**2 + dy**2) ** 0.5
        velocity = dist / dt
        state.last_x, state.last_y, state.last_time = x, y, now
        if velocity < config.velocity_threshold_px:
            return None
        if dist < config.min_displacement_px:
            return None
        import math
        angle_deg = math.degrees(math.atan2(dy, dx))
        tol = config.direction_tolerance_deg
        # Right: ~0 deg, Left: ~180/-180 deg, Down: ~90 deg, Up: ~-90 deg
        if -tol <= angle_deg <= tol:
            return "SWIPE_RIGHT"
        if angle_deg >= (180 - tol) or angle_deg <= (-180 + tol):
            return "SWIPE_LEFT"
        if (90 - tol) <= angle_deg <= (90 + tol):
            return "SWIPE_DOWN"
        if (-90 - tol) <= angle_deg <= (-90 + tol):
            return "SWIPE_UP"
        return None
    except Exception as e:
        logger.warning(f"detect_swipe error: {e}")
        return None