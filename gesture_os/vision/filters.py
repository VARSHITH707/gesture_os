import logging
import math
from dataclasses import dataclass, field, field as _field
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class LowPassState:
    """Holds state for a single low-pass filter channel."""
    value: float = 0.0
    initialized: bool = False


@dataclass
class OneEuroState:
    """State for one channel of the One Euro Filter."""
    min_cutoff: float = 0.35
    beta: float = 0.035
    d_cutoff: float = 0.8
    _x_filter: LowPassState = field(default_factory=LowPassState)
    _dx_filter: LowPassState = field(default_factory=LowPassState)
    _last_time: float = -1.0


def _low_pass_filter(
    state: LowPassState, x: float, alpha: float
) -> float:
    """Apply one step of a low-pass filter to scalar x."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if not state.initialized:
        state.value = x
        state.initialized = True
    else:
        state.value = alpha * x + (1.0 - alpha) * state.value
    return state.value


def one_euro_filter(
    state: OneEuroState, x: float, timestamp: float
) -> float:
    """Filter scalar x with the One Euro Filter algorithm."""
    try:
        if state._last_time < 0:
            state._last_time = timestamp
            return _low_pass_filter(state._x_filter, x, 1.0)
        dt = max(timestamp - state._last_time, 1e-6)
        rate = 1.0 / dt
        dx = (x - state._x_filter.value) * rate if state._x_filter.initialized else 0.0
        a_d = 1.0 / (1.0 + (1.0 / (2 * math.pi * state.d_cutoff)) * rate)
        dx_hat = _low_pass_filter(state._dx_filter, dx, a_d)
        cutoff = state.min_cutoff + state.beta * abs(dx_hat)
        tau = 1.0 / (2 * math.pi * cutoff)
        alpha = float(np.clip(1.0 / (1.0 + tau * rate), 0.0, 1.0))
        state._last_time = timestamp
        return _low_pass_filter(state._x_filter, x, alpha)
    except Exception as e:
        logger.warning(f"one_euro_filter error: {e}")
        return x


@dataclass
class OneEuroState2D:
    """Paired One Euro Filter state for 2D (x, y) coordinates."""
    x_state: OneEuroState = _field(default_factory=OneEuroState)
    y_state: OneEuroState = _field(default_factory=OneEuroState)


def filter_point_2d(
    state: OneEuroState2D,
    point: tuple[float, float],
    timestamp: float,
) -> tuple[float, float]:
    """Apply One Euro Filter independently to x and y of a 2D point."""
    px, py = point
    if not math.isfinite(px) or not math.isfinite(py):
        logger.warning("filter_point_2d received non-finite input.")
        return point
    fx = one_euro_filter(state.x_state, px, timestamp)
    fy = one_euro_filter(state.y_state, py, timestamp)
    return fx, fy


def apply_dead_zone(
    current: tuple[float, float],
    previous: tuple[float, float],
    threshold_px: float = 2.0,
) -> tuple[float, float]:
    """Suppress cursor micro-drift below threshold_px distance."""
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    dist = (dx**2 + dy**2) ** 0.5
    return previous if dist < threshold_px else current


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    state = OneEuroState2D()
    samples = [0.5 + float(np.random.normal(0, 0.01)) for _ in range(20)]
    for i, val in enumerate(samples):
        ts = time.perf_counter()
        fx, fy = filter_point_2d(state, (val, val), ts)
        if i < 5 or i >= 15:
            print(f"raw={val:.5f}  filtered={fx:.5f}")