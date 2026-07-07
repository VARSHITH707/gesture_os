"""TEST 7: One Euro Filter produces stable smoothed coordinates.

Covers:
- Stationary jitter reduction
- Moving response (no excessive lag)
- First-sample safety
- Non-finite input safety
- Reacquisition after long gap
"""
import time
import math
import pytest

from gesture_os.vision.filters import (
    OneEuroState,
    OneEuroState2D,
    filter_point_2d,
    one_euro_filter,
)


def test_first_sample_returns_input():
    """First sample is passed through directly (no delay)."""
    state = OneEuroState()
    result = one_euro_filter(state, 0.5, time.perf_counter())
    assert result == pytest.approx(0.5, abs=1e-6)


def test_stationary_jitter_reduction():
    """Stationary noisy signal converges to near the true value."""
    import random
    state = OneEuroState2D()
    true_val = 0.5
    t = 0.0
    last_x = true_val
    for _ in range(100):
        t += 0.033
        noisy = true_val + random.gauss(0, 0.01)
        last_x, _ = filter_point_2d(state, (noisy, noisy), t)
    assert abs(last_x - true_val) < 0.02, f"Filter did not converge: {last_x}"


def test_moving_response():
    """Filter follows a linearly moving signal without excessive lag."""
    state = OneEuroState2D()
    t = 0.0
    for i in range(30):
        t += 0.033
        v = i / 30.0
        fx, _ = filter_point_2d(state, (v, 0.5), t)
    # After 30 frames of smooth motion, filter should be within 35% of the
    # final true value. One Euro Filter has intentional lag on slow ramps;
    # this confirms it follows the signal direction without freezing.
    assert abs(fx - 1.0) < 0.35, f"Filter too laggy at end of ramp: {fx}"


def test_non_finite_input_passes_through():
    """Non-finite (NaN/Inf) input is returned unchanged — does not crash."""
    state = OneEuroState2D()
    result = filter_point_2d(state, (float("nan"), 0.5), 0.1)
    assert math.isnan(result[0])


def test_long_gap_does_not_jump():
    """A long time gap resets dt gracefully — no enormous jump in output."""
    state = OneEuroState2D()
    filter_point_2d(state, (0.5, 0.5), 1.0)
    # Jump in time by 60 seconds (loss of tracking)
    fx, fy = filter_point_2d(state, (0.5, 0.5), 61.0)
    assert math.isfinite(fx) and math.isfinite(fy)
    assert abs(fx - 0.5) < 0.1
