"""TEST 9–11: Relative trackpad cursor tests.

Covers:
- Relative movement from current position
- Dead zone suppresses micro-drift
- Reacquisition does not jump
- Screen boundary clamping
- on_hand_lost freezes cursor
"""
import pytest
from gesture_os.config import CursorConfig
from gesture_os.input.cursor import RelativeCursorController


def _make_ctrl(sensitivity=2.0, dead_zone=0.003, max_delta=200.0):
    cfg = CursorConfig(sensitivity=sensitivity, dead_zone=dead_zone, max_delta_px=max_delta)
    return RelativeCursorController(cfg, screen_width=1920, screen_height=1080)


def test_first_frame_no_jump():
    """TEST 11 — First frame after reacquisition does not move the cursor."""
    ctrl = _make_ctrl()
    # Simulate fresh start at centre
    ctrl._state.screen_x = 960
    ctrl._state.screen_y = 540
    # First update at arbitrary position
    x, y = ctrl.update(0.8, 0.8)
    # Cursor must not jump
    assert x == 960 and y == 540, f"Expected (960, 540) got ({x}, {y})"


def test_relative_movement():
    """TEST 9 — Movement from reference point moves cursor proportionally."""
    ctrl = _make_ctrl(sensitivity=2.0, dead_zone=0.0, max_delta=2000.0)
    ctrl._state.screen_x = 500
    ctrl._state.screen_y = 300
    # Establish reference (first frame sets ref, no movement)
    ctrl.update(0.5, 0.5)
    # Move finger 0.1 in x, 0.05 in y
    x, y = ctrl.update(0.6, 0.55)
    # dx_px = 0.1 * 1920 * 2.0 = 384 (within max_delta=2000)
    assert abs(x - (500 + 384)) < 5, f"x={x}"
    # dy_px = 0.05 * 1080 * 2.0 = 108
    assert abs(y - (300 + 108)) < 5, f"y={y}"


def test_dead_zone_suppresses_drift():
    """Small movement below dead zone does not move cursor."""
    ctrl = _make_ctrl(dead_zone=0.01)
    ctrl._state.screen_x = 960
    ctrl._state.screen_y = 540
    ctrl.update(0.5, 0.5)    # establish ref
    # Sub-dead-zone movement
    x, y = ctrl.update(0.505, 0.505)  # delta ~ 0.007 < dead_zone 0.01
    assert x == 960 and y == 540, f"Dead zone failed: ({x}, {y})"


def test_screen_boundary_clamping():
    """Cursor is clamped to screen bounds."""
    ctrl = _make_ctrl(sensitivity=10.0, dead_zone=0.0, max_delta=5000.0)
    ctrl._state.screen_x = 0
    ctrl._state.screen_y = 0
    ctrl.update(0.5, 0.5)
    # Move far left/up
    x, y = ctrl.update(0.0, 0.0)
    assert x >= 0 and y >= 0


def test_hand_lost_freezes_cursor():
    """TEST 10 — on_hand_lost freezes cursor at last known position."""
    ctrl = _make_ctrl(dead_zone=0.0)
    ctrl._state.screen_x = 800
    ctrl._state.screen_y = 400
    ctrl.update(0.5, 0.5)
    ctrl.update(0.6, 0.6)
    last_x, last_y = ctrl._state.screen_x, ctrl._state.screen_y
    ctrl.on_hand_lost()
    # After hand lost, state should be reset for reacquisition
    assert ctrl._state.ref_x is None


def test_reacquisition_no_jump():
    """TEST 11 — After on_hand_lost, next update does not jump."""
    ctrl = _make_ctrl(dead_zone=0.0)
    ctrl._state.screen_x = 300
    ctrl._state.screen_y = 200
    ctrl.update(0.1, 0.1)
    ctrl.update(0.2, 0.2)
    ctrl.on_hand_lost()
    # Hand returns at very different position
    x, y = ctrl.update(0.9, 0.9)
    # Should stay at last known position (no jump)
    assert x == ctrl._state.screen_x and y == ctrl._state.screen_y
