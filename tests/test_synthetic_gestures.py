"""Synthetic gesture sequence tests using realistic landmark arrays.

These tests drive the full pipeline (feature extraction → FSM) using
landmark_factory poses without any camera or physical hand. They prove
the complete gesture recognition path end-to-end.
"""
import pytest
import numpy as np

from gesture_os.config import GestureConfig
from gesture_os.input.gesture_fsm import (
    GestureFSM,
    GestureMode,
    extract_features,
    reset_fsm,
    tick_fsm,
)
from tests.landmark_factory import (
    open_palm,
    index_thumb_pinch,
    middle_thumb_pinch,
    three_finger_pinch,
    scroll_pose,
    closed_fist,
    idle_hand,
    pointing_hand,
)


def _cfg():
    return GestureConfig(
        debounce_ms=80.0,
        long_press_ms=500.0,
        double_tap_ms=300.0,
        fist_pause_ms=2000.0,
        wispr_debounce_ms=80.0,
    )


def _feat(lm: np.ndarray, t: float):
    return extract_features(lm, _cfg(), t)


def _tick(fsm, lm, t):
    return tick_fsm(fsm, _feat(lm, t), t)


def _hold(fsm, lm, dt, step=0.005):
    """Hold a pose for dt seconds. Returns last output."""
    out = None
    t = 0.0
    end = dt
    while t < end:
        t += step
        out = _tick(fsm, lm, t)
    return out


def _any_event(fsm, lm, start_t, dt, step=0.005, **kw):
    """Return True if any tick within dt has all kw attributes matching."""
    t = start_t
    end = start_t + dt
    while t < end:
        t += step
        out = tick_fsm(fsm, _feat(lm, t), t)
        if all(getattr(out, k, None) == v for k, v in kw.items()):
            return True
    return False


# ------------------------------------------------------------------
# Open palm → swipe
# ------------------------------------------------------------------

class TestSyntheticSwipe:

    def _swipe_sequence(self, dx, dy):
        """Simulate open-palm starting at (0.5,0.5) moving by (dx,dy)."""
        fsm = GestureFSM(config=_cfg())
        # Establish swipe start
        _tick(fsm, open_palm(0.5, 0.5), 0.0)
        assert fsm.mode == GestureMode.SWIPE_TRACK

        # Move in small steps
        for i in range(1, 10):
            t = i * 0.033
            x = 0.5 + dx * i / 9
            y = 0.5 + dy * i / 9
            out = _tick(fsm, open_palm(x, y), t)
            if out.swipe:
                return out.swipe
        return None

    def test_swipe_right(self):
        assert self._swipe_sequence(0.3, 0.0) == "RIGHT"

    def test_swipe_left(self):
        assert self._swipe_sequence(-0.3, 0.0) == "LEFT"

    def test_swipe_down(self):
        assert self._swipe_sequence(0.0, 0.3) == "DOWN"

    def test_swipe_up(self):
        assert self._swipe_sequence(0.0, -0.3) == "UP"

    def test_swipe_fires_once(self):
        """Swipe fires exactly once and FSM returns to IDLE."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, open_palm(0.5, 0.5), 0.0)
        swipes = []
        for i in range(1, 12):
            t = i * 0.033
            x = 0.5 + 0.3 * i / 11
            out = _tick(fsm, open_palm(x, 0.5), t)
            if out.swipe:
                swipes.append(out.swipe)
        assert len(swipes) == 1, f"Expected 1 swipe, got {len(swipes)}: {swipes}"

    def test_idle_returns_after_swipe(self):
        """FSM returns to IDLE when palm is lowered after swipe fires."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, open_palm(0.5, 0.5), 0.0)
        swipe_fired = False
        for i in range(1, 10):
            out = _tick(fsm, open_palm(0.5 + 0.3 * i / 9, 0.5), i * 0.033)
            if out.swipe:
                swipe_fired = True
        assert swipe_fired, "Expected swipe to fire"
        # pointing_hand: index extended only (is_open_palm=False), so FSM
        # leaves SWIPE_TRACK → IDLE and stays there
        for j in range(3):
            _tick(fsm, pointing_hand(), 0.3 + j * 0.033)
        assert fsm.mode == GestureMode.IDLE


# ------------------------------------------------------------------
# Index-thumb pinch → click
# ------------------------------------------------------------------

class TestSyntheticTap:

    def test_index_pinch_enters_fsm(self):
        """Index-thumb pinch enters PINCH_ENTER state."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, index_thumb_pinch(), 0.0)
        assert fsm.mode == GestureMode.PINCH_ENTER

    def test_single_tap_fires_left_click(self):
        """Full index-thumb tap sequence → left click."""
        fsm = GestureFSM(config=_cfg())
        # Pinch in
        _tick(fsm, index_thumb_pinch(), 0.0)
        # Hold through debounce
        for i in range(1, 20):  # 0.095s
            _tick(fsm, index_thumb_pinch(), i * 0.005)
        assert fsm.mode == GestureMode.PINCH_HELD
        # Release
        _tick(fsm, idle_hand(), 0.1)
        assert fsm.mode == GestureMode.TAP_SETTLE
        # Wait out double-tap window
        fired = _any_event(fsm, idle_hand(), 0.1, 0.4, left_click=True)
        assert fired, "Single tap must produce left_click"

    def test_double_tap_fires_right_click(self):
        """Two pinch-release cycles within window → right click."""
        fsm = GestureFSM(config=_cfg())
        # First tap
        _tick(fsm, index_thumb_pinch(), 0.0)
        for i in range(1, 20):
            _tick(fsm, index_thumb_pinch(), i * 0.005)
        _tick(fsm, idle_hand(), 0.1)
        assert fsm.mode == GestureMode.TAP_SETTLE
        # Second tap within window
        out = _tick(fsm, index_thumb_pinch(), 0.2)
        assert out.right_click, "Second tap must produce right_click"
        assert not out.left_click


# ------------------------------------------------------------------
# Index-thumb long press → drag
# ------------------------------------------------------------------

class TestSyntheticDrag:

    def test_long_press_enters_drag(self):
        """Pinch held > 500ms → DRAG state."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, index_thumb_pinch(), 0.0)
        # Hold for 600ms
        for i in range(1, 121):  # 120 * 5ms = 600ms
            _tick(fsm, index_thumb_pinch(), i * 0.005)
        assert fsm.mode == GestureMode.DRAG

    def test_drag_emits_press_not_click(self):
        """Drag must emit mouse_press, not left_click."""
        fsm = GestureFSM(config=_cfg())
        press_seen = False
        click_seen = False
        _tick(fsm, index_thumb_pinch(), 0.0)
        for i in range(1, 121):
            out = _tick(fsm, index_thumb_pinch(), i * 0.005)
            if out.mouse_press:
                press_seen = True
            if out.left_click:
                click_seen = True
        assert press_seen, "Drag must emit mouse_press"
        assert not click_seen, "Drag must NOT emit left_click"

    def test_drag_release(self):
        """Releasing pinch after drag emits mouse_release."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, index_thumb_pinch(), 0.0)
        for i in range(1, 121):
            _tick(fsm, index_thumb_pinch(), i * 0.005)
        out = _tick(fsm, idle_hand(), 0.61)
        assert out.mouse_release


# ------------------------------------------------------------------
# Closed fist
# ------------------------------------------------------------------

class TestSyntheticFist:

    def test_fist_enters_fist_enter(self):
        """Closed fist → FIST_ENTER state."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, closed_fist(), 0.0)
        assert fsm.mode == GestureMode.FIST_ENTER

    def test_short_fist_becomes_drag(self):
        """Fist released before 2s → FIST_DRAG."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, closed_fist(), 0.0)
        for i in range(1, 60):  # 300ms
            _tick(fsm, closed_fist(), i * 0.005)
        _tick(fsm, idle_hand(), 0.31)
        assert fsm.mode == GestureMode.FIST_DRAG

    def test_long_fist_triggers_pause(self):
        """Fist held 2s → toggle_pause."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, closed_fist(), 0.0)
        fired = _any_event(fsm, closed_fist(), 0.0, 2.2, toggle_pause=True)
        assert fired

    def test_short_fist_no_pause(self):
        """Short fist must NOT trigger pause."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, closed_fist(), 0.0)
        for i in range(1, 60):
            out = _tick(fsm, closed_fist(), i * 0.005)
            assert not out.toggle_pause
        out2 = _tick(fsm, idle_hand(), 0.31)
        assert not out2.toggle_pause

    def test_fist_open_releases_drag(self):
        """Opening hand after FIST_DRAG emits mouse_release."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, closed_fist(), 0.0)
        for i in range(1, 60):
            _tick(fsm, closed_fist(), i * 0.005)
        _tick(fsm, idle_hand(), 0.31)
        assert fsm.mode == GestureMode.FIST_DRAG
        out = _tick(fsm, idle_hand(), 0.35)
        assert out.mouse_release


# ------------------------------------------------------------------
# Scroll
# ------------------------------------------------------------------

class TestSyntheticScroll:

    def test_scroll_pose_enters_scroll(self):
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, scroll_pose(0.5), 0.0)
        assert fsm.mode == GestureMode.SCROLL

    def test_scroll_down(self):
        """Fingers moving downward (increasing y) → positive scroll_dy."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, scroll_pose(0.4), 0.0)
        out = None
        for i in range(1, 10):
            t = i * 0.033
            out = _tick(fsm, scroll_pose(0.4 + i * 0.01), t)
        assert out is not None and out.scroll_dy > 0

    def test_scroll_up(self):
        """Fingers moving upward (decreasing y) → negative scroll_dy."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, scroll_pose(0.6), 0.0)
        out = None
        for i in range(1, 10):
            t = i * 0.033
            out = _tick(fsm, scroll_pose(0.6 - i * 0.01), t)
        assert out is not None and out.scroll_dy < 0

    def test_scroll_suppresses_cursor(self):
        """cursor_active is False during scroll."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, scroll_pose(0.5), 0.0)
        out = _tick(fsm, scroll_pose(0.51), 0.033)
        assert not out.cursor_active

    def test_scroll_speed_scales(self):
        """Larger y delta → larger |scroll_dy|."""
        def run(delta_per_frame):
            fsm = GestureFSM(config=_cfg())
            _tick(fsm, scroll_pose(0.5), 0.0)
            out = None
            for i in range(1, 5):
                out = _tick(fsm, scroll_pose(0.5 + i * delta_per_frame), i * 0.033)
            return abs(out.scroll_dy) if out else 0

        slow = run(0.005)
        fast = run(0.020)
        assert fast >= slow, f"Fast={fast} must be >= slow={slow}"


# ------------------------------------------------------------------
# Middle-thumb (Wispr)
# ------------------------------------------------------------------

class TestSyntheticWispr:

    def _complete_wispr_tap(self, start_t=0.0):
        """Drive a complete middle-thumb tap. Returns (fsm, output)."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, middle_thumb_pinch(), start_t)
        for i in range(1, 20):
            _tick(fsm, middle_thumb_pinch(), start_t + i * 0.005)
        assert fsm.mode == GestureMode.WISPR_HELD
        out = _tick(fsm, idle_hand(), start_t + 0.1)
        return fsm, out

    def test_wispr_tap_fires_shortcut(self):
        """Middle-thumb tap fires wispr_shortcut."""
        _, out = self._complete_wispr_tap()
        assert out.wispr_shortcut

    def test_wispr_no_click(self):
        """Wispr tap must not produce left_click or right_click."""
        _, out = self._complete_wispr_tap()
        assert not out.left_click
        assert not out.right_click

    def test_wispr_second_tap_fires_again(self):
        """Second complete middle-thumb tap fires a second shortcut."""
        fsm = GestureFSM(config=_cfg())
        # First tap
        _tick(fsm, middle_thumb_pinch(), 0.0)
        for i in range(1, 20):
            _tick(fsm, middle_thumb_pinch(), i * 0.005)
        _tick(fsm, idle_hand(), 0.1)
        assert fsm.mode == GestureMode.IDLE
        # Second tap
        _tick(fsm, middle_thumb_pinch(), 0.2)
        for i in range(1, 20):
            _tick(fsm, middle_thumb_pinch(), 0.2 + i * 0.005)
        out = _tick(fsm, idle_hand(), 0.3)
        assert out.wispr_shortcut

    def test_holding_wispr_does_not_repeat(self):
        """Holding middle-thumb pinch past debounce never fires twice."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, middle_thumb_pinch(), 0.0)
        shortcuts = []
        for i in range(1, 60):
            out = _tick(fsm, middle_thumb_pinch(), i * 0.005)
            if out.wispr_shortcut:
                shortcuts.append(i)
        assert not shortcuts, f"Held wispr fired: {shortcuts}"

    def test_wispr_priority_over_index_pinch(self):
        """Middle-thumb pinch must not activate as index-thumb pinch."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, middle_thumb_pinch(), 0.0)
        assert fsm.mode == GestureMode.WISPR_ENTER
        assert fsm.mode != GestureMode.PINCH_ENTER


# ------------------------------------------------------------------
# Hand loss and reacquisition (FSM side)
# ------------------------------------------------------------------

class TestSyntheticHandLoss:

    def test_hand_loss_resets_fsm(self):
        """reset_fsm() after hand loss returns to IDLE."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, index_thumb_pinch(), 0.0)
        assert fsm.mode == GestureMode.PINCH_ENTER
        reset_fsm(fsm)
        assert fsm.mode == GestureMode.IDLE

    def test_reacquisition_after_drag(self):
        """Hand lost during drag: after reset, no press/drag state lingers."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, index_thumb_pinch(), 0.0)
        for i in range(1, 121):
            _tick(fsm, index_thumb_pinch(), i * 0.005)
        assert fsm.mode == GestureMode.DRAG
        assert fsm._mouse_pressed
        reset_fsm(fsm)
        assert fsm.mode == GestureMode.IDLE
        assert not fsm._mouse_pressed

    def test_reacquisition_no_spurious_click(self):
        """After reset, first idle frame does not produce any click."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, index_thumb_pinch(), 0.0)
        reset_fsm(fsm)
        out = _tick(fsm, idle_hand(), 0.0)
        assert not out.left_click
        assert not out.right_click


# ------------------------------------------------------------------
# Gesture priority
# ------------------------------------------------------------------

class TestGesturePriority:

    def test_fist_overrides_pinch(self):
        """If hand is fist, FIST_ENTER takes priority over index pinch."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, closed_fist(), 0.0)
        assert fsm.mode == GestureMode.FIST_ENTER

    def test_scroll_not_active_during_open_palm(self):
        """Open palm does not enter SCROLL (wrong pose)."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, open_palm(), 0.0)
        assert fsm.mode != GestureMode.SCROLL

    def test_wispr_overrides_scroll(self):
        """Middle-thumb pinch takes priority over scroll pose check."""
        fsm = GestureFSM(config=_cfg())
        _tick(fsm, middle_thumb_pinch(), 0.0)
        assert fsm.mode == GestureMode.WISPR_ENTER
        assert fsm.mode != GestureMode.SCROLL
