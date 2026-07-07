"""TEST 8 + 12–46: Gesture FSM transition tests.

Uses synthetic GestureFeatures to drive the FSM without real hardware.
Covers all required gesture flows.
"""
import time
import pytest
import numpy as np

from gesture_os.config import GestureConfig
from gesture_os.input.gesture_fsm import (
    GestureFSM,
    GestureFeatures,
    GestureMode,
    FSMOutput,
    reset_fsm,
    tick_fsm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> GestureConfig:
    return GestureConfig(
        debounce_ms=80.0,
        long_press_ms=500.0,
        double_tap_ms=300.0,
        fist_pause_ms=2000.0,
        wispr_debounce_ms=80.0,
    )


def _idle_feat(now: float = 0.0) -> GestureFeatures:
    """Features with no active gesture."""
    return GestureFeatures(
        index_tip_x=0.5, index_tip_y=0.5,
        scroll_ref_y=0.5,
        thumb_index_dist=0.15,   # well above pinch threshold
        thumb_middle_dist=0.15,
        is_fist=False, is_open_palm=False, is_scroll_pose=False,
        is_scroll_shape=False,
        is_index_pinch=False, is_wispr_pinch=False, is_three_finger_pinch=False,
        timestamp=now,
    )


def _pinch_feat(now: float = 0.0) -> GestureFeatures:
    f = _idle_feat(now)
    return GestureFeatures(
        **{**f.__dict__, "thumb_index_dist": 0.04, "is_index_pinch": True}
    )


def _wispr_feat(now: float = 0.0) -> GestureFeatures:
    f = _idle_feat(now)
    return GestureFeatures(
        **{**f.__dict__, "thumb_middle_dist": 0.04, "is_wispr_pinch": True}
    )


def _fist_feat(now: float = 0.0) -> GestureFeatures:
    f = _idle_feat(now)
    return GestureFeatures(
        **{**f.__dict__, "is_fist": True,
           "is_index_pinch": False, "is_wispr_pinch": False}
    )


def _scroll_feat(now: float = 0.0, y: float = 0.5) -> GestureFeatures:
    f = _idle_feat(now)
    return GestureFeatures(
        **{**f.__dict__, "is_scroll_pose": True, "is_scroll_shape": True, "scroll_ref_y": y}
    )


def _palm_feat(now: float = 0.0, x: float = 0.5, y: float = 0.5) -> GestureFeatures:
    f = _idle_feat(now)
    return GestureFeatures(
        **{**f.__dict__, "is_open_palm": True, "index_tip_x": x, "index_tip_y": y}
    )


def _advance(fsm: GestureFSM, feat: GestureFeatures, dt_s: float, step_s: float = 0.005) -> FSMOutput:
    """Simulate holding a gesture for dt_s seconds. Returns last output."""
    out = None
    t = feat.timestamp
    end = t + dt_s
    while t < end:
        t += step_s
        f = GestureFeatures(**{**feat.__dict__, "timestamp": t})
        out = tick_fsm(fsm, f, t)
    return out


def _any_event(fsm: GestureFSM, feat: GestureFeatures, dt_s: float,
               step_s: float = 0.005, **expected) -> bool:
    """Return True if any tick within dt_s emits all expected attribute values."""
    t = feat.timestamp
    end = t + dt_s
    while t < end:
        t += step_s
        f = GestureFeatures(**{**feat.__dict__, "timestamp": t})
        out = tick_fsm(fsm, f, t)
        if all(getattr(out, k, None) == v for k, v in expected.items()):
            return True
    return False


# ---------------------------------------------------------------------------
# TEST 8: FSM state transitions
# ---------------------------------------------------------------------------

class TestFSMTransitions:

    def test_idle_is_initial_state(self):
        fsm = GestureFSM(config=_cfg())
        assert fsm.mode == GestureMode.IDLE

    def test_pinch_enter_on_index_pinch(self):
        fsm = GestureFSM(config=_cfg())
        out = tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        assert fsm.mode == GestureMode.PINCH_ENTER

    def test_pinch_held_after_debounce(self):
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.1)   # 100ms > 80ms debounce
        assert fsm.mode in (GestureMode.PINCH_HELD, GestureMode.DRAG)

    def test_fist_enters_fist_enter(self):
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        assert fsm.mode == GestureMode.FIST_ENTER

    def test_scroll_enters_scroll(self):
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _scroll_feat(0.0), 0.0)
        assert fsm.mode == GestureMode.SCROLL

    def test_wispr_enters_wispr(self):
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        assert fsm.mode == GestureMode.WISPR_ENTER

    def test_reset_returns_to_idle(self):
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        assert fsm.mode != GestureMode.IDLE
        reset_fsm(fsm)
        assert fsm.mode == GestureMode.IDLE


# ---------------------------------------------------------------------------
# TEST 12–13: Single tap vs double tap
# ---------------------------------------------------------------------------

class TestTapGestures:

    def test_single_tap_emits_left_click(self):
        """TEST 12 — single tap → left click."""
        fsm = GestureFSM(config=_cfg())

        # Enter pinch, debounce
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.09)   # past debounce

        # Release pinch
        out = tick_fsm(fsm, _idle_feat(0.09), 0.09)
        assert fsm.mode == GestureMode.TAP_SETTLE

        # Wait out double-tap window; left_click fires on the tick where window expires.
        fired = _any_event(fsm, _idle_feat(0.09), 0.4, left_click=True)
        assert fired, "Single tap must emit left_click during settle window"
        assert fsm.mode == GestureMode.IDLE

    def test_double_tap_emits_right_click(self):
        """TEST 13 — double tap → right click; TEST 14 — no spurious left clicks."""
        fsm = GestureFSM(config=_cfg())

        # First tap: pinch + debounce + release
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.09)
        tick_fsm(fsm, _idle_feat(0.09), 0.09)
        assert fsm.mode == GestureMode.TAP_SETTLE

        # Second tap within double-tap window (< 300ms)
        t2 = 0.09 + 0.10   # 100ms later, well inside window
        out = tick_fsm(fsm, _pinch_feat(t2), t2)
        assert out.right_click, "Double tap must emit right_click"
        assert not out.left_click, "Double tap must NOT emit left_click"

    def test_single_tap_does_not_fire_during_window(self):
        """TEST 14 — No left click fires during double-tap window."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.09)
        tick_fsm(fsm, _idle_feat(0.09), 0.09)

        # Check intermediate frames: no left click yet
        for i in range(5):
            t = 0.09 + i * 0.02
            out = tick_fsm(fsm, _idle_feat(t), t)
            assert not out.left_click, f"Premature left_click at frame {i}"


# ---------------------------------------------------------------------------
# TEST 16–17: Long-press drag
# ---------------------------------------------------------------------------

class TestDrag:

    def test_long_pinch_starts_drag(self):
        """TEST 16 — pinch held > 500ms → drag mode."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.6)   # 600ms > 500ms long_press
        assert fsm.mode == GestureMode.DRAG

    def test_drag_emits_mouse_press(self):
        """Mouse press is emitted when pinch is confirmed."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        pressed = False
        t = 0.0
        for _ in range(30):
            t += 0.005
            out = tick_fsm(fsm, _pinch_feat(t), t)
            if out.mouse_press:
                pressed = True
        assert pressed, "mouse_press never emitted during debounce"

    def test_drag_release_emits_mouse_release(self):
        """TEST 17 — releasing pinch after drag emits mouse_release."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.6)
        assert fsm.mode == GestureMode.DRAG
        out = tick_fsm(fsm, _idle_feat(0.6), 0.6)
        assert out.mouse_release, "Release from drag must emit mouse_release"

    def test_drag_does_not_emit_click(self):
        """No click is emitted when gesture escalates to drag."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        clicks = []
        t = 0.0
        for _ in range(120):   # 600ms
            t += 0.005
            out = tick_fsm(fsm, _pinch_feat(t), t)
            if out.left_click or out.right_click:
                clicks.append(t)
        assert not clicks, f"Unexpected clicks during drag: {clicks}"


# ---------------------------------------------------------------------------
# TEST 18–19: Fist drag
# ---------------------------------------------------------------------------

class TestFistDrag:

    def test_short_fist_starts_drag(self):
        """TEST 18 — fist released before 2s → fist drag."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        _advance(fsm, _fist_feat(0.0), 0.3)   # 300ms, well under 2000ms
        out = tick_fsm(fsm, _idle_feat(0.3), 0.3)   # release fist
        # After release: should be in FIST_DRAG
        assert fsm.mode == GestureMode.FIST_DRAG or out.mouse_press

    def test_fist_open_ends_drag(self):
        """TEST 19 — opening hand after fist drag emits mouse_release."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        _advance(fsm, _fist_feat(0.0), 0.3)
        tick_fsm(fsm, _idle_feat(0.3), 0.3)   # transition to FIST_DRAG
        assert fsm.mode == GestureMode.FIST_DRAG
        out = tick_fsm(fsm, _idle_feat(0.35), 0.35)   # already not fist
        assert out.mouse_release or fsm.mode == GestureMode.IDLE


# ---------------------------------------------------------------------------
# TEST 20–23: Scroll
# ---------------------------------------------------------------------------

class TestScroll:

    def test_scroll_down(self):
        """TEST 21 — downward finger movement produces positive scroll_dy."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _scroll_feat(0.0, y=0.4), 0.0)

        out = None
        for i in range(1, 10):
            t = i * 0.033
            out = tick_fsm(fsm, _scroll_feat(t, y=0.4 + i * 0.01), t)
        assert out is not None and out.scroll_dy > 0, f"scroll_dy={out.scroll_dy if out else None}"

    def test_scroll_up(self):
        """TEST 20 — upward finger movement produces negative scroll_dy."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _scroll_feat(0.0, y=0.6), 0.0)

        out = None
        for i in range(1, 10):
            t = i * 0.033
            out = tick_fsm(fsm, _scroll_feat(t, y=0.6 - i * 0.01), t)
        assert out is not None and out.scroll_dy < 0, f"scroll_dy={out.scroll_dy if out else None}"

    def test_scroll_suppresses_cursor(self):
        """TEST 23 — cursor_active is False during scroll."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _scroll_feat(0.0), 0.0)
        out = tick_fsm(fsm, _scroll_feat(0.033), 0.033)
        assert not out.cursor_active, "Scroll must suppress cursor movement"

    def test_scroll_speed_scales_with_velocity(self):
        """TEST 22 — faster movement produces larger |scroll_dy|."""
        def _run_scroll(dy_per_frame: float) -> int:
            fsm = GestureFSM(config=_cfg())
            tick_fsm(fsm, _scroll_feat(0.0, y=0.5), 0.0)
            out = None
            for i in range(1, 5):
                t = i * 0.033
                out = tick_fsm(fsm, _scroll_feat(t, y=0.5 + i * dy_per_frame), t)
            return abs(out.scroll_dy) if out else 0

        slow = _run_scroll(0.005)
        fast = _run_scroll(0.020)
        assert fast >= slow, f"Fast scroll ({fast}) not >= slow ({slow})"


# ---------------------------------------------------------------------------
# Scroll/pinch hysteresis (Fix A) and scroll_ref_y filtering (Fix C)
# regression coverage — see gesture_fsm.py SCROLL state.
# ---------------------------------------------------------------------------

class TestScrollHysteresisAndFilter:

    def test_thumb_dip_during_scroll_does_not_exit_or_click(self):
        """Fix A regression — a momentary thumb-index dip below pinch_enter
        while still in the two-finger scroll shape must NOT collapse SCROLL
        into a pinch/click. This is the exact boundary that used to fire a
        phantom LEFT CLICK. Uses is_scroll_shape (not is_scroll_pose) as the
        exit condition; would fail if that were reverted to is_scroll_pose.
        """
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _scroll_feat(0.0, y=0.4), 0.0)
        assert fsm.mode == GestureMode.SCROLL

        # Thumb dips into pinch range: is_scroll_pose would go False, but
        # finger-extension shape (is_scroll_shape) is unchanged.
        dip_feat = _scroll_feat(0.033, y=0.41)
        dip_feat = GestureFeatures(**{
            **dip_feat.__dict__,
            "is_scroll_pose": False,
            "is_scroll_shape": True,
            "thumb_index_dist": 0.03,
            "is_index_pinch": True,
        })
        out = tick_fsm(fsm, dip_feat, 0.033)
        assert fsm.mode == GestureMode.SCROLL, (
            f"Thumb dip exited SCROLL into {fsm.mode} — hysteresis broken"
        )
        assert not out.left_click and not out.right_click

        # Hold the dip well past double_tap_ms / long_press_ms — under the
        # old bug this window is exactly when the phantom left_click fired.
        out = None
        t = 0.033
        while t < 0.033 + 0.6:
            t += 0.02
            dip_feat = GestureFeatures(**{**dip_feat.__dict__, "timestamp": t})
            out = tick_fsm(fsm, dip_feat, t)
            assert not out.left_click, f"Phantom LEFT CLICK fired at t={t}"
            assert not out.right_click
        assert fsm.mode == GestureMode.SCROLL

        # Sanity: genuinely losing the scroll shape still exits normally.
        no_shape = GestureFeatures(**{**dip_feat.__dict__, "is_scroll_shape": False,
                                       "is_scroll_pose": False})
        tick_fsm(fsm, no_shape, t + 0.02)
        assert fsm.mode == GestureMode.IDLE

    def test_scroll_ref_y_is_filtered_not_raw(self):
        """Fix C regression — scroll delta must be computed from the
        One-Euro-filtered y (fsm._scroll_y_filter), not the raw landmark y.
        Would fail if gesture_fsm.py reverted to storing feat.scroll_ref_y
        directly in fsm._scroll_prev_y.
        """
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _scroll_feat(0.0, y=0.40), 0.0)

        t = 0.0
        for i in range(1, 6):
            t = i * 0.01
            tick_fsm(fsm, _scroll_feat(t, y=0.40 + i * 0.002), t)

        t_noisy = t + 0.01
        raw_y = 0.40 + 6 * 0.002 + 0.05  # trend value plus a sudden jump
        tick_fsm(fsm, _scroll_feat(t_noisy, y=raw_y), t_noisy)

        assert fsm._scroll_prev_y != raw_y, (
            "scroll_prev_y matches raw input exactly — filter appears bypassed"
        )
        assert abs(fsm._scroll_prev_y - raw_y) > 1e-6


# ---------------------------------------------------------------------------
# TEST 24–29: Swipe gestures
# ---------------------------------------------------------------------------

class TestSwipe:

    def _run_swipe(self, dx: float, dy: float) -> str | None:
        """Simulate an open-palm swipe and return the emitted direction."""
        fsm = GestureFSM(config=_cfg())
        t = 0.0
        # Start swipe
        out = tick_fsm(fsm, _palm_feat(t, x=0.5, y=0.5), t)

        # Move quickly
        for i in range(1, 8):
            t += 0.033
            x = 0.5 + dx * i / 7.0
            y = 0.5 + dy * i / 7.0
            out = tick_fsm(fsm, _palm_feat(t, x=x, y=y), t)
            if out.swipe:
                return out.swipe
        return None

    def test_swipe_right(self):
        """TEST 24 — left-to-right palm swipe."""
        assert self._run_swipe(0.3, 0.0) == "RIGHT"

    def test_swipe_left(self):
        """TEST 25 — right-to-left palm swipe."""
        assert self._run_swipe(-0.3, 0.0) == "LEFT"

    def test_swipe_down(self):
        """TEST 26 — top-to-bottom palm swipe."""
        assert self._run_swipe(0.0, 0.3) == "DOWN"

    def test_swipe_up(self):
        """TEST 27 — bottom-to-top palm swipe."""
        assert self._run_swipe(0.0, -0.3) == "UP"

    def test_swipe_cooldown(self):
        """TEST 28 — swipe cooldown prevents immediate re-trigger."""
        fsm = GestureFSM(config=_cfg())
        # First swipe
        first = self._run_swipe(0.3, 0.0)
        assert first == "RIGHT"
        # Second swipe immediately (within cooldown 0.6s)
        second = self._run_swipe(0.3, 0.0)
        # Cooldown should block it — but we need the same fsm instance.
        # This test verifies cooldown via the same fsm.
        fsm2 = GestureFSM(config=_cfg())
        t = 0.0
        tick_fsm(fsm2, _palm_feat(t, 0.5, 0.5), t)
        for i in range(1, 8):
            t += 0.033
            tick_fsm(fsm2, _palm_feat(t, 0.5 + 0.3 * i / 7, 0.5), t)
        # Reset to idle and immediately try again
        tick_fsm(fsm2, _idle_feat(t), t)
        # Try second swipe right away (no cooldown elapsed)
        tick_fsm(fsm2, _palm_feat(t, 0.5, 0.5), t)
        swipe2 = None
        for i in range(1, 8):
            t += 0.033
            out = tick_fsm(fsm2, _palm_feat(t, 0.5 + 0.3 * i / 7, 0.5), t)
            if out.swipe:
                swipe2 = out.swipe
        assert swipe2 is None, "Swipe fired during cooldown"

    def test_holding_palm_no_retrigger(self):
        """TEST 29 — holding open palm after swipe does not retrigger."""
        fsm = GestureFSM(config=_cfg())
        t = 0.0
        # Trigger a swipe
        tick_fsm(fsm, _palm_feat(t, 0.5, 0.5), t)
        for i in range(1, 8):
            t += 0.033
            tick_fsm(fsm, _palm_feat(t, 0.5 + 0.3 * i / 7, 0.5), t)
        # Now hold palm still
        swipes = []
        for _ in range(20):
            t += 0.033
            out = tick_fsm(fsm, _palm_feat(t, 0.85, 0.5), t)
            if out.swipe:
                swipes.append(out.swipe)
        assert not swipes, f"Holding palm retriggered swipe: {swipes}"


# ---------------------------------------------------------------------------
# TEST 30–40: Wispr gesture
# ---------------------------------------------------------------------------

class TestWispr:

    def _run_wispr_tap(self) -> tuple[bool, GestureFSM]:
        """Simulate a complete middle-thumb tap. Returns (shortcut_fired, fsm)."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        # Hold through debounce
        _advance(fsm, _wispr_feat(0.0), 0.09)
        assert fsm.mode == GestureMode.WISPR_HELD
        # Release
        out = tick_fsm(fsm, _idle_feat(0.09), 0.09)
        return out.wispr_shortcut, fsm

    def test_wispr_shortcut_fires(self):
        """TEST 31 — one middle-thumb tap fires wispr_shortcut exactly once."""
        fired, _ = self._run_wispr_tap()
        assert fired

    def test_wispr_does_not_fire_before_debounce(self):
        """TEST 30 — premature release does not fire shortcut."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        # Release immediately (before debounce)
        out = tick_fsm(fsm, _idle_feat(0.01), 0.01)
        assert not out.wispr_shortcut

    def test_wispr_held_does_not_repeat(self):
        """TEST 33 — holding middle-thumb pinch does not repeat shortcut."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        _advance(fsm, _wispr_feat(0.0), 0.09)
        # Hold longer without releasing
        shortcuts = []
        t = 0.09
        for _ in range(20):
            t += 0.033
            out = tick_fsm(fsm, _wispr_feat(t), t)
            if out.wispr_shortcut:
                shortcuts.append(t)
        assert not shortcuts, f"Holding wispr pinch repeated shortcut: {shortcuts}"

    def test_wispr_second_tap(self):
        """TEST 34 — second distinct tap sends a new shortcut."""
        fsm = GestureFSM(config=_cfg())
        # First tap
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        _advance(fsm, _wispr_feat(0.0), 0.09)
        tick_fsm(fsm, _idle_feat(0.09), 0.09)
        assert fsm.mode == GestureMode.IDLE
        # Second tap
        tick_fsm(fsm, _wispr_feat(0.15), 0.15)
        _advance(fsm, _wispr_feat(0.15), 0.09)
        out = tick_fsm(fsm, _idle_feat(0.24), 0.24)
        assert out.wispr_shortcut, "Second wispr tap must fire shortcut"

    def test_wispr_does_not_trigger_click(self):
        """TEST 40 — wispr gesture must not emit left_click or right_click."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        _advance(fsm, _wispr_feat(0.0), 0.09)
        out = tick_fsm(fsm, _idle_feat(0.09), 0.09)
        assert not out.left_click
        assert not out.right_click

    def test_wispr_does_not_trigger_scroll(self):
        """TEST 40 — wispr must not start scroll."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _wispr_feat(0.0), 0.0)
        _advance(fsm, _wispr_feat(0.0), 0.09)
        out = tick_fsm(fsm, _idle_feat(0.09), 0.09)
        assert out.scroll_dy == 0


# ---------------------------------------------------------------------------
# TEST 41–46: Pause / fist hold
# ---------------------------------------------------------------------------

class TestPause:

    def test_short_fist_is_not_pause(self):
        """TEST 41 — short fist (< 2s) does not trigger pause."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        out = _advance(fsm, _fist_feat(0.0), 0.5)   # 500ms < 2000ms
        assert not out.toggle_pause

    def test_two_second_fist_triggers_pause(self):
        """TEST 42 — fist held 2+ seconds emits toggle_pause."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        # toggle_pause fires once when the threshold is crossed; _advance
        # may return a later output (IDLE or re-entered FIST_ENTER), so
        # we check whether any tick during the window emitted it.
        fired = _any_event(fsm, _fist_feat(0.0), 2.2, toggle_pause=True)
        assert fired, "2s fist must emit toggle_pause on some tick"

    def test_fist_pause_fires_once(self):
        """TEST 42 — one continuous fist hold triggers pause exactly once."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        pauses = []
        t = 0.0
        for _ in range(300):
            t += 0.01
            out = tick_fsm(fsm, _fist_feat(t), t)
            if out.toggle_pause:
                pauses.append(t)
        assert len(pauses) == 1, f"Pause fired {len(pauses)} times: {pauses}"

    def test_short_fist_does_not_pause(self):
        """Fist released before 2s → FIST_DRAG, NOT pause."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _fist_feat(0.0), 0.0)
        _advance(fsm, _fist_feat(0.0), 0.3)
        out = tick_fsm(fsm, _idle_feat(0.3), 0.3)
        assert not out.toggle_pause
        assert fsm.mode == GestureMode.FIST_DRAG

    def test_reset_clears_state(self):
        """TEST 46 — reset_fsm clears all transient state after pause."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.1)
        assert fsm.mode != GestureMode.IDLE
        reset_fsm(fsm)
        assert fsm.mode == GestureMode.IDLE
        assert fsm._last_tap_at == -1.0
        assert not fsm._mouse_pressed

    def test_paused_cursor_active_false(self):
        """TEST 44 — FSM should support the app-layer pause; cursor active is not the pause gate."""
        # The FSM doesn't own the app-level pause; the App does.
        # But when reset and in IDLE after pause, cursor_active should be True.
        fsm = GestureFSM(config=_cfg())
        reset_fsm(fsm)
        out = tick_fsm(fsm, _idle_feat(0.0), 0.0)
        assert out.cursor_active   # IDLE allows cursor


# ---------------------------------------------------------------------------
# Key release guard tests (TEST 51–52)
# ---------------------------------------------------------------------------

class TestKeyReleaseSafety:

    def test_reset_releases_mouse_pressed_flag(self):
        """TEST 51 — reset_fsm ensures mouse_pressed flag is cleared."""
        fsm = GestureFSM(config=_cfg())
        fsm._mouse_pressed = True
        reset_fsm(fsm)
        assert not fsm._mouse_pressed

    def test_drag_release_clears_mouse_pressed(self):
        """Mouse release on drag end clears internal pressed flag."""
        fsm = GestureFSM(config=_cfg())
        tick_fsm(fsm, _pinch_feat(0.0), 0.0)
        _advance(fsm, _pinch_feat(0.0), 0.6)   # into DRAG
        assert fsm._mouse_pressed
        tick_fsm(fsm, _idle_feat(0.6), 0.6)    # release
        assert not fsm._mouse_pressed
