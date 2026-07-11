"""One authoritative gesture finite state machine.

All gesture states are coordinated here. No global mutable state —
every piece of state lives inside GestureFSM. All frame-critical
lookups use precomputed landmark index constants.
"""
import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from gesture_os.config import GestureConfig
from gesture_os.vision.filters import OneEuroState, one_euro_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Precomputed landmark index constants (evaluated once at import time)
# ---------------------------------------------------------------------------
LM_WRIST       = 0
LM_THUMB_TIP   = 4
LM_INDEX_MCP   = 5
LM_INDEX_PIP   = 6
LM_INDEX_TIP   = 8
LM_MIDDLE_MCP  = 9
LM_MIDDLE_PIP  = 10
LM_MIDDLE_TIP  = 12
LM_RING_MCP    = 13
LM_RING_PIP    = 14
LM_RING_TIP    = 16
LM_PINKY_MCP   = 17
LM_PINKY_PIP   = 18
LM_PINKY_TIP   = 20

_TIPS = (LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP)
_PIPS = (LM_INDEX_PIP, LM_MIDDLE_PIP, LM_RING_PIP, LM_PINKY_PIP)
_MCPS = (LM_INDEX_MCP, LM_MIDDLE_MCP, LM_RING_MCP, LM_PINKY_MCP)

MIN_HAND_SIZE = 1e-4  # floor to avoid divide-by-zero on degenerate landmark input


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class GestureFeatures:
    """Per-frame features extracted from one hand's landmark array."""
    index_tip_x: float
    index_tip_y: float
    scroll_ref_y: float          # average y of index+middle tips for scroll
    thumb_index_dist: float      # hand-size-normalized ratio, not a raw frame distance
    thumb_middle_dist: float     # hand-size-normalized ratio, not a raw frame distance
    is_fist: bool
    is_open_palm: bool
    is_scroll_pose: bool
    is_scroll_shape: bool        # finger extension only, ignores thumb distance
    is_index_pinch: bool         # thumb-index only (middle NOT pinching)
    is_wispr_pinch: bool         # thumb-middle only (index NOT pinching)
    is_three_finger_pinch: bool  # both index AND middle close to thumb
    timestamp: float
    hand_size: float = 0.2       # wrist-to-middle-MCP distance; the normalization reference for the ratios above


def extract_features(
    landmarks: np.ndarray,
    cfg: GestureConfig,
    timestamp: float,
) -> GestureFeatures:
    """Extract all gesture features from a (21, 3) float32 array."""

    def _dist2d(a: int, b: int) -> float:
        dx = float(landmarks[a, 0] - landmarks[b, 0])
        dy = float(landmarks[a, 1] - landmarks[b, 1])
        return (dx * dx + dy * dy) ** 0.5

    # Hand-size reference (wrist -> middle MCP), same convention as AirBench's
    # hands.js. Every distance below is divided by this, so pinch/fist/open
    # thresholds mean the same thing whether the hand is close to the camera
    # or far from it — a fixed frame-space distance only ever worked at one
    # specific hand-to-camera distance.
    hand_size = max(_dist2d(LM_WRIST, LM_MIDDLE_MCP), MIN_HAND_SIZE)

    ti_dist = _dist2d(LM_THUMB_TIP, LM_INDEX_TIP) / hand_size
    tm_dist = _dist2d(LM_THUMB_TIP, LM_MIDDLE_TIP) / hand_size

    # Per-finger extension ratio: tip-to-MCP distance over hand size — a real
    # geometric curl measurement, robust to hand rotation/tilt. Replaces the
    # old `tip.y < pip.y` screen-space heuristic, which only worked when the
    # hand was held upright facing the camera (ported from AirBench's
    # tip-to-MCP extension metric; same landmark model, same reasoning).
    #
    # Aggregation stays count-based (not averaged) on purpose: this app has a
    # "pointing" pose (index extended, other 3 curled) that must read as
    # neither a fist nor an open palm, and an average can't tell "2 of 4
    # extended" (scroll) apart from "all 4 partially extended" the way a
    # per-finger count threshold can. AirBench doesn't have this ambiguity
    # (no pointing-only pose, and its fist check excludes index entirely) so
    # its average/exclusion approach doesn't transfer as-is — only the
    # underlying distance metric does.
    ext_ratio = [_dist2d(t, m) / hand_size for t, m in zip(_TIPS, _MCPS)]
    ext = [r > cfg.finger_extend_ratio for r in ext_ratio]
    n_ext = sum(ext)

    is_fist = (n_ext == 0)
    is_open_palm = (n_ext >= cfg.open_palm_min_fingers)

    # Scroll shape: index AND middle extended, ring AND pinky curled.
    # Deliberately excludes thumb distance so callers can use it for exit
    # hysteresis (a momentary thumb dip mid-scroll shouldn't collapse to pinch).
    scroll_shape = ext[0] and ext[1] and not ext[2] and not ext[3] and not is_fist

    # Scroll pose (entry gate): scroll shape AND thumb clear of both pinches.
    is_scroll = (
        scroll_shape
        and ti_dist >= cfg.pinch_enter
        and tm_dist >= cfg.wispr_enter
    )

    # Exclusive pinch classification
    is_idx = (ti_dist < cfg.pinch_enter and tm_dist >= cfg.wispr_enter and not is_fist)
    is_wsp = (tm_dist < cfg.wispr_enter and ti_dist >= cfg.pinch_enter and not is_fist)
    is_3f  = (ti_dist < cfg.three_finger_enter and tm_dist < cfg.three_finger_enter and not is_fist)

    scroll_ref = (float(landmarks[LM_INDEX_TIP, 1]) + float(landmarks[LM_MIDDLE_TIP, 1])) * 0.5

    return GestureFeatures(
        index_tip_x=float(landmarks[LM_INDEX_TIP, 0]),
        index_tip_y=float(landmarks[LM_INDEX_TIP, 1]),
        scroll_ref_y=scroll_ref,
        hand_size=hand_size,
        thumb_index_dist=ti_dist,
        thumb_middle_dist=tm_dist,
        is_fist=is_fist,
        is_open_palm=is_open_palm,
        is_scroll_pose=is_scroll,
        is_scroll_shape=scroll_shape,
        is_index_pinch=is_idx,
        is_wispr_pinch=is_wsp,
        is_three_finger_pinch=is_3f,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# FSM state enum
# ---------------------------------------------------------------------------

class GestureMode(Enum):
    IDLE         = "idle"
    PINCH_ENTER  = "pinch_enter"   # debouncing initial index-thumb pinch
    PINCH_HELD   = "pinch_held"    # confirmed pinch; may become tap or drag
    TAP_SETTLE   = "tap_settle"    # pinch released; waiting for double-tap window
    DRAG         = "drag"          # long-press drag (left mouse held)
    THREE_ENTER  = "three_enter"   # three-finger pinch held (first tap)
    THREE_SETTLE = "three_settle"  # first three-finger tap released; wait for second
    FIST_ENTER   = "fist_enter"    # fist held; timing for pause vs short drag
    FIST_DRAG    = "fist_drag"     # short fist drag active
    SCROLL       = "scroll"        # two-finger scroll pose active
    SWIPE_TRACK  = "swipe_track"   # open palm; tracking for swipe
    WISPR_ENTER  = "wispr_enter"   # middle-thumb pinch debouncing
    WISPR_HELD   = "wispr_held"    # confirmed wispr pinch; fire on release


# ---------------------------------------------------------------------------
# FSM output
# ---------------------------------------------------------------------------

@dataclass
class FSMOutput:
    """Events emitted by one tick_fsm call."""
    left_click:    bool = False
    right_click:   bool = False
    mouse_press:   bool = False
    mouse_release: bool = False
    scroll_dy:     int  = 0          # positive = down, negative = up
    swipe:         str | None = None # "LEFT" | "RIGHT" | "UP" | "DOWN"
    wispr_shortcut: bool = False
    toggle_pause:  bool = False
    cursor_active: bool = True       # False suppresses cursor movement
    mode:          GestureMode = GestureMode.IDLE


# ---------------------------------------------------------------------------
# FSM mutable state container
# ---------------------------------------------------------------------------

@dataclass
class GestureFSM:
    """Owns all gesture state. Must be initialised once; reused every frame."""
    config: GestureConfig = field(default_factory=GestureConfig)

    mode:        GestureMode = field(default=GestureMode.IDLE, init=False)
    _entered_at: float       = field(default=0.0, init=False)

    # Tap timing
    _last_tap_at:   float = field(default=-1.0, init=False)
    _three_tap_at:  float = field(default=-1.0, init=False)

    # Scroll
    _scroll_prev_y: float = field(default=0.0, init=False)
    _scroll_prev_t: float = field(default=0.0, init=False)
    _scroll_y_filter: OneEuroState = field(default_factory=OneEuroState, init=False)

    # Swipe
    _swipe_start_x: float = field(default=0.0, init=False)
    _swipe_start_y: float = field(default=0.0, init=False)
    _swipe_start_t: float = field(default=0.0, init=False)
    _swipe_cooldowns: dict = field(default_factory=dict, init=False)

    # Wispr local toggle
    _wispr_active: bool = field(default=False, init=False)

    # Mouse pressed guard (prevents double-press)
    _mouse_pressed: bool = field(default=False, init=False)


# ---------------------------------------------------------------------------
# Core tick function
# ---------------------------------------------------------------------------

def tick_fsm(fsm: GestureFSM, feat: GestureFeatures, now: float) -> FSMOutput:
    """Advance the gesture FSM by one frame. Mutates fsm. Returns FSMOutput."""
    cfg = fsm.config
    out = FSMOutput(mode=fsm.mode)

    def _enter(m: GestureMode) -> None:
        fsm.mode = m
        fsm._entered_at = now
        out.mode = m

    def _ms() -> float:
        return (now - fsm._entered_at) * 1000.0

    # ------------------------------------------------------------------
    # IDLE
    # ------------------------------------------------------------------
    if fsm.mode == GestureMode.IDLE:
        out.cursor_active = True

        if feat.is_fist:
            _enter(GestureMode.FIST_ENTER)

        elif feat.is_wispr_pinch:
            _enter(GestureMode.WISPR_ENTER)

        elif feat.is_three_finger_pinch:
            _enter(GestureMode.THREE_ENTER)

        elif feat.is_scroll_pose:
            fsm._scroll_y_filter = OneEuroState()
            fsm._scroll_prev_y = one_euro_filter(fsm._scroll_y_filter, feat.scroll_ref_y, now)
            fsm._scroll_prev_t = now
            _enter(GestureMode.SCROLL)
            out.cursor_active = False

        elif feat.is_index_pinch:
            _enter(GestureMode.PINCH_ENTER)

        elif feat.is_open_palm:
            # Only enter swipe tracking if not in swipe cooldown for all directions
            fsm._swipe_start_x = feat.index_tip_x
            fsm._swipe_start_y = feat.index_tip_y
            fsm._swipe_start_t = now
            _enter(GestureMode.SWIPE_TRACK)
            out.cursor_active = False

    # ------------------------------------------------------------------
    # PINCH_ENTER — debouncing initial index-thumb pinch
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.PINCH_ENTER:
        out.cursor_active = True

        if feat.thumb_index_dist >= cfg.pinch_exit:
            # Released before debounce window — treat as noise
            _enter(GestureMode.IDLE)
        elif _ms() >= cfg.debounce_ms:
            # Debounced: confirm pinch. Emit virtual press for drag tracking.
            if not fsm._mouse_pressed:
                out.mouse_press = True
                fsm._mouse_pressed = True
            _enter(GestureMode.PINCH_HELD)

    # ------------------------------------------------------------------
    # PINCH_HELD — waiting for release or long-press threshold
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.PINCH_HELD:
        out.cursor_active = True

        if feat.thumb_index_dist >= cfg.pinch_exit:
            # Quick release → potential tap
            if fsm._mouse_pressed:
                out.mouse_release = True
                fsm._mouse_pressed = False
            fsm._last_tap_at = now
            _enter(GestureMode.TAP_SETTLE)

        elif _ms() >= cfg.long_press_ms:
            # Held long enough → drag begins; mouse is already pressed
            _enter(GestureMode.DRAG)

    # ------------------------------------------------------------------
    # TAP_SETTLE — waiting for double-tap window
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.TAP_SETTLE:
        out.cursor_active = True

        if feat.is_index_pinch:
            # Second tap within window → right click (double-tap action)
            out.right_click = True
            fsm._last_tap_at = -1.0
            _enter(GestureMode.IDLE)

        elif _ms() >= cfg.double_tap_ms:
            # No second tap → emit single left click
            out.left_click = True
            _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # DRAG — long-press drag (mouse button held throughout)
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.DRAG:
        out.cursor_active = True

        if feat.thumb_index_dist >= cfg.pinch_exit:
            if fsm._mouse_pressed:
                out.mouse_release = True
                fsm._mouse_pressed = False
            _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # THREE_ENTER — first three-finger pinch held
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.THREE_ENTER:
        out.cursor_active = True

        if not feat.is_three_finger_pinch:
            # First tap released → start settle window
            fsm._three_tap_at = now
            _enter(GestureMode.THREE_SETTLE)
        elif _ms() >= cfg.long_press_ms:
            # Held too long → cancel, treat as nothing
            _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # THREE_SETTLE — waiting for second three-finger tap
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.THREE_SETTLE:
        out.cursor_active = True

        if feat.is_three_finger_pinch:
            # Second tap → left click
            out.left_click = True
            fsm._three_tap_at = -1.0
            _enter(GestureMode.IDLE)
        elif _ms() >= cfg.double_tap_ms:
            # Window expired — no action
            _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # FIST_ENTER — timing fist duration: short = drag, long = pause
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.FIST_ENTER:
        out.cursor_active = False

        if not feat.is_fist:
            # Fist released before pause threshold → short fist drag
            if not fsm._mouse_pressed:
                out.mouse_press = True
                fsm._mouse_pressed = True
            _enter(GestureMode.FIST_DRAG)

        elif _ms() >= cfg.fist_pause_ms:
            # Held 2+ seconds → pause toggle
            out.toggle_pause = True
            _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # FIST_DRAG — short fist drag active
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.FIST_DRAG:
        out.cursor_active = True

        if not feat.is_fist:
            if fsm._mouse_pressed:
                out.mouse_release = True
                fsm._mouse_pressed = False
            _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # SCROLL — two-finger scroll pose
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.SCROLL:
        out.cursor_active = False

        # Hysteresis: only exit on losing the scroll finger shape (or a fist),
        # not on thumb proximity — a momentary pinch-distance dip mid-scroll
        # must not collapse this into PINCH_ENTER.
        if not feat.is_scroll_shape:
            _enter(GestureMode.IDLE)
        else:
            filtered_y = one_euro_filter(fsm._scroll_y_filter, feat.scroll_ref_y, now)
            delta_y = filtered_y - fsm._scroll_prev_y

            if abs(delta_y) >= cfg.scroll_dead_zone:
                ticks = delta_y * cfg.scroll_speed_scale
                ticks = max(-cfg.max_scroll_speed,
                            min(int(round(ticks)), cfg.max_scroll_speed))
                out.scroll_dy = ticks

            fsm._scroll_prev_y = filtered_y
            fsm._scroll_prev_t = now

    # ------------------------------------------------------------------
    # SWIPE_TRACK — open palm, tracking for swipe
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.SWIPE_TRACK:
        out.cursor_active = False

        if not feat.is_open_palm:
            _enter(GestureMode.IDLE)
        else:
            dx = feat.index_tip_x - fsm._swipe_start_x
            dy = feat.index_tip_y - fsm._swipe_start_y
            dt = max(now - fsm._swipe_start_t, 1e-6)
            dist = (dx * dx + dy * dy) ** 0.5
            velocity = dist / dt

            if velocity >= cfg.swipe_min_velocity and dist >= cfg.swipe_min_displacement:
                direction = _classify_swipe(dx, dy, cfg.swipe_direction_tolerance_deg)
                if direction:
                    last_t = fsm._swipe_cooldowns.get(direction, float("-inf"))
                    if (now - last_t) >= cfg.swipe_cooldown_s:
                        out.swipe = direction
                        fsm._swipe_cooldowns[direction] = now
                _enter(GestureMode.IDLE)

    # ------------------------------------------------------------------
    # WISPR_ENTER — debouncing middle-thumb pinch
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.WISPR_ENTER:
        out.cursor_active = True

        if feat.thumb_middle_dist >= cfg.wispr_exit:
            # Released before debounce — ignore
            _enter(GestureMode.IDLE)
        elif _ms() >= cfg.wispr_debounce_ms:
            _enter(GestureMode.WISPR_HELD)

    # ------------------------------------------------------------------
    # WISPR_HELD — confirmed; fire shortcut on release
    # ------------------------------------------------------------------
    elif fsm.mode == GestureMode.WISPR_HELD:
        out.cursor_active = True

        if feat.thumb_middle_dist >= cfg.wispr_exit:
            out.wispr_shortcut = True
            fsm._wispr_active = not fsm._wispr_active
            _enter(GestureMode.IDLE)

    out.mode = fsm.mode
    return out


def _classify_swipe(dx: float, dy: float, tol_deg: float) -> str | None:
    angle = math.degrees(math.atan2(dy, dx))
    if -tol_deg <= angle <= tol_deg:
        return "RIGHT"
    if angle >= (180 - tol_deg) or angle <= (-180 + tol_deg):
        return "LEFT"
    if (90 - tol_deg) <= angle <= (90 + tol_deg):
        return "DOWN"
    if (-90 - tol_deg) <= angle <= (-90 + tol_deg):
        return "UP"
    return None


def reset_fsm(fsm: GestureFSM) -> None:
    """Reset all transient FSM state. Call on pause, resume, or hand reacquisition."""
    fsm.mode = GestureMode.IDLE
    fsm._entered_at = 0.0
    fsm._last_tap_at = -1.0
    fsm._three_tap_at = -1.0
    fsm._scroll_prev_y = 0.0
    fsm._scroll_prev_t = 0.0
    fsm._scroll_y_filter = OneEuroState()
    fsm._swipe_start_x = 0.0
    fsm._swipe_start_y = 0.0
    fsm._swipe_start_t = 0.0
    if fsm._mouse_pressed:
        fsm._mouse_pressed = False
