"""Relative trackpad-style cursor controller.

Finger displacement in normalized camera space is converted to relative
screen pixel displacement, mirroring laptop trackpad behavior.

The hand does NOT represent an absolute screen location. When the hand
leaves the frame the cursor freezes. When it returns, movement resumes
from the cursor's current screen position without jumping.
"""
import logging
from dataclasses import dataclass

from gesture_os.config import CursorConfig

logger = logging.getLogger(__name__)


@dataclass
class CursorState:
    """Mutable state for relative cursor tracking."""
    # Last known normalized finger position (None = hand not seen yet)
    ref_x: float | None = None
    ref_y: float | None = None
    # Screen pixel position that the controller last wrote
    screen_x: int = 960
    screen_y: int = 540
    # Whether the next detection is a reacquisition frame
    reacquiring: bool = False


class RelativeCursorController:
    """Converts normalized finger deltas to absolute screen positions."""

    def __init__(
        self,
        config: CursorConfig,
        screen_width: int,
        screen_height: int,
    ) -> None:
        self._cfg = config
        self._sw = screen_width
        self._sh = screen_height
        self._state = CursorState(screen_x=screen_width // 2, screen_y=screen_height // 2)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_hand_lost(self) -> None:
        """Call every frame where no hand landmarks are detected."""
        self._state.ref_x = None
        self._state.ref_y = None
        self._state.reacquiring = True

    def update(self, finger_x: float, finger_y: float) -> tuple[int, int]:
        """Compute new screen position from current finger location.

        Returns the new (x, y) screen position. On the first frame after
        reacquisition, returns the frozen cursor position unchanged and
        establishes the new reference — preventing jumps.
        """
        state = self._state

        if state.ref_x is None or state.reacquiring:
            # Reacquisition: anchor reference, do not move cursor.
            state.ref_x = finger_x
            state.ref_y = finger_y
            state.reacquiring = False
            return state.screen_x, state.screen_y

        dx_norm = finger_x - state.ref_x
        dy_norm = finger_y - state.ref_y

        # Dead zone suppresses micro-tremor.
        dist = (dx_norm ** 2 + dy_norm ** 2) ** 0.5
        if dist < self._cfg.dead_zone:
            state.ref_x = finger_x
            state.ref_y = finger_y
            return state.screen_x, state.screen_y

        # Convert normalized delta to pixel delta, apply sensitivity and clamp.
        dx_px = dx_norm * self._sw * self._cfg.sensitivity
        dy_px = dy_norm * self._sh * self._cfg.sensitivity

        max_d = self._cfg.max_delta_px
        dx_px = max(-max_d, min(dx_px, max_d))
        dy_px = max(-max_d, min(dy_px, max_d))

        new_x = int(state.screen_x + dx_px)
        new_y = int(state.screen_y + dy_px)

        # Clamp to screen bounds.
        new_x = max(0, min(new_x, self._sw - 1))
        new_y = max(0, min(new_y, self._sh - 1))

        state.screen_x = new_x
        state.screen_y = new_y
        state.ref_x = finger_x
        state.ref_y = finger_y

        return new_x, new_y

    def force_position(self, x: int, y: int) -> None:
        """Directly set screen position (used by external moves)."""
        self._state.screen_x = x
        self._state.screen_y = y

    def reset(self) -> None:
        """Reset reference on pause/resume to prevent jump on resume."""
        self._state.ref_x = None
        self._state.ref_y = None
        self._state.reacquiring = True
