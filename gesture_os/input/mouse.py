from dataclasses import dataclass
import logging
from pynput import mouse as _pynput_mouse

logger = logging.getLogger(__name__)


@dataclass
class MonitorLayout:
    """Virtual desktop bounds across all monitors."""
    total_width: int = 1920
    total_height: int = 1080
    origin_x: int = 0
    origin_y: int = 0


def build_monitor_layout() -> MonitorLayout:
    """Detect all monitors and return unified desktop bounds."""
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        min_x = min(m.x for m in monitors)
        min_y = min(m.y for m in monitors)
        max_x = max(m.x + m.width for m in monitors)
        max_y = max(m.y + m.height for m in monitors)
        return MonitorLayout(
            total_width=max_x - min_x,
            total_height=max_y - min_y,
            origin_x=min_x,
            origin_y=min_y,
        )
    except Exception as e:
        logger.warning(f"Monitor detection failed, using defaults: {e}")
        return MonitorLayout()


def normalized_to_screen(
    nx: float, ny: float, layout: MonitorLayout
) -> tuple[int, int]:
    """Map normalized 0-1 coords to absolute screen pixel coords."""
    px = int(layout.origin_x + nx * layout.total_width)
    py = int(layout.origin_y + ny * layout.total_height)
    px = max(layout.origin_x, min(px, layout.origin_x + layout.total_width - 1))
    py = max(layout.origin_y, min(py, layout.origin_y + layout.total_height - 1))
    return px, py


class MouseController:
    """Wraps pynput mouse controller for safe OS event injection."""

    def __init__(self) -> None:
        """Initialize the pynput mouse controller."""
        try:
            self._ctrl = _pynput_mouse.Controller()
            self._last_pos: tuple[int, int] = (0, 0)
        except Exception as e:
            raise RuntimeError(f"MouseController init failed: {e}") from e

    def _btn(self, name: str) -> _pynput_mouse.Button:
        """Resolve button name string to pynput Button enum."""
        return _pynput_mouse.Button.right if name == "right" \
            else _pynput_mouse.Button.left

    def move_to(self, x: int, y: int) -> None:
        """Move cursor to absolute screen position (x, y)."""
        try:
            self._ctrl.position = (x, y)
            self._last_pos = (x, y)
        except Exception as e:
            logger.warning(f"move_to error: {e}")

    def click(self, button: str = "left") -> None:
        """Emit a single click event."""
        try:
            self._ctrl.click(self._btn(button), 1)
        except Exception as e:
            logger.warning(f"click error: {e}")

    def double_click(self, button: str = "left") -> None:
        """Emit a double-click event."""
        try:
            self._ctrl.click(self._btn(button), 2)
        except Exception as e:
            logger.warning(f"double_click error: {e}")

    def press(self, button: str = "left") -> None:
        """Press and hold a mouse button."""
        try:
            self._ctrl.press(self._btn(button))
        except Exception as e:
            logger.warning(f"press error: {e}")

    def release(self, button: str = "left") -> None:
        """Release a held mouse button."""
        try:
            self._ctrl.release(self._btn(button))
        except Exception as e:
            logger.warning(f"release error: {e}")

    def scroll(self, dy: int) -> None:
        """Scroll vertically. Positive dy = scroll down, negative = up."""
        try:
            self._ctrl.scroll(0, -dy)
        except Exception as e:
            logger.warning(f"scroll error: {e}")

    def release_all(self) -> None:
        """Unconditionally release both mouse buttons. Called on shutdown/pause."""
        for btn in (_pynput_mouse.Button.left, _pynput_mouse.Button.right):
            try:
                self._ctrl.release(btn)
            except Exception:
                pass

    @property
    def position(self) -> tuple[int, int]:
        """Return the current cursor screen position."""
        try:
            pos = self._ctrl.position
            return (int(pos[0]), int(pos[1]))
        except Exception:
            return self._last_pos