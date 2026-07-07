"""System tray interface for GestureOS.

Runs pystray in a background thread. The tray icon reflects the
current application state (active / paused) and provides pause,
resume, and exit actions.
"""
import logging
import threading
from typing import Callable

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def _make_icon(active: bool) -> Image.Image:
    """Draw a simple 64x64 tray icon. Green = active, grey = paused."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (0, 200, 80) if active else (140, 140, 140)
    draw.ellipse([4, 4, 60, 60], fill=color)
    # Hand silhouette: a simple upward-pointing triangle
    draw.polygon([(32, 12), (20, 48), (44, 48)], fill=(255, 255, 255, 200))
    return img


class SystemTray:
    """Wraps pystray to provide a Windows system tray interface."""

    def __init__(
        self,
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_exit: Callable[[], None],
        is_active_fn: Callable[[], bool],
    ) -> None:
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_exit = on_exit
        self._is_active_fn = is_active_fn
        self._icon = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the tray icon in a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="SystemTrayThread",
        )
        self._thread.start()
        logger.info("System tray thread started.")

    def update_icon(self) -> None:
        """Refresh the tray icon to reflect current active/paused state."""
        if self._icon is not None:
            try:
                active = self._is_active_fn()
                self._icon.icon = _make_icon(active)
                label = "GestureOS — Active" if active else "GestureOS — Paused"
                self._icon.title = label
            except Exception as e:
                logger.debug("update_icon error: %s", e)

    def stop(self) -> None:
        """Stop the tray icon."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import pystray
        except ImportError:
            logger.error("pystray not installed — tray unavailable.")
            return

        def _pause_action(icon, item) -> None:  # noqa: ARG001
            try:
                if self._is_active_fn():
                    self._on_pause()
                else:
                    self._on_resume()
                self.update_icon()
            except Exception as e:
                logger.warning("Tray pause/resume error: %s", e)

        def _exit_action(icon, item) -> None:  # noqa: ARG001
            try:
                self._on_exit()
            except Exception as e:
                logger.warning("Tray exit error: %s", e)
            icon.stop()

        def _dynamic_label(item) -> str:  # noqa: ARG001
            return "Pause" if self._is_active_fn() else "Resume"

        active = self._is_active_fn()
        self._icon = pystray.Icon(
            "GestureOS",
            _make_icon(active),
            "GestureOS — Active" if active else "GestureOS — Paused",
            menu=pystray.Menu(
                pystray.MenuItem(_dynamic_label, _pause_action, default=False),
                pystray.MenuItem("Exit", _exit_action),
            ),
        )
        try:
            self._icon.run()
        except Exception as e:
            logger.error("Tray icon crashed: %s", e)
