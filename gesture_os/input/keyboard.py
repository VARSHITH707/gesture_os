"""Safe keyboard shortcut execution with guaranteed key release.

All hotkey functions use try/finally to ensure modifier keys are
released even if an exception occurs mid-sequence.
"""
import logging
from pynput.keyboard import Controller as KbController, Key

logger = logging.getLogger(__name__)

_keyboard = KbController()

# Named key lookup table — all imports at module level, no in-loop lookups.
_KEY_MAP: dict = {
    "ctrl": Key.ctrl,
    "alt": Key.alt,
    "shift": Key.shift,
    "super": Key.cmd,
    "win": Key.cmd,
    "tab": Key.tab,
    "f4": Key.f4,
    "volumemute": Key.media_volume_mute,
    "volumeup": Key.media_volume_up,
    "volumedown": Key.media_volume_down,
    "m": "m",
}

# Modifier keys that must always be released on cleanup.
_MODIFIER_KEYS = (Key.ctrl, Key.alt, Key.shift, Key.cmd)


def release_all_modifiers() -> None:
    """Release all known modifier keys unconditionally."""
    for k in _MODIFIER_KEYS:
        try:
            _keyboard.release(k)
        except Exception:
            pass


def send_hotkey(keys: list) -> None:
    """Press and release a sequence of keys atomically.

    Keys are pressed in order and released in reverse order.
    Modifier keys are always released in the finally block.
    """
    resolved: list = []
    for k in keys:
        if isinstance(k, str):
            resolved.append(_KEY_MAP.get(k.lower(), k))
        else:
            resolved.append(k)

    pressed: list = []
    try:
        for k in resolved:
            _keyboard.press(k)
            pressed.append(k)
        for k in reversed(pressed):
            _keyboard.release(k)
        logger.debug("Hotkey sent: %s", keys)
    except Exception as e:
        logger.warning("send_hotkey error: %s", e)
        for k in reversed(pressed):
            try:
                _keyboard.release(k)
            except Exception:
                pass
    finally:
        release_all_modifiers()


def send_ctrl_alt() -> None:
    """Send a complete Ctrl+Alt press-and-release (Wispr Flow toggle)."""
    try:
        _keyboard.press(Key.ctrl)
        _keyboard.press(Key.alt)
        _keyboard.release(Key.alt)
        _keyboard.release(Key.ctrl)
        logger.info("Sent Ctrl+Alt (Wispr toggle)")
    except Exception as e:
        logger.warning("send_ctrl_alt error: %s", e)
    finally:
        try:
            _keyboard.release(Key.alt)
        except Exception:
            pass
        try:
            _keyboard.release(Key.ctrl)
        except Exception:
            pass


def send_win_m() -> None:
    """Send Win+M (minimize all)."""
    try:
        _keyboard.press(Key.cmd)
        _keyboard.press("m")
        _keyboard.release("m")
        _keyboard.release(Key.cmd)
    except Exception as e:
        logger.warning("send_win_m error: %s", e)
    finally:
        try:
            _keyboard.release(Key.cmd)
        except Exception:
            pass


def send_win_shift_m() -> None:
    """Send Win+Shift+M (restore all minimized)."""
    try:
        _keyboard.press(Key.cmd)
        _keyboard.press(Key.shift)
        _keyboard.press("m")
        _keyboard.release("m")
        _keyboard.release(Key.shift)
        _keyboard.release(Key.cmd)
    except Exception as e:
        logger.warning("send_win_shift_m error: %s", e)
    finally:
        try:
            _keyboard.release(Key.shift)
        except Exception:
            pass
        try:
            _keyboard.release(Key.cmd)
        except Exception:
            pass


def send_alt_tab_forward() -> None:
    """Send Ctrl+Tab (move forward in switcher)."""
    try:
        _keyboard.press(Key.ctrl)
        _keyboard.press(Key.tab)
        _keyboard.release(Key.tab)
        _keyboard.release(Key.ctrl)
    except Exception as e:
        logger.warning("send_alt_tab_forward error: %s", e)
    finally:
        try:
            _keyboard.release(Key.ctrl)
        except Exception:
            pass


def send_alt_tab_backward() -> None:
    """Send Ctrl+Shift+Tab (move backward in switcher)."""
    try:
        _keyboard.press(Key.ctrl)
        _keyboard.press(Key.shift)
        _keyboard.press(Key.tab)
        _keyboard.release(Key.tab)
        _keyboard.release(Key.shift)
        _keyboard.release(Key.ctrl)
    except Exception as e:
        logger.warning("send_alt_tab_backward error: %s", e)
    finally:
        try:
            _keyboard.release(Key.shift)
        except Exception:
            pass
        try:
            _keyboard.release(Key.ctrl)
        except Exception:
            pass
