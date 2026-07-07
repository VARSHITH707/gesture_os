"""Windows auto-start integration for GestureOS.

Uses the HKCU Run registry key — no administrator privileges required.
Verifies and deduplicates entries before writing.
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)

_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _open_run_key(write: bool = False):
    """Open the HKCU Run registry key. Returns (key handle, winreg module)."""
    import winreg
    access = winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE if write else winreg.KEY_QUERY_VALUE
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_PATH, 0, access)
    return key, winreg


def _build_launch_command() -> str:
    """Return the command string to launch GestureOS using the current Python."""
    python_exe = sys.executable
    main_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gesture_os",
        "main.py",
    )
    return f'"{python_exe}" "{main_script}"'


def register_autostart(name: str = "GestureOS") -> bool:
    """Register GestureOS to start automatically on Windows login.

    Returns True if the entry was created or already correct.
    Returns False if registration failed.
    """
    try:
        import winreg
        cmd = _build_launch_command()

        # Check for existing entry first.
        existing = get_autostart_command(name)
        if existing == cmd:
            logger.info("Autostart entry already correct: %s", cmd)
            return True

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_PATH,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
        )
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(key)
        logger.info("Autostart registered: %s = %s", name, cmd)
        return True
    except Exception as e:
        logger.error("register_autostart failed: %s", e)
        return False


def unregister_autostart(name: str = "GestureOS") -> bool:
    """Remove the autostart registry entry if it exists."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_PATH,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
        )
        try:
            winreg.DeleteValue(key, name)
            logger.info("Autostart entry removed: %s", name)
        except FileNotFoundError:
            logger.info("No autostart entry to remove for: %s", name)
        finally:
            winreg.CloseKey(key)
        return True
    except Exception as e:
        logger.error("unregister_autostart failed: %s", e)
        return False


def get_autostart_command(name: str = "GestureOS") -> str | None:
    """Read the current autostart command from the registry. Returns None if absent."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_PATH,
            0,
            winreg.KEY_QUERY_VALUE,
        )
        try:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
        except FileNotFoundError:
            return None
        finally:
            winreg.CloseKey(key)
    except Exception as e:
        logger.warning("get_autostart_command error: %s", e)
        return None


def verify_autostart(name: str = "GestureOS") -> dict:
    """Verify the autostart registration is present and points to the correct target.

    Returns a dict with keys: 'registered', 'command', 'expected', 'match'.
    """
    expected = _build_launch_command()
    current = get_autostart_command(name)
    return {
        "registered": current is not None,
        "command": current,
        "expected": expected,
        "match": current == expected,
    }
