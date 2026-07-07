import subprocess
import logging
from pynput.keyboard import Controller as KbController, Key

logger = logging.getLogger(__name__)
_keyboard = KbController()

_KEY_MAP = {
    "ctrl": Key.ctrl, "alt": Key.alt, "shift": Key.shift,
    "super": Key.cmd, "tab": Key.tab, "F4": Key.f4,
    "volumemute": Key.media_volume_mute,
    "volumeup": Key.media_volume_up,
    "volumedown": Key.media_volume_down,
}


def match_intent(transcript: str, registry: dict) -> dict | None:
    """Match a transcript string to a command in the registry."""
    cleaned = transcript.lower().strip()
    if cleaned in registry:
        return registry[cleaned]
    for key in registry:
        if key in cleaned:
            return registry[key]
    return None


def execute_command(command: dict) -> None:
    """Execute a matched voice command dict."""
    try:
        action = command.get("action")
        if action == "launch":
            target = command.get("target", "")
            subprocess.Popen([target], shell=False)
            logger.info(f"Launched: {target}")
        elif action == "hotkey":
            keys = [_KEY_MAP.get(k, k) for k in command.get("keys", [])]
            for k in keys:
                _keyboard.press(k)
            for k in reversed(keys):
                _keyboard.release(k)
            logger.info(f"Hotkey: {command.get('keys')}")
        else:
            logger.warning(f"Unknown action: {action}")
    except Exception as e:
        logger.warning(f"execute_command error: {e}")