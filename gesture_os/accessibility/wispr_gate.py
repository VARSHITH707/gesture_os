import subprocess
import logging
import time
from dataclasses import dataclass
from pynput.keyboard import Controller as KbController, Key, HotKey
from gesture_os.accessibility.focus_monitor import FocusState, FocusConfig, focus_state_generator

logger = logging.getLogger(__name__)
_keyboard = KbController()


@dataclass
class WisprConfig:
    """Configuration for the Wispr Flow dictation trigger."""
    activation_hotkey: list = None
    arming_delay_ms: float = 150.0

    def __post_init__(self):
        """Set default hotkey if none provided."""
        if self.activation_hotkey is None:
            self.activation_hotkey = [Key.ctrl, Key.shift]


def arm_wispr(config: WisprConfig) -> None:
    """Trigger Wispr Flow dictation via its hotkey."""
    try:
        time.sleep(config.arming_delay_ms / 1000.0)
        for k in config.activation_hotkey:
            _keyboard.press(k)
        for k in reversed(config.activation_hotkey):
            _keyboard.release(k)
        logger.info("Wispr Flow armed via hotkey.")
    except Exception as e:
        logger.warning(f"arm_wispr error: {e}")


def disarm_wispr(config: WisprConfig) -> None:
    """Stop Wispr Flow dictation via its hotkey (toggle off)."""
    try:
        for k in config.activation_hotkey:
            _keyboard.press(k)
        for k in reversed(config.activation_hotkey):
            _keyboard.release(k)
        logger.info("Wispr Flow disarmed.")
    except Exception as e:
        logger.warning(f"disarm_wispr error: {e}")


def focus_routing_loop(
    wispr_config: WisprConfig | None = None,
    focus_config: FocusConfig | None = None,
) -> None:
    """Watch focus state and arm or disarm Wispr accordingly."""
    wispr_config = wispr_config or WisprConfig()
    focus_config = focus_config or FocusConfig()
    wispr_armed = False
    try:
        for state in focus_state_generator(focus_config):
            if state == FocusState.EDITABLE and not wispr_armed:
                arm_wispr(wispr_config)
                wispr_armed = True
            elif state != FocusState.EDITABLE and wispr_armed:
                disarm_wispr(wispr_config)
                wispr_armed = False
    except KeyboardInterrupt:
        if wispr_armed:
            disarm_wispr(wispr_config)
        logger.info("Focus routing loop stopped.")
    except Exception as e:
        logger.error(f"focus_routing_loop crashed: {e}")