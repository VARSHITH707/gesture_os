import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Generator

logger = logging.getLogger(__name__)


class FocusState(Enum):
    """OS focus state for the current foreground element."""
    UNKNOWN = "unknown"
    EDITABLE = "editable"
    NON_EDITABLE = "non_editable"


@dataclass
class FocusConfig:
    """Configuration for the accessibility focus monitor."""
    poll_interval_ms: float = 200.0
    editable_roles: tuple = (
        "Edit", "Document", "TextArea",
        "AXTextField", "AXTextArea", "AXComboBox",
        "AXWebArea",
    )


def get_focused_element_role() -> str:
    """Return the control-type role string of the current focused element."""
    try:
        if sys.platform == "win32":
            from pywinauto import Desktop
            el = Desktop(backend="uia").get_active()
            if el is None:
                return "unknown"
            return el.element_info.control_type or "unknown"
        elif sys.platform == "darwin":
            from AppKit import NSWorkspace
            import objc
            system_wide = objc.lookUpClass(
                "AXUIElement"
            ).systemWideElement()
            focused = system_wide.accessibilityAttributeValue_(
                "AXFocusedUIElement"
            )
            if focused is None:
                return "unknown"
            role = focused.accessibilityAttributeValue_("AXRole")
            return role or "unknown"
        else:
            import pyatspi
            el = pyatspi.Registry.getDesktop(0)
            focused = pyatspi.findDescendant(
                el, lambda x: x.getState().contains(
                    pyatspi.STATE_FOCUSED
                )
            )
            if focused is None:
                return "unknown"
            return focused.getRoleName() or "unknown"
    except Exception as e:
        logger.warning(f"get_focused_element_role error: {e}")
        return "unknown"


def classify_focus(role: str, config: FocusConfig) -> FocusState:
    """Map a raw role string to a FocusState enum value."""
    for editable_role in config.editable_roles:
        if editable_role.lower() in role.lower():
            return FocusState.EDITABLE
    if role == "unknown":
        return FocusState.UNKNOWN
    return FocusState.NON_EDITABLE


def focus_state_generator(
    config: FocusConfig,
) -> Generator[FocusState, None, None]:
    """Poll OS focus at config.poll_interval_ms and yield FocusState changes."""
    interval = config.poll_interval_ms / 1000.0
    last_state = FocusState.UNKNOWN
    while True:
        try:
            role = get_focused_element_role()
            current = classify_focus(role, config)
            if current != last_state:
                logger.info(f"Focus changed: {last_state} → {current}")
                last_state = current
                yield current
            time.sleep(interval)
        except GeneratorExit:
            break
        except Exception as e:
            logger.warning(f"focus_state_generator error: {e}")
            time.sleep(interval)