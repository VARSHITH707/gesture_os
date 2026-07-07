"""TEST 1: All modules import without error."""
import importlib
import pytest

MODULES = [
    "gesture_os.config",
    "gesture_os.app",
    "gesture_os.tray",
    "gesture_os.startup",
    "gesture_os.vision.camera",
    "gesture_os.vision.filters",
    "gesture_os.vision.landmarks",
    "gesture_os.input.mouse",
    "gesture_os.input.keyboard",
    "gesture_os.input.cursor",
    "gesture_os.input.gesture_fsm",
    "gesture_os.main",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    """TEST 1 — every required module imports cleanly."""
    mod = importlib.import_module(module)
    assert mod is not None
