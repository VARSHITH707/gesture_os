"""TEST 53–55: Windows startup integration tests."""
import pytest
from gesture_os.startup import (
    get_autostart_command,
    register_autostart,
    unregister_autostart,
    verify_autostart,
    _build_launch_command,
)

TEST_NAME = "GestureOS_Test_Entry"


@pytest.fixture(autouse=True)
def cleanup():
    """Remove test registry entry before and after each test."""
    unregister_autostart(TEST_NAME)
    yield
    unregister_autostart(TEST_NAME)


def test_register_creates_entry():
    """TEST 53 — autostart entry is created."""
    result = register_autostart(TEST_NAME)
    assert result, "register_autostart returned False"
    cmd = get_autostart_command(TEST_NAME)
    assert cmd is not None, "Registry entry not found after registration"


def test_startup_target_correct():
    """TEST 54 — startup target points to the correct executable."""
    register_autostart(TEST_NAME)
    v = verify_autostart(TEST_NAME)
    assert v["registered"]
    assert v["match"], f"Command mismatch:\n  got: {v['command']}\n  expected: {v['expected']}"


def test_no_duplicate_entries():
    """TEST 55 — registering twice does not create duplicate entries."""
    register_autostart(TEST_NAME)
    cmd1 = get_autostart_command(TEST_NAME)
    register_autostart(TEST_NAME)
    cmd2 = get_autostart_command(TEST_NAME)
    assert cmd1 == cmd2, "Command changed on second registration"


def test_unregister_removes_entry():
    """unregister_autostart cleanly removes the entry."""
    register_autostart(TEST_NAME)
    assert get_autostart_command(TEST_NAME) is not None
    unregister_autostart(TEST_NAME)
    assert get_autostart_command(TEST_NAME) is None


def test_build_launch_command_contains_python():
    """Launch command references a Python executable."""
    cmd = _build_launch_command()
    assert "python" in cmd.lower() or ".exe" in cmd.lower()
    assert "main.py" in cmd
