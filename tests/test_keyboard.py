"""TEST 32, 51–52: Keyboard safety and key release tests.

These tests verify the module imports and the safety logic, but cannot
verify actual OS key injection without a real UI session. Integration
evidence for actual shortcut firing is marked as HUMAN_VALIDATION_REQUIRED.
"""
import pytest
from gesture_os.input.keyboard import (
    release_all_modifiers,
    send_ctrl_alt,
    send_win_m,
    send_win_shift_m,
    send_alt_tab_forward,
    send_alt_tab_backward,
)


def test_release_all_modifiers_no_exception():
    """TEST 52 — release_all_modifiers must not raise even if no key is held."""
    release_all_modifiers()   # must not raise


def test_send_ctrl_alt_no_exception():
    """TEST 31, 32 — send_ctrl_alt executes without raising."""
    send_ctrl_alt()           # AUTOMATED: no exception. INTEGRATION: see manual test.


def test_send_win_m_no_exception():
    """TEST 26 keyboard side — Win+M executes without raising."""
    send_win_m()


def test_send_win_shift_m_no_exception():
    """TEST 27 keyboard side — Win+Shift+M executes without raising."""
    send_win_shift_m()


def test_send_alt_tab_forward_no_exception():
    """TEST 24 keyboard side — Ctrl+Tab executes without raising."""
    send_alt_tab_forward()


def test_send_alt_tab_backward_no_exception():
    """TEST 25 keyboard side — Ctrl+Shift+Tab executes without raising."""
    send_alt_tab_backward()
