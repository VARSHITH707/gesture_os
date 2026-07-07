"""Synthetic landmark feature-extraction tests.

These tests use landmark_factory to build realistic hand poses and verify
that extract_features() correctly classifies each pose.

No camera or physical hand required.
"""
import pytest
import numpy as np

from gesture_os.config import GestureConfig
from gesture_os.input.gesture_fsm import extract_features
from tests.landmark_factory import (
    open_palm,
    index_thumb_pinch,
    middle_thumb_pinch,
    three_finger_pinch,
    scroll_pose,
    closed_fist,
    idle_hand,
    verify_poses,
)


@pytest.fixture
def cfg() -> GestureConfig:
    return GestureConfig()


def feat(lm: np.ndarray, cfg: GestureConfig = None):
    if cfg is None:
        cfg = GestureConfig()
    return extract_features(lm, cfg, 0.0)


# ------------------------------------------------------------------
# Pose factory sanity
# ------------------------------------------------------------------

def test_pose_factory_valid():
    """All synthetic poses produce valid (21, 3) float32 arrays."""
    verify_poses()


# ------------------------------------------------------------------
# Open palm
# ------------------------------------------------------------------

def test_open_palm_detected(cfg):
    f = feat(open_palm(), cfg)
    assert f.is_open_palm, "open_palm() must be classified as open palm"


def test_open_palm_not_fist(cfg):
    f = feat(open_palm(), cfg)
    assert not f.is_fist


def test_open_palm_not_pinch(cfg):
    f = feat(open_palm(), cfg)
    assert not f.is_index_pinch
    assert not f.is_wispr_pinch


# ------------------------------------------------------------------
# Closed fist
# ------------------------------------------------------------------

def test_closed_fist_detected(cfg):
    f = feat(closed_fist(), cfg)
    assert f.is_fist, "closed_fist() must be classified as fist"


def test_fist_not_open_palm(cfg):
    f = feat(closed_fist(), cfg)
    assert not f.is_open_palm


def test_fist_not_pinch(cfg):
    """Fist suppresses pinch flags."""
    f = feat(closed_fist(), cfg)
    assert not f.is_index_pinch
    assert not f.is_wispr_pinch
    assert not f.is_three_finger_pinch


# ------------------------------------------------------------------
# Index-thumb pinch
# ------------------------------------------------------------------

def test_index_pinch_detected(cfg):
    lm = index_thumb_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert f.is_index_pinch, f"thumb_index_dist={f.thumb_index_dist:.4f}"


def test_index_pinch_not_wispr(cfg):
    """Index-thumb pinch must not trigger Wispr."""
    lm = index_thumb_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert not f.is_wispr_pinch


def test_no_pinch_when_apart(cfg):
    f = feat(idle_hand(), cfg)
    assert not f.is_index_pinch
    assert not f.is_wispr_pinch


def test_pinch_distance_small(cfg):
    """Strong pinch has small thumb_index_dist."""
    lm = index_thumb_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert f.thumb_index_dist < cfg.pinch_enter, f"dist={f.thumb_index_dist}"


def test_no_pinch_distance_large(cfg):
    f = feat(idle_hand(), cfg)
    assert f.thumb_index_dist > cfg.pinch_exit


# ------------------------------------------------------------------
# Middle-thumb (Wispr) pinch
# ------------------------------------------------------------------

def test_wispr_pinch_detected(cfg):
    lm = middle_thumb_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert f.is_wispr_pinch, f"thumb_middle_dist={f.thumb_middle_dist:.4f}"


def test_wispr_not_index_pinch(cfg):
    """Middle-thumb pinch must not trigger index pinch."""
    lm = middle_thumb_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert not f.is_index_pinch


# ------------------------------------------------------------------
# Three-finger pinch
# ------------------------------------------------------------------

def test_three_finger_pinch_detected(cfg):
    lm = three_finger_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert f.is_three_finger_pinch, (
        f"ti={f.thumb_index_dist:.4f}  tm={f.thumb_middle_dist:.4f}"
    )


def test_three_finger_not_wispr_only(cfg):
    """Three-finger pinch has both index and middle close."""
    lm = three_finger_pinch(strength=1.0)
    f = feat(lm, cfg)
    assert f.thumb_index_dist < cfg.three_finger_enter
    assert f.thumb_middle_dist < cfg.three_finger_enter


# ------------------------------------------------------------------
# Scroll pose
# ------------------------------------------------------------------

def test_scroll_pose_detected(cfg):
    lm = scroll_pose()
    f = feat(lm, cfg)
    assert f.is_scroll_pose, "scroll_pose() must be classified as scroll pose"


def test_scroll_not_open_palm(cfg):
    """Scroll pose has only 2 extended fingers — below open_palm threshold."""
    lm = scroll_pose()
    f = feat(lm, cfg)
    # open_palm requires 3+; scroll has exactly 2 extended
    assert not f.is_open_palm


def test_scroll_not_fist(cfg):
    f = feat(scroll_pose(), cfg)
    assert not f.is_fist


# ------------------------------------------------------------------
# Index tip coordinates
# ------------------------------------------------------------------

def test_index_tip_coordinates(cfg):
    """index_tip_x and index_tip_y match landmark 8."""
    from tests.landmark_factory import LM_INDEX_TIP
    lm = idle_hand(0.6, 0.4)
    f = feat(lm, cfg)
    assert abs(f.index_tip_x - float(lm[LM_INDEX_TIP, 0])) < 1e-5
    assert abs(f.index_tip_y - float(lm[LM_INDEX_TIP, 1])) < 1e-5


def test_scroll_ref_y_is_average(cfg):
    """scroll_ref_y is average of index tip y and middle tip y."""
    from tests.landmark_factory import LM_INDEX_TIP, LM_MIDDLE_TIP
    lm = scroll_pose(0.55)
    f = feat(lm, cfg)
    expected = (float(lm[LM_INDEX_TIP, 1]) + float(lm[LM_MIDDLE_TIP, 1])) * 0.5
    assert abs(f.scroll_ref_y - expected) < 1e-5
