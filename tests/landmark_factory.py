"""Synthetic landmark array factory for gesture tests.

Builds realistic (21, 3) float32 numpy arrays that mimic specific hand poses,
allowing full feature-extraction and FSM tests without a camera.

Coordinate system: normalized 0-1, y increases downward (0=top, 1=bottom).
Hand is assumed to be held upright with fingers pointing up:
  - Fingertips have small y values (near top of image)
  - Palm/wrist have large y values (near bottom)
  - Finger extended: tip.y < pip.y
  - Finger curled:   tip.y > pip.y
"""
import numpy as np

# Landmark index constants (same as gesture_fsm.py)
LM_WRIST      = 0
LM_THUMB_CMC  = 1
LM_THUMB_MCP  = 2
LM_THUMB_IP   = 3
LM_THUMB_TIP  = 4
LM_INDEX_MCP  = 5
LM_INDEX_PIP  = 6
LM_INDEX_DIP  = 7
LM_INDEX_TIP  = 8
LM_MIDDLE_MCP = 9
LM_MIDDLE_PIP = 10
LM_MIDDLE_DIP = 11
LM_MIDDLE_TIP = 12
LM_RING_MCP   = 13
LM_RING_PIP   = 14
LM_RING_DIP   = 15
LM_RING_TIP   = 16
LM_PINKY_MCP  = 17
LM_PINKY_PIP  = 18
LM_PINKY_DIP  = 19
LM_PINKY_TIP  = 20


def _base() -> np.ndarray:
    """Build a neutral hand skeleton in normalized image coordinates."""
    lm = np.zeros((21, 3), dtype=np.float32)

    # Wrist
    lm[LM_WRIST] = [0.50, 0.90, 0.0]

    # Thumb (roughly extended to the side)
    lm[LM_THUMB_CMC] = [0.45, 0.82, 0.0]
    lm[LM_THUMB_MCP] = [0.38, 0.76, 0.0]
    lm[LM_THUMB_IP]  = [0.33, 0.70, 0.0]
    lm[LM_THUMB_TIP] = [0.30, 0.65, 0.0]

    # Index (extended upward)
    lm[LM_INDEX_MCP] = [0.45, 0.72, 0.0]
    lm[LM_INDEX_PIP] = [0.44, 0.60, 0.0]
    lm[LM_INDEX_DIP] = [0.43, 0.50, 0.0]
    lm[LM_INDEX_TIP] = [0.43, 0.42, 0.0]

    # Middle (extended)
    lm[LM_MIDDLE_MCP] = [0.50, 0.71, 0.0]
    lm[LM_MIDDLE_PIP] = [0.50, 0.59, 0.0]
    lm[LM_MIDDLE_DIP] = [0.50, 0.49, 0.0]
    lm[LM_MIDDLE_TIP] = [0.50, 0.41, 0.0]

    # Ring (extended)
    lm[LM_RING_MCP] = [0.55, 0.72, 0.0]
    lm[LM_RING_PIP] = [0.55, 0.60, 0.0]
    lm[LM_RING_DIP] = [0.55, 0.50, 0.0]
    lm[LM_RING_TIP] = [0.55, 0.42, 0.0]

    # Pinky (extended)
    lm[LM_PINKY_MCP] = [0.60, 0.74, 0.0]
    lm[LM_PINKY_PIP] = [0.59, 0.62, 0.0]
    lm[LM_PINKY_DIP] = [0.59, 0.53, 0.0]
    lm[LM_PINKY_TIP] = [0.59, 0.46, 0.0]

    return lm


def _curl_finger(lm: np.ndarray, pip: int, dip: int, tip: int) -> np.ndarray:
    """Curl one finger so its tip is below its PIP joint."""
    pip_y = float(lm[pip, 1])
    lm[dip, 1] = pip_y + 0.04
    lm[tip, 1] = pip_y + 0.08
    return lm


# ------------------------------------------------------------------
# Public pose constructors — each returns a (21, 3) float32 array
# ------------------------------------------------------------------

def open_palm(index_x: float = 0.5, index_y: float = 0.5) -> np.ndarray:
    """All 4 fingers extended. Thumb separated. Optionally shift index tip."""
    lm = _base().copy()
    dx = index_x - float(lm[LM_INDEX_TIP, 0])
    dy = index_y - float(lm[LM_INDEX_TIP, 1])
    lm += np.array([dx, dy, 0.0], dtype=np.float32)  # shift whole hand
    return lm


def index_thumb_pinch(strength: float = 1.0) -> np.ndarray:
    """Thumb tip moved close to index tip. Middle/ring/pinky extended."""
    lm = _base().copy()
    ix, iy = float(lm[LM_INDEX_TIP, 0]), float(lm[LM_INDEX_TIP, 1])
    tx, ty = float(lm[LM_THUMB_TIP, 0]), float(lm[LM_THUMB_TIP, 1])
    # Move thumb tip toward index tip by strength (0=apart, 1=touching)
    lm[LM_THUMB_TIP, 0] = tx + (ix - tx) * strength
    lm[LM_THUMB_TIP, 1] = ty + (iy - ty) * strength
    return lm


def middle_thumb_pinch(strength: float = 1.0) -> np.ndarray:
    """Thumb tip close to middle tip. Index extended and NOT pinching.

    Index tip is placed at (0.65, 0.36) so it's ~0.16 normalized units from the
    thumb tip (well above pinch_enter=0.06 and three_finger_enter=0.07).
    """
    lm = _base().copy()
    mx, my = float(lm[LM_MIDDLE_TIP, 0]), float(lm[LM_MIDDLE_TIP, 1])
    tx, ty = float(lm[LM_THUMB_TIP, 0]), float(lm[LM_THUMB_TIP, 1])
    lm[LM_THUMB_TIP, 0] = tx + (mx - tx) * strength
    lm[LM_THUMB_TIP, 1] = ty + (my - ty) * strength
    # Keep index far from thumb so ti_dist >> pinch_enter, three_finger_pinch stays False
    lm[LM_INDEX_TIP, 0] = 0.65
    lm[LM_INDEX_TIP, 1] = 0.36
    return lm


def three_finger_pinch(strength: float = 1.0) -> np.ndarray:
    """Both index AND middle tips close to thumb. Ring/pinky extended."""
    lm = index_thumb_pinch(strength)
    mx, my = float(_base()[LM_MIDDLE_TIP, 0]), float(_base()[LM_MIDDLE_TIP, 1])
    tx, ty = float(lm[LM_THUMB_TIP, 0]), float(lm[LM_THUMB_TIP, 1])
    lm[LM_MIDDLE_TIP, 0] = tx + (mx - tx) * (1 - strength)
    lm[LM_MIDDLE_TIP, 1] = ty + (my - ty) * (1 - strength)
    return lm


def scroll_pose(ref_y: float = 0.5) -> np.ndarray:
    """Index + middle extended, ring + pinky curled. Suitable for scroll."""
    lm = _base().copy()
    # Curl ring and pinky
    _curl_finger(lm, LM_RING_PIP,  LM_RING_DIP,  LM_RING_TIP)
    _curl_finger(lm, LM_PINKY_PIP, LM_PINKY_DIP, LM_PINKY_TIP)
    # Shift so average of index+middle tip y is ref_y
    avg_y = (float(lm[LM_INDEX_TIP, 1]) + float(lm[LM_MIDDLE_TIP, 1])) * 0.5
    dy = ref_y - avg_y
    lm[:, 1] += dy
    return lm


def closed_fist() -> np.ndarray:
    """All 4 fingers curled (tips below PIPs). Thumb tucked."""
    lm = _base().copy()
    _curl_finger(lm, LM_INDEX_PIP,  LM_INDEX_DIP,  LM_INDEX_TIP)
    _curl_finger(lm, LM_MIDDLE_PIP, LM_MIDDLE_DIP, LM_MIDDLE_TIP)
    _curl_finger(lm, LM_RING_PIP,   LM_RING_DIP,   LM_RING_TIP)
    _curl_finger(lm, LM_PINKY_PIP,  LM_PINKY_DIP,  LM_PINKY_TIP)
    return lm


def idle_hand(index_x: float = 0.5, index_y: float = 0.5) -> np.ndarray:
    """Open hand with no pinch — index tip at specified position.

    All 4 fingers extended (is_open_palm=True). Use pointing_hand() for
    cases that need a non-palm neutral pose (e.g. after a swipe).
    """
    lm = open_palm(index_x, index_y)
    # Ensure thumb is well separated from index
    lm[LM_THUMB_TIP, 0] = float(lm[LM_INDEX_TIP, 0]) - 0.12
    lm[LM_THUMB_TIP, 1] = float(lm[LM_INDEX_TIP, 1]) + 0.08
    return lm


def pointing_hand(index_x: float = 0.5, index_y: float = 0.5) -> np.ndarray:
    """Index finger extended, all others curled. Thumb well separated.

    n_ext=1 → is_open_palm=False, is_fist=False, is_scroll_pose=False.
    In IDLE the FSM stays in IDLE (cursor mode). Useful as a neutral pose
    after gestures that require open_palm to exit (e.g. after swipe).
    """
    lm = _base().copy()
    # Curl middle, ring, pinky
    _curl_finger(lm, LM_MIDDLE_PIP, LM_MIDDLE_DIP, LM_MIDDLE_TIP)
    _curl_finger(lm, LM_RING_PIP,   LM_RING_DIP,   LM_RING_TIP)
    _curl_finger(lm, LM_PINKY_PIP,  LM_PINKY_DIP,  LM_PINKY_TIP)
    # Shift so index tip is at requested position
    dx = index_x - float(lm[LM_INDEX_TIP, 0])
    dy = index_y - float(lm[LM_INDEX_TIP, 1])
    lm += np.array([dx, dy, 0.0], dtype=np.float32)
    # Thumb well separated
    lm[LM_THUMB_TIP, 0] = float(lm[LM_INDEX_TIP, 0]) - 0.15
    lm[LM_THUMB_TIP, 1] = float(lm[LM_INDEX_TIP, 1]) + 0.10
    return lm


def verify_poses() -> None:
    """Quick sanity-check that all factory poses produce valid landmarks."""
    for name, lm in [
        ("open_palm",          open_palm()),
        ("index_thumb_pinch",  index_thumb_pinch()),
        ("middle_thumb_pinch", middle_thumb_pinch()),
        ("three_finger_pinch", three_finger_pinch()),
        ("scroll_pose",        scroll_pose()),
        ("closed_fist",        closed_fist()),
        ("idle_hand",          idle_hand()),
    ]:
        assert lm.shape == (21, 3), f"{name}: wrong shape {lm.shape}"
        assert lm.dtype == np.float32, f"{name}: wrong dtype"
        assert np.all(np.isfinite(lm)), f"{name}: non-finite values"


if __name__ == "__main__":
    verify_poses()
    print("All landmark poses verified.")
