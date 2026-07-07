import logging
import time
from dataclasses import dataclass
import numpy as np

from gesture_os.config import LandmarkConfig  # single authoritative definition

logger = logging.getLogger(__name__)


@dataclass
class HandLandmarks:
    """Stores extracted hand landmark data for one hand."""
    landmarks: np.ndarray        # shape (21, 3), float32
    handedness: str              # "Left" or "Right"
    detection_confidence: float
    presence_confidence: float


def create_hands_detector(
    config: LandmarkConfig,
) -> "HandLandmarker":
    """Instantiate and return a MediaPipe HandLandmarker detector in VIDEO mode."""
    try:
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.model_asset_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        detector = HandLandmarker.create_from_options(options)
        return detector
    except Exception as e:
        raise RuntimeError(f"Failed to create HandLandmarker: {e}") from e


def extract_landmarks(
    frame_bgr: np.ndarray,
    detector: "HandLandmarker",
    timestamp_ms: int | None = None,
) -> list[HandLandmarks]:
    """Extract hand landmarks from a BGR frame using VIDEO mode."""
    if frame_bgr is None or frame_bgr.size == 0:
        logger.warning("extract_landmarks received empty frame.")
        return []
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    try:
        from mediapipe import Image, ImageFormat
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect_for_video(mp_image, timestamp_ms)
        if not results.hand_landmarks:
            return []
        hands = []
        for i, hand_lm in enumerate(results.hand_landmarks):
            lm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm],
                dtype=np.float32,
            )
            # Flip x-coordinate because webcam is mirrored (selfie view)
            lm_array[:, 0] = 1.0 - lm_array[:, 0]
            label = "Right"
            score = 1.0
            if results.handedness and i < len(results.handedness):
                h = results.handedness[i]
                if h:
                    label = h[0].category_name if hasattr(h[0], 'category_name') else h[0].label
                    score = h[0].score if hasattr(h[0], 'score') else 1.0
            hands.append(HandLandmarks(
                landmarks=lm_array,
                handedness=label,
                detection_confidence=score,
                presence_confidence=1.0,
            ))
        return hands
    except Exception as e:
        logger.warning(f"extract_landmarks error: {e}")
        return []


def get_index_tip(hand: HandLandmarks) -> tuple[float, float]:
    """Return normalized (x, y) of index finger tip (landmark 8)."""
    return float(hand.landmarks[8, 0]), float(hand.landmarks[8, 1])


def get_thumb_tip(hand: HandLandmarks) -> tuple[float, float]:
    """Return normalized (x, y) of thumb tip (landmark 4)."""
    return float(hand.landmarks[4, 0]), float(hand.landmarks[4, 1])


def compute_pinch_ratio(hand: HandLandmarks) -> float:
    """Euclidean distance between thumb tip and index tip in 2D normalized space."""
    dx = hand.landmarks[4, 0] - hand.landmarks[8, 0]
    dy = hand.landmarks[4, 1] - hand.landmarks[8, 1]
    return float((dx**2 + dy**2) ** 0.5)


if __name__ == "__main__":
    import logging
    import time
    from gesture_os.vision.camera import CameraConfig, frame_generator
    logging.basicConfig(level=logging.INFO)
    detector = create_hands_detector(LandmarkConfig())
    try:
        for frame in frame_generator(CameraConfig()):
            timestamp_ms = int(time.time() * 1000)
            hands = extract_landmarks(frame, detector, timestamp_ms)
            if hands:
                print(f"conf={hands[0].detection_confidence:.2f} "
                      f"pinch={compute_pinch_ratio(hands[0]):.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()