import cv2
import time
import logging
from gesture_os.vision.camera import CameraConfig, frame_generator
from gesture_os.vision.landmarks import LandmarkConfig, create_hands_detector, extract_landmarks, compute_pinch_ratio, get_index_tip
from gesture_os.vision.filters import OneEuroState2D, filter_point_2d, apply_dead_zone
from gesture_os.input.gesture_fsm import GestureFSM, FSMConfig, GestureState, tick_fsm
from gesture_os.input.mouse import MouseController, build_monitor_layout
from gesture_os.input.swipe import SwipeConfig, SwipeState, detect_swipe
from gesture_os.input.macro_executor import execute_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_phase1(show_debug_window: bool = False):
    detector = create_hands_detector(LandmarkConfig())
    fsm = GestureFSM(FSMConfig())

    # Initialize mouse and load layout bounds
    mouse = MouseController()
    layout = build_monitor_layout()

    filter_state = OneEuroState2D()
    swipe_state = SwipeState()
    swipe_config = SwipeConfig()
    last_swipe_time: dict[str, float] = {}
    swipe_cooldown_s = 0.6
    prev_filtered_x = 0.5
    prev_filtered_y = 0.5

    logger.info("Starting Phase 1 Live Pipeline with Optimized Responsiveness...")
    try:
        for frame in frame_generator(CameraConfig()):
            now = time.perf_counter()
            timestamp_ms = int(now * 1000)
            hands = extract_landmarks(frame, detector, timestamp_ms)

            if hands:
                hand = hands[0]
                raw_x, raw_y = get_index_tip(hand)

                # EDGE GUARD: Reduced margin to preserve tracking at screen edges
                if 0.015 < raw_x < 0.985 and 0.015 < raw_y < 0.985:
                    smooth_x, smooth_y = filter_point_2d(filter_state, (raw_x, raw_y), now)

                    # Apply dead zone in NORMALIZED space before coordinate mapping (reduces latency)
                    smooth_x, smooth_y = apply_dead_zone(
                        (smooth_x, smooth_y),
                        (prev_filtered_x, prev_filtered_y),
                        0.0015  # ~2-3px at 1920px width
                    )
                    prev_filtered_x, prev_filtered_y = smooth_x, smooth_y

                    # Convert normalized 0.0-1.0 decimals to absolute screen pixel integers
                    pixel_x = int(layout.origin_x + (smooth_x * layout.total_width))
                    pixel_y = int(layout.origin_y + (smooth_y * layout.total_height))

                    pinch_ratio = compute_pinch_ratio(hand)
                    out = tick_fsm(fsm, pinch_ratio, now)

                    # Map FSM states to mouse actions
                    if out.state in (GestureState.IDLE, GestureState.PINCH_PENDING):
                        mouse.move_to(pixel_x, pixel_y)
                    elif out.state == GestureState.CLICK_HELD:
                        mouse.move_to(pixel_x, pixel_y)
                    elif out.state == GestureState.DRAG_ACTIVE:
                        mouse.move_to(pixel_x, pixel_y)

                    # Handle click/drag events from FSM output
                    if out.emit_mouse_press:
                        mouse.press("left")
                    if out.emit_mouse_release:
                        mouse.release("left")
                    if out.emit_left_click:
                        mouse.click("left")
                    if out.emit_double_click:
                        mouse.double_click("left")

                    # Swipe detection for tab switching and window management (only when not pinching)
                    # Open palm = at least 3 of 4 fingers extended (tip.y < pip.y)
                    landmarks = hand.landmarks
                    tips = [8, 12, 16, 20]
                    pips = [6, 10, 14, 18]
                    extended = sum(1 for t, p in zip(tips, pips) if landmarks[t, 1] < landmarks[p, 1])
                    is_open_palm = extended >= 3

                    if is_open_palm:
                        swipe_dir = detect_swipe(swipe_state, pixel_x, pixel_y, now, swipe_config)
                        if swipe_dir:
                            last_trigger = last_swipe_time.get(swipe_dir, 0.0)
                            if (now - last_trigger) > swipe_cooldown_s:
                                if swipe_dir == "SWIPE_RIGHT":
                                    execute_command({"action": "hotkey", "keys": ["ctrl", "tab"]})
                                    logger.info("Swipe RIGHT → Next Tab (Ctrl+Tab)")
                                elif swipe_dir == "SWIPE_LEFT":
                                    execute_command({"action": "hotkey", "keys": ["ctrl", "shift", "tab"]})
                                    logger.info("Swipe LEFT → Previous Tab (Ctrl+Shift+Tab)")
                                elif swipe_dir == "SWIPE_DOWN":
                                    execute_command({"action": "hotkey", "keys": ["super", "m"]})
                                    logger.info("Swipe DOWN → Minimize All (Win+M)")
                                elif swipe_dir == "SWIPE_UP":
                                    execute_command({"action": "hotkey", "keys": ["super", "shift", "m"]})
                                    logger.info("Swipe UP → Restore All (Win+Shift+M)")
                                last_swipe_time[swipe_dir] = now
                else:
                    logger.debug("Tracking near edge, skipping frame")

            if show_debug_window:
                cv2.imshow("Gesture OS - Phase 1 Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_phase1(show_debug_window=True)