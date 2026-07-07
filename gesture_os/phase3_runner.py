import logging
import threading
from gesture_os.accessibility.wispr_gate import focus_routing_loop, WisprConfig
from gesture_os.accessibility.focus_monitor import FocusConfig
from gesture_os.phase2_runner import run_phase2

logger = logging.getLogger(__name__)


def run_phase3(show_debug_window: bool = False) -> None:
    """Run Phase 1 + Phase 2 + Phase 3 accessibility layer concurrently."""
    focus_thread = threading.Thread(
        target=focus_routing_loop,
        kwargs={
            "wispr_config": WisprConfig(),
            "focus_config": FocusConfig(),
        },
        daemon=True,
        name="FocusRoutingThread",
    )
    focus_thread.start()
    logger.info("Focus routing thread started.")
    run_phase2(show_debug_window=show_debug_window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase3(show_debug_window=True)