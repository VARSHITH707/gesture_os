"""GestureOS entry point."""
import argparse
import logging
import os
import sys

# Suppress MediaPipe / TensorFlow Lite C++ noise before any imports
os.environ.setdefault("GLOG_minloglevel", "3")       # suppress INFO/WARNING/ERROR from glog
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress TF C++ logs
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")  # avoid GPU init warnings

# Ensure the project root (parent of this package) is on sys.path when the
# script is run directly (e.g. python gesture_os/main.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gesture_os.app import GestureApp
from gesture_os.config import AppConfig
from gesture_os.startup import register_autostart


def main() -> None:
    parser = argparse.ArgumentParser(description="GestureOS")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show the debug overlay window (mode, FPS, pinch distance).",
    )
    args = parser.parse_args()

    cfg = AppConfig()
    if args.debug:
        cfg.show_debug_window = True

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Register autostart on first launch (safe; deduplicates)
    register_autostart(cfg.autostart_name)

    app = GestureApp(cfg)
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
        app.stop()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        app.stop()
        sys.exit(1)

    stats = app.get_stats()
    logger.info("Final stats: %s", stats)


if __name__ == "__main__":
    main()
