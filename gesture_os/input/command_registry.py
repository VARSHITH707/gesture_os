import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "commands.json"


def load_registry(path: Path = _DEFAULT_PATH) -> dict:
    """Load the voice command registry from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        logger.info(f"Loaded {len(registry)} commands from {path}")
        return registry
    except Exception as e:
        logger.warning(f"Registry load failed: {e}")
        return {}