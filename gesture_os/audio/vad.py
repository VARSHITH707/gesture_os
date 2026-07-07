from dataclasses import dataclass
import logging
import numpy as np
import torch
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Configuration for Silero VAD gate."""
    sample_rate: int = 16000
    threshold: float = 0.5
    min_speech_ms: int = 100


def load_vad_model() -> tuple:
    """Load Silero VAD model and utils from torch.hub."""
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        logger.info("Silero VAD loaded successfully.")
        return model, utils
    except Exception as e:
        raise RuntimeError(f"VAD model load failed: {e}") from e


def is_speech(
    chunk: bytes,
    model,
    config: VADConfig,
) -> bool:
    """Return True if the audio chunk contains speech."""
    try:
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        tensor = torch.tensor(audio)
        prob = model(tensor, config.sample_rate).item()
        return prob >= config.threshold
    except Exception as e:
        logger.warning(f"is_speech error: {e}")
        return False


def speech_chunk_generator(
    raw_gen: Generator[bytes, None, None],
    model,
    config: VADConfig,
) -> Generator[bytes, None, None]:
    """Yield only chunks classified as speech by the VAD model."""
    for chunk in raw_gen:
        if is_speech(chunk, model, config):
            yield chunk