from dataclasses import dataclass
import json
import logging
from typing import Generator
from vosk import KaldiRecognizer, Model

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """Configuration for Vosk ASR engine."""
    model_path: str = "models/vosk-model-small-en-us"
    sample_rate: int = 16000


def load_asr_model(config: ASRConfig) -> tuple[Model, KaldiRecognizer]:
    """Load Vosk model and recognizer from local path."""
    try:
        model = Model(config.model_path)
        recognizer = KaldiRecognizer(model, config.sample_rate)
        recognizer.SetWords(False)
        logger.info(f"Vosk model loaded from {config.model_path}")
        return model, recognizer
    except Exception as e:
        raise RuntimeError(f"ASR model load failed: {e}") from e


def transcribe_chunk(
    chunk: bytes,
    recognizer: KaldiRecognizer,
) -> str | None:
    """Feed one chunk to Vosk. Return final transcript or None."""
    try:
        if recognizer.AcceptWaveform(chunk):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            return text if text else None
        return None
    except Exception as e:
        logger.warning(f"transcribe_chunk error: {e}")
        return None


def transcript_generator(
    speech_gen: Generator[bytes, None, None],
    recognizer: KaldiRecognizer,
) -> Generator[str, None, None]:
    """Yield non-empty final transcripts from a speech chunk stream."""
    for chunk in speech_gen:
        result = transcribe_chunk(chunk, recognizer)
        if result:
            yield result