import logging
import pyaudio
from dataclasses import dataclass
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for microphone audio capture."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    device_index: int | None = None


def open_mic(config: AudioConfig) -> pyaudio.PyAudio:
    """Open a PyAudio stream for microphone capture."""
    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=config.sample_rate,
            channels=config.channels,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=config.device_index,
            frames_per_buffer=config.chunk_size,
        )
        pa._stream = stream
        return pa
    except Exception as e:
        raise RuntimeError(f"Mic open failed: {e}") from e


def read_chunk(pa: pyaudio.PyAudio, config: AudioConfig) -> bytes | None:
    """Read one audio chunk from the open mic stream."""
    try:
        return pa._stream.read(
            config.chunk_size, exception_on_overflow=False
        )
    except Exception as e:
        logger.warning(f"read_chunk error: {e}")
        return None


def release_mic(pa: pyaudio.PyAudio) -> None:
    """Stop and close the PyAudio mic stream safely."""
    try:
        pa._stream.stop_stream()
        pa._stream.close()
        pa.terminate()
    except Exception as e:
        logger.warning(f"release_mic error: {e}")


def chunk_generator(
    config: AudioConfig,
) -> Generator[bytes, None, None]:
    """Yield raw PCM audio chunks from the microphone indefinitely."""
    pa = open_mic(config)
    try:
        while True:
            chunk = read_chunk(pa, config)
            if chunk is not None:
                yield chunk
    except GeneratorExit:
        pass
    finally:
        release_mic(pa)