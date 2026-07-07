import time
import logging
import threading
from gesture_os.audio.capture import AudioConfig, chunk_generator
from gesture_os.audio.vad import VADConfig, load_vad_model, speech_chunk_generator
from gesture_os.audio.asr import ASRConfig, load_asr_model, transcript_generator
from gesture_os.input.command_registry import load_registry
from gesture_os.input.macro_executor import match_intent, execute_command
from gesture_os.phase1_runner import run_phase1

logger = logging.getLogger(__name__)


def run_voice_macro_loop() -> None:
    """Background thread: mic → VAD → ASR → intent → execute."""
    while True:
        try:
            vad_model, _ = load_vad_model()
            _, recognizer = load_asr_model(ASRConfig())
            registry = load_registry()
            audio_cfg = AudioConfig()
            vad_cfg = VADConfig()
            raw = chunk_generator(audio_cfg)
            speech = speech_chunk_generator(raw, vad_model, vad_cfg)
            for transcript in transcript_generator(speech, recognizer):
                logger.info(f"Heard: {transcript}")
                cmd = match_intent(transcript, registry)
                if cmd:
                    execute_command(cmd)
        except Exception as e:
            logger.error(f"Voice loop crashed: {e}. Restarting in 3s...")
            time.sleep(3.0)


def run_phase2(show_debug_window: bool = False) -> None:
    """Run Phase 1 gesture control + Phase 2 voice macros concurrently."""
    voice_thread = threading.Thread(
        target=run_voice_macro_loop,
        daemon=True,
        name="VoiceMacroThread",
    )
    voice_thread.start()
    logger.info("Voice macro thread started.")
    run_phase1(show_debug_window=show_debug_window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_phase2(show_debug_window=True)