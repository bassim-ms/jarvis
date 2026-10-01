"""
Speech-to-text backed by Vosk (Apache-2.0, fully offline, no account/API
key) - a lightweight local recognizer well suited to short, already-
segmented utterances rather than continuous streaming transcription. The
user downloads a language model once (e.g. `vosk-model-small-en-us-0.15`
from https://alphacephei.com/vosk/models) and points
JARVIS_VOICE_VOSK_MODEL_PATH at the extracted folder - never downloaded
automatically by this module.
"""

import json
import os

import vosk

from app.voice.base import SpeechToTextError, SpeechToTextProvider

# Vosk logs its own diagnostics straight to stderr by default (including
# on every recognizer call) - silenced so app/voice's own [jarvis] status
# lines stay the only console output.
vosk.SetLogLevel(-1)


class VoskSTTProvider(SpeechToTextProvider):
    """SpeechToTextProvider implementation backed by a local Vosk model."""

    def __init__(self, model_path: str):
        if not model_path:
            raise SpeechToTextError(
                "No Vosk model configured - set JARVIS_VOICE_VOSK_MODEL_PATH "
                "(see .env.example)."
            )
        if not os.path.isdir(model_path):
            raise SpeechToTextError(
                f"Vosk model not found at '{model_path}' - download one from "
                "https://alphacephei.com/vosk/models and extract it there."
            )
        try:
            self._model = vosk.Model(model_path)
        except Exception as exc:  # noqa: BLE001 - vosk raises a plain Exception
            raise SpeechToTextError(
                f"Failed to load Vosk model at '{model_path}': {exc}"
            ) from exc

    def transcribe(self, audio: bytes, sample_rate: int) -> str:
        try:
            recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
            recognizer.AcceptWaveform(audio)
            result = json.loads(recognizer.FinalResult())
        except Exception as exc:  # noqa: BLE001 - isolate a single bad utterance
            raise SpeechToTextError(f"Vosk transcription failed: {exc}") from exc

        return result.get("text", "").strip()
