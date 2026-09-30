"""
Voice Session (M10): the hands-free
`wake word -> listen -> STT -> /chat -> TTS -> back to wake word` loop.

Deliberately talks to Jarvis Core only through `ChatClient`
(app/voice/chat_client.py), which only ever calls the server's HTTP
endpoints (app/main.py) - `VoiceSession` never imports `app.core.core`.
Voice is just another client of an already-running Rhasspy server, the
same way curl or a future UI would be; this milestone needed (and made)
zero changes to Core, ToolRegistry, or any Tool. As run by
`python -m app.voice.run` the ChatClient is the registered, authenticated
PC (Device Gateway, `/device/chat`); VoiceSession itself neither knows nor
handles any credential - it just passes the transcribed text through.

Raw microphone audio only ever exists as an in-memory `bytes` buffer
between `_capture_utterance` and `_transcribe` (via
SpeechToTextProvider.transcribe) - it is never written to disk, never
logged, and is discarded (falls out of scope) the moment transcription
returns. Only the *transcribed text* - already the exact same kind of
data a typed `/chat` request carries - ever leaves this module.

Failure isolation mirrors ReminderScheduler/AutomationEngine (M7/M8):
`run_one_cycle` raises the underlying typed error normally (so a direct
caller, e.g. a test, sees it); `run_forever` is what isolates a single
bad interaction so the process never crashes because of it.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from app.voice.base import (
    Microphone,
    SpeechToTextProvider,
    TextToSpeechError,
    TextToSpeechProvider,
    WakeWordProvider,
)
from app.voice.chat_client import ChatClient

#: Every piece of this subsystem (microphone, wake-word model, Vosk)
#: shares this single fixed sample rate - there is no resampling
#: anywhere in this milestone.
VOICE_SAMPLE_RATE = 16000


class VoiceState(Enum):
    """Where a VoiceSession currently is in its own loop - set
    synchronously at each phase transition, so a caller (or a test) can
    always read `session.state` and get an accurate answer."""

    WAKE_WORD_LISTENING = "wake_word_listening"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


def _status(message: str) -> None:
    print(f"[rhasspy] {message}")


class VoiceSession:
    """Runs the hands-free voice loop against injected providers - never
    constructs its own Microphone/WakeWordProvider/SpeechToTextProvider/
    TextToSpeechProvider/ChatClient (see app/voice/factory.py for the
    application's default wiring), so tests can exercise the full state
    machine with fakes and no real hardware/model/network (see
    tests/test_voice_session.py)."""

    def __init__(
        self,
        microphone: Microphone,
        wake_word_provider: WakeWordProvider,
        stt_provider: SpeechToTextProvider,
        tts_provider: TextToSpeechProvider,
        chat_client: ChatClient,
        silence_seconds: float = 1.2,
        max_utterance_seconds: float = 15.0,
        speech_amplitude_threshold: int = 500,
    ):
        self.microphone = microphone
        self.wake_word_provider = wake_word_provider
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.chat_client = chat_client
        self.silence_seconds = silence_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.speech_amplitude_threshold = speech_amplitude_threshold

        self.state = VoiceState.WAKE_WORD_LISTENING
        self.conversation_id: str | None = None
        self._stop_requested = False

    def stop(self) -> None:
        """Ask `run_forever` to exit after the current cycle finishes."""
        self._stop_requested = True

    def run_forever(self) -> None:
        """The application's real entry point (see app/voice/run.py):
        loops `run_one_cycle` until `stop()` is called or the process is
        interrupted (Ctrl+C), isolating any single failed interaction so
        the process itself never crashes because of it."""
        _status("wake-word listener active")
        try:
            while not self._stop_requested:
                try:
                    self.run_one_cycle()
                except Exception as exc:  # noqa: BLE001 - one bad interaction must not crash the process
                    print(f"[rhasspy] error: {exc}")
                    print("[rhasspy] returning to wake-word listening")
                    self.state = VoiceState.WAKE_WORD_LISTENING
        except KeyboardInterrupt:
            print()
        finally:
            self.tts_provider.stop()
            _status("stopped")

    def run_one_cycle(self) -> None:
        """Run exactly one `wake word -> ... -> back to wake word` cycle.
        Raises the underlying typed error (MicrophoneError/
        WakeWordError/SpeechToTextError/ChatClientError) on failure -
        `run_forever` is what isolates that per-cycle; calling this
        directly (as tests do) surfaces failures normally."""
        self._wait_for_wake_word()
        _status("wake word detected")

        audio = self._capture_utterance()

        self.state = VoiceState.PROCESSING
        _status("processing...")
        text = self.stt_provider.transcribe(audio, VOICE_SAMPLE_RATE).strip()
        if not text:
            print("[rhasspy] didn't catch anything - returning to wake-word listening")
            self.state = VoiceState.WAKE_WORD_LISTENING
            return

        reply, self.conversation_id = self.chat_client.send(text, self.conversation_id)

        self.state = VoiceState.SPEAKING
        _status("speaking...")
        try:
            self.tts_provider.speak(reply)
        except TextToSpeechError as exc:
            print(f"[rhasspy] speech synthesis failed ({exc}) - reply was: {reply}")

        _status("returning to wake-word listening")
        self.state = VoiceState.WAKE_WORD_LISTENING

    def _wait_for_wake_word(self) -> None:
        self.state = VoiceState.WAKE_WORD_LISTENING
        frame_length = self.wake_word_provider.frame_length
        while True:
            frame = self.microphone.read(frame_length)
            if self.wake_word_provider.process_frame(frame):
                return

    def _capture_utterance(self) -> bytes:
        """Record until `silence_seconds` of continuous quiet is seen -
        whether that's "the user never said anything after the wake
        word" or "the user spoke, then stopped" - or `max_utterance_seconds`
        elapses, whichever comes first. One unified rule rather than two
        separate cases: both are "stop capturing on sustained silence,"
        and STT correctly returns "" for the former either way (see
        `run_one_cycle`'s empty-transcription handling).

        A simple mean-absolute-amplitude threshold, not a dedicated VAD
        model - enough to segment one push-free utterance without adding
        another ML dependency on top of the wake-word/STT models this
        milestone already needs."""
        self.state = VoiceState.LISTENING
        _status("listening...")

        frame_length = self.wake_word_provider.frame_length
        frame_seconds = frame_length / VOICE_SAMPLE_RATE
        max_frames = max(1, round(self.max_utterance_seconds / frame_seconds))
        silence_frame_limit = max(1, round(self.silence_seconds / frame_seconds))

        frames: list[bytes] = []
        silence_run = 0

        for _ in range(max_frames):
            frame = self.microphone.read(frame_length)
            frames.append(frame)

            samples = np.frombuffer(frame, dtype=np.int16)
            amplitude = float(np.abs(samples).mean()) if samples.size else 0.0

            if amplitude >= self.speech_amplitude_threshold:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= silence_frame_limit:
                    break

        return b"".join(frames)
