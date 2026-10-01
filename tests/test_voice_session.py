"""
Tests for VoiceSession (app/voice/session.py): the hands-free
wake-word -> listen -> STT -> /chat -> TTS -> back-to-wake-word state
machine. Every dependency (microphone, wake-word/STT/TTS providers, chat
client) is a small fake defined here - no real audio hardware, model, or
network call is ever involved, matching this project's testing
convention.
"""

import numpy as np
import pytest

from app.voice.base import (
    Microphone,
    MicrophoneError,
    SpeechToTextProvider,
    TextToSpeechError,
    TextToSpeechProvider,
    WakeWordProvider,
)
from app.voice.chat_client import ChatClientError
from app.voice.session import VoiceSession, VoiceState

FRAME_LENGTH = 1280


def _frame(value: int) -> bytes:
    return np.full(FRAME_LENGTH, value, dtype=np.int16).tobytes()


WAKE_FRAME = _frame(1)
LOUD_FRAME = _frame(1000)
QUIET_FRAME = _frame(0)


class FakeMicrophone(Microphone):
    def __init__(self, frames: list[bytes]):
        self.sample_rate = 16000
        self._frames = list(frames)
        self.on_read = None
        self.read_count = 0
        self.closed = False

    def read(self, num_frames: int) -> bytes:
        self.read_count += 1
        if self.on_read:
            self.on_read()
        if not self._frames:
            raise MicrophoneError("FakeMicrophone ran out of frames")
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeWakeWordProvider(WakeWordProvider):
    frame_length = FRAME_LENGTH

    def __init__(self, detect_on_frame: bytes = WAKE_FRAME, error: Exception | None = None):
        self.detect_on_frame = detect_on_frame
        self.error = error

    def process_frame(self, frame: bytes) -> bool:
        if self.error:
            raise self.error
        return frame == self.detect_on_frame


class FakeSTTProvider(SpeechToTextProvider):
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error
        self.received_audio: bytes | None = None

    def transcribe(self, audio: bytes, sample_rate: int) -> str:
        self.received_audio = audio
        if self.error:
            raise self.error
        return self.text


class FakeTTSProvider(TextToSpeechProvider):
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.spoken: list[str] = []
        self.stopped = False

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        if self.error:
            raise self.error

    def stop(self) -> None:
        self.stopped = True


class FakeChatClient:
    def __init__(
        self, reply: str = "ok", conversation_id: str = "conv-1", error: Exception | None = None
    ):
        self.reply = reply
        self.conversation_id = conversation_id
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def send(self, message: str, conversation_id: str | None) -> tuple[str, str]:
        self.calls.append((message, conversation_id))
        if self.error:
            raise self.error
        return self.reply, self.conversation_id


def make_session(
    frames: list[bytes],
    wake_word_provider: WakeWordProvider | None = None,
    stt_provider: SpeechToTextProvider | None = None,
    tts_provider: TextToSpeechProvider | None = None,
    chat_client=None,
    silence_seconds: float = 0.16,  # 2 frames at 1280/16000s each
    max_utterance_seconds: float = 10.0,
) -> tuple[VoiceSession, FakeMicrophone]:
    mic = FakeMicrophone(frames)
    session = VoiceSession(
        microphone=mic,
        wake_word_provider=wake_word_provider or FakeWakeWordProvider(),
        stt_provider=stt_provider or FakeSTTProvider(text="hello jarvis"),
        tts_provider=tts_provider or FakeTTSProvider(),
        chat_client=chat_client or FakeChatClient(),
        silence_seconds=silence_seconds,
        max_utterance_seconds=max_utterance_seconds,
        speech_amplitude_threshold=500,
    )
    return session, mic


# --- full happy-path cycle -------------------------------------------------


def test_run_one_cycle_full_happy_path():
    frames = [WAKE_FRAME, LOUD_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME]
    chat = FakeChatClient(reply="Nothing today.", conversation_id="conv-1")
    stt = FakeSTTProvider(text="what's on my calendar")
    tts = FakeTTSProvider()
    session, mic = make_session(frames, stt_provider=stt, tts_provider=tts, chat_client=chat)

    session.run_one_cycle()

    assert chat.calls == [("what's on my calendar", None)]
    assert tts.spoken == ["Nothing today."]
    assert session.conversation_id == "conv-1"
    assert session.state == VoiceState.WAKE_WORD_LISTENING
    assert mic._frames == []  # exactly the frames needed were consumed


def test_conversation_id_persists_across_cycles():
    frames = [WAKE_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME] * 2
    chat = FakeChatClient(reply="hi", conversation_id="conv-99")
    session, _ = make_session(frames, chat_client=chat, stt_provider=FakeSTTProvider(text="hello"))

    session.run_one_cycle()
    assert chat.calls[0] == ("hello", None)
    assert session.conversation_id == "conv-99"

    session.run_one_cycle()
    assert chat.calls[1] == ("hello", "conv-99")


# --- state transitions ---------------------------------------------------


def test_states_transition_through_wake_word_then_listening_then_back():
    frames = [WAKE_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME]
    recorded: list[VoiceState] = []
    session, mic = make_session(frames)
    mic.on_read = lambda: recorded.append(session.state)

    session.run_one_cycle()

    # The wake-word frame is read while still WAKE_WORD_LISTENING.
    assert recorded[0] == VoiceState.WAKE_WORD_LISTENING
    # Every subsequent read (utterance capture) happens while LISTENING.
    assert all(s == VoiceState.LISTENING for s in recorded[1:])
    # Back to wake-word listening once the cycle completes.
    assert session.state == VoiceState.WAKE_WORD_LISTENING


def test_state_is_speaking_while_tts_provider_is_invoked():
    seen_state = {}

    class RecordingTTS(FakeTTSProvider):
        def speak(self, text):
            seen_state["state"] = session.state
            super().speak(text)

    frames = [WAKE_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME]
    session, _ = make_session(frames, tts_provider=RecordingTTS())

    session.run_one_cycle()

    assert seen_state["state"] == VoiceState.SPEAKING


# --- silence / empty speech ------------------------------------------------


def test_pure_silence_ends_capture_without_waiting_for_speech():
    """No speech at all (never crosses the amplitude threshold) still
    ends the utterance after `silence_seconds`, not the full
    `max_utterance_seconds` - the same rule as ending capture after
    speech, not a separate timeout."""
    frames = [WAKE_FRAME, QUIET_FRAME, QUIET_FRAME]
    session, mic = make_session(frames)

    session.run_one_cycle()

    assert mic._frames == []  # only the 2 silence frames were consumed
    assert mic.read_count == 3  # 1 wake-word + 2 silence


def test_empty_transcription_skips_chat_and_tts():
    frames = [WAKE_FRAME, QUIET_FRAME, QUIET_FRAME]
    chat = FakeChatClient()
    tts = FakeTTSProvider()
    session, _ = make_session(frames, stt_provider=FakeSTTProvider(text=""), chat_client=chat, tts_provider=tts)

    session.run_one_cycle()

    assert chat.calls == []
    assert tts.spoken == []
    assert session.state == VoiceState.WAKE_WORD_LISTENING


def test_whitespace_only_transcription_is_treated_as_empty():
    frames = [WAKE_FRAME, QUIET_FRAME, QUIET_FRAME]
    chat = FakeChatClient()
    session, _ = make_session(frames, stt_provider=FakeSTTProvider(text="   "), chat_client=chat)

    session.run_one_cycle()

    assert chat.calls == []


def test_capture_stops_at_max_utterance_seconds_even_without_silence():
    frame_seconds = FRAME_LENGTH / 16000
    max_capture_frames = 3
    # Far more loud frames available than the cap should ever consume.
    frames = [WAKE_FRAME] + [LOUD_FRAME] * (max_capture_frames + 10)
    session, mic = make_session(
        frames,
        silence_seconds=100.0,  # never triggers - continuously loud
        max_utterance_seconds=max_capture_frames * frame_seconds,
    )

    session.run_one_cycle()

    assert mic.read_count == 1 + max_capture_frames  # wake frame + capped capture


# --- failure recovery -------------------------------------------------


def test_tts_failure_falls_back_to_printing_the_reply(capsys):
    frames = [WAKE_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME]
    chat = FakeChatClient(reply="the reply text", conversation_id="c1")
    tts = FakeTTSProvider(error=TextToSpeechError("synth broke"))
    session, _ = make_session(frames, chat_client=chat, tts_provider=tts)

    session.run_one_cycle()  # must not raise

    out = capsys.readouterr().out
    assert "synth broke" in out
    assert "the reply text" in out
    assert session.state == VoiceState.WAKE_WORD_LISTENING


def test_run_one_cycle_propagates_chat_client_errors():
    frames = [WAKE_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME]
    chat = FakeChatClient(error=ChatClientError("server down"))
    session, _ = make_session(frames, chat_client=chat)

    with pytest.raises(ChatClientError):
        session.run_one_cycle()


def test_run_one_cycle_propagates_wake_word_errors():
    session, _ = make_session([WAKE_FRAME], wake_word_provider=FakeWakeWordProvider(error=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        session.run_one_cycle()


def test_run_one_cycle_propagates_microphone_errors():
    session, _ = make_session([])  # no frames at all

    with pytest.raises(MicrophoneError):
        session.run_one_cycle()


def test_run_forever_isolates_a_failed_cycle_and_continues(capsys):
    """A failed /chat call (or any other failure) must not crash the
    process - the loop keeps running and processes the next wake word
    normally, exactly like ReminderScheduler/AutomationEngine isolate a
    single bad interaction."""
    frames = [WAKE_FRAME, LOUD_FRAME, QUIET_FRAME, QUIET_FRAME] * 2
    session_holder: list[VoiceSession] = []

    class FailOnceThenStopChatClient:
        def __init__(self):
            self.calls = 0

        def send(self, message, conversation_id):
            self.calls += 1
            if self.calls == 1:
                raise ChatClientError("server down")
            session_holder[0].stop()
            return "ok", "conv-1"

    chat = FailOnceThenStopChatClient()
    session, _ = make_session(frames, chat_client=chat)
    session_holder.append(session)

    session.run_forever()

    assert chat.calls == 2
    out = capsys.readouterr().out
    assert "server down" in out
    assert out.count("returning to wake-word listening") >= 1
    assert "wake-word listener active" in out
    assert "stopped" in out


def test_run_forever_stops_tts_playback_on_the_way_out():
    tts = FakeTTSProvider()
    session, _ = make_session([], tts_provider=tts)
    session.stop()  # so the loop body never even runs

    session.run_forever()

    assert tts.stopped is True
