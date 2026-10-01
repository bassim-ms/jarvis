"""
Tests for VoskSTTProvider (app/voice/stt.py). Vosk's own `Model`/
`KaldiRecognizer` classes are monkeypatched - no real model file or
recognizer is ever loaded, matching this project's "no downloaded model
needed for the test suite" convention.
"""

import json

import pytest

from app.voice import stt as stt_module
from app.voice.base import SpeechToTextError
from app.voice.stt import VoskSTTProvider


class FakeKaldiRecognizer:
    def __init__(self, model, sample_rate, text: str = "", error: Exception | None = None):
        self.model = model
        self.sample_rate = sample_rate
        self.text = text
        self.error = error
        self.accepted: bytes | None = None

    def AcceptWaveform(self, audio: bytes) -> bool:
        if self.error:
            raise self.error
        self.accepted = audio
        return True

    def FinalResult(self) -> str:
        return json.dumps({"text": self.text})


# --- construction / missing model -----------------------------------------


def test_requires_a_configured_model_path():
    with pytest.raises(SpeechToTextError, match="No Vosk model configured"):
        VoskSTTProvider(model_path="")


def test_raises_when_model_directory_does_not_exist(tmp_path):
    missing = str(tmp_path / "vosk-model-small-en-us")

    with pytest.raises(SpeechToTextError, match="not found"):
        VoskSTTProvider(model_path=missing)


def test_wraps_a_model_construction_failure(tmp_path, monkeypatch):
    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir()

    def raising_model(path):
        raise Exception("Failed to create a model")

    monkeypatch.setattr(stt_module.vosk, "Model", raising_model)

    with pytest.raises(SpeechToTextError, match="Failed to load Vosk model"):
        VoskSTTProvider(model_path=str(model_dir))


# --- transcribe -----------------------------------------------------


def _make_provider(tmp_path, monkeypatch, text: str = "", error: Exception | None = None) -> VoskSTTProvider:
    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir()
    monkeypatch.setattr(stt_module.vosk, "Model", lambda path: object())
    monkeypatch.setattr(
        stt_module.vosk,
        "KaldiRecognizer",
        lambda model, sample_rate: FakeKaldiRecognizer(model, sample_rate, text=text, error=error),
    )
    return VoskSTTProvider(model_path=str(model_dir))


def test_transcribe_returns_recognized_text(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, text="what's on my calendar today")

    result = provider.transcribe(b"\x00\x01" * 100, 16000)

    assert result == "what's on my calendar today"


def test_transcribe_strips_surrounding_whitespace(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, text="  hello jarvis  ")

    assert provider.transcribe(b"\x00\x01" * 100, 16000) == "hello jarvis"


def test_transcribe_returns_empty_string_for_silence(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, text="")

    assert provider.transcribe(b"\x00\x00" * 100, 16000) == ""


def test_transcribe_raises_speech_to_text_error_on_engine_failure(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, error=RuntimeError("decoder crashed"))

    with pytest.raises(SpeechToTextError, match="Vosk transcription failed"):
        provider.transcribe(b"\x00\x01" * 100, 16000)
