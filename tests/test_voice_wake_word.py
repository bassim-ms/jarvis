"""
Tests for OpenWakeWordProvider (app/voice/wake_word.py). openWakeWord's
own `Model` class is monkeypatched - no real model file or ONNX runtime
inference is ever exercised, matching this project's "no real
microphone/downloaded model needed for the test suite" convention.
"""

import numpy as np
import pytest

from app.voice import wake_word as wake_word_module
from app.voice.base import WakeWordError
from app.voice.wake_word import OpenWakeWordProvider


class FakeOwwModel:
    """Stands in for openwakeword.model.Model."""

    def __init__(self, wakeword_models, inference_framework, score=0.0, error=None):
        self.wakeword_models = wakeword_models
        self.inference_framework = inference_framework
        self.score = score
        self.error = error
        self.reset_calls = 0

    def predict(self, x):
        if self.error:
            raise self.error
        key = _model_key(self.wakeword_models[0])
        return {key: self.score}

    def reset(self):
        self.reset_calls += 1


def _model_key(path: str) -> str:
    import os

    return os.path.splitext(os.path.basename(path))[0]


def _frame() -> bytes:
    return np.zeros(1280, dtype=np.int16).tobytes()


# --- construction / missing model -----------------------------------------


def test_requires_a_configured_model_path():
    with pytest.raises(WakeWordError, match="No wake-word model configured"):
        OpenWakeWordProvider(wakeword_model_path="")


def test_raises_when_model_file_does_not_exist(tmp_path):
    missing = str(tmp_path / "hey_jarvis_v0.1.onnx")

    with pytest.raises(WakeWordError, match="not found"):
        OpenWakeWordProvider(wakeword_model_path=missing)


def test_wraps_a_model_construction_failure(tmp_path, monkeypatch):
    model_path = tmp_path / "hey_jarvis_v0.1.onnx"
    model_path.write_bytes(b"not a real model")

    def raising_model(*args, **kwargs):
        raise RuntimeError("bad onnx file")

    monkeypatch.setattr(wake_word_module, "Model", raising_model)

    with pytest.raises(WakeWordError, match="Failed to load wake-word model"):
        OpenWakeWordProvider(wakeword_model_path=str(model_path))


# --- process_frame -----------------------------------------------------


def _make_provider(tmp_path, monkeypatch, score: float = 0.0, error=None) -> tuple[OpenWakeWordProvider, list]:
    model_path = tmp_path / "hey_jarvis_v0.1.onnx"
    model_path.write_bytes(b"not a real model")
    created: list[FakeOwwModel] = []

    def fake_model(wakeword_models, inference_framework):
        fake = FakeOwwModel(wakeword_models, inference_framework, score=score, error=error)
        created.append(fake)
        return fake

    monkeypatch.setattr(wake_word_module, "Model", fake_model)
    provider = OpenWakeWordProvider(wakeword_model_path=str(model_path), threshold=0.5)
    return provider, created


def test_process_frame_returns_false_below_threshold(tmp_path, monkeypatch):
    provider, fakes = _make_provider(tmp_path, monkeypatch, score=0.2)

    assert provider.process_frame(_frame()) is False
    assert fakes[0].reset_calls == 0


def test_process_frame_returns_true_at_or_above_threshold_and_resets(tmp_path, monkeypatch):
    provider, fakes = _make_provider(tmp_path, monkeypatch, score=0.9)

    assert provider.process_frame(_frame()) is True
    assert fakes[0].reset_calls == 1


def test_process_frame_raises_wake_word_error_on_prediction_failure(tmp_path, monkeypatch):
    provider, _ = _make_provider(tmp_path, monkeypatch, error=RuntimeError("inference crashed"))

    with pytest.raises(WakeWordError, match="Wake-word detection failed"):
        provider.process_frame(_frame())


def test_frame_length_matches_openwakewords_recommended_chunk_size(tmp_path, monkeypatch):
    provider, _ = _make_provider(tmp_path, monkeypatch)

    assert provider.frame_length == 1280
