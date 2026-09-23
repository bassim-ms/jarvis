"""
Tests for SoundDeviceMicrophone (app/voice/microphone.py). `sounddevice`'s
own `RawInputStream` is monkeypatched - no real audio device is ever
opened, matching this project's "no real microphone needed for the test
suite" convention.
"""

import pytest
import sounddevice as sd

from app.voice import microphone as microphone_module
from app.voice.base import MicrophoneError
from app.voice.microphone import SoundDeviceMicrophone


class FakeStream:
    def __init__(self, samplerate, channels, dtype, device, fail_to_open: bool = False):
        if fail_to_open:
            raise sd.PortAudioError("no default input device")
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.started = False
        self.stopped = False
        self.closed = False
        self.read_calls: list[int] = []
        self.fail_on_read = False

    def start(self):
        self.started = True

    def read(self, num_frames):
        self.read_calls.append(num_frames)
        if self.fail_on_read:
            raise sd.PortAudioError("device disconnected")
        return (b"\x00\x01" * num_frames, False)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_opens_and_starts_the_stream_on_construction(monkeypatch):
    created = {}

    def fake_raw_input_stream(**kwargs):
        stream = FakeStream(**kwargs)
        created["stream"] = stream
        return stream

    monkeypatch.setattr(microphone_module.sd, "RawInputStream", fake_raw_input_stream)

    mic = SoundDeviceMicrophone(sample_rate=16000, device="Test Mic")

    assert mic.sample_rate == 16000
    assert created["stream"].started is True
    assert created["stream"].device == "Test Mic"
    assert created["stream"].channels == 1
    assert created["stream"].dtype == "int16"


def test_raises_microphone_error_when_the_device_cannot_be_opened(monkeypatch):
    def fake_raw_input_stream(**kwargs):
        return FakeStream(fail_to_open=True, **kwargs)

    monkeypatch.setattr(microphone_module.sd, "RawInputStream", fake_raw_input_stream)

    with pytest.raises(MicrophoneError, match="Failed to open microphone"):
        SoundDeviceMicrophone(sample_rate=16000)


def test_read_returns_bytes_from_the_stream(monkeypatch):
    def fake_raw_input_stream(**kwargs):
        return FakeStream(**kwargs)

    monkeypatch.setattr(microphone_module.sd, "RawInputStream", fake_raw_input_stream)
    mic = SoundDeviceMicrophone(sample_rate=16000)

    data = mic.read(4)

    assert data == b"\x00\x01" * 4


def test_read_raises_microphone_error_on_stream_failure(monkeypatch):
    def fake_raw_input_stream(**kwargs):
        stream = FakeStream(**kwargs)
        stream.fail_on_read = True
        return stream

    monkeypatch.setattr(microphone_module.sd, "RawInputStream", fake_raw_input_stream)
    mic = SoundDeviceMicrophone(sample_rate=16000)

    with pytest.raises(MicrophoneError, match="Failed to read from microphone"):
        mic.read(4)


def test_close_stops_and_closes_the_stream(monkeypatch):
    created = {}

    def fake_raw_input_stream(**kwargs):
        stream = FakeStream(**kwargs)
        created["stream"] = stream
        return stream

    monkeypatch.setattr(microphone_module.sd, "RawInputStream", fake_raw_input_stream)
    mic = SoundDeviceMicrophone(sample_rate=16000)

    mic.close()

    assert created["stream"].stopped is True
    assert created["stream"].closed is True


def test_close_never_raises_even_if_the_stream_is_already_broken(monkeypatch):
    def fake_raw_input_stream(**kwargs):
        stream = FakeStream(**kwargs)

        def broken_stop():
            raise sd.PortAudioError("already closed")

        stream.stop = broken_stop
        return stream

    monkeypatch.setattr(microphone_module.sd, "RawInputStream", fake_raw_input_stream)
    mic = SoundDeviceMicrophone(sample_rate=16000)

    mic.close()  # must not raise
