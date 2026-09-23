"""
Microphone input, backed by `sounddevice` (a thin, permissively-licensed
binding over PortAudio, with prebuilt Windows wheels - no system install
needed beyond `pip install`). Opens the input stream once at construction
time, so an unavailable/misconfigured microphone is reported once, at
startup (see app/voice/run.py), rather than failing deep inside the
wake-word listening loop.
"""

import sounddevice as sd

from app.voice.base import Microphone, MicrophoneError


class SoundDeviceMicrophone(Microphone):
    """Microphone implementation backed by a `sounddevice.RawInputStream`."""

    def __init__(self, sample_rate: int, device: str | int | None = None):
        self.sample_rate = sample_rate
        try:
            self._stream = sd.RawInputStream(
                samplerate=sample_rate, channels=1, dtype="int16", device=device
            )
            self._stream.start()
        except (sd.PortAudioError, OSError, ValueError) as exc:
            raise MicrophoneError(
                f"Failed to open microphone (device={device!r}): {exc}"
            ) from exc

    def read(self, num_frames: int) -> bytes:
        try:
            data, _overflowed = self._stream.read(num_frames)
        except (sd.PortAudioError, OSError) as exc:
            raise MicrophoneError(f"Failed to read from microphone: {exc}") from exc
        return bytes(data)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except (sd.PortAudioError, OSError):
            pass
