"""
Voice abstraction (M10): microphone -> wake word -> STT -> (existing
`/chat`) -> TTS -> speaker, entirely as a *client* of Jarvis - see
app/voice/session.py's module docstring for why this never imports
`app.core.core` directly.

Four small interfaces, the same "interface + swappable backend" shape as
AIProvider (app/providers/base.py), SearchProvider (app/search/base.py),
and CalendarProvider (app/calendar/base.py):

- `Microphone` - raw audio in.
- `WakeWordProvider` - "was the wake word just said?"
- `SpeechToTextProvider` - audio -> text.
- `TextToSpeechProvider` - text -> spoken audio.

Every concrete implementation (app/voice/microphone.py,
app/voice/wake_word.py, app/voice/stt.py, app/voice/tts.py) is local-only
and free: no cloud wake-word/STT/TTS service is ever called, and no
credential of any kind is needed for any of them. All four raise their
own typed *Error - never a raw third-party exception - so
app/voice/session.py can catch failures uniformly and keep the voice
loop running after a single bad interaction, the same convention
CalendarProviderError/SearchProviderError/ProviderError already use at
their own boundaries.

Audio is always raw 16-bit signed PCM, mono, at a single fixed sample
rate shared by every piece of this subsystem
(app/voice/session.py::VOICE_SAMPLE_RATE) - there is no resampling
anywhere in this milestone, so every provider is expected to produce/
consume audio at that rate directly.
"""

from abc import ABC, abstractmethod


class MicrophoneError(RuntimeError):
    """Raised when the microphone can't be opened or read - no input
    device, a device that's in use elsewhere, a driver error, etc."""


class Microphone(ABC):
    """A single raw audio input device, already opened at construction
    time (so "microphone unavailable" is discovered once, at startup,
    not deep inside a later read)."""

    #: Sample rate frames are captured at - always VOICE_SAMPLE_RATE in
    #: this milestone; kept as an attribute rather than a bare constant
    #: so a future provider could differ without changing the interface.
    sample_rate: int

    @abstractmethod
    def read(self, num_frames: int) -> bytes:
        """Block until `num_frames` samples are available and return them
        as raw 16-bit signed PCM mono bytes (`num_frames * 2` bytes).
        Raises MicrophoneError if the device fails while reading."""

    def close(self) -> None:
        """Release the underlying audio device. Default no-op - not
        every implementation holds an OS resource worth closing."""


class WakeWordError(RuntimeError):
    """Raised when the wake-word engine can't be initialized (missing
    package, missing/invalid model file) or fails while processing
    audio."""


class WakeWordProvider(ABC):
    """Detects a single fixed wake phrase in a continuous stream of audio
    frames, fed one frame at a time - never full-sentence speech
    recognition, and never anything beyond "was the wake word just said,
    yes or no"."""

    #: How many samples `process_frame` expects per call - callers read
    #: exactly this many frames from a Microphone before each call.
    frame_length: int

    @abstractmethod
    def process_frame(self, frame: bytes) -> bool:
        """Feed one frame of raw 16-bit PCM mono audio (`frame_length`
        samples). Returns True at most once per utterance of the wake
        word - implementations debounce/reset their own internal state
        after a detection so a single utterance can't fire twice."""


class SpeechToTextError(RuntimeError):
    """Raised when the STT engine can't be initialized (missing package,
    missing/invalid model) or fails while transcribing."""


class SpeechToTextProvider(ABC):
    """Transcribes one already-captured utterance. Stateless across
    calls - each `transcribe` is a single, self-contained recognition,
    not a running/streaming session."""

    @abstractmethod
    def transcribe(self, audio: bytes, sample_rate: int) -> str:
        """Transcribe raw 16-bit PCM mono `audio` and return the
        recognized text, stripped of surrounding whitespace. Returns ""
        (never raises) for silence/unintelligible audio - that's a valid,
        expected outcome, not a failure; only a genuine engine failure
        raises SpeechToTextError."""


class TextToSpeechError(RuntimeError):
    """Raised when the TTS engine can't be initialized or fails while
    synthesizing/playing speech."""


class TextToSpeechProvider(ABC):
    """Synthesizes and plays text as speech through the local speaker."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak `text` aloud, blocking until playback finishes (or
        `stop()` cuts it short). Raises TextToSpeechError on failure -
        callers fall back to printing `text` rather than losing the
        reply entirely (see app/voice/session.py)."""

    def stop(self) -> None:
        """Interrupt in-progress speech, if any - the "user interruption
        where reasonably possible" this milestone asks for, short of
        full barge-in. Default no-op; callers may call this at any time,
        including when nothing is speaking."""
