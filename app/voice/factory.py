"""
Builds the application's default voice-subsystem components from config -
the one place construction happens, same role app/reminders/factory.py
plays for the reminder store/scheduler.

Each factory function can raise its own typed error (MicrophoneError/
WakeWordError/SpeechToTextError) if a prerequisite isn't in place yet - a
missing package, a missing downloaded model, or a microphone that can't
be opened. app/voice/run.py is what catches those at startup and reports
them clearly rather than crashing partway through construction.
"""

from app.config import settings
from app.voice.base import Microphone, SpeechToTextProvider, TextToSpeechProvider, WakeWordProvider
from app.voice.chat_client import ChatClient
from app.voice.microphone import SoundDeviceMicrophone
from app.voice.session import VOICE_SAMPLE_RATE
from app.voice.stt import VoskSTTProvider
from app.voice.tts import WindowsSapiTTSProvider
from app.voice.wake_word import OpenWakeWordProvider


def create_default_microphone() -> Microphone:
    return SoundDeviceMicrophone(
        sample_rate=VOICE_SAMPLE_RATE, device=settings.voice_input_device or None
    )


def create_default_wake_word_provider() -> WakeWordProvider:
    return OpenWakeWordProvider(
        wakeword_model_path=settings.voice_wakeword_model_path,
        threshold=settings.voice_wakeword_threshold,
    )


def create_default_stt_provider() -> SpeechToTextProvider:
    return VoskSTTProvider(model_path=settings.voice_vosk_model_path)


def create_default_tts_provider() -> TextToSpeechProvider:
    return WindowsSapiTTSProvider(
        voice_name=settings.voice_tts_voice_name, rate=settings.voice_tts_rate
    )


def create_default_chat_client() -> ChatClient:
    return ChatClient(
        base_url=settings.voice_chat_url, timeout=settings.voice_chat_timeout_seconds
    )
