"""
Run Rhasspy's hands-free voice interface:

    python -m app.voice.run

Requires the regular Rhasspy server to already be running separately
(`uvicorn app.main:app`) - this process is only ever a *client* of the
existing `/chat` endpoint (see app/voice/chat_client.py); it never
imports `app.core.core` and never starts its own copy of Core, the
reminder scheduler, or the automation engine.

Also requires the one-time setup described in .env.example /
JARVIS_VOICE_WAKEWORD_MODEL_PATH / JARVIS_VOICE_VOSK_MODEL_PATH to
already be done - see this repo's README's Voice Interface section for
exactly what to download and where. Any missing prerequisite (no
microphone, no wake-word model, no Vosk model) is reported clearly here,
once, at startup, rather than failing deep inside the listening loop.
"""

from app.voice.base import (
    MicrophoneError,
    SpeechToTextError,
    TextToSpeechError,
    WakeWordError,
)
from app.voice.factory import (
    create_default_chat_client,
    create_default_microphone,
    create_default_stt_provider,
    create_default_tts_provider,
    create_default_wake_word_provider,
)
from app.voice.session import VoiceSession


def main() -> None:
    try:
        microphone = create_default_microphone()
        wake_word_provider = create_default_wake_word_provider()
        stt_provider = create_default_stt_provider()
        tts_provider = create_default_tts_provider()
    except (MicrophoneError, WakeWordError, SpeechToTextError, TextToSpeechError) as exc:
        print(f"[rhasspy] startup failed: {exc}")
        raise SystemExit(1) from exc

    session = VoiceSession(
        microphone=microphone,
        wake_word_provider=wake_word_provider,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        chat_client=create_default_chat_client(),
    )
    try:
        session.run_forever()
    finally:
        microphone.close()


if __name__ == "__main__":
    main()
