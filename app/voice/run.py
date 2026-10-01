"""
Run Jarvis's hands-free voice interface:

    python -m app.voice.run

Requires the regular Jarvis server to already be running separately
(`uvicorn app.main:app`) - this process is only ever a *client* of the
existing `/chat` endpoint (see app/voice/chat_client.py); it never
imports `app.core.core` and never starts its own copy of Core, the
reminder scheduler, or the automation engine.

It runs as the registered, trusted PC: it authenticates to the Device Gateway
with the credential in `jarvis_pc_device.json` (written once by
`python -m app.devices.admin register-pc`), checks in at startup, and talks to
the authenticated `/device/chat`. Without a usable credential it stops with a
clear message - it never falls back to the anonymous `/chat`.

Also requires the one-time setup described in .env.example /
JARVIS_VOICE_WAKEWORD_MODEL_PATH / JARVIS_VOICE_VOSK_MODEL_PATH to
already be done - see this repo's README's Voice Interface section for
exactly what to download and where. Any missing prerequisite (no
microphone, no wake-word model, no Vosk model) is reported clearly here,
once, at startup, rather than failing deep inside the listening loop.
"""

from app.voice.chat_client import ChatClient, ChatClientError, DeviceAuthError, DeviceCredentialError
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


def check_in_at_startup(chat_client: ChatClient) -> None:
    """The PC says "I am alive" (authenticated) so the registry shows it ONLINE. A rejected credential stops
    startup (SystemExit) - the PC does not carry on as an unauthenticated client. A server that simply is not
    reachable yet is not fatal: the same check-in is retried before the first authenticated message."""
    try:
        chat_client.check_in()
    except (DeviceAuthError, DeviceCredentialError) as exc:  # rejected, or no credential at all: never carry on anonymously
        print(f"[jarvis] startup failed: {exc}")
        raise SystemExit(1) from exc
    except ChatClientError as exc:
        print(f"[jarvis] could not check in yet ({exc}) - will retry when the first request is sent")
        return
    print("[jarvis] checked in as the registered PC")


def main() -> None:
    try:
        microphone = create_default_microphone()
        wake_word_provider = create_default_wake_word_provider()
        stt_provider = create_default_stt_provider()
        tts_provider = create_default_tts_provider()
        chat_client = create_default_chat_client()  # needs the PC's device credential; no anonymous fallback
        if not chat_client.authenticated:  # defence in depth: the PC voice client is never the anonymous /chat client
            raise DeviceCredentialError("The voice client has no device credential, so it will not start.")
    except (MicrophoneError, WakeWordError, SpeechToTextError, TextToSpeechError, ChatClientError) as exc:
        print(f"[jarvis] startup failed: {exc}")
        raise SystemExit(1) from exc
    check_in_at_startup(chat_client)

    session = VoiceSession(
        microphone=microphone,
        wake_word_provider=wake_word_provider,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        chat_client=chat_client,
    )
    try:
        session.run_forever()
    finally:
        microphone.close()


if __name__ == "__main__":
    main()
