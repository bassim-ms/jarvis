"""
Tests for Jarvis Core: it must delegate to whatever AIProvider it's given,
without depending on Gemini specifically.
"""

from app.core import JarvisCore
from app.providers.base import AIProvider
from app.providers.gemini import GeminiProvider


class FakeProvider(AIProvider):
    def __init__(self, reply: str):
        self.reply = reply
        self.last_message = None

    def generate_reply(self, message: str) -> str:
        self.last_message = message
        return self.reply


def test_handle_chat_delegates_to_provider():
    provider = FakeProvider("hi there")
    jarvis = JarvisCore(provider=provider)

    assert jarvis.handle_chat("hello") == "hi there"
    assert provider.last_message == "hello"


def test_default_provider_is_gemini():
    assert isinstance(JarvisCore().provider, GeminiProvider)
