"""
Tests for Jarvis Core: it must delegate to whatever AIProvider it's given,
using a MemoryStore for conversation history and stored facts, without
depending on Gemini or SQLite specifically.
"""

import pytest

from app.core import ConversationNotFoundError, JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.providers.gemini import GeminiProvider


class FakeProvider(AIProvider):
    def __init__(self, reply: str):
        self.reply = reply
        self.last_message = None

    def generate_reply(self, message: str) -> str:
        self.last_message = message
        return self.reply


def make_core(reply: str = "hi there") -> tuple[JarvisCore, FakeProvider]:
    provider = FakeProvider(reply)
    jarvis = JarvisCore(provider=provider, memory=SQLiteMemoryStore(":memory:"))
    return jarvis, provider


def test_handle_chat_delegates_to_provider():
    jarvis, provider = make_core("hi there")

    reply, conversation_id = jarvis.handle_chat("hello")

    assert reply == "hi there"
    assert provider.last_message == "hello"
    assert conversation_id


def test_default_provider_is_gemini():
    assert isinstance(JarvisCore().provider, GeminiProvider)


def test_handle_chat_creates_new_conversation_when_none_given():
    jarvis, _ = make_core()

    _, first_id = jarvis.handle_chat("hello")
    _, second_id = jarvis.handle_chat("hello again")

    assert first_id != second_id


def test_handle_chat_reuses_and_persists_conversation_history():
    jarvis, provider = make_core()

    _, conversation_id = jarvis.handle_chat("my name is Bassim")
    jarvis.handle_chat("what did I just say?", conversation_id=conversation_id)

    assert "my name is Bassim" in provider.last_message
    assert len(jarvis.memory.get_messages(conversation_id)) == 4


def test_handle_chat_raises_for_unknown_conversation_id():
    jarvis, provider = make_core()

    with pytest.raises(ConversationNotFoundError):
        jarvis.handle_chat("hello", conversation_id="does-not-exist")

    assert provider.last_message is None


def test_handle_chat_includes_saved_facts_in_context():
    jarvis, provider = make_core()
    jarvis.save_memory("name", "Bassim")

    jarvis.handle_chat("hi")

    assert "Bassim" in provider.last_message


def test_save_and_get_memories():
    jarvis, _ = make_core()

    jarvis.save_memory("favorite_language", "Python")

    assert jarvis.get_memories() == {"favorite_language": "Python"}
