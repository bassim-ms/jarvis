"""
A chat reply's prompt carries only a conversation's most recent messages (settings.chat_history_max_messages); the stored
conversation is never trimmed. Real JarvisCore, real SQLite memory (in-memory), a recording fake provider - no network.
"""

import pytest

from app.capability import Capability, CapabilityDecision
from app.config import Settings, settings
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider


class RecordingProvider(AIProvider):
    def __init__(self):
        self.prompts = []

    def generate_reply(self, message: str) -> str:
        self.prompts.append(message)
        return f"reply {len(self.prompts)}"


class DirectRouter:
    def route(self, message):
        return CapabilityDecision(capability=Capability.DIRECT)


def make_core():
    provider = RecordingProvider()
    memory = SQLiteMemoryStore(":memory:")
    return JarvisCore(provider=provider, memory=memory, capability_router=DirectRouter()), provider, memory


def talk(core, turns, conversation_id=None):
    for index in range(turns):
        conversation_id = core.handle_chat(f"user message {index}", conversation_id)[1]
    return conversation_id


def test_the_default_limit_is_forty_messages():
    assert Settings.model_fields["chat_history_max_messages"].default == 40


def test_a_long_conversation_sends_only_the_most_recent_messages(monkeypatch):
    monkeypatch.setattr(settings, "chat_history_max_messages", 6)
    core, provider, memory = make_core()

    conversation_id = talk(core, 10)  # 10 turns = 20 stored messages
    core.handle_chat("the newest question", conversation_id)
    prompt = provider.prompts[-1]

    assert "user message 9" in prompt and "reply 10" in prompt  # the recent ones are there
    assert "user message 0" not in prompt and "reply 1\n" not in prompt  # the oldest are not
    assert prompt.rstrip().endswith("User: the newest question")
    history_lines = [line for line in prompt.splitlines() if line.startswith(("user: ", "assistant: "))]
    assert len(history_lines) == 6


def test_the_prompt_says_plainly_that_earlier_messages_are_not_shown(monkeypatch):
    monkeypatch.setattr(settings, "chat_history_max_messages", 4)
    core, provider, _ = make_core()

    conversation_id = talk(core, 5)
    core.handle_chat("next", conversation_id)

    assert "the 4 most recent messages; 6 earlier messages are not shown" in provider.prompts[-1]


def test_a_short_conversation_is_sent_whole_with_the_ordinary_heading(monkeypatch):
    monkeypatch.setattr(settings, "chat_history_max_messages", 40)
    core, provider, _ = make_core()

    conversation_id = talk(core, 3)
    core.handle_chat("next", conversation_id)
    prompt = provider.prompts[-1]

    assert "Conversation so far:\n" in prompt and "not shown" not in prompt
    assert all(f"user message {index}" in prompt for index in range(3))


def test_the_stored_conversation_is_never_trimmed(monkeypatch):
    monkeypatch.setattr(settings, "chat_history_max_messages", 2)
    core, _, memory = make_core()

    conversation_id = talk(core, 12)

    stored = memory.get_messages(conversation_id)
    assert len(stored) == 24
    assert stored[0].content == "user message 0" and stored[-1].content == "reply 12"


@pytest.mark.parametrize("unlimited", [0, -1])
def test_zero_means_no_limit(monkeypatch, unlimited):
    monkeypatch.setattr(settings, "chat_history_max_messages", unlimited)
    core, provider, _ = make_core()

    conversation_id = talk(core, 30)
    core.handle_chat("next", conversation_id)
    prompt = provider.prompts[-1]

    assert "user message 0" in prompt and "user message 29" in prompt and "not shown" not in prompt


def test_the_prompt_stops_growing_with_the_conversation(monkeypatch):
    monkeypatch.setattr(settings, "chat_history_max_messages", 8)
    core, provider, _ = make_core()

    conversation_id = talk(core, 40)
    sizes = [len(prompt) for prompt in provider.prompts[-10:]]

    assert max(sizes) - min(sizes) < 40  # steady: only the digit widths of the message numbers differ
    assert conversation_id
