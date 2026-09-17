"""
Tests for Jarvis Core: it must delegate to whatever AIProvider it's given,
using a MemoryStore for conversation history and stored facts, without
depending on Gemini or SQLite specifically.
"""

import pytest

from app.core import ConversationNotFoundError, JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.search.answer import (
    GeminiSearchAnswerGenerator,
    SearchAnswer,
    SearchAnswerError,
    SearchAnswerGenerator,
)
from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.router import SearchRouter
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


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


# --- M6.9: SearchRouter / SEARCH path --------------------------------------


class FakeSearchProvider(SearchProvider):
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error

    def search(self, query, max_results=5):
        if self.error:
            raise self.error
        return self.results


class FakeAnswerGenerator(SearchAnswerGenerator):
    """Records the SearchResponse it was given and returns a fixed
    SearchAnswer, or raises - decoupled from Gemini's citation-matching
    logic, which is covered by tests/test_search_answer.py."""

    def __init__(self, answer: str = "search answer", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.last_response = None

    def generate(self, response):
        self.last_response = response
        if self.error:
            raise self.error
        return SearchAnswer(answer=self.answer, sources=[])


def make_search_core(
    reply: str = "direct reply",
    search_provider: SearchProvider | None = None,
    answer_generator: SearchAnswerGenerator | None = None,
    register_web_search: bool = True,
) -> tuple[JarvisCore, FakeProvider, ToolRegistry]:
    provider = FakeProvider(reply)
    tools = ToolRegistry()
    if register_web_search:
        tools.register(WebSearchTool(provider=search_provider or FakeSearchProvider()))
    jarvis = JarvisCore(
        provider=provider,
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        answer_generator=answer_generator,
    )
    return jarvis, provider, tools


def test_default_router_is_search_router():
    assert isinstance(JarvisCore().router, SearchRouter)


def test_default_answer_generator_is_gemini_backed_and_reuses_the_provider():
    core = JarvisCore()

    assert isinstance(core.answer_generator, GeminiSearchAnswerGenerator)
    assert core.answer_generator.provider is core.provider


def test_handle_chat_uses_direct_path_for_a_stable_question():
    answer_generator = FakeAnswerGenerator()
    jarvis, provider, _ = make_search_core(
        reply="gemini answer", answer_generator=answer_generator
    )

    reply, _ = jarvis.handle_chat("what is quantum entanglement?")

    assert reply == "gemini answer"
    assert provider.last_message is not None
    assert answer_generator.last_response is None  # SEARCH path never ran


def test_handle_chat_uses_search_path_for_a_search_triggering_message():
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, provider, _ = make_search_core(answer_generator=answer_generator)

    reply, _ = jarvis.handle_chat("what's the latest news on AI?")

    assert reply == "grounded answer"
    assert provider.last_message is None  # DIRECT path never ran
    assert answer_generator.last_response is not None
    assert answer_generator.last_response.query == "what's the latest news on AI?"


def test_search_path_persists_the_final_reply_to_memory():
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, _, _ = make_search_core(answer_generator=answer_generator)

    _, conversation_id = jarvis.handle_chat("what's the weather today?")

    messages = jarvis.memory.get_messages(conversation_id)
    assert [m.content for m in messages] == [
        "what's the weather today?",
        "grounded answer",
    ]


def test_search_path_mixes_correctly_with_direct_turns_in_one_conversation():
    """Proves conversation history/facts persistence keeps working when a
    conversation mixes DIRECT and SEARCH turns."""
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, _, _ = make_search_core(
        reply="direct reply", answer_generator=answer_generator
    )

    _, conversation_id = jarvis.handle_chat("hello")
    jarvis.handle_chat("what's the latest news?", conversation_id=conversation_id)

    messages = jarvis.memory.get_messages(conversation_id)
    assert [m.content for m in messages] == [
        "hello",
        "direct reply",
        "what's the latest news?",
        "grounded answer",
    ]


def test_search_path_passes_the_search_response_to_the_answer_generator():
    answer_generator = FakeAnswerGenerator()
    search_provider = FakeSearchProvider(
        results=[SearchResult(title="T", url="https://x.com", snippet="s")]
    )
    jarvis, _, _ = make_search_core(
        search_provider=search_provider, answer_generator=answer_generator
    )

    jarvis.handle_chat("what's the latest news?")

    response = answer_generator.last_response
    assert response.results == [
        SearchResult(title="T", url="https://x.com", snippet="s")
    ]


def test_search_path_provider_failure_raises_provider_error():
    search_provider = FakeSearchProvider(error=SearchProviderError("search is down"))
    jarvis, _, _ = make_search_core(search_provider=search_provider)

    with pytest.raises(ProviderError, match="search is down"):
        jarvis.handle_chat("what's the latest news?")


def test_search_path_answer_generator_failure_raises_provider_error():
    answer_generator = FakeAnswerGenerator(error=SearchAnswerError("gemini is down"))
    jarvis, _, _ = make_search_core(answer_generator=answer_generator)

    with pytest.raises(ProviderError, match="gemini is down"):
        jarvis.handle_chat("what's the latest news?")


def test_search_path_without_a_registered_web_search_tool_raises_provider_error():
    jarvis, _, _ = make_search_core(register_web_search=False)

    with pytest.raises(ProviderError, match="web_search"):
        jarvis.handle_chat("what's the latest news?")
