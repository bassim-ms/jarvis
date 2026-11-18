"""
Tests for Jarvis Core: it must delegate to whatever AIProvider it's given,
using a MemoryStore for conversation history and stored facts, without
depending on Gemini or SQLite specifically.
"""

import pytest

from app.capability import (
    Capability,
    CapabilityDecision,
    CapabilityDecisionError,
    CapabilityDecisionProvider,
    CapabilityRouter,
    DeterministicCapabilityRouter,
    HybridCapabilityRouter,
)
from app.core import ConversationNotFoundError, JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.orchestration.base import JobStatus
from app.providers.base import AIProvider, ProviderError
from app.providers.manager import AIProviderEntry, AIProviderManager
from app.providers.requirements import TaskType
from app.search.answer import (
    GeminiSearchAnswerGenerator,
    SearchAnswer,
    SearchAnswerError,
    SearchAnswerGenerator,
    SearchSource,
)
from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.query_builder import DeterministicSearchQueryBuilder, SearchQueryBuilder
from app.search.router import SearchRouter
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.decision import ToolCall, ToolCallBuilder, ToolCallError, ToolCallProvider
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

    reply, conversation_id, sources = jarvis.handle_chat("hello")

    assert reply == "hi there"
    assert provider.last_message == "hello"
    assert conversation_id
    assert sources == []


def test_default_provider_is_the_ai_provider_manager(monkeypatch):
    """Core's own default (when no provider is injected) is the ordered fallback manager (app/providers/manager.py),
    built from JARVIS_AI_PROVIDER_ORDER - here the code default, the user's fixed Groq -> OpenRouter -> Gemini ->
    Cloudflare - not a bare GeminiProvider. The order is pinned to the code default and FreeLLMAPI is cleared so the
    developer's own .env decides neither."""
    from app.config import Settings, settings

    monkeypatch.setattr(settings, "ai_provider_order", Settings.model_fields["ai_provider_order"].default)
    monkeypatch.setattr(settings, "freellmapi_api_key", "")
    provider = JarvisCore().provider
    assert isinstance(provider, AIProviderManager)
    assert [health.name for health in provider.get_health()] == [
        "groq",
        "openrouter",
        "gemini",
        "cloudflare",
    ]


def test_handle_chat_creates_new_conversation_when_none_given():
    jarvis, _ = make_core()

    _, first_id, _ = jarvis.handle_chat("hello")
    _, second_id, _ = jarvis.handle_chat("hello again")

    assert first_id != second_id


def test_handle_chat_reuses_and_persists_conversation_history():
    jarvis, provider = make_core()

    _, conversation_id, _ = jarvis.handle_chat("my name is Bassim")
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
        self.last_query = None

    def search(self, query, max_results=5):
        self.last_query = query
        if self.error:
            raise self.error
        return self.results


class FakeAnswerGenerator(SearchAnswerGenerator):
    """Records the SearchResponse and conversation_context it was given
    and returns a fixed SearchAnswer, or raises - decoupled from Gemini's
    citation-matching/prompt logic, which is covered by
    tests/test_search_answer.py."""

    def __init__(
        self,
        answer: str = "search answer",
        sources: list[SearchSource] | None = None,
        error: Exception | None = None,
    ):
        self.answer = answer
        self.sources = sources if sources is not None else []
        self.error = error
        self.last_response = None
        self.last_context = None

    def generate(self, response, conversation_context=""):
        self.last_response = response
        self.last_context = conversation_context
        if self.error:
            raise self.error
        return SearchAnswer(answer=self.answer, sources=self.sources)


class RecordingRouter(SearchRouter):
    """Wraps the real deterministic routing but records the exact message
    it was asked to route, to prove Core never appends conversation
    context before routing - M6.10 only affects what happens AFTER a
    SEARCH decision has already been made, never routing itself."""

    def __init__(self):
        super().__init__()
        self.last_message = None

    def route(self, message):
        self.last_message = message
        return super().route(message)


def make_search_core(
    reply: str = "direct reply",
    search_provider: SearchProvider | None = None,
    answer_generator: SearchAnswerGenerator | None = None,
    register_web_search: bool = True,
    router: SearchRouter | None = None,
    query_builder: SearchQueryBuilder | None = None,
    capability_router: CapabilityRouter | None = None,
) -> tuple[JarvisCore, FakeProvider, ToolRegistry]:
    provider = FakeProvider(reply)
    tools = ToolRegistry()
    if register_web_search:
        tools.register(WebSearchTool(provider=search_provider or FakeSearchProvider()))
    jarvis = JarvisCore(
        provider=provider,
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        router=router,
        answer_generator=answer_generator,
        query_builder=query_builder,
        capability_router=capability_router,
    )
    return jarvis, provider, tools


def test_default_router_is_search_router():
    assert isinstance(JarvisCore().router, SearchRouter)


def test_default_answer_generator_is_gemini_backed_and_reuses_the_provider():
    core = JarvisCore()

    assert isinstance(core.answer_generator, GeminiSearchAnswerGenerator)
    assert core.answer_generator.provider is core.provider


def test_default_query_builder_is_deterministic_search_query_builder():
    assert isinstance(JarvisCore().query_builder, DeterministicSearchQueryBuilder)


def test_default_capability_router_is_deterministic_capability_router():
    assert isinstance(JarvisCore().capability_router, DeterministicCapabilityRouter)


def test_handle_chat_uses_direct_path_for_a_stable_question():
    answer_generator = FakeAnswerGenerator()
    jarvis, provider, _ = make_search_core(
        reply="gemini answer", answer_generator=answer_generator
    )

    reply, _, sources = jarvis.handle_chat("what is quantum entanglement?")

    assert reply == "gemini answer"
    assert provider.last_message is not None
    assert answer_generator.last_response is None  # SEARCH path never ran
    assert answer_generator.last_context is None  # never even called
    assert sources == []


def test_handle_chat_uses_search_path_for_a_search_triggering_message():
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, provider, _ = make_search_core(answer_generator=answer_generator)

    reply, _, sources = jarvis.handle_chat("what's the latest news on AI?")

    assert reply == "grounded answer"
    assert provider.last_message is None  # DIRECT path never ran
    assert answer_generator.last_response is not None
    assert answer_generator.last_response.query == "what's the latest news on AI?"
    assert sources == []


def test_search_path_persists_the_final_reply_to_memory():
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, _, _ = make_search_core(answer_generator=answer_generator)

    _, conversation_id, _ = jarvis.handle_chat("what's the weather today?")

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

    _, conversation_id, _ = jarvis.handle_chat("hello")
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


# --- M6.11: SearchAnswer.sources surfaced through handle_chat ---------------


def test_search_path_surfaces_the_answer_generators_sources_verbatim():
    sources = [
        SearchSource(title="Python.org", url="https://python.org", query="python"),
        SearchSource(
            title="Python Docs", url="https://docs.python.org", query="python"
        ),
    ]
    answer_generator = FakeAnswerGenerator(answer="grounded answer", sources=sources)
    jarvis, _, _ = make_search_core(answer_generator=answer_generator)

    _, _, returned_sources = jarvis.handle_chat("what's the latest news on AI?")

    assert returned_sources == sources


def test_search_path_with_no_cited_sources_returns_empty_list():
    answer_generator = FakeAnswerGenerator(answer="grounded answer", sources=[])
    jarvis, _, _ = make_search_core(answer_generator=answer_generator)

    _, _, sources = jarvis.handle_chat("what's the latest news on AI?")

    assert sources == []


def test_direct_path_always_returns_empty_sources_even_with_facts_and_history():
    jarvis, _, _ = make_search_core(reply="direct reply")
    jarvis.save_memory("name", "Bassim")

    _, _, sources = jarvis.handle_chat("what is quantum entanglement?")

    assert sources == []


# --- M6.12: deterministic conversational query enrichment --------------------


class FakeSearchQueryBuilder(SearchQueryBuilder):
    """Records the message/context it was given and returns either a fixed
    value or the message unchanged, or raises - decoupled from
    DeterministicSearchQueryBuilder's own pattern-matching logic, which is
    covered by tests/test_search_query_builder.py."""

    def __init__(
        self, fixed_result: str | None = None, error: Exception | None = None
    ):
        self.fixed_result = fixed_result
        self.error = error
        self.last_message = None
        self.last_context = None
        self.call_count = 0

    def build(self, message, recent_context):
        self.call_count += 1
        self.last_message = message
        self.last_context = recent_context
        if self.error:
            raise self.error
        return self.fixed_result if self.fixed_result is not None else message


def test_search_path_enriches_a_context_dependent_follow_up_before_searching():
    search_provider = FakeSearchProvider(results=[])
    answer_generator = FakeAnswerGenerator()
    jarvis, _, _ = make_search_core(
        search_provider=search_provider, answer_generator=answer_generator
    )

    _, conversation_id, _ = jarvis.handle_chat("what's the latest news about SpaceX?")
    jarvis.handle_chat(
        "is there anything more recent on that?", conversation_id=conversation_id
    )

    assert "SpaceX" in search_provider.last_query
    assert "is there anything more recent on that?" in search_provider.last_query


def test_search_response_query_reflects_the_enriched_query_not_the_raw_message():
    """Important architectural decision (M6.12): SearchResponse.query (and
    therefore SearchResult.query provenance downstream) records whatever
    was actually sent to the search pipeline - the enriched query, not the
    raw follow-up message - without any schema change to either
    dataclass."""
    search_provider = FakeSearchProvider(results=[])
    answer_generator = FakeAnswerGenerator()
    jarvis, _, _ = make_search_core(
        search_provider=search_provider, answer_generator=answer_generator
    )

    _, conversation_id, _ = jarvis.handle_chat("what's the latest news about SpaceX?")
    jarvis.handle_chat(
        "is there anything more recent on that?", conversation_id=conversation_id
    )

    response = answer_generator.last_response
    assert response.query == search_provider.last_query
    assert "SpaceX" in response.query


def test_self_contained_follow_up_search_query_is_sent_unchanged():
    search_provider = FakeSearchProvider(results=[])
    jarvis, _, _ = make_search_core(search_provider=search_provider)

    _, conversation_id, _ = jarvis.handle_chat("what's the latest news about SpaceX?")
    jarvis.handle_chat(
        "what's the latest news about NASA?", conversation_id=conversation_id
    )

    assert search_provider.last_query == "what's the latest news about NASA?"


def test_first_turn_search_query_is_sent_unchanged_with_no_prior_context():
    search_provider = FakeSearchProvider(results=[])
    jarvis, _, _ = make_search_core(search_provider=search_provider)

    jarvis.handle_chat("is there anything more recent on that?")

    assert search_provider.last_query == "is there anything more recent on that?"


def test_search_router_still_receives_only_the_raw_message_when_the_query_is_enriched():
    router = RecordingRouter()
    search_provider = FakeSearchProvider(results=[])
    jarvis, _, _ = make_search_core(router=router, search_provider=search_provider)

    _, conversation_id, _ = jarvis.handle_chat("what's the latest news about SpaceX?")
    jarvis.handle_chat(
        "is there anything more recent on that?", conversation_id=conversation_id
    )

    assert router.last_message == "is there anything more recent on that?"
    assert "SpaceX" in search_provider.last_query  # the query WAS enriched
    assert router.last_message != search_provider.last_query  # but not what routed


def test_direct_path_never_invokes_the_query_builder():
    query_builder = FakeSearchQueryBuilder(
        error=AssertionError("query builder should not run on the DIRECT path")
    )
    jarvis, _, _ = make_search_core(reply="direct reply", query_builder=query_builder)

    reply, _, _ = jarvis.handle_chat("what is quantum entanglement?")

    assert reply == "direct reply"
    assert query_builder.call_count == 0


def test_query_builder_is_constructor_injectable_and_used_verbatim():
    query_builder = FakeSearchQueryBuilder(fixed_result="a completely different query")
    search_provider = FakeSearchProvider(results=[])
    jarvis, _, _ = make_search_core(
        query_builder=query_builder, search_provider=search_provider
    )

    jarvis.handle_chat("what's the latest news?")

    assert search_provider.last_query == "a completely different query"


def test_query_builder_receives_the_same_bounded_context_as_the_answer_generator():
    query_builder = FakeSearchQueryBuilder()
    answer_generator = FakeAnswerGenerator()
    search_provider = FakeSearchProvider(results=[])
    jarvis, _, _ = make_search_core(
        query_builder=query_builder,
        answer_generator=answer_generator,
        search_provider=search_provider,
    )

    _, conversation_id, _ = jarvis.handle_chat("what's the latest news about SpaceX?")
    jarvis.handle_chat(
        "what's the latest news about NASA?", conversation_id=conversation_id
    )

    assert query_builder.last_context == answer_generator.last_context


def test_query_builder_context_is_bounded_to_the_configured_message_window():
    """M6.10's bounded-context limits are reused unchanged for query
    building - no new/wider context system."""
    query_builder = FakeSearchQueryBuilder()
    jarvis, _, _ = make_search_core(reply="ok", query_builder=query_builder)

    _, conversation_id, _ = jarvis.handle_chat("oldest message, should drop out")
    for i in range(5):
        jarvis.handle_chat(f"filler {i}", conversation_id=conversation_id)
    jarvis.handle_chat("what's the latest news?", conversation_id=conversation_id)

    context = query_builder.last_context
    assert "oldest message, should drop out" not in context
    assert "filler 0" not in context
    assert "filler 4" in context  # within the last _SEARCH_CONTEXT_MAX_MESSAGES


# --- M6.10: conversation context for the SEARCH answer step -----------------


def test_search_path_passes_recent_conversation_context_to_the_answer_generator():
    """The TUM example: a follow-up SEARCH question gets the prior DIRECT
    turn in its conversation_context, so "the latest application deadline"
    can be understood to mean TUM's deadline."""
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, _, _ = make_search_core(
        reply="TUM is a university in Munich.", answer_generator=answer_generator
    )

    _, conversation_id, _ = jarvis.handle_chat("tell me about TUM")
    jarvis.handle_chat(
        "what's the latest application deadline?", conversation_id=conversation_id
    )

    assert answer_generator.last_context
    assert "TUM" in answer_generator.last_context


def test_search_path_context_includes_the_earlier_relevant_turn():
    answer_generator = FakeAnswerGenerator()
    jarvis, _, _ = make_search_core(
        reply="TUM is a university in Munich.", answer_generator=answer_generator
    )

    _, conversation_id, _ = jarvis.handle_chat("tell me about TUM")
    jarvis.handle_chat(
        "what's the latest application deadline?", conversation_id=conversation_id
    )

    context = answer_generator.last_context
    assert "tell me about TUM" in context
    assert "TUM is a university in Munich." in context


def test_search_path_context_is_bounded_to_the_configured_message_window():
    answer_generator = FakeAnswerGenerator()
    jarvis, _, _ = make_search_core(reply="ok", answer_generator=answer_generator)

    _, conversation_id, _ = jarvis.handle_chat("oldest message, should drop out")
    for i in range(5):
        jarvis.handle_chat(f"filler {i}", conversation_id=conversation_id)
    jarvis.handle_chat("what's the latest news?", conversation_id=conversation_id)

    context = answer_generator.last_context
    assert "oldest message, should drop out" not in context
    assert "filler 0" not in context
    assert "filler 1" not in context
    assert "filler 4" in context  # within the last _SEARCH_CONTEXT_MAX_MESSAGES


def test_search_path_context_has_a_defensive_character_cap():
    answer_generator = FakeAnswerGenerator()
    jarvis, _, _ = make_search_core(
        reply="x" * 2000, answer_generator=answer_generator
    )

    _, conversation_id, _ = jarvis.handle_chat("tell me something long")
    jarvis.handle_chat("what's the latest news?", conversation_id=conversation_id)

    assert len(answer_generator.last_context) <= JarvisCore._SEARCH_CONTEXT_MAX_CHARS


def test_search_router_still_receives_only_the_raw_current_message():
    """M6.10 must not change what SearchRouter sees - routing is decided
    entirely from the current message, same as M6.9."""
    router = RecordingRouter()
    jarvis, _, _ = make_search_core(
        reply="TUM is a university in Munich.", router=router
    )

    _, conversation_id, _ = jarvis.handle_chat("tell me about TUM")
    jarvis.handle_chat(
        "what's the latest application deadline?", conversation_id=conversation_id
    )

    assert router.last_message == "what's the latest application deadline?"


def test_web_search_still_receives_only_the_raw_current_message():
    """M6.10 must not change what WebSearchTool.search() is asked to
    search - the search query stays exactly the current raw message,
    never enriched with conversation context."""
    search_provider = FakeSearchProvider(results=[])
    jarvis, _, _ = make_search_core(
        reply="TUM is a university in Munich.", search_provider=search_provider
    )

    _, conversation_id, _ = jarvis.handle_chat("tell me about TUM")
    jarvis.handle_chat(
        "what's the latest application deadline?", conversation_id=conversation_id
    )

    assert search_provider.last_query == "what's the latest application deadline?"


def test_search_response_propagation_to_the_answer_generator_is_unchanged():
    """SearchResponse itself (query/mode/queries_used/results) still flows
    through to the answer generator exactly as in M6.9 - only the new,
    separate conversation_context argument was added."""
    answer_generator = FakeAnswerGenerator()
    search_provider = FakeSearchProvider(
        results=[SearchResult(title="T", url="https://x.com", snippet="s")]
    )
    jarvis, _, _ = make_search_core(
        search_provider=search_provider, answer_generator=answer_generator
    )

    jarvis.handle_chat("what's the latest news?")

    response = answer_generator.last_response
    assert response.query == "what's the latest news?"
    assert response.results == [
        SearchResult(title="T", url="https://x.com", snippet="s")
    ]


# --- M6.13: Capability Decision Layer dispatch --------------------------------


class FakeCapabilityRouter(CapabilityRouter):
    """Always returns a fixed CapabilityDecision, ignoring the message -
    proves Core dispatches purely on the router's decision, not on any
    routing logic of its own."""

    def __init__(self, capability: Capability, tool_name: str | None = None):
        self.capability = capability
        self.tool_name = tool_name
        self.last_message = None

    def route(self, message):
        self.last_message = message
        return CapabilityDecision(capability=self.capability, tool_name=self.tool_name)


def test_core_dispatches_direct_when_the_capability_router_says_direct():
    """Even a message that would normally trigger SEARCH (via SearchRouter's
    own keyword matching) goes DIRECT if that's what the injected
    CapabilityRouter decides - proving Core no longer has its own
    DIRECT/SEARCH logic, only dispatch on the decision it's handed."""
    capability_router = FakeCapabilityRouter(Capability.DIRECT)
    jarvis, provider, _ = make_search_core(
        reply="direct reply", capability_router=capability_router
    )

    reply, _, sources = jarvis.handle_chat("what's the latest news on AI?")

    assert reply == "direct reply"
    assert provider.last_message is not None
    assert sources == []
    assert capability_router.last_message == "what's the latest news on AI?"


def test_core_dispatches_search_when_the_capability_router_says_search():
    """Even a message that would normally stay DIRECT goes through the
    SEARCH pipeline if that's what the injected CapabilityRouter
    decides."""
    capability_router = FakeCapabilityRouter(Capability.SEARCH)
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, provider, _ = make_search_core(
        answer_generator=answer_generator, capability_router=capability_router
    )

    reply, _, _ = jarvis.handle_chat("what is quantum entanglement?")

    assert reply == "grounded answer"
    assert provider.last_message is None  # DIRECT path never ran
    assert answer_generator.last_response is not None
    assert capability_router.last_message == "what is quantum entanglement?"


def test_capability_router_is_constructor_injectable():
    capability_router = FakeCapabilityRouter(Capability.DIRECT)

    jarvis = JarvisCore(
        provider=FakeProvider("direct reply"),
        memory=SQLiteMemoryStore(":memory:"),
        capability_router=capability_router,
    )

    jarvis.handle_chat("hello")

    assert capability_router.last_message == "hello"


def test_default_capability_router_still_uses_the_injected_search_router():
    """Backward-compatibility guarantee: injecting a `router=` (as every
    pre-M6.13 test does) still drives the DIRECT/SEARCH decision, because
    the default CapabilityRouter wraps whatever `router` Core was given -
    Core doesn't need a `capability_router` override just to keep using a
    custom SearchRouter."""
    router = RecordingRouter()
    jarvis, provider, _ = make_search_core(reply="direct reply", router=router)

    jarvis.handle_chat("what is quantum entanglement?")

    assert router.last_message == "what is quantum entanglement?"
    assert provider.last_message is not None  # DIRECT path actually ran


# --- M6.14: TOOL capability dispatch -------------------------------------------


class EchoTool(Tool):
    """Fake READ tool: automatic, no confirmation needed."""

    name = "echo"
    description = "Echoes back a fixed message."
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(message="echoed")


class AddTool(Tool):
    """A second, differently-named fake tool - proves Core's TOOL dispatch
    is generic (no `if tool_name == "echo"` branch) by working identically
    for an unrelated tool."""

    name = "add"
    description = "Pretends to add two numbers."
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(data=42)


class ConfirmTool(Tool):
    """Fake CONFIRM tool - proves Core surfaces the Registry's existing
    permission enforcement rather than bypassing or reimplementing it."""

    name = "confirm_me"
    description = "Requires confirmation before running."
    permission = PermissionLevel.CONFIRM

    def execute(self) -> ToolResult:
        return ToolResult.success(message="confirmed and ran")


def make_tool_core(
    capability_router,
    tools: ToolRegistry | None = None,
    tool_call_builder: ToolCallBuilder | None = None,
) -> JarvisCore:
    registry = tools or ToolRegistry()
    return JarvisCore(
        provider=FakeProvider("direct reply"),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=capability_router,
        tool_call_builder=tool_call_builder,
    )


def test_core_dispatches_tool_based_on_the_injected_capability_decision():
    registry = ToolRegistry()
    registry.register(EchoTool())
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="echo")
    jarvis = make_tool_core(capability_router, tools=registry)

    reply, _, sources = jarvis.handle_chat("use tool echo")

    assert reply == "echoed"
    assert sources == []


def test_core_tool_dispatch_is_generic_across_different_registered_tools():
    """No `if tool_name == "echo"` branch in Core: two differently-named
    tools both dispatch correctly through the exact same code path."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(AddTool())

    echo_router = FakeCapabilityRouter(Capability.TOOL, tool_name="echo")
    add_router = FakeCapabilityRouter(Capability.TOOL, tool_name="add")

    echo_reply, _, _ = make_tool_core(echo_router, tools=registry).handle_chat(
        "use tool echo"
    )
    add_reply, _, _ = make_tool_core(add_router, tools=registry).handle_chat(
        "use tool add"
    )

    assert echo_reply == "echoed"
    assert add_reply == "42"


def test_core_tool_dispatch_reports_unknown_tool_via_existing_registry_error():
    """An unrecognized tool_name needs no special-casing in Core - the
    Registry's own existing "Unknown tool" error surfaces as the reply."""
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="does_not_exist")
    jarvis = make_tool_core(capability_router)

    reply, _, sources = jarvis.handle_chat("use tool does_not_exist")

    assert "does_not_exist" in reply
    assert sources == []


def test_core_tool_dispatch_with_no_identified_tool_returns_explicit_message():
    """CapabilityDecision(TOOL, tool_name=None) - the CapabilityRouter
    recognized an explicit tool request but couldn't name one - is handled
    as an explicit, typed non-execution outcome, not a crash or a guess."""
    capability_router = FakeCapabilityRouter(Capability.TOOL)
    jarvis = make_tool_core(capability_router)

    reply, _, sources = jarvis.handle_chat("use tool")

    assert reply
    assert sources == []


def test_core_tool_dispatch_respects_confirm_permission():
    """Core doesn't bypass the Registry's permission enforcement: a
    CONFIRM-level tool returns the confirmation-required outcome as the
    reply, and is never actually run without confirmation."""
    registry = ToolRegistry()
    registry.register(ConfirmTool())
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="confirm_me")
    jarvis = make_tool_core(capability_router, tools=registry)

    reply, _, _ = jarvis.handle_chat("use tool confirm_me")

    assert "confirm" in reply.lower()
    assert reply != "confirmed and ran"  # the tool itself never actually ran


def test_core_tool_dispatch_never_produces_sources_and_persists_reply_to_memory():
    registry = ToolRegistry()
    registry.register(EchoTool())
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="echo")
    jarvis = make_tool_core(capability_router, tools=registry)

    _, conversation_id, sources = jarvis.handle_chat("use tool echo")

    assert sources == []
    messages = jarvis.memory.get_messages(conversation_id)
    assert [m.content for m in messages] == ["use tool echo", "echoed"]


# --- M6.15: hybrid capability routing integration ------------------------------


class FailingCapabilityRouter(CapabilityRouter):
    """Raises CapabilityDecisionError directly - isolates Core's own
    translation of that error into ProviderError, the same way
    FakeAnswerGenerator(error=...) isolates _handle_search's translation
    of SearchAnswerError."""

    def route(self, message):
        raise CapabilityDecisionError("capability decision provider is down")


def test_core_translates_capability_decision_error_into_provider_error():
    jarvis = make_tool_core(FailingCapabilityRouter())

    with pytest.raises(ProviderError, match="capability decision provider is down"):
        jarvis.handle_chat("use tool")


class FakeCapabilityDecisionProviderForCore(CapabilityDecisionProvider):
    """Returns a fixed CapabilityDecision - has no reference to, and no
    way to reach, any ToolRegistry - proving structurally that the
    decision provider itself cannot execute a tool."""

    def __init__(self, decision: CapabilityDecision):
        self.decision = decision
        self.last_message = None

    def decide(self, message):
        self.last_message = message
        return self.decision


def test_hybrid_ambiguous_tool_decision_reaches_toolregistry_and_executes():
    """End-to-end M6.15 TOOL safety: the AI decision provider only ever
    returns a CapabilityDecision; JarvisCore is what calls
    ToolRegistry.execute(...) - the exact same _handle_tool path a
    deterministic TOOL decision already uses."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    decision_provider = FakeCapabilityDecisionProviderForCore(
        CapabilityDecision(capability=Capability.TOOL, tool_name="echo")
    )
    capability_router = HybridCapabilityRouter(decision_provider=decision_provider)
    jarvis = make_tool_core(capability_router, tools=registry)

    reply, _, sources = jarvis.handle_chat("use tool")

    assert reply == "echoed"
    assert sources == []
    assert decision_provider.last_message == "use tool"


def test_hybrid_ambiguous_direct_decision_never_touches_toolregistry():
    decision_provider = FakeCapabilityDecisionProviderForCore(
        CapabilityDecision(capability=Capability.DIRECT)
    )
    capability_router = HybridCapabilityRouter(decision_provider=decision_provider)
    jarvis, provider, _ = make_search_core(
        reply="a direct answer", capability_router=capability_router
    )

    reply, _, sources = jarvis.handle_chat("use tool")

    assert reply == "a direct answer"
    assert sources == []


def test_hybrid_unknown_tool_named_by_ai_decision_uses_existing_registry_error():
    """Gemini naming an unregistered tool needs no special-casing in Core
    - ToolRegistry's own existing "Unknown tool" error surfaces exactly as
    it does for a deterministic TOOL decision (M6.14)."""
    decision_provider = FakeCapabilityDecisionProviderForCore(
        CapabilityDecision(capability=Capability.TOOL, tool_name="does_not_exist")
    )
    capability_router = HybridCapabilityRouter(decision_provider=decision_provider)
    jarvis = make_tool_core(capability_router)

    reply, _, sources = jarvis.handle_chat("use tool")

    assert "does_not_exist" in reply
    assert sources == []


def test_hybrid_router_default_deterministic_fast_path_never_calls_ai_decision_provider():
    """Reconfirms, at the full JarvisCore level, that obvious messages
    never reach the AI decision provider even when Core is wired with a
    HybridCapabilityRouter."""
    decision_provider = FakeCapabilityDecisionProviderForCore(
        CapabilityDecision(capability=Capability.TOOL, tool_name="should_not_be_used")
    )
    capability_router = HybridCapabilityRouter(decision_provider=decision_provider)
    jarvis, provider, _ = make_search_core(
        reply="direct reply", capability_router=capability_router
    )

    reply, _, _ = jarvis.handle_chat("what is quantum entanglement?")

    assert reply == "direct reply"
    assert decision_provider.last_message is None  # never called


# --- M6.16: natural-language tool selection ------------------------------------


class MessageEchoTool(Tool):
    """Fake READ tool with one required string parameter - distinct from
    the zero-parameter EchoTool above, used to exercise M6.16's parameter
    construction (deterministic and AI-backed) without disturbing any
    M6.14 test that relies on the zero-parameter one."""

    name = "echo"
    description = "Echoes back a message."
    permission = PermissionLevel.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="message", type="string", free_text=True)]

    def execute(self, message: str = "") -> ToolResult:
        return ToolResult.success(data=message)


class AddNumbersTool(Tool):
    """Fake READ tool with two required parameters - needs AI-backed
    construction even when the tool name is already known."""

    name = "add"
    description = "Adds two numbers."
    permission = PermissionLevel.READ
    # Prose arguments ("add 2 and 3"): the one kind of tool an explicit
    # command may hand to the AI ToolCallProvider.
    natural_language_arguments = True

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="a", type="number"),
            ToolParameter(name="b", type="number"),
        ]

    def execute(self, a: float = 0, b: float = 0) -> ToolResult:
        return ToolResult.success(data=a + b)


class FakeToolCallProviderForCore(ToolCallProvider):
    """Records whether/how it was called and returns a fixed ToolCall, or
    raises - has no reference to any ToolRegistry, proving structurally
    that it cannot execute a tool itself."""

    def __init__(self, tool_call=None, error=None):
        self.tool_call = tool_call
        self.error = error
        self.call_count = 0
        self.last_message = None
        self.last_candidate_tools = None

    def decide(self, message, candidate_tools):
        self.call_count += 1
        self.last_message = message
        self.last_candidate_tools = candidate_tools
        if self.error:
            raise self.error
        return self.tool_call


def test_core_dispatches_deterministic_tool_call_with_parameters():
    """An explicitly named tool invocation with leftover text fills the
    single required parameter deterministically - no AI call at all."""
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    call_provider = FakeToolCallProviderForCore()
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="echo")
    jarvis = make_tool_core(
        capability_router, tools=registry, tool_call_builder=ToolCallBuilder(call_provider)
    )

    reply, _, sources = jarvis.handle_chat("use tool echo hello world")

    assert reply == "hello world"
    assert sources == []
    assert call_provider.call_count == 0


def test_core_end_to_end_handles_the_name_before_tool_phrasing():
    """M6.16 routing fix, exercised through the real, unmocked
    CapabilityRouter and ToolCallBuilder: "use the echo tool to say
    hello" now reaches TOOL (via app/capability.py's
    _TOOL_MENTION_PATTERN, not just the original "use tool echo" syntax)
    and is filled deterministically - no AI call needed or made."""
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    call_provider = FakeToolCallProviderForCore()
    jarvis = JarvisCore(
        provider=FakeProvider("direct reply"),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        tool_call_builder=ToolCallBuilder(call_provider),
    )

    reply, _, sources = jarvis.handle_chat("use the echo tool to say hello")

    assert reply == "to say hello"
    assert sources == []
    assert call_provider.call_count == 0


def test_core_dispatches_natural_language_tool_call_via_ai():
    """decision.tool_name is None (an explicit-but-nameless TOOL request,
    or a future capability decision that leaves it unset) - the
    AI-backed ToolCallProvider selects a real registered tool and builds
    its parameters, and Core executes it through the ordinary
    ToolRegistry path. The fake provider has zero access to the
    registry, proving Gemini itself never executes anything."""
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    registry.register(AddNumbersTool())
    call_provider = FakeToolCallProviderForCore(
        tool_call=ToolCall(tool_name="echo", parameters={"message": "hello"})
    )
    capability_router = FakeCapabilityRouter(Capability.TOOL)
    jarvis = make_tool_core(
        capability_router, tools=registry, tool_call_builder=ToolCallBuilder(call_provider)
    )

    reply, _, sources = jarvis.handle_chat("say hello using the echo tool")

    assert reply == "hello"
    assert sources == []
    assert call_provider.call_count == 1
    assert {t.name for t in call_provider.last_candidate_tools} == {"echo", "add"}


def test_core_tool_call_incomplete_error_becomes_a_graceful_reply_not_a_502():
    """Missing required parameters is a normal, typed "not enough
    information" outcome - a plain reply, never a 502."""
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="echo")
    jarvis = make_tool_core(capability_router, tools=registry)

    reply, _, sources = jarvis.handle_chat("use tool echo")

    assert "message" in reply.lower() or "echo" in reply.lower()
    assert sources == []


def test_core_tool_call_error_becomes_provider_error():
    """A ToolCallProvider failure (or invalid output) is translated to
    ProviderError, the same convention CapabilityDecisionError/
    SearchAnswerError already use - never a silent guess, never exposed
    as a raw internal error."""
    registry = ToolRegistry()
    registry.register(AddNumbersTool())
    call_provider = FakeToolCallProviderForCore(error=ToolCallError("Gemini is down"))
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="add")
    jarvis = make_tool_core(
        capability_router, tools=registry, tool_call_builder=ToolCallBuilder(call_provider)
    )

    with pytest.raises(ProviderError, match="Gemini is down"):
        jarvis.handle_chat("use tool add 2 and 3")


def test_core_tool_dispatch_still_respects_permission_with_ai_constructed_parameters():
    """AI-constructed parameters don't bypass ToolRegistry's permission
    enforcement: a CONFIRM-level tool still requires confirmation and is
    never actually run without it."""

    class ConfirmWithParamTool(Tool):
        name = "confirm_add"
        description = "Adds two numbers, but requires confirmation."
        permission = PermissionLevel.CONFIRM
        natural_language_arguments = True

        @property
        def parameters(self):
            return [
                ToolParameter(name="a", type="number"),
                ToolParameter(name="b", type="number"),
            ]

        def execute(self, a: float = 0, b: float = 0) -> ToolResult:
            return ToolResult.success(message="ran for real")

    registry = ToolRegistry()
    registry.register(ConfirmWithParamTool())
    call_provider = FakeToolCallProviderForCore(
        tool_call=ToolCall(tool_name="confirm_add", parameters={"a": 2, "b": 3})
    )
    capability_router = FakeCapabilityRouter(Capability.TOOL, tool_name="confirm_add")
    jarvis = make_tool_core(
        capability_router, tools=registry, tool_call_builder=ToolCallBuilder(call_provider)
    )

    reply, _, _ = jarvis.handle_chat("use tool confirm_add 2 and 3")

    assert "confirm" in reply.lower()
    assert reply != "ran for real"  # the tool itself never actually ran


def test_core_tool_call_dispatch_is_generic_across_ai_selected_tools():
    """No `if tool_name == "echo"` branch in Core: two differently-named,
    differently-schemad tools both dispatch correctly through the exact
    same AI-backed code path."""
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    registry.register(AddNumbersTool())

    echo_provider = FakeToolCallProviderForCore(
        tool_call=ToolCall(tool_name="echo", parameters={"message": "hi"})
    )
    add_provider = FakeToolCallProviderForCore(
        tool_call=ToolCall(tool_name="add", parameters={"a": 1, "b": 2})
    )

    echo_reply, _, _ = make_tool_core(
        FakeCapabilityRouter(Capability.TOOL),
        tools=registry,
        tool_call_builder=ToolCallBuilder(echo_provider),
    ).handle_chat("say hi with echo")
    add_reply, _, _ = make_tool_core(
        FakeCapabilityRouter(Capability.TOOL),
        tools=registry,
        tool_call_builder=ToolCallBuilder(add_provider),
    ).handle_chat("add 1 and 2")

    assert echo_reply == "hi"
    assert add_reply == "3"


def test_core_tool_call_builder_is_constructor_injectable():
    call_provider = FakeToolCallProviderForCore(
        tool_call=ToolCall(tool_name="echo", parameters={"message": "hi"})
    )
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    jarvis = JarvisCore(
        provider=FakeProvider("direct reply"),
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL),
        tool_call_builder=ToolCallBuilder(call_provider),
    )

    jarvis.handle_chat("say hi with echo")

    assert call_provider.call_count == 1


def test_core_default_tool_call_builder_is_a_tool_call_builder():
    assert isinstance(JarvisCore().tool_call_builder, ToolCallBuilder)


def test_direct_behavior_is_unaffected_by_the_tool_decision_layer():
    jarvis, provider, _ = make_search_core(reply="a stable answer")

    reply, _, sources = jarvis.handle_chat("what is quantum entanglement?")

    assert reply == "a stable answer"
    assert sources == []


def test_search_behavior_is_unaffected_by_the_tool_decision_layer():
    answer_generator = FakeAnswerGenerator(answer="grounded answer")
    jarvis, _, _ = make_search_core(answer_generator=answer_generator)

    reply, _, sources = jarvis.handle_chat("what's the latest news on AI?")

    assert reply == "grounded answer"


# --- M14: wiring RequestRequirements into the real DIRECT/SEARCH/TOOL paths --
#
# ProviderRouter/AIProviderManager themselves are already covered in
# tests/test_provider_router.py and tests/test_ai_provider_manager.py; these
# tests only cover the NEW seam - that JarvisCore's DIRECT path and the
# SEARCH path (via GeminiSearchAnswerGenerator, tested more thoroughly in
# tests/test_search_answer.py) actually build a RequestRequirements from real
# execution context and route through it, without breaking anything for a
# plain (non-manager) AIProvider.


def test_direct_path_routes_through_the_manager_and_records_a_decision():
    manager = AIProviderManager([AIProviderEntry("only", FakeProvider("direct reply"))])
    jarvis = JarvisCore(provider=manager, memory=SQLiteMemoryStore(":memory:"))

    reply, _, _ = jarvis.handle_chat("hello there")

    assert reply == "direct reply"
    assert manager.last_routing_decision is not None
    assert manager.last_routing_decision.requirements.task_type is TaskType.GENERAL


def test_direct_path_classifies_coding_task_type_from_the_message():
    manager = AIProviderManager([AIProviderEntry("only", FakeProvider("ok"))])
    jarvis = JarvisCore(provider=manager, memory=SQLiteMemoryStore(":memory:"))

    jarvis.handle_chat("can you help me debug this function")

    assert manager.last_routing_decision.requirements.task_type is TaskType.CODING


def test_direct_path_long_context_flag_reflects_the_assembled_prompt_length():
    manager = AIProviderManager([AIProviderEntry("only", FakeProvider("ok"))])
    jarvis = JarvisCore(provider=manager, memory=SQLiteMemoryStore(":memory:"))
    # Pushes the assembled prompt (facts + history + message) well past
    # LONG_CONTEXT_CHAR_THRESHOLD - this is real execution context (the
    # actual payload size), not a keyword match.
    jarvis.save_memory("bio", "x" * 7000)

    jarvis.handle_chat("hi")

    assert manager.last_routing_decision.requirements.long_context is True


def test_direct_path_short_message_is_not_flagged_long_context():
    manager = AIProviderManager([AIProviderEntry("only", FakeProvider("ok"))])
    jarvis = JarvisCore(provider=manager, memory=SQLiteMemoryStore(":memory:"))

    jarvis.handle_chat("hi")

    assert manager.last_routing_decision.requirements.long_context is False


def test_direct_path_stays_backward_compatible_with_a_plain_provider():
    # A plain (non-manager) AIProvider - every pre-M14 injection pattern,
    # and every other test fake in this suite - must keep working exactly
    # as before, with no TypeError from an unexpected `requirements`
    # keyword argument reaching a provider that never asked for one.
    jarvis, provider = make_core("hi there")

    reply, _, _ = jarvis.handle_chat("hello")

    assert reply == "hi there"
    assert provider.last_message == "hello"


def test_tool_path_never_builds_or_routes_requirements():
    manager = AIProviderManager([AIProviderEntry("only", FakeProvider("unused"))])
    call_provider = FakeToolCallProviderForCore(
        tool_call=ToolCall(tool_name="echo", parameters={"message": "hi"})
    )
    registry = ToolRegistry()
    registry.register(MessageEchoTool())
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL),
        tool_call_builder=ToolCallBuilder(call_provider),
    )

    reply, _, _ = jarvis.handle_chat("say hi with echo")

    assert reply == "hi"
    assert call_provider.call_count == 1
    # An explicitly injected tool_call_builder (as here) never touches
    # `manager` at all, so no routing decision is ever recorded on it.
    # Since M17.1-E, Core's DEFAULT tool_call_builder does call through
    # `manager` when one is injected as `provider` (see the fallback tests
    # below) - but never with `requirements=`, so `last_routing_decision`
    # stays None either way (see AIProviderManager.generate_reply).
    assert manager.last_routing_decision is None


class FailableFakeProvider(AIProvider):
    """Like the module's own FakeProvider, but scriptable with a queue of
    errors to raise before it starts succeeding (mirrors
    tests/test_ai_provider_manager.py's FakeProvider) - needed here to
    simulate "gemini 429s, groq answers" sequences. Kept local to this
    section rather than changing FakeProvider itself, which many other
    tests in this file rely on for its exact simple (reply-only, no
    errors) signature."""

    def __init__(self, reply: str = "ok", errors: list[Exception] | None = None):
        self.reply = reply
        self._errors = list(errors) if errors is not None else []
        self.calls = 0

    def generate_reply(self, message: str) -> str:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self.reply


# --- M17.1-E: multi-provider fallback for tool-call construction ---------------
#
# Root cause of the M17.1-D live smoke-test bug: Core's default
# tool_call_builder built its GeminiToolCallProvider around a fresh,
# hardcoded GeminiProvider() - never `self.provider` - so a Gemini 429
# during tool-call construction (any tool needing more than one required
# parameter, e.g. a computer-control tool like `move_mouse`) had no
# fallback at all, unlike DIRECT (which already routes through
# AIProviderManager - see the M14 tests above). These tests exercise
# Core's DEFAULT wiring (no `tool_call_builder=` override) with a real
# AIProviderManager as `provider`, proving the fallback chain now covers
# TOOL too, without weakening permission enforcement or retrying an
# invalid tool call indefinitely.


class AddNumbersToolRequiringConfirmation(Tool):
    """Two required number parameters (forces AI-backed construction, like
    AddNumbersTool above) AND a non-automatic permission - lets one test
    exercise both the fallback path and permission enforcement at once."""

    name = "add"
    description = "Adds two numbers, but requires confirmation."
    permission = PermissionLevel.CONFIRM
    natural_language_arguments = True

    @property
    def parameters(self):
        return [
            ToolParameter(name="a", type="number"),
            ToolParameter(name="b", type="number"),
        ]

    def execute(self, a: float = 0, b: float = 0) -> ToolResult:
        return ToolResult.success(message="ran for real")


def test_tool_path_falls_back_across_providers_on_gemini_429():
    """The exact shape of the live bug: the first provider (gemini) 429s
    constructing the tool call; Core's default tool_call_builder must fall
    back to the next configured provider and still succeed, instead of
    surfacing a ProviderError to the user."""
    gemini = FailableFakeProvider(errors=[ProviderError("Gemini returned 429: quota exceeded")])
    groq = FailableFakeProvider(reply='{"tool_name": "add", "parameters": {"a": 2, "b": 3}}')
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("groq", groq)]
    )
    registry = ToolRegistry()
    registry.register(AddNumbersTool())
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="add"),
    )

    reply, _, _ = jarvis.handle_chat("use tool add 2 and 3")

    assert reply == "5"  # AddNumbersTool: a=2, b=3 -> 5, via the FALLBACK provider
    assert gemini.calls == 1
    assert groq.calls == 1


def test_tool_path_all_providers_failing_raises_a_clear_error():
    """No provider left to try must still be one clear, typed failure -
    never an unbounded retry/fallback loop, and never a silent success."""
    gemini = FailableFakeProvider(errors=[ProviderError("Gemini returned 429: quota exceeded")])
    groq = FailableFakeProvider(errors=[ProviderError("Groq returned 503: overloaded")])
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("groq", groq)],
        retry_count=0,
    )
    registry = ToolRegistry()
    registry.register(AddNumbersTool())
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="add"),
    )

    with pytest.raises(ProviderError, match="gemini"):
        jarvis.handle_chat("use tool add 2 and 3")


def test_tool_path_malformed_reply_moves_to_the_next_provider_and_the_tool_runs_exactly_once():
    """A provider that answers but with garbage is passed over (its reply is never used as a result), the next provider
    in the configured order constructs the call - and the tool itself runs exactly ONCE: asking a second provider for the
    tool call must never execute anything twice."""
    executions = []

    class CountingAddTool(AddNumbersTool):
        def execute(self, a: float = 0, b: float = 0) -> ToolResult:
            executions.append((a, b))
            return super().execute(a=a, b=b)

    gemini = FailableFakeProvider(errors=[ProviderError("Gemini returned 429: quota exceeded")])
    groq = FailableFakeProvider(reply="not json at all")
    third = FailableFakeProvider(reply='{"tool_name": "add", "parameters": {"a": 1, "b": 1}}')
    manager = AIProviderManager(
        [
            AIProviderEntry("gemini", gemini),
            AIProviderEntry("groq", groq),
            AIProviderEntry("third", third),
        ]
    )
    registry = ToolRegistry()
    registry.register(CountingAddTool())
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="add"),
    )

    reply = jarvis.handle_chat("use tool add 2 and 3")[0]

    assert "2" in reply  # 1 + 1, from the third provider's valid call
    assert (gemini.calls, groq.calls, third.calls) == (1, 1, 1)  # each asked once, in order
    assert executions == [(1, 1)]  # the tool ran exactly once


def test_tool_path_with_no_valid_reply_anywhere_is_a_clean_error_and_nothing_runs():
    """Every provider answers garbage: a clean ProviderError (from ToolCallError), each provider asked exactly once - no
    retry loop - and the tool never runs."""
    executions = []

    class CountingAddTool(AddNumbersTool):
        def execute(self, a: float = 0, b: float = 0) -> ToolResult:
            executions.append((a, b))
            return super().execute(a=a, b=b)

    first = FailableFakeProvider(reply="not json at all")
    second = FailableFakeProvider(reply="still not json")
    manager = AIProviderManager([AIProviderEntry("first", first), AIProviderEntry("second", second)])
    registry = ToolRegistry()
    registry.register(CountingAddTool())
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="add"),
    )

    with pytest.raises(ProviderError, match="Invalid JSON"):
        jarvis.handle_chat("use tool add 2 and 3")

    assert (first.calls, second.calls) == (1, 1)
    assert executions == []


def test_tool_path_permission_enforcement_is_authoritative_after_fallback():
    """A CONFIRM-level tool still requires confirmation and is never
    actually run, even when its parameters were constructed by the
    FALLBACK provider (not the first-choice one) - fallback never grants
    a tool extra permissions or bypasses ToolRegistry's own check."""
    gemini = FailableFakeProvider(errors=[ProviderError("Gemini returned 429: quota exceeded")])
    groq = FailableFakeProvider(reply='{"tool_name": "add", "parameters": {"a": 2, "b": 3}}')
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("groq", groq)]
    )
    registry = ToolRegistry()
    registry.register(AddNumbersToolRequiringConfirmation())
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=FakeCapabilityRouter(Capability.TOOL, tool_name="add"),
    )

    reply, _, _ = jarvis.handle_chat("use tool add 2 and 3")

    assert "confirm" in reply.lower()
    assert reply != "ran for real"  # the tool itself never actually ran
    assert groq.calls == 1  # fallback did construct the call...
    # ...but ToolRegistry.execute still gated it on confirmation, exactly
    # as it does for the first-choice provider (see the pre-existing
    # test_core_tool_dispatch_still_respects_permission_with_ai_constructed_parameters).


def test_direct_path_still_falls_back_the_same_way_tool_now_does():
    """Regression guard: DIRECT's pre-existing fallback behavior (M12/M14)
    is completely unaffected by M17.1-E - it never went through
    GeminiToolCallProvider and still doesn't."""
    gemini = FailableFakeProvider(errors=[ProviderError("Gemini returned 429: quota exceeded")])
    groq = FailableFakeProvider(reply="direct fallback reply")
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("groq", groq)]
    )
    jarvis = JarvisCore(provider=manager, memory=SQLiteMemoryStore(":memory:"))

    reply, _, _ = jarvis.handle_chat("hello there")

    assert reply == "direct fallback reply"
    assert gemini.calls == 1
    assert groq.calls == 1


# --- Explicit multi-tool sequence execution ------------------------------------
#
# These tests use JarvisCore's REAL (default) DeterministicCapabilityRouter and
# the real ExplicitToolCommandPlanner (app/tools/command.py) - not
# FakeCapabilityRouter - since the sequence is understood and validated from
# the message itself; every other M6.14/M6.16 TOOL test in this file fixes
# the decision directly because it isn't testing that. The parser's own edge
# cases live in tests/test_tool_command.py, the /chat runtime path in
# tests/test_tool_command_runtime.py.


class PingTool(Tool):
    """A third, distinct zero-parameter READ tool - proves a 3-step
    sequence is generic across tools, not special-cased for 2."""

    name = "ping"
    description = "Returns pong."
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(message="pong")


class RecordingTool(Tool):
    """Zero-parameter READ tool that records every execution - used to
    prove a step AFTER a stopped/rejected sequence never actually runs,
    not merely that the reply text looks right."""

    name = "record"
    description = "Records that it ran."
    permission = PermissionLevel.READ

    def __init__(self):
        self.runs = 0

    def execute(self) -> ToolResult:
        self.runs += 1
        return ToolResult.success(message="recorded")


class SubtractNumbersTool(Tool):
    """Second AI-backed (two required number parameters) tool, distinct
    from AddNumbersTool above - used to prove one step's clause never
    leaks into another step's parameter construction."""

    name = "subtract"
    description = "Subtracts two numbers."
    permission = PermissionLevel.READ
    natural_language_arguments = True

    @property
    def parameters(self):
        return [
            ToolParameter(name="a", type="number"),
            ToolParameter(name="b", type="number"),
        ]

    def execute(self, a: float = 0, b: float = 0) -> ToolResult:
        return ToolResult.success(data=a - b)


class RecordingToolCallProviderByName(ToolCallProvider):
    """Records the EXACT message/candidate-tool-names of every `decide`
    call (one per sequence step) and returns a fixed ToolCall looked up
    by the offered tool's own name - lets a test assert precisely what
    text each step's construction saw, proving no cross-step leakage."""

    def __init__(self, results: dict[str, ToolCall]):
        self.results = results
        self.calls: list[tuple[str, list[str]]] = []

    def decide(self, message: str, candidate_tools: list[Tool]) -> ToolCall:
        self.calls.append((message, [t.name for t in candidate_tools]))
        return self.results[candidate_tools[0].name]


def make_sequence_core(tools: ToolRegistry, tool_call_builder=None) -> JarvisCore:
    return JarvisCore(
        provider=FakeProvider("unused"),
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        tool_call_builder=tool_call_builder,
    )


def test_two_explicit_tools_execute_in_order():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(AddTool())
    jarvis = make_sequence_core(registry)

    reply, _, sources = jarvis.handle_chat("use tool echo, then use tool add")

    assert reply == "1. echoed\n2. 42"
    assert sources == []


def test_three_explicit_tools_execute_in_order():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(AddTool())
    registry.register(PingTool())
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat(
        "use tool echo, then use tool add, then use tool ping"
    )

    assert reply == "1. echoed\n2. 42\n3. pong"


def test_a_sequence_with_an_unauthorized_step_is_rejected_and_nothing_executes():
    """Each step is independently permission-checked (ToolRegistry's own
    gate, via check_permission) BEFORE the first step runs: a CONFIRM-
    gated step anywhere rejects the WHOLE sequence, so not even the
    earlier, individually-permitted RecordingTool step executes."""
    registry = ToolRegistry()
    recorder = RecordingTool()
    registry.register(recorder)
    registry.register(ConfirmTool())
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat(
        "use tool record, then use tool confirm_me, then use tool record"
    )

    assert "Step 2 (confirm_me)" in reply
    assert "requires confirmation" in reply
    assert "Nothing was executed" in reply
    assert recorder.runs == 0  # neither the first nor the third step ran


def test_malformed_sequence_is_rejected_before_anything_executes():
    """A trailing "...then use tool" with nothing named is malformed -
    NOTHING in the sequence runs, not even the well-formed first step."""
    registry = ToolRegistry()
    recorder = RecordingTool()
    registry.register(recorder)
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat("use tool record, then use tool")

    assert "step 2" in reply.lower()
    assert recorder.runs == 0


def test_a_sequence_naming_an_unregistered_tool_is_rejected_before_any_step_runs():
    """An unknown tool anywhere in the sequence rejects the whole thing
    at validation time - the well-formed steps around it never execute
    (previously step 1 ran and only then was step 2 found to be unknown)."""
    registry = ToolRegistry()
    recorder = RecordingTool()
    registry.register(recorder)
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat(
        "use tool record, then use tool nonexistent, then use tool record"
    )

    assert "Unknown tool: 'nonexistent'" in reply
    assert "Nothing was executed" in reply
    assert recorder.runs == 0


def test_single_tool_call_behavior_is_unchanged_by_sequence_support():
    """Regression guard: an ordinary single "use tool X" message (no
    "then") behaves exactly as before sequence support existed."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    jarvis = make_sequence_core(registry)

    reply, _, sources = jarvis.handle_chat("use tool echo")

    assert reply == "echoed"
    assert sources == []


def test_sequence_steps_never_leak_arguments_into_each_other():
    """The exact bug this fix closes: an AI-backed step's clause must be
    ONLY that step's own wording - never the full message, and never
    another step's clause. Uses two DISTINCT AI-backed (multi-required-
    number-parameter) tools so a shared fixed ToolCall couldn't mask
    leakage - each result is only correct if the right message reached
    the right tool."""
    registry = ToolRegistry()
    registry.register(AddNumbersTool())
    registry.register(SubtractNumbersTool())
    call_provider = RecordingToolCallProviderByName(
        {
            "add": ToolCall(tool_name="add", parameters={"a": 1, "b": 2}),
            "subtract": ToolCall(tool_name="subtract", parameters={"a": 9, "b": 3}),
        }
    )
    jarvis = make_sequence_core(registry, tool_call_builder=ToolCallBuilder(call_provider))

    reply, _, _ = jarvis.handle_chat("use tool add 1 2, then use tool subtract 9 3")

    assert reply == "1. 3\n2. 6"
    assert call_provider.calls == [
        ("use tool add 1 2", ["add"]),
        ("use tool subtract 9 3", ["subtract"]),
    ]
    # Neither call's message contains the OTHER step's own clause text.
    assert "subtract" not in call_provider.calls[0][0]
    assert "add 1 2" not in call_provider.calls[1][0]


class SelectivelyFailingToolCallProvider(ToolCallProvider):
    """Returns a fixed ToolCall for every tool EXCEPT `fails_for`, for
    which it raises ToolCallError - simulates a genuine AI construction
    failure (e.g. every configured provider 429ing/erroring) for exactly
    one step, with no real provider or network call needed."""

    def __init__(self, results: dict[str, ToolCall], fails_for: str):
        self.results = results
        self.fails_for = fails_for
        self.calls: list[str] = []

    def decide(self, message: str, candidate_tools: list[Tool]) -> ToolCall:
        name = candidate_tools[0].name
        self.calls.append(name)
        if name == self.fails_for:
            raise ToolCallError(
                f"Failed to construct a tool call: All AI providers failed for '{name}'"
            )
        return self.results[name]


def test_sequence_does_not_stop_after_the_first_successful_tool():
    """Explicit regression guard for the exact defect class under
    investigation: a SUCCESS result must never be treated as a reason to
    stop the sequence - only a non-SUCCESS result or a construction
    failure does."""
    registry = ToolRegistry()
    recorder = RecordingTool()
    registry.register(recorder)
    registry.register(PingTool())
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat("use tool record, then use tool ping")

    assert recorder.runs == 1
    assert reply == "1. recorded\n2. pong"


def test_the_second_tools_execute_method_is_actually_invoked():
    """Direct proof the SECOND tool's own execute() runs - not merely
    that ToolCallBuilder produced two ToolCall objects. RecordingTool
    increments a real counter inside its own execute(), so this can only
    pass if execution genuinely reached it."""
    registry = ToolRegistry()
    registry.register(PingTool())
    recorder = RecordingTool()
    registry.register(recorder)
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat("use tool ping, then use tool record")

    assert recorder.runs == 1
    assert reply == "1. pong\n2. recorded"


def test_first_step_execution_failure_stops_the_sequence_and_is_reported_normally():
    """"If the first tool fails" (item 4): when it FAILS TO EXECUTE
    cleanly (here, an unknown tool name - ToolRegistry's own existing
    typed outcome), that's already a normal, safely-typed reply, never a
    502 - and the sequence correctly stops before any later step runs."""
    registry = ToolRegistry()
    recorder = RecordingTool()
    registry.register(recorder)
    jarvis = make_sequence_core(registry)

    reply, _, _ = jarvis.handle_chat("use tool nonexistent, then use tool record")

    assert "Unknown tool: 'nonexistent'" in reply
    assert recorder.runs == 0


def test_first_step_construction_failure_still_raises_provider_error():
    """"If the first tool fails" (item 4), the OTHER way it can fail:
    tool-CALL CONSTRUCTION itself fails (e.g. every AI provider erroring)
    on the very FIRST step. Documents and tests the INTENTIONAL current
    behavior - nothing has executed yet, so this still raises
    ProviderError exactly like a single-tool request already does; there
    is no prior progress to lose (and, since construction is now phase 1 of
    the whole sequence, there never is - see the test after this one)."""
    registry = ToolRegistry()
    registry.register(AddNumbersTool())
    registry.register(SubtractNumbersTool())
    call_provider = SelectivelyFailingToolCallProvider(
        results={"subtract": ToolCall(tool_name="subtract", parameters={"a": 9, "b": 3})},
        fails_for="add",
    )
    jarvis = make_sequence_core(registry, tool_call_builder=ToolCallBuilder(call_provider))

    with pytest.raises(ProviderError):
        jarvis.handle_chat("use tool add 1 2, then use tool subtract 9 3")

    assert call_provider.calls == ["add"]  # subtract's step was never attempted


def test_a_later_steps_construction_failure_now_prevents_every_earlier_step_from_running():
    """Supersedes the M17.1-G behaviour. There, step 1 executed and a
    later step's tool-call construction failure was reported alongside it
    (to avoid a bare 502 hiding a completed side effect). Construction is
    now PHASE 1 of the whole sequence: it fails before step 1 executes, so
    the ProviderError convention (502) loses nothing - and, crucially, no
    earlier step ever runs for a sequence that can't be completed."""

    class RecordingAddTool(AddNumbersTool):
        runs = 0

        def execute(self, a: float = 0, b: float = 0) -> ToolResult:
            type(self).runs += 1
            return super().execute(a=a, b=b)

    registry = ToolRegistry()
    registry.register(RecordingAddTool())
    registry.register(SubtractNumbersTool())
    call_provider = SelectivelyFailingToolCallProvider(
        results={"add": ToolCall(tool_name="add", parameters={"a": 1, "b": 2})},
        fails_for="subtract",
    )
    jarvis = make_sequence_core(registry, tool_call_builder=ToolCallBuilder(call_provider))

    with pytest.raises(ProviderError):
        jarvis.handle_chat("use tool add 1 2, then use tool subtract 9 3")

    assert call_provider.calls == ["add", "subtract"]  # both were constructed...
    assert RecordingAddTool.runs == 0  # ...but nothing executed


def test_computer_tools_remain_unreachable_from_m15_even_as_sequence_step_names():
    """A sequence naming tools that only exist on core.computer_tools must
    still be "Unknown tool" through core.tools/JobEngine's own registry -
    M17.1-F adds no new path into M15's tool candidate pool."""
    from app.computer.defaults import create_computer_tools_registry

    core_tools = ToolRegistry()  # the same shape as M15's real core.tools
    jarvis = make_sequence_core(core_tools)
    computer_tools = create_computer_tools_registry()

    assert core_tools.has("move_mouse") is False
    assert core_tools.has("click_mouse") is False
    assert computer_tools.has("move_mouse") is True

    reply, _, _ = jarvis.handle_chat(
        "use tool move_mouse 1 1, then use tool click_mouse 1 1"
    )

    assert "Unknown tool: 'move_mouse'" in reply


def test_search_path_routes_through_the_manager_via_gemini_search_answer_generator():
    manager = AIProviderManager(
        [
            AIProviderEntry(
                "only",
                FakeProvider("Grounded answer. See https://python.org for details."),
            )
        ]
    )
    tools = ToolRegistry()
    tools.register(
        WebSearchTool(
            provider=FakeSearchProvider(
                results=[
                    SearchResult(
                        title="Python.org",
                        url="https://python.org",
                        snippet="Official Python site",
                        source="searxng",
                        query="latest news on AI",
                    )
                ]
            )
        )
    )
    jarvis = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=tools,
        answer_generator=GeminiSearchAnswerGenerator(provider=manager),
    )

    reply, _, sources = jarvis.handle_chat("what's the latest news on AI?")

    assert "python.org" in reply
    assert manager.last_routing_decision is not None
    assert manager.last_routing_decision.requirements.search_synthesis is True
    assert manager.last_routing_decision.requirements.task_type is TaskType.SEARCH_SYNTHESIS
    # SEARCH_SYNTHESIS is a soft preference (see app/providers/router.py) -
    # the single registered provider is still eligible, never excluded.
    assert manager.last_routing_decision.excluded == []


# --- M15: TASK capability dispatch ------------------------------------------


def test_handle_task_creates_a_job_via_the_create_job_tool():
    from app.orchestration.sqlite_store import SQLiteJobStore
    from app.orchestration.tools import CreateJobTool

    provider = FakeProvider("should not be used")
    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    tools.register(CreateJobTool(store=job_store))
    jarvis = JarvisCore(provider=provider, memory=SQLiteMemoryStore(":memory:"), tools=tools)

    reply, _, sources = jarvis.handle_chat("Prepare everything for tomorrow's robotics meeting.")

    assert sources == []
    jobs = job_store.list()
    assert len(jobs) == 1
    assert jobs[0].goal == "Prepare everything for tomorrow's robotics meeting."
    assert "task" in reply.lower() or jobs[0].id in reply
    # DIRECT's own provider is never consulted for a TASK request - job
    # creation/planning is entirely separate from the chat provider call.
    assert provider.last_message is None


def test_handle_task_uses_explicit_invocation_goal_text():
    from app.orchestration.sqlite_store import SQLiteJobStore
    from app.orchestration.tools import CreateJobTool

    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    tools.register(CreateJobTool(store=job_store))
    jarvis, _ = make_core()
    jarvis.tools = tools

    jarvis.handle_chat("create a goal: organize my downloads folder")

    assert job_store.list()[0].goal == "organize my downloads folder"


def test_handle_task_with_no_create_job_tool_registered_returns_plain_reply():
    jarvis, _ = make_core()  # default empty ToolRegistry, no create_job

    reply, conversation_id, sources = jarvis.handle_chat("Organize my Downloads folder.")

    assert sources == []
    assert "create_job" in reply or "Unknown tool" in reply
    # Still persisted as a normal conversation turn, same as any other reply.
    history = jarvis.memory.get_messages(conversation_id)
    assert history[-1].content == reply


def test_handle_task_never_blocks_on_planning_or_execution():
    """TASK only ever creates a PENDING job and returns - planning/
    execution is JobEngine's job, never JarvisCore's."""
    from app.orchestration.base import JobStatus
    from app.orchestration.sqlite_store import SQLiteJobStore
    from app.orchestration.tools import CreateJobTool

    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    tools.register(CreateJobTool(store=job_store))
    jarvis, _ = make_core()
    jarvis.tools = tools

    jarvis.handle_chat("plan a birthday party for next week")

    assert job_store.list()[0].status == JobStatus.PENDING


def test_simple_direct_question_is_unaffected_by_task_capability_existing():
    jarvis, provider = make_core("Tokyo.")

    reply, _, sources = jarvis.handle_chat("What is the capital of Japan?")

    assert reply == "Tokyo."
    assert sources == []


# --- M15 fixes: handle_chat_with_job_id, TASK_STATUS, CONFIRMATION ---------


def _orchestration_core(reply: str = "direct reply", provider: FakeProvider | None = None):
    """A JarvisCore wired with real orchestration tools/store (in-memory),
    for end-to-end coverage of TASK/TASK_STATUS/CONFIRMATION through
    handle_chat - not just each tool in isolation."""
    from app.orchestration.sqlite_store import SQLiteJobStore
    from app.orchestration.tools import (
        CancelJobTool,
        ConfirmJobTool,
        CreateJobTool,
        GetJobTool,
        ListJobsTool,
        PauseJobTool,
        ResumeJobTool,
    )

    job_store = SQLiteJobStore(":memory:")
    tools = ToolRegistry()
    for tool in (
        CreateJobTool(store=job_store),
        ListJobsTool(store=job_store),
        GetJobTool(store=job_store),
        ConfirmJobTool(store=job_store),
        PauseJobTool(store=job_store),
        ResumeJobTool(store=job_store),
        CancelJobTool(store=job_store),
    ):
        tools.register(tool)
    provider = provider or FakeProvider(reply)
    jarvis = JarvisCore(provider=provider, memory=SQLiteMemoryStore(":memory:"), tools=tools)
    return jarvis, job_store, provider


def test_handle_chat_with_job_id_returns_the_created_jobs_id():
    jarvis, job_store, _ = _orchestration_core()

    reply, conversation_id, sources, job_id = jarvis.handle_chat_with_job_id(
        "prepare a report for the meeting"
    )

    assert sources == []
    assert job_id is not None
    assert job_store.get(job_id) is not None
    assert job_store.get(job_id).goal == "prepare a report for the meeting"


def test_handle_chat_with_job_id_is_none_for_non_task_messages():
    jarvis, _, _ = _orchestration_core("Tokyo.")

    reply, _, _, job_id = jarvis.handle_chat_with_job_id("What is the capital of Japan?")

    assert reply == "Tokyo."
    assert job_id is None


def test_handle_chat_still_returns_a_plain_three_tuple():
    """handle_chat's own signature/behavior is completely unchanged -
    still exactly 3 values, no job_id anywhere in it."""
    jarvis, _, _ = _orchestration_core()

    result = jarvis.handle_chat("prepare a report")

    assert len(result) == 3
    reply, conversation_id, sources = result
    assert isinstance(reply, str)


# --- TASK_STATUS: natural-language status queries ---------------------------


def test_status_query_with_no_tasks_at_all():
    jarvis, _, _ = _orchestration_core()

    reply, _, sources = jarvis.handle_chat("what's the status of my task?")

    assert sources == []
    assert "don't have any tasks" in reply.lower()


def test_is_that_finished_reports_a_completed_task_in_the_same_conversation():
    jarvis, job_store, _ = _orchestration_core()

    _, conversation_id, _ = jarvis.handle_chat("prepare a report")
    job = job_store.list()[0]
    job.status = JobStatus.COMPLETED
    job.final_result = "The report is ready."
    from datetime import datetime, timezone

    job.updated_at = datetime.now(timezone.utc).isoformat()
    job_store.save(job)

    reply, _, sources = jarvis.handle_chat("is that finished?", conversation_id=conversation_id)

    assert sources == []
    assert "finished" in reply.lower()
    assert "report is ready" in reply.lower() or "prepare a report" in reply.lower()


def test_status_query_reports_a_running_task():
    jarvis, job_store, _ = _orchestration_core()

    _, conversation_id, _ = jarvis.handle_chat("organize my downloads folder")
    job = job_store.list()[0]
    job.status = JobStatus.RUNNING
    job_store.save(job)

    reply, _, _ = jarvis.handle_chat(
        "what's the status of my task?", conversation_id=conversation_id
    )

    assert "in progress" in reply.lower() or "running" in reply.lower()


def test_status_query_reports_a_failed_task_with_its_reason():
    jarvis, job_store, _ = _orchestration_core()

    _, conversation_id, _ = jarvis.handle_chat("plan a party")
    job = job_store.list()[0]
    job.status = JobStatus.FAILED
    job.error = "no venue tool available"
    job_store.save(job)

    reply, _, _ = jarvis.handle_chat("is that finished?", conversation_id=conversation_id)

    assert "failed" in reply.lower()
    assert "no venue tool available" in reply


def test_status_query_scopes_to_its_own_conversation():
    jarvis, job_store, _ = _orchestration_core()

    _, conv_a, _ = jarvis.handle_chat("prepare report A")
    job_a = job_store.list()[0]
    job_a.status = JobStatus.COMPLETED
    job_a.final_result = "Report A done."
    job_store.save(job_a)

    _, conv_b, _ = jarvis.handle_chat("prepare report B")
    job_b = next(j for j in job_store.list() if j.goal == "prepare report B")
    job_b.status = JobStatus.RUNNING
    job_store.save(job_b)

    reply_a, _, _ = jarvis.handle_chat("is that finished?", conversation_id=conv_a)
    reply_b, _, _ = jarvis.handle_chat("is that finished?", conversation_id=conv_b)

    assert "report a" in reply_a.lower()
    assert "report b" in reply_b.lower()


def test_show_me_my_running_tasks_lists_active_jobs_only():
    jarvis, job_store, _ = _orchestration_core()

    jarvis.tools.execute("create_job", params={"goal": "task one"})
    jarvis.tools.execute("create_job", params={"goal": "task two"})
    jobs_by_goal = {j.goal: j for j in job_store.list()}
    jobs_by_goal["task one"].status = JobStatus.RUNNING
    job_store.save(jobs_by_goal["task one"])
    jobs_by_goal["task two"].status = JobStatus.COMPLETED
    job_store.save(jobs_by_goal["task two"])

    reply, _, _ = jarvis.handle_chat("show me my running tasks")

    assert "task one" in reply.lower()
    assert "task two" not in reply.lower()


# --- CONFIRMATION: natural-language yes/no ----------------------------------


def test_yes_with_no_pending_confirmation_falls_back_to_direct():
    jarvis, _, provider = _orchestration_core("Sure, happy to help!")

    reply, _, sources = jarvis.handle_chat("yes")

    assert reply == "Sure, happy to help!"
    assert sources == []
    assert provider.last_message is not None  # the DIRECT provider path really ran


def test_yes_confirms_a_pending_task_in_the_same_conversation():
    jarvis, job_store, _ = _orchestration_core()

    _, conversation_id, _ = jarvis.handle_chat("prepare a report")
    job = job_store.list()[0]
    from app.orchestration.base import Plan, PlanStep

    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="send_email", permission="confirm")])
    job.status = JobStatus.AWAITING_CONFIRMATION
    job.pending_step_ids = ["s1"]
    job.confirmation_message = "Send the report by email?"
    job_store.save(job)

    reply, _, sources = jarvis.handle_chat("yes", conversation_id=conversation_id)

    assert sources == []
    fetched = job_store.get(job.id)
    assert fetched.status == JobStatus.RUNNING
    assert fetched.plan.get("s1").confirmed is True


def test_send_it_and_go_ahead_also_confirm():
    for phrase in ("send it", "go ahead"):
        jarvis, job_store, _ = _orchestration_core()
        _, conversation_id, _ = jarvis.handle_chat("prepare a report")
        job = job_store.list()[0]
        from app.orchestration.base import Plan, PlanStep

        job.plan = Plan(steps=[PlanStep(id="s1", tool_name="send_email", permission="confirm")])
        job.status = JobStatus.AWAITING_CONFIRMATION
        job.pending_step_ids = ["s1"]
        job.confirmation_message = "Send it?"
        job_store.save(job)

        jarvis.handle_chat(phrase, conversation_id=conversation_id)

        assert job_store.get(job.id).status == JobStatus.RUNNING, "failed for phrase: " + phrase


def test_no_declines_a_pending_confirmation():
    jarvis, job_store, _ = _orchestration_core()

    _, conversation_id, _ = jarvis.handle_chat("prepare a report")
    job = job_store.list()[0]
    from app.orchestration.base import Plan, PlanStep

    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="send_email", permission="confirm")])
    job.status = JobStatus.AWAITING_CONFIRMATION
    job.pending_step_ids = ["s1"]
    job.confirmation_message = "Send it?"
    job_store.save(job)

    jarvis.handle_chat("no", conversation_id=conversation_id)

    fetched = job_store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert "declined" in fetched.error.lower()


def test_ambiguous_confirmation_across_conversations_asks_which_one():
    jarvis, job_store, provider = _orchestration_core()
    from app.orchestration.base import Plan, PlanStep

    _, conv_a, _ = jarvis.handle_chat("prepare report A")
    _, conv_b, _ = jarvis.handle_chat("prepare report B")
    for job in job_store.list():
        job.plan = Plan(steps=[PlanStep(id="s1", tool_name="send_email", permission="confirm")])
        job.status = JobStatus.AWAITING_CONFIRMATION
        job.pending_step_ids = ["s1"]
        job.confirmation_message = "Send it?"
        job_store.save(job)

    # Ask "yes" from a THIRD, unrelated conversation - neither pending
    # job is linked to it, so both remain candidates and it must ask
    # rather than guess.
    reply, _, _ = jarvis.handle_chat("yes")

    assert "more than one" in reply.lower() or "which one" in reply.lower()
    for job in job_store.list():
        assert job.status == JobStatus.AWAITING_CONFIRMATION  # neither was touched
    assert provider.last_message is None  # never fell through to DIRECT either


def test_confirmation_scoped_to_conversation_resolves_unambiguously():
    """Even with two pending confirmations system-wide, replying "yes"
    FROM one of their own conversations resolves to that one directly -
    no disambiguation needed, since conversation-scoping narrows it to
    exactly one candidate."""
    jarvis, job_store, _ = _orchestration_core()
    from app.orchestration.base import Plan, PlanStep

    _, conv_a, _ = jarvis.handle_chat("prepare report A")
    _, conv_b, _ = jarvis.handle_chat("prepare report B")
    jobs_by_goal = {j.goal: j for j in job_store.list()}
    for job in jobs_by_goal.values():
        job.plan = Plan(steps=[PlanStep(id="s1", tool_name="send_email", permission="confirm")])
        job.status = JobStatus.AWAITING_CONFIRMATION
        job.pending_step_ids = ["s1"]
        job.confirmation_message = "Send it?"
        job_store.save(job)

    jarvis.handle_chat("yes", conversation_id=conv_a)

    assert job_store.get(jobs_by_goal["prepare report A"].id).status == JobStatus.RUNNING
    assert (
        job_store.get(jobs_by_goal["prepare report B"].id).status
        == JobStatus.AWAITING_CONFIRMATION
    )
