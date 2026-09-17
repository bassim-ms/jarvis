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
from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
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


def test_default_provider_is_gemini():
    assert isinstance(JarvisCore().provider, GeminiProvider)


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
        return [ToolParameter(name="message", type="string")]

    def execute(self, message: str = "") -> ToolResult:
        return ToolResult.success(data=message)


class AddNumbersTool(Tool):
    """Fake READ tool with two required parameters - needs AI-backed
    construction even when the tool name is already known."""

    name = "add"
    description = "Adds two numbers."
    permission = PermissionLevel.READ

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
