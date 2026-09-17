"""
Jarvis Core - the orchestrator for handling a chat request.

Core depends only on the AIProvider and MemoryStore interfaces, not on any
specific AI vendor or storage backend, so providers can be added or swapped
without touching this module. Since M6.9, it also depends on
SearchAnswerGenerator (deterministic/interface-level) and on WebSearchTool
by name specifically (see _handle_search), a deliberate, narrow exception
to "Core doesn't know about tool types": M6.9's whole point is that Core
now orchestrates a fixed SEARCH path, not that it becomes tool-agnostic
tool-calling. Since M6.12, it also depends on SearchQueryBuilder
(deterministic, see app/search/query_builder.py) to decide what text to
actually search once SEARCH has already been chosen.

Since M6.13, the DIRECT-vs-SEARCH decision itself is no longer Core's own
logic: Core asks an injected CapabilityRouter (app/capability.py) for a
CapabilityDecision and dispatches on it. `router`/`SearchRouter` is kept as
its own constructor parameter purely so the underlying deterministic
keyword matching stays independently injectable/testable (see
DeterministicCapabilityRouter, which delegates to it) - Core itself no
longer branches on SearchDecision directly.

Since M6.14, TOOL is a third capability: _handle_tool dispatches to
`self.tools` (the existing ToolRegistry, M5/M6.9) by `decision.tool_name`
alone, via `ToolRegistry.execute(...)` - Core never branches on which tool
it is, only on the generic ToolResult it gets back, the same "Registry is
the abstraction boundary" design as the SEARCH path already has with
WebSearchTool.
"""

from app.capability import (
    Capability,
    CapabilityDecision,
    CapabilityRouter,
    DeterministicCapabilityRouter,
)
from app.config import settings
from app.memory.base import MemoryStore, Message
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.search.answer import (
    GeminiSearchAnswerGenerator,
    SearchAnswerError,
    SearchAnswerGenerator,
    SearchSource,
)
from app.search.base import SearchProviderError
from app.search.query_builder import DeterministicSearchQueryBuilder, SearchQueryBuilder
from app.search.router import SearchRouter
from app.tools.base import ToolResult, ToolResultStatus
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class ConversationNotFoundError(RuntimeError):
    """Raised when a client supplies a conversation_id that doesn't exist."""


class JarvisCore:
    """Handles chat requests: builds AI context from conversation history and
    stored facts, delegates to an AI provider, then persists the exchange."""

    # Bounds for the conversation context passed to the SEARCH path's
    # answer generator (M6.10) and query builder (M6.12) - small and
    # fixed, so a search-answer prompt (or an enriched query) doesn't grow
    # with conversation length the way DIRECT's prompt can. Not applied to
    # DIRECT, which keeps its existing unbounded behavior.
    _SEARCH_CONTEXT_MAX_MESSAGES = 6
    _SEARCH_CONTEXT_MAX_CHARS = 800

    def __init__(
        self,
        provider: AIProvider | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        router: SearchRouter | None = None,
        answer_generator: SearchAnswerGenerator | None = None,
        query_builder: SearchQueryBuilder | None = None,
        capability_router: CapabilityRouter | None = None,
    ):
        self.provider = provider or GeminiProvider()
        self.memory = memory or SQLiteMemoryStore(settings.memory_db_path)
        # Empty by default - no real tools exist yet (M5 is architecture
        # only). AI tool-calling isn't wired up here either; Core just has
        # a place to reach the registry without knowing any tool's type.
        self.tools = tools or ToolRegistry()
        self.router = router or SearchRouter()
        # Reuses self.provider by default, so injecting a fake AIProvider
        # into Core (as tests do) covers the SEARCH path's answer
        # generation too, without a second independent GeminiProvider().
        self.answer_generator = answer_generator or GeminiSearchAnswerGenerator(
            self.provider
        )
        self.query_builder = query_builder or DeterministicSearchQueryBuilder()
        # M6.13: defaults to wrapping self.router, so an injected/default
        # `router` still drives the DIRECT/SEARCH decision unchanged (e.g.
        # a test's RecordingRouter keeps recording exactly as before) -
        # `capability_router` is a separate, optional override for
        # injecting a whole different CapabilityRouter (e.g. once a second
        # real capability exists and DIRECT/SEARCH is no longer the only
        # choice).
        self.capability_router = capability_router or DeterministicCapabilityRouter(
            search_router=self.router
        )

    def handle_chat(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, str, list[SearchSource]]:
        """Route a user message to the AI provider and return its reply.

        If conversation_id is None, a new conversation is started. If it's
        given but unknown, ConversationNotFoundError is raised rather than
        silently starting a new conversation under that ID. Otherwise the
        conversation's prior messages are loaded and combined with any
        stored user facts to give the provider context for this turn.

        CapabilityRouter (deterministic, no AI call - M6.13/M6.14) decides
        DIRECT vs SEARCH vs TOOL before that, not Core itself: DIRECT keeps
        the existing prompt -> provider.generate_reply path unchanged;
        SEARCH runs the message through WebSearchTool -> SearchResponse ->
        SearchAnswerGenerator, using the generated answer as the reply;
        TOOL dispatches to the existing ToolRegistry by name (see
        _handle_tool). Either way, the final reply is persisted exactly the
        same way. Both DIRECT and SEARCH reuse the same `history`/`facts`
        already loaded below - neither the SEARCH nor the TOOL path issues
        its own memory query.

        The third return value is the SearchAnswer's validated sources
        (M6.11) - always `[]` for DIRECT and TOOL, since neither produces a
        SearchAnswer.
        """
        if conversation_id is None:
            conversation_id = self.memory.create_conversation()
        elif not self.memory.conversation_exists(conversation_id):
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' does not exist."
            )

        history = self.memory.get_messages(conversation_id)
        facts = self.memory.get_facts()

        decision = self.capability_router.route(message)
        if decision.capability is Capability.SEARCH:
            reply, sources = self._handle_search(message, history, facts)
        elif decision.capability is Capability.TOOL:
            reply, sources = self._handle_tool(decision)
        else:
            prompt = self._build_prompt(message, history, facts)
            reply = self.provider.generate_reply(prompt)
            sources = []

        self.memory.add_message(conversation_id, "user", message)
        self.memory.add_message(conversation_id, "assistant", reply)

        return reply, conversation_id, sources

    def _handle_search(
        self, message: str, history: list[Message], facts: dict[str, str]
    ) -> tuple[str, list[SearchSource]]:
        """SEARCH path: SearchQueryBuilder -> WebSearchTool ->
        SearchResponse -> SearchAnswerGenerator, using SearchAnswer.answer
        as the reply and returning SearchAnswer.sources (M6.11) alongside
        it for the caller to surface as it sees fit.

        The CapabilityRouter has already decided SEARCH from `message`
        alone, before this method is even called - it never sees the
        bounded context built here or the enriched query built below.
        WebSearchTool.search(), by contrast, now (M6.12) receives whatever
        `self.query_builder.build()` returns: `message` unchanged for a
        self-contained SEARCH query (the common case), or `message` plus
        the previous user turn for a narrow, explicit set of
        context-dependent follow-ups ("what about that?", "the second
        one", ...) - see app/search/query_builder.py. Either way, the
        bounded context is computed once here and reused for both the
        query builder and the answer generator (M6.10), never rebuilt or
        widened for this new step.

        Failures from either step are raised as ProviderError, the same
        error type the DIRECT path already raises on an AI-provider
        failure, so callers (like /chat's existing error handling) need no
        changes to keep treating a failed reply as a clean 502.
        """
        web_search = self.tools.get("web_search")
        if not isinstance(web_search, WebSearchTool):
            raise ProviderError("web_search tool is not registered.")

        context = self._format_context(
            history, facts, max_messages=self._SEARCH_CONTEXT_MAX_MESSAGES
        )
        context = context[-self._SEARCH_CONTEXT_MAX_CHARS :]

        query = self.query_builder.build(message, context)

        try:
            response = web_search.search(query)
        except SearchProviderError as exc:
            raise ProviderError(f"Search failed: {exc}") from exc

        try:
            search_answer = self.answer_generator.generate(
                response, conversation_context=context
            )
        except SearchAnswerError as exc:
            raise ProviderError(f"Failed to generate a search answer: {exc}") from exc

        return search_answer.answer, search_answer.sources

    def _handle_tool(self, decision: CapabilityDecision) -> tuple[str, list[SearchSource]]:
        """TOOL path (M6.14): dispatch to the existing ToolRegistry
        (app/tools/registry.py) by `decision.tool_name` alone - Core never
        branches on which tool it is, never inspects a specific tool's
        parameters, and never attempts to extract them from the message
        (no NLP/AI tool selection exists yet; see app/capability.py).
        `sources` is always `[]` - TOOL never produces search evidence.

        `decision.tool_name` is `None` when the CapabilityRouter recognized
        an explicit tool-invocation phrasing but couldn't identify which
        tool (e.g. "use tool" with nothing named) - reported back as a
        clear, typed non-execution outcome rather than guessing a tool.

        An unregistered `tool_name` needs no special handling here either:
        `ToolRegistry.execute()` already returns a typed "Unknown tool"
        `ToolResult.error(...)` for it, same as for any other tool failure.
        Permission enforcement (CONFIRM/DANGEROUS -> CONFIRMATION_REQUIRED/
        DENIED) is likewise entirely the Registry's existing behavior - the
        `/chat` API has no confirmation round-trip yet, so those outcomes
        simply surface as the reply text, same as an error would.
        """
        if decision.tool_name is None:
            return "I couldn't tell which tool to use for that request.", []

        result = self.tools.execute(decision.tool_name)
        return self._format_tool_result(result), []

    @staticmethod
    def _format_tool_result(result: ToolResult) -> str:
        """Turn a generic ToolResult into reply text - Core inspects only
        the typed `status`/`message`/`data`, never which tool produced
        them."""
        if result.status is ToolResultStatus.SUCCESS:
            if result.message:
                return result.message
            if result.data is not None:
                return str(result.data)
            return "Done."
        return result.message

    def save_memory(self, key: str, value: str) -> None:
        """Programmatic hook to save a user fact. Manual for now; automatic
        memory extraction can call this later without changing Core."""
        self.memory.save_fact(key, value)

    def get_memories(self) -> dict[str, str]:
        """Programmatic hook to retrieve all stored user facts."""
        return self.memory.get_facts()

    @staticmethod
    def _format_context(
        history: list[Message],
        facts: dict[str, str],
        max_messages: int | None = None,
    ) -> str:
        """Format stored facts and conversation history into plain text -
        shared by `_build_prompt` (DIRECT, unbounded) and `_handle_search`
        (SEARCH, bounded to a small recent window via `max_messages`) so
        both build context the same way instead of duplicating this
        formatting. Returns "" when there's nothing to include."""
        parts = []
        if facts:
            facts_text = "\n".join(f"- {key}: {value}" for key, value in facts.items())
            parts.append(f"Known facts about the user:\n{facts_text}")
        if history:
            recent = history if max_messages is None else history[-max_messages:]
            history_text = "\n".join(f"{m.role}: {m.content}" for m in recent)
            parts.append(f"Conversation so far:\n{history_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(
        message: str, history: list[Message], facts: dict[str, str]
    ) -> str:
        """Combine stored facts and conversation history into context for
        the provider. With no facts or history, the raw message is sent
        unchanged, matching the original single-turn behavior."""
        context = JarvisCore._format_context(history, facts)
        if not context:
            return message
        return f"{context}\n\nUser: {message}"


core = JarvisCore()
