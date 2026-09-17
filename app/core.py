"""
Jarvis Core - the orchestrator for handling a chat request.

Core depends only on the AIProvider and MemoryStore interfaces, not on any
specific AI vendor or storage backend, so providers can be added or swapped
without touching this module. Since M6.9, it also depends on SearchRouter
and SearchAnswerGenerator - both deterministic/interface-level - and on
WebSearchTool by name specifically (see _handle_search), a deliberate,
narrow exception to "Core doesn't know about tool types": M6.9's whole
point is that Core now orchestrates a fixed SEARCH path, not that it
becomes tool-agnostic tool-calling.
"""

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
from app.search.router import SearchDecision, SearchRouter
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class ConversationNotFoundError(RuntimeError):
    """Raised when a client supplies a conversation_id that doesn't exist."""


class JarvisCore:
    """Handles chat requests: builds AI context from conversation history and
    stored facts, delegates to an AI provider, then persists the exchange."""

    # Bounds for the conversation context passed to the SEARCH path's
    # answer generator (M6.10) - small and fixed, so a search-answer prompt
    # doesn't grow with conversation length the way DIRECT's prompt can.
    # Not applied to DIRECT, which keeps its existing unbounded behavior.
    _SEARCH_CONTEXT_MAX_MESSAGES = 6
    _SEARCH_CONTEXT_MAX_CHARS = 800

    def __init__(
        self,
        provider: AIProvider | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        router: SearchRouter | None = None,
        answer_generator: SearchAnswerGenerator | None = None,
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

    def handle_chat(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, str, list[SearchSource]]:
        """Route a user message to the AI provider and return its reply.

        If conversation_id is None, a new conversation is started. If it's
        given but unknown, ConversationNotFoundError is raised rather than
        silently starting a new conversation under that ID. Otherwise the
        conversation's prior messages are loaded and combined with any
        stored user facts to give the provider context for this turn.

        SearchRouter (deterministic, no AI call) decides DIRECT vs SEARCH
        before that: DIRECT keeps the existing prompt -> provider.
        generate_reply path unchanged; SEARCH runs the message through
        WebSearchTool -> SearchResponse -> SearchAnswerGenerator instead,
        using the generated answer as the reply. Either way, the final
        reply is persisted exactly the same way. Both paths reuse the same
        `history`/`facts` already loaded below - the SEARCH path never
        issues its own memory query.

        The third return value is the SearchAnswer's validated sources
        (M6.11) - always `[]` for DIRECT, since it never produces a
        SearchAnswer at all.
        """
        if conversation_id is None:
            conversation_id = self.memory.create_conversation()
        elif not self.memory.conversation_exists(conversation_id):
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' does not exist."
            )

        history = self.memory.get_messages(conversation_id)
        facts = self.memory.get_facts()

        if self.router.route(message) is SearchDecision.SEARCH:
            reply, sources = self._handle_search(message, history, facts)
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
        """SEARCH path: WebSearchTool -> SearchResponse ->
        SearchAnswerGenerator, using SearchAnswer.answer as the reply and
        returning SearchAnswer.sources (M6.11) alongside it for the caller
        to surface as it sees fit.

        The search query is `message` alone, exactly as before M6.10 -
        conversation context is never passed to SearchRouter.route() or
        WebSearchTool.search(), only to the answer generator afterwards
        (M6.10), so it can resolve references like "it" or "the deadline"
        in the question without changing what was actually searched.

        Failures from either step are raised as ProviderError, the same
        error type the DIRECT path already raises on an AI-provider
        failure, so callers (like /chat's existing error handling) need no
        changes to keep treating a failed reply as a clean 502.
        """
        web_search = self.tools.get("web_search")
        if not isinstance(web_search, WebSearchTool):
            raise ProviderError("web_search tool is not registered.")

        try:
            response = web_search.search(message)
        except SearchProviderError as exc:
            raise ProviderError(f"Search failed: {exc}") from exc

        context = self._format_context(
            history, facts, max_messages=self._SEARCH_CONTEXT_MAX_MESSAGES
        )
        context = context[-self._SEARCH_CONTEXT_MAX_CHARS :]

        try:
            search_answer = self.answer_generator.generate(
                response, conversation_context=context
            )
        except SearchAnswerError as exc:
            raise ProviderError(f"Failed to generate a search answer: {exc}") from exc

        return search_answer.answer, search_answer.sources

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
