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

Since M6.15, `capability_router.route(message)` can itself fail - a
HybridCapabilityRouter's Gemini fallback (app/capability.py) may raise
CapabilityDecisionError for a genuinely ambiguous message. Core re-raises
that as ProviderError right where it's called, the same translation
_handle_search already does for SearchAnswerError, so /chat's existing
error handling (a clean 502) needs no changes to cover it. Core's default
capability_router is still the fully deterministic DeterministicCapability-
Router (unchanged from M6.14) - this failure path only matters for a
caller that explicitly injects a HybridCapabilityRouter.

Since M6.16, _handle_tool no longer calls ToolRegistry.execute() with a
bare tool_name and no arguments: it first asks a ToolCallBuilder
(app/tools/decision.py) to construct a validated ToolCall (tool_name +
parameters) - deterministically when the message already contains
enough information, via a Gemini-backed ToolCallProvider only when
genuinely necessary - then passes that straight into the same
ToolRegistry.execute(name, params=...) call. A ToolCallIncompleteError
(missing required parameters, or nothing to select from) becomes a
plain reply, the same way an unknown tool name already did; any other
ToolCallError is translated to ProviderError, the same convention
CapabilityDecisionError/SearchAnswerError already use.

Since M14, the DIRECT branch of handle_chat builds a RequestRequirements
(app/providers/requirements.py) from the message, the already-decided
Capability, and the assembled prompt's length, then calls the provider
through call_with_requirements (app/providers/manager.py) instead of
generate_reply directly - this is what lets M13's ProviderRouter actually
pick an initial provider for real DIRECT requests. It's a no-op whenever
`self.provider` isn't an AIProviderManager (a directly-injected
AIProvider, every existing test fake, ...) - see call_with_requirements's
own docstring. TOOL never reaches an AI provider through `self.provider`
at all (tool-call construction, when needed, goes through its own
deliberately Gemini-pinned ToolCallProvider - see app/tools/decision.py),
so there's nothing to route there; SEARCH's routing lives in
GeminiSearchAnswerGenerator instead (app/search/answer.py), since that's
where the actual provider call happens.

Since M15, TASK is a fourth capability (app/capability.py) alongside
DIRECT/SEARCH/TOOL: `_handle_task` dispatches to the existing
ToolRegistry's `create_job` tool by name, exactly like `_handle_tool`
dispatches to any other tool - Core gains no orchestration-specific
logic beyond that one line. Planning/validation/execution of the
resulting Job all happen later, on JobEngine's own background thread
(app/orchestration/engine.py, wired up in app/main.py the same way
AutomationEngine/ReminderScheduler already are) - never synchronously
inside `handle_chat`, so an ordinary DIRECT/SEARCH/TOOL request's latency
and behavior are completely unaffected by TASK existing at all.

Right after that initial TASK pass, two more capabilities closed the gap
between "a Job exists" and "the user can actually find out what happened
to it" through ordinary conversation, without changing what `handle_chat`
returns to any existing caller:

- TASK_STATUS (`_handle_task_status`): a natural-language status question
  ("what's the status of my task?", "is that finished?", "what are you
  doing?", "show me my running tasks") is answered by calling the
  existing `list_jobs` tool and resolving/formatting the result - never a
  second store, never a bypass of ToolRegistry.
- CONFIRMATION (`_handle_confirmation`): a short reply ("yes", "send it",
  "go ahead", "no") that MIGHT be answering a Job's pending confirmation
  is resolved against real Job state (via `list_jobs`) and, if something
  is actually awaiting confirmation, dispatched to the existing
  `confirm_job` tool. If nothing is pending, `_handle_confirmation`
  returns `None` and `handle_chat` falls back to ordinary `_handle_direct`
  handling - so a stray "yes"/"ok" in normal conversation, with no Job
  awaiting anything, behaves exactly as it always has.

`handle_chat`'s own public signature and behavior (a 3-tuple: reply,
conversation_id, sources) are completely unchanged by any of this - every
existing caller (this project's whole test suite, the voice client, ...)
is unaffected. A Job's id, when one is created, is instead exposed
through the additive `handle_chat_with_job_id` (a 4-tuple, the 4th being
`job_id: str | None`) that only `app/main.py`'s `/chat` endpoint uses, so
API clients get a real structured field instead of having to parse the
reply's prose for it. Both methods share one internal implementation
(`_dispatch`) so there is exactly one place capability dispatch and
conversation persistence happen, never two copies to keep in sync.
"""

import re

from app.capability import (
    Capability,
    CapabilityDecision,
    CapabilityDecisionError,
    CapabilityRouter,
    DeterministicCapabilityRouter,
)
from app.config import settings
from app.context_engine.engine import ContextEngine
from app.context_engine.request_builder import build_request_context
from app.memory.base import MemoryStore, Message
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider, ProviderError
from app.providers.factory import create_default_ai_provider_manager
from app.providers.manager import call_with_requirements
from app.providers.requirements import classify_request
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
from app.tools.decision import ToolCallBuilder, ToolCallError, ToolCallIncompleteError
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool
from app.world_model.base import ContextSnapshot


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
        tool_call_builder: ToolCallBuilder | None = None,
        context_engine: ContextEngine | None = None,
    ):
        self.provider = provider or create_default_ai_provider_manager()
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
        self.tool_call_builder = tool_call_builder or ToolCallBuilder()
        # M16.2: `None` by default (including for the module-level `core`
        # singleton below) - no existing caller/test passes one, so every
        # existing behavior is completely unaffected. When given, `_dispatch`
        # additionally builds a RequestContext/ContextSnapshot for every
        # message (any capability) and threads it through as a purely
        # ADDITIVE output - see `handle_chat_with_context`. Nothing reads
        # or acts on the snapshot itself here: it never reaches a prompt,
        # a tool call, or a Job's context - Core only ever computes it and
        # hands it back, so provider routing, M15 planning/execution, and
        # ToolRegistry dispatch are all byte-for-byte unaffected by this
        # parameter's mere presence.
        self.context_engine = context_engine

    def handle_chat(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, str, list[SearchSource]]:
        """Route a user message to the AI provider and return its reply.

        If conversation_id is None, a new conversation is started. If it's
        given but unknown, ConversationNotFoundError is raised rather than
        silently starting a new conversation under that ID. Otherwise the
        conversation's prior messages are loaded and combined with any
        stored user facts to give the provider context for this turn.

        CapabilityRouter (deterministic, no AI call - M6.13/M6.14, extended
        in M15) decides DIRECT vs SEARCH vs TOOL vs TASK vs TASK_STATUS vs
        CONFIRMATION before that, not Core itself - see this module's own
        docstring for what each of those does. Either way, the final reply
        is persisted exactly the same way.

        The third return value is the SearchAnswer's validated sources
        (M6.11) - always `[]` for every capability except SEARCH, since
        none of the others produce a SearchAnswer.

        This method's own signature/behavior is unchanged by M15's
        TASK_STATUS/CONFIRMATION additions - see `handle_chat_with_job_id`
        for the one additive way a created Job's id is exposed structurally
        instead of only in the reply's prose, and `handle_chat_with_context`
        for the same treatment of M16.2's ContextSnapshot.
        """
        reply, conversation_id, sources, _job_id, _snapshot = self._dispatch(
            message, conversation_id
        )
        return reply, conversation_id, sources

    def handle_chat_with_job_id(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, str, list[SearchSource], str | None]:
        """Identical to `handle_chat` in every respect, plus a 4th return
        value: the id of the Job `_handle_task` just created, if this
        message was a TASK request - `None` for every other capability, or
        if TASK dispatch didn't actually create one (e.g. `create_job`
        isn't registered). This is the one thing `handle_chat`'s own
        existing 3-tuple contract can't grow to carry without breaking
        every one of its current callers, so `app/main.py`'s `/chat`
        endpoint uses this method instead, to give API clients a real
        structured `job_id` field rather than one embedded in reply text.
        """
        reply, conversation_id, sources, job_id, _snapshot = self._dispatch(
            message, conversation_id
        )
        return reply, conversation_id, sources, job_id

    def handle_chat_with_context(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, str, list[SearchSource], str | None, ContextSnapshot | None]:
        """Identical to `handle_chat_with_job_id`, plus a 5th return value:
        the ContextSnapshot `_dispatch` built for this message (M16.2), or
        `None` when no `context_engine` was configured on this JarvisCore
        instance. Purely additive, the same "new method, existing ones
        untouched" pattern `handle_chat_with_job_id` already established
        for `job_id` - `handle_chat`/`handle_chat_with_job_id` keep their
        exact existing signatures and behavior; this is simply where a
        caller that actually wants the snapshot (e.g. a future caller
        that decides what to do with it) can get it. Nothing in this
        module itself reads or acts on the returned snapshot - it's
        computed and handed back, never consumed here."""
        return self._dispatch(message, conversation_id)

    def _dispatch(
        self, message: str, conversation_id: str | None
    ) -> tuple[str, str, list[SearchSource], str | None, ContextSnapshot | None]:
        """The single implementation `handle_chat`/`handle_chat_with_job_id`/
        `handle_chat_with_context` all call - capability dispatch and
        conversation persistence happen exactly once, here, so the three
        public methods can never drift out of sync with each other."""
        if conversation_id is None:
            conversation_id = self.memory.create_conversation()
        elif not self.memory.conversation_exists(conversation_id):
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' does not exist."
            )

        history = self.memory.get_messages(conversation_id)
        facts = self.memory.get_facts()

        try:
            decision = self.capability_router.route(message)
        except CapabilityDecisionError as exc:
            raise ProviderError(f"Failed to decide a capability: {exc}") from exc

        # M16.2: purely additive - computed only when a context_engine was
        # actually configured (never for the default `core` singleton or
        # any existing test, none of which pass one), and never consumed
        # by anything below. `build_request_context` reuses the SAME
        # `decision.capability` this method already computed above - no
        # second classifier - and ContextEngine.build_snapshot is
        # read-only (see app/context_engine/engine.py's own module
        # docstring), so this can never change `reply`/`sources`/`job_id`,
        # never touches a provider, ToolRegistry, or a Job.
        context_snapshot: ContextSnapshot | None = None
        if self.context_engine is not None:
            request_context = build_request_context(message, capability=decision.capability)
            context_snapshot = self.context_engine.build_snapshot(request_context)

        job_id: str | None = None

        if decision.capability is Capability.SEARCH:
            reply, sources = self._handle_search(message, history, facts)
        elif decision.capability is Capability.TOOL:
            reply, sources = self._handle_tool(message, decision)
        elif decision.capability is Capability.TASK:
            reply, sources, job_id = self._handle_task(decision, conversation_id)
        elif decision.capability is Capability.TASK_STATUS:
            reply, sources = self._handle_task_status(message, conversation_id)
        elif decision.capability is Capability.CONFIRMATION:
            confirmation_result = self._handle_confirmation(decision, conversation_id)
            if confirmation_result is not None:
                reply, sources = confirmation_result
            else:
                # Nothing is actually awaiting confirmation right now - the
                # deterministic "yes"/"ok"/... pattern was only ever a
                # hint (see Capability.CONFIRMATION's own docstring), so a
                # stray affirmative/negative reply falls back to ordinary
                # conversation, unaffected by CONFIRMATION existing at all.
                reply, sources = self._handle_direct(
                    message, history, facts, Capability.DIRECT, context_snapshot
                )
        else:
            reply, sources = self._handle_direct(
                message, history, facts, decision.capability, context_snapshot
            )

        self.memory.add_message(conversation_id, "user", message)
        self.memory.add_message(conversation_id, "assistant", reply)

        return reply, conversation_id, sources, job_id, context_snapshot

    def _handle_direct(
        self,
        message: str,
        history: list[Message],
        facts: dict[str, str],
        capability: Capability,
        context_snapshot: ContextSnapshot | None = None,
    ) -> tuple[str, list[SearchSource]]:
        """DIRECT path (extracted, M15, from what was previously inlined
        in `handle_chat`'s own `else` branch) - also reused by
        `_dispatch`'s CONFIRMATION fallback, so a "yes"/"ok" with nothing
        actually pending is handled by the exact same code as an ordinary
        DIRECT message, not a second copy of it. `capability` is passed
        through to `classify_request` (M14) rather than hardcoded, exactly
        as the original inline branch already did - always
        `Capability.DIRECT` for both callers today.

        `context_snapshot` (M16.2) is the ONE place a ContextSnapshot
        actually reaches "existing reasoning" - see `_build_prompt`'s own
        docstring for exactly how. `None` (no context_engine configured)
        or an empty snapshot (nothing retrieved) both produce byte-
        identical prompt text to before this parameter existed - see
        `_format_world_model_context`."""
        prompt = self._build_prompt(message, history, facts, context_snapshot)
        # `context_char_count=len(prompt)` is the exact payload about to be
        # sent - the most direct "how much context is this" signal
        # available, no separate length heuristic invented. See
        # app/providers/requirements.py for what this actually classifies
        # (task type, long_context, ...) and
        # app/providers/manager.py:call_with_requirements for why this is
        # safe to call even when self.provider is a plain AIProvider that
        # knows nothing about requirements.
        requirements = classify_request(
            message, capability=capability, context_char_count=len(prompt)
        )
        reply = call_with_requirements(self.provider, prompt, requirements)
        return reply, []

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

    def _handle_tool(
        self, message: str, decision: CapabilityDecision
    ) -> tuple[str, list[SearchSource]]:
        """TOOL path: ToolCallBuilder (M6.16) constructs a validated
        ToolCall - deterministically when the message already has
        everything needed, via a Gemini-backed ToolCallProvider only when
        genuinely necessary - then dispatches to the existing
        ToolRegistry (app/tools/registry.py) by name + parameters. Core
        never branches on which tool it is, never inspects a specific
        tool's own parameter semantics, and never extracts parameters
        from the message itself - that's ToolCallBuilder's job. `sources`
        is always `[]` - TOOL never produces search evidence.

        A ToolCallIncompleteError (required parameters missing, or no
        tools registered to choose from) is a normal, typed "not enough
        information" outcome - reported back as the reply text, not a
        502, since it's about what the user said rather than a system
        failure; this replaces M6.14's fixed "couldn't tell which tool"
        reply for `decision.tool_name is None` with a real attempt at
        natural-language tool selection, falling back to the same kind of
        plain reply only when that genuinely can't succeed (e.g. no tools
        registered at all). Any other ToolCallError (a failed AI call, or
        AI output that fails validation) is translated to ProviderError,
        the same 502-producing convention CapabilityDecisionError/
        SearchAnswerError already use.

        An unregistered `tool_name` still needs no special handling here:
        ToolCallBuilder passes it through unchanged, and
        `ToolRegistry.execute()` returns its existing typed "Unknown
        tool" `ToolResult.error(...)` for it, same as before M6.16.
        Permission enforcement (CONFIRM/DANGEROUS -> CONFIRMATION_REQUIRED/
        DENIED) is likewise entirely the Registry's existing behavior,
        unaffected by how the parameters were constructed - the `/chat`
        API has no confirmation round-trip yet, so those outcomes simply
        surface as the reply text, same as an error would.
        """
        try:
            tool_call = self.tool_call_builder.build(
                message, self.tools, decision.tool_name
            )
        except ToolCallIncompleteError as exc:
            return str(exc), []
        except ToolCallError as exc:
            raise ProviderError(f"Failed to construct a tool call: {exc}") from exc

        result = self.tools.execute(tool_call.tool_name, params=tool_call.parameters)
        return self._format_tool_result(result), []

    def _handle_task(
        self, decision: CapabilityDecision, conversation_id: str
    ) -> tuple[str, list[SearchSource], str | None]:
        """TASK path (M15): dispatches to the existing ToolRegistry's
        `create_job` tool (app/orchestration/tools.py) by name, exactly
        the same "Core never branches on which tool it is" pattern
        `_handle_tool` already uses - the only difference is TASK's tool
        name and argument (`goal`) are already fully known from the
        CapabilityDecision itself (see Capability.TASK's own docstring),
        so no ToolCallBuilder round-trip (deterministic or AI-backed) is
        needed to construct the call. `conversation_id` is passed through
        as `create_job`'s own `conversation_id` argument, so the new Job
        is linked back to this conversation - what lets
        `_handle_task_status`/`_handle_confirmation` later resolve "my
        task"/a pending confirmation reply against it specifically.

        This method only ever creates a PENDING Job and returns
        immediately - planning and execution happen later, on
        JobEngine's own background thread (app/orchestration/engine.py),
        so a TASK request never blocks `/chat` waiting for a multi-step
        plan to actually run. If `create_job` isn't registered (e.g. a
        bare ToolRegistry with no orchestration tools, as in most of this
        project's own unit tests), the registry's existing "Unknown
        tool" ToolResult.error(...) is surfaced as plain reply text, same
        convention as an unregistered TOOL-path tool_name, and the
        returned job_id is `None`.

        Returns a 3-tuple (reply, sources, job_id) rather than
        `_handle_tool`'s 2-tuple - job_id is the one piece of structured
        data `_dispatch` needs to thread through to
        `handle_chat_with_job_id`'s own 4th return value; every other
        handler still returns a plain 2-tuple since none of them create
        anything with an id worth exposing this way."""
        goal = decision.goal or ""
        result = self.tools.execute(
            "create_job", params={"goal": goal, "conversation_id": conversation_id}
        )
        job_id = None
        if result.status is ToolResultStatus.SUCCESS and isinstance(result.data, dict):
            job_id = result.data.get("id")
        return self._format_tool_result(result), [], job_id

    #: "show me my running tasks" / "list me my active tasks" (list intent)
    #: vs "is that finished?" / "what's the status of my task?" (single-job
    #: intent) - a narrow, local re-check of the SAME message
    #: DeterministicCapabilityRouter already matched as TASK_STATUS (see
    #: app/capability.py's own `_TASK_STATUS_PATTERNS`), just to pick which
    #: of the two response shapes to build. Never a second capability
    #: decision - `_handle_task_status` is only ever reached once the
    #: router has already decided TASK_STATUS.
    _TASK_STATUS_LIST_INTENT_PATTERN = re.compile(r"\b(?:show|list)\b.*\btasks\b", re.IGNORECASE)

    #: Statuses that count as "still going" for "show me my running
    #: tasks" - a terminal Job (COMPLETED/FAILED/CANCELLED) is deliberately
    #: excluded from that listing, even though get_job/list_jobs still
    #: return it (a status query about ALL history isn't what "running
    #: tasks" asks for).
    _ACTIVE_JOB_STATUS_VALUES = frozenset(
        {"pending", "planning", "running", "awaiting_confirmation", "paused"}
    )

    def _handle_task_status(
        self, message: str, conversation_id: str
    ) -> tuple[str, list[SearchSource]]:
        """TASK_STATUS path (M15): answers a natural-language question
        about a Job's progress/outcome ("what's the status of my task?",
        "is that finished?", "what are you doing?", "show me my running
        tasks") using the existing `list_jobs` tool - never a second
        store, never a ToolRegistry bypass. `list_jobs`'s own summary dict
        already carries `final_result`/`error`/`confirmation_message`
        alongside `status` (see app/orchestration/tools.py), so this never
        needs a second `get_job` round-trip to describe an outcome.

        Resolution for a single-job question prefers a Job linked to
        THIS conversation (`conversation_id`, set by `_handle_task` at
        creation time), falling back to considering every Job if none
        match - reasonable for a single-user assistant where "my task"
        outside its own conversation is still unambiguous when there's
        only one candidate. Guessing "the most recently updated matching
        Job" is safe here specifically because this is a read-only query
        - contrast with `_handle_confirmation`, which never guesses among
        multiple candidates for a consequential approval action."""
        result = self.tools.execute("list_jobs", params={})
        if result.status is not ToolResultStatus.SUCCESS:
            return self._format_tool_result(result), []

        jobs: list[dict] = result.data or []
        if not jobs:
            return "You don't have any tasks right now.", []

        if self._TASK_STATUS_LIST_INTENT_PATTERN.search(message):
            return self._format_job_list(jobs), []

        return self._format_single_job_status(jobs, conversation_id), []

    @classmethod
    def _format_job_list(cls, jobs: list[dict]) -> str:
        active = [j for j in jobs if j["status"] in cls._ACTIVE_JOB_STATUS_VALUES]
        if not active:
            return "You have no running tasks right now."
        lines = [f"- {j['goal']} ({j['status']})" for j in active]
        return "Your running tasks:\n" + "\n".join(lines)

    @staticmethod
    def _resolve_relevant_job(jobs: list[dict], conversation_id: str) -> dict:
        """Pick the single most relevant Job for a status/confirmation
        query: the most recently updated one linked to THIS conversation,
        or - if none are - the most recently updated one overall. Shared
        by `_handle_task_status` (safe to guess, read-only) -
        `_handle_confirmation` uses the same conversation-scoping idea but
        never calls this when more than one candidate remains, since
        approving/declining is consequential (see its own docstring)."""
        same_conversation = [j for j in jobs if j.get("conversation_id") == conversation_id]
        candidates = same_conversation or jobs
        return max(candidates, key=lambda j: j["updated_at"])

    @classmethod
    def _format_single_job_status(cls, jobs: list[dict], conversation_id: str) -> str:
        job = cls._resolve_relevant_job(jobs, conversation_id)
        goal, status = job["goal"], job["status"]

        if status == "completed":
            outcome = job.get("final_result") or "it finished successfully"
            return f"Yes - your task '{goal}' is finished. {outcome}"
        if status == "failed":
            reason = job.get("error") or "no reason was recorded"
            return f"Your task '{goal}' failed: {reason}"
        if status == "awaiting_confirmation":
            ask = job.get("confirmation_message") or "it needs your confirmation to continue"
            return f"Your task '{goal}' is waiting on you: {ask}"
        if status == "cancelled":
            return f"Your task '{goal}' was cancelled."
        if status == "paused":
            return f"Your task '{goal}' is paused."
        return f"Your task '{goal}' is still in progress ({status})."

    def _handle_confirmation(
        self, decision: CapabilityDecision, conversation_id: str
    ) -> tuple[str, list[SearchSource]] | None:
        """CONFIRMATION path (M15): resolves a short affirmative/negative
        reply against whatever Job(s) are actually AWAITING_CONFIRMATION
        right now (via the existing `list_jobs` tool), and - if exactly
        one applies - dispatches to the existing `confirm_job` tool with
        `decision.approve`. Returns `None` (never a reply) when there's
        nothing pending at all, so `_dispatch` can fall back to ordinary
        `_handle_direct` handling instead - the deterministic "yes"/"no"
        pattern in app/capability.py is only ever a hint this message
        MIGHT be a confirmation reply, and this is where that hint is
        actually checked against reality.

        Safe ambiguity handling: if MORE than one Job is awaiting
        confirmation (after preferring ones linked to THIS conversation -
        see `_resolve_relevant_job`), this never guesses which one the
        user means - approving/declining is a consequential, potentially
        irreversible action (see app/orchestration/engine.py's own
        permission discipline), unlike a read-only status question. It
        asks the user to disambiguate instead, and no `confirm_job` call
        is made at all until they do.
        """
        result = self.tools.execute("list_jobs", params={"status": "awaiting_confirmation"})
        if result.status is not ToolResultStatus.SUCCESS:
            return None  # can't tell anything is pending - not a confirmation reply

        pending: list[dict] = result.data or []
        if not pending:
            return None

        same_conversation = [j for j in pending if j.get("conversation_id") == conversation_id]
        candidates = same_conversation or pending

        if len(candidates) > 1:
            options = "; ".join(f"'{j['goal']}' (id {j['id']})" for j in candidates)
            return (
                "You have more than one task awaiting confirmation: "
                f"{options}. Which one do you mean?",
                [],
            )

        target = candidates[0]
        confirm_result = self.tools.execute(
            "confirm_job", params={"job_id": target["id"], "approve": bool(decision.approve)}
        )
        return self._format_tool_result(confirm_result), []

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
        world_model_context: str = "",
    ) -> str:
        """Format stored facts, conversation history, and (M16.2,
        optional) World Model context into plain text - shared by
        `_build_prompt` (DIRECT, unbounded, the only caller that ever
        passes `world_model_context`) and `_handle_search` (SEARCH,
        bounded to a small recent window via `max_messages`, never
        passes `world_model_context` - SEARCH's own routing/query-
        building stays completely untouched by M16.2) so both build
        context the same way instead of duplicating this formatting.
        Returns "" when there's nothing to include."""
        parts = []
        if facts:
            facts_text = "\n".join(f"- {key}: {value}" for key, value in facts.items())
            parts.append(f"Known facts about the user:\n{facts_text}")
        if world_model_context:
            parts.append(world_model_context)
        if history:
            recent = history if max_messages is None else history[-max_messages:]
            history_text = "\n".join(f"{m.role}: {m.content}" for m in recent)
            parts.append(f"Conversation so far:\n{history_text}")
        return "\n\n".join(parts)

    #: Every ContextSnapshot category this formatter knows how to
    #: serialize, in a fixed display order - (attribute name, display
    #: label, per-item -> str). This is NOT a relevance whitelist: all
    #: nine ContextSnapshot fields are listed here (the same nine
    #: RequiredContextCategory names, app/context_engine/base.py), so
    #: every category ContextEngine can ever populate has somewhere to
    #: go. Whether a category actually HAS anything to show is decided
    #: entirely by ContextEngine (an empty list here just means "not
    #: populated," never "excluded by the formatter") - see
    #: `_format_world_model_context`'s own docstring. `relevant_tasks`
    #: has no human-readable name on the TaskLink itself (see
    #: app/world_model/base.py:TaskLink's own docstring on why ref_id is
    #: opaque) - shown by ref_type:ref_id, the only thing this formatter
    #: can honestly say about it without importing app.tasks/
    #: app.orchestration (which it must never do). `relevant_memory` is
    #: already `list[str]` (ContextSnapshot's own shape - see
    #: app/world_model/base.py) - the identity function.
    _WORLD_MODEL_CATEGORIES: tuple = (
        ("active_projects", "Active projects", lambda p: p.name),
        ("relevant_goals", "Relevant goals", lambda g: g.title),
        ("relevant_tasks", "Relevant tasks", lambda t: f"{t.ref_type.value}:{t.ref_id}"),
        ("upcoming_commitments", "Upcoming commitments", lambda c: c.title),
        ("upcoming_events", "Upcoming events", lambda e: e.title),
        ("active_blockers", "Active blockers", lambda b: b.title),
        ("relevant_resources", "Relevant resources", lambda r: r.name),
        ("relevant_preferences", "Relevant preferences", lambda p: f"{p.key}={p.value}"),
        ("relevant_memory", "Relevant memory", lambda m: m),
    )

    @classmethod
    def _format_world_model_context(cls, snapshot: ContextSnapshot | None) -> str:
        """The ONE place a ContextSnapshot (M16.2, app/context_engine/)
        actually reaches "existing reasoning" - a pure SERIALIZER, never
        a second decision-maker. Returns "" - contributing NOTHING to
        the prompt - when `snapshot` is `None` (no context_engine
        configured on this JarvisCore instance) or when every category
        is empty (nothing relevant in the World Model for this request,
        or that category simply wasn't in RequestContext.required_context
        - ContextEngine.build_snapshot only ever populates a category's
        list when it was actually requested, so an empty list here
        already means "not relevant/not requested," with no separate
        check needed), so an ordinary request with no meaningful
        world-model context produces byte-identical prompt text to
        before this integration existed (M16.2's own explicit backward-
        compatibility requirement).

        This method makes exactly one decision of its own: which of
        ContextSnapshot's nine fields get a line AT ALL when non-empty
        (`_WORLD_MODEL_CATEGORIES` above, covering all nine - not a
        smaller whitelist) and what label/short string to show for each
        item. It never re-ranks, filters by relevance, truncates, or
        applies any item/category limit of its own: every category's
        items are joined in EXACTLY the order `snapshot.<field>` already
        holds them in - the order ContextEngine.build_snapshot's own
        relevance sort produced - and every item ContextEngine selected
        appears, however many that budget-selection actually decided on
        (app/context_engine/base.py:ContextBudget is the only place a
        count limit is ever decided). Only ever reads from the snapshot -
        never queries the World Model itself (that's
        ContextEngine.build_snapshot's job, already done by the time
        this runs), and never touches MemoryStore directly either -
        `relevant_memory` is serialized exactly as ContextEngine already
        populated it, nothing added, nothing summarized, nothing
        extracted here."""
        if snapshot is None:
            return ""

        lines: list[str] = []
        for field_name, label, describe in cls._WORLD_MODEL_CATEGORIES:
            items = getattr(snapshot, field_name)
            if not items:
                continue
            lines.append(f"- {label}: " + ", ".join(describe(item) for item in items))

        if not lines:
            return ""
        return "What Rhasspy currently knows from the user's world model:\n" + "\n".join(lines)

    @staticmethod
    def _build_prompt(
        message: str,
        history: list[Message],
        facts: dict[str, str],
        context_snapshot: ContextSnapshot | None = None,
    ) -> str:
        """Combine stored facts, conversation history, and (M16.2,
        optional) World Model context into context for the provider.
        With none of those, the raw message is sent unchanged, matching
        the original single-turn behavior."""
        world_model_context = JarvisCore._format_world_model_context(context_snapshot)
        context = JarvisCore._format_context(
            history, facts, world_model_context=world_model_context
        )
        if not context:
            return message
        return f"{context}\n\nUser: {message}"


core = JarvisCore()
