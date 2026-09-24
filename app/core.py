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
from app.context_engine.base import ContextIntent, RequestContext
from app.context_engine.engine import ContextEngine
from app.context_engine.request_builder import build_request_context
from app.context_planning.base import PlanningDirective
from app.context_planning.planner import ContextPlanner, ContextPlanningError
from app.context_reasoning.base import ReasoningResult
from app.context_reasoning.reasoner import ContextReasoner, ContextReasonerError
from app.memory.base import MemoryStore, Message
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider, ProviderError
from app.providers.factory import create_default_ai_provider_manager
from app.providers.manager import call_with_requirements
from app.providers.requirements import classify_request
from app.self_development.base import DevelopmentSession
from app.self_development.engine import SelfDevelopmentEngine
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
        context_reasoner: ContextReasoner | None = None,
        context_planner: ContextPlanner | None = None,
        identity_context: str | None = None,
        self_development_engine: SelfDevelopmentEngine | None = None,
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
        # M16.3: `None` by default, same "purely additive, off unless
        # explicitly configured" convention as `context_engine` above -
        # no existing caller/test passes one. When given (and a
        # ContextSnapshot was actually built - i.e. `context_engine` is
        # also configured), `_dispatch` asks it to reason over that
        # snapshot ONLY for the capability branches that actually read
        # the result (DIRECT, and CONFIRMATION's fallback to DIRECT) -
        # see `_reason_about_context`'s own docstring - and threads the
        # resulting ReasoningResult into `_handle_direct`/`_build_prompt`
        # the same way `context_snapshot` already is - see
        # `_format_reasoning_result`.
        # ContextReasoner itself never executes anything or mutates the
        # World Model (see app/context_reasoning/reasoner.py), so this
        # parameter's mere presence can never affect provider routing,
        # M15 planning/execution, or ToolRegistry dispatch beyond the
        # DIRECT-path prompt text it's designed to enrich.
        self.context_reasoner = context_reasoner
        # M16.4: `None` by default, same "purely additive, off unless
        # explicitly configured" convention as `context_reasoner` above.
        # When given, `_dispatch` asks it to classify a strategic
        # PlanningDirective ONLY for the same two branches that already
        # reason (DIRECT, and CONFIRMATION's fallback to DIRECT), AND
        # only when `request_context.context_intent` is itself planning/
        # action-oriented (PLANNING/TASK_EXECUTION/DECISION_SUPPORT) -
        # see `_plan_about_context`'s own docstring for exactly why an
        # informational request never even asks. Threaded into
        # `_handle_direct`/`_build_prompt` the same way `reasoning_result`
        # already is - see `_format_planning_directive`. ContextPlanner
        # itself never executes anything, never mutates the World Model,
        # and never builds a competing executable plan - M15
        # (app/orchestration/) remains the sole authoritative executable
        # planning/execution system (see app/context_planning/planner.py's
        # own module docstring), so this parameter's mere presence can
        # never affect provider routing, M15 planning/execution, or
        # ToolRegistry dispatch beyond the DIRECT-path prompt text it's
        # designed to enrich.
        self.context_planner = context_planner
        # Identity context (personality.md/user_profile.md - see
        # app/identity.py): `None` by default, same "purely additive, off
        # unless explicitly configured" convention as context_engine/
        # context_reasoner/context_planner above - no existing caller/
        # test passes one. When given, `_handle_direct` includes it in
        # every DIRECT prompt as plain, static, already-loaded text (see
        # `_build_prompt`) - it is never re-read per request, never
        # parsed for tool-call syntax, never reaches ToolRegistry or the
        # World Model, and never affects SEARCH/TOOL/TASK/TASK_STATUS/
        # CONFIRMATION dispatch, which don't call `_handle_direct` at all
        # (a resolved CONFIRMATION included - only its fallback-to-DIRECT
        # branch does).
        self.identity_context = identity_context
        # Self-Development Engine (app/self_development/): `None` by
        # default, same "purely additive, off unless explicitly
        # configured" convention as every M16.x layer above - no
        # existing caller/test passes one. When given, a message the
        # deterministic router identifies as SELF_DEVELOPMENT (see
        # Capability.SELF_DEVELOPMENT, app/capability.py) dispatches to
        # `_handle_self_development` instead of ordinary DIRECT handling;
        # with no engine configured, that same message falls back to
        # ordinary DIRECT handling exactly like an unresolved CONFIRMATION
        # does. The engine has its OWN, entirely separate ToolRegistry
        # (app/self_development/factory.py) - it never touches `self.tools`
        # and `self.tools` never touches it, so self-development stays
        # completely unreachable through the ordinary TOOL capability.
        self.self_development_engine = self_development_engine

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
        reply, conversation_id, sources, _job_id, _snapshot, _reasoning, _plan = self._dispatch(
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
        reply, conversation_id, sources, job_id, _snapshot, _reasoning, _plan = self._dispatch(
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
        computed and handed back, never consumed here. Kept as its own
        5-tuple method (rather than growing to 6 or 7) so a caller that
        only ever wanted M16.2's snapshot is unaffected by M16.3/M16.4
        existing - see `handle_chat_with_reasoning`/`handle_chat_with_planning`
        for the wider tuples that also expose those."""
        reply, conversation_id, sources, job_id, snapshot, _reasoning, _plan = self._dispatch(
            message, conversation_id
        )
        return reply, conversation_id, sources, job_id, snapshot

    def handle_chat_with_reasoning(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[
        str, str, list[SearchSource], str | None, ContextSnapshot | None, ReasoningResult | None
    ]:
        """Identical to `handle_chat_with_context`, plus a 6th return
        value: the ReasoningResult `_dispatch` built for this message
        (M16.3), or `None` when no `context_reasoner` was configured on
        this JarvisCore instance, no `context_engine` was configured (a
        ReasoningResult needs a ContextSnapshot to reason over), the
        message's capability never reaches a branch that actually
        consumes one (SEARCH/TOOL/TASK/TASK_STATUS/a resolved
        CONFIRMATION - see `_reason_about_context`'s own docstring for
        exactly which two branches do), or the reasoner itself failed
        (see `_reason_about_context`'s safe-degradation handling of
        ContextReasonerError). Purely additive, the same "new method,
        existing ones untouched" pattern `handle_chat_with_context`
        already established for the ContextSnapshot itself. Kept as its
        own 6-tuple (rather than growing to 7) so a caller that only
        ever wanted M16.3's result is unaffected by M16.4 existing - see
        `handle_chat_with_planning` for the 7-tuple that also exposes
        the PlanningDirective."""
        reply, conversation_id, sources, job_id, snapshot, reasoning, _plan = self._dispatch(
            message, conversation_id
        )
        return reply, conversation_id, sources, job_id, snapshot, reasoning

    def handle_chat_with_planning(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[
        str,
        str,
        list[SearchSource],
        str | None,
        ContextSnapshot | None,
        ReasoningResult | None,
        PlanningDirective | None,
    ]:
        """Identical to `handle_chat_with_reasoning`, plus a 7th return
        value: the PlanningDirective `_dispatch` built for this message
        (M16.4), or `None` when no `context_planner` was configured, no
        `reasoning_result` was produced, the request's own
        `RequestContext.context_intent` isn't planning/action-oriented
        (see `_plan_about_context`'s own docstring for the exact set),
        or the planner itself failed (safe-degradation, mirroring
        `_reason_about_context`). Purely additive, the same "new
        method, existing ones untouched" pattern this module has used
        for every milestone since M15's own `job_id`."""
        return self._dispatch(message, conversation_id)

    def _dispatch(
        self, message: str, conversation_id: str | None
    ) -> tuple[
        str,
        str,
        list[SearchSource],
        str | None,
        ContextSnapshot | None,
        ReasoningResult | None,
        PlanningDirective | None,
    ]:
        """The single implementation every `handle_chat*` method calls -
        capability dispatch and conversation persistence happen exactly
        once, here, so the public methods can never drift out of sync
        with each other."""
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
        request_context = None
        if self.context_engine is not None:
            request_context = build_request_context(message, capability=decision.capability)
            context_snapshot = self.context_engine.build_snapshot(request_context)

        # M16.3: `reasoning_result` starts `None` and is only ever
        # computed inside the two branches below that actually pass it
        # to `_handle_direct` - the ordinary DIRECT branch (the final
        # `else`) and CONFIRMATION's fallback-to-DIRECT branch. It is
        # deliberately NOT computed here, unconditionally, before the
        # capability branch: SEARCH/TOOL/TASK/TASK_STATUS/a resolved
        # CONFIRMATION never read a ReasoningResult at all (see
        # `_handle_search`/`_handle_tool`/`_handle_task`/
        # `_handle_task_status`/`_handle_confirmation`'s own signatures),
        # so reasoning over `context_snapshot` for one of those would be
        # a real LLM call whose result is immediately discarded - see
        # `_reason_about_context`'s own docstring for the full rationale.
        job_id: str | None = None
        reasoning_result: ReasoningResult | None = None
        planning_directive: PlanningDirective | None = None

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
                # conversation, unaffected by CONFIRMATION existing at all -
                # including M16.3's reasoning step and M16.4's planning
                # step, only reached here because this fallback is about
                # to call `_handle_direct`.
                reasoning_result = self._reason_about_context(
                    message, request_context, context_snapshot
                )
                planning_directive = self._plan_about_context(
                    message, request_context, context_snapshot, reasoning_result
                )
                reply, sources = self._handle_direct(
                    message,
                    history,
                    facts,
                    Capability.DIRECT,
                    context_snapshot,
                    reasoning_result,
                    planning_directive,
                )
        elif decision.capability is Capability.SELF_DEVELOPMENT:
            if self.self_development_engine is None:
                # No engine configured (every existing caller/test, and
                # the module-level `core` singleton unless app/main.py
                # wires one) - a clearly-identified development request
                # falls back to ordinary DIRECT handling rather than a
                # dead end, the exact same "hint, not a guarantee" fallback
                # CONFIRMATION's own branch above already uses.
                reasoning_result = self._reason_about_context(
                    message, request_context, context_snapshot
                )
                planning_directive = self._plan_about_context(
                    message, request_context, context_snapshot, reasoning_result
                )
                reply, sources = self._handle_direct(
                    message,
                    history,
                    facts,
                    Capability.DIRECT,
                    context_snapshot,
                    reasoning_result,
                    planning_directive,
                )
            else:
                reply, sources = self._handle_self_development(decision, message)
        else:
            reasoning_result = self._reason_about_context(
                message, request_context, context_snapshot
            )
            planning_directive = self._plan_about_context(
                message, request_context, context_snapshot, reasoning_result
            )
            reply, sources = self._handle_direct(
                message,
                history,
                facts,
                decision.capability,
                context_snapshot,
                reasoning_result,
                planning_directive,
            )

        self.memory.add_message(conversation_id, "user", message)
        self.memory.add_message(conversation_id, "assistant", reply)

        return (
            reply,
            conversation_id,
            sources,
            job_id,
            context_snapshot,
            reasoning_result,
            planning_directive,
        )

    def _reason_about_context(
        self,
        message: str,
        request_context: RequestContext | None,
        context_snapshot: ContextSnapshot | None,
    ) -> ReasoningResult | None:
        """M16.3: called ONLY from the two `_dispatch` branches that
        actually consume a ReasoningResult (the DIRECT branch and
        CONFIRMATION's fallback-to-DIRECT branch) - never for SEARCH/
        TOOL/TASK/TASK_STATUS/a resolved CONFIRMATION, so this milestone
        never spends an LLM call reasoning over context for a reply that
        was never going to read the result. This is a WIRING decision
        only: no new `context_intent`/`required_context` heuristic is
        added here - M16.2 already decided what's relevant, and
        ContextReasoner.reason's own empty-ContextSnapshot short circuit
        (app/context_reasoning/reasoner.py) still independently skips
        the provider call entirely for an empty snapshot, unchanged.

        Returns `None` when no `context_reasoner` is configured, no
        `context_snapshot` exists (no `context_engine` configured), or
        the reasoner fails - read-only and never executes anything (see
        app/context_reasoning/reasoner.py's own module docstring), so
        this can never change `reply`/`sources`/`job_id`, never touches
        ToolRegistry, the World Model, or a Job."""
        if self.context_reasoner is None or context_snapshot is None:
            return None
        try:
            return self.context_reasoner.reason(message, request_context, context_snapshot)
        except ContextReasonerError:
            return None

    #: RequestContext.context_intent values M16.4 treats as "planning/
    #: action-oriented" - the ONLY gate `_plan_about_context` applies
    #: beyond "a reasoning_result exists." Deliberately a small, fixed
    #: set (the same discipline every other closed-set gate in this
    #: codebase uses), reusing ContextIntent - itself already derived
    #: from M13/M14's Capability plus M16.2's own deterministic pattern
    #: matching (app/context_engine/request_builder.py) - rather than
    #: adding a new classifier. PROJECT_FOCUS/STATUS/SCHEDULING/GENERAL
    #: are deliberately excluded: "what's happening with my robotics
    #: project?" (PROJECT_FOCUS) and "what's the status of X" (STATUS)
    #: are informational, not planning-oriented, even though they're
    #: still DIRECT capability and still get a ReasoningResult.
    _PLANNING_ORIENTED_INTENTS = frozenset(
        {ContextIntent.PLANNING, ContextIntent.TASK_EXECUTION, ContextIntent.DECISION_SUPPORT}
    )

    def _plan_about_context(
        self,
        message: str,
        request_context: RequestContext | None,
        context_snapshot: ContextSnapshot | None,
        reasoning_result: ReasoningResult | None,
    ) -> PlanningDirective | None:
        """M16.4: called from the SAME two `_dispatch` branches as
        `_reason_about_context` (DIRECT, and CONFIRMATION's fallback to
        DIRECT) - never for SEARCH/TOOL/TASK/TASK_STATUS/a resolved
        CONFIRMATION, for the identical "never pay for a call whose
        result is discarded" reason `_reason_about_context`'s own
        docstring explains. On TOP of that, this method applies one
        further gate M16.3's own wiring doesn't need: it only actually
        asks `context_planner` when `request_context.context_intent` is
        planning/action-oriented (`_PLANNING_ORIENTED_INTENTS` above) -
        "do not make every chat message generate a planning directive"
        (M16.4's own explicit requirement). An informational DIRECT
        request (e.g. PROJECT_FOCUS/STATUS/GENERAL) still gets ordinary
        reasoning enrichment from M16.3, but never reaches M16.4 at all.

        Returns `None` when no `context_planner` is configured, no
        `context_snapshot`/`reasoning_result` exists (a PlanningDirective
        needs both - M16.4's own stated inputs), the intent isn't
        planning-oriented, or the planner fails - read-only and never
        executes anything (see app/context_planning/planner.py's own
        module docstring), so this can never change `reply`/`sources`/
        `job_id`, never touches ToolRegistry, the World Model, or a Job,
        and never creates a competing executable plan (M15 remains the
        sole authority for that)."""
        if self.context_planner is None or context_snapshot is None or reasoning_result is None:
            return None
        if (
            request_context is None
            or request_context.context_intent not in self._PLANNING_ORIENTED_INTENTS
        ):
            return None
        try:
            return self.context_planner.plan(
                message, request_context, context_snapshot, reasoning_result
            )
        except ContextPlanningError:
            return None

    def _handle_direct(
        self,
        message: str,
        history: list[Message],
        facts: dict[str, str],
        capability: Capability,
        context_snapshot: ContextSnapshot | None = None,
        reasoning_result: ReasoningResult | None = None,
        planning_directive: PlanningDirective | None = None,
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
        `_format_world_model_context`. `reasoning_result` (M16.3) is the
        same treatment for a ReasoningResult - see
        `_format_reasoning_result`, which returns "" (no prompt change)
        for `None` or an empty/trivial result, exactly mirroring
        `_format_world_model_context`'s own backward-compatibility
        guarantee one layer up. `planning_directive` (M16.4) gets the
        same treatment again via `_format_planning_directive`.

        `self.identity_context` (personality.md/user_profile.md, see
        app/identity.py) is read here rather than threaded through
        `_dispatch` - it's a static string computed once at Core
        construction, never derived per-request the way the other three
        are, so there's nothing for `_dispatch` to compute or pass
        down."""
        prompt = self._build_prompt(
            message,
            history,
            facts,
            context_snapshot,
            reasoning_result,
            planning_directive,
            self.identity_context,
        )
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

    def _handle_self_development(
        self, decision: CapabilityDecision, message: str
    ) -> tuple[str, list[SearchSource]]:
        """SELF_DEVELOPMENT path: dispatches to
        `self.self_development_engine.start(...)`, which proposes a
        DevelopmentPlan and stops there - no branch is created, no file
        is touched, nothing is committed or rolled back (see
        app/self_development/engine.py:SelfDevelopmentEngine.start's own
        docstring). `sources` is always `[]`.

        V1's own chat integration goes exactly this far: proposing and
        reporting a plan. Approving that plan, approving a commit, and
        rolling back are all available through SelfDevelopmentEngine's
        own directly-callable API (`approve_plan`/`approve_commit`/
        `rollback`) - not yet through a second chat turn. This is a
        deliberate, explicit V1 scope boundary (see this project's own
        Self-Development Engine documentation) rather than a rushed,
        under-tested multi-turn approval flow for a subsystem whose
        entire point is that nothing happens without explicit approval."""
        request_text = decision.goal or message
        session = self.self_development_engine.start(request_text)
        return self._format_self_dev_session(session), []

    @staticmethod
    def _format_self_dev_session(session: DevelopmentSession) -> str:
        """Pure serializer - describes `session`'s own already-decided
        state, never a second decision-maker. Never includes a secret
        file's contents (nothing here reads file content at all) and
        never claims an action happened that didn't (a plan is only
        ever described as "proposed," never as "done")."""
        from app.self_development.base import SessionStage

        if session.stage is SessionStage.BLOCKED_DIRTY_TREE:
            return (
                "I can't start a self-development session right now - the "
                "repository already has uncommitted changes. Please commit or "
                "stash them first, then ask again."
            )
        if session.plan is None:
            reason = session.messages[-1] if session.messages else "an unknown error"
            return f"I couldn't put together a development plan for that ({reason})."

        plan = session.plan
        files = ", ".join(f.path for f in plan.files_affected) or "no files (investigation only)"
        risks = ", ".join(plan.risks) if plan.risks else "none noted"
        lines = [
            f"Here's my proposed development plan (session {session.id}):",
            f"- Objective: {plan.objective}",
            f"- Files affected: {files}",
            f"- Risks: {risks}",
        ]
        # Surfaced clearly BEFORE editing begins (V1's own explicit
        # requirement) - a plan touching a protected security-critical
        # path (app/self_development/safety.py:is_protected_system_path)
        # will need a SEPARATE SECURITY_EDIT approval that ordinary plan
        # approval never grants - see app/self_development/engine.py's
        # own module docstring.
        protected = plan.protected_file_paths()
        if protected:
            lines.append(
                "- ⚠ Protected security-critical file(s) in this plan: "
                f"{', '.join(sorted(protected))} - these require a SEPARATE "
                "SECURITY_EDIT approval, which ordinary plan approval does not grant."
            )
        lines.append(
            "\nI won't create a branch, edit anything, or commit anything until "
            "this plan is explicitly approved."
        )
        return "\n".join(lines)

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
        reasoning_context: str = "",
        planning_context: str = "",
        identity_context: str = "",
    ) -> str:
        """Format identity context, stored facts, conversation history,
        and (M16.2/M16.3/M16.4, optional) World Model context/reasoning/
        planning into plain text - shared by `_build_prompt` (DIRECT,
        unbounded, the only caller that ever passes any of these) and
        `_handle_search` (SEARCH, bounded to a small recent window via
        `max_messages`, never passes any of them - SEARCH's own routing/
        query-building stays completely untouched) so both build context
        the same way instead of duplicating this formatting. Returns ""
        when there's nothing to include. `identity_context` (app/
        identity.py) goes FIRST - static personality/user-profile
        framing, not derived from this request, so it reads naturally
        ahead of everything else. `reasoning_context` is placed
        immediately after `world_model_context`, and `planning_context`
        immediately after that - each is downstream of, and about, the
        one before it, so they read naturally in that order, before the
        conversation history."""
        parts = []
        if identity_context:
            parts.append(identity_context)
        if facts:
            facts_text = "\n".join(f"- {key}: {value}" for key, value in facts.items())
            parts.append(f"Known facts about the user:\n{facts_text}")
        if world_model_context:
            parts.append(world_model_context)
        if reasoning_context:
            parts.append(reasoning_context)
        if planning_context:
            parts.append(planning_context)
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
    def _format_reasoning_result(reasoning_result: ReasoningResult | None) -> str:
        """The ONE place a ReasoningResult (M16.3, app/context_reasoning/)
        actually reaches "existing response generation" - a pure
        SERIALIZER, exactly the same role `_format_world_model_context`
        plays for a ContextSnapshot one layer up, and the same
        backward-compatibility guarantee: returns "" - contributing
        NOTHING to the prompt - when `reasoning_result` is `None` (no
        context_reasoner configured, no context_engine configured, or
        the reasoner failed and `_dispatch` degraded to `None`) or when
        `reasoning_result.is_empty()` (nothing was found worth reasoning
        about - see app/context_reasoning/reasoner.py's own empty-
        ContextSnapshot short circuit), so a request with nothing
        relevant produces byte-identical prompt text to before M16.3
        existed.

        Never re-derives, re-scores, or second-guesses anything in
        `reasoning_result` - only decides how to lay its already-
        validated fields out as text. `provenance` identifiers are
        deliberately NOT shown here (they're internal grounding
        metadata for app/context_reasoning/reasoner.py's own
        anti-hallucination validation, not something the end user or
        the downstream prompt needs to see)."""
        if reasoning_result is None or reasoning_result.is_empty():
            return ""

        lines = [f"Interpretation: {reasoning_result.interpretation}"]
        for field_name, label in (
            ("relevant_facts", "Relevant facts"),
            ("constraints", "Constraints"),
            ("conflicts", "Conflicts"),
            ("opportunities", "Opportunities"),
            ("conclusions", "Conclusions"),
        ):
            items = getattr(reasoning_result, field_name)
            if items:
                lines.append(f"{label}: " + "; ".join(item.text for item in items))
        if reasoning_result.recommended_next_step:
            lines.append(f"Recommended next step: {reasoning_result.recommended_next_step}")
        if reasoning_result.requires_clarification:
            lines.append(
                f"This request is ambiguous given what's known - ask the user: "
                f"{reasoning_result.clarification_question}"
            )
        return "What Rhasspy's reasoning about this request concluded:\n" + "\n".join(
            f"- {line}" for line in lines
        )

    @staticmethod
    def _format_planning_directive(planning_directive: PlanningDirective | None) -> str:
        """The ONE place a PlanningDirective (M16.4, app/context_planning/)
        actually reaches "existing response generation" - a pure
        SERIALIZER, exactly the same role `_format_reasoning_result`
        plays for a ReasoningResult one layer down, and the same
        backward-compatibility guarantee: returns "" - contributing
        NOTHING to the prompt - when `planning_directive` is `None` (no
        context_planner configured, no reasoning_result to plan from,
        the request's intent wasn't planning-oriented, or the planner
        failed) or when `planning_directive.is_empty()` (NONE, with
        nothing else - see app/context_planning/planner.py's own
        empty-context short circuit), so an informational/non-planning
        request produces byte-identical prompt text to before M16.4
        existed.

        Never re-derives, re-classifies, or second-guesses anything in
        `planning_directive` - only decides how to lay its already-
        validated fields out as text. `provenance` identifiers are
        deliberately NOT shown here, same as `_format_reasoning_result`
        (internal grounding metadata, not end-user-facing text). This
        method NEVER creates, suggests creating verbatim, or dispatches
        to `create_job`/any tool - it only ever describes the
        directive's own already-validated fields as plain text; whether
        and how a reply built from this actually invites the user to
        start a task remains entirely up to the provider's own
        response, exactly as it already is for any other prompt
        content."""
        if planning_directive is None or planning_directive.is_empty():
            return ""

        lines = [
            f"Objective: {planning_directive.objective}",
            f"Planning type: {planning_directive.planning_type.value}",
        ]
        for field_name, label in (
            ("contextual_constraints", "Contextual constraints"),
            ("relevant_resources", "Relevant resources"),
            ("dependencies", "Dependencies"),
        ):
            items = getattr(planning_directive, field_name)
            if items:
                lines.append(f"{label}: " + "; ".join(item.text for item in items))
        if planning_directive.assumptions:
            lines.append("Assumptions: " + "; ".join(planning_directive.assumptions))
        if planning_directive.suggested_approach:
            lines.append(f"Suggested approach: {planning_directive.suggested_approach}")
        if planning_directive.requires_execution:
            lines.append(
                "This may require actually starting a task - only if the user "
                "asks for that, it goes through Rhasspy's existing task system."
            )
        return "What Rhasspy's planning about this request concluded:\n" + "\n".join(
            f"- {line}" for line in lines
        )

    @staticmethod
    def _build_prompt(
        message: str,
        history: list[Message],
        facts: dict[str, str],
        context_snapshot: ContextSnapshot | None = None,
        reasoning_result: ReasoningResult | None = None,
        planning_directive: PlanningDirective | None = None,
        identity_context: str | None = None,
    ) -> str:
        """Combine identity context, stored facts, conversation history,
        and (M16.2/M16.3/M16.4, optional) World Model context/reasoning/
        planning into context for the provider. With none of those, the
        raw message is sent unchanged, matching the original single-turn
        behavior. `identity_context` (app/identity.py) is already a
        plain, pre-formatted string (or `None`/"") - this method makes
        no decision about its content, only where it goes: first, ahead
        of everything else, since it's static background identity/
        personality framing rather than anything derived from this
        specific request."""
        world_model_context = JarvisCore._format_world_model_context(context_snapshot)
        reasoning_context = JarvisCore._format_reasoning_result(reasoning_result)
        planning_context = JarvisCore._format_planning_directive(planning_directive)
        context = JarvisCore._format_context(
            history,
            facts,
            world_model_context=world_model_context,
            reasoning_context=reasoning_context,
            planning_context=planning_context,
            identity_context=identity_context or "",
        )
        if not context:
            return message
        return f"{context}\n\nUser: {message}"


core = JarvisCore()
