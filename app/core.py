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
own docstring. SEARCH's routing lives in GeminiSearchAnswerGenerator
instead (app/search/answer.py), since that's where the actual provider
call happens.

Until M17.1-E, TOOL never reached an AI provider through `self.provider`
at all: tool-call construction, when needed, went through its own
GeminiToolCallProvider (app/tools/decision.py) built around a hardcoded
`GeminiProvider()` - a real gap, not a deliberate exclusion from routing:
a live smoke test found that a Gemini 429 during tool-call construction
(e.g. a computer-control tool needing more than one parameter, like
`move_mouse`'s x/y) surfaced straight to the user instead of falling back
to Groq/OpenRouter/Cloudflare/FreeLLMAPI the way DIRECT already does.
M17.1-E fixes this the same way M14 wired DIRECT: Core's default
`tool_call_builder` now builds its GeminiToolCallProvider around
`self.provider` itself (see `__init__` below) instead of a fresh
`GeminiProvider()`, so when `self.provider` is the real
AIProviderManager, `GeminiToolCallProvider.decide`'s call to
`self.provider.generate_reply(prompt)` gets the SAME ordered
retry/cooldown/fallback behavior AIProviderManager.generate_reply already
implements (app/providers/manager.py) - no new routing/retry/fallback
logic anywhere, and no `requirements=`/ProviderRouter involvement (tool
calls still don't build a RequestRequirements), just a plain
multi-provider `generate_reply` call, exactly like every pre-M13 caller
already gets. `GeminiToolCallProvider`'s own strict JSON/schema
validation is completely unchanged and still runs only once, against
whichever provider ultimately answered - an invalid/malformed reply from
that provider is never retried, only a provider-level ProviderError
(429/5xx/timeout/etc, classified exactly as before) triggers a fallback
to the next provider in the SAME configured order. Tool-call construction
still never sees a tool's permission level, and ToolRegistry.execute
still independently enforces it after the ToolCall comes back - which
provider's text happened to answer the prompt changes nothing about
permission enforcement or which tools were ever offered as candidates.

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

The Self-Development Engine (app/self_development/) integration follows
the exact same "hint, not a guarantee, resolved against real state"
shape:

- SELF_DEVELOPMENT (`_handle_self_development`): a clearly-identified
  development request ("fix the voice module") starts a new session via
  `self_development_engine.start(...)`, which proposes a plan and stops -
  nothing is branched, edited, or committed yet.
- SELF_DEV_APPROVAL (`_handle_self_dev_approval`): a LATER message
  explicitly naming a gate ("approve the plan", "reject the security
  edit", "approve the commit") is resolved against whichever session in
  THIS SAME conversation is actually awaiting THAT exact gate right now,
  then dispatched to the engine's own existing `approve_plan`/
  `approve_security_edit`/`approve_commit`. If no such session exists,
  this returns `None` and falls back to ordinary DIRECT handling, same as
  CONFIRMATION's own fallback - and, unlike CONFIRMATION, NEVER widens
  the search to other conversations if none matches this one (SECURITY_
  EDIT/COMMIT are DANGEROUS-tier, so association must be exact, not
  merely "the only one pending anywhere").

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

M17.1-C connects the computer-control tools (app/computer/, M17.1-A/B)
to ordinary interactive TOOL-capability chat, WITHOUT making them
available to M15 Job planning or AutomationEngine: `computer_tools`
(`__init__`) is a wholly separate `ToolRegistry` from `self.tools` - the
exact registry M15's JobEngine and AutomationEngine are built with in
app/main.py - so `_handle_tool`'s new `_resolve_tool_registry` can route
an interactive, explicitly-named computer-tool request there while
JobEngine/AutomationEngine simply never hold a reference to it at all.
See `_handle_tool`/`_resolve_tool_registry`'s own docstrings for exactly
how a request is routed and why that routing can never be influenced by
the message/decision itself.
"""

import json
import re
from datetime import datetime

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
from app.memory_policy.derived import with_derived_facts
from app.providers.base import AIProvider, ProviderError
from app.providers.factory import create_default_ai_provider_manager
from app.providers.manager import call_with_requirements
from app.providers.requirements import classify_request
from app.self_development.base import DevelopmentSession, SessionStage
from app.self_development.engine import SelfDevelopmentEngine
from app.search.answer import (
    GeminiSearchAnswerGenerator,
    SearchAnswerError,
    SearchAnswerGenerator,
    SearchSource,
)
from app.search.base import SearchProviderError
from app.search.query_builder import DeterministicSearchQueryBuilder, SearchQueryBuilder
from app.search.router import SearchRouter, is_explicit_search_request
from app.tools.base import ToolResult, ToolResultStatus
from app.tools.command import CommandRejection, ExplicitToolCommandPlanner, ToolSequence
from app.tools.decision import (
    GeminiToolCallProvider,
    ToolCallBuilder,
    ToolCallError,
    ToolCallIncompleteError,
)
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
        computer_tools: ToolRegistry | None = None,
        interaction_resolver=None,
        handle_this=None,
        memory_writer=None,
        activity_context=None,
        escalation_surface=None,
        morning_brief=None,
        device_operations=None,
        reminder_operations=None,
        scheduled_job_operations=None,
        focus=None,
    ):
        self.provider = provider or create_default_ai_provider_manager()
        self.memory = memory or SQLiteMemoryStore(settings.memory_db_path)
        # Empty by default - no real tools exist yet (M5 is architecture
        # only). AI tool-calling isn't wired up here either; Core just has
        # a place to reach the registry without knowing any tool's type.
        self.tools = tools or ToolRegistry()
        # Computer-control tools (M17.1-C, app/computer/) - `None` by
        # default, same "purely additive, off unless explicitly
        # configured" convention as context_engine/context_reasoner/
        # context_planner/self_development_engine above: no existing
        # caller/test passes one, so every existing behavior (including
        # every current TOOL-capability request) is byte-for-byte
        # unaffected. This is DELIBERATELY a SEPARATE ToolRegistry from
        # `self.tools`, never merged into it: `self.tools` is the exact
        # registry M15's JobEngine and AutomationEngine are constructed
        # with in the real running app (see app/main.py) - JobEngine's
        # planner offers `self.tools.list_tools()` to its planner as
        # candidates, and PlanValidator rejects any plan step naming a
        # tool `self.tools` doesn't contain (app/orchestration/
        # validator.py). Registering computer tools onto `self.tools`
        # would therefore silently make them available to M15 Job
        # planning and to AutomationEngine's automation actions - neither
        # of which this milestone permits. Keeping them in a wholly
        # separate registry object that JobEngine/AutomationEngine are
        # never given a reference to is what makes that exclusion
        # DETERMINISTIC in application code (an absent object, not a
        # runtime check an LLM or a Job plan could ever influence) -
        # see `_resolve_tool_registry` below for the only place this
        # registry is ever consulted.
        self.computer_tools = computer_tools
        # Routes a user's reply to a WAITING interaction (a question or
        # confirmation a background Job is waiting on - see app/orchestration/
        # interaction_resolver.py). `None` (every existing caller/test) means
        # interactions aren't in use and chat behaves exactly as before. It
        # decides only WHICH interaction a message answers and hands that,
        # with the untrusted text, to JobEngine's atomic transition; it never
        # runs a tool.
        self.interaction_resolver = interaction_resolver
        # Understands a goal-driven (TASK) request before a Job is started for
        # it (see app/handle_this/service.py): a duck-typed object whose
        # `handle(text, conversation_id)` returns an outcome carrying `reply`,
        # `job_params` (None unless a Job should be started) and
        # `already_running_reply`. `None` (every existing caller/test) means a
        # TASK request starts a Job exactly as it always did. When set, the
        # service only ever RETURNS data: the Job is still created here,
        # through the ToolRegistry's own `create_job` tool, and planned,
        # validated and executed by the existing M15 machinery. A request that
        # needs clarification, is unsupported or could not be understood
        # creates no Job and calls no tool.
        self.handle_this = handle_this
        # Persists what the USER plainly says or asks to be remembered ("I
        # prefer dark mode", "remember that ...", "set my goal to ..."), under a
        # deterministic write policy (see app/memory_policy/): a duck-typed
        # object with `command_reply(message, conversation_id)`, `observe(message,
        # conversation_id)` and
        # `redact(message)` (the form of a message that may be kept in the
        # conversation history). It reads
        # only the user's own message, only on the ordinary conversation path
        # (DIRECT), and never calls a model. `None` (every existing caller/test)
        # means nothing is ever written from a chat message, exactly as before.
        self.memory_writer = memory_writer
        # The user's own statement of what they are doing right now ("I'm in class", "I'm free now") - a
        # duck-typed object with `command_reply(message, conversation_id)` (see
        # app/notifications/activity_context.py). It records a TEMPORARY context for the attention
        # policy: never memory, never inferred. Only on the ordinary conversation path (DIRECT), and
        # only when the whole message is such a statement. `None` (every existing caller/test): off.
        self.activity_context = activity_context
        # Duck-typed: `take_escalations()` -> items with `.event.text` (see AttentionGate). URGENT events
        # that no permitted channel could carry are shown in the next chat reply - the one surface the
        # user is looking at - instead of being lost. `None` (the default): nothing is shown.
        self.escalation_surface = escalation_surface
        # The read-only Morning Brief (app/morning_brief/): a duck-typed object with
        # `command_reply(message, conversation_id)` that answers a message which is, in its
        # entirety, a request for the brief ("morning brief", "what do I need to know today?").
        # It only READS existing systems and never runs a tool, makes a Job or changes any state.
        # `None` (every existing caller/test): a message like that is ordinary conversation.
        self.morning_brief = morning_brief
        # The user-facing device operations (app/devices/operations.py): a duck-typed object with
        # `command_reply(message, conversation_id)` that answers a few whole-message device phrasings
        # ("what devices do you have?", "is my phone connected?", "revoke my phone" - the last one only
        # ASKS, through the existing confirmation). `None` (every existing caller/test): ordinary chat.
        self.device_operations = device_operations
        # The user-facing reminder operations (app/reminders/operations.py): a duck-typed object with
        # `command_reply(message, conversation_id)` that answers a whole-message reminder request ("remind me tomorrow at
        # 7 PM to study", "what reminders do I have?", "pause my study reminder"). It acts only through the reminder
        # tools on `self.tools`. `None` (every existing caller/test): such a message is ordinary conversation.
        self.reminder_operations = reminder_operations
        # The user-facing Scheduled Job operations (app/scheduled_operations/operations.py): a duck-typed object with
        # `command_reply(message, conversation_id, device_context=None)` that answers a whole-message request to schedule recurring or
        # timed READ-ONLY work ("every morning at 8, research X"), or to list / pause / resume / cancel one. Creating one only ASKS
        # (the existing confirmation); nothing runs until it is approved. `None` (every existing caller/test): ordinary chat.
        self.scheduled_job_operations = scheduled_job_operations
        # Focus (app/focus/): a duck-typed object with `command_reply(message, conversation_id)` that answers a whole-message
        # "I need to finish my <X> project today" / "what should I work on?" from the existing task, calendar and World Model
        # data, ranked deterministically. READ-ONLY: it starts nothing, writes nothing and calls no model. The last door on the
        # ordinary conversation path - every other door claims its own phrases first. `None` (every existing caller/test): ordinary chat.
        self.focus = focus
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
        # M17.1-E: when `tool_call_builder` isn't explicitly injected, its
        # GeminiToolCallProvider is built around `self.provider` - the SAME
        # AIProviderManager (or directly-injected AIProvider/test fake)
        # every other path already uses - instead of a hardcoded, always-
        # Gemini-only `GeminiProvider()`. See GeminiToolCallProvider's own
        # docstring for why this closes the M14-era tool-call fallback gap
        # without changing any of its request/response/validation logic:
        # `generate_reply` is called exactly the same way, only now against
        # whatever provider Core itself was constructed with.
        self.tool_call_builder = tool_call_builder or ToolCallBuilder(
            GeminiToolCallProvider(self.provider)
        )
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
        self, message: str, conversation_id: str | None = None, *, device_context=None
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
            message, conversation_id, device_context
        )
        return reply, conversation_id, sources, job_id

    def handle_chat_with_context(
        self, message: str, conversation_id: str | None = None, *, device_context=None
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
            message, conversation_id, device_context
        )
        return reply, conversation_id, sources, job_id, snapshot

    def handle_chat_with_reasoning(
        self, message: str, conversation_id: str | None = None, *, device_context=None
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
            message, conversation_id, device_context
        )
        return reply, conversation_id, sources, job_id, snapshot, reasoning

    def handle_chat_with_planning(
        self, message: str, conversation_id: str | None = None, *, device_context=None
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
        return self._dispatch(message, conversation_id, device_context)

    def _dispatch(
        self, message: str, conversation_id: str | None, device_context=None
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
        facts = with_derived_facts(self.memory.get_facts(), datetime.now().astimezone().date())

        try:
            decision = self.capability_router.route(message)
        except CapabilityDecisionError as exc:
            raise ProviderError(f"Failed to decide a capability: {exc}") from exc

        # A reply to a waiting interaction (an explicit `answer <code>: ...`,
        # or a yes/no / free-text answer to something surfaced to THIS
        # conversation) is handled here, before ordinary dispatch. Only a
        # DIRECT-classified message (or a bare yes/no) can be an implicit
        # answer: tool use, search, tasks, status and self-development are
        # commands/questions and always win. A resolver fault must never
        # break ordinary chat - it just means "not an interaction reply".
        interaction_reply: str | None = None
        if self.interaction_resolver is not None:
            try:
                outcome = self.interaction_resolver.handle(
                    message,
                    conversation_id,
                    may_be_answer=decision.capability
                    in (Capability.DIRECT, Capability.CONFIRMATION),
                    # A message routed to SEARCH only by a freshness word ("...I
                    # can spend right now") may still answer a question that
                    # positively recognises it; an explicit search never does.
                    may_answer_if_recognised=decision.capability is Capability.SEARCH
                    and not is_explicit_search_request(message),
                )
                if outcome.handled:
                    interaction_reply = outcome.reply
            except Exception:  # noqa: BLE001 - see above
                interaction_reply = None

        # An explicit memory command ("remember that I prefer X", "set my goal
        # to X") is answered by the write policy itself - it is not a question
        # for the model. Only the user's own message, only on the ordinary
        # conversation path; a fault here is "not a memory command".
        memory_reply: str | None = None
        # An activity statement is a narrow whole-message grammar, so it may also come in as SEARCH
        # when the router classified it only on a freshness word ("I'm free now") - but never an
        # explicit search request. Anything else (tool, task, status, self-development) always wins.
        if (
            self.activity_context is not None
            and interaction_reply is None
            and (
                decision.capability is Capability.DIRECT
                or (decision.capability is Capability.SEARCH and not is_explicit_search_request(message))
            )
        ):
            try:
                memory_reply = self.activity_context.command_reply(message, conversation_id)
            except Exception:  # noqa: BLE001 - a fault here is "not an activity statement"
                memory_reply = None
        # The Morning Brief is read the same way: a whole-message request only, on the ordinary
        # conversation path (DIRECT, or SEARCH when only a freshness word like "today" put it
        # there - never an explicit search, tool, task, status or self-development request).
        if (
            memory_reply is None
            and self.morning_brief is not None
            and interaction_reply is None
            and (
                decision.capability is Capability.DIRECT
                or (decision.capability is Capability.SEARCH and not is_explicit_search_request(message))
            )
        ):
            try:
                memory_reply = self.morning_brief.command_reply(message, conversation_id)
            except Exception:  # noqa: BLE001 - a fault here is "not a brief request"
                memory_reply = None
        # Device operations are read the same way: a whole-message phrasing only, on the ordinary
        # conversation path. Reads answer from the registry; "revoke my phone" only ASKS (existing
        # confirmation). Registering and trusting a device have no chat path at all.
        if (
            memory_reply is None
            and self.device_operations is not None
            and interaction_reply is None
            and (
                decision.capability is Capability.DIRECT
                or (decision.capability is Capability.SEARCH and not is_explicit_search_request(message))
            )
        ):
            try:
                memory_reply = (
                    self.device_operations.command_reply(message, conversation_id, device_context=device_context)
                    if device_context is not None
                    else self.device_operations.command_reply(message, conversation_id)
                )
            except Exception:  # noqa: BLE001 - a fault here is "not a device request"
                memory_reply = None
        # Reminders are read the same way: a whole-message request only, on the ordinary conversation path (DIRECT, or
        # SEARCH when only a freshness word like "tomorrow" put it there - never an explicit search, tool, task, status
        # or self-development request, and never a multi-step goal, which Handle This receives as a TASK).
        if (
            memory_reply is None
            and self.reminder_operations is not None
            and interaction_reply is None
            and (
                decision.capability is Capability.DIRECT
                or (decision.capability is Capability.SEARCH and not is_explicit_search_request(message))
            )
        ):
            try:
                memory_reply = self.reminder_operations.command_reply(message, conversation_id)
            except Exception:  # noqa: BLE001 - a fault here is "not a reminder request"
                memory_reply = None
        # Scheduled Jobs are read the same way, with one difference: a request to do something EVERY day / at a TIME is claimed
        # here even when the router would otherwise have sent it to Handle This (TASK) or to search - otherwise "research X, compare
        # Y and report every Monday at 8" would be started once, immediately, instead of being proposed. The door's own whole-message
        # grammar (a timing AND an action verb) decides whether it is theirs; anything else is left exactly as routed. An explicit
        # tool command, status, confirmation or self-development request never reaches it.
        if (
            memory_reply is None
            and self.scheduled_job_operations is not None
            and interaction_reply is None
            and decision.capability in (Capability.DIRECT, Capability.SEARCH, Capability.TASK)
        ):
            try:
                memory_reply = (
                    self.scheduled_job_operations.command_reply(message, conversation_id, device_context=device_context)
                    if device_context is not None
                    else self.scheduled_job_operations.command_reply(message, conversation_id)
                )
            except Exception:  # noqa: BLE001 - a fault here is "not a scheduled job request"
                memory_reply = None
        if (
            memory_reply is None
            and self.memory_writer is not None
            and interaction_reply is None
            and decision.capability is Capability.DIRECT
        ):
            try:
                memory_reply = self.memory_writer.command_reply(message, conversation_id)
            except Exception:  # noqa: BLE001
                memory_reply = None
        # The user's own questions about their memory ("what did I note down?", "what are my preferences?", "how old
        # am I?") and an explicit "forget that preference" are answered from the existing stores by the same service -
        # on the ordinary conversation path only (DIRECT, or SEARCH when only a freshness word like "today" put it there).
        if (
            memory_reply is None
            and self.memory_writer is not None
            and interaction_reply is None
            and (
                decision.capability is Capability.DIRECT
                or (decision.capability is Capability.SEARCH and not is_explicit_search_request(message))
            )
        ):
            try:
                memory_reply = self.memory_writer.recall_reply(message, conversation_id)
            except Exception:  # noqa: BLE001 - a fault here is "not a memory question"
                memory_reply = None
        # Focus is the last door, read the same way: a whole-message request only, on the ordinary conversation path (DIRECT,
        # or SEARCH when only a freshness word like "today" put it there). Never an explicit search, tool, task, status,
        # confirmation or self-development request, never a message another door already claimed, and never while an
        # interaction (a question or a confirmation) is waiting on this conversation's next message - that message belongs to
        # the interaction flow even when the resolver did not itself take it. Anything its own grammar does not recognise
        # stays ordinary conversation.
        if (
            memory_reply is None
            and self.focus is not None
            and interaction_reply is None
            and (
                decision.capability is Capability.DIRECT
                or (decision.capability is Capability.SEARCH and not is_explicit_search_request(message))
            )
            and not self._interaction_awaiting_reply(conversation_id)  # last: a lookup, asked only of a message that could be Focus's
        ):
            try:
                memory_reply = self.focus.command_reply(message, conversation_id)
            except Exception:  # noqa: BLE001 - a fault here is "not a focus request"
                memory_reply = None

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
            # `device_context` is who the Device Gateway AUTHENTICATED as the sender (None for the ordinary,
            # unauthenticated /chat). It rides on the RequestContext - request-scoped, never derived from the
            # message, never stored, never a permission - and the Context Engine carries it into the snapshot.
            request_context = build_request_context(
                message,
                capability=decision.capability,
                **({"device_context": device_context} if device_context is not None else {}),
            )
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

        # Computed ONCE, before dispatch, and reused by the SELF_
        # DEVELOPMENT/SELF_DEV_APPROVAL/plain-fallback branches below -
        # see `_active_self_dev_sessions`'s own docstring for exactly
        # what "active" means and why this is conversation-scoped only.
        active_self_dev_sessions = self._active_self_dev_sessions(conversation_id)

        if interaction_reply is not None:
            reply, sources = interaction_reply, []
        elif memory_reply is not None:
            reply, sources = memory_reply, []
        elif decision.capability is Capability.SEARCH:
            reply, sources = self._handle_search(message, history, facts)
        elif decision.capability is Capability.TOOL:
            reply, sources = self._handle_tool(message, decision)
        elif decision.capability is Capability.TASK:
            reply, sources, job_id = self._handle_task(
                decision, conversation_id, context_snapshot, message=message,
                **({"device_context": device_context} if device_context is not None else {}),
            )
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
                # to call `_handle_direct`. Deliberately NEVER guarded by
                # `active_self_dev_sessions` - an M15 Job's own "yes"/"no"
                # must keep working exactly as before regardless of any
                # self-development session also pending in this same
                # conversation (see `_active_self_dev_sessions`'s own
                # docstring), and a self-development session's approval
                # has its own, separate SELF_DEV_APPROVAL capability.
                reply, sources, reasoning_result, planning_directive = (
                    self._fallback_to_direct(
                        message, history, facts, request_context, context_snapshot
                    )
                )
        elif decision.capability is Capability.SELF_DEVELOPMENT:
            if active_self_dev_sessions:
                # A self-development session is ALREADY awaiting an
                # explicit decision in THIS conversation - a new
                # development request must never silently start a
                # second, competing session (which would effectively
                # abandon the first) - see `_format_pending_self_dev_
                # notice`. No engine call, no LLM call.
                reply, sources = (
                    self._format_pending_self_dev_notice(active_self_dev_sessions),
                    [],
                )
            elif self.self_development_engine is None:
                # No engine configured (every existing caller/test, and
                # the module-level `core` singleton unless app/main.py
                # wires one) - a clearly-identified development request
                # falls back to ordinary DIRECT handling rather than a
                # dead end, the exact same "hint, not a guarantee" fallback
                # CONFIRMATION's own branch above already uses.
                reply, sources, reasoning_result, planning_directive = (
                    self._fallback_to_direct(
                        message, history, facts, request_context, context_snapshot
                    )
                )
            else:
                reply, sources = self._handle_self_development(
                    decision, message, conversation_id
                )
        elif decision.capability is Capability.SELF_DEV_APPROVAL:
            approval_result = (
                self._handle_self_dev_approval(decision, conversation_id)
                if self.self_development_engine is not None
                else None
            )
            if approval_result is not None:
                reply, sources = approval_result
            elif active_self_dev_sessions:
                # The named gate ("approve the commit", say) didn't match
                # what's actually pending (e.g. still awaiting plan
                # approval) - tell the user the real pending gate instead
                # of falling through to a generic, LLM-authored reply.
                reply, sources = (
                    self._format_pending_self_dev_notice(active_self_dev_sessions),
                    [],
                )
            else:
                # Either no engine is configured, or no session in ANY
                # conversation is actually awaiting the named gate right
                # now - the same "hint, not a guarantee" fallback
                # CONFIRMATION/SELF_DEVELOPMENT already use above.
                reply, sources, reasoning_result, planning_directive = (
                    self._fallback_to_direct(
                        message, history, facts, request_context, context_snapshot
                    )
                )
        elif active_self_dev_sessions:
            # Every other capability this router can produce for an
            # otherwise-unclassified message (DIRECT/SEARCH's own
            # "else") - a self-development session is awaiting an
            # explicit decision in THIS conversation, so this message is
            # treated as a self-development follow-up rather than being
            # handed to a generic, ungrounded LLM reply that could
            # invent unrelated content (V1's own explicit requirement).
            # SEARCH/TOOL/TASK/TASK_STATUS/CONFIRMATION are unaffected -
            # they're each already handled by their own branch above,
            # never reaching here.
            reply, sources = self._format_pending_self_dev_notice(active_self_dev_sessions), []
        else:
            reply, sources, reasoning_result, planning_directive = self._fallback_to_direct(
                message, history, facts, request_context, context_snapshot
            )

        # A plain first-person statement in an ordinary message ("I prefer dark
        # mode.") may be remembered under the write policy; the user is told
        # briefly when something was saved, asked about or refused (never for a
        # candidate or a passing remark). Best-effort: never breaks chat.
        if (
            self.memory_writer is not None
            and interaction_reply is None
            and memory_reply is None
            and decision.capability is Capability.DIRECT
        ):
            try:
                note = self.memory_writer.observe(message, conversation_id)
            except Exception:  # noqa: BLE001
                note = None
            if note:
                reply = f"{reply}\n\n{note}"

        # An URGENT event that no permitted channel could carry is not lost: it is told to the user here,
        # in the reply to their own message. Nothing is delivered through any channel and no channel
        # restriction is lifted; each such event is shown once. Best-effort: never breaks chat.
        if self.escalation_surface is not None:
            try:
                urgent_items = self.escalation_surface.take_escalations()
            except Exception:  # noqa: BLE001
                urgent_items = []
            if urgent_items:
                lines = " ".join(item.event.text for item in urgent_items)
                reply = f"{reply}\n\nThis was urgent and I had no way to reach you: {lines}"

        # Surface anything now waiting on the user in this conversation (its
        # own Jobs' and background Jobs' questions/confirmations), which is
        # also what arms a plain reply to it. Best-effort: never breaks chat.
        if self.interaction_resolver is not None:
            try:
                notice = self.interaction_resolver.surface(conversation_id)
            except Exception:  # noqa: BLE001
                notice = None
            if notice:
                reply = f"{reply}\n\n{notice}"

        stored_message = message
        if self.memory_writer is not None:
            # A message that looks like it holds a credential is not kept verbatim
            # in the conversation history (see MemoryWriteService.redact).
            try:
                stored_message = self.memory_writer.redact(message)
            except Exception:  # noqa: BLE001
                stored_message = message
        self.memory.add_message(conversation_id, "user", stored_message)
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

        M17.1-C: `decision.tool_name`, when given, is resolved against
        `self.computer_tools` FIRST (via `_resolve_tool_registry` below),
        falling back to `self.tools` otherwise - so an interactive chat
        request naming a registered computer tool reaches it, while
        M15/AutomationEngine (which only ever hold `self.tools`) never
        see it exist at all. The AMBIGUOUS case (`decision.tool_name is
        None` - nothing named, ToolCallBuilder must pick from a full
        tool list) deliberately still only ever offers `self.tools`'s own
        candidates: computer tools are reachable ONLY by an explicit,
        named invocation, never by an LLM picking one from an ambiguous
        "use tool" message - a narrower, more deliberate reachability
        requirement than this milestone strictly needs, kept anyway
        since nothing calls for offering them any other way yet.

        Explicit tool commands (a message that itself says "use tool X
        ..." / "use the X tool ...", one step or several joined with
        "then" / "and then" / "followed by") are understood by
        app/tools/command.py's ExplicitToolCommandPlanner, NOT by the
        legacy ToolCallBuilder path below: the whole command is parsed,
        every tool and argument resolved against the registered tool's own
        schema, and every step validated (types, the tool's own value
        check, permission) BEFORE anything runs. Only a fully valid
        `ToolSequence` is executed (`_execute_tool_sequence`); any problem
        anywhere rejects the entire command with NOTHING executed. A
        message that isn't an explicit named command (a bare "use tool",
        or a decision a router made from other wording) still takes the
        legacy path below, byte-for-byte unchanged.
        """
        planner = ExplicitToolCommandPlanner(
            self._resolve_tool_registry, self.tool_call_builder
        )
        try:
            outcome = planner.plan(message)
        except ToolCallError as exc:
            # Raised only while planning (nothing has executed), so the
            # 502 convention below can't hide a completed side effect.
            raise ProviderError(f"Failed to construct a tool call: {exc}") from exc

        if isinstance(outcome, CommandRejection):
            return outcome.render(), []
        if outcome is not None:
            return self._execute_tool_sequence(outcome), []

        registry = self._resolve_tool_registry(decision.tool_name)
        try:
            tool_call = self.tool_call_builder.build(
                message, registry, decision.tool_name
            )
        except ToolCallIncompleteError as exc:
            return str(exc), []
        except ToolCallError as exc:
            raise ProviderError(f"Failed to construct a tool call: {exc}") from exc

        result = registry.execute(tool_call.tool_name, params=tool_call.parameters)
        return self._format_tool_result(result), []

    def _execute_tool_sequence(self, sequence: ToolSequence) -> str:
        """PHASE 2: run an already fully validated `ToolSequence`, in
        order, one `ToolRegistry.execute` per step - which still applies
        the registry's own permission gate to every step. The sequence
        (not the original sentence) is the only input: nothing is
        re-parsed, no tool is added, and steps never repeat.

        Execution stops after the first step that doesn't return
        ToolResultStatus.SUCCESS; that step's real result is reported, the
        earlier steps' real results are kept, and the remaining steps are
        stated as NOT run - never reported as if they had happened. A
        single-step sequence replies with just that step's result."""
        steps = sequence.steps
        lines: list[str] = []
        for position, step in enumerate(steps):
            registry = self._resolve_tool_registry(step.tool_name)
            result = registry.execute(step.tool_name, params=step.parameters)
            text = self._format_tool_result(result)
            lines.append(text if len(steps) == 1 else f"{step.index}. {text}")
            if result.status is not ToolResultStatus.SUCCESS and position < len(steps) - 1:
                skipped = ", ".join(str(later.index) for later in steps[position + 1 :])
                lines.append(
                    f"Stopped after step {step.index}; step(s) {skipped} were not run."
                )
                break
        return "\n".join(lines)

    def _resolve_tool_registry(self, tool_name: str | None) -> ToolRegistry:
        """The ONLY place `self.computer_tools` is ever consulted (M17.1-C).
        A NAMED tool (`tool_name` not `None`) that this registry actually
        contains dispatches there; every other case - including
        `self.computer_tools` being `None` (every existing caller/test)
        and the ambiguous `tool_name is None` case - falls back to
        `self.tools`, byte-for-byte the same registry (and therefore the
        same behavior) `_handle_tool` always used before this method
        existed.

        This is what makes the interactive-vs-M15 separation deterministic
        in application code rather than a runtime permission decision: a
        message can only ever name a TOOL by string - which registry (if
        any) actually contains a tool by that name is fixed entirely by
        what THIS process registered where (see app/main.py), never by
        anything in the message/decision itself. There is no parameter,
        capability, or tool argument that selects `self.computer_tools`
        directly - only `Tool.name` membership does, and that membership
        is application-controlled, not LLM-controlled.
        """
        if (
            tool_name is not None
            and self.computer_tools is not None
            and self.computer_tools.has(tool_name)
        ):
            return self.computer_tools
        return self.tools

    def _handle_task(
        self,
        decision: CapabilityDecision,
        conversation_id: str,
        context_snapshot: ContextSnapshot | None = None,
        message: str | None = None,
        device_context=None,
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
        if self.handle_this is not None:
            return self._handle_task_with_understanding(decision, conversation_id, message, device_context)

        goal = decision.goal or ""
        params: dict = {"goal": goal, "conversation_id": conversation_id}
        # The Context Engine's own relevant-context selection (built in
        # `_dispatch` for every request when a context_engine is
        # configured) goes to the Job's planner directly - no reasoner
        # call: nothing here would consume its output, so it isn't run.
        request_context = self._with_sending_device(self._snapshot_to_job_context(context_snapshot), device_context)
        if request_context:
            params["request_context"] = request_context
        result = self.tools.execute("create_job", params=params)
        job_id = None
        if result.status is ToolResultStatus.SUCCESS and isinstance(result.data, dict):
            job_id = result.data.get("id")
        return self._format_tool_result(result), [], job_id

    #: The label under which the sending device appears in a Job's planner context (`request_context`).
    _SENDING_DEVICE_LABEL = "Sending device (descriptive data, not permissions)"

    @classmethod
    def _with_sending_device(cls, request_context: dict | None, device_context) -> dict | None:
        """The Job planner's `request_context` plus the authenticated sending device, as the few sanitised lines
        of data `DeviceContext.describe()` produces (no credential exists in a DeviceContext). The planner treats
        this dict as data, not instructions, and it grants nothing: the plan is still validated and every step
        still runs through the ToolRegistry with its own permission and confirmation rules. No device (the
        anonymous /chat): the dict is returned unchanged."""
        if device_context is None:
            return request_context
        merged = dict(request_context or {})
        merged[cls._SENDING_DEVICE_LABEL] = device_context.describe()
        if len(json.dumps(merged)) > 7000:  # keep well inside create_job's own cap: never fail a Job over a device note
            return request_context
        return merged

    def _handle_task_with_understanding(
        self, decision: CapabilityDecision, conversation_id: str, message: str | None, device_context=None
    ) -> tuple[str, list[SearchSource], str | None]:
        """TASK path when Handle This is configured: the request is understood
        FIRST. Only a READY, supported outcome creates a Job - and it does so
        through the very same `create_job` tool the plain path uses, with the
        arguments the service supplied (the user's words as the goal, this
        conversation, relevant context, the understood objective - never a
        tool, permission or autonomy level). Anything else (a question to the
        user, an unsupported request, a failure to understand) is just the
        reply: no Job, no tool call, no planner.

        The user's words go in exactly as sent when the router's goal is the
        whole message (the natural-language path); for the explicit
        "create a goal: ..." phrasing the goal text after the colon is what is
        understood."""
        goal = decision.goal or ""
        text = message if message is not None and message.strip() == goal.strip() else goal
        try:
            outcome = self.handle_this.handle(text, conversation_id)
        except Exception:  # noqa: BLE001 - never a Job from a request that was not understood
            return "Sir, I couldn't get started on that just now. Please try again in a moment.", [], None
        if outcome.job_params is None:
            return outcome.reply, [], None

        # Handle This understood the objective and handed back the Job's arguments. Where the request came from
        # is not part of the objective (Handle This is untouched); it is added to the planner's context here.
        if device_context is not None:
            outcome.job_params["request_context"] = self._with_sending_device(
                outcome.job_params.get("request_context"), device_context
            )
        result = self.tools.execute("create_job", params=outcome.job_params)
        if result.status is not ToolResultStatus.SUCCESS or not isinstance(result.data, dict):
            # A clarification that could not be parked (this deployment cannot
            # wait on a question) falls back to asking in plain words: nothing
            # was created.
            fallback = getattr(outcome, "fallback_reply", None)
            return (fallback or self._format_tool_result(result)), [], None
        pending = result.data.get("pending_interaction")
        if isinstance(pending, dict) and self.interaction_resolver is not None:
            # The reply below IS the question being put to the user in this
            # conversation: record that, so a plain answer here is recognised.
            try:
                self.interaction_resolver.mark_surfaced(pending["id"], conversation_id)
            except Exception:  # noqa: BLE001 - best effort; the question is also surfaced next turn
                pass
        if not result.data.get("already_started"):
            reply = outcome.reply
        elif isinstance(pending, dict) or not getattr(outcome, "underway_reply", None):
            reply = outcome.already_running_reply  # still waiting on that very question
        else:
            reply = outcome.underway_reply  # its question was answered: it is simply under way
        return reply, [], result.data.get("id")

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
        {"pending", "planning", "running", "awaiting_input", "awaiting_confirmation", "paused"}
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
        if self.interaction_resolver is not None:
            # A Job with a pending interaction is the interaction system's to
            # resolve (recency-, conversation- and DANGEROUS-aware); a bare
            # "yes" must not approve it through this older, direct path.
            pending = [j for j in pending if not self.interaction_resolver.governs_job(j["id"])]
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
        self, decision: CapabilityDecision, message: str, conversation_id: str
    ) -> tuple[str, list[SearchSource]]:
        """SELF_DEVELOPMENT path: dispatches to
        `self.self_development_engine.start(...)`, which proposes a
        DevelopmentPlan and stops there - no branch is created, no file
        is touched, nothing is committed or rolled back (see
        app/self_development/engine.py:SelfDevelopmentEngine.start's own
        docstring). `sources` is always `[]`.

        `conversation_id` is passed through as `start()`'s own
        `conversation_id` argument purely so a LATER chat message in
        THIS SAME conversation can be deterministically associated back
        to the resulting session (see `_handle_self_dev_approval`) -
        `start()`'s own decision of what plan to propose is completely
        unaffected by it.

        Continuing this session - approving the plan, a SECURITY_EDIT,
        or a commit - now happens through `_handle_self_dev_approval`
        (a later chat message naming the gate explicitly), not only
        through SelfDevelopmentEngine's own directly-callable Python API
        - see that method's own docstring for the exact association/
        approval rules."""
        request_text = decision.goal or message
        session = self.self_development_engine.start(
            request_text, conversation_id=conversation_id
        )
        return self._format_self_dev_session(session), []

    @staticmethod
    def _format_self_dev_session(session: DevelopmentSession) -> str:
        """Pure serializer - describes `session`'s own already-decided
        state, never a second decision-maker. Never includes a secret
        file's contents (nothing here reads file content at all) and
        never claims an action happened that didn't (a plan is only
        ever described as "proposed," never as "done")."""
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
            "this plan is explicitly approved. Reply 'Approve the plan.' to "
            "continue, or 'Reject the plan.' to cancel."
        )
        return "\n".join(lines)

    #: Maps a SELF_DEV_APPROVAL gate name (app/capability.py:
    #: _SELF_DEV_APPROVAL_PATTERN) to the exact SessionStage a session
    #: must be in for that gate's approval to apply to it - the only
    #: place this association is made, so "approve the commit" can never
    #: accidentally resolve a session that's actually still awaiting
    #: plan approval, or vice versa.
    _SELF_DEV_GATE_STAGES = {
        "plan": SessionStage.PLAN_PROPOSED,
        "security_edit": SessionStage.AWAITING_SECURITY_APPROVAL,
        "commit": SessionStage.AWAITING_COMMIT_APPROVAL,
    }

    #: Derived from `_SELF_DEV_GATE_STAGES` (never a second, hand-
    #: maintained list) - the stages that mean "this session is
    #: genuinely awaiting one of the three explicit approval gates
    #: right now," used by `_active_self_dev_sessions` to decide
    #: whether a conversation has a session actively awaiting
    #: interaction. Deliberately excludes every terminal/declined stage
    #: (COMMITTED, PLAN_REJECTED, SECURITY_EDIT_DECLINED, COMMIT_
    #: DECLINED, ROLLED_BACK, CHECKS_FAILED, ...) - once a session is
    #: resolved one way or another, it stops being "active" and a new
    #: request in the same conversation is free to start a fresh one.
    _AWAITING_SELF_DEV_INTERACTION_STAGES = frozenset(_SELF_DEV_GATE_STAGES.values())

    def _interaction_awaiting_reply(self, conversation_id: str) -> bool:
        """Whether a waiting interaction owns this conversation's next message (the resolver's own definition of "armed"). With no
        resolver there is none. If the resolver cannot say, the answer is "yes": a door that would otherwise take the message
        stands down rather than guess."""
        if self.interaction_resolver is None:
            return False
        try:
            return bool(self.interaction_resolver.awaiting_reply(conversation_id))
        except Exception:  # noqa: BLE001 - fail closed
            return True

    def _active_self_dev_sessions(self, conversation_id: str) -> list[DevelopmentSession]:
        """Every session THIS SAME conversation itself started that is
        still sitting at one of the three explicit approval gates -
        NEVER a session from a different conversation (see `_handle_
        self_dev_approval`'s own docstring for why cross-conversation
        lookup is never done at all: SECURITY_EDIT/COMMIT are
        DANGEROUS-tier, so association must be exact).

        Used for two things, both in `_dispatch`: refusing to start a
        REDUNDANT new self-development session while one is already
        pending in this conversation (which would otherwise silently
        abandon the pending one), and refusing to let a message that
        doesn't resolve the pending gate fall through to a generic,
        ungrounded DIRECT/SEARCH reply - the exact failure mode a live
        test surfaced: a follow-up message that didn't match any
        approval gate reached ordinary DIRECT handling and the LLM
        answered with an invented, unrelated "plan" of its own, with no
        connection to the real, still-pending session at all.

        Deliberately does NOT affect TOOL/TASK/TASK_STATUS/CONFIRMATION -
        each of those is handled by its own `_dispatch` branch before
        this is ever consulted, so an M15 Job's own pending confirmation
        keeps working exactly as before regardless of a self-development
        session also being active in the same conversation, and vice
        versa (see tests/test_core_self_dev_approval_integration.py's
        own M15-isolation tests)."""
        if self.self_development_engine is None:
            return []
        return [
            session
            for session in self.self_development_engine.sessions.values()
            if session.conversation_id == conversation_id
            and session.stage in self._AWAITING_SELF_DEV_INTERACTION_STAGES
        ]

    @staticmethod
    def _format_pending_self_dev_notice(sessions: list[DevelopmentSession]) -> str:
        """Pure, deterministic, LLM-free text - never calls the
        provider, never starts or advances any session. Tells the user
        exactly which reply resolves the pending session, and restates
        the conversation_id so a client that isn't already tracking it
        has an explicit, unmissable reminder to send it back (the
        actual client-contract gap a live test exposed: `/chat` already
        returns `conversation_id` in every response, and its own
        docstring in app/main.py already says to pass it back, but
        nothing in the reply text itself said so)."""
        if len(sessions) > 1:
            ids = ", ".join(session.id for session in sessions)
            return (
                "You have more than one self-development session awaiting "
                f"a decision in this conversation: {ids}. Resolve one "
                "before starting or continuing another."
            )

        session = sessions[0]
        gate_phrase = {
            SessionStage.PLAN_PROPOSED: "the plan",
            SessionStage.AWAITING_SECURITY_APPROVAL: "the security edit",
            SessionStage.AWAITING_COMMIT_APPROVAL: "the commit",
        }[session.stage]
        return (
            f"You already have a self-development session ({session.id}) "
            f"awaiting a decision on {gate_phrase}. Reply 'Approve "
            f"{gate_phrase}.' or 'Reject {gate_phrase}.' before starting a "
            "new request. Keep using conversation_id "
            f"'{session.conversation_id}' so this reply continues the same "
            "session."
        )

    def _fallback_to_direct(
        self,
        message: str,
        history: list[Message],
        facts,
        request_context: RequestContext | None,
        context_snapshot: ContextSnapshot | None,
    ) -> tuple[str, list[SearchSource], ReasoningResult | None, PlanningDirective | None]:
        """The shared "nothing more specific actually applies, answer
        directly" tail every fallback case uses (CONFIRMATION/SELF_
        DEVELOPMENT/SELF_DEV_APPROVAL's own "nothing pending" cases, and
        the plain DIRECT/SEARCH catch-all) - one copy instead of four,
        so they can never drift out of sync. Behavior is identical to
        each of those branches' own inlined version before this was
        factored out."""
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
        return reply, sources, reasoning_result, planning_directive

    def _handle_self_dev_approval(
        self, decision: CapabilityDecision, conversation_id: str
    ) -> tuple[str, list[SearchSource]] | None:
        """SELF_DEV_APPROVAL path: resolves an explicit "approve/reject
        the plan/security edit/commit" chat message against whichever of
        THIS conversation's own self-development sessions is actually
        sitting at the matching gate right now, then dispatches to the
        existing, already-hardened SelfDevelopmentEngine.approve_plan/
        approve_security_edit/approve_commit - never a new approval
        mechanism.

        `decision.approve`/`decision.self_dev_gate` come ONLY from
        `_SELF_DEV_APPROVAL_PATTERN` matching the user's OWN raw message
        (app/capability.py) - nothing here ever reads plan content, file
        content, or any other LLM output to decide whether something is
        approved, so approval can never be manufactured by the LLM or by
        text living in a repository file.

        Returns `None` (never a reply) when no session in THIS
        conversation is actually awaiting that specific gate right now,
        so `_dispatch` falls back to ordinary DIRECT handling - the same
        "hint, not a guarantee" discipline `_handle_confirmation` already
        uses.

        Deliberately STRICTER than `_handle_confirmation`'s own
        disambiguation: `_handle_confirmation` falls back to considering
        every Job pending across ALL conversations if none match this
        one; this method never widens the search - SECURITY_EDIT and
        COMMIT are DANGEROUS-tier actions (app/tools/base.py), so a
        session started in a DIFFERENT conversation is never even a
        candidate. More than one matching session in THIS conversation
        (only possible if the same person started more than one
        self-development request from here without resolving the first)
        is reported back for the user to disambiguate by session id -
        never guessed."""
        gate = decision.self_dev_gate
        target_stage = self._SELF_DEV_GATE_STAGES.get(gate) if gate else None
        if target_stage is None:
            return None

        candidates = [
            session
            for session in self.self_development_engine.sessions.values()
            if session.stage is target_stage and session.conversation_id == conversation_id
        ]
        if not candidates:
            return None

        if len(candidates) > 1:
            ids = ", ".join(session.id for session in candidates)
            return (
                "You have more than one self-development session awaiting "
                f"{gate} approval: {ids}. Which one do you mean?",
                [],
            )

        session = candidates[0]
        approve = bool(decision.approve)

        if gate == "plan":
            session = self.self_development_engine.approve_plan(session, approved=approve)
        elif gate == "security_edit":
            session = self.self_development_engine.approve_security_edit(
                session, confirmed=approve
            )
        else:  # "commit"
            # A deterministic, already-reviewed commit message - the
            # plan's own objective (already shown to the user in the
            # original plan preview) or the finalize step's own summary
            # of it, NEVER anything freshly generated at approval time,
            # so approving a commit can never be tricked into running
            # arbitrary attacker-influenced text as the commit message
            # either.
            commit_message = (
                session.summary.change_summary
                if session.summary is not None
                else (session.plan.objective if session.plan is not None else "")
            )
            session = self.self_development_engine.approve_commit(
                session, confirmed=approve, message=commit_message
            )

        return self._format_self_dev_approval_result(session, gate), []

    @staticmethod
    def _format_self_dev_approval_result(session: DevelopmentSession, gate: str) -> str:
        """Pure serializer for the state `_handle_self_dev_approval`'s
        engine call just produced - same "describe, never decide"
        discipline as `_format_self_dev_session`."""
        if session.stage is SessionStage.PLAN_REJECTED:
            return "Plan rejected - no branch was created and no files were touched."

        if session.stage is SessionStage.AWAITING_SECURITY_APPROVAL:
            protected = ", ".join(sorted(session.plan.protected_file_paths()))
            return (
                f"Plan approved - development branch '{session.dev_branch}' created. "
                f"This plan touches protected security-critical file(s): {protected}. "
                "That requires a SEPARATE SECURITY_EDIT approval before those files "
                "are edited. Reply 'Approve the security edit.' or 'Reject the "
                "security edit.'"
            )

        if session.stage is SessionStage.SECURITY_EDIT_DECLINED:
            return (
                "SECURITY_EDIT declined - protected files were not edited. The "
                "session's branch still exists for manual review or rollback."
            )

        if session.stage is SessionStage.CHECKS_FAILED:
            return (
                f"Checks failed after {session.repair_attempts} bounded repair "
                "attempt(s) - stopping. No commit approval requested; the branch "
                "still exists for manual review or rollback."
            )

        if session.stage is SessionStage.AWAITING_COMMIT_APPROVAL:
            files = ", ".join(session.edited_files) or "(none)"
            diff = session.summary.git_diff_summary if session.summary else ""
            return (
                f"Checks passed on branch '{session.dev_branch}'. Files changed: "
                f"{files}.\n\nDiff:\n{diff[:2000]}\n\nReply 'Approve the commit.' "
                "or 'Reject the commit.'"
            )

        if session.stage is SessionStage.COMMITTED:
            return f"Committed on branch '{session.dev_branch}'. Nothing was pushed."

        if session.stage is SessionStage.COMMIT_DECLINED:
            return (
                "Commit declined. The branch and its uncommitted changes still "
                "exist for manual review or rollback."
            )

        return f"Session {session.id} is now in stage '{session.stage.value}'."

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
    def _snapshot_to_job_context(
        cls, snapshot: ContextSnapshot | None
    ) -> dict[str, list[str]] | None:
        """A ContextSnapshot as the plain `label -> [strings]` dict a Job
        carries for its planner (`create_job`'s `request_context`), built
        from the SAME `_WORLD_MODEL_CATEGORIES` table
        `_format_world_model_context` uses, so the two can't disagree
        about what a category is called or how an item is described.
        Pure serialization: never re-ranks, filters or adds anything to
        what ContextEngine selected, and never queries the World Model.
        `None` when there is no snapshot or nothing was selected."""
        if snapshot is None:
            return None
        context = {
            label: [describe(item) for item in getattr(snapshot, field_name)]
            for field_name, label, describe in cls._WORLD_MODEL_CATEGORIES
            if getattr(snapshot, field_name)
        }
        return context or None

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

        candidates = getattr(snapshot, "unconfirmed_candidates", None) or []
        if candidates:
            # Not a fact category (and never part of `_WORLD_MODEL_CATEGORIES`, so never
            # handed to a Job's planner): hypotheses, each marked unconfirmed.
            lines.append(
                "- Unconfirmed hunches (NOT facts - the user has not confirmed these; treat only as "
                "possible clues and never state them as true): "
                + ", ".join(f"{c.key}={c.value} (confidence {c.confidence:.2f})" for c in candidates)
            )

        if not lines:
            return ""
        return "What Jarvis currently knows from the user's world model:\n" + "\n".join(lines)

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
        return "What Jarvis's reasoning about this request concluded:\n" + "\n".join(
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
                "asks for that, it goes through Jarvis's existing task system."
            )
        return "What Jarvis's planning about this request concluded:\n" + "\n".join(
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
