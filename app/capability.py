"""
Capability Decision Layer (M6.13, extended in M6.14 with TOOL, and M6.15
with hybrid/AI-fallback routing).

The high-level "who should handle this message" decision - DIRECT, SEARCH,
and TOOL today, more capabilities (MEMORY, EMAIL, CALENDAR, PRINTER,
ROBOTIC_ARM, ...) later - lives here, not in JarvisCore. JarvisCore
(app/core.py) only orchestrates: it asks a CapabilityRouter for a
CapabilityDecision and dispatches on it, never encoding routing rules
itself.

DIRECT, SEARCH, TOOL, and (M15) TASK are implemented - no other
Capability member is added "for the future"; see the README's Postponed
decisions for why. Adding a real future capability means adding an enum
member plus whatever decision logic picks it, without touching
JarvisCore's dispatch mechanics - TASK (M15) is exactly that: a new
member plus a new narrow, deterministic detector in
DeterministicCapabilityRouter, dispatched by JarvisCore._handle_task
(app/core.py) to the existing ToolRegistry's `create_job` tool
(app/orchestration/tools.py) - JarvisCore itself never gains any
orchestration-specific logic beyond that one dispatch line, the same
"Core only orchestrates" discipline TOOL already established.

DeterministicCapabilityRouter generalizes M6.9's SearchRouter
(app/search/router.py) into this layer by delegating to it for DIRECT vs
SEARCH rather than duplicating its regex rules - SearchRouter (and its own
tests, tests/test_search_router.py) stays the single source of truth for
that matching; it's no longer the top-level interface JarvisCore talks to
directly for the decision. TOOL detection (M6.14, extended in M6.16 with a
second narrow phrasing) is a separate, purely textual check - see
DeterministicCapabilityRouter's docstring.

M6.15 adds HybridCapabilityRouter: deterministic routing stays the fast
path for every message it can already confidently resolve (which is still
almost every message); only a message the deterministic layer itself
flags as genuinely ambiguous - see HybridCapabilityRouter's docstring for
the exact, narrow boundary - falls back to GeminiCapabilityDecisionProvider,
built entirely on the existing AIProvider/GeminiProvider (no second Gemini
client, no new config, no paid API). Gemini only ever returns a decision;
it never executes anything - JarvisCore._handle_tool and ToolRegistry
remain the only things that ever run a tool, exactly as in M6.14, and a
Gemini decision is strictly validated before JarvisCore ever sees it.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.search.router import SearchDecision, SearchRouter
from app.tools.command import match_invocation_head


class Capability(Enum):
    """A handler Rhasspy can dispatch a message to.

    DIRECT, SEARCH, TOOL, and TASK were the M15 set; TASK_STATUS and
    CONFIRMATION were added right after, to close two gaps the initial
    TASK pass left open: a Job's outcome was only ever retrievable
    through explicit tool-invocation syntax, and a Job's own confirmation
    request had no path back from a natural reply like "yes"/"send it".
    MEMORY/EMAIL/CALENDAR/PRINTER/ROBOTIC_ARM/etc. are still deliberately
    not added as members until a milestone actually implements the
    decision logic for them - an unused enum value isn't "future-facing
    architecture," just dead code.
    """

    DIRECT = "direct"
    SEARCH = "search"
    TOOL = "tool"
    #: M15: an open-ended, possibly multi-step goal, handled by starting
    #: a persistent, AI-planned Job (app/orchestration/) rather than
    #: answering/searching/calling one tool synchronously - see
    #: JarvisCore._handle_task.
    TASK = "task"
    #: A natural-language question about a Job's progress/outcome -
    #: "what's the status of my task?", "is that finished?", "what are
    #: you doing?", "show me my running tasks" - resolved against the
    #: existing list_jobs tool, scoped to the current conversation where
    #: possible. See JarvisCore._handle_task_status.
    TASK_STATUS = "task_status"
    #: A short affirmative/negative reply ("yes", "send it", "go ahead",
    #: "no", "cancel that") that MIGHT be answering a Job's pending
    #: confirmation request. Deliberately resolved, not assumed, by
    #: JarvisCore._handle_confirmation: if nothing is actually awaiting
    #: confirmation right now, Core falls back to ordinary DIRECT
    #: handling - this capability is a hint the message COULD be a
    #: confirmation reply, never a guarantee that it is one.
    CONFIRMATION = "confirmation"
    #: A clearly-identified request to modify/inspect Jarvis's OWN
    #: codebase - "add X to your code", "fix the voice module", "modify
    #: the personality system", "inspect why this test is failing" -
    #: dispatched to the Self-Development Engine (app/self_development/)
    #: rather than an ordinary DIRECT reply or M15 TASK. Deliberately
    #: narrow (see `_SELF_DEV_VERB_PATTERN`/`_SELF_DEV_TARGET_PATTERN`
    #: below): an ordinary conversational request ("search for X",
    #: "remind me tomorrow", "what's on my calendar?") must never trigger
    #: this - see JarvisCore._handle_self_development's own safe fallback
    #: to DIRECT when no engine is configured.
    SELF_DEVELOPMENT = "self_development"
    #: An explicit "approve/reject the plan/security edit/commit" reply
    #: continuing an ALREADY-STARTED self-development session
    #: (Capability.SELF_DEVELOPMENT, above) through a later chat turn -
    #: see `_SELF_DEV_APPROVAL_PATTERN` below. Deliberately its own
    #: capability, never folded into CONFIRMATION: CONFIRMATION's own
    #: "yes"/"no" patterns are intentionally bare and ambiguous (resolved
    #: against whatever Job happens to be pending), which is fine for an
    #: M15 Job but never acceptable for SECURITY_EDIT/COMMIT - DANGEROUS-
    #: tier actions (app/tools/base.py) that must never be approved by an
    #: ambiguous word. This capability instead requires the explicit gate
    #: name ("the plan"/"the security edit"/"the commit") in the message
    #: itself - see JarvisCore._handle_self_dev_approval, which is also
    #: where "does a matching session actually exist" is checked; this
    #: capability alone is only ever a hint, exactly like CONFIRMATION.
    SELF_DEV_APPROVAL = "self_development_approval"


@dataclass
class CapabilityDecision:
    """The result of a CapabilityRouter decision.

    `capability` is the only thing JarvisCore needs to dispatch DIRECT or
    SEARCH correctly - both use JarvisCore's own already-loaded
    message/history/facts, no extra payload required.

    `tool_name` (M6.14) is used only when `capability` is `TOOL`: the name
    of the registered `Tool` (app/tools/base.py) to dispatch to via
    `ToolRegistry.execute(...)`, exactly as the user's message named it -
    never guessed, inferred, or fuzzy-matched. It's `None` when TOOL was
    clearly the right capability (an explicit tool-invocation phrasing was
    used) but no concrete tool name could be identified from the message
    (e.g. "use tool" with nothing named) - an explicit, typed way to
    represent "TOOL, but nothing to dispatch to" rather than guessing a
    tool or silently falling back to DIRECT. It's also `None`, and simply
    unused, for DIRECT/SEARCH decisions.

    `goal` (M15) is used only when `capability` is `TASK`: the goal text
    JarvisCore._handle_task passes to `create_job` - the explicit-
    invocation phrasing's captured text ("create a goal: <goal>"), or the
    raw message itself when TASK was detected from its own wording alone
    (e.g. "Prepare everything for tomorrow's meeting" - the whole message
    IS the goal). `None` and unused for every other capability.

    `approve` is used only when `capability` is `CONFIRMATION`: `True` for
    an affirmative reply ("yes", "send it", "go ahead"), `False` for a
    negative one ("no", "cancel that"). `None` and unused otherwise.

    `goal` is ALSO used when `capability` is `SELF_DEVELOPMENT`: the
    development request text JarvisCore._handle_self_development passes
    to the Self-Development Engine's own `start()` - either the explicit-
    invocation phrasing's captured text ("self-develop: <request>") or
    the raw message itself, the same "captured text, or the whole
    message" convention TASK's own `goal` already uses.

    `approve` is ALSO used when `capability` is `SELF_DEV_APPROVAL`: `True`
    for "approve ...", `False` for "reject"/"decline"/"deny ...", exactly
    the same meaning as for CONFIRMATION, just for a different pending
    thing.

    A multi-tool sequence ("use tool X ..., then use tool Y ...") is
    still just `TOOL` here, with `tool_name` set to the FIRST step's name:
    which steps exist, what their arguments are, and whether they are all
    valid depends on the tool registry, so that understanding lives in
    app/tools/command.py (ExplicitToolCommandPlanner), run by
    JarvisCore._handle_tool - never in this registry-free router.

    `self_dev_gate` is used ONLY when `capability` is `SELF_DEV_APPROVAL`:
    which of the Self-Development Engine's three explicit approval gates
    the message named - `"plan"`, `"security_edit"`, or `"commit"` (see
    `_SELF_DEV_APPROVAL_PATTERN` below). JarvisCore._handle_self_dev_approval
    uses this to find the one pending session actually awaiting THAT gate -
    never the LLM, never plan/file content, only this field, which is
    itself set only from the user's own raw message text. `None` and
    unused for every other capability.
    """

    capability: Capability
    tool_name: str | None = None
    goal: str | None = None
    approve: bool | None = None
    self_dev_gate: str | None = None


class CapabilityRouter(ABC):
    """Decides which capability should handle a message.

    Answers only "which capability?" - it never performs the capability
    itself (that's JarvisCore's dispatch + the capability's own handler,
    e.g. WebSearchTool/SearchAnswerGenerator for SEARCH).
    """

    @abstractmethod
    def route(self, message: str) -> CapabilityDecision:
        """Return the CapabilityDecision for `message`.

        `message` is the raw current message only, matching every
        deterministic router in this codebase (SearchRouter, M6.9) - no
        conversation context is used to decide WHICH capability handles a
        request. Context-dependent behavior after a capability is already
        chosen (e.g. SEARCH's query enrichment, M6.12) is a separate,
        lower-level concern this router never touches.
        """


# TOOL detection (M6.14/M6.16) is deliberately conservative: only an
# explicit, command-like invocation at the START of the message counts -
# "use tool echo" / "use the echo tool" (verbs use/run/call/invoke) - never
# a message that merely mentions "a tool" in passing ("what tool should I
# use for this job?" stays DIRECT). The phrasing itself is defined once, in
# app/tools/command.py (`match_invocation_head`), which is also what parses
# the command's steps and arguments - so this router and the parser can
# never disagree about what an explicit tool command looks like.


#: M15: the explicit, unambiguous TASK invocation - "create/start/begin a
#: goal/job/task: <goal text>" - the same "an explicit command-like phrase
#: is the entire extent of understanding, no fuzzy matching" discipline
#: _TOOL_INVOCATION_PATTERN already uses for TOOL. Takes priority even
#: over the goal-verb heuristic below, since it's unambiguous.
_TASK_INVOCATION_PATTERN = re.compile(
    r"^\s*(?:create|start|begin)\s+(?:a\s+)?(?:goal|job|task)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)

#: M15: a small, fixed, documented goal-verb list for the natural-language
#: goal phrasing this milestone's own examples use ("Prepare everything I
#: need for tomorrow's robotics meeting," "Organize my Downloads folder,"
#: "Research robotics competitions and prepare a shortlist") - an
#: accepted, deliberate trade-off, the same kind SearchRouter's own
#: keyword list and requirements.py's _CODING_KEYWORDS/_REASONING_KEYWORDS
#: already are: it catches the milestone's stated examples without
#: attempting general goal/intent NLP, and never fires on a question (a
#: "?" anywhere in the message defers to the existing DIRECT/SEARCH
#: delegation instead - "prepare? what do you mean?" should never become
#: a Job). Checked only at the START of the message, exactly like
#: _TOOL_INVOCATION_PATTERN, so a message that merely mentions one of
#: these words later ("let's discuss how to prepare for finals") is
#: unaffected.
_TASK_GOAL_VERB_PATTERN = re.compile(
    r"^\s*(?:prepare|organize|research|plan|arrange|set\s+up|put\s+together)\b",
    re.IGNORECASE,
)

#: A short affirmative/negative reply, anchored to the WHOLE message (not
#: merely present somewhere in it) - the same "explicit, closed, narrow
#: signal" discipline every other deterministic pattern in this file
#: uses. Matching the whole message rather than a substring keeps this
#: from ever misfiring on an ordinary sentence that happens to contain
#: "yes"/"no" ("yes, and also tell me about Rome" stays DIRECT - only a
#: message that IS essentially just the reply, plus optional trailing
#: punctuation, counts). This alone does NOT mean a confirmation is
#: actually pending - JarvisCore._handle_confirmation checks real Job
#: state and falls back to ordinary DIRECT handling when nothing is
#: awaiting confirmation, so an unrelated "yes"/"ok" in normal
#: conversation behaves exactly as it always has whenever there's
#: nothing to confirm.
_CONFIRMATION_YES_PATTERN = re.compile(
    r"^\s*(?:yes|yep|yeah|yup|sure|ok(?:ay)?|confirm(?:ed)?|approve[ds]?|"
    r"go\s*ahead|send\s+it|do\s+it|proceed)[\.!]*\s*$",
    re.IGNORECASE,
)
_CONFIRMATION_NO_PATTERN = re.compile(
    r"^\s*(?:no|nope|nah|don'?t(?:\s+do\s+that)?|cancel(?:\s+that)?|"
    r"decline[d]?|stop)[\.!]*\s*$",
    re.IGNORECASE,
)

#: SELF_DEV_APPROVAL: the explicit "approve/reject/decline/deny the
#: plan/security edit/commit" phrasing - anchored to the WHOLE message
#: (plus optional leading "please" and trailing punctuation), the same
#: "explicit, closed, narrow signal, never a substring match" discipline
#: `_CONFIRMATION_YES_PATTERN` already uses. Deliberately requires the
#: gate's own name spelled out - bare "approve"/"yes"/"ok" is left to
#: CONFIRMATION (which "approve[ds]?" is already one of), and never
#: resolves anything here: an ambiguous word must never be capable of
#: approving a DANGEROUS-tier SECURITY_EDIT or COMMIT (V1's own explicit
#: requirement). The captured gate name is normalized to
#: "plan"/"security_edit"/"commit" by `DeterministicCapabilityRouter.
#: route` below.
_SELF_DEV_APPROVAL_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(approve|reject|decline|deny)\s+(?:the\s+)?"
    r"(plan|security\s+edit|commit)\s*[.!]?\s*$",
    re.IGNORECASE,
)

#: A natural-language question about a Job's status/progress - a small,
#: fixed set of phrasings matching this project's own worked examples
#: ("what's the status of my task?", "what are you doing?", "is that
#: finished?", "show me my running tasks"), same discipline as every
#: other pattern here. Deliberately does NOT match a bare "my tasks" -
#: that phrase alone is genuinely ambiguous with app.tasks (Rhasspy
#: Tasks, M9's unrelated simple todo list) and is left to existing
#: DIRECT/SEARCH/TOOL handling; every pattern below instead requires a
#: word implying an in-progress GOAL specifically ("status," "doing,"
#: "finished/done," "running") that a plain todo item has no concept of.
_TASK_STATUS_PATTERNS = (
    re.compile(r"\bstatus\s+of\s+(?:my|the)\s+(?:task|job|goal)\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s?\s+(?:is\s+)?(?:the\s+)?status\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+you\s+(?:currently\s+)?doing\b", re.IGNORECASE),
    re.compile(r"\bis\s+(?:that|it)\s+(?:finished|done|complete)\b", re.IGNORECASE),
    re.compile(r"\bis\s+(?:my|the)\s+task\s+(?:finished|done|complete)\b", re.IGNORECASE),
    re.compile(r"\bare\s+you\s+(?:done|finished)\b", re.IGNORECASE),
    # "running"/"active"/"in-progress" is deliberately REQUIRED here (not
    # optional) - "show me my tasks" alone, with no such qualifier, is
    # left unmatched precisely because it's genuinely ambiguous with
    # app.tasks' own (currently NL-inert, tool-invocation-only) simple
    # todo list - see this constant's own module-level comment above.
    re.compile(
        r"\b(?:show|list)\s+me\s+my\s+(?:running|active|in[- ]progress)\s+tasks\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bmy\s+(?:running|active|in[- ]progress)\s+tasks\b", re.IGNORECASE),
)

#: SELF_DEVELOPMENT (Self-Development Engine, app/self_development/) is
#: deliberately gated on BOTH a dev-action verb AND an explicit
#: self-referential target appearing anywhere in the message - neither
#: alone is enough. "fix the printer" (verb, no matching target) and
#: "your code is great" (target-ish phrase, no verb) both stay DIRECT;
#: only a message naming BOTH ("fix the voice module", "add X to your
#: code", "modify the personality system", "inspect why this test is
#: failing") routes here. This is the same "narrow, explicit, no fuzzy
#: matching" discipline every other pattern in this file already uses,
#: chosen specifically because a normal conversational request must
#: NEVER accidentally invoke self-development (V1's own explicit
#: requirement) - see tests/test_capability_self_development.py for the
#: worked trigger/non-trigger examples this combination was checked
#: against.
_SELF_DEV_VERB_PATTERN = re.compile(
    r"\b(?:add|fix|modify|update|change|implement|refactor|debug|patch|improve|inspect)\b",
    re.IGNORECASE,
)
_SELF_DEV_TARGET_PATTERN = re.compile(
    r"\byour\s+(?:own\s+)?code(?:base)?\b|\bthe\s+codebase\b|\bthis\s+repo(?:sitory)?\b|"
    r"\byour\s+system\b|\byourself\b|\bthe\s+\w+\s+(?:module|system|engine)\b|"
    r"\bthis\s+test\b|\bthe\s+tests?\b|\byour\s+source\s+code\b",
    re.IGNORECASE,
)

#: The explicit, unambiguous form - "self-develop: <request>"/"dev
#: task: <request>"/"modify your code: <request>" - the same
#: colon-invocation convention `_TASK_INVOCATION_PATTERN` already uses,
#: taking priority even over the verb+target heuristic since it's
#: unambiguous.
_SELF_DEV_INVOCATION_PATTERN = re.compile(
    r"^\s*(?:self[- ]develop|dev(?:elopment)?\s+task|modify\s+your\s+code)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


class DeterministicCapabilityRouter(CapabilityRouter):
    """The only capability decision maker for now: TOOL, then CONFIRMATION,
    then SELF_DEV_APPROVAL, then SELF_DEVELOPMENT, then TASK_STATUS, then
    TASK, then DIRECT vs SEARCH, all without any Gemini/LLM call.

    TOOL is checked first, via the narrow explicit-invocation phrasings in
    app/tools/command.py (`match_invocation_head`: "use tool echo", "use
    the echo tool") - the most deliberate, unambiguous signal, so it takes
    priority even over a message that also happens to contain a
    SEARCH-triggering word (e.g. "use tool weather" is TOOL, not SEARCH).
    This is pure text matching - it neither looks at what tools are
    registered nor executes anything; a multi-step "..., then ..." message
    is still one TOOL decision, whose steps JarvisCore._handle_tool has the
    registry-aware ExplicitToolCommandPlanner understand and validate.

    CONFIRMATION is checked next: `_CONFIRMATION_YES_PATTERN`/
    `_CONFIRMATION_NO_PATTERN` match a short affirmative/negative reply
    anchored to the WHOLE message. This is only ever a HINT the message
    might be answering a pending confirmation - JarvisCore._handle_
    confirmation checks whether anything is actually awaiting
    confirmation and falls back to ordinary DIRECT handling if not, so a
    stray "yes"/"ok" with nothing pending behaves exactly as before.
    Checked before TASK_STATUS/TASK since neither of those patterns can
    match a bare "yes" anyway, but placed early for clarity of intent.

    SELF_DEV_APPROVAL is checked next: `_SELF_DEV_APPROVAL_PATTERN`
    ("approve the plan", "reject the security edit", "approve the
    commit", ...) is, like CONFIRMATION, only ever a HINT that this
    message continues a pending self-development session -
    JarvisCore._handle_self_dev_approval checks whether any session in
    THIS conversation is actually awaiting that exact gate and falls
    back to ordinary DIRECT handling if not. Checked before
    SELF_DEVELOPMENT so that, e.g., "approve the plan" (no dev-action
    verb, no self-referential target) is never left to fall all the way
    through to DIRECT/SEARCH before this narrower, more specific pattern
    gets a chance.

    SELF_DEVELOPMENT is checked next: `_SELF_DEV_INVOCATION_PATTERN` (an
    explicit "self-develop: ..."/"dev task: ..." phrasing) or the
    combination of `_SELF_DEV_VERB_PATTERN` AND `_SELF_DEV_TARGET_PATTERN`
    both matching somewhere in the message - see those patterns' own
    module-level comment for why BOTH are required and worked trigger/
    non-trigger examples. Checked before TASK_STATUS/TASK since a
    self-development request ("fix the voice module") is a more specific
    signal than either.

    TASK_STATUS is checked next: `_TASK_STATUS_PATTERNS` match a small
    fixed set of natural-language status questions ("what's the status of
    my task?", "is that finished?", "what are you doing?", "show me my
    running tasks") - resolved by JarvisCore._handle_task_status against
    the real list_jobs data, never guessed here.

    TASK is checked next (M15), via `_TASK_INVOCATION_PATTERN` (the same
    kind of explicit "create a goal: ..." phrasing as TOOL's) or
    `_TASK_GOAL_VERB_PATTERN` (a small fixed goal-verb list matching this
    milestone's own examples - "prepare," "organize," "research," etc. -
    at the very start of the message, skipped entirely if the message
    contains "?" so a genuine question is never misrouted). Checked after
    TOOL (so "use tool prepare_report" still means TOOL, not TASK) but
    before SEARCH/DIRECT, since a goal-verb message would otherwise often
    also match SearchRouter's own keyword patterns.

    Everything that doesn't match a TOOL, CONFIRMATION, TASK_STATUS, or
    TASK pattern is delegated to an injected SearchRouter (defaulting to
    a bare SearchRouter()) for the actual DIRECT/SEARCH keyword/pattern
    matching, exactly as in M6.13.

    Injecting a `search_router` here is how JarvisCore keeps its existing
    `router`/`SearchRouter` constructor parameter meaningful (see
    app/core.py): whatever SearchRouter instance Core was given still
    makes the DIRECT/SEARCH decision, just reached through this layer
    instead of Core calling it directly.
    """

    def __init__(self, search_router: SearchRouter | None = None):
        self.search_router = search_router or SearchRouter()

    def route(self, message: str) -> CapabilityDecision:
        head = match_invocation_head(message)
        if head is not None:
            return CapabilityDecision(capability=Capability.TOOL, tool_name=head.tool_name)

        if _CONFIRMATION_YES_PATTERN.match(message):
            return CapabilityDecision(capability=Capability.CONFIRMATION, approve=True)
        if _CONFIRMATION_NO_PATTERN.match(message):
            return CapabilityDecision(capability=Capability.CONFIRMATION, approve=False)

        self_dev_approval_match = _SELF_DEV_APPROVAL_PATTERN.match(message)
        if self_dev_approval_match:
            verb = self_dev_approval_match.group(1).lower()
            gate = re.sub(r"\s+", "_", self_dev_approval_match.group(2).lower())
            return CapabilityDecision(
                capability=Capability.SELF_DEV_APPROVAL,
                approve=(verb == "approve"),
                self_dev_gate=gate,
            )

        self_dev_invocation_match = _SELF_DEV_INVOCATION_PATTERN.match(message)
        if self_dev_invocation_match:
            return CapabilityDecision(
                capability=Capability.SELF_DEVELOPMENT,
                goal=self_dev_invocation_match.group(1).strip(),
            )
        if _SELF_DEV_VERB_PATTERN.search(message) and _SELF_DEV_TARGET_PATTERN.search(message):
            return CapabilityDecision(capability=Capability.SELF_DEVELOPMENT, goal=message.strip())

        if any(pattern.search(message) for pattern in _TASK_STATUS_PATTERNS):
            return CapabilityDecision(capability=Capability.TASK_STATUS)

        task_invocation_match = _TASK_INVOCATION_PATTERN.match(message)
        if task_invocation_match:
            return CapabilityDecision(
                capability=Capability.TASK, goal=task_invocation_match.group(1).strip()
            )
        if "?" not in message and _TASK_GOAL_VERB_PATTERN.match(message):
            return CapabilityDecision(capability=Capability.TASK, goal=message.strip())

        decision = self.search_router.route(message)
        capability = (
            Capability.SEARCH
            if decision is SearchDecision.SEARCH
            else Capability.DIRECT
        )
        return CapabilityDecision(capability=capability)


class CapabilityDecisionError(RuntimeError):
    """Raised when a CapabilityDecisionProvider can't produce a valid
    CapabilityDecision - a failed AI call, malformed output, or output
    that fails strict schema validation. Never silently guessed around:
    JarvisCore (app/core.py) catches this and re-raises it as
    ProviderError, the same 502-producing convention SearchAnswerError
    already uses at the SEARCH path's boundary (M6.8)."""


class CapabilityDecisionProvider(ABC):
    """Asks an AI for a CapabilityDecision on a message the deterministic
    fast path couldn't confidently resolve on its own.

    Only ever consulted for genuinely ambiguous messages - see
    HybridCapabilityRouter - never for the deterministic fast path, and
    never to execute anything: it only returns a decision for JarvisCore
    to dispatch through the existing DIRECT/SEARCH/TOOL paths, exactly
    like any other CapabilityDecision. It has no access to, and no
    knowledge of, ToolRegistry or any tool's implementation.
    """

    @abstractmethod
    def decide(self, message: str) -> CapabilityDecision:
        """Return a validated CapabilityDecision for `message`, or raise
        CapabilityDecisionError if the underlying AI call fails or its
        output isn't valid, safe, structured output - never a guess."""


# Only "capability"/"tool_name" - see GeminiCapabilityDecisionProvider's
# docstring for the exact validation rules applied to each.
_VALID_DECISION_KEYS = {"capability", "tool_name"}
_VALID_CAPABILITY_NAMES = {member.name for member in Capability}

# Deliberately small and fixed - the same message every time, no per-call
# variation beyond the user's own message. No tool names, descriptions, or
# registry contents are included: Gemini is never given enough to attempt
# real tool selection, only enough to pick one of three capability names.
_DECISION_PROMPT = (
    "You are a strict capability classifier for a personal assistant. "
    "Choose exactly ONE capability for the user's message:\n"
    "- DIRECT: answerable directly from general knowledge, reasoning, "
    "writing, or calculation - no current/external information or a tool "
    "is needed.\n"
    "- SEARCH: needs current, external, or live information from the web.\n"
    "- TOOL: the user is explicitly asking to invoke a specific named "
    "tool or action, not just asking a question about one.\n\n"
    "Respond with ONLY one JSON object, no prose, no markdown formatting, "
    'in exactly this shape: {"capability": "DIRECT|SEARCH|TOOL", '
    '"tool_name": null or a string}. tool_name must be null unless '
    "capability is TOOL, and if capability is TOOL, tool_name must be a "
    "real, concrete, non-empty tool name - never TOOL with a null "
    "tool_name.\n\n"
    "User message: "
)

# Gemini often wraps JSON in a markdown code fence even when told not to;
# stripping a leading/trailing fence is tolerant formatting cleanup, not
# guessing at content - the JSON underneath still has to pass full
# validation below.
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class GeminiCapabilityDecisionProvider(CapabilityDecisionProvider):
    """CapabilityDecisionProvider backed by the existing AIProvider/Gemini
    integration - the same free-tier client GeminiProvider (and
    GeminiSearchAnswerGenerator) already use. No second Gemini client, no
    new API key/config, no billing.

    `_parse_decision` enforces strict structured output before any
    CapabilityDecision is ever constructed from it:
    - the response must be valid JSON (after stripping an optional
      markdown code fence) and must decode to an object.
    - only "capability" and "tool_name" keys are allowed - any other key
      makes the whole decision invalid.
    - "capability" must be a string matching DIRECT/SEARCH/TOOL
      case-insensitively; anything else is invalid.
    - if capability is TOOL, "tool_name" must be a non-empty string -
      TOOL with a null/missing/blank tool_name is invalid (this AI path
      exists specifically to resolve that exact ambiguity, so an
      unresolved TOOL answer is a failure, not a valid decision).
    - if capability is DIRECT or SEARCH, "tool_name" must be null/absent -
      a non-null tool_name there is invalid, not silently ignored.
    Any failure raises CapabilityDecisionError - never a fallback guess.
    """

    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or GeminiProvider()

    def decide(self, message: str) -> CapabilityDecision:
        prompt = _DECISION_PROMPT + message

        try:
            raw = self.provider.generate_reply(prompt)
        except ProviderError as exc:
            raise CapabilityDecisionError(
                f"Failed to get a capability decision: {exc}"
            ) from exc

        return self._parse_decision(raw)

    @staticmethod
    def _parse_decision(raw: str) -> CapabilityDecision:
        cleaned = _CODE_FENCE_PATTERN.sub("", raw.strip()).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise CapabilityDecisionError(
                f"Invalid JSON from capability decision provider: {raw!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise CapabilityDecisionError(
                f"Expected a JSON object, got: {raw!r}"
            )

        extra_keys = set(payload) - _VALID_DECISION_KEYS
        if extra_keys:
            raise CapabilityDecisionError(
                f"Unexpected fields in capability decision: {sorted(extra_keys)}"
            )

        capability_raw = payload.get("capability")
        if not isinstance(capability_raw, str):
            raise CapabilityDecisionError(
                f"Missing or invalid 'capability' in decision: {raw!r}"
            )

        capability_name = capability_raw.strip().upper()
        if capability_name not in _VALID_CAPABILITY_NAMES:
            raise CapabilityDecisionError(f"Unknown capability: {capability_raw!r}")

        capability = Capability[capability_name]
        tool_name = payload.get("tool_name")

        if capability is Capability.TOOL:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise CapabilityDecisionError(
                    "TOOL decision must include a non-empty 'tool_name'"
                )
        elif tool_name is not None:
            raise CapabilityDecisionError(
                "'tool_name' must be null unless capability is TOOL, got: "
                f"{tool_name!r}"
            )

        return CapabilityDecision(capability=capability, tool_name=tool_name)


class HybridCapabilityRouter(CapabilityRouter):
    """Deterministic-first, AI-fallback-only capability routing (M6.15).

    Tries `deterministic_router` (defaulting to a bare
    DeterministicCapabilityRouter()) first. Every DIRECT/SEARCH decision,
    and every TOOL decision with a concrete `tool_name`, is returned
    exactly as-is - zero AI calls, and no behavior change from M6.14 for
    any of those messages. `decision_provider` (defaulting to
    GeminiCapabilityDecisionProvider()) is consulted ONLY when the
    deterministic router itself reports genuine ambiguity, and at most
    once per message - never for the deterministic fast path, never twice
    for one message, never as a second opinion on a confident decision.

    Ambiguity boundary: a `CapabilityDecision(TOOL, tool_name=None)` - an
    explicit tool-invocation phrasing ("use tool") with nothing actually
    named - is deliberately the ONLY signal treated as ambiguous. This
    reuses M6.14's own existing typed "TOOL, but nothing to dispatch to"
    representation rather than inventing a new ambiguity heuristic or a
    confidence score: every other message the deterministic router
    resolves (DIRECT by default, SEARCH via SearchRouter's patterns, TOOL
    with a named tool) is treated as fully confident, matching M6.9-M6.14
    behavior exactly - see the README's Hybrid Capability Routing section
    for the reasoning. A multi-step message whose first step names a tool
    ("use tool A, then use tool B") has a `tool_name` and so is never
    ambiguous here - its steps are understood, and validated as a whole,
    by app/tools/command.py inside JarvisCore._handle_tool.

    Whatever CapabilityDecision the decision_provider returns still goes
    through JarvisCore's normal dispatch (app/core.py) - for TOOL, that
    means `JarvisCore._handle_tool` -> `ToolRegistry.execute(...)`, the
    exact same path a deterministic TOOL decision already uses. Gemini
    never touches ToolRegistry and never executes a tool itself.
    """

    def __init__(
        self,
        deterministic_router: CapabilityRouter | None = None,
        decision_provider: CapabilityDecisionProvider | None = None,
    ):
        self.deterministic_router = (
            deterministic_router or DeterministicCapabilityRouter()
        )
        self.decision_provider = decision_provider or GeminiCapabilityDecisionProvider()

    def route(self, message: str) -> CapabilityDecision:
        decision = self.deterministic_router.route(message)
        if decision.capability is Capability.TOOL and decision.tool_name is None:
            return self.decision_provider.decide(message)
        return decision
