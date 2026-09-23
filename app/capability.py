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
    """

    capability: Capability
    tool_name: str | None = None
    goal: str | None = None
    approve: bool | None = None


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


#: Deliberately conservative (M6.14): only an explicit, command-like
#: invocation at the start of the message counts as a TOOL request - never
#: a message that merely mentions "a tool" in passing ("what tool should I
#: use for this job?" stays DIRECT). The tool name itself, if given, is
#: captured verbatim - no fuzzy matching, no case-folding, no synonym
#: mapping. There is currently no safe, general way to map arbitrary
#: natural language to a specific registered tool, so none is invented
#: here; this pattern is the entire extent of TOOL "understanding" for
#: M6.14.
_TOOL_INVOCATION_PATTERN = re.compile(
    r"^\s*(?:use|run|call|invoke)\s+tool\b\s*:?\s*(\S+)?", re.IGNORECASE
)

#: M6.16: the mirror-image phrasing - the tool name given BEFORE the
#: literal word "tool" instead of after it ("use the echo tool[...]"),
#: still anchored to the same explicit verb set and still requiring the
#: literal word "tool" right after the name. Not a general NLP pattern -
#: just the other natural word order for the same explicit intent. The
#: negative lookahead stops the optional article from backtracking into
#: the capture group itself (without it, "use a tool to..." would wrongly
#: match with "a" captured as the tool name).
_TOOL_MENTION_PATTERN = re.compile(
    r"^\s*(?:use|run|call|invoke)\s+(?:(?:the|a|an)\s+)?"
    r"(?!(?:the|a|an)\b)(\S+)\s+tool\b",
    re.IGNORECASE,
)


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


class DeterministicCapabilityRouter(CapabilityRouter):
    """The only capability decision maker for now: TOOL, then CONFIRMATION,
    then TASK_STATUS, then TASK, then DIRECT vs SEARCH, all without any
    Gemini/LLM call.

    TOOL is checked first, via two narrow explicit-invocation phrasings -
    `_TOOL_INVOCATION_PATTERN` ("use tool echo") and `_TOOL_MENTION_PATTERN`
    ("use the echo tool") - the most deliberate, unambiguous signal of the
    three, so either takes priority even over a message that also happens
    to contain a SEARCH-triggering word (e.g. "use tool weather" is TOOL,
    not SEARCH, even though "weather" alone would be SEARCH). Both checks
    are pure text parsing - neither looks at what tools are actually
    registered (that's ToolRegistry's job, at dispatch time in
    JarvisCore._handle_tool - see app/core.py) and neither executes
    anything.

    CONFIRMATION is checked next: `_CONFIRMATION_YES_PATTERN`/
    `_CONFIRMATION_NO_PATTERN` match a short affirmative/negative reply
    anchored to the WHOLE message. This is only ever a HINT the message
    might be answering a pending confirmation - JarvisCore._handle_
    confirmation checks whether anything is actually awaiting
    confirmation and falls back to ordinary DIRECT handling if not, so a
    stray "yes"/"ok" with nothing pending behaves exactly as before.
    Checked before TASK_STATUS/TASK since neither of those patterns can
    match a bare "yes" anyway, but placed early for clarity of intent.

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
        tool_match = _TOOL_INVOCATION_PATTERN.match(
            message
        ) or _TOOL_MENTION_PATTERN.match(message)
        if tool_match:
            return CapabilityDecision(
                capability=Capability.TOOL, tool_name=tool_match.group(1)
            )

        if _CONFIRMATION_YES_PATTERN.match(message):
            return CapabilityDecision(capability=Capability.CONFIRMATION, approve=True)
        if _CONFIRMATION_NO_PATTERN.match(message):
            return CapabilityDecision(capability=Capability.CONFIRMATION, approve=False)

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
    for the reasoning.

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
