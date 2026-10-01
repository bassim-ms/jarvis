"""
High-level objective detection: does this message look like the user DELEGATING
an outcome ("work out the steps and handle it"), as opposed to chatting,
asking a question, giving a simple command or explicitly asking for a search?

This answers ONE question and nothing else. It does not understand the
objective (Handle This's intake does), classify its family (signals.py /
the intake), check the capability boundary, retrieve context or decide whether
anything is missing. It only decides whether the message should ENTER Handle
This at all - and Handle This still applies all of its own checks afterwards.
Routing into Handle This is not permission to do anything.

THE SIGNAL IS STRUCTURE, NOT KEYWORDS. A single mention of "research" or "find"
decides nothing ("Research what a black hole is", "Find the capital of France"
are ordinary information requests). What marks a delegated objective is
several KINDS of intended work in one request:

    GATHER       finding / researching / looking into
    COMPARE      comparing, ranking, evaluating, analysing
    SYNTHESIS    summarising, explaining, telling me what matters
    DELIVERABLE  a produced result: a plan, shortlist, report, checklist
    ORGANIZE     organising / prioritising
    STORE        saving what was found
    REMIND       a reminder, timer, or scheduled item

A message is a delegated objective only if it contains at least two different
kinds, at least one of them SUBSTANTIVE (GATHER, COMPARE, ORGANIZE or
DELIVERABLE - saving and reminding alone are simple commands, and summarising
alone is an ordinary answer) and at least one of them STRONG (COMPARE,
ORGANIZE, DELIVERABLE, STORE or REMIND - "find X and summarise it" is what the
existing search path already does in one step). Or: an explicit delegation cue
("handle", "take care of", "figure this out", "work it out") together with a
concrete substantive action - "handle this" on its own delegates nothing.

One more shape counts, because it is a delegation even in a single sentence:
CHOOSING something to acquire for a stated personal purpose - "research the best
laptop for my robotics work". A gather request about a product-choice topic
(signals.is_purchase_oriented) that names what it is FOR ("for my ...") is
asking Jarvis to work out a recommendation for the user's own situation, not
to look a fact up. Without the personal purpose ("Find me a laptop", "Find
current prices for the Elegoo printer", "What is the best 3D printer?") it
stays an ordinary request.

Slice 5.1 adds two more shapes a person plainly means as "go and do this for me":
asking for a choice of something to acquire ("Find me a printer", "Look for a
good 3D printer for my robotics projects" - `_ACQUIRE`), and handing over a named
thing with a delegation cue ("Handle my schedule for tomorrow", "Take care of
this for me" - `_DELEGATE_START`). Questions ("What is a 3D printer?"), lookups
("Find the definition of robotics") and the bare cue ("handle this") are
unchanged. Routing into Handle This is still not permission: its own intake
still checks the boundary and fails closed.

One read-only request is recognised here as well, and deliberately NOT sent into Handle This: the Morning
Brief ("morning brief", "what do I need to know today?", "catch me up for today"). It is a supported
information/review objective - it gathers what the existing systems already know and reports it - so it
never becomes a Job, a Plan or a Planner run (route MORNING_BRIEF; see app/morning_brief/).

Conservative by construction: a question, a request to explain, or anything
short of that structure is EXISTING - the existing routing, unchanged. The
decision is fully deterministic, adds no AI call and no I/O, and carries only
informational evidence: nothing downstream treats it as authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.handle_this.signals import is_computer_workflow, is_informational, is_purchase_oriented, mentions_product
from app.morning_brief.recognition import is_morning_brief_request

MAX_MESSAGE_CHARS = 4000


class ObjectiveRoute(Enum):
    HANDLE_THIS = "handle_this"  # a delegated high-level objective: enters Handle This
    EXISTING = "existing"  # anything else: the existing routing, untouched
    #: A supported READ-ONLY review request (the Morning Brief). It is recognised here so it is never mistaken
    #: for a delegated objective, but it is NOT one: no Job, no Plan, no Planner - it goes straight to the
    #: existing read-only Morning Brief capability.
    MORNING_BRIEF = "morning_brief"


class ObjectiveReason(Enum):
    MULTI_STEP_OBJECTIVE = "multi_step_objective"
    DELEGATED_OBJECTIVE = "delegated_objective"
    QUESTION = "question"
    SINGLE_ACTION = "single_action"
    SELECTION_FOR_A_PURPOSE = "selection_for_a_purpose"
    PRODUCT_SEARCH = "product_search"
    NO_SUBSTANTIVE_ACTION = "no_substantive_action"
    NOT_A_REQUEST = "not_a_request"
    READ_ONLY_REVIEW = "read_only_review"
    COMPUTER_WORKFLOW = "computer_workflow"


@dataclass(frozen=True)
class ObjectiveRoutingDecision:
    """The verdict, with the original text and INFORMATIONAL evidence (which
    kinds of work were seen). `route` is the only thing acted on; evidence is
    never a score or an authorisation."""

    route: ObjectiveRoute
    reason: ObjectiveReason
    original_request: str
    evidence: tuple[str, ...] = ()

    @property
    def is_high_level_objective(self) -> bool:
        return self.route is ObjectiveRoute.HANDLE_THIS


_I = re.IGNORECASE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


# --- kinds of intended work ----------------------------------------------------------------------------------------------

_KINDS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "gather",
        _rx(
            r"\b(?:research(?:ing)?|investigate|investigating|look(?:ing)?\s+(?:up|into|for)|find(?:ing)?(?:\s+out)?|go\s+through|"
            r"search(?:ing)?\s+(?:google\s+)?(?:for|the\s+web|online)|gather|collect|compile|dig\s+into)\b"
        ),
    ),
    (
        "compare",
        _rx(
            r"\b(?:compare|comparing|comparison|contrast|rank|ranking|weigh|evaluate|assess|analy[sz]e|"
            r"pros\s+and\s+cons)\b"
        ),
    ),
    (
        "synthesis",
        _rx(
            r"\b(?:summari[sz]e|summary|explain\s+(?:the\s+)?(?:trade-?offs?|differences?|evidence|results?|options?)|"
            r"tell\s+me\s+(?:which|what|how|whether)|let\s+me\s+know\s+(?:which|what)|identify\s+(?:what|which|the)|"
            r"figure\s+out\s+what|recommend|what\s+i\s+should\s+(?:consider|do|prioriti[sz]e)|"
            r"which\s+(?:specifications?|specs|features)\s+matter|what\s+should\s+be\s+done)\b"
        ),
    ),
    (
        "deliverable",
        _rx(
            r"\bgive\s+me\s+(?:a|an|the)\s+(?:\w+\s+){0,2}(?:plan|shortlist|report|list|comparison|breakdown|overview|"
            r"recommendation|checklist)\b|"
            r"\b(?:write|draft|prepare|produce|put\s+together|create|build|make)\s+(?:me\s+)?(?:a|an|the)\s+(?:\w+\s+){0,2}"
            r"(?:plan|shortlist|report|comparison|overview|checklist|roadmap)\b|"
            r"\btroubleshooting\s+plan\b"
        ),
    ),
    ("organize", _rx(r"\b(?:organi[sz]e|prioriti[sz]e|sort\s+out|triage)\b")),
    (
        "store",
        _rx(
            r"\b(?:save|store|keep|write\s+down|note\s+down|jot\s+down)\s+(?:it|them|this|these|those|the\s+\w+(?:\s+\w+)?|"
            r"a\s+(?:copy|note)|notes?)\b"
        ),
    ),
    (
        "remind",
        _rx(
            r"\b(?:remind(?:er)?s?|set\s+(?:a\s+)?(?:timer|alarm)|schedule|"
            r"(?:put|add)\s+.+\s+(?:on|to)\s+my\s+calendar)\b"
        ),
    ),
)

_SUBSTANTIVE = frozenset({"gather", "compare", "organize", "deliverable"})
_STRONG = frozenset({"compare", "organize", "deliverable", "store", "remind"})

#: An explicit hand-over of the work. On its own it delegates nothing.
_PURPOSE = _rx(r"\bfor\s+(?:my|our)\s+[a-z0-9]")

_DELEGATE = _rx(
    r"\b(?:handle|take\s+care\s+of|figure\s+(?:this|it|that)\s+out|work\s+(?:this|it|that)\s+out|"
    r"sort\s+(?:this|it)\s+out|deal\s+with)\b"
)

#: "Find me a printer", "Get me a laptop under 2500 AED", "Look for a good 3D
#: printer for my robotics projects": asking Jarvis to come back with a
#: choice of something to acquire. It is the "me / a" that makes it a request
#: for a choice; "Find the definition of X" and "Find current prices for the
#: Elegoo printer" are lookups and are not matched.
_ACQUIRE = _rx(
    r"^\s*(?:(?:find|get)\s+me\b|look(?:ing)?\s+for\s+(?:a|an|some)\b|help\s+me\s+(?:find|choose|pick)\b)"
)

#: Things one asks to be TOLD, whatever product is mentioned ("find me the
#: definition of a printer").
_FACT_LOOKUP = _rx(r"\b(?:definition|meaning|capital|news|weather|history|explanation|difference|specs?\s+of)\b")

#: A delegation cue that starts the request AND names what is being handed over
#: ("handle my schedule for tomorrow", "take care of this for me"). The cue with
#: nothing to act on ("handle this", "take care of it") still delegates nothing.
_DELEGATE_START = _rx(
    r"^\s*(?:handle|take\s+care\s+of|sort\s+out|deal\s+with)\s+"
    r"(?:(?:my|our|the|all\s+of\s+(?:my|our|the)|these|those)\s+\w+|(?:this|it|that)\s+for\s+(?:me|us)\b)"
)

# --- questions and politeness --------------------------------------------------------------------------------------------------

#: A polite request wrapper ("Can you ...", "Please ...", "I'd like you to ..."),
#: possibly after a greeting: the request follows it, and a trailing "?" on such
#: a request is not a question.
_POLITE_PREFIX = _rx(
    r"^\s*(?:(?:hey|hi|ok|okay)\s+)?(?:jarvis[,!]?\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+|i(?:'d|\s+would)\s+like\s+you\s+to\s+|"
    r"i\s+(?:want|need)\s+you\s+to\s+)"
)
_QUESTION_OPENER = _rx(
    r"^\s*(?:what|whats|what's|why|how|when|where|who|whom|whose|which|is|are|was|were|does|do|did|"
    r"should|shall|has|have|had|am)\b"
)


def _decision(
    route: ObjectiveRoute, reason: ObjectiveReason, message: str, evidence: tuple[str, ...] = ()
) -> ObjectiveRoutingDecision:
    return ObjectiveRoutingDecision(route, reason, message, evidence)


def detect_objective(message: object) -> ObjectiveRoutingDecision:
    """Whether `message` is a delegated high-level objective. Never raises;
    anything that is not clearly one is EXISTING."""
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        return _decision(ObjectiveRoute.EXISTING, ObjectiveReason.NOT_A_REQUEST, message if isinstance(message, str) else "")

    if is_morning_brief_request(message):  # "morning brief", "what do I need to know today?", ... (whole message)
        return _decision(ObjectiveRoute.MORNING_BRIEF, ObjectiveReason.READ_ONLY_REVIEW, message, ("morning_brief",))

    text = message
    polite = _POLITE_PREFIX.match(text)
    if polite:
        text = text[polite.end():].rstrip()
        if text.endswith("?"):
            text = text[:-1]  # "Can you ...?" is a request, not a question
    if "?" in text or _QUESTION_OPENER.match(text):
        return _decision(ObjectiveRoute.EXISTING, ObjectiveReason.QUESTION, message)

    present = tuple(name for name, pattern in _KINDS if pattern.search(text))
    kinds = set(present)
    if is_computer_workflow(text):
        # "Open my browser and go to example.com", "Open X and scroll to Y": operating the computer is a delegated
        # objective. Entering Handle This grants nothing - whether each action asked for is actually available is
        # the capability boundary's decision, and an unavailable one (a click, typing) is declined there.
        return _decision(
            ObjectiveRoute.HANDLE_THIS, ObjectiveReason.COMPUTER_WORKFLOW, message, present + ("computer",)
        )
    if _DELEGATE_START.match(text):
        return _decision(
            ObjectiveRoute.HANDLE_THIS, ObjectiveReason.DELEGATED_OBJECTIVE, message, present or ("delegate",)
        )
    acquiring = (
        bool(_ACQUIRE.match(text))
        and not is_informational(text)
        and not _FACT_LOOKUP.search(text)
        and (is_purchase_oriented(text) or mentions_product(text))
    )
    if not kinds & _SUBSTANTIVE:
        if acquiring:  # "Get me a laptop under 2500 AED"
            return _decision(ObjectiveRoute.HANDLE_THIS, ObjectiveReason.PRODUCT_SEARCH, message, present or ("gather",))
        return _decision(ObjectiveRoute.EXISTING, ObjectiveReason.NO_SUBSTANTIVE_ACTION, message, present)

    delegated = bool(_DELEGATE.search(text))
    if len(kinds) >= 2 and kinds & _STRONG:
        return _decision(ObjectiveRoute.HANDLE_THIS, ObjectiveReason.MULTI_STEP_OBJECTIVE, message, present)
    if delegated and kinds:
        return _decision(ObjectiveRoute.HANDLE_THIS, ObjectiveReason.DELEGATED_OBJECTIVE, message, present)
    if "gather" in kinds and _PURPOSE.search(text) and is_purchase_oriented(text):
        return _decision(ObjectiveRoute.HANDLE_THIS, ObjectiveReason.SELECTION_FOR_A_PURPOSE, message, present)
    if acquiring:
        return _decision(ObjectiveRoute.HANDLE_THIS, ObjectiveReason.PRODUCT_SEARCH, message, present)
    return _decision(ObjectiveRoute.EXISTING, ObjectiveReason.SINGLE_ACTION, message, present)
