"""
Handle This (Slice 1): the domain model of an UNDERSTOOD high-level request.

"Handle This" is an interpretation layer. Given something like "find me the
best 3D printer for my robotics projects within my budget", it works out WHAT
THE USER WANTS - the objective, the broad family of work it belongs to, the
constraints that are known, whether something material is genuinely missing,
and whether it is inside what Jarvis may currently attempt on its own.

What it deliberately is NOT:

- Not a planner. It never chooses steps, tools, dependencies, ordering or an
  execution strategy: that is the existing Planner's job, later
  (app/orchestration/planner.py). A `HandleThisRequest` names FAMILIES of work
  ("research", "organization"), never tools.
- Not an executor. Nothing in this package holds a ToolRegistry, a JobStore,
  a subprocess, a file handle or a network client, so an interpretation can
  not turn into an action. Everything Jarvis eventually does about a request
  still goes Planner -> validation -> JobEngine -> capability checks ->
  ToolRegistry -> tool.
- Not a permission system. "Handle This may attempt this KIND of objective"
  (app/handle_this/boundary.py) says nothing about whether any specific action
  is allowed; ToolRegistry and the capability system stay authoritative.

Everything an AI produces here is untrusted data: it is validated against the
closed vocabularies in this module and dropped or rejected if it does not fit.
The user's own words are never replaced: `original_request` is stored exactly
as received, separate from the `interpreted_goal`.
"""

from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

MAX_REQUEST_CHARS = 4000
MAX_GOAL_CHARS = 400
MAX_ITEM_CHARS = 200
MAX_LIST_ITEMS = 8
MAX_CLARIFICATION_NEEDS = 3


class HandleThisError(ValueError):
    """A HandleThisRequest (or a part of one) is structurally invalid."""


# --- closed vocabularies --------------------------------------------------------------------------------------------


class GoalFamily(Enum):
    """The v1 families - deliberately few. MIXED means ONE objective that
    clearly combines several supported families (it still becomes one Job,
    with its steps decided later by the Planner). UNSUPPORTED means the
    objective needs something outside the approved v1 boundary."""

    RESEARCH = "research"
    ANALYSIS = "analysis"
    ORGANIZATION = "organization"
    COMPUTER_WORKFLOW = "computer_workflow"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"


#: The families that name real work. (MIXED is a combination of these;
#: UNSUPPORTED is the absence of one.)
SUPPORTED_FAMILIES: tuple[GoalFamily, ...] = (
    GoalFamily.RESEARCH,
    GoalFamily.ANALYSIS,
    GoalFamily.ORGANIZATION,
    GoalFamily.COMPUTER_WORKFLOW,
)


class HandleThisStatus(Enum):
    READY = "ready"  # understood, supported, nothing material missing
    NEEDS_CLARIFICATION = "needs_clarification"  # something material is genuinely missing
    UNSUPPORTED = "unsupported"  # outside the v1 boundary (or not currently available)


class Confidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Topic(Enum):
    """What a constraint, an assumption or a missing piece of information is
    ABOUT. Closed on purpose: nothing outside this list can be asked for or
    claimed to be known."""

    SUBJECT = "subject"  # what the request is about
    BUDGET = "budget"
    USE_CASE = "use_case"  # what it is for
    WHEN = "when"  # a time a reminder/event needs
    QUANTITY = "quantity"
    TIMEFRAME = "timeframe"
    PREFERENCE = "preference"
    INTENT = "intent"  # what outcome the user wants at all


#: The topics a stated CONSTRAINT can be about.
CONSTRAINT_TOPICS = frozenset(
    {Topic.BUDGET, Topic.QUANTITY, Topic.TIMEFRAME, Topic.USE_CASE, Topic.PREFERENCE}
)


class ConstraintSource(Enum):
    REQUEST = "request"  # stated in the user's own words
    CONTEXT = "context"  # known from the Context Engine


class UnsupportedCategory(Enum):
    """Why an objective is outside the boundary. Categories of WHAT WOULD BE
    NEEDED, never tool names."""

    COMMUNICATION = "communication"  # contacting other people
    TRANSACTION = "transaction"  # buying, paying, booking
    FILE_OR_SYSTEM_CHANGE = "file_or_system_change"  # deleting/installing/running code
    SELF_MODIFICATION = "self_modification"  # changing Jarvis itself
    PERMISSION_CHANGE = "permission_change"  # loosening limits or confirmations
    HARDWARE_CONTROL = "hardware_control"  # driving printers/robots/devices
    UNAVAILABLE_CAPABILITY = "unavailable_capability"  # a supported family with nothing to do it with right now
    OTHER = "other"


class FailureCode(Enum):
    INVALID_REQUEST = "invalid_request"
    UNCLEAR_OBJECTIVE = "unclear_objective"
    INTERPRETER_UNAVAILABLE = "interpreter_unavailable"
    INTERPRETER_INVALID_OUTPUT = "interpreter_invalid_output"
    CONTEXT_UNAVAILABLE = "context_unavailable"
    INTERNAL_ERROR = "internal_error"


class ContextStatus(Enum):
    RETRIEVED = "retrieved"
    NOT_CONFIGURED = "not_configured"  # no Context Engine exists: understood from the request alone


# --- text hygiene -----------------------------------------------------------------------------------------------------


def clean_line(text: object, limit: int) -> str:
    """Untrusted text as one bounded line without control/format characters."""
    if not isinstance(text, str):
        return ""
    cleaned = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in text
    )
    collapsed = " ".join(cleaned.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3].rstrip() + "..."


# --- parts ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    """Something known that limits or shapes the objective ("under $500",
    "top three", "for my robotics projects"), and where it came from."""

    topic: Topic
    text: str
    source: ConstraintSource = ConstraintSource.REQUEST

    def __post_init__(self) -> None:
        if self.topic not in CONSTRAINT_TOPICS:
            raise HandleThisError(f"'{getattr(self.topic, 'value', self.topic)}' is not a constraint topic.")
        if not isinstance(self.source, ConstraintSource):
            raise HandleThisError("A constraint's source must be a ConstraintSource.")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > MAX_ITEM_CHARS:
            raise HandleThisError(f"A constraint's text must be 1-{MAX_ITEM_CHARS} characters.")


@dataclass(frozen=True)
class Assumption:
    """A safe default Jarvis will proceed on INSTEAD of asking (recorded so it
    is visible and can be corrected, never silently baked in)."""

    topic: Topic
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.topic, Topic):
            raise HandleThisError("An assumption's topic must be a Topic.")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > MAX_ITEM_CHARS:
            raise HandleThisError(f"An assumption's text must be 1-{MAX_ITEM_CHARS} characters.")


@dataclass(frozen=True)
class ClarificationNeed:
    """One thing that is genuinely missing: WHAT (`missing`), WHY it is needed
    (`why`), and whether it materially affects the objective (`material`).
    Both texts come from fixed templates, never from the request or a model."""

    topic: Topic
    missing: str
    why: str
    material: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.topic, Topic):
            raise HandleThisError("A need's topic must be a Topic.")
        for name in ("missing", "why"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > MAX_ITEM_CHARS:
                raise HandleThisError(f"A need's {name} must be 1-{MAX_ITEM_CHARS} characters.")
        if not isinstance(self.material, bool):
            raise HandleThisError("A need's material flag must be a bool.")


@dataclass(frozen=True)
class Clarification:
    """What must be answered before the objective can safely go on. Only
    MATERIAL needs are ever represented here: absent optional information is
    not a question. (Turning this into an InteractionRequest is a later
    slice's job; nothing here creates one.)"""

    needs: tuple[ClarificationNeed, ...]

    def __post_init__(self) -> None:
        needs = tuple(self.needs)
        if not needs or len(needs) > MAX_CLARIFICATION_NEEDS:
            raise HandleThisError(f"A clarification needs 1-{MAX_CLARIFICATION_NEEDS} items.")
        if not all(isinstance(n, ClarificationNeed) and n.material for n in needs):
            raise HandleThisError("A clarification may only contain material needs.")
        object.__setattr__(self, "needs", needs)


@dataclass(frozen=True)
class UnsupportedReason:
    """Why the objective is outside the boundary: a category, and - when the
    reason is that a supported family has nothing to run on right now - which
    families were unavailable."""

    category: UnsupportedCategory
    unavailable_families: tuple[GoalFamily, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.category, UnsupportedCategory):
            raise HandleThisError("An unsupported reason needs an UnsupportedCategory.")
        families = tuple(self.unavailable_families)
        if not all(f in SUPPORTED_FAMILIES for f in families):
            raise HandleThisError("unavailable_families may only name supported families.")
        object.__setattr__(self, "unavailable_families", families)


@dataclass(frozen=True)
class ContextHandoff:
    """The RELEVANT slice of what Jarvis already knows, ready for a Planner
    later: selective (only the categories this kind of objective needs),
    bounded (the Context Engine's own budget) and read-only. `categories` maps
    a display label to short strings; `known_constraints` are the constraints
    the context settles (e.g. a stored budget). Nothing here is written back
    anywhere."""

    status: ContextStatus
    requested_categories: tuple[str, ...] = ()
    categories: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    known_constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ContextStatus):
            raise HandleThisError("A context handoff needs a ContextStatus.")
        object.__setattr__(self, "requested_categories", tuple(self.requested_categories))
        object.__setattr__(
            self,
            "categories",
            MappingProxyType({k: tuple(v) for k, v in dict(self.categories).items()}),
        )
        object.__setattr__(self, "known_constraints", tuple(self.known_constraints))


@dataclass(frozen=True)
class UnderstandingMeta:
    """How the understanding was reached and how sure it is."""

    method: str  # "deterministic" or "llm"
    confidence: Confidence

    def __post_init__(self) -> None:
        if self.method not in ("deterministic", "llm"):
            raise HandleThisError("An understanding method is 'deterministic' or 'llm'.")
        if not isinstance(self.confidence, Confidence):
            raise HandleThisError("An understanding needs a Confidence.")


# --- the request ------------------------------------------------------------------------------------------------------


def new_request_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class HandleThisRequest:
    """One understood user objective. Immutable.

    `original_request` is the user's text EXACTLY as received (never cleaned,
    trimmed or replaced by an interpretation); `interpreted_goal` is a
    separate, bounded restatement. `goal_family` / `required_families` say
    which broad kinds of work are involved - MIXED is still ONE request, and
    it is not split into anything here.

    The invariants below tie the pieces together so an inconsistent request
    cannot exist: READY has neither a clarification nor an unsupported reason;
    NEEDS_CLARIFICATION has a clarification; UNSUPPORTED has an unsupported
    reason and family UNSUPPORTED (and no required families)."""

    id: str
    original_request: str
    interpreted_goal: str
    goal_family: GoalFamily
    required_families: tuple[GoalFamily, ...]
    status: HandleThisStatus
    understanding: UnderstandingMeta
    constraints: tuple[Constraint, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    success_criteria: tuple[str, ...] = ()
    context: ContextHandoff | None = None
    clarification: Clarification | None = None
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise HandleThisError("A request needs an id.")
        if not isinstance(self.original_request, str) or not self.original_request.strip():
            raise HandleThisError("A request needs its original text.")
        if len(self.original_request) > MAX_REQUEST_CHARS:
            raise HandleThisError(f"A request can be at most {MAX_REQUEST_CHARS} characters.")
        if (
            not isinstance(self.interpreted_goal, str)
            or not self.interpreted_goal.strip()
            or len(self.interpreted_goal) > MAX_GOAL_CHARS
            or clean_line(self.interpreted_goal, MAX_GOAL_CHARS) != self.interpreted_goal
        ):
            raise HandleThisError(
                f"The interpreted goal must be 1-{MAX_GOAL_CHARS} characters on a single clean line."
            )
        if not isinstance(self.goal_family, GoalFamily) or not isinstance(self.status, HandleThisStatus):
            raise HandleThisError("A request needs a GoalFamily and a HandleThisStatus.")
        if not isinstance(self.understanding, UnderstandingMeta):
            raise HandleThisError("A request needs UnderstandingMeta.")

        required = tuple(self.required_families)
        if not all(isinstance(f, GoalFamily) and f in SUPPORTED_FAMILIES for f in required):
            raise HandleThisError("required_families may only contain supported families.")
        if len(set(required)) != len(required):
            raise HandleThisError("required_families must not repeat a family.")
        object.__setattr__(self, "required_families", required)

        if self.goal_family is GoalFamily.MIXED:
            if len(required) < 2:
                raise HandleThisError("A MIXED objective must require at least two families.")
        elif self.goal_family is GoalFamily.UNSUPPORTED:
            if required:
                raise HandleThisError("An UNSUPPORTED objective requires no families.")
        elif required != (self.goal_family,):
            raise HandleThisError("A single-family objective must require exactly that family.")

        for name, limit_items in (("constraints", MAX_LIST_ITEMS), ("assumptions", MAX_LIST_ITEMS)):
            items = tuple(getattr(self, name))
            if len(items) > limit_items:
                raise HandleThisError(f"At most {limit_items} {name}.")
            object.__setattr__(self, name, items)
        if not all(isinstance(c, Constraint) for c in self.constraints):
            raise HandleThisError("constraints must be Constraint objects.")
        if not all(isinstance(a, Assumption) for a in self.assumptions):
            raise HandleThisError("assumptions must be Assumption objects.")
        criteria = tuple(self.success_criteria)
        if len(criteria) > MAX_LIST_ITEMS or not all(
            isinstance(c, str) and c.strip() and clean_line(c, MAX_ITEM_CHARS) == c for c in criteria
        ):
            raise HandleThisError("success_criteria must be a few short clean lines.")
        object.__setattr__(self, "success_criteria", criteria)

        if self.context is not None and not isinstance(self.context, ContextHandoff):
            raise HandleThisError("context must be a ContextHandoff.")

        supported_status = (self.goal_family is GoalFamily.UNSUPPORTED) == (
            self.status is HandleThisStatus.UNSUPPORTED
        )
        if not supported_status:
            raise HandleThisError("Family UNSUPPORTED and status UNSUPPORTED go together.")
        if self.status is HandleThisStatus.UNSUPPORTED:
            if not isinstance(self.unsupported, UnsupportedReason) or self.clarification is not None:
                raise HandleThisError("An UNSUPPORTED request has an unsupported reason and no clarification.")
        elif self.unsupported is not None:
            raise HandleThisError("Only an UNSUPPORTED request has an unsupported reason.")
        if self.status is HandleThisStatus.NEEDS_CLARIFICATION:
            if not isinstance(self.clarification, Clarification):
                raise HandleThisError("A request that needs clarification must say what it needs.")
        elif self.status is HandleThisStatus.READY and self.clarification is not None:
            raise HandleThisError("A READY request has nothing left to clarify.")

    @property
    def is_ready(self) -> bool:
        return self.status is HandleThisStatus.READY


# --- the outcome ------------------------------------------------------------------------------------------------------


_FAILURE_MESSAGES = {
    FailureCode.INVALID_REQUEST: "That request could not be read.",
    FailureCode.UNCLEAR_OBJECTIVE: "I could not tell what outcome you want.",
    FailureCode.INTERPRETER_UNAVAILABLE: "I could not interpret that request right now.",
    FailureCode.INTERPRETER_INVALID_OUTPUT: "I could not interpret that request reliably.",
    FailureCode.CONTEXT_UNAVAILABLE: "I could not look up the context this request depends on.",
    FailureCode.INTERNAL_ERROR: "Something went wrong while understanding that request.",
}


@dataclass(frozen=True)
class UnderstandingFailure:
    """Understanding did not complete. A fixed, safe message per code - never
    the request's text, a model's output, or an exception message."""

    code: FailureCode

    def __post_init__(self) -> None:
        if not isinstance(self.code, FailureCode):
            raise HandleThisError("A failure needs a FailureCode.")

    @property
    def message(self) -> str:
        return _FAILURE_MESSAGES[self.code]


@dataclass(frozen=True)
class UnderstandingResult:
    """Exactly one of: an understood request (READY, NEEDS_CLARIFICATION or
    UNSUPPORTED), or a failure. Never a guess dressed up as an understanding."""

    request: HandleThisRequest | None = None
    failure: UnderstandingFailure | None = None

    def __post_init__(self) -> None:
        if (self.request is None) == (self.failure is None):
            raise HandleThisError("A result is either a request or a failure, never both or neither.")

    @property
    def ok(self) -> bool:
        return self.request is not None
