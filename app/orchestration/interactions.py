"""
Interactions (M15): the durable record of "Jarvis is waiting for the user
about a Job" - a question to answer (INPUT) or steps to approve
(CONFIRMATION).

THIS MODULE IS A DATA/STATE FOUNDATION ONLY. It defines what an
InteractionRequest is, which transitions it may make, and how it is
persisted (see app/orchestration/sqlite_interaction_store.py). It does NOT
route chat replies, match natural-language answers, notify anyone, expire
anything on a schedule, or resume a Job - those belong to later stages, and
nothing here can run a tool or change what a step is allowed to do.

Relationship to a Job (deliberately loose):
  * An interaction belongs to a Job by `job_id`; a Job may have several
    over its life (its history is the audit trail) and, by policy of later
    stages, normally at most one PENDING at a time.
  * The interaction does NOT own Job execution and is NOT a second state
    machine. `JobStatus.AWAITING_CONFIRMATION` / `AWAITING_INPUT` stay
    DERIVED views of the Job - `derive_waiting_status` computes which one
    applies from the Job's PENDING interactions, and nothing else.
  * Transitions here change ONLY the interaction row. A resolved or
    declined interaction therefore cannot make any step runnable: steps run
    only when the engine advances a RUNNING Job, and only the engine (the
    single writer of plan state, in a later stage) will translate a
    resolved interaction into Job/step changes.

Lifecycle (terminal states are final):

    PENDING --resolve--> RESOLVED    (INPUT: the answer; CONFIRMATION: approved)
    PENDING --decline--> DECLINED    (the user said no / abandoned it)
    PENDING --expire---> EXPIRED     (only once `expires_at` has passed)
    PENDING --cancel---> CANCELLED   (withdrawn, e.g. its Job was cancelled)

Every transition is optimistic-concurrency guarded exactly like Job saves:
the caller names the `version` it read, the write applies only if that is
still current AND the row is still PENDING, and otherwise fails closed
(`StaleInteractionError` / `InvalidInteractionTransitionError`) leaving the
row untouched. Each transition also records an audit entry (`resolution`).

Stored text is untrusted: a prompt may embed tool/LLM-derived text, and a
response is whatever the user typed. Both are stored verbatim (bounded) and
must only ever be treated as data by whatever reads them back.
"""

from __future__ import annotations

import json
import re
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from app.orchestration.base import Job, JobStatus

# --- bounds (one place, easy to change) ------------------------------------------------------------------

#: Agreed v1 limits per Job (see JobEngine._check_interaction_limits): at most
#: this many INPUT questions, and this many interactions in total, may be
#: answered. Creation-side enforcement belongs to whatever creates
#: interactions (a later stage); this is the defence at the point a reply is
#: applied.
MAX_INPUT_QUESTIONS_PER_JOB = 2
MAX_INTERACTIONS_PER_JOB = 3

MAX_PROMPT_CHARS = 4000
MAX_RESPONSE_CHARS = 2000
MAX_DETAIL_CHARS = 4000
MAX_REASON_CHARS = 200
MAX_STEP_IDS = 50

#: Interaction codes: short, typeable, opaque. Random (never derived from a
#: job/interaction id, goal or prompt), lowercase, no look-alike characters
#: (no i, l, o, 0, 1) - ~887 million possibilities at length 6, and unique
#: across ALL interactions ever created so a stale code can never match a
#: newer request.
CODE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
CODE_LENGTH = 6
_CODE_PATTERN = re.compile(rf"^[{CODE_ALPHABET}]{{{CODE_LENGTH}}}$")

#: `via` is an identifier for the channel/actor of a transition ("chat",
#: "engine", ...) - deliberately not free text.
_VIA_PATTERN = re.compile(r"^[a-z0-9_.:-]{1,50}$")


class InteractionType(Enum):
    """What kind of thing the user is being asked. Extensible: a future
    CHOICE/SELECTION kind is a new member plus a handler, with no schema
    change (kind-specific data lives in `detail`)."""

    INPUT = "input"
    CONFIRMATION = "confirmation"


class InteractionStatus(Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    DECLINED = "declined"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


TERMINAL_INTERACTION_STATUSES = frozenset(
    {
        InteractionStatus.RESOLVED,
        InteractionStatus.DECLINED,
        InteractionStatus.EXPIRED,
        InteractionStatus.CANCELLED,
    }
)

#: The whole transition table. Anything not listed is rejected.
ALLOWED_TRANSITIONS: dict[InteractionStatus, frozenset[InteractionStatus]] = {
    InteractionStatus.PENDING: TERMINAL_INTERACTION_STATUSES,
    **{status: frozenset() for status in TERMINAL_INTERACTION_STATUSES},
}


class InteractionError(RuntimeError):
    """Base class for every interaction persistence failure. All are safe
    failures: nothing was written."""


class InteractionNotFoundError(InteractionError):
    """The interaction (or, on create, its Job) does not exist."""


class StaleInteractionError(InteractionError):
    """The interaction changed since the caller read it (its persisted
    version is not the one the caller named). The write was refused; the
    caller's view is out of date and must be re-read - never retried
    blindly."""


class InvalidInteractionTransitionError(InteractionError):
    """The requested transition isn't allowed from the interaction's
    current state (it is already terminal, its expiry rules forbid it, or
    the input is invalid for this kind)."""


class InteractionExpiredError(InvalidInteractionTransitionError):
    """A resolve/decline arrived after `expires_at` but before the
    interaction was expired: the reply is refused and the row is left
    PENDING for `expire` to close."""


class InteractionLimitError(InteractionError):
    """Applying this interaction would exceed the per-Job interaction limits
    (MAX_INPUT_QUESTIONS_PER_JOB / MAX_INTERACTIONS_PER_JOB). Nothing was
    changed."""


class InteractionJobStateError(InteractionError):
    """The interaction's Job (or the steps it names) is not in a state where
    this interaction can be applied - e.g. the Job was cancelled, is no
    longer waiting, or a gated step already ran. Raised while applying an
    interaction to its Job; nothing was changed."""


@dataclass
class InteractionRequest:
    """One persisted interaction. See the module docstring.

    `prompt` is what the user is shown (bounded, untrusted-derived text).
    `code` is the opaque short code a reply can name to match this request
    from any conversation. `step_ids` are the plan steps a CONFIRMATION is
    about (empty for INPUT); `detail` carries kind-specific JSON-safe data
    (e.g. tool names and an argument digest for a CONFIRMATION).

    `resolved_at` is when the interaction LEFT the PENDING state, whatever
    the outcome (resolved, declined, expired or cancelled). `response` is
    the user's text for RESOLVED (required for INPUT) or optionally for
    DECLINED - untrusted data. `resolution` is the audit entry of the
    terminal transition: `{"previous_status", "status", "previous_version",
    "at", "via", "reason"}`; `None` while PENDING.

    `version` is the optimistic-concurrency counter (0 at creation, +1 per
    transition), set by the store, never by callers.
    """

    id: str
    job_id: str
    type: InteractionType
    status: InteractionStatus
    prompt: str
    code: str
    created_at: str
    expires_at: str | None = None
    resolved_at: str | None = None
    response: str | None = None
    step_ids: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    resolution: dict[str, Any] | None = None
    version: int = 0
    #: When, and to which conversation, this interaction was last SURFACED to
    #: the user (shown in a chat reply). Only ever set while PENDING, via
    #: `InteractionStore.mark_presented`. It is presentation METADATA, not
    #: lifecycle state: setting it never changes `status` or `version`. A
    #: free-text answer or a bare yes/no is accepted only in the conversation
    #: it was last surfaced to, within a recent window (see
    #: app/orchestration/interaction_resolver.py).
    presented_at: str | None = None
    presented_conversation_id: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status is InteractionStatus.PENDING


class InteractionStore(ABC):
    """Persistent storage for InteractionRequests. Depends only on the types
    above, never on SQLite (same split as JobStore).

    Every transition method takes the `expected_version` the caller read and
    returns the updated InteractionRequest. Failure modes, all leaving the
    row untouched: InteractionNotFoundError (no such interaction),
    StaleInteractionError (persisted version != expected_version),
    InvalidInteractionTransitionError (not PENDING, or the kind/expiry rules
    forbid it). `now` (an aware datetime) exists so tests control the clock;
    real callers omit it."""

    @abstractmethod
    def create(
        self,
        job_id: str,
        type: InteractionType,
        prompt: str,
        *,
        step_ids: list[str] | tuple[str, ...] = (),
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """Create a PENDING interaction for an existing Job. `ttl_seconds`
        sets `expires_at` (None = never expires). A CONFIRMATION must name
        at least one step. Raises InteractionNotFoundError if the Job does
        not exist, ValueError for invalid input."""

    @abstractmethod
    def get(self, interaction_id: str) -> InteractionRequest | None:
        """Look up one interaction, or None."""

    @abstractmethod
    def list_for_job(self, job_id: str) -> list[InteractionRequest]:
        """Every interaction of the Job in any state, oldest first - the
        Job's interaction history / audit trail."""

    @abstractmethod
    def list_pending_for_job(self, job_id: str) -> list[InteractionRequest]:
        """The Job's PENDING interactions, oldest first."""

    @abstractmethod
    def get_by_code(self, code: str) -> InteractionRequest | None:
        """Look up an interaction by its (unique, opaque) code, in any
        state, or None."""

    @abstractmethod
    def list_pending(self) -> list[InteractionRequest]:
        """Every PENDING interaction across all Jobs, oldest first - for
        the reply resolver, which then applies its own eligibility rules."""

    @abstractmethod
    def mark_presented(
        self,
        interaction_id: str,
        conversation_id: str,
        *,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """Record that the user was just shown this PENDING interaction in
        `conversation_id`. Metadata only: it does not change the lifecycle
        `status` or bump `version` (so it never makes an in-flight resolver
        stale). Raises InteractionNotFoundError, or
        InvalidInteractionTransitionError if it is no longer PENDING."""

    @abstractmethod
    def resolve(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        response: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """PENDING -> RESOLVED. An INPUT requires a non-empty `response`
        (the answer); for a CONFIRMATION it is optional (the raw reply, for
        audit). Refused with InteractionExpiredError once `expires_at` has
        passed."""

    @abstractmethod
    def decline(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        response: str | None = None,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """PENDING -> DECLINED. Refused with InteractionExpiredError once
        `expires_at` has passed."""

    @abstractmethod
    def expire(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """PENDING -> EXPIRED, only once `expires_at` is set and has
        passed - it can never be used to end an interaction early."""

    @abstractmethod
    def cancel(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """PENDING -> CANCELLED (withdrawn - e.g. its Job was cancelled)."""

    @abstractmethod
    def transition_with_job(
        self,
        interaction_id: str,
        target: "InteractionStatus",
        *,
        expected_version: int,
        expected_job_version: int,
        mutate_job: Callable[[Job, InteractionRequest], None],
        check: Callable[[Job, InteractionRequest, list[InteractionRequest]], None] | None = None,
        response: str | None = None,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> tuple[InteractionRequest, Job]:
        """The COUPLED transition: move the interaction to `target` AND
        apply its consequence to its Job, in ONE transaction - both are
        committed together or neither is.

        Everything the plain transitions check is checked (freshness of
        `expected_version`, the transition table, payload and expiry
        rules), plus the Job's own freshness: the Job must still be at
        `expected_job_version`. `mutate_job(job, interaction)` then edits a
        freshly loaded in-memory copy of the Job (raising
        InteractionJobStateError if the Job/steps are no longer in a state
        that permits this). The Job write and the interaction write are each
        version-guarded; if either guard refuses, or anything fails before
        the single commit, EVERYTHING rolls back and a Stale*/error is
        raised. Returns (updated interaction, updated Job).

        `check(job, interaction, siblings)`, if given, runs inside the same
        transaction just before `mutate_job`, with every interaction of the
        Job (this one included) as read in that transaction: it may raise an
        InteractionError to refuse (e.g. a limit), so a rule that depends on
        the Job's other interactions is evaluated atomically with the write.

        `mutate_job` and `check` run while the store's lock is held, so they
        must be pure and must not call any store.
        This is the engine's primitive (JobEngine is the only intended
        caller): the interaction never makes its Job runnable by itself."""

    @abstractmethod
    def create_job_awaiting_input(
        self,
        *,
        goal: str,
        context: dict[str, Any],
        conversation_id: str | None,
        prompt: str,
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, InteractionRequest]:
        """Create a NEW Job directly in AWAITING_INPUT together with its one
        PENDING INPUT interaction, in ONE transaction: both exist or neither
        does, and there is never a moment when the Job exists without the
        question it waits on - nor a PENDING Job the engine could plan before
        the question is answered. The Job takes the same defaults as any other
        (ASSISTED autonomy, default autonomous permissions). Returns
        (Job, interaction)."""

    @abstractmethod
    def park_job_for_input(
        self,
        job_id: str,
        *,
        expected_job_version: int,
        mutate_job: Callable[[Job], None],
        prompt: str,
        detail: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        check: Callable[[Job, list[InteractionRequest]], None] | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, InteractionRequest]:
        """Make an EXISTING Job wait on a NEW INPUT interaction, in ONE
        transaction: the Job must still be at `expected_job_version`
        (StaleJobError otherwise); `check(job, siblings)` may refuse (e.g. a
        per-Job limit) with an InteractionError; `mutate_job(job)` edits the
        freshly loaded copy (it sets the waiting status). Both writes commit
        together or neither does. `check` and `mutate_job` run under the
        store's lock and must be pure. Returns (Job, interaction)."""

    @abstractmethod
    def cancel_with_job_check(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        check: Callable[[Job, InteractionRequest], None],
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> InteractionRequest:
        """PENDING -> CANCELLED, but only if `check(job, interaction)` - run
        inside the same transaction, against the Job as that transaction sees
        it - does not raise. It withdraws an interaction whose Job no longer
        waits on it; the Job itself is never written. `check` must be pure
        (it runs under the store's lock). Raises what `check` raises, plus the
        usual not-found / stale / invalid-transition errors; on any failure
        nothing is changed."""

    @abstractmethod
    def cancel_job_with_interactions(
        self,
        job_id: str,
        *,
        expected_job_version: int,
        mutate_job: Callable[[Job], None],
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> tuple[Job, list[InteractionRequest]]:
        """Cancel a Job AND withdraw every one of its PENDING interactions
        (PENDING -> CANCELLED) in ONE transaction: both commit or neither
        does, so a cancelled Job can never be left with a live interaction
        that could later resume it. The Job must still be at
        `expected_job_version` (StaleJobError otherwise). `mutate_job(job)`
        edits a freshly loaded copy (it decides what "cancelled" means for the
        Job) and must be pure. Returns (updated Job, the interactions that
        were withdrawn)."""


# --- shared helpers -----------------------------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_interaction_code() -> str:
    """A fresh random code (see CODE_ALPHABET). Uniqueness across
    interactions is enforced by the store, which retries on a collision."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def is_valid_interaction_code(code: str) -> bool:
    return isinstance(code, str) and bool(_CODE_PATTERN.match(code))


def is_expired(request: InteractionRequest, now: datetime) -> bool:
    """Whether `request` has an expiry that has passed as of `now`."""
    if request.expires_at is None:
        return False
    return datetime.fromisoformat(request.expires_at) <= now


def expires_at_for(created: datetime, ttl_seconds: float | None) -> str | None:
    if ttl_seconds is None:
        return None
    return (created + timedelta(seconds=ttl_seconds)).isoformat()


def default_ttl_seconds(type: InteractionType) -> float:
    """The configured lifetime for a kind of interaction (settings:
    24 hours for CONFIRMATION, 7 days for INPUT by default). Callers pass
    it to `create(ttl_seconds=...)`; the store itself applies no default."""
    from app.config import settings

    if type is InteractionType.CONFIRMATION:
        return settings.orchestration_confirmation_ttl_seconds
    return settings.orchestration_input_ttl_seconds


def validate_new_interaction(
    type: InteractionType,
    prompt: Any,
    step_ids: list[str] | tuple[str, ...],
    detail: dict[str, Any] | None,
    ttl_seconds: float | None,
) -> dict[str, Any]:
    """Validate creation input; returns the JSON-safe `detail` dict to
    store. Raises ValueError. Nothing is truncated: oversize input is
    rejected, never silently cut."""
    if not isinstance(type, InteractionType):
        raise ValueError(f"type must be an InteractionType, got: {type!r}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("An interaction needs a non-empty prompt.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"An interaction prompt can be at most {MAX_PROMPT_CHARS} characters.")
    if not all(isinstance(s, str) and s for s in step_ids) or len(step_ids) > MAX_STEP_IDS:
        raise ValueError(f"step_ids must be at most {MAX_STEP_IDS} non-empty strings.")
    if type is InteractionType.CONFIRMATION and not step_ids:
        raise ValueError("A CONFIRMATION must name the step(s) it is about.")
    if ttl_seconds is not None and (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or ttl_seconds <= 0
    ):
        raise ValueError("ttl_seconds must be a positive number or None.")
    detail = dict(detail or {})
    try:
        rendered = json.dumps(detail)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"detail must be JSON-serializable: {exc}") from exc
    if len(rendered) > MAX_DETAIL_CHARS:
        raise ValueError(f"detail can be at most {MAX_DETAIL_CHARS} characters when serialized.")
    return detail


def validate_transition_input(
    request_type: InteractionType,
    target: InteractionStatus,
    response: str | None,
    reason: str | None,
    via: str | None,
) -> None:
    """Validate the payload of a transition. Raises
    InvalidInteractionTransitionError."""
    if response is not None:
        if not isinstance(response, str):
            raise InvalidInteractionTransitionError("response must be text.")
        if len(response) > MAX_RESPONSE_CHARS:
            raise InvalidInteractionTransitionError(
                f"A response can be at most {MAX_RESPONSE_CHARS} characters."
            )
    if (
        target is InteractionStatus.RESOLVED
        and request_type is InteractionType.INPUT
        and (response is None or not response.strip())
    ):
        raise InvalidInteractionTransitionError("Resolving an INPUT needs a non-empty response.")
    if reason is not None and (not isinstance(reason, str) or len(reason) > MAX_REASON_CHARS):
        raise InvalidInteractionTransitionError(
            f"reason must be text of at most {MAX_REASON_CHARS} characters."
        )
    if via is not None and not (isinstance(via, str) and _VIA_PATTERN.match(via)):
        raise InvalidInteractionTransitionError(
            "via must be a short identifier (lowercase letters, digits, '_', '.', ':', '-')."
        )


def derive_waiting_status(pending: list[InteractionRequest]) -> JobStatus | None:
    """The waiting-state VIEW a Job derives from its PENDING interactions,
    or None if it isn't waiting on any: a pending CONFIRMATION means
    AWAITING_CONFIRMATION, otherwise a pending INPUT means AWAITING_INPUT
    (a confirmation takes precedence if, unusually, both are pending).
    Non-pending interactions are ignored. Pure: it neither reads nor writes
    the Job - the engine remains the only thing that sets a Job's status."""
    pending = [i for i in pending if i.is_pending]
    if any(i.type is InteractionType.CONFIRMATION for i in pending):
        return JobStatus.AWAITING_CONFIRMATION
    if any(i.type is InteractionType.INPUT for i in pending):
        return JobStatus.AWAITING_INPUT
    return None
