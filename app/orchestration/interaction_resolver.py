"""
Interaction resolver (M15): the layer between a user's chat message and a
waiting interaction.

    incoming message
        -> which waiting interactions could this be for?      (deterministic)
        -> exactly one, unambiguously?                        (else ask / leave alone)
        -> "resolve/decline THIS interaction with THIS untrusted text"
        -> JobEngine's coupled, atomic transition             (the only writer)

It decides ONLY which interaction a message is a reply to, and then hands
exactly that to JobEngine.resolve_interaction / decline_interaction, which
validates the Job and steps and commits everything atomically. It never
executes a tool, never names or selects one, never touches a step, a
permission or an autonomy level, and never calls a store's plain transition:
the user's text is data ("the answer"), never a command. No AI is involved -
matching is by fixed rules.

WHAT COUNTS AS A REPLY

  * Explicit, by code - works from ANY conversation for a background Job (one
    with no conversation) and from the Job's own conversation:

        answer <code>: <response>

    The code (6 opaque characters, shown when the question is surfaced) names
    ONE interaction. For an INPUT the response is the answer, verbatim. For a
    CONFIRMATION it must be exactly a yes/no word. Every way the lookup can
    fail - unknown code, already answered, expired, cancelled, or a Job
    belonging to a different conversation - gets the SAME reply, so a code
    reveals nothing about which Jobs exist or what state they are in.

  * Implicit - no code needed - but ONLY for an interaction that was
    SURFACED to THIS conversation within the recent-surface window
    (`orchestration_interaction_surface_window_seconds`, default 30 minutes;
    see InteractionRequest.presented_*):

      - a bare yes/no word (yes, yeah, yep, yup, confirm(ed), approve(d) /
        no, nope, nah, decline(d), reject(ed) - the whole message, optional
        trailing . or !) approves/declines the ONE surfaced CONFIRMATION;
      - any other short, non-question free text answers the ONE surfaced INPUT
        ("armed on presentation"), passed on exactly as typed.

    Deliberately NOT matched: "ok", "sure", "go ahead", "stop", ... (too
    loose); a message containing "?"; anything longer than a response may be;
    and anything the router already classified as a command or a question
    (tool use, search, task, status, self-development) - see `may_be_answer`.

  * More than one plausible candidate: nothing is resolved; the user is asked
    which one they mean. If a message might just be ordinary conversation, it
    is left to ordinary conversation and no interaction is touched.

DANGEROUS. A CONFIRMATION whose steps include a DANGEROUS tool can never be
approved by a bare yes/no or by `answer <code>: yes` in this stage (the hook
`_requires_explicit_code` marks it, so the explicit-code check can be added
there without redesign). Declining is the safe direction: `answer <code>: no`
works, a bare no does not.

SURFACING. `surface(conversation_id)` renders the pending interactions
belonging to a conversation (its own Jobs' and background Jobs') into a
short, sanitised notice and records that they were shown to it - which is
what arms the implicit replies above. Nothing else marks an interaction as
presented.

Limits per Job (2 questions / 3 interactions) are enforced by the engine
inside the atomic transition; a refusal comes back here as a REJECTED outcome.

THE CONVERSATION AROUND A QUESTION (`dialogue`, optional). A person with a
question open does not only answer it. When a `dialogue` is supplied (a
duck-typed object - this module knows nothing about who asked; see
app/handle_this/dialogue.py) the resolver also understands, deterministically:

  * CANCEL   - "cancel", "never mind", "forget it": the Job the user is talking
    about is cancelled and its waiting questions withdrawn in ONE transaction
    (`JobEngine.cancel_job`). The words are not an answer; nothing is resolved,
    planned or run. With ONE task waiting (or just answered) in the
    conversation that is the one; with more than one NOTHING is cancelled - the
    user is asked which ("cancel the printer one" / "the printer" is read against
    the words of each task's own request). A waiting CONFIRMATION in the same
    conversation keeps its own, older, yes/no/cancel handling.
  * REVISION - "actually, make that 3000 AED": the answer to a question that was
    already answered (recently, on a Job still going) is replaced through
    `JobEngine.revise_answer` - the SAME Job, no new question.
  * STALE    - an answer-shaped message to a question that was put to the user
    longer ago than the window is NOT applied: the question is shown again (and
    so armed again) and the user answers it there. Nothing is created,
    resolved or duplicated.
  * NEXT QUESTION - after an answer, if the same Job now needs something else,
    that question is opened and put in the very reply that thanks the user.

Without a `dialogue` none of this exists and the resolver behaves exactly as
described above.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from app.orchestration.base import Job, JobStatus, JobStore, StaleJobError
from app.orchestration.interactions import (
    MAX_RESPONSE_CHARS,
    InteractionError,
    InteractionLimitError,
    InteractionRequest,
    InteractionStatus,
    InteractionStore,
    InteractionType,
    StaleInteractionError,
    is_expired,
    is_valid_interaction_code,
    utc_now,
)
from app.tools.base import PermissionLevel

if TYPE_CHECKING:  # the resolver only CALLS the engine's two public operations
    from app.orchestration.engine import JobEngine

#: Most interactions listed in one ambiguity question or one surfaced notice.
MAX_LISTED = 3
_MAX_PROMPT_DISPLAY_CHARS = 300

_YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "confirm", "confirmed", "approve", "approved"})
_NO_WORDS = frozenset({"no", "nope", "nah", "decline", "declined", "reject", "rejected"})
_BARE_WORD = re.compile(r"^\s*([a-z]+)\s*[.!]*\s*$", re.IGNORECASE)
_CODE_REPLY = re.compile(r"^\s*answer\s+([a-z0-9]{6})\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)

# What the user is told. None of these repeats a Job's goal, prompt or state.
_NOT_FOUND = "I couldn't find a waiting question with that code."
_NOT_WAITING = "That question is no longer waiting for an answer, so nothing was changed."
_CHANGED = "That question changed just now, so I didn't apply your answer. Please try again."
_LIMIT = "That task has reached its limit of questions, so nothing was changed."
_CANCELLED = "Okay - I've dropped that and I won't go any further with it."
_ALREADY_FINISHED = "That one has already finished, so there was nothing left to cancel."
_REVISED = "Got it - I've updated that and I'll use your new answer."
_ALREADY_HAVE = "Got it - I already have that."
#: A cancel or revision is unambiguous, so if the Job moved on between reading
#: it and writing (the engine ticked), it is read again and retried.
_RETRIES = 3
#: The answer did not settle it and no further question may be asked (the
#: per-Job limit), so the task was stopped - and the user is told exactly that.
_STOPPED = (
    "I'm sorry - I still don't have enough to go on, so I've stopped there. "
    "If you'd still like it done, ask me again and include the details up front."
)


class OutcomeKind(Enum):
    NOT_HANDLED = "not_handled"  # not an interaction reply: ordinary conversation
    RESOLVED = "resolved"
    DECLINED = "declined"
    AMBIGUOUS = "ambiguous"  # nothing resolved; the user was asked which one
    REJECTED = "rejected"  # a reply was recognised but safely refused; nothing changed
    CANCELLED = "cancelled"  # the user dropped the task; it and its questions were withdrawn
    REVISED = "revised"  # an answer already given was replaced by the user's latest
    REMINDED = "reminded"  # the question had gone stale: it was shown again, nothing applied


@dataclass(frozen=True)
class ResolverOutcome:
    kind: OutcomeKind
    reply: str | None = None
    interaction_id: str | None = None

    @property
    def handled(self) -> bool:
        return self.kind is not OutcomeKind.NOT_HANDLED


_NOT_HANDLED = ResolverOutcome(OutcomeKind.NOT_HANDLED)


def parse_yes_no(message: str) -> bool | None:
    """True/False if `message` is exactly one accepted yes/no word (see the
    module docstring), else None. Deliberately narrow: no "ok", "sure",
    "go ahead", "stop"."""
    match = _BARE_WORD.match(message or "")
    if not match:
        return None
    word = match.group(1).lower()
    if word in _YES_WORDS:
        return True
    if word in _NO_WORDS:
        return False
    return None


def _clean(text: str | None, limit: int) -> str:
    """Single line, no control/format characters, bounded: interaction
    prompts can carry tool- or model-derived text and are shown to the user."""
    cleaned = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in (text or "")
    )
    collapsed = " ".join(cleaned.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


class InteractionResolver:
    """See the module docstring. `engine` supplies the atomic transition;
    `interactions` and `jobs` are only read from here."""

    def __init__(
        self,
        interactions: InteractionStore,
        jobs: JobStore,
        engine: "JobEngine",
        *,
        surface_window_seconds: float | None = None,
        clock: Callable[[], datetime] | None = None,
        answer_gate: Callable[[InteractionRequest, str], bool] | None = None,
        dialogue=None,
    ):
        # `answer_gate(interaction, message)` decides whether a free-text
        # message PLAUSIBLY answers an armed INPUT question: deterministic,
        # side-effect free, injected (this module knows nothing about who asked
        # the question). Only a message that passes is treated as an answer -
        # so ordinary conversation never consumes a question. `None` (the
        # default) means every armed INPUT is a candidate, as before. It does
        # not apply to the explicit `answer <code>: ...` form, which the user
        # has labelled as an answer.
        self.answer_gate = answer_gate
        # See "THE CONVERSATION AROUND A QUESTION" in the module docstring.
        # Duck-typed: owns(interaction), cancel_requested(message),
        # could_revise(message), accepted_answers(interactions),
        # revision(message, topics), choose(interactions), reminder(interaction).
        self.dialogue = dialogue
        # A "which task should I cancel?" that is waiting for its answer, per
        # conversation: (when it was asked, the tasks offered). Deliberately NOT
        # persisted: nothing has changed yet, so losing it (a restart) only means
        # the user says "cancel" again - never a wrong cancellation.
        self._choices: dict[str, tuple[datetime, list[str]]] = {}
        self._choices_lock = threading.Lock()
        self.interactions = interactions
        self.jobs = jobs
        self.engine = engine
        if surface_window_seconds is None:
            from app.config import settings

            surface_window_seconds = settings.orchestration_interaction_surface_window_seconds
        self.surface_window = timedelta(seconds=surface_window_seconds)
        self._clock = clock or utc_now

    # -- public ------------------------------------------------------------------------------

    def handle(
        self,
        message: str,
        conversation_id: str,
        *,
        may_be_answer: bool = True,
        may_answer_if_recognised: bool = False,
    ) -> ResolverOutcome:
        """Decide whether `message` (sent in `conversation_id`) replies to a
        waiting interaction and, if so and unambiguously, resolve it through
        the engine. `may_be_answer` is False when the router already
        classified the message as a command or question, which forbids the
        IMPLICIT forms (the explicit `answer <code>:` form is unaffected)."""
        code_match = _CODE_REPLY.match(message or "")
        if code_match:
            return self._handle_code(code_match.group(1).lower(), code_match.group(2), conversation_id)
        if self.dialogue is not None:
            outcome = self._handle_dialogue(
                message, conversation_id, may_revise=may_be_answer or may_answer_if_recognised
            )
            if outcome is not None:
                return outcome
        if not may_be_answer and not may_answer_if_recognised:
            return _NOT_HANDLED
        return self._handle_implicit(message, conversation_id, recognised_only=not may_be_answer)

    def surface(self, conversation_id: str) -> str | None:
        """A short notice listing the pending interactions this conversation
        should see - its own Jobs' and background Jobs' - and, as a side
        effect, record that they were surfaced to it (arming implicit
        replies for the recent-surface window). Ones already surfaced to
        this conversation within the window are not repeated. None if there
        is nothing to show."""
        now = self._clock()
        shown: list[str] = []
        for interaction in self.interactions.list_pending():
            if len(shown) >= MAX_LISTED:
                break
            job = self.jobs.get(interaction.job_id)
            if (
                job is None
                or not self._job_waiting_for(interaction, job)
                or not self._in_scope(job, conversation_id)
                or is_expired(interaction, now)
                or self._surfaced_recently_to(interaction, conversation_id, now)
            ):
                continue
            try:
                self.interactions.mark_presented(interaction.id, conversation_id, now=now)
            except InteractionError:
                continue  # it stopped being pending meanwhile: nothing to show
            shown.append(self._render(interaction, job))
        if not shown:
            return None
        return "Waiting on you:\n" + "\n".join(shown)

    def mark_surfaced(self, interaction_id: str, conversation_id: str) -> bool:
        """Record that `interaction_id` was just put to the user in
        `conversation_id` - by the very reply that asked the question - which is
        what lets a plain reply there answer it (and stops `surface` repeating
        the same question underneath). Metadata only, exactly as `surface` does
        it; False if the interaction is no longer pending."""
        try:
            self.interactions.mark_presented(interaction_id, conversation_id, now=self._clock())
        except InteractionError:
            return False
        return True

    def governs_job(self, job_id: str) -> bool:
        """Whether the Job has a pending interaction. Such a Job's
        confirmation is the interaction system's to resolve; the legacy
        confirm-by-bare-yes path must leave it alone."""
        return bool(self.interactions.list_pending_for_job(job_id))

    # -- explicit: answer <code>: <response> -------------------------------------------------------

    def _handle_code(self, code: str, response: str, conversation_id: str) -> ResolverOutcome:
        if not is_valid_interaction_code(code):
            return ResolverOutcome(OutcomeKind.REJECTED, _NOT_FOUND)
        now = self._clock()
        interaction = self.interactions.get_by_code(code)
        job = self.jobs.get(interaction.job_id) if interaction is not None else None
        if (
            interaction is None
            or job is None
            or not interaction.is_pending
            or is_expired(interaction, now)
            or not self._job_waiting_for(interaction, job)
            or not self._in_scope(job, conversation_id)
        ):
            # One reply for every failure: a code reveals nothing about which
            # Jobs exist, their state, or which conversation owns them.
            return ResolverOutcome(OutcomeKind.REJECTED, _NOT_FOUND)

        if interaction.type is InteractionType.INPUT:
            return self._apply(
                interaction, job, approved=True, response=response, via="chat-code", conversation_id=conversation_id
            )

        verdict = parse_yes_no(response)
        if verdict is None:
            return ResolverOutcome(
                OutcomeKind.REJECTED,
                f"For a confirmation, reply 'answer {code}: yes' or 'answer {code}: no'.",
            )
        if verdict and self._requires_explicit_code(interaction, job):
            return ResolverOutcome(OutcomeKind.REJECTED, self._dangerous_message(interaction))
        return self._apply(
            interaction, job, approved=verdict, response=response, via="chat-code", conversation_id=conversation_id
        )

    # -- implicit: only for what was surfaced to this conversation -----------------------------------

    def _handle_implicit(
        self, message: str, conversation_id: str, *, recognised_only: bool = False
    ) -> ResolverOutcome:
        """`recognised_only`: the router called this message a search only
        because of freshness wording ("...the most I can spend right now"). It
        may still be an answer, but ONLY to a question whose answer gate
        positively recognises it - never to a confirmation, and never to a
        question with no gate."""
        armed = self._armed(conversation_id)
        if recognised_only:
            armed = [(i, j) for i, j in armed if i.type is InteractionType.INPUT]
        if not armed:
            return self._remind_stale(message, conversation_id, recognised_only=recognised_only) or _NOT_HANDLED

        confirmations = [(i, j) for i, j in armed if i.type is InteractionType.CONFIRMATION]
        inputs = [(i, j) for i, j in armed if i.type is InteractionType.INPUT]
        verdict = parse_yes_no(message)

        if verdict is not None and confirmations:
            # A yes/no word belongs to a confirmation.
            if inputs or len(confirmations) > 1:
                return self._ambiguous(armed)
            interaction, job = confirmations[0]
            if self._requires_explicit_code(interaction, job):
                return ResolverOutcome(
                    OutcomeKind.REJECTED, self._dangerous_message(interaction), interaction.id
                )
            return self._apply(
                interaction, job, approved=verdict, response=message, via="chat", conversation_id=conversation_id
            )

        # Otherwise it can only be free text for an INPUT (a yes/no word with no
        # confirmation waiting is just a short free-text answer).
        if not inputs:
            return _NOT_HANDLED
        if "?" in message or not message.strip() or len(message) > MAX_RESPONSE_CHARS:
            return _NOT_HANDLED  # a question, empty, or too long to be an answer: ordinary chat
        # Which of the armed questions does this message plausibly ANSWER? None:
        # it is ordinary conversation and nothing at all is touched.
        inputs = [(i, j) for i, j in inputs if self._plausible(i, message, required=recognised_only)]
        if not inputs:
            return self._remind_stale(message, conversation_id, recognised_only=recognised_only) or _NOT_HANDLED
        if len(inputs) > 1 and self.dialogue is not None:
            # Two questions that ask the same thing: the newest is the one in front of the user.
            chosen = self.dialogue.choose([i for i, _ in inputs])
            if chosen is not None:
                inputs = [(i, j) for i, j in inputs if i.id == chosen.id]
        if len(inputs) > 1:
            return self._ambiguous(inputs)
        interaction, job = inputs[0]
        return self._apply(
            interaction, job, approved=True, response=message, via="chat", conversation_id=conversation_id
        )

    def _plausible(self, interaction: InteractionRequest, message: str, *, required: bool = False) -> bool:
        if self.answer_gate is None:
            return not required
        try:
            return bool(self.answer_gate(interaction, message))
        except Exception:  # noqa: BLE001 - when unsure, do not consume the question
            return False

    # -- the conversation around a question (only with a `dialogue`) -----------------------------------------

    def _handle_dialogue(self, message: str, conversation_id: str, *, may_revise: bool) -> ResolverOutcome | None:
        """Cancel, then revision. None means "neither": the message goes on to
        the ordinary rules. Any fault here is "neither" too - a message is never
        lost, and never half-applied (each move is one atomic engine call)."""
        try:
            if self.dialogue.cancel_requested(message):
                return self._handle_cancel(conversation_id)
            picked = self._handle_cancel_choice(message, conversation_id)
            if picked is not None:
                return picked
            if may_revise and self.dialogue.could_revise(message):
                return self._handle_revision(message, conversation_id)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _recent_focus(self, conversation_id: str) -> list[tuple[datetime, Job, list[InteractionRequest]]]:
        """The conversation's own Jobs, still going, that the dialogue asked
        something of and that were last ANSWERED within the surface window:
        (when it was answered, the Job, its interactions). These are the Jobs
        an answer can still be revised on, or a cancel can still mean."""
        now = self._clock()
        found = []
        for status in (JobStatus.PENDING, JobStatus.PLANNING, JobStatus.RUNNING, JobStatus.AWAITING_INPUT):
            for job in self.jobs.list(status=status):
                if job.conversation_id != conversation_id:
                    continue
                history = self.interactions.list_for_job(job.id)
                answered = [
                    datetime.fromisoformat(i.resolved_at)
                    for i in history
                    if self.dialogue.owns(i) and i.status is InteractionStatus.RESOLVED and i.resolved_at
                ]
                if not answered or now - max(answered) > self.surface_window:
                    continue
                found.append((max(answered), job, history))
        return found

    def _cancel_candidates(self, conversation_id: str) -> list[Job] | None:
        """Every task in this conversation that "cancel" could mean: one
        Handle This has a question open on (whether or not it was surfaced
        recently), or one it was answered on recently and is still going.
        Oldest first. None if a waiting CONFIRMATION is armed here: that keeps
        its own, older, yes / no / cancel handling."""
        armed = self._armed(conversation_id)
        if any(i.type is InteractionType.CONFIRMATION for i, _ in armed):
            return None
        now = self._clock()
        found: dict[str, Job] = {}
        for interaction in self.interactions.list_pending():
            if (
                interaction.type is not InteractionType.INPUT
                or not self.dialogue.owns(interaction)
                or is_expired(interaction, now)
            ):
                continue
            job = self.jobs.get(interaction.job_id)
            if job is not None and job.conversation_id == conversation_id and self._job_waiting_for(interaction, job):
                found[job.id] = job
        for _, job, _ in self._recent_focus(conversation_id):
            found[job.id] = job
        return sorted(found.values(), key=lambda j: j.created_at)

    def _handle_cancel(self, conversation_id: str) -> ResolverOutcome | None:
        """A bare "cancel". One task waiting: it is cancelled. More than one:
        NOTHING is cancelled and no question is used up - the user is asked
        which, and the answer (or "cancel the printer one") is read by
        `_handle_cancel_choice`."""
        candidates = self._cancel_candidates(conversation_id)
        if not candidates:
            return None
        if len(candidates) == 1:
            return self._cancel(candidates[0])
        return self._ask_which(conversation_id, candidates)

    def _ask_which(self, conversation_id: str, candidates: list[Job]) -> ResolverOutcome:
        with self._choices_lock:
            self._choices[conversation_id] = (self._clock(), [j.id for j in candidates])
        return ResolverOutcome(OutcomeKind.AMBIGUOUS, self.dialogue.which_task_reply(candidates))

    def _handle_cancel_choice(self, message: str, conversation_id: str) -> ResolverOutcome | None:
        """The user naming which task to drop - as the answer to "which task
        should I cancel?" or in the same breath ("cancel the printer one") -
        read deterministically against the words of each task's own request.
        Only ever picks ONE task, and only when the words tell them apart;
        otherwise the message is not a cancel at all."""
        with self._choices_lock:
            asked = self._choices.get(conversation_id)
            if asked is not None and self._clock() - asked[0] > self.surface_window:
                del self._choices[conversation_id]
                asked = None
        named = self.dialogue.cancel_target(message)
        if asked is None and named is None:
            return None
        candidates = self._cancel_candidates(conversation_id)
        if candidates is None or len(candidates) < 2:
            return None
        if asked is not None and named is None:  # a bare answer picks among the tasks that were on offer
            offered = [j for j in candidates if j.id in asked[1]]
            candidates = offered if len(offered) >= 2 else candidates
        chosen = self.dialogue.choose_task(named if named is not None else message, candidates)
        if chosen is None:
            return None
        with self._choices_lock:
            self._choices.pop(conversation_id, None)
        return self._cancel(chosen)

    def _cancel(self, job: Job) -> ResolverOutcome:
        for _ in range(_RETRIES):
            try:
                self.engine.cancel_job(job.id, expected_job_version=job.version, via="chat", now=self._clock())
            except StaleJobError:
                fresh = self.jobs.get(job.id)
                if fresh is None:
                    return ResolverOutcome(OutcomeKind.REJECTED, _NOT_WAITING)
                if fresh.status is JobStatus.CANCELLED:  # the same message twice: already done
                    return ResolverOutcome(OutcomeKind.CANCELLED, _CANCELLED)
                job = fresh
                continue
            except InteractionError:  # already finished: nothing to cancel
                fresh = self.jobs.get(job.id)
                if fresh is not None and fresh.status is JobStatus.CANCELLED:
                    return ResolverOutcome(OutcomeKind.CANCELLED, _CANCELLED)
                return ResolverOutcome(OutcomeKind.REJECTED, _ALREADY_FINISHED)
            return ResolverOutcome(OutcomeKind.CANCELLED, _CANCELLED)
        return ResolverOutcome(OutcomeKind.REJECTED, _CHANGED)

    def _handle_revision(self, message: str, conversation_id: str) -> ResolverOutcome | None:
        candidates = []
        for at, job, history in self._recent_focus(conversation_id):
            found = self.dialogue.revision(message, self.dialogue.accepted_answers(history).keys())
            if found is not None:
                candidates.append((at, job, found))
        if not candidates:
            return None
        _, job, (topic, text) = max(candidates, key=lambda item: item[0])
        for _ in range(_RETRIES):
            try:
                self.engine.revise_answer(job.id, topic, text, expected_job_version=job.version)
            except StaleJobError:
                fresh = self.jobs.get(job.id)
                if fresh is None:
                    return None
                job = fresh
                continue
            except InteractionError:
                return None  # cannot be applied (finished, unusable): ordinary conversation
            return ResolverOutcome(OutcomeKind.REVISED, _REVISED)
        return ResolverOutcome(OutcomeKind.REJECTED, _CHANGED)

    def _remind_stale(self, message: str, conversation_id: str, *, recognised_only: bool) -> ResolverOutcome | None:
        """An answer-shaped message to a question that was put to this
        conversation longer ago than the window: not applied. The question is
        shown again (which arms it again) and the user answers it there. Nothing
        is created, resolved, or duplicated - it is the same interaction."""
        if self.dialogue is None or self.answer_gate is None:
            return None
        if "?" in message or not message.strip() or len(message) > MAX_RESPONSE_CHARS:
            return None
        now = self._clock()
        stale = []
        for interaction in self.interactions.list_pending():
            if (
                interaction.type is not InteractionType.INPUT
                or not self.dialogue.owns(interaction)
                or is_expired(interaction, now)
                or self._surfaced_recently_to(interaction, conversation_id, now)
            ):
                continue
            job = self.jobs.get(interaction.job_id)
            if (
                job is None
                or job.conversation_id != conversation_id
                or not self._job_waiting_for(interaction, job)
                or not self._plausible(interaction, message, required=recognised_only)
            ):
                continue
            stale.append(interaction)
        if not stale:
            return None
        newest = max(stale, key=lambda i: i.created_at)
        try:
            self.interactions.mark_presented(newest.id, conversation_id, now=now)
        except InteractionError:
            return None  # it stopped being pending meanwhile
        return ResolverOutcome(OutcomeKind.REMINDED, self.dialogue.reminder(newest), newest.id)

    # -- shared -------------------------------------------------------------------------------------------

    def _armed(self, conversation_id: str) -> list[tuple[InteractionRequest, Job]]:
        """Pending, unexpired interactions whose Job is still waiting for them
        and that were surfaced to THIS conversation within the window."""
        now = self._clock()
        armed = []
        for interaction in self.interactions.list_pending():
            if (
                interaction.presented_conversation_id != conversation_id
                or not self._surfaced_recently_to(interaction, conversation_id, now)
                or is_expired(interaction, now)
            ):
                continue
            job = self.jobs.get(interaction.job_id)
            if job is None or not self._job_waiting_for(interaction, job) or not self._in_scope(job, conversation_id):
                continue
            armed.append((interaction, job))
        return armed

    def _surfaced_recently_to(
        self, interaction: InteractionRequest, conversation_id: str, now: datetime
    ) -> bool:
        if interaction.presented_at is None or interaction.presented_conversation_id != conversation_id:
            return False
        return now - datetime.fromisoformat(interaction.presented_at) <= self.surface_window

    @staticmethod
    def _in_scope(job: Job, conversation_id: str) -> bool:
        """A Job with no conversation is background (answerable from any);
        otherwise only from its own conversation."""
        return job.conversation_id is None or job.conversation_id == conversation_id

    @staticmethod
    def _job_waiting_for(interaction: InteractionRequest, job: Job) -> bool:
        expected = (
            JobStatus.AWAITING_INPUT
            if interaction.type is InteractionType.INPUT
            else JobStatus.AWAITING_CONFIRMATION
        )
        return job.status is expected

    def _requires_explicit_code(self, interaction: InteractionRequest, job: Job) -> bool:
        """True if approving this needs more than a plain reply - today, if any
        of its steps is DANGEROUS (or can't be resolved to a known step: fail
        closed). THE HOOK for the later explicit-code check: when that exists
        it belongs here; until then such a confirmation cannot be approved
        through the resolver at all."""
        if job.plan is None:
            return True
        for step_id in interaction.step_ids:
            step = job.plan.get(step_id)
            if step is None or step.permission == PermissionLevel.DANGEROUS.value:
                return True
        return False

    @staticmethod
    def _dangerous_message(interaction: InteractionRequest) -> str:
        return (
            "That action is irreversible, so a plain yes can't approve it - it needs an "
            "explicit confirmation code, which isn't available yet. You can decline it "
            f"with 'answer {interaction.code}: no'."
        )

    def _ambiguous(self, candidates: list[tuple[InteractionRequest, Job]]) -> ResolverOutcome:
        lines = [
            f"- [{i.code}] {_clean(i.prompt, _MAX_PROMPT_DISPLAY_CHARS)}"
            for i, _ in candidates[:MAX_LISTED]
        ]
        reply = (
            "I have more than one question waiting. Which one do you mean? Reply with "
            "'answer <code>: <your reply>' for the one you mean.\n" + "\n".join(lines)
        )
        return ResolverOutcome(OutcomeKind.AMBIGUOUS, reply)

    def _render(self, interaction: InteractionRequest, job: Job) -> str:
        prompt = _clean(interaction.prompt, _MAX_PROMPT_DISPLAY_CHARS)
        if interaction.type is InteractionType.INPUT:
            how = "reply with your answer"
        elif self._requires_explicit_code(interaction, job):
            how = f"irreversible: needs explicit confirmation, not available yet; decline with 'answer {interaction.code}: no'"
        else:
            how = "reply yes or no"
        return f"- [{interaction.code}] {prompt} ({how})"

    def _apply(
        self,
        interaction: InteractionRequest,
        job: Job,
        *,
        approved: bool,
        response: str,
        via: str,
        conversation_id: str | None = None,
    ) -> ResolverOutcome:
        """Hand exactly one interaction and the user's untrusted text to the
        engine's atomic transition, using the versions just read. Every
        failure leaves nothing changed."""
        operation = self.engine.resolve_interaction if approved else self.engine.decline_interaction
        try:
            operation(
                interaction.id,
                expected_version=interaction.version,
                expected_job_version=job.version,
                response=response,
                via=via,
                now=self._clock(),
            )
        except (StaleInteractionError, StaleJobError):
            if self._is_duplicate(interaction, response):
                return ResolverOutcome(OutcomeKind.RESOLVED, _ALREADY_HAVE, interaction.id)
            return ResolverOutcome(OutcomeKind.REJECTED, _CHANGED, interaction.id)
        except InteractionLimitError:
            return ResolverOutcome(OutcomeKind.REJECTED, _LIMIT, interaction.id)
        except InteractionError:  # expired, Job/steps no longer waiting, invalid, not found
            return ResolverOutcome(OutcomeKind.REJECTED, _NOT_WAITING, interaction.id)

        if interaction.type is InteractionType.INPUT:
            follow_up = self._open_follow_up(interaction, job, conversation_id)
            if follow_up is not None:
                return ResolverOutcome(OutcomeKind.RESOLVED, follow_up, interaction.id)
            return ResolverOutcome(
                OutcomeKind.RESOLVED,
                "Thanks - I've passed your answer on and the task is continuing.",
                interaction.id,
            )
        if approved:
            return ResolverOutcome(OutcomeKind.RESOLVED, "Approved - the task will continue.", interaction.id)
        return ResolverOutcome(OutcomeKind.DECLINED, "Declined - that step won't run.", interaction.id)

    def _is_duplicate(self, interaction: InteractionRequest, response: str) -> bool:
        """The same message arrived twice at once: the other copy already
        resolved this very question with these very words. (Only with a
        `dialogue`; otherwise a lost race is reported as before.)"""
        if self.dialogue is None or interaction.type is not InteractionType.INPUT:
            return False
        current = self.interactions.get(interaction.id)
        return (
            current is not None
            and current.status is InteractionStatus.RESOLVED
            and current.response == response
        )

    def _open_follow_up(
        self, answered: InteractionRequest, job: Job, conversation_id: str | None
    ) -> str | None:
        """If the answer leaves the SAME Job needing something else (or the
        same thing, unusably answered), open that question now and return the
        text that puts it to the user; None if the Job simply carries on.
        Only with a `dialogue`. Best effort: on any doubt the engine's next
        tick does exactly what it always did."""
        if self.dialogue is None or conversation_id is None:
            return None
        opener = getattr(self.engine, "follow_up", None)
        if opener is None:
            return None
        try:
            outcome = opener(job.id)
            if getattr(outcome, "stopped", False) is True:
                return _STOPPED  # the Job could not go on: say so, never "it is continuing"
            follow_up = getattr(outcome, "question", None)
            if not isinstance(follow_up, InteractionRequest):
                return None
            self.interactions.mark_presented(follow_up.id, conversation_id, now=self._clock())
        except Exception:  # noqa: BLE001
            return None
        question = _clean(follow_up.prompt, _MAX_PROMPT_DISPLAY_CHARS)
        if follow_up.detail.get("topic") == answered.detail.get("topic"):
            return f"I still need that to get started: {question}"
        return f"Thanks. {question}"
