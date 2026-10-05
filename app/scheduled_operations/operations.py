"""
The user-facing Scheduled Job operations - through the EXISTING request path, not a new one.

JarvisCore hands a message to `command_reply(message, conversation_id)` (the same duck-typed hook the reminders, the device operations
and the Morning Brief use) only when the WHOLE message is one of a few phrasings (app/scheduled_operations/language.py - a fixed
grammar; no model here):

  CREATE   "Every morning at 8, research AI robotics news and give me a summary" / "Tomorrow at 5 PM, research X and report the
           findings". Nothing is scheduled by saying it: the request is understood, checked, stored as a PROPOSAL, and the user is
           ASKED - in plain words: what Jarvis will do, when, how often, for how long, that it is read-only, and that every run still
           goes through the normal security checks. Only an approved confirmation makes it live.
  LIST / SHOW / PAUSE / RESUME / CANCEL   "what scheduled jobs do I have?", "pause my AI news scheduled job".

THE APPROVAL IS THE EXISTING ONE. The question is asked through the same Job / interaction / ToolRegistry confirmation the memory policy
and the device revocation use: a Job that is already WAITING on a confirmation, whose one step is the CONFIRM-level, non-plannable,
non-automatable `commit_scheduled_job_approval`. Only an approved confirmation of THAT Job runs the tool, and the tool only checks it and
calls the service's atomic approval - so a "no", an expiry or silence schedules nothing, and a forged call schedules nothing. The proposal
is a stored row (not an in-memory note), so an approval survives a restart. A request is never carried out on several Scheduled Jobs at
once, and a question about WHICH one is answered by listing and asking the user to say more of its words - no hidden state is kept.

The Scheduled Job's own words are DATA: stored and shown as plain text; nothing in them is ever parsed as a command or fed to a model here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.notifications.text import clean_text
from app.orchestration.base import Job, JobStatus, Plan, PlanStep
from app.orchestration.interactions import InteractionStatus, InteractionType, default_ttl_seconds
from app.reminders.timefmt import local_phrase
from app.scheduled_jobs.base import RunOutcome, ScheduledJob, ScheduledJobRun, ScheduledJobStatus, StatusReason
from app.scheduled_operations.language import ManageAction, ManageRequest, CreateRequest, parse_request, topic_words
from app.scheduled_operations.service import (
    UNATTENDED_REFUSAL,
    Approved,
    Proposed,
    ScheduledJobRequestError,
    ScheduledJobService,
)
from app.scheduling.schedule import ScheduleKind
from app.tools.base import PermissionLevel

logger = logging.getLogger("jarvis.scheduled_operations")

COMMIT_TOOL_NAME = "commit_scheduled_job_approval"
ORIGIN = "scheduled_jobs"
GOAL = "Schedule a job"
_STEP_ID = "s1"
_NO_REPLANS = 10_000
MAX_LISTED = 15
MAX_CHOICES = 5
_SHOWN = 160

_S, _R = ScheduledJobStatus, StatusReason

BULK = (
    "I won't change several scheduled jobs at once. Tell me which one - for example \"pause my AI news scheduled job\" - "
    "or ask \"what scheduled jobs do I have?\" to see them."
)
_PAUSE_REASONS = {
    _R.USER: "paused by you",
    _R.BREAKER: "paused after several failed runs in a row",
    _R.DEVICE_REVOKED: "paused (the device it came from is no longer trusted)",
    _R.BLOCKED: "paused (something it needs isn't available)",
}
_DONE_REASONS = {
    _R.ONE_TIME_DONE: "finished (it ran once)",
    _R.ONE_TIME_MISSED: "finished (its time passed while I wasn't running)",
    _R.LIFETIME_ENDED: "its approved period has ended - ask me to schedule it again to renew",
}
_OUTCOMES = {
    RunOutcome.SKIPPED_MISSED: "skipped (I wasn't running at the time)",
    RunOutcome.SKIPPED_OVERLAP: "skipped (the previous run was still going)",
    RunOutcome.SKIPPED_BLOCKED: "skipped (something it needed wasn't available)",
    RunOutcome.SKIPPED_DISABLED: "skipped (scheduled jobs were switched off)",
    RunOutcome.SKIPPED_CAP: "skipped (it had reached its daily limit)",
    RunOutcome.SKIPPED_EXPIRED: "skipped (its approved period had ended)",
}
_RESULTS = {"completed": "completed", "failed": "failed", "cancelled": "was cancelled"}


class ScheduledJobApprovalConfirmer:
    """Asks the user before a Scheduled Job becomes live, through the existing Job / interaction machinery (see
    app/devices/operations.py and app/memory_policy/confirmation.py, whose shape this follows)."""

    def __init__(self, interactions, *, mark_surfaced: Callable[[str, str], object] | None = None):
        self.interactions = interactions
        self.mark_surfaced = mark_surfaced

    def approved(self, job_id: str, scheduled_job_id: str) -> str | None:
        """The id of the confirmation the user APPROVED for exactly this Scheduled Job, or None. Read from the interaction store, so a
        forged call - or the step of a declined, expired or different Job - can never approve."""
        for interaction in self.interactions.list_for_job(job_id):
            if (
                interaction.type is InteractionType.CONFIRMATION
                and interaction.status is InteractionStatus.RESOLVED
                and interaction.detail.get("origin") == ORIGIN
                and interaction.detail.get("scheduled_job_id") == scheduled_job_id
            ):
                return interaction.id
        return None

    def ask(self, scheduled_job_id: str, prompt: str, conversation_id: str | None) -> str:
        from app.orchestration.sqlite_store import SQLiteJobStore

        job: Job = SQLiteJobStore.build_job(GOAL, conversation_id=conversation_id, status=JobStatus.AWAITING_CONFIRMATION)
        job.plan = Plan(
            steps=[
                PlanStep(
                    id=_STEP_ID, tool_name=COMMIT_TOOL_NAME, arguments={"scheduled_job_id": scheduled_job_id},
                    permission=PermissionLevel.CONFIRM.value,
                )
            ]
        )
        job.total_step_count = 1
        job.pending_step_ids = [_STEP_ID]
        job.replan_count = _NO_REPLANS  # never re-planned by a model
        job.confirmation_message = prompt
        created, interaction = self.interactions.create_job_awaiting_confirmation(
            job, prompt=prompt, step_ids=[_STEP_ID],
            detail={"origin": ORIGIN, "dangerous": False, "scheduled_job_id": scheduled_job_id},
            ttl_seconds=default_ttl_seconds(InteractionType.CONFIRMATION),
        )
        if self.mark_surfaced is not None and conversation_id:
            try:
                self.mark_surfaced(interaction.id, conversation_id)
            except Exception:  # noqa: BLE001 - best effort: the ordinary "waiting on you" notice shows it next
                pass
        return created.id


# ====================================== words ============================================================================


def _when_words(sj: ScheduledJob) -> str:
    return sj.schedule.describe()


def _at(instant, tz: str | None) -> str:
    return local_phrase(instant, tz)


def proposal_words(proposed: Proposed) -> str:
    """The approval question: concise, natural language, nothing technical."""
    sj, schedule = proposed.scheduled_job, proposed.scheduled_job.schedule
    once = schedule.kind is ScheduleKind.ONCE
    lines = [
        "Here's what I'll set up:",
        f"- I'll: {clean_text(sj.intent_text, _SHOWN)}",
        f"- When: {schedule.describe()}",
        f"- First run: {_at(proposed.first_due, schedule.tz)}",
    ]
    if once:
        lines.append("- It runs once, then it's done.")
    else:
        lines.append("- It keeps going for up to 90 days from your approval; after that I'll ask you before continuing.")
    lines += [
        "- It's read-only: I can research, read public web pages and look at your tasks, notes, events and reminders - "
        "I can't change anything, send anything or control your computer.",
        "- Every run still goes through my normal security checks.",
    ]
    if proposed.rolled:
        lines.append("(That time has already passed today, so the first run is tomorrow.)")
    if proposed.warning:
        lines.append(f"Note: {proposed.warning}")
    lines.append("Shall I schedule it?")
    return "\n".join(lines)


def approved_words(approved: Approved) -> str:
    sj = approved.scheduled_job
    if approved.already:
        return f"'{clean_text(sj.title, 80)}' is already scheduled."
    message = f"Scheduled: '{clean_text(sj.title, 80)}' - {sj.schedule.describe()}. First run: {_at(sj.due_at, sj.schedule.tz)}."
    if approved.lifetime.expires_at is not None:
        message += f" It runs until {_at(approved.lifetime.expires_at, sj.schedule.tz)}; I'll ask before it continues after that."
    return message


def _state_words(sj: ScheduledJob) -> str:
    tz = sj.schedule.tz
    if sj.status is _S.ACTIVE:
        return f"{_when_words(sj)}; next {_at(sj.due_at, tz)}"
    if sj.status is _S.PAUSED:
        return f"{_PAUSE_REASONS.get(sj.status_reason, 'paused')}; {_when_words(sj)}"
    if sj.status is _S.COMPLETED:
        return _DONE_REASONS.get(sj.status_reason, "finished")
    if sj.status is _S.PROPOSED:
        return "waiting for your approval"
    return sj.status.value


def _line(sj: ScheduledJob) -> str:
    return f"{clean_text(sj.title, 70)} - {_state_words(sj)}"


def _outcome_words(run: ScheduledJobRun | None, tz: str | None) -> str:
    if run is None:
        return "it hasn't run yet"
    when = _at(run.scheduled_for, tz)
    if run.outcome in _OUTCOMES:
        return f"{_OUTCOMES[run.outcome]} ({when})"
    return f"{_RESULTS.get(run.result_state, 'started, not finished yet')} ({when})"


def detail_words(sj: ScheduledJob, last: ScheduledJobRun | None) -> str:
    tz = sj.schedule.tz
    lines = [
        f"{clean_text(sj.title, 80)}",
        f"- What: {clean_text(sj.intent_text, _SHOWN)}",
        f"- State: {_state_words(sj)}",
        f"- Runs so far: {sj.runs_total}; last: {_outcome_words(last, tz)}",
    ]
    if sj.consecutive_failures:
        lines.append(f"- Failed runs in a row: {sj.consecutive_failures}")
    if sj.expires_at is not None:
        lines.append(f"- Approved until: {_at(sj.expires_at, tz)}")
    lines.append("- Read-only: it can't change anything, send anything or control your computer.")
    return "\n".join(lines)


# ====================================== the chat hook ====================================================================


class ScheduledJobOperations:
    def __init__(self, service: ScheduledJobService, confirmer: ScheduledJobApprovalConfirmer | None = None):
        self.service = service
        self.confirmer = confirmer

    def command_reply(self, message: str, conversation_id: str | None = None, device_context=None) -> str | None:
        """`device_context` is the Gateway-authenticated DeviceContext of THIS request (None on the anonymous /chat). It is only used
        to remember which trusted device a Scheduled Job came from, so a later revocation can stop it - never to grant anything."""
        request = parse_request(message)
        if request is None:
            return None
        try:
            if isinstance(request, CreateRequest):
                return self._create(request, conversation_id, device_context)
            return self._manage(request)
        except ScheduledJobRequestError as exc:
            return str(exc)

    # -- creating (asked, never just done) ----------------------------------------------------------------------------------------

    def _create(self, request: CreateRequest, conversation_id: str | None, device_context) -> str:
        if not request.reads_only:  # "send / add / buy / open ... every day": V1 schedules read-only work only - said aloud, not stored
            return UNATTENDED_REFUSAL
        device_id = None
        if device_context is not None and getattr(device_context, "is_authenticated", False):
            device_id = getattr(device_context, "device_id", None)
        proposed = self.service.propose(request.text, request.when, conversation_id=conversation_id, device_id=device_id)
        if proposed.duplicate:
            if proposed.scheduled_job.status is _S.PROPOSED:
                return "I'm already waiting for your answer on that one - say yes to schedule it, or no to drop it."
            return f"You already have that scheduled: {_line(proposed.scheduled_job)}."
        if self.confirmer is None:
            self.service.discard(proposed.scheduled_job.id)
            return "I can't ask for that confirmation right now, so nothing was scheduled."
        prompt = proposal_words(proposed)
        try:
            self.confirmer.ask(proposed.scheduled_job.id, prompt, conversation_id)
        except Exception:  # noqa: BLE001 - it cannot be asked: nothing is scheduled, nothing stays proposed
            logger.warning("scheduled job confirmation failed")
            self.service.discard(proposed.scheduled_job.id)
            return "I couldn't ask for that confirmation, so nothing was scheduled."
        return f"{prompt}\nPlease confirm - say yes or no."

    def commit_approval(self, scheduled_job_id: str, job_id: str | None) -> str:
        """Make live the Scheduled Job a confirmation was asked about - called ONLY by the CONFIRM-level tool, i.e. only when the user
        approved the confirmation of THAT Job (`job_id` comes from the engine, never from a plan)."""
        approval = self.confirmer.approved(job_id, scheduled_job_id) if self.confirmer is not None and job_id else None
        if approval is None:
            raise ScheduledJobRequestError("that can only be done by the confirmation that was asked")
        return approved_words(self.service.approve(scheduled_job_id, approval))

    # -- looking and managing -------------------------------------------------------------------------------------------------------

    def _manage(self, request: ManageRequest) -> str:
        if request.action is ManageAction.LIST:
            return self._list()
        if request.everything:
            return BULK
        eligible = self._eligible(request)
        if not eligible:
            return self._nothing_to_do(request)
        if len(eligible) > 1:
            return self._which(request, eligible)
        return self._apply(request.action, eligible[0])

    def _list(self) -> str:
        shown = self.service.listing(MAX_LISTED + 1)
        if not shown:
            return "You don't have any scheduled jobs."
        lines = [f"You have {len(shown) if len(shown) <= MAX_LISTED else MAX_LISTED}{'+' if len(shown) > MAX_LISTED else ''} scheduled job{'' if len(shown) == 1 else 's'}:"]
        lines += [f"{n}. {_line(sj)}" for n, sj in enumerate(shown[:MAX_LISTED], start=1)]
        return "\n".join(lines)

    _STATES = {
        ManageAction.PAUSE: (_S.ACTIVE,),
        ManageAction.RESUME: (_S.PAUSED,),
        ManageAction.CANCEL: (_S.ACTIVE, _S.PAUSED),
        ManageAction.SHOW: (_S.ACTIVE, _S.PAUSED, _S.COMPLETED),
    }

    def _matching(self, request: ManageRequest) -> list[ScheduledJob]:
        words = set(topic_words(request.topic))
        return [
            sj for sj in self.service.listing(100)
            if not words or words <= set(topic_words(f"{sj.title} {sj.intent_text}"))
        ]

    def _eligible(self, request: ManageRequest) -> list[ScheduledJob]:
        return [sj for sj in self._matching(request) if sj.status in self._STATES[request.action]]

    def _nothing_to_do(self, request: ManageRequest) -> str:
        about = self._matching(request)
        if not about:
            what = f" about \"{clean_text(request.topic, 40)}\"" if request.topic else ""
            return f"I couldn't find a scheduled job{what}. Ask \"what scheduled jobs do I have?\" to see them."
        first = about[0]
        if request.action is ManageAction.PAUSE:
            return f"'{clean_text(first.title, 60)}' is {'already paused' if first.status is _S.PAUSED else 'not running'}, so there is nothing to pause."
        if request.action is ManageAction.RESUME:
            return f"'{clean_text(first.title, 60)}' isn't paused."
        return f"'{clean_text(first.title, 60)}' isn't running, so there is nothing to cancel."

    def _which(self, request: ManageRequest, eligible: list[ScheduledJob]) -> str:
        if len(eligible) > MAX_CHOICES:
            return f"That matches {len(eligible)} scheduled jobs - tell me more of what it says so I can pick the right one."
        listing = "\n".join(f"- {_line(sj)}" for sj in eligible)
        return f"That matches more than one scheduled job:\n{listing}\nTell me more of what the one you mean says, and I'll {request.action.value} it."

    def _apply(self, action: ManageAction, sj: ScheduledJob) -> str:
        name = clean_text(sj.title, 60)
        if action is ManageAction.SHOW:
            return detail_words(sj, self.service.last_run(sj.id))
        if action is ManageAction.PAUSE:
            self.service.pause(sj.id)
            return f"Paused: '{name}'. It won't run until you resume it."
        if action is ManageAction.RESUME:
            _, upcoming = self.service.resume(sj.id)
            return f"Resumed: '{name}'. Next run: {_at(upcoming, sj.schedule.tz)}."
        self.service.cancel(sj.id)
        return f"Cancelled: '{name}'. It won't run again."
