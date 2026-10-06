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
from datetime import datetime, timedelta
from collections.abc import Callable

from app.notifications.text import clean_text
from app.orchestration.base import Job, JobStatus, Plan, PlanStep
from app.orchestration.interactions import InteractionStatus, InteractionType, default_ttl_seconds
from app.reminders.timefmt import local_phrase
from app.scheduled_jobs.base import RunOutcome, ScheduledJob, ScheduledJobRun, ScheduledJobStatus, StatusReason
from app.scheduled_operations.presenter import JobFacts, span
from app.scheduled_operations.language import CreateRequest, ManageAction, ManageRequest, hint_name, parse_request, topic_words
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

    def approval(self, job_id: str, scheduled_job_id: str):
        """The confirmation the user APPROVED for exactly this Scheduled Job, or None. Read from the interaction store, so a forged call -
        or the step of a declined, expired or different Job - can never approve."""
        for interaction in self.interactions.list_for_job(job_id):
            if (
                interaction.type is InteractionType.CONFIRMATION
                and interaction.status is InteractionStatus.RESOLVED
                and interaction.detail.get("origin") == ORIGIN
                and interaction.detail.get("scheduled_job_id") == scheduled_job_id
            ):
                return interaction
        return None

    def approved(self, job_id: str, scheduled_job_id: str) -> str | None:
        """The id of that confirmation, or None."""
        interaction = self.approval(job_id, scheduled_job_id)
        return interaction.id if interaction is not None else None

    def ask(self, scheduled_job_id: str, prompt: str, conversation_id: str | None, renews: str | None = None) -> str:
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
            detail={"origin": ORIGIN, "dangerous": False, "scheduled_job_id": scheduled_job_id, **({"renews": renews} if renews else {})},
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
        "Here's what I'll renew:" if proposed.renews else "Here's what I'll set up:",
        f"- I'll: {clean_text(sj.intent_text, _SHOWN)}",
        f"- When: {schedule.describe()}",
        f"- First run: {_at(proposed.first_due, schedule.tz)}",
    ]
    if once:
        lines.append("- It runs once, then it's done.")
    elif proposed.renews:
        lines.append("- This is a fresh approval: the current one ends when you say yes, and the new period lasts up to 90 days from then. Nothing renews itself.")
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
    lines.append("Shall I renew it?" if proposed.renews else "Shall I schedule it?")
    return "\n".join(lines)


def approved_words(approved: Approved) -> str:
    sj = approved.scheduled_job
    if approved.already:
        return f"'{clean_text(sj.title, 80)}' is already {'renewed' if approved.renewed else 'scheduled'}."
    message = f"{'Renewed' if approved.renewed else 'Scheduled'}: '{clean_text(sj.title, 80)}' - {sj.schedule.describe()}. First run: {_at(sj.due_at, sj.schedule.tz)}."
    if approved.lifetime.expires_at is not None:
        message += f" It runs until {_at(approved.lifetime.expires_at, sj.schedule.tz)}; I'll ask before it continues after that."
    return message


#: How close to the end of its approved period a Scheduled Job is when Jarvis starts mentioning that it can be renewed.
RENEWAL_HINT_WITHIN = timedelta(days=7)


def _state_words(sj: ScheduledJob, now: datetime | None = None) -> str:
    tz = sj.schedule.tz
    if sj.status is _S.ACTIVE:
        text = f"{_when_words(sj)}; next {_at(sj.due_at, tz)}"
        if now is not None and sj.expires_at is not None and sj.expires_at - now <= RENEWAL_HINT_WITHIN:
            text += f"; its approved period ends {_at(sj.expires_at, tz)} - say \"renew my {hint_name(sj.title)} scheduled job\" to keep it going"
        return text
    if sj.status is _S.PAUSED:
        return f"{_PAUSE_REASONS.get(sj.status_reason, 'paused')}; {_when_words(sj)}"
    if sj.status is _S.COMPLETED:
        if sj.status_reason is _R.LIFETIME_ENDED:
            return f"its approved period has ended - say \"renew my {hint_name(sj.title)} scheduled job\" to start a new one"
        return _DONE_REASONS.get(sj.status_reason, "finished")
    if sj.status is _S.PROPOSED:
        return "waiting for your approval"
    return sj.status.value


def _line(sj: ScheduledJob, now: datetime | None = None) -> str:
    return f"{clean_text(sj.title, 70)} - {_state_words(sj, now)}"


def _outcome_words(run: ScheduledJobRun | None, tz: str | None) -> str:
    if run is None:
        return "it hasn't run yet"
    when = _at(run.scheduled_for, tz)
    if run.outcome in _OUTCOMES:
        return f"{_OUTCOMES[run.outcome]} ({when})"
    return f"{_RESULTS.get(run.result_state, 'started, not finished yet')} ({when})"


def detail_words(sj: ScheduledJob, last: ScheduledJobRun | None, facts: JobFacts | None = None, now: datetime | None = None) -> str:
    tz = sj.schedule.tz
    lines = [
        f"{clean_text(sj.title, 80)}",
        f"- What: {clean_text(sj.intent_text, _SHOWN)}",
        f"- State: {_state_words(sj, now)}",
        f"- Runs so far: {sj.runs_total}; last: {_outcome_words(last, tz)}",
    ]
    if facts is not None and facts.result_excerpt:  # what the last finished run found: a bounded excerpt of its own report, quoted as data
        lines.append(f"- What it found: \"{facts.result_excerpt}\"")
    if sj.consecutive_failures:
        lines.append(f"- Failed runs in a row: {sj.consecutive_failures}")
    if sj.expires_at is not None:
        lines.append(f"- Approved until: {_at(sj.expires_at, tz)}")
    lines.append("- Read-only: it can't change anything, send anything or control your computer.")
    return "\n".join(lines)


_REFUSED = {
    "disabled": "scheduled jobs are switched off right now",
    "overlap": "its previous run is still going",
    "cap": "it has already run the most it may in 24 hours",
    "global_cap": "I've reached the daily limit for scheduled runs across all your scheduled jobs",
    "busy": "other scheduled runs are still in progress - try again in a moment",
    "device": "the device it was set up from isn't trusted any more",
    "blocked": "something it needs isn't available to an unattended run right now",
    "changed": "a run was just started (or it just changed), so I didn't start another",
    "one_time": "it's a one-time job that runs once, at its time - running it now as well would make it run twice. Cancel it and ask me again if you want it done now",
}


def run_words(sj: ScheduledJob, result, now: datetime) -> str:
    """What to say about a "run it now" request, from what the scheduler answered (`launched`, `refusal`). A refusal started nothing."""
    name = f"'{clean_text(sj.title, 60)}'"
    if getattr(result, "launched", False):
        nxt = f" Its regular schedule is unchanged - next run {_at(sj.due_at, sj.schedule.tz)}." if sj.due_at is not None else ""
        return f"Started: {name} is running now. I'll tell you when it's done.{nxt}"
    refusal = getattr(getattr(result, "refusal", None), "value", "")
    if refusal == "not_active":
        if sj.status is _S.PAUSED:
            return f"{name} is {_PAUSE_REASONS.get(sj.status_reason, 'paused')}, so I haven't run it. Resume it first."
        if sj.status is _S.COMPLETED:
            renew = f" Say \"renew my {hint_name(sj.title)} scheduled job\" to start a new approved period." if sj.status_reason is _R.LIFETIME_ENDED else ""
            return f"{name} has finished, so there is nothing to run.{renew}"
        return f"{name} isn't running, so I haven't run it."
    if refusal == "expired":
        return f"The period you approved for {name} has ended, so I haven't run it. Say \"renew my {hint_name(sj.title)} scheduled job\" to start a new one."
    if refusal == "breaker":
        return f"{name} has failed {sj.consecutive_failures} runs in a row, so I won't run it again until you've looked at it - I'm about to pause it."
    if refusal == "not_found":
        return "I couldn't find that scheduled job, so nothing was run."
    return f"I haven't run {name}: {_REFUSED.get(refusal, 'it could not be started right now')}."


# ====================================== the chat hook ====================================================================


class ScheduledJobOperations:
    def __init__(
        self,
        service: ScheduledJobService,
        confirmer: ScheduledJobApprovalConfirmer | None = None,
        job_facts: Callable[[str], JobFacts | None] | None = None,
    ):
        self.service = service
        self.confirmer = confirmer
        self._job_facts = job_facts

    def command_reply(self, message: str, conversation_id: str | None = None, device_context=None) -> str | None:
        """`device_context` is the Gateway-authenticated DeviceContext of THIS request (None on the anonymous /chat). It is only used
        to remember which trusted device a Scheduled Job came from, so a later revocation can stop it - never to grant anything."""
        request = parse_request(message)
        if request is None:
            return None
        try:
            if isinstance(request, CreateRequest):
                return self._create(request, conversation_id, device_context)
            return self._manage(request, conversation_id, device_context)
        except ScheduledJobRequestError as exc:
            return str(exc)

    # -- creating (asked, never just done) ----------------------------------------------------------------------------------------

    @staticmethod
    def _device_id(device_context) -> str | None:
        if device_context is not None and getattr(device_context, "is_authenticated", False):
            return getattr(device_context, "device_id", None)
        return None

    def _create(self, request: CreateRequest, conversation_id: str | None, device_context) -> str:
        if not request.reads_only:  # "send / add / buy / open ... every day": V1 schedules read-only work only - said aloud, not stored
            return UNATTENDED_REFUSAL
        proposed = self.service.propose(
            request.text, request.when, conversation_id=conversation_id, device_id=self._device_id(device_context)
        )
        return self._ask(proposed, conversation_id)

    def _ask(self, proposed, conversation_id: str | None) -> str:
        """Put a stored proposal (a new one, or a renewal) to the user through the existing confirmation. Nothing is live until it is
        approved; if it cannot be asked, it is withdrawn."""
        if proposed.duplicate:
            if proposed.scheduled_job.status is _S.PROPOSED:
                return "I'm already waiting for your answer on that one - say yes to schedule it, or no to drop it."
            return f"You already have that scheduled: {_line(proposed.scheduled_job)}."
        if self.confirmer is None:
            self.service.discard(proposed.scheduled_job.id)
            return "I can't ask for that confirmation right now, so nothing was scheduled."
        prompt = proposal_words(proposed)
        try:
            self.confirmer.ask(proposed.scheduled_job.id, prompt, conversation_id, renews=proposed.renews)
        except Exception:  # noqa: BLE001 - it cannot be asked: nothing is scheduled, nothing stays proposed
            logger.warning("scheduled job confirmation failed")
            self.service.discard(proposed.scheduled_job.id)
            return "I couldn't ask for that confirmation, so nothing was scheduled."
        return f"{prompt}\nPlease confirm - say yes or no."

    def commit_approval(self, scheduled_job_id: str, job_id: str | None) -> str:
        """Make live the Scheduled Job a confirmation was asked about - called ONLY by the CONFIRM-level tool, i.e. only when the user
        approved the confirmation of THAT Job (`job_id` comes from the engine, never from a plan)."""
        approval = self.confirmer.approval(job_id, scheduled_job_id) if self.confirmer is not None and job_id else None
        if approval is None:
            raise ScheduledJobRequestError("that can only be done by the confirmation that was asked")
        renews = approval.detail.get("renews")  # set only by `ask`, when the proposal is a renewal; the store re-checks it atomically
        return approved_words(self.service.approve(scheduled_job_id, approval.id, renews=renews if isinstance(renews, str) and renews else None))

    # -- looking and managing -------------------------------------------------------------------------------------------------------

    def _manage(self, request: ManageRequest, conversation_id: str | None = None, device_context=None) -> str:
        if request.action is ManageAction.LIST:
            return self._list()
        if request.everything:
            return BULK
        eligible = self._eligible(request)
        if not eligible:
            return self._nothing_to_do(request)
        if len(eligible) > 1:
            return self._which(request, eligible)
        if request.action is ManageAction.RENEW:
            return self._ask(
                self.service.propose_renewal(eligible[0].id, conversation_id=conversation_id, device_id=self._device_id(device_context)),
                conversation_id,
            )
        return self._apply(request.action, eligible[0])

    def _list(self) -> str:
        shown = self.service.listing(MAX_LISTED + 1)
        if not shown:
            return "You don't have any scheduled jobs."
        lines = [f"You have {len(shown) if len(shown) <= MAX_LISTED else MAX_LISTED}{'+' if len(shown) > MAX_LISTED else ''} scheduled job{'' if len(shown) == 1 else 's'}:"]
        now = self.service.now()
        lines += [f"{n}. {_line(sj, now)}" for n, sj in enumerate(shown[:MAX_LISTED], start=1)]
        return "\n".join(lines)

    _STATES = {
        ManageAction.PAUSE: (_S.ACTIVE,),
        ManageAction.RESUME: (_S.PAUSED,),
        ManageAction.CANCEL: (_S.ACTIVE, _S.PAUSED),
        ManageAction.SHOW: (_S.ACTIVE, _S.PAUSED, _S.COMPLETED),
        # asked about in any live or finished state, so that the answer can say exactly WHY it can't be done (paused, finished, ...)
        ManageAction.RUN: (_S.ACTIVE, _S.PAUSED, _S.COMPLETED),
        ManageAction.RENEW: (_S.ACTIVE, _S.PAUSED, _S.COMPLETED),
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
        if request.action in (ManageAction.RUN, ManageAction.RENEW):
            return f"I couldn't find a running scheduled job like that to {request.action.value}."
        return f"'{clean_text(first.title, 60)}' isn't running, so there is nothing to cancel."

    def _which(self, request: ManageRequest, eligible: list[ScheduledJob]) -> str:
        if len(eligible) > MAX_CHOICES:
            return f"That matches {len(eligible)} scheduled jobs - tell me more of what it says so I can pick the right one."
        listing = "\n".join(f"- {_line(sj)}" for sj in eligible)
        return f"That matches more than one scheduled job:\n{listing}\nTell me more of what the one you mean says, and I'll {request.action.value} it."

    def _apply(self, action: ManageAction, sj: ScheduledJob) -> str:
        name = clean_text(sj.title, 60)
        if action is ManageAction.SHOW:
            last = self.service.last_run(sj.id)
            facts = self._job_facts(last.job_id) if self._job_facts is not None and last is not None and last.job_id else None
            return detail_words(sj, last, facts, self.service.now())
        if action is ManageAction.RUN:
            current, result = self.service.run_now(sj.id)
            return run_words(self.service.store.get(sj.id) or current, result, self.service.now())
        if action is ManageAction.PAUSE:
            self.service.pause(sj.id)
            return f"Paused: '{name}'. It won't run until you resume it."
        if action is ManageAction.RESUME:
            _, upcoming = self.service.resume(sj.id)
            return f"Resumed: '{name}'. Next run: {_at(upcoming, sj.schedule.tz)}."
        self.service.cancel(sj.id)
        return f"Cancelled: '{name}'. It won't run again."
