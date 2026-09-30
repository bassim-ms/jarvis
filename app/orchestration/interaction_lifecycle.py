"""
The interaction lifecycle: what happens to a question or confirmation WHILE
it waits for the user - telling the user, reminding them (a little, then
never again), and closing it when it runs out of time.

One `process()` pass, run at the end of every JobEngine tick (and therefore
once right after a restart, which is how overdue interactions get closed):

1. ADOPT. A Job parked in AWAITING_CONFIRMATION that has no pending
   confirmation gets one - the engine parks Jobs by status alone, and the
   interaction (with its code, its lifetime, its reminders) is what the rest
   of the system speaks about. Idempotent, and bounded by the per-Job
   interaction limit.
2. For every PENDING interaction, look at its Job:
   - the Job is finished / cancelled / moved on -> the interaction is stale:
     withdrawn, silently (`JobEngine.discard_stale_interaction`);
   - the Job is PAUSED -> left alone, no reminders (nothing is being asked
     of the user while the Job is paused);
   - its lifetime has passed -> EXPIRED through `JobEngine.expire_interaction`
     (the same atomic Job+interaction transaction resolve/decline use), then
     the user is told once;
   - otherwise it is waiting: the notification schedule below decides whether
     something is due.

NOTIFICATION SCHEDULE (per interaction type, configurable; see
ReminderSchedule): the first notice is immediate (`waiting`), then reminders
at fixed offsets after the interaction was opened, then ONE final warning a
fixed lead time before it expires, then (once) the expiry notice. Reminders
plus the final warning are capped by `max_reminders` - a hard limit, so
nobody is notified forever. If the process was down, only the LATEST stage
that is due is sent (never a burst of everything missed).

DE-DUPLICATION is the notification router's: every stage has a stable key
("interaction:<id>:reminder:2"), and the router delivers a key at most once,
across ticks, retries and restarts. A resolved / declined / cancelled /
expired interaction is no longer pending, so nothing about it is ever sent
again - a pending reminder is simply never reached.

Nothing here resolves anything, executes any tool, or changes a Job except
through the engine's own transition (expiry). Notification failures cannot
reach a Job: the router never raises, and every step here is isolated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from app.notifications.base import DEFAULT_USER, NotificationEvent, NotificationRouter
from app.notifications.text import clean_text
from app.orchestration.base import Job, JobStatus, JobStore
from app.orchestration.engine import JobEngine
from app.orchestration.interactions import (
    MAX_INTERACTIONS_PER_JOB,
    InteractionError,
    InteractionRequest,
    InteractionStore,
    InteractionType,
    StaleInteractionError,
    default_ttl_seconds,
    is_expired,
    utc_now,
)
from app.orchestration.base import StaleJobError
from app.orchestration.notification_events import (
    expired_event,
    expiry_warning_event,
    reminder_event,
    reminders_prefix,
    waiting_event,
)
from app.tools.base import PermissionLevel

#: No configuration may raise the cap on reminders + the final warning above
#: this: a misconfigured schedule must not become a nuisance.
HARD_MAX_REMINDERS = 10


@dataclass(frozen=True)
class Stage:
    kind: Literal["waiting", "reminder", "warning"]
    ordinal: int  # 0 for waiting/warning; the reminder's 1-based position in the schedule
    at: datetime


@dataclass(frozen=True)
class ReminderSchedule:
    """When to notify about one kind of interaction.

    `reminder_offsets_seconds`: seconds after the interaction was opened at
    which a reminder is due (strictly increasing). `final_warning_lead_seconds`:
    how long before expiry the single final warning is due (None = none).
    `max_reminders`: hard cap on reminders + the final warning together."""

    reminder_offsets_seconds: tuple[float, ...] = ()
    final_warning_lead_seconds: float | None = None
    max_reminders: int = 4

    def __post_init__(self) -> None:
        offsets = tuple(self.reminder_offsets_seconds)
        if any(isinstance(o, bool) or not isinstance(o, (int, float)) or o <= 0 for o in offsets):
            raise ValueError("reminder offsets must be positive numbers of seconds.")
        if any(b <= a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("reminder offsets must be strictly increasing.")
        lead = self.final_warning_lead_seconds
        if lead is not None and (isinstance(lead, bool) or not isinstance(lead, (int, float)) or lead <= 0):
            raise ValueError("final_warning_lead_seconds must be a positive number or None.")
        if (
            isinstance(self.max_reminders, bool)
            or not isinstance(self.max_reminders, int)
            or not 0 <= self.max_reminders <= HARD_MAX_REMINDERS
        ):
            raise ValueError(f"max_reminders must be an integer from 0 to {HARD_MAX_REMINDERS}.")
        object.__setattr__(self, "reminder_offsets_seconds", offsets)

    @property
    def warning_allowed(self) -> bool:
        return self.final_warning_lead_seconds is not None and self.max_reminders >= 1

    def stages(self, created: datetime, expires_at: datetime | None) -> list[Stage]:
        """The notification stages for an interaction opened at `created`
        that expires at `expires_at` (or never), in time order. Never more
        than `max_reminders` reminder-like stages (reminders + warning), and
        nothing at or after the expiry."""
        stages = [Stage("waiting", 0, created)]

        warning_at: datetime | None = None
        if self.warning_allowed and expires_at is not None:
            candidate = expires_at - timedelta(seconds=self.final_warning_lead_seconds)
            if candidate > created:
                warning_at = candidate

        budget = self.max_reminders - (1 if warning_at is not None else 0)
        taken = 0
        for ordinal, offset in enumerate(self.reminder_offsets_seconds, start=1):
            if taken >= budget:
                break
            at = created + timedelta(seconds=offset)
            if expires_at is not None and at >= expires_at:
                break
            if warning_at is not None and at >= warning_at:
                break  # the final warning takes over from here
            stages.append(Stage("reminder", ordinal, at))
            taken += 1

        if warning_at is not None:
            stages.append(Stage("warning", 0, warning_at))
        return stages


def _parse_offsets(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in str(raw).split(",") if part.strip())


def schedules_from_settings() -> dict[InteractionType, ReminderSchedule]:
    """The configured schedules. A malformed setting falls back to the
    built-in default for that kind: a typo must never disable the schedule
    entirely or, worse, make it noisy."""
    from app.config import settings

    cap = settings.notifications_max_reminders_per_interaction
    cap = cap if isinstance(cap, int) and 0 <= cap <= HARD_MAX_REMINDERS else 4

    def build(offsets: str, lead: float, default: ReminderSchedule) -> ReminderSchedule:
        try:
            return ReminderSchedule(_parse_offsets(offsets), lead, cap)
        except (ValueError, TypeError):
            return default

    return {
        InteractionType.CONFIRMATION: build(
            settings.orchestration_confirmation_reminder_offsets_seconds,
            settings.orchestration_confirmation_final_warning_lead_seconds,
            ReminderSchedule((7200, 21600), 14400, 4),
        ),
        InteractionType.INPUT: build(
            settings.orchestration_input_reminder_offsets_seconds,
            settings.orchestration_input_final_warning_lead_seconds,
            ReminderSchedule((86400, 259200), 86400, 4),
        ),
    }


class InteractionLifecycle:
    """See the module docstring. `engine` provides expiry and stale-discard;
    `router` delivers notifications; `interactions`/`jobs` are read here."""

    def __init__(
        self,
        interactions: InteractionStore,
        jobs: JobStore,
        engine: JobEngine,
        router: NotificationRouter,
        *,
        schedules: dict[InteractionType, ReminderSchedule] | None = None,
        clock: Callable[[], datetime] | None = None,
        user: str = DEFAULT_USER,
        adopt_confirmations: bool = True,
    ):
        self.interactions = interactions
        self.jobs = jobs
        self.engine = engine
        self.router = router
        self.schedules = schedules if schedules is not None else schedules_from_settings()
        self._clock = clock or utc_now
        self.user = user
        self.adopt_confirmations = adopt_confirmations

    def attach(self) -> "InteractionLifecycle":
        """Run `process` at the end of every engine tick."""
        self.engine.add_tick_hook(self.process)
        return self

    # -- one pass ------------------------------------------------------------------------------

    def process(self) -> None:
        """One housekeeping pass. Never raises: each interaction is isolated,
        and a failure just means it is looked at again next tick."""
        now = self._clock()
        if self.adopt_confirmations:
            try:
                self._adopt(now)
            except Exception:  # noqa: BLE001
                pass
        try:
            pending = self.interactions.list_pending()
        except Exception:  # noqa: BLE001
            return
        for interaction in pending:
            try:
                self._process_one(interaction, now)
            except Exception:  # noqa: BLE001
                continue

    # -- adoption ------------------------------------------------------------------------------

    def _adopt(self, now: datetime) -> None:
        for job in self.jobs.list(status=JobStatus.AWAITING_CONFIRMATION):
            try:
                self._adopt_job(job, now)
            except Exception:  # noqa: BLE001 - one Job never blocks the rest
                continue

    def _adopt_job(self, job: Job, now: datetime) -> None:
        if job.plan is None or not job.pending_step_ids:
            return
        if any(i.type is InteractionType.CONFIRMATION for i in self.interactions.list_pending_for_job(job.id)):
            return
        if len(self.interactions.list_for_job(job.id)) >= MAX_INTERACTIONS_PER_JOB:
            return  # the per-Job limit: the legacy confirm path remains for it
        fresh = self.jobs.get(job.id)  # nothing may have moved it since it was listed
        if fresh is None or fresh.version != job.version or fresh.status is not JobStatus.AWAITING_CONFIRMATION:
            return
        steps = [s for s in (job.plan.get(sid) for sid in job.pending_step_ids) if s is not None]
        if len(steps) != len(job.pending_step_ids):
            return
        self.interactions.create(
            job.id,
            InteractionType.CONFIRMATION,
            self._confirmation_prompt(job, steps),
            step_ids=list(job.pending_step_ids),
            detail={
                "origin": "job-engine",
                "dangerous": any(s.permission == PermissionLevel.DANGEROUS.value for s in steps),
            },
            ttl_seconds=default_ttl_seconds(InteractionType.CONFIRMATION),
            now=now,
        )

    def _confirmation_prompt(self, job: Job, steps) -> str:
        """What the user is asked: the goal and, in plain words, what the
        gated step(s) do (the tool's own description - never its name or
        arguments, which may be internal or private)."""
        described = []
        for step in steps[:2]:
            tool = self.engine.tools.get(step.tool_name)
            first = (tool.description or "").split(". ")[0].rstrip(".") if tool is not None else ""
            described.append(clean_text(first, 80) or "an action")
        more = len(steps) - len(described)
        what = "; ".join(described) + (f" (and {more} more)" if more > 0 else "")
        step_word = "this step" if len(steps) == 1 else "these steps"
        return f'Shall I go ahead with {step_word} of "{clean_text(job.goal, 60)}": {what}?'

    # -- per interaction -----------------------------------------------------------------------

    def _process_one(self, interaction: InteractionRequest, now: datetime) -> None:
        job = self.jobs.get(interaction.job_id)
        if job is None:
            return
        state = self.engine.interaction_state(interaction, job)
        if state == JobEngine.STALE:
            self.engine.discard_stale_interaction(interaction.id, expected_version=interaction.version)
            return
        if state == JobEngine.PAUSED:
            return
        if is_expired(interaction, now):
            self._expire(interaction, job, now)
            return
        self._notify_stage(interaction, job, now)

    def _expire(self, interaction: InteractionRequest, job: Job, now: datetime) -> None:
        try:
            applied = self.engine.expire_interaction(
                interaction.id,
                expected_version=interaction.version,
                expected_job_version=job.version,
                now=now,
            )
        except (StaleInteractionError, StaleJobError):
            return  # something changed it meanwhile: looked at again next tick
        except InteractionError:
            return  # not answerable/expirable any more: the next pass re-classifies it
        self._send(expired_event(applied.interaction, applied.job))

    def _notify_stage(self, interaction: InteractionRequest, job: Job, now: datetime) -> None:
        schedule = self.schedules.get(interaction.type)
        if schedule is None:
            return
        created = datetime.fromisoformat(interaction.created_at)
        expires = datetime.fromisoformat(interaction.expires_at) if interaction.expires_at else None
        stages = schedule.stages(created, expires)
        due = [s for s in stages if s.at <= now]
        if not due:
            return
        stage = due[-1]  # only the latest due stage: never a burst of missed ones

        if stage.kind == "waiting":
            event = waiting_event(interaction, job)
        elif stage.kind == "reminder":
            reminder_cap = sum(1 for s in stages if s.kind == "reminder")
            if self.router.sent_count(reminders_prefix(interaction.id)) >= reminder_cap:
                return  # the hard maximum
            event = reminder_event(interaction, job, stage.ordinal)
        else:
            remaining = (expires - now).total_seconds() if expires else 0.0
            event = expiry_warning_event(interaction, job, remaining)

        if self.router.already_sent(event.key):
            return
        self._send(event)

    def _send(self, event: NotificationEvent) -> None:
        try:
            self.router.notify(self.user, event)
        except Exception:  # noqa: BLE001 - the router never raises; belt and braces
            pass
