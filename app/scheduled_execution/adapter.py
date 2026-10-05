"""
The execution adapter: one due occurrence of one approved Scheduled Job -> one ordinary, PENDING Job, written atomically.

    adapter.build_request(scheduled_job, occurrence)   what the Job WILL be (fresh context built now); persists nothing
    adapter.launch(scheduled_job, occurrence)          builds it, then in ONE transaction: compare-and-set the Scheduled Job,
                                                       write the ledger row, insert the Job (Slice 1's `record_run`)

WHAT THE JOB IS. An ordinary Job in every respect the rest of the system can see: goal = the user's own words, context = the stable
objective (validated by the very same check `create_job` applies) plus FRESH context for this run, autonomy ASSISTED with no
autonomous permissions, no conversation (a background Job). What differs is only the reduction it carries: `run_profile` (the READ-only
unattended profile, which can only take capability away - app/orchestration/profiles.py) and its own `deadline_at`. The existing
JobEngine then plans it with the existing Planner, validates the plan, and runs every step through the existing ToolRegistry with its
own permission checks. Nothing here plans, runs a tool, notifies, writes memory, or decides what is due - and an occurrence that was
already acted on cannot make a second Job (the ledger's UNIQUE constraint, not this code, is what guarantees it).

FAIL CLOSED. Anything unexpected - an inactive Scheduled Job, a profile this system does not define, an objective the usual validation
rejects, no capability available to the profile, a capability boundary (the existing Handle This one) that no longer holds for what
the objective needs, an occurrence that does not match the Scheduled Job - raises `ExecutionRefused` and writes nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.handle_this.base import GoalFamily
from app.handle_this.boundary import DEFAULT_BOUNDARY
from app.handle_this.signals import detect_computer_actions
from app.orchestration.base import (
    OBJECTIVE_KEY,
    REQUEST_CONTEXT_KEY,
    AutonomyLevel,
    Job,
    JobEvent,
    JobPriority,
)
from app.orchestration.profiles import PROFILES, RunProfile, permitted_tools
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import _validate_objective
from app.scheduled_execution.context import FreshContextBuilder
from app.scheduled_execution.occurrence import ScheduledOccurrence
from app.scheduled_jobs.base import (
    KNOWN_PROFILES,
    NewRun,
    ScheduledJob,
    ScheduledJobError,
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobStore,
)


class ExecutionRefused(ScheduledJobError):
    """The occurrence cannot be turned into a Job. Nothing was written."""


@dataclass(frozen=True)
class ExecutionRequest:
    """Everything a Job is created with, decided and visible BEFORE anything is persisted."""

    goal: str
    context: dict[str, Any]
    run_profile: str
    deadline_at: str
    autonomy: AutonomyLevel
    autonomous_permissions: frozenset[str]
    priority: JobPriority
    conversation_id: str | None
    events: tuple[JobEvent, ...]

    def to_job(self) -> Job:
        job = SQLiteJobStore.build_job(
            self.goal,
            priority=self.priority,
            autonomy=self.autonomy,
            context=self.context,
            conversation_id=self.conversation_id,
            autonomous_permissions=self.autonomous_permissions,
            run_profile=self.run_profile,
            deadline_at=self.deadline_at,
        )
        job.events.extend(self.events)
        return job


@dataclass(frozen=True)
class LaunchedOccurrence:
    job: Job
    run: ScheduledJobRun


class ScheduledExecutionAdapter:
    """`scheduled_jobs` is the Slice 1 store; `tools` is the live tool registry, used ONLY to ask which tools the profile
    currently permits (it is never called); `context_builder` builds each run's fresh context; `run_deadline` bounds one run from
    its Job's creation. All injected: the adapter owns no context, no memory, no registry and no clock of its own."""

    def __init__(
        self,
        *,
        scheduled_jobs: ScheduledJobStore,
        tools,
        context_builder: FreshContextBuilder,
        run_deadline: timedelta,
        clock: Callable[[], datetime] | None = None,
    ):
        if not isinstance(run_deadline, timedelta) or run_deadline <= timedelta(0):
            raise ValueError("A run needs a positive deadline.")
        self._store = scheduled_jobs
        self._tools = tools
        self._context = context_builder
        self._run_deadline = run_deadline
        self._clock = clock

    def _now(self) -> datetime:
        return (self._clock() if self._clock is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    def check_objective(self, objective: dict[str, Any], profile_name: str) -> RunProfile:
        """Whether an unattended run of `objective` under `profile_name` could be made RIGHT NOW; returns the profile, or raises
        ExecutionRefused. This is the ONE check, used both when a Scheduled Job is created (so one that could never run is refused
        up front, never weakened) and again by every run (`build_request`): the profile must be one this system defines, the tools it
        permits are re-read from the live registry (a tool that is gone, or no longer read-only, is simply not there), the objective goes
        through the very check `create_job` applies, and the existing capability boundary is re-run against what the Planner would be
        offered. It only ever refuses: it grants nothing and persists nothing."""
        if profile_name not in KNOWN_PROFILES or profile_name not in PROFILES:
            raise ExecutionRefused(f"Unknown capability profile '{profile_name}'.")
        profile = PROFILES[profile_name]
        permitted = permitted_tools(profile, self._tools)
        if not permitted:
            raise ExecutionRefused("No capability is available for an unattended run right now.")
        problem = _validate_objective(objective)
        if problem:
            raise ExecutionRefused(f"The stored objective is not acceptable: {problem}")
        self._reassess_boundary(objective, frozenset(tool.name for tool in permitted))
        return profile

    # -- what the Job will be --------------------------------------------------------------------------------------------------

    def build_request(self, scheduled_job: ScheduledJob, occurrence: ScheduledOccurrence) -> ExecutionRequest:
        """The Job this occurrence would become, with FRESH context built now. Persists nothing; refuses (ExecutionRefused) anything
        that does not hold together."""
        if scheduled_job.status is not ScheduledJobStatus.ACTIVE:
            raise ExecutionRefused(f"A {scheduled_job.status.value} Scheduled Job does not run.")
        if occurrence.scheduled_job_id != scheduled_job.id:
            raise ExecutionRefused("That occurrence belongs to a different Scheduled Job.")
        if scheduled_job.due_at != occurrence.expected_due_at:
            raise ExecutionRefused("That occurrence was decided for an earlier state of this Scheduled Job.")

        # The very check a NEW Scheduled Job passes at creation, applied again now to the live registry.
        profile = self.check_objective(scheduled_job.objective, scheduled_job.profile)

        built = self._context.build(scheduled_job, occurrence)  # fresh, every time
        fingerprint = hashlib.sha256(occurrence.occurrence.encode("utf-8")).hexdigest()[:32]
        context: dict[str, Any] = dict(built.facts)
        context[OBJECTIVE_KEY] = {**scheduled_job.objective, "fingerprint": fingerprint}  # unique per occurrence
        context[REQUEST_CONTEXT_KEY] = built.request_context

        stamp = self._now().isoformat()
        events = [JobEvent(timestamp=stamp, kind="scheduled_run", detail=occurrence.occurrence)]
        events += [JobEvent(timestamp=stamp, kind="context_note", detail=note) for note in built.notes]
        return ExecutionRequest(
            goal=scheduled_job.intent_text,
            context=context,
            run_profile=profile.name,
            deadline_at=(self._now() + self._run_deadline).isoformat(),
            autonomy=AutonomyLevel.ASSISTED,
            autonomous_permissions=frozenset(),  # AUTONOMOUS would auto-run CONFIRM steps; a scheduled run has none to auto-run
            priority=JobPriority.MEDIUM,
            conversation_id=None,  # a background Job: there is no conversation to answer in
            events=tuple(events),
        )

    @staticmethod
    def _reassess_boundary(objective: dict[str, Any], offered: frozenset[str]) -> None:
        """Re-run, at run time, the existing Handle This capability boundary (the one intake applied when the objective was first
        understood): every family the stored objective needs must still have a tool on offer, and every computer action its wording
        asks for must still have all of its tools on offer. `offered` is the profile-filtered, live set the Planner will be given - a
        subset of what intake saw - so this can only refuse, never admit. A family the system cannot read fails closed. Refusal
        raises ExecutionRefused, before anything is built or written."""
        try:
            values = objective["required_families"]
            if isinstance(values, str):  # a bare string would be read character by character
                raise TypeError("required_families must be a list")
            required = tuple(GoalFamily(value) for value in values)
        except (KeyError, TypeError, ValueError):
            raise ExecutionRefused("The stored objective names a kind of work this system does not recognise.") from None
        if not required:  # an objective that needs nothing is not one this boundary can vouch for
            raise ExecutionRefused("The stored objective names a kind of work this system does not recognise.")
        unavailable = DEFAULT_BOUNDARY.assess(required, offered).unavailable
        if unavailable:
            kinds = ", ".join(family.value for family in unavailable)
            raise ExecutionRefused(f"What this objective needs ({kinds}) is not available to an unattended run right now.")
        if GoalFamily.COMPUTER_WORKFLOW in required:
            missing = DEFAULT_BOUNDARY.assess_actions(detect_computer_actions(objective["original_request"]), offered)
            if missing:
                raise ExecutionRefused(
                    f"This objective asks for computer actions ({', '.join(missing)}) that an unattended run cannot take."
                )

    # -- making it a Job, atomically ---------------------------------------------------------------------------------------------

    def launch(self, scheduled_job: ScheduledJob, occurrence: ScheduledOccurrence) -> LaunchedOccurrence:
        """Create the Job and its ledger row in ONE transaction (the Scheduled Job's `due_at` moves on in the same step). If the
        occurrence was already acted on, or the Scheduled Job changed since it was read, nothing is written and the store's
        DuplicateOccurrenceError / StaleScheduledJobError says which - in particular no second Job is ever created."""
        request = self.build_request(scheduled_job, occurrence)
        job = request.to_job()
        run = self._store.record_run(
            scheduled_job.id,
            NewRun(
                occurrence=occurrence.occurrence,
                scheduled_for=occurrence.scheduled_for,
                outcome=occurrence.outcome,
                kind=occurrence.kind,
                job_id=job.id,
                missed_count=occurrence.missed_count,
                late=occurrence.late,
            ),
            expected_version=scheduled_job.version,
            expected_due_at=occurrence.expected_due_at,
            next_due_at=occurrence.next_due_at,
            complete_reason=occurrence.complete_reason,
            insert_job=lambda connection: SQLiteJobStore.insert_job_locked(connection, job),
            via="scheduled_execution",
            now=self._now(),
        )
        return LaunchedOccurrence(job=job, run=run)
