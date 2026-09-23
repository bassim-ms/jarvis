"""
Job Engine (M15): a background poller that turns PENDING Jobs into
validated Plans and RUNNING Jobs into completed/failed ones - the direct
sibling of AutomationEngine (app/automation/engine.py) and
ReminderScheduler (app/reminders/scheduler.py), reusing the exact same
"poll on a daemon thread, isolate every per-item failure, isolate the
poll itself too" shape.

This is the ONE place plan execution actually happens, and it is built
entirely on the existing M5 tool system: `_run_step` only ever calls
`ToolRegistry.execute(tool_name, params=..., confirmed=...)` for an
already-validated PlanStep - never Python/shell directly, never a tool
outside the registry, and never `confirmed=True` for a step whose
permission requires confirmation until an explicit ConfirmJobTool
approval set `step.confirmed = True` first (see `_needs_confirmation`/
`_confirmed_value`). Permission enforcement is entirely ToolRegistry's,
unchanged - this milestone adds a gate in FRONT of it (deciding whether
to even attempt a step this tick), never a bypass behind it.

Core safety discipline - "never claim success when an action actually
failed" - is structural, not a convention: a Job's status can only ever
become COMPLETED in `_advance`'s completion check, which requires every
step to be SUCCEEDED or SKIPPED; a single unresolved FAILED step makes
the whole Job FAILED, with `error` naming the step and reason, no matter
how much of the rest of the plan succeeded.

Restart recovery is mostly a property of persistence alone - exactly
like ReminderScheduler/AutomationEngine, a Job's current status/plan is
just read back from JobStore on the next tick, so nothing "disappears"
across a restart. The one thing that needs explicit handling is a step
caught mid-execution when the process died: `_run_step` persists
`status=RUNNING` (with `attempts` already incremented) BEFORE calling
ToolRegistry.execute, and persists the step's OUTCOME again immediately
after determining it - not only via the next step's own pre-call save or
`_advance`'s trailing save at the end of a whole batch. Both saves matter:
the first means a crash during the call itself leaves a durable RUNNING
marker; the second means a crash any time AFTER the call returns can
never leave an already-decided outcome (SUCCEEDED, in particular) sitting
on disk as RUNNING. Together they guarantee that the ONLY step
`_recover_interrupted_steps` can ever find RUNNING on the next tick is
the single step that was genuinely still in-flight at the moment of the
crash - never one that had already finished. That step's real outcome is
still unknown (it may have already happened), so it's never blindly
re-run: it's marked FAILED and routed through the normal failure/replan
path, the same "never automatically repeat an irreversible action merely
because Jarvis restarted" requirement this project calls out explicitly.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from app.orchestration.base import (
    AutonomyLevel,
    Job,
    JobEvent,
    JobStatus,
    JobStore,
    Plan,
    PlanStep,
    RetryPolicy,
    StepStatus,
    TERMINAL_STEP_STATUSES,
)
from app.orchestration.planner import Planner, PlanningError, ReplanContext
from app.orchestration.validator import PlanValidationError, validate_plan
from app.tools.base import AUTOMATIC_PERMISSIONS, PermissionLevel, ToolResult, ToolResultStatus
from app.tools.registry import ToolRegistry

#: How often JobEngine polls by default when none is configured - matches
#: ReminderScheduler/AutomationEngine's own default of 30s, though
#: settings.orchestration_poll_interval_seconds defaults lower (15s,
#: see app/config.py) since a Job in progress benefits more from being
#: advanced promptly than a due reminder does.
DEFAULT_POLL_INTERVAL_SECONDS = 15.0

#: Statuses JobEngine's own tick ever needs to touch - PAUSED and every
#: terminal status are deliberately excluded, so a paused/cancelled/
#: finished Job is never re-planned or re-advanced (see `tick`).
_ACTIVE_STATUSES = (JobStatus.PENDING, JobStatus.PLANNING, JobStatus.RUNNING)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_tool_result(result: ToolResult) -> dict:
    """A small, JSON-safe summary of a ToolResult for persistence -
    never the live object. `data` is kept as-is only if it's already
    JSON-safe; otherwise stringified, so a step's stored result never
    breaks JobStore's own JSON encoding regardless of what a tool
    returns."""
    data = result.data
    if data is not None and not isinstance(data, (dict, list, str, int, float, bool)):
        data = str(data)
    return {
        "status": result.status.value,
        "message": result.message,
        "data": data,
    }


class JobEngine:
    """Polls `store` for active Jobs and advances each one: planning a
    PENDING Job, then executing a RUNNING Job's ready steps, one batch at
    a time, respecting dependencies/permissions/retries/replans.

    `tick()` is the actual check-and-run step, fully synchronous and with
    all its side effects visible before it returns - tests call it
    directly instead of racing a background thread; `start()`/`stop()`
    just wrap it in a daemon polling loop for the running application,
    same convention as AutomationEngine/ReminderScheduler.
    """

    def __init__(
        self,
        store: JobStore,
        tools: ToolRegistry,
        planner: Planner,
        *,
        max_steps: int,
        max_total_steps: int,
        max_retries_per_step: int,
        max_replans: int,
        retry_delay_seconds: float,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        context_provider=None,
        sleep=time.sleep,
    ):
        self.store = store
        self.tools = tools
        self.planner = planner
        self.max_steps = max_steps
        self.max_total_steps = max_total_steps
        self.max_retries_per_step = max_retries_per_step
        self.max_replans = max_replans
        self.retry_delay_seconds = retry_delay_seconds
        self.poll_interval_seconds = poll_interval_seconds
        # Optional zero-arg callable returning a fresh JSON-safe context
        # dict (see app/orchestration/tools.py:CreateJobTool - the same
        # kind of provider, typically `memory.get_facts`). `None` (the
        # default, and every existing test's default) means "never
        # refresh" - a Job's context stays exactly the one-time snapshot
        # taken at creation, byte-for-byte the pre-refresh behavior.
        self.context_provider = context_provider
        self._sleep = sleep
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- background thread wiring (mirrors AutomationEngine/ReminderScheduler) --

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-job-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a failed poll must not kill the loop
                pass
            self._stop_event.wait(self.poll_interval_seconds)

    # -- the actual work --

    def tick(self) -> int:
        """Advance every currently-active Job once and return how many
        were processed. Never raises past a single Job: each one is
        isolated in its own try/except, the same failure-isolation
        philosophy ReminderScheduler/AutomationEngine already apply."""
        processed = 0
        for status in _ACTIVE_STATUSES:
            for job in self.store.list(status=status):
                try:
                    self._process(job)
                except Exception:  # noqa: BLE001 - isolate one Job's failure
                    continue
                processed += 1
        return processed

    def _process(self, job: Job) -> None:
        self._recover_interrupted_steps(job)

        if job.status in (JobStatus.PENDING, JobStatus.PLANNING):
            self._plan(job)
            if job.status != JobStatus.RUNNING:
                return  # planning failed - already saved as FAILED

        if job.status == JobStatus.RUNNING:
            self._advance(job)

    # -- restart recovery --

    def _recover_interrupted_steps(self, job: Job) -> None:
        """Find any step this process left at RUNNING - only possible if
        a previous process died mid-`_run_step` (see module docstring) -
        and fail it without re-executing anything, then let the normal
        failure/replan path decide what happens next. A no-op on every
        ordinary tick."""
        if job.plan is None:
            return
        interrupted = [s for s in job.plan.steps if s.status == StepStatus.RUNNING]
        if not interrupted:
            return
        for step in interrupted:
            step.status = StepStatus.FAILED
            step.error = "Interrupted by a process restart - not automatically retried."
            self._log(job, "step_interrupted", step_id=step.id, detail=step.error)
            if job.status == JobStatus.RUNNING:
                # Same failure/replan decision a normal exhausted-retry
                # failure gets (see `_handle_step_failure`) - an
                # interrupted step's real outcome is unknown, so it's
                # never just silently retried in place.
                self._handle_step_failure(job, step)
        job.updated_at = _now()
        self.store.save(job)

    # -- planning --

    def _refresh_context(self, job: Job) -> None:
        """Refresh `job.context` from `self.context_provider` immediately
        before a planning call - never relying solely on the one-time
        snapshot CreateJobTool took when the Job was first created, which
        can go stale if a Job sits PENDING for a while, or is paused and
        resumed later. Only ever touches `job.context` - `job.goal`
        (the original goal) and `job.plan`/`job.events` (this Job's own
        execution history) are never altered here, so both survive a
        refresh unchanged, exactly as required.

        A no-op when `context_provider` wasn't given (every pre-existing
        test/caller) - preserves the prior "context is a fixed snapshot"
        behavior exactly. A provider that raises is swallowed: a failed
        context refresh must never block planning, since the alternative
        (a stale-but-present snapshot) is strictly safer than aborting the
        Job outright over a transient memory-store hiccup.
        """
        if self.context_provider is None:
            return
        try:
            fresh = self.context_provider()
        except Exception:  # noqa: BLE001 - a refresh failure must not block planning
            return
        if fresh:
            job.context = dict(fresh)

    def _plan(self, job: Job) -> None:
        job.status = JobStatus.PLANNING
        job.updated_at = _now()
        self.store.save(job)

        self._refresh_context(job)
        available_tools = self.tools.list_tools()
        try:
            raw_steps = self.planner.plan(
                job.goal,
                available_tools=available_tools,
                context=job.context,
                autonomy=job.autonomy,
            )
            steps = validate_plan(
                raw_steps,
                self.tools,
                max_steps=self.max_steps,
                default_retry=RetryPolicy(
                    max_attempts=self.max_retries_per_step,
                    delay_seconds=self.retry_delay_seconds,
                ),
            )
        except (PlanningError, PlanValidationError) as exc:
            job.status = JobStatus.FAILED
            job.error = f"Could not create a plan: {exc}"
            self._log(job, "planning_failed", detail=str(exc))
            job.updated_at = _now()
            self.store.save(job)
            return

        job.plan = Plan(steps=steps)
        job.total_step_count = len(steps)
        job.status = JobStatus.RUNNING
        self._log(job, "planned", detail=f"{len(steps)} step(s)")
        job.updated_at = _now()
        self.store.save(job)

    # -- execution --

    def _advance(self, job: Job) -> None:
        assert job.plan is not None
        progressed = True

        while progressed and job.status == JobStatus.RUNNING:
            progressed = False

            if self._cascade_skips(job):
                progressed = True

            ready = [
                step
                for step in job.plan.steps
                if step.status == StepStatus.PENDING
                and all(self._dependency_satisfied(job.plan, dep) for dep in step.depends_on)
            ]
            if not ready:
                break

            # Run every ready step that doesn't need confirmation FIRST -
            # "Jarvis has prepared the email and selected the recipients;
            # sending it will contact 4 people - send?" (see this
            # project's own Confirmation example) means auto steps in the
            # same ready batch as a gated one still run, and only the
            # gated ones actually pause the Job - never the whole batch.
            auto_ready = [s for s in ready if not self._needs_confirmation(s, job)]
            confirm_ready = [s for s in ready if self._needs_confirmation(s, job)]

            for step in auto_ready:
                self._run_step(job, step)
                progressed = True
                if step.status == StepStatus.FAILED:
                    self._handle_step_failure(job, step)
                    if job.status != JobStatus.RUNNING:
                        break

            if job.status != JobStatus.RUNNING:
                break

            if confirm_ready:
                job.status = JobStatus.AWAITING_CONFIRMATION
                job.pending_step_ids = [s.id for s in confirm_ready]
                job.confirmation_message = self._build_confirmation_message(confirm_ready)
                self._log(
                    job,
                    "confirmation_requested",
                    detail=job.confirmation_message,
                )
                break

        # One last cascade pass regardless of outcome, so a Job that ended
        # FAILED still has accurate SKIPPED bookkeeping for any step whose
        # dependency chain was never going to succeed - never left
        # ambiguously PENDING forever (see `_cascade_skips`).
        while self._cascade_skips(job):
            pass

        if job.status == JobStatus.RUNNING:
            self._maybe_finish(job)

        job.updated_at = _now()
        self.store.save(job)

    @staticmethod
    def _dependency_satisfied(plan: Plan, dep_id: str) -> bool:
        dep_step = plan.get(dep_id)
        return dep_step is not None and dep_step.status == StepStatus.SUCCEEDED

    def _cascade_skips(self, job: Job) -> bool:
        """Mark any PENDING step whose dependency terminally FAILED or
        was itself SKIPPED as SKIPPED too - it can never become eligible
        to run. Returns whether anything changed, so `_advance`'s loop
        knows to re-scan for newly-ready steps."""
        changed = False
        for step in job.plan.steps:
            if step.status != StepStatus.PENDING:
                continue
            blocking = [
                dep
                for dep in step.depends_on
                if (dep_step := job.plan.get(dep)) is not None
                and dep_step.status in (StepStatus.FAILED, StepStatus.SKIPPED)
            ]
            if blocking:
                step.status = StepStatus.SKIPPED
                step.error = f"Skipped: dependency '{blocking[0]}' did not succeed."
                self._log(job, "step_skipped", step_id=step.id, detail=step.error)
                changed = True
        return changed

    def _needs_confirmation(self, step: PlanStep, job: Job) -> bool:
        if step.confirmed:
            return False
        permission = PermissionLevel(step.permission) if step.permission else PermissionLevel.READ
        if permission in AUTOMATIC_PERMISSIONS:
            return False
        if permission is PermissionLevel.DANGEROUS:
            # Structural, not configurable: DANGEROUS always requires
            # confirmation, regardless of autonomy or `job.
            # autonomous_permissions` - see AutonomyLevel's docstring and
            # parse_autonomous_permissions (app/orchestration/base.py),
            # which makes it impossible to even construct a Job whose
            # autonomous_permissions contains "dangerous" in the first
            # place. This check is the second, independent layer of that
            # same guarantee.
            return True
        # permission is CONFIRM here: gated unless THIS Job's own
        # configured `autonomous_permissions` (not a hardcoded engine
        # constant) says AUTONOMOUS may auto-run it.
        if job.autonomy is AutonomyLevel.AUTONOMOUS and permission.value in job.autonomous_permissions:
            return False
        return True

    def _confirmed_value(self, step: PlanStep, job: Job) -> bool | None:
        permission = PermissionLevel(step.permission) if step.permission else PermissionLevel.READ
        if permission in AUTOMATIC_PERMISSIONS:
            return None  # ToolRegistry ignores `confirmed` for automatic permissions anyway
        if step.confirmed:
            return True
        if (
            permission is PermissionLevel.CONFIRM
            and job.autonomy is AutonomyLevel.AUTONOMOUS
            and permission.value in job.autonomous_permissions
        ):
            return True
        return None

    @staticmethod
    def _build_confirmation_message(steps: list[PlanStep]) -> str:
        described = "; ".join(f"{s.tool_name}({s.arguments})" for s in steps)
        plural = "s" if len(steps) != 1 else ""
        return (
            f"Jarvis is ready to run {len(steps)} step{plural} that "
            f"need{'s' if len(steps) == 1 else ''} your confirmation: "
            f"{described}. Confirm to proceed?"
        )

    def _run_step(self, job: Job, step: PlanStep) -> None:
        step.status = StepStatus.RUNNING
        step.attempts += 1
        job.updated_at = _now()
        self.store.save(job)  # persist BEFORE the (possibly irreversible) call - see module docstring
        self._log(job, "step_started", step_id=step.id, detail=f"attempt {step.attempts}")

        try:
            result = self.tools.execute(
                step.tool_name,
                params=step.arguments,
                confirmed=self._confirmed_value(step, job),
            )
        except Exception as exc:  # noqa: BLE001 - a safety net; ToolRegistry already isolates tool failures
            result = ToolResult.error(f"Executor error running '{step.tool_name}': {exc}")

        step.result = _serialize_tool_result(result)

        if result.status is ToolResultStatus.SUCCESS:
            step.status = StepStatus.SUCCEEDED
            step.error = None
            self._log(job, "step_succeeded", step_id=step.id, detail=result.message)
        elif result.status is ToolResultStatus.DENIED:
            step.status = StepStatus.FAILED
            step.error = result.message or "Permission denied."
            self._log(job, "step_denied", step_id=step.id, detail=step.error)
        elif result.status is ToolResultStatus.CONFIRMATION_REQUIRED:
            # Shouldn't happen given `_needs_confirmation` gates this
            # before `_run_step` is ever called - handled defensively
            # anyway: revert to PENDING so the next `_advance` pass's
            # confirmation gate picks it up cleanly rather than
            # recording a spurious failure.
            step.status = StepStatus.PENDING
            step.attempts -= 1
            self._log(job, "step_confirmation_unexpected", step_id=step.id)
        else:  # ERROR
            if step.attempts < step.retry_policy.max_attempts:
                step.status = StepStatus.PENDING
                self._log(
                    job,
                    "step_retry_scheduled",
                    step_id=step.id,
                    detail=result.message,
                )
                if step.retry_policy.delay_seconds > 0:
                    self._sleep(step.retry_policy.delay_seconds)
            else:
                step.status = StepStatus.FAILED
                step.error = result.message or "Tool execution failed."
                self._log(job, "step_failed", step_id=step.id, detail=step.error)

        # Persist this step's OUTCOME immediately, not just its RUNNING
        # start (see the save above and this module's docstring). Without
        # this, a step that actually succeeded stays recorded as RUNNING
        # on disk until either the next step's own pre-call save or
        # `_advance`'s trailing save - a crash inside that window would
        # make `_recover_interrupted_steps` treat an already-SUCCEEDED
        # step as interrupted and potentially replan an alternative
        # action that duplicates one that already really happened.
        # Saving here closes that window: at most the single step
        # actually mid-flight at crash time can ever be found RUNNING on
        # restart.
        job.updated_at = _now()
        self.store.save(job)

    def _handle_step_failure(self, job: Job, step: PlanStep) -> None:
        """A step exhausted its retries. Decide whether replanning is
        worth attempting (bounded by max_replans and max_total_steps) or
        the Job should simply fail - never an uncontrolled replanning
        loop."""
        if job.replan_count >= self.max_replans:
            job.status = JobStatus.FAILED
            job.error = (
                f"Step '{step.id}' ({step.tool_name}) failed: {step.error} "
                f"(replan limit of {self.max_replans} reached)."
            )
            self._log(job, "job_failed", detail=job.error)
            return

        completed = [s for s in job.plan.steps if s.status == StepStatus.SUCCEEDED]
        remaining = [
            s
            for s in job.plan.steps
            if s.status == StepStatus.PENDING and s.id != step.id
        ]
        known_ids = frozenset(s.id for s in completed)

        self._refresh_context(job)  # a replan is a planning call too - see _refresh_context
        try:
            raw_new_steps = self.planner.plan(
                job.goal,
                available_tools=self.tools.list_tools(),
                context=job.context,
                autonomy=job.autonomy,
                replan_context=ReplanContext(
                    completed_steps=completed,
                    failed_step=step,
                    failure_reason=step.error or "unknown error",
                    remaining_steps=remaining,
                ),
            )
            if len(completed) + len(raw_new_steps) > self.max_total_steps:
                raise PlanValidationError(
                    f"Replan would exceed the total step safety limit of "
                    f"{self.max_total_steps}."
                )
            new_steps = validate_plan(
                raw_new_steps,
                self.tools,
                max_steps=self.max_steps,
                default_retry=RetryPolicy(
                    max_attempts=self.max_retries_per_step,
                    delay_seconds=self.retry_delay_seconds,
                ),
                known_step_ids=known_ids,
            )
        except (PlanningError, PlanValidationError) as exc:
            job.status = JobStatus.FAILED
            job.error = (
                f"Step '{step.id}' ({step.tool_name}) failed: {step.error}. "
                f"Replanning also failed: {exc}"
            )
            self._log(job, "replan_failed", detail=str(exc))
            return

        job.plan.steps = completed + new_steps
        job.total_step_count += len(new_steps)
        job.replan_count += 1
        self._log(
            job,
            "replanned",
            step_id=step.id,
            detail=f"{len(new_steps)} replacement step(s) after '{step.id}' failed",
        )
        # job.status stays RUNNING - `_advance`'s own loop re-scans for
        # newly-ready steps on its next iteration.

    def _maybe_finish(self, job: Job) -> None:
        statuses = {step.status for step in job.plan.steps}
        if not statuses <= TERMINAL_STEP_STATUSES:
            return  # something is still PENDING/RUNNING - not done yet

        failed = [s for s in job.plan.steps if s.status == StepStatus.FAILED]
        if failed:
            job.status = JobStatus.FAILED
            names = ", ".join(f"{s.id} ({s.tool_name})" for s in failed)
            job.error = f"Job failed - unresolved failed step(s): {names}."
            self._log(job, "job_failed", detail=job.error)
            return

        job.status = JobStatus.COMPLETED
        succeeded = [s for s in job.plan.steps if s.status == StepStatus.SUCCEEDED]
        skipped = [s for s in job.plan.steps if s.status == StepStatus.SKIPPED]
        summary_parts = [
            f"{s.tool_name}: {s.result.get('message') or s.result.get('data') or 'done'}"
            for s in succeeded
            if s.result
        ]
        summary = "; ".join(str(p) for p in summary_parts) or "All steps completed."
        if skipped:
            summary += f" ({len(skipped)} step(s) skipped as unnecessary.)"
        job.final_result = summary
        self._log(job, "job_completed", detail=summary)

    def _log(self, job: Job, kind: str, *, step_id: str | None = None, detail: str = "") -> None:
        job.events.append(
            JobEvent(timestamp=_now(), kind=kind, detail=detail or "", step_id=step_id)
        )
