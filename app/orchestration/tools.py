"""
Orchestration tools (M15): adapt JobStore (and JobEngine's confirmation
protocol) into the Tool interface, so create/list/get/confirm/pause/
resume/cancel all reach a Job through the same ToolRegistry
(app/tools/registry.py) and TOOL capability (app/capability.py) every
other tool in this project already uses - no special-casing in
JarvisCore beyond `_handle_task` calling `create_job` (see app/core.py).

Every tool here is local metadata (creating/reading/transitioning a
Job's own persisted record) - never itself the thing that runs a plan
step - so, like the reminder/task/automation control tools, every one of
them is READ or SAFE_ACTION. The actual plan steps a Job runs still go
through ToolRegistry with their OWN tool's permission enforced exactly
as it always is (see app/orchestration/engine.py) - approving a Job's
pending confirmation here never itself grants elevated trust to whatever
tool that confirmation was about.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.orchestration.base import (
    AutonomousPermissionError,
    AutonomyError,
    AutonomyLevel,
    Job,
    JobPriorityError,
    JobStatus,
    JobStore,
    StepStatus,
    TERMINAL_JOB_STATUSES,
    parse_autonomous_permissions,
    parse_autonomy,
    parse_priority,
)
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_DEFAULT_AUTONOMY = AutonomyLevel.ASSISTED.value


def _step_to_dict(step) -> dict[str, Any]:
    return {
        "id": step.id,
        "tool": step.tool_name,
        "arguments": step.arguments,
        "depends_on": step.depends_on,
        "status": step.status.value,
        "attempts": step.attempts,
        "error": step.error,
        "result": step.result,
    }


def _job_to_summary_dict(job: Job) -> dict[str, Any]:
    """A compact view for list_jobs - full step-level plan/events omitted
    (use get_job for that), but enough to answer a natural-language
    status question in one call: `final_result`/`error`/
    `confirmation_message` are included precisely so
    JarvisCore._handle_task_status/_handle_confirmation (app/core.py) can
    resolve and describe "is that finished?"/"what's the status of my
    task?" from a single list_jobs call, without a second get_job
    round-trip. `conversation_id` is what lets those same handlers scope
    "my task" to the conversation it was actually created in."""
    return {
        "id": job.id,
        "goal": job.goal,
        "status": job.status.value,
        "priority": job.priority.value,
        "autonomy": job.autonomy.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "conversation_id": job.conversation_id,
        "final_result": job.final_result,
        "error": job.error,
        "confirmation_message": job.confirmation_message,
    }


def _job_to_detail_dict(job: Job) -> dict[str, Any]:
    detail = _job_to_summary_dict(job)  # already carries final_result/error/confirmation_message
    detail.update(
        {
            "plan": [_step_to_dict(s) for s in job.plan.steps] if job.plan else None,
            "replan_count": job.replan_count,
            "pending_step_ids": job.pending_step_ids,
            "autonomous_permissions": sorted(job.autonomous_permissions),
            "events": [
                {
                    "timestamp": e.timestamp,
                    "kind": e.kind,
                    "detail": e.detail,
                    "step_id": e.step_id,
                }
                for e in job.events
            ],
        }
    )
    return detail


class CreateJobTool(Tool):
    """Creates a new goal-driven Job. SAFE_ACTION: it only ever adds a
    new PENDING row for JobEngine to plan/execute later - purely local,
    no side effects of its own. `goal` is the only required parameter
    (mirrors CreateTaskTool, app/tasks/tools.py), so ToolCallBuilder can
    always build the call deterministically from trailing message text."""

    name = "create_job"
    description = (
        "Start a new goal-driven task: Jarvis will plan and execute the "
        "steps needed to achieve the given goal using its available "
        "tools, in the background."
    )
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: JobStore, context_provider=None):
        self.store = store
        # Optional zero-arg callable returning a JSON-safe dict, used to
        # snapshot "relevant context" (e.g. known user facts) at creation
        # time - see app/tools/defaults.py, which wires this to
        # `memory.get_facts`. Defaults to an empty context rather than
        # requiring every caller/test to supply one.
        self.context_provider = context_provider or (lambda: {})

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="goal", type="string", description="What to accomplish"),
            ToolParameter(
                name="priority",
                type="string",
                description="low, medium, or high (default medium)",
                required=False,
                default="medium",
            ),
            ToolParameter(
                name="autonomy",
                type="string",
                description="manual, assisted, or autonomous (default assisted)",
                required=False,
                default=_DEFAULT_AUTONOMY,
            ),
            ToolParameter(
                name="conversation_id",
                type="string",
                description=(
                    "The conversation this task was requested from, so a later "
                    "'is that finished?' in the same conversation resolves to it"
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="autonomous_permissions",
                type="string",
                description=(
                    "Comma-separated permission levels AUTONOMOUS mode may "
                    "auto-run without confirmation - only 'confirm' is valid "
                    "('dangerous' can never be auto-approved). An empty string "
                    "opts this task out of AUTONOMOUS auto-approval entirely. "
                    "Omit to use the default ('confirm')."
                ),
                required=False,
                default=None,
            ),
        ]

    def execute(
        self,
        goal: str,
        priority: str = "medium",
        autonomy: str = _DEFAULT_AUTONOMY,
        conversation_id: str | None = None,
        autonomous_permissions: str | None = None,
    ) -> ToolResult:
        try:
            parsed_priority = parse_priority(priority)
        except JobPriorityError as exc:
            return ToolResult.error(str(exc))
        try:
            parsed_autonomy = parse_autonomy(autonomy)
        except AutonomyError as exc:
            return ToolResult.error(str(exc))
        try:
            parsed_autonomous_permissions = parse_autonomous_permissions(
                # A comma-separated string, same convention as the existing
                # *_PROVIDER_ORDER settings (app/config.py) - `None` means
                # "not configured" (use the default); "" means an explicit,
                # deliberate empty set (opt out of AUTONOMOUS auto-approval).
                None if autonomous_permissions is None else
                [v.strip() for v in autonomous_permissions.split(",") if v.strip()]
            )
        except AutonomousPermissionError as exc:
            return ToolResult.error(str(exc))

        context = self.context_provider() or {}
        job = self.store.create(
            goal=goal,
            priority=parsed_priority,
            autonomy=parsed_autonomy,
            context=context,
            conversation_id=conversation_id,
            autonomous_permissions=parsed_autonomous_permissions,
        )
        return ToolResult.success(
            data=_job_to_summary_dict(job),
            message=(
                f"Started task '{job.id}' for goal: {goal!r}. Jarvis will plan "
                "and work on it in the background."
            ),
        )


class ListJobsTool(Tool):
    """Lists Jobs, optionally filtered by status. READ: no side effects."""

    name = "list_jobs"
    description = "List goal-driven tasks, optionally filtered by status."
    permission = PermissionLevel.READ

    def __init__(self, store: JobStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                type="string",
                description=(
                    "Filter by status: pending, planning, awaiting_confirmation, "
                    "running, paused, failed, completed, or cancelled"
                ),
                required=False,
                default=None,
            )
        ]

    def execute(self, status: str | None = None) -> ToolResult:
        parsed_status = None
        if status is not None:
            try:
                parsed_status = JobStatus(status)
            except ValueError:
                return ToolResult.error(
                    f"Unknown status: '{status}'. Must be one of: "
                    f"{sorted(s.value for s in JobStatus)}"
                )
        jobs = self.store.list(status=parsed_status)
        return ToolResult.success(
            data=[_job_to_summary_dict(j) for j in jobs],
            message=f"{len(jobs)} task(s) found.",
        )


class GetJobTool(Tool):
    """Returns full detail (plan, step results, events) for one Job.
    READ: no side effects."""

    name = "get_job"
    description = "Get full detail (plan, progress, results) for one task by id."
    permission = PermissionLevel.READ

    def __init__(self, store: JobStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="job_id", type="string", description="The task's id")]

    def execute(self, job_id: str) -> ToolResult:
        job = self.store.get(job_id)
        if job is None:
            return ToolResult.error(f"No such task: '{job_id}'")
        return ToolResult.success(data=_job_to_detail_dict(job))


class ConfirmJobTool(Tool):
    """Resolves a Job's current AWAITING_CONFIRMATION batch. SAFE_ACTION:
    approving only ever authorizes the specific steps JobEngine already
    named in `pending_step_ids` for THIS batch - never the whole
    remaining plan, and never a tool's own permission requirement (see
    this module's docstring). Declining fails the Job outright with an
    accurate explanation, rather than leaving it stuck forever."""

    name = "confirm_job"
    description = (
        "Approve or decline a task's pending confirmation request "
        "(see its confirmation_message via get_job)."
    )
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: JobStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="job_id", type="string", description="The task's id"),
            ToolParameter(
                name="approve", type="boolean", description="True to approve, False to decline"
            ),
        ]

    def execute(self, job_id: str, approve: bool) -> ToolResult:
        job = self.store.get(job_id)
        if job is None:
            return ToolResult.error(f"No such task: '{job_id}'")
        if job.status != JobStatus.AWAITING_CONFIRMATION:
            return ToolResult.error(
                f"Task '{job_id}' is not awaiting confirmation (status: {job.status.value})."
            )

        now = _now()

        if approve:
            for step_id in job.pending_step_ids:
                step = job.plan.get(step_id) if job.plan else None
                if step is not None:
                    step.confirmed = True
            job.pending_step_ids = []
            job.confirmation_message = None
            job.status = JobStatus.RUNNING
            message = f"Task '{job_id}' confirmed - resuming execution."
        else:
            declined = job.confirmation_message
            for step_id in job.pending_step_ids:
                step = job.plan.get(step_id) if job.plan else None
                if step is not None:
                    step.error = "Confirmation declined by user."
                    step.status = StepStatus.FAILED
            job.pending_step_ids = []
            job.confirmation_message = None
            job.status = JobStatus.FAILED
            job.error = f"Task cancelled: user declined confirmation ({declined})."
            message = f"Task '{job_id}' declined - task marked as failed."

        job.updated_at = now
        self.store.save(job)
        return ToolResult.success(data=_job_to_summary_dict(job), message=message)


def _transition(store: JobStore, job_id: str, *, from_ok, to_status: JobStatus, verb: str) -> ToolResult:
    job = store.get(job_id)
    if job is None:
        return ToolResult.error(f"No such task: '{job_id}'")
    if not from_ok(job.status):
        return ToolResult.error(
            f"Cannot {verb} task '{job_id}' from status '{job.status.value}'."
        )
    job.status = to_status
    job.updated_at = _now()
    store.save(job)
    return ToolResult.success(
        data=_job_to_summary_dict(job), message=f"Task '{job_id}' {verb}d."
    )


class PauseJobTool(Tool):
    """Pauses an active Job - JobEngine skips a PAUSED Job entirely until
    resumed. SAFE_ACTION: reversible, purely local."""

    name = "pause_job"
    description = "Pause an in-progress task so it stops advancing until resumed."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: JobStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="job_id", type="string", description="The task's id")]

    def execute(self, job_id: str) -> ToolResult:
        non_terminal_active = {
            JobStatus.PENDING,
            JobStatus.PLANNING,
            JobStatus.AWAITING_CONFIRMATION,
            JobStatus.RUNNING,
        }
        return _transition(
            self.store,
            job_id,
            from_ok=lambda s: s in non_terminal_active,
            to_status=JobStatus.PAUSED,
            verb="pause",
        )


class ResumeJobTool(Tool):
    """Resumes a PAUSED Job back to RUNNING - JobEngine's next tick picks
    it back up exactly where its plan left off. SAFE_ACTION."""

    name = "resume_job"
    description = "Resume a paused task."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: JobStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="job_id", type="string", description="The task's id")]

    def execute(self, job_id: str) -> ToolResult:
        job = self.store.get(job_id)
        if job is None:
            return ToolResult.error(f"No such task: '{job_id}'")
        if job.status != JobStatus.PAUSED:
            return ToolResult.error(
                f"Cannot resume task '{job_id}' from status '{job.status.value}'."
            )
        # A Job with no plan yet (paused before planning ever completed)
        # resumes as PENDING so JobEngine plans it; otherwise RUNNING so
        # `_advance` re-evaluates ready steps from the persisted plan -
        # never re-running an already-SUCCEEDED step (see engine.py).
        target = JobStatus.PENDING if job.plan is None else JobStatus.RUNNING
        return _transition(
            self.store, job_id, from_ok=lambda s: s == JobStatus.PAUSED, to_status=target, verb="resume"
        )


class CancelJobTool(Tool):
    """Cancels a Job - terminal; JobEngine never advances a CANCELLED Job
    again, and no further steps ever run. SAFE_ACTION: cancelling your
    own task is the user's prerogative, same convention as
    CancelReminderTool/CancelAutomationTool/CancelTaskTool."""

    name = "cancel_job"
    description = "Cancel a task. It will not run any further steps."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: JobStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="job_id", type="string", description="The task's id")]

    def execute(self, job_id: str) -> ToolResult:
        return _transition(
            self.store,
            job_id,
            from_ok=lambda s: s not in TERMINAL_JOB_STATUSES,
            to_status=JobStatus.CANCELLED,
            verb="cancel",
        )
