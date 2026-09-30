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

import hashlib
import json
import threading
import unicodedata
from datetime import datetime, timezone
from typing import Any

from app.orchestration.base import (
    DECLINED_STEP_ERROR,
    OBJECTIVE_KEY,
    AutonomousPermissionError,
    AutonomyError,
    AutonomyLevel,
    Job,
    JobPriorityError,
    REQUEST_CONTEXT_KEY,
    JobStatus,
    JobStore,
    StaleJobError,
    StepStatus,
    TERMINAL_JOB_STATUSES,
    parse_autonomous_permissions,
    parse_autonomy,
    parse_priority,
)
from app.orchestration.interactions import InteractionStore, default_ttl_seconds, InteractionType
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_DEFAULT_AUTONOMY = AutonomyLevel.ASSISTED.value

#: What a tool that modifies a Job reports when the Job changed underneath
#: it (JobStore.save refused the stale write): nothing was changed, and the
#: user can safely look at the task again and retry.
_CHANGED_MESSAGE = (
    "Task '{job_id}' changed while this was being applied, so nothing was "
    "changed. Check its current status and try again."
)

#: Upper bound on the serialized size of a Job's request-context handoff.
#: The Context Engine already budget-limits what it selects; this is a
#: second, independent cap so an oversized value can never be persisted
#: into a Job and later sent to the planner.
_MAX_REQUEST_CONTEXT_CHARS = 8000


def _validate_request_context(value: Any) -> str | None:
    """Return why `value` isn't an acceptable request-context handoff, or
    None. Strict shape: a dict of non-empty string label -> list of
    strings, within `_MAX_REQUEST_CONTEXT_CHARS` once serialized."""
    if not isinstance(value, dict):
        return "request_context must be an object."
    for label, items in value.items():
        if not isinstance(label, str) or not label:
            return "request_context labels must be non-empty strings."
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            return f"request_context['{label}'] must be a list of strings."
    if len(json.dumps(value)) > _MAX_REQUEST_CONTEXT_CHARS:
        return f"request_context exceeds {_MAX_REQUEST_CONTEXT_CHARS} characters."
    return None


#: The structured objective Handle This (app/handle_this/) hands a Job. A
#: strict, closed schema: the user's words, an interpreted goal, the family of
#: work, constraints, success criteria and assumptions - all DATA. It cannot
#: name a tool, a permission, an autonomy level or a step, and it can never
#: name the computer-control family: computer workflows are deliberately not
#: something a Job is handed (see tests/test_computer_m15_isolation.py). The
#: literal sets below mirror app/handle_this/base.py (a test pins them
#: together); this module stays free of any dependency on it.
_MAX_OBJECTIVE_CHARS = 8000
_OBJECTIVE_KEYS = frozenset(
    {
        "original_request",
        "interpreted_goal",
        "goal_family",
        "required_families",
        "constraints",
        "success_criteria",
        "assumptions",
    }
)
_OBJECTIVE_FAMILIES = frozenset({"research", "analysis", "organization"})
_OBJECTIVE_GOAL_FAMILIES = _OBJECTIVE_FAMILIES | {"mixed"}
_OBJECTIVE_CONSTRAINT_TOPICS = frozenset({"budget", "quantity", "timeframe", "use_case", "preference"})
_OBJECTIVE_ASSUMPTION_TOPICS = _OBJECTIVE_CONSTRAINT_TOPICS | {"subject", "when", "intent"}
_OBJECTIVE_SOURCES = frozenset({"request", "context"})
_MAX_LIST = 8


def _is_clean_line(text: Any, limit: int) -> bool:
    return (
        isinstance(text, str)
        and bool(text.strip())
        and len(text) <= limit
        and not any(unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") for ch in text)
    )


def _validate_objective(value: Any) -> str | None:
    """Why `value` isn't an acceptable objective handoff, or None. The
    original request is the user's own text and is kept verbatim, so it is
    only bounded; every other text is a short clean line."""
    if not isinstance(value, dict) or set(value) != _OBJECTIVE_KEYS:
        return "objective must be an object with exactly the expected fields."
    original = value["original_request"]
    if not isinstance(original, str) or not original.strip() or len(original) > 4000:
        return "objective.original_request must be the request text (at most 4000 characters)."
    if not _is_clean_line(value["interpreted_goal"], 400):
        return "objective.interpreted_goal must be one short clean line."
    family = value["goal_family"]
    if family not in _OBJECTIVE_GOAL_FAMILIES:
        return "objective.goal_family is not a family a task can be started for."
    required = value["required_families"]
    if (
        not isinstance(required, list)
        or not all(isinstance(f, str) and f in _OBJECTIVE_FAMILIES for f in required)
        or len(set(required)) != len(required)
    ):
        return "objective.required_families must be distinct, supported families."
    if (family == "mixed" and len(required) < 2) or (family != "mixed" and required != [family]):
        return "objective.goal_family and required_families do not agree."
    for name, keys, topics in (
        ("constraints", {"topic", "text", "source"}, _OBJECTIVE_CONSTRAINT_TOPICS),
        ("assumptions", {"topic", "text"}, _OBJECTIVE_ASSUMPTION_TOPICS),
    ):
        items = value[name]
        if not isinstance(items, list) or len(items) > _MAX_LIST:
            return f"objective.{name} must be a short list."
        for item in items:
            if not isinstance(item, dict) or set(item) != keys:
                return f"objective.{name} entries must have exactly {sorted(keys)}."
            if item["topic"] not in topics or not _is_clean_line(item["text"], 200):
                return f"objective.{name} has an invalid entry."
            if "source" in keys and item["source"] not in _OBJECTIVE_SOURCES:
                return f"objective.{name} has an invalid source."
    criteria = value["success_criteria"]
    if (
        not isinstance(criteria, list)
        or len(criteria) > _MAX_LIST
        or not all(_is_clean_line(c, 200) for c in criteria)
    ):
        return "objective.success_criteria must be a few short clean lines."
    if len(json.dumps(value)) > _MAX_OBJECTIVE_CHARS:
        return f"objective exceeds {_MAX_OBJECTIVE_CHARS} characters."
    return None


#: A question the created Job must WAIT on before it is planned. Closed: a
#: topic from a fixed list and one short clean line (Handle This's own fixed
#: question templates). The Job is created already waiting, together with its
#: INPUT interaction, in one transaction.
_CLARIFICATION_KEYS = frozenset({"question", "topic"})
_CLARIFICATION_TOPICS = frozenset({"subject", "intent", "budget", "use_case", "when"})


def _validate_clarification(value: Any) -> str | None:
    if not isinstance(value, dict) or set(value) != _CLARIFICATION_KEYS:
        return "clarification must be an object with exactly 'question' and 'topic'."
    if value["topic"] not in _CLARIFICATION_TOPICS:
        return "clarification.topic is not a topic that can be asked about."
    if not _is_clean_line(value["question"], 200):
        return "clarification.question must be one short clean line."
    return None


def _objective_fingerprint(conversation_id: str | None, original_request: str) -> str:
    """Identifies "this request, in this conversation" so the same request
    processed twice (a client retry, a repeated message) starts one Job."""
    return hashlib.sha256(f"{conversation_id or ''}\n{original_request}".encode("utf-8")).hexdigest()[:32]


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
        "expectation": step.expectation,
        "evaluation": step.evaluation,
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
    plannable = False  # Jarvis-control tool: never offered to / accepted from a plan

    def __init__(
        self,
        store: JobStore,
        context_provider=None,
        interactions: InteractionStore | None = None,
    ):
        self.store = store
        # Needed only to create a Job that WAITS on a question (see the
        # `clarification` parameter); without it such a request is refused.
        self.interactions = interactions
        # Serialises "is this request already running?" with "create it", so
        # the same request arriving twice at once still starts one Job.
        self._create_lock = threading.Lock()
        # Optional zero-arg callable returning a JSON-safe dict, used to
        # snapshot "relevant context" (e.g. known user facts) at creation
        # time - see app/tools/defaults.py, which wires this to
        # `memory.get_facts`. Defaults to an empty context rather than
        # requiring every caller/test to supply one.
        self.context_provider = context_provider or (lambda: {})

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="goal", type="string", description="What to accomplish", free_text=True
            ),
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
            ToolParameter(
                name="request_context",
                type="object",
                description=(
                    "Relevant World Model context for this goal (label -> list "
                    "of strings), supplied by Jarvis itself when the goal is "
                    "received - never by a plan"
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="clarification",
                type="object",
                description=(
                    "A question the user must answer before this task is planned "
                    "({question, topic}), supplied by Jarvis itself when its "
                    "request-understanding step needs something material. The task "
                    "is created already WAITING on that question (never planned "
                    "first), and resumes as the same task when it is answered. "
                    "Requires an objective; takes the default settings."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="objective",
                type="object",
                description=(
                    "The structured objective Jarvis's request-understanding step "
                    "produced for this goal (the user's words, interpreted goal, "
                    "family of work, constraints, success criteria, assumptions), "
                    "supplied by Jarvis itself - never by a plan. Data for the "
                    "planner; it grants nothing. Also makes creation idempotent: "
                    "the same request in the same conversation, while still "
                    "running, is not started twice."
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
        request_context: dict[str, list[str]] | None = None,
        objective: dict[str, Any] | None = None,
        clarification: dict[str, Any] | None = None,
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

        context = dict(self.context_provider() or {})
        if request_context:
            problem = _validate_request_context(request_context)
            if problem:
                return ToolResult.error(problem)
            context[REQUEST_CONTEXT_KEY] = request_context

        if clarification is not None:
            return self._create_waiting(
                goal, context, conversation_id, objective, clarification,
                parsed_priority, parsed_autonomy, autonomous_permissions,
            )

        if objective is None:
            job = self._create(
                goal, parsed_priority, parsed_autonomy, context, conversation_id,
                parsed_autonomous_permissions,
            )
        else:
            problem = _validate_objective(objective)
            if problem:
                return ToolResult.error(problem)
            fingerprint = _objective_fingerprint(conversation_id, objective["original_request"])
            context[OBJECTIVE_KEY] = {**objective, "fingerprint": fingerprint}
            with self._create_lock:
                running = next(
                    (
                        j
                        for j in self.store.list()
                        if j.status not in TERMINAL_JOB_STATUSES
                        and isinstance(j.context.get(OBJECTIVE_KEY), dict)
                        and j.context[OBJECTIVE_KEY].get("fingerprint") == fingerprint
                    ),
                    None,
                )
                if running is not None:
                    return ToolResult.success(
                        data={**_job_to_summary_dict(running), "already_started": True},
                        message=f"Task '{running.id}' is already in progress for that request.",
                    )
                job = self._create(
                    goal, parsed_priority, parsed_autonomy, context, conversation_id,
                    parsed_autonomous_permissions,
                )
        return ToolResult.success(
            data=_job_to_summary_dict(job),
            message=(
                f"Started task '{job.id}' for goal: {goal!r}. Jarvis will plan "
                "and work on it in the background."
            ),
        )


    def _create_waiting(
        self,
        goal,
        context,
        conversation_id,
        objective,
        clarification,
        priority,
        autonomy,
        autonomous_permissions,
    ) -> ToolResult:
        """Create a Job that starts out WAITING on one INPUT question - the
        Job row and the interaction row are written in one transaction, so
        the engine can never plan it before the question is answered, and it
        can never exist without the question it waits on. Nothing is created
        at all if anything here is refused."""
        if objective is None:
            return ToolResult.error("A task that waits for an answer needs an objective.")
        problem = _validate_objective(objective) or _validate_clarification(clarification)
        if problem:
            return ToolResult.error(problem)
        if (
            priority.value != "medium"
            or autonomy.value != _DEFAULT_AUTONOMY
            or autonomous_permissions is not None
        ):
            return ToolResult.error("A task that waits for an answer takes the default settings.")
        if self.interactions is None:
            return ToolResult.error("This assistant cannot ask for input, so that task was not started.")

        fingerprint = _objective_fingerprint(conversation_id, objective["original_request"])
        context[OBJECTIVE_KEY] = {**objective, "fingerprint": fingerprint}
        with self._create_lock:
            running = next(
                (
                    j
                    for j in self.store.list()
                    if j.status not in TERMINAL_JOB_STATUSES
                    and isinstance(j.context.get(OBJECTIVE_KEY), dict)
                    and j.context[OBJECTIVE_KEY].get("fingerprint") == fingerprint
                ),
                None,
            )
            if running is not None:
                data = {**_job_to_summary_dict(running), "already_started": True}
                # The same request sent again while it still waits on its
                # question: hand back that question (not a second one), so the
                # caller can put it to the user again.
                waiting = [
                    i
                    for i in self.interactions.list_pending_for_job(running.id)
                    if i.type is InteractionType.INPUT
                ]
                if waiting:
                    data["pending_interaction"] = {"id": waiting[-1].id, "code": waiting[-1].code}
                return ToolResult.success(
                    data=data,
                    message=f"Task '{running.id}' is already in progress for that request.",
                )
            job, interaction = self.interactions.create_job_awaiting_input(
                goal=goal,
                context=context,
                conversation_id=conversation_id,
                prompt=clarification["question"],
                detail={"origin": "handle_this", "topic": clarification["topic"]},
                ttl_seconds=default_ttl_seconds(InteractionType.INPUT),
            )
        return ToolResult.success(
            data={
                **_job_to_summary_dict(job),
                "pending_interaction": {"id": interaction.id, "code": interaction.code},
            },
            message=f"Started task '{job.id}' for goal: {goal!r}; it is waiting for an answer.",
        )

    def _create(self, goal, priority, autonomy, context, conversation_id, autonomous_permissions):
        return self.store.create(
            goal=goal,
            priority=priority,
            autonomy=autonomy,
            context=context,
            conversation_id=conversation_id,
            autonomous_permissions=autonomous_permissions,
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
    plannable = False  # Jarvis-control tool: never offered to / accepted from a plan

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
                    step.error = DECLINED_STEP_ERROR
                    step.status = StepStatus.FAILED
            job.pending_step_ids = []
            job.confirmation_message = None
            job.status = JobStatus.FAILED
            job.error = f"Task cancelled: user declined confirmation ({declined})."
            message = f"Task '{job_id}' declined - task marked as failed."

        job.updated_at = now
        try:
            self.store.save(job)
        except StaleJobError:
            return ToolResult.error(_CHANGED_MESSAGE.format(job_id=job_id))
        return ToolResult.success(data=_job_to_summary_dict(job), message=message)


def _transition(
    store: JobStore,
    job_id: str,
    *,
    from_ok,
    to_status: JobStatus,
    verb: str,
    interactions: InteractionStore | None = None,
) -> ToolResult:
    job = store.get(job_id)
    if job is None:
        return ToolResult.error(f"No such task: '{job_id}'")
    if not from_ok(job.status):
        return ToolResult.error(
            f"Cannot {verb} task '{job_id}' from status '{job.status.value}'."
        )
    read_version = job.version
    job.status = to_status
    job.updated_at = _now()
    try:
        if interactions is not None and to_status is JobStatus.CANCELLED:
            # Cancelling also withdraws the Job's pending interactions - in
            # the SAME transaction, so a cancelled Job can never be left with
            # a live question/confirmation that could later resume it.
            stamp = job.updated_at

            def mutate(fresh: Job) -> None:
                fresh.status = to_status
                fresh.updated_at = stamp

            interactions.cancel_job_with_interactions(
                job_id,
                expected_job_version=read_version,
                mutate_job=mutate,
                reason="job cancelled",
                via="job-cancel",
            )
            job.version = read_version + 1
        else:
            store.save(job)
    except StaleJobError:
        return ToolResult.error(_CHANGED_MESSAGE.format(job_id=job_id))
    return ToolResult.success(
        data=_job_to_summary_dict(job), message=f"Task '{job_id}' {verb}d."
    )


class PauseJobTool(Tool):
    """Pauses an active Job - JobEngine skips a PAUSED Job entirely until
    resumed. SAFE_ACTION: reversible, purely local."""

    name = "pause_job"
    description = "Pause an in-progress task so it stops advancing until resumed."
    permission = PermissionLevel.SAFE_ACTION
    plannable = False  # Jarvis-control tool: never offered to / accepted from a plan

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
    plannable = False  # Jarvis-control tool: never offered to / accepted from a plan

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
    plannable = False  # Jarvis-control tool: never offered to / accepted from a plan

    def __init__(self, store: JobStore, interactions: InteractionStore | None = None):
        self.store = store
        # When given, cancelling also withdraws the Job's pending
        # interactions atomically (see `_transition`).
        self.interactions = interactions

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
            interactions=self.interactions,
        )
