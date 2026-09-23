"""
Tests for the orchestration Tools (app/orchestration/tools.py):
create/list/get/confirm/pause/resume/cancel, each adapting JobStore (and
JobEngine's own confirmation protocol) into the Tool interface - the same
"reach a Job through the same ToolRegistry every other tool uses"
convention every other *_tools.py test file already proves for its own
domain.
"""

from app.orchestration.base import (
    AutonomyLevel,
    JobPriority,
    JobStatus,
    Plan,
    PlanStep,
    StepStatus,
)
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import (
    CancelJobTool,
    ConfirmJobTool,
    CreateJobTool,
    GetJobTool,
    ListJobsTool,
    PauseJobTool,
    ResumeJobTool,
)
from app.tools.base import PermissionLevel, ToolResultStatus


def make_store() -> SQLiteJobStore:
    return SQLiteJobStore(":memory:")


# -- CreateJobTool -----------------------------------------------------------


def test_create_job_tool_is_safe_action():
    assert CreateJobTool(store=make_store()).permission == PermissionLevel.SAFE_ACTION


def test_create_job_tool_creates_a_pending_job():
    store = make_store()
    tool = CreateJobTool(store=store)

    result = tool.execute(goal="prepare the report")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["status"] == "pending"
    jobs = store.list()
    assert len(jobs) == 1
    assert jobs[0].goal == "prepare the report"


def test_create_job_tool_uses_context_provider():
    store = make_store()
    tool = CreateJobTool(store=store, context_provider=lambda: {"user_name": "Bassim"})

    tool.execute(goal="goal")

    assert store.list()[0].context == {"user_name": "Bassim"}


def test_create_job_tool_defaults_to_medium_priority_and_assisted_autonomy():
    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal")

    job = store.list()[0]
    assert job.priority == JobPriority.MEDIUM
    assert job.autonomy == AutonomyLevel.ASSISTED


def test_create_job_tool_accepts_explicit_priority_and_autonomy():
    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal", priority="high", autonomy="autonomous")

    job = store.list()[0]
    assert job.priority == JobPriority.HIGH
    assert job.autonomy == AutonomyLevel.AUTONOMOUS


def test_create_job_tool_rejects_unknown_priority():
    tool = CreateJobTool(store=make_store())
    result = tool.execute(goal="goal", priority="urgent")
    assert result.status == ToolResultStatus.ERROR


def test_create_job_tool_rejects_unknown_autonomy():
    tool = CreateJobTool(store=make_store())
    result = tool.execute(goal="goal", autonomy="godmode")
    assert result.status == ToolResultStatus.ERROR


# -- ListJobsTool / GetJobTool -----------------------------------------------------------


def test_list_jobs_tool_is_read():
    assert ListJobsTool(store=make_store()).permission == PermissionLevel.READ


def test_list_jobs_tool_returns_all_jobs():
    store = make_store()
    store.create("a")
    store.create("b")
    tool = ListJobsTool(store=store)

    result = tool.execute()

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == 2


def test_list_jobs_tool_filters_by_status():
    store = make_store()
    store.create("a")
    running = store.create("b")
    running.status = JobStatus.RUNNING
    store.save(running)
    tool = ListJobsTool(store=store)

    result = tool.execute(status="running")

    assert [j["id"] for j in result.data] == [running.id]


def test_list_jobs_tool_rejects_unknown_status():
    tool = ListJobsTool(store=make_store())
    result = tool.execute(status="not-a-status")
    assert result.status == ToolResultStatus.ERROR


def test_get_job_tool_returns_full_detail():
    store = make_store()
    job = store.create("goal")
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="echo", status=StepStatus.SUCCEEDED)])
    store.save(job)
    tool = GetJobTool(store=store)

    result = tool.execute(job_id=job.id)

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["plan"][0]["id"] == "s1"


def test_get_job_tool_errors_for_unknown_id():
    tool = GetJobTool(store=make_store())
    result = tool.execute(job_id="nope")
    assert result.status == ToolResultStatus.ERROR


# -- ConfirmJobTool -----------------------------------------------------------


def _make_awaiting_confirmation_job(store):
    job = store.create("goal", autonomy=AutonomyLevel.MANUAL)
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="confirm_tool", permission="confirm")])
    job.status = JobStatus.AWAITING_CONFIRMATION
    job.pending_step_ids = ["s1"]
    job.confirmation_message = "Run confirm_tool?"
    store.save(job)
    return job


def test_confirm_job_tool_is_safe_action():
    assert ConfirmJobTool(store=make_store()).permission == PermissionLevel.SAFE_ACTION


def test_confirm_job_tool_approve_marks_step_confirmed_and_resumes():
    store = make_store()
    job = _make_awaiting_confirmation_job(store)
    tool = ConfirmJobTool(store=store)

    result = tool.execute(job_id=job.id, approve=True)

    assert result.status == ToolResultStatus.SUCCESS
    fetched = store.get(job.id)
    assert fetched.status == JobStatus.RUNNING
    assert fetched.pending_step_ids == []
    assert fetched.plan.get("s1").confirmed is True


def test_confirm_job_tool_decline_fails_the_job():
    store = make_store()
    job = _make_awaiting_confirmation_job(store)
    tool = ConfirmJobTool(store=store)

    result = tool.execute(job_id=job.id, approve=False)

    assert result.status == ToolResultStatus.SUCCESS  # the tool call itself succeeded
    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert "declined" in fetched.error.lower()
    assert fetched.plan.get("s1").status == StepStatus.FAILED


def test_confirm_job_tool_errors_if_not_awaiting_confirmation():
    store = make_store()
    job = store.create("goal")  # PENDING
    tool = ConfirmJobTool(store=store)

    result = tool.execute(job_id=job.id, approve=True)

    assert result.status == ToolResultStatus.ERROR


def test_confirm_job_tool_errors_for_unknown_id():
    tool = ConfirmJobTool(store=make_store())
    result = tool.execute(job_id="nope", approve=True)
    assert result.status == ToolResultStatus.ERROR


# -- Pause / Resume / Cancel -----------------------------------------------------------


def test_pause_resume_cancel_tools_are_safe_action():
    store = make_store()
    assert PauseJobTool(store=store).permission == PermissionLevel.SAFE_ACTION
    assert ResumeJobTool(store=store).permission == PermissionLevel.SAFE_ACTION
    assert CancelJobTool(store=store).permission == PermissionLevel.SAFE_ACTION


def test_pause_job_tool_pauses_a_running_job():
    store = make_store()
    job = store.create("goal")
    job.status = JobStatus.RUNNING
    store.save(job)

    result = PauseJobTool(store=store).execute(job_id=job.id)

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(job.id).status == JobStatus.PAUSED


def test_pause_job_tool_rejects_a_terminal_job():
    store = make_store()
    job = store.create("goal")
    job.status = JobStatus.COMPLETED
    store.save(job)

    result = PauseJobTool(store=store).execute(job_id=job.id)

    assert result.status == ToolResultStatus.ERROR
    assert store.get(job.id).status == JobStatus.COMPLETED


def test_resume_job_tool_resumes_a_planned_job_to_running():
    store = make_store()
    job = store.create("goal")
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="echo")])
    job.status = JobStatus.PAUSED
    store.save(job)

    result = ResumeJobTool(store=store).execute(job_id=job.id)

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(job.id).status == JobStatus.RUNNING


def test_resume_job_tool_resumes_an_unplanned_job_to_pending():
    store = make_store()
    job = store.create("goal")
    job.status = JobStatus.PAUSED
    store.save(job)

    ResumeJobTool(store=store).execute(job_id=job.id)

    assert store.get(job.id).status == JobStatus.PENDING


def test_resume_job_tool_rejects_a_non_paused_job():
    store = make_store()
    job = store.create("goal")  # PENDING, not PAUSED

    result = ResumeJobTool(store=store).execute(job_id=job.id)

    assert result.status == ToolResultStatus.ERROR


def test_cancel_job_tool_cancels_a_non_terminal_job():
    store = make_store()
    job = store.create("goal")

    result = CancelJobTool(store=store).execute(job_id=job.id)

    assert result.status == ToolResultStatus.SUCCESS
    assert store.get(job.id).status == JobStatus.CANCELLED


def test_cancel_job_tool_rejects_an_already_terminal_job():
    store = make_store()
    job = store.create("goal")
    job.status = JobStatus.CANCELLED
    store.save(job)

    result = CancelJobTool(store=store).execute(job_id=job.id)

    assert result.status == ToolResultStatus.ERROR


# -- job_id propagation / conversation_id / autonomous_permissions (item 2/4) --


def test_create_job_tool_returns_a_real_job_id_in_structured_data():
    """The id is a first-class field of the structured result, not
    something a caller has to parse out of the message string."""
    store = make_store()
    tool = CreateJobTool(store=store)

    result = tool.execute(goal="prepare the report")

    created = store.list()[0]
    assert result.data["id"] == created.id
    assert isinstance(result.data["id"], str) and result.data["id"]


def test_create_job_tool_links_the_job_to_its_conversation():
    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal", conversation_id="conv-123")

    assert store.list()[0].conversation_id == "conv-123"


def test_create_job_tool_conversation_id_defaults_to_none():
    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal")

    assert store.list()[0].conversation_id is None


def test_create_job_tool_accepts_comma_separated_autonomous_permissions():
    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal", autonomy="autonomous", autonomous_permissions="confirm")

    assert store.list()[0].autonomous_permissions == frozenset({"confirm"})


def test_create_job_tool_empty_autonomous_permissions_is_a_real_opt_out():
    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal", autonomy="autonomous", autonomous_permissions="")

    assert store.list()[0].autonomous_permissions == frozenset()


def test_create_job_tool_unset_autonomous_permissions_uses_the_default():
    from app.orchestration.base import DEFAULT_AUTONOMOUS_PERMISSIONS

    store = make_store()
    tool = CreateJobTool(store=store)

    tool.execute(goal="goal", autonomy="autonomous")

    assert store.list()[0].autonomous_permissions == DEFAULT_AUTONOMOUS_PERMISSIONS


def test_create_job_tool_rejects_dangerous_in_autonomous_permissions():
    store = make_store()
    tool = CreateJobTool(store=store)

    result = tool.execute(goal="goal", autonomous_permissions="dangerous")

    assert result.status == ToolResultStatus.ERROR
    assert store.list() == []  # rejected before creating anything


def test_list_jobs_summary_includes_conversation_id_and_outcome_fields():
    store = make_store()
    job = store.create("goal", conversation_id="conv-1")
    job.status = JobStatus.COMPLETED
    job.final_result = "All done."
    store.save(job)

    result = ListJobsTool(store=store).execute()

    entry = result.data[0]
    assert entry["conversation_id"] == "conv-1"
    assert entry["final_result"] == "All done."
