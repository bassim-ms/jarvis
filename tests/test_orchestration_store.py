"""
Tests for SQLiteJobStore (app/orchestration/sqlite_store.py): create/get/
list/save round-trip a Job's full state, including its Plan/PlanSteps and
event log, through real SQLite (never mocked) - the same "prove
persistence with a real (temp/in-memory) database" convention every
other store's own tests use.
"""

from app.orchestration.base import (
    AutonomyLevel,
    Job,
    JobEvent,
    JobPriority,
    JobStatus,
    Plan,
    PlanStep,
    RetryPolicy,
    StepStatus,
)
from app.orchestration.sqlite_store import SQLiteJobStore


def make_store() -> SQLiteJobStore:
    return SQLiteJobStore(":memory:")


def test_create_persists_a_pending_job_with_no_plan():
    store = make_store()

    job = store.create("prepare the report")

    assert job.status == JobStatus.PENDING
    assert job.priority == JobPriority.MEDIUM
    assert job.autonomy == AutonomyLevel.ASSISTED
    assert job.plan is None
    assert job.goal == "prepare the report"
    assert job.id


def test_create_accepts_priority_autonomy_and_context():
    store = make_store()

    job = store.create(
        "goal",
        priority=JobPriority.HIGH,
        autonomy=AutonomyLevel.AUTONOMOUS,
        context={"fact": "value"},
    )

    assert job.priority == JobPriority.HIGH
    assert job.autonomy == AutonomyLevel.AUTONOMOUS
    assert job.context == {"fact": "value"}


def test_get_returns_none_for_unknown_id():
    store = make_store()
    assert store.get("does-not-exist") is None


def test_get_round_trips_a_freshly_created_job():
    store = make_store()
    job = store.create("goal")

    fetched = store.get(job.id)

    assert fetched.id == job.id
    assert fetched.goal == "goal"
    assert fetched.status == JobStatus.PENDING


def test_list_returns_most_recently_created_first():
    store = make_store()
    first = store.create("first")
    second = store.create("second")

    jobs = store.list()

    assert [j.id for j in jobs] == [second.id, first.id]


def test_list_filters_by_status():
    store = make_store()
    pending = store.create("pending goal")
    running = store.create("running goal")
    running.status = JobStatus.RUNNING
    store.save(running)

    result = store.list(status=JobStatus.RUNNING)

    assert [j.id for j in result] == [running.id]
    assert pending.id not in [j.id for j in result]


def test_save_persists_a_full_plan_with_steps_and_results():
    store = make_store()
    job = store.create("goal")

    job.plan = Plan(
        steps=[
            PlanStep(
                id="s1",
                tool_name="echo",
                arguments={"text": "hi"},
                depends_on=[],
                permission="read",
                retry_policy=RetryPolicy(max_attempts=3, delay_seconds=2.0),
                status=StepStatus.SUCCEEDED,
                attempts=1,
                confirmed=True,
                result={"status": "success", "message": "ok", "data": "hi"},
                error=None,
            ),
            PlanStep(id="s2", tool_name="echo", depends_on=["s1"]),
        ]
    )
    job.status = JobStatus.RUNNING
    job.total_step_count = 2
    job.replan_count = 1
    job.pending_step_ids = ["s2"]
    job.confirmation_message = "please confirm"
    job.updated_at = "2030-01-01T00:00:00+00:00"
    store.save(job)

    fetched = store.get(job.id)

    assert fetched.status == JobStatus.RUNNING
    assert fetched.total_step_count == 2
    assert fetched.replan_count == 1
    assert fetched.pending_step_ids == ["s2"]
    assert fetched.confirmation_message == "please confirm"
    assert fetched.updated_at == "2030-01-01T00:00:00+00:00"

    assert len(fetched.plan.steps) == 2
    step1 = fetched.plan.get("s1")
    assert step1.tool_name == "echo"
    assert step1.arguments == {"text": "hi"}
    assert step1.permission == "read"
    assert step1.retry_policy == RetryPolicy(max_attempts=3, delay_seconds=2.0)
    assert step1.status == StepStatus.SUCCEEDED
    assert step1.attempts == 1
    assert step1.confirmed is True
    assert step1.result == {"status": "success", "message": "ok", "data": "hi"}

    step2 = fetched.plan.get("s2")
    assert step2.depends_on == ["s1"]
    assert step2.status == StepStatus.PENDING


def test_save_persists_final_result_and_error():
    store = make_store()
    completed = store.create("goal a")
    completed.status = JobStatus.COMPLETED
    completed.final_result = "All done."
    store.save(completed)

    failed = store.create("goal b")
    failed.status = JobStatus.FAILED
    failed.error = "step x failed"
    store.save(failed)

    assert store.get(completed.id).final_result == "All done."
    assert store.get(completed.id).error is None
    assert store.get(failed.id).error == "step x failed"


def test_save_persists_events():
    store = make_store()
    job = store.create("goal")
    job.events.append(JobEvent(timestamp="t1", kind="planned", detail="2 steps"))
    job.events.append(JobEvent(timestamp="t2", kind="step_succeeded", detail="ok", step_id="s1"))
    store.save(job)

    fetched = store.get(job.id)

    assert len(fetched.events) == 2
    assert fetched.events[0] == JobEvent(timestamp="t1", kind="planned", detail="2 steps")
    assert fetched.events[1].step_id == "s1"


def test_save_does_not_change_immutable_fields():
    store = make_store()
    job = store.create("original goal")
    original_created_at = job.created_at

    job.status = JobStatus.RUNNING
    store.save(job)

    fetched = store.get(job.id)
    assert fetched.goal == "original goal"
    assert fetched.created_at == original_created_at


def test_store_survives_a_fresh_instance_against_the_same_file(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    store1 = SQLiteJobStore(db_path)
    job = store1.create("persisted goal")
    job.status = JobStatus.RUNNING
    store1.save(job)

    store2 = SQLiteJobStore(db_path)
    fetched = store2.get(job.id)

    assert fetched is not None
    assert fetched.goal == "persisted goal"
    assert fetched.status == JobStatus.RUNNING
