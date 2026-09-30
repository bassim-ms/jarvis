"""
Optimistic versioning of Job saves.

`JobStore.save` is a compare-and-set: a write is applied only if the row
still carries the version the caller read; otherwise it raises
`StaleJobError` and writes NOTHING. This is what stops a stale writer - an
engine tick working from an old copy, a chat-side cancel/confirm racing the
engine - from silently overwriting newer state.

Before this, `save` overwrote the whole row unconditionally: a user's cancel
issued while the engine was mid-tick was silently undone by the engine's next
save, and the cancelled Job ran to COMPLETED.
"""

import sqlite3
import threading
import uuid

import pytest

from app.orchestration.base import (
    AutonomyLevel,
    Job,
    JobPriority,
    JobStatus,
    StaleJobError,
    StepStatus,
)
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner, PlanningError
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import (
    CancelJobTool,
    ConfirmJobTool,
    PauseJobTool,
    ResumeJobTool,
)
from app.reminders.notifier import Notifier
from app.tools.base import PermissionLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry


# --- 1. the store: versioned save ---------------------------------------------------------------------


def test_a_new_job_starts_at_version_zero_and_reads_report_it():
    store = SQLiteJobStore(":memory:")

    job = store.create("goal")

    assert job.version == 0
    assert store.get(job.id).version == 0
    assert [j.version for j in store.list()] == [0]


def test_a_versioned_save_succeeds_and_advances_the_version_in_the_store_and_the_object():
    store = SQLiteJobStore(":memory:")
    job = store.create("goal")

    job.status = JobStatus.RUNNING
    job.error = "first"
    store.save(job)

    assert job.version == 1
    persisted = store.get(job.id)
    assert persisted.version == 1
    assert (persisted.status, persisted.error) == (JobStatus.RUNNING, "first")


def test_a_holder_that_keeps_saving_the_same_object_keeps_working():
    store = SQLiteJobStore(":memory:")
    job = store.create("goal")

    for expected in (1, 2, 3, 4):
        job.updated_at = f"t{expected}"
        store.save(job)
        assert job.version == expected

    assert store.get(job.id).version == 4
    assert store.get(job.id).updated_at == "t4"


def test_a_stale_save_is_rejected():
    store = SQLiteJobStore(":memory:")
    first = store.create("goal")
    stale = store.get(first.id)  # a second reader, same version

    first.status = JobStatus.RUNNING
    store.save(first)  # the first writer wins

    stale.status = JobStatus.FAILED
    stale.error = "written from a stale copy"
    with pytest.raises(StaleJobError, match="changed since it was read"):
        store.save(stale)

    assert stale.version == 0  # a refused write leaves the caller's version alone


def test_a_stale_writer_never_overwrites_newer_state():
    store = SQLiteJobStore(":memory:")
    original = store.create("goal", context={"k": "v"})
    stale = store.get(original.id)

    # Newer state: the job was cancelled.
    original.status = JobStatus.CANCELLED
    original.error = "cancelled by the user"
    store.save(original)

    # A stale writer tries to move it on, changing many fields at once.
    stale.status = JobStatus.RUNNING
    stale.final_result = "stale result"
    stale.context = {"k": "stale"}
    stale.replan_count = 9
    stale.total_step_count = 99
    stale.confirmation_message = "stale"
    with pytest.raises(StaleJobError):
        store.save(stale)

    current = store.get(original.id)
    assert current.status == JobStatus.CANCELLED  # not resurrected
    assert current.error == "cancelled by the user"
    assert current.final_result is None
    assert current.context == {"k": "v"}
    assert (current.replan_count, current.total_step_count) == (0, 0)  # no partial write
    assert current.confirmation_message is None
    assert current.version == 1


def test_a_refused_save_leaves_the_row_byte_for_byte_unchanged(tmp_path):
    db = str(tmp_path / "jobs.db")
    store = SQLiteJobStore(db)
    job = store.create("goal")
    stale = store.get(job.id)
    job.status = JobStatus.PAUSED
    store.save(job)
    raw = lambda: sqlite3.connect(db).execute("SELECT * FROM jobs").fetchall()  # noqa: E731
    before = raw()

    stale.status = JobStatus.FAILED
    with pytest.raises(StaleJobError):
        store.save(stale)

    assert raw() == before


def test_after_a_stale_rejection_a_fresh_read_can_be_saved():
    store = SQLiteJobStore(":memory:")
    job = store.create("goal")
    stale = store.get(job.id)
    job.status = JobStatus.RUNNING
    store.save(job)
    with pytest.raises(StaleJobError):
        store.save(stale)

    fresh = store.get(job.id)
    fresh.status = JobStatus.COMPLETED
    store.save(fresh)

    assert store.get(job.id).status == JobStatus.COMPLETED
    assert store.get(job.id).version == 2


def test_repeated_stale_attempts_keep_failing_and_never_consume_a_version():
    store = SQLiteJobStore(":memory:")
    job = store.create("goal")
    stale = store.get(job.id)
    job.status = JobStatus.RUNNING
    store.save(job)

    for _ in range(3):
        with pytest.raises(StaleJobError):
            store.save(stale)

    assert store.get(job.id).version == 1


def test_saving_a_job_that_does_not_exist_is_refused_and_creates_nothing():
    store = SQLiteJobStore(":memory:")
    ghost = Job(
        id=str(uuid.uuid4()),
        goal="ghost",
        status=JobStatus.RUNNING,
        priority=JobPriority.MEDIUM,
        autonomy=AutonomyLevel.ASSISTED,
        created_at="t",
        updated_at="t",
    )

    with pytest.raises(StaleJobError, match="no longer exists"):
        store.save(ghost)

    assert store.list() == [] and store.get(ghost.id) is None


def test_versions_are_per_job():
    store = SQLiteJobStore(":memory:")
    a, b = store.create("a"), store.create("b")

    a.status = JobStatus.RUNNING
    store.save(a)
    store.save(a)

    assert store.get(a.id).version == 2
    assert store.get(b.id).version == 0
    b.status = JobStatus.PAUSED
    store.save(b)  # b's own copy is still current
    assert store.get(b.id).version == 1


def test_concurrent_writers_exactly_one_wins():
    """Eight threads all read the same version and try to save different
    states at once: the compare-and-set lets exactly one through."""
    store = SQLiteJobStore(":memory:")
    job = store.create("goal")
    copies = [store.get(job.id) for _ in range(8)]
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    lock = threading.Lock()

    def writer(index, copy):
        copy.error = f"writer-{index}"
        barrier.wait()
        try:
            store.save(copy)
            result = "won"
        except StaleJobError:
            result = "stale"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=writer, args=(i, c)) for i, c in enumerate(copies)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sorted(outcomes) == ["stale"] * 7 + ["won"]
    persisted = store.get(job.id)
    assert persisted.version == 1
    assert persisted.error in {f"writer-{i}" for i in range(8)}


def test_the_version_survives_a_new_store_instance(tmp_path):
    db = str(tmp_path / "jobs.db")
    first = SQLiteJobStore(db)
    job = first.create("goal")
    job.status = JobStatus.RUNNING
    first.save(job)
    first.save(job)
    del first

    reopened = SQLiteJobStore(db)
    read = reopened.get(job.id)

    assert read.version == 2
    read.status = JobStatus.PAUSED
    reopened.save(read)
    assert reopened.get(job.id).version == 3


def test_a_database_created_before_versioning_is_migrated_in_place(tmp_path):
    """An on-disk jobs table without the `version` column picks it up (at
    0) and its rows keep working."""
    db = str(tmp_path / "old.db")
    old = sqlite3.connect(db)
    old.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
            priority TEXT NOT NULL, autonomy TEXT NOT NULL, plan TEXT,
            context TEXT NOT NULL, replan_count INTEGER NOT NULL,
            total_step_count INTEGER NOT NULL, pending_step_ids TEXT NOT NULL,
            confirmation_message TEXT, final_result TEXT, error TEXT,
            events TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            conversation_id TEXT, autonomous_permissions TEXT NOT NULL DEFAULT '[]'
        );
        INSERT INTO jobs VALUES ('old-1', 'an old goal', 'running', 'medium', 'assisted', NULL,
            '{"k": "v"}', 0, 0, '[]', NULL, NULL, NULL, '[]', 't0', 't0', 'conv-1', '["confirm"]');
        """
    )
    old.commit()
    old.close()

    store = SQLiteJobStore(db)
    job = store.get("old-1")

    assert job.version == 0
    assert (job.goal, job.context, job.conversation_id) == ("an old goal", {"k": "v"}, "conv-1")
    job.status = JobStatus.PAUSED
    store.save(job)
    assert store.get("old-1").version == 1
    stale = store.get("old-1")
    stale.version = 0
    with pytest.raises(StaleJobError):
        store.save(stale)


# --- 2. the engine ------------------------------------------------------------------------------------------


def CountingTool(name, on_execute=None):  # noqa: N802
    """A tool instance that counts its executions (`.calls`) and can run a
    callback while executing (to simulate another writer)."""

    def execute(self) -> ToolResult:
        self.calls += 1
        if on_execute:
            on_execute()
        return ToolResult.success(message=f"{name} done")

    cls = type(
        f"CountingTool_{name}",
        (Tool,),
        {
            "name": name,
            "description": "counts executions",
            "permission": PermissionLevel.SAFE_ACTION,
            "execute": execute,
        },
    )
    tool = cls()
    tool.calls = 0
    return tool


class OnePlan(Planner):
    def __init__(self, *steps, on_plan=None):
        self.steps = list(steps)
        self.on_plan = on_plan

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        if self.on_plan:
            self.on_plan()
        if replan_context is not None:
            raise PlanningError("no replan")
        return self.steps


def step(step_id, tool, depends_on=()):
    return {"id": step_id, "tool": tool, "arguments": {}, "depends_on": list(depends_on)}


def make_engine(store, tools, planner, **overrides):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    defaults = dict(
        max_steps=10,
        max_total_steps=20,
        max_retries_per_step=1,
        max_replans=0,
        retry_delay_seconds=0,
        poll_interval_seconds=1000,
    )
    defaults.update(overrides)
    return JobEngine(store=store, tools=registry, planner=planner, **defaults)


class RecordingStore(SQLiteJobStore):
    """Records the version each successful save produced."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_versions: list[int] = []

    def save(self, job):
        super().save(job)
        self.saved_versions.append(job.version)


def test_normal_execution_is_unchanged_and_versions_advance_without_gaps():
    store = RecordingStore(":memory:")
    a, b, c = CountingTool("a"), CountingTool("b"), CountingTool("c")
    engine = make_engine(
        store, [a, b, c], OnePlan(step("s1", "a"), step("s2", "b", ["s1"]), step("s3", "c", ["s2"]))
    )
    job = store.create("goal")

    processed = engine.tick()

    finished = store.get(job.id)
    assert processed == 1
    assert finished.status == JobStatus.COMPLETED
    assert [a.calls, b.calls, c.calls] == [1, 1, 1]
    assert all(s.status == StepStatus.SUCCEEDED for s in finished.plan.steps)
    # every save was a clean +1 on the previous one: no stale write, no gap
    assert store.saved_versions == list(range(1, len(store.saved_versions) + 1))
    assert finished.version == store.saved_versions[-1]


def test_a_cancel_issued_mid_tick_now_wins_and_no_further_step_runs():
    """The bug this hardening fixes: cancelling while the engine is running
    step 1 used to be silently overwritten - step 2 ran and the Job ended
    COMPLETED."""
    store = SQLiteJobStore(":memory:")
    holder = {}
    step_two = CountingTool("two")
    step_one = CountingTool(
        "one", on_execute=lambda: CancelJobTool(store=store).execute(job_id=holder["id"])
    )
    engine = make_engine(store, [step_one, step_two], OnePlan(step("s1", "one"), step("s2", "two", ["s1"])))
    job = store.create("goal")
    holder["id"] = job.id

    processed = engine.tick()  # must not raise

    after = store.get(job.id)
    assert after.status == JobStatus.CANCELLED  # the newer state stands
    assert step_one.calls == 1 and step_two.calls == 0  # nothing more ran from a stale view
    assert processed == 0  # the dropped tick isn't counted as processed
    assert after.final_result is None
    assert after.plan.get("s2").status == StepStatus.PENDING


def test_a_pause_issued_mid_tick_stays_paused():
    store = SQLiteJobStore(":memory:")
    holder = {}
    step_two = CountingTool("two")
    step_one = CountingTool(
        "one", on_execute=lambda: PauseJobTool(store=store).execute(job_id=holder["id"])
    )
    engine = make_engine(store, [step_one, step_two], OnePlan(step("s1", "one"), step("s2", "two", ["s1"])))
    job = store.create("goal")
    holder["id"] = job.id

    engine.tick()

    assert store.get(job.id).status == JobStatus.PAUSED
    assert step_two.calls == 0
    # ...and a later tick leaves a paused job alone, as always
    engine.tick()
    assert store.get(job.id).status == JobStatus.PAUSED and step_two.calls == 0


def test_a_job_cancelled_while_it_is_being_planned_is_not_resurrected_or_run():
    store = SQLiteJobStore(":memory:")
    holder = {}
    action = CountingTool("one")
    planner = OnePlan(
        step("s1", "one"),
        on_plan=lambda: CancelJobTool(store=store).execute(job_id=holder["id"]),
    )
    engine = make_engine(store, [action], planner)
    job = store.create("goal")
    holder["id"] = job.id

    engine.tick()

    after = store.get(job.id)
    assert after.status == JobStatus.CANCELLED
    assert after.plan is None  # the stale planning result was not saved
    assert action.calls == 0


def test_no_completion_notification_is_sent_for_a_state_that_was_never_persisted():
    sent = []

    class Spy(Notifier):
        def notify(self, title, message):
            sent.append((title, message))

    store = SQLiteJobStore(":memory:")
    holder = {}
    last = CountingTool(
        "last", on_execute=lambda: CancelJobTool(store=store).execute(job_id=holder["id"])
    )
    engine = make_engine(
        store, [last], OnePlan(step("s1", "last")), notifier=Spy(), notification_dispatcher=lambda fn: fn()
    )
    job = store.create("goal")
    holder["id"] = job.id

    engine.tick()

    assert store.get(job.id).status == JobStatus.CANCELLED
    assert sent == []  # the engine's COMPLETED never reached the store, so nobody is told it did


def test_a_stale_abort_after_a_tool_ran_never_re_runs_that_tool_on_the_next_tick():
    """Recovery is safe: the tool ran, the engine's outcome write was
    refused as stale (an unrelated writer bumped the row meanwhile). The
    durable RUNNING marker written BEFORE the call means the next tick
    recovers the step as interrupted - it does not execute it again."""
    store = SQLiteJobStore(":memory:")
    holder = {}

    def unrelated_writer_touches_the_row():
        other = store.get(holder["id"])
        other.updated_at = "touched by another writer"
        store.save(other)

    action = CountingTool("one", on_execute=unrelated_writer_touches_the_row)
    engine = make_engine(store, [action], OnePlan(step("s1", "one")))
    job = store.create("goal")
    holder["id"] = job.id

    engine.tick()
    mid = store.get(job.id)
    assert action.calls == 1
    # The engine's own outcome write was refused, so the durable record of
    # that step is the RUNNING marker written before the call - or, because
    # a tick also scans RUNNING jobs after PENDING ones, its interrupted-step
    # recovery of exactly that marker. Never SUCCEEDED (that write was refused).
    assert mid.plan.get("s1").status in (StepStatus.RUNNING, StepStatus.FAILED)

    engine.tick()
    engine.tick()

    after = store.get(job.id)
    assert action.calls == 1  # NOT executed a second time, however many ticks follow
    assert after.plan.get("s1").status == StepStatus.FAILED
    assert "Interrupted" in after.plan.get("s1").error
    assert after.status == JobStatus.FAILED  # no replan available: fails closed, honestly


def test_the_engine_can_always_resume_from_what_is_really_persisted_after_a_stale_abort():
    store = SQLiteJobStore(":memory:")
    holder = {}
    calls = {"n": 0}

    class Toucher(Tool):
        name = "toucher"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            calls["n"] += 1
            if calls["n"] == 1:  # only the first execution races
                other = store.get(holder["id"])
                other.updated_at = "raced"
                store.save(other)
            return ToolResult.success(message="ok")

    engine = make_engine(store, [Toucher(), CountingTool("after")], OnePlan(step("s1", "toucher"), step("s2", "after", ["s1"])), max_replans=1)
    job = store.create("goal")
    holder["id"] = job.id

    engine.tick()  # aborted by the stale write
    engine.tick()  # recovers the interrupted step and continues from real state

    after = store.get(job.id)
    assert after.plan.get("s1").status == StepStatus.FAILED  # interrupted, never blindly re-run
    assert calls["n"] == 1


# --- 3. tools that modify a Job fail cleanly when it changed under them ------------------------------------


class RacyStore(SQLiteJobStore):
    """On the `race_on_get`-th get() after arming, another writer saves the
    same row - so the copy that get() returned is already stale when its
    caller saves it."""

    race_on_get = 0  # 0 = not armed
    _gets_since_armed = 0

    def arm(self, nth_get=1):
        self.race_on_get, self._gets_since_armed = nth_get, 0

    def get(self, job_id):
        job = super().get(job_id)
        if self.race_on_get:
            self._gets_since_armed += 1
            if self._gets_since_armed == self.race_on_get:
                self.race_on_get = 0
                other = super().get(job_id)
                other.updated_at = "raced by another writer"
                super().save(other)
        return job


@pytest.mark.parametrize(
    "make_tool,starting_status,saved_read",
    [
        (CancelJobTool, JobStatus.RUNNING, 1),
        (PauseJobTool, JobStatus.RUNNING, 1),
        (ResumeJobTool, JobStatus.PAUSED, 2),  # resume reads once to choose a target, then _transition re-reads
    ],
    ids=["cancel", "pause", "resume"],
)
def test_pause_resume_cancel_report_a_clean_error_and_change_nothing_when_stale(
    make_tool, starting_status, saved_read
):
    store = RacyStore(":memory:")
    job = store.create("goal")
    job.status = starting_status
    store.save(job)
    store.arm(saved_read)

    result = make_tool(store=store).execute(job_id=job.id)

    assert result.status.value == "error"
    assert "changed while this was being applied" in result.message
    assert store.get(job.id).status == starting_status  # untouched


def test_a_stale_confirm_does_not_approve_anything():
    store = RacyStore(":memory:")
    from app.orchestration.base import Plan, PlanStep

    job = store.create("goal")
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="x")])
    job.status = JobStatus.AWAITING_CONFIRMATION
    job.pending_step_ids = ["s1"]
    job.confirmation_message = "confirm?"
    store.save(job)
    store.arm(1)

    approved = ConfirmJobTool(store=store).execute(job_id=job.id, approve=True)
    declined_after = store.get(job.id)

    assert approved.status.value == "error"
    assert declined_after.status == JobStatus.AWAITING_CONFIRMATION
    assert declined_after.plan.get("s1").confirmed is False
    assert declined_after.pending_step_ids == ["s1"]

    # a retry against fresh state works
    retry = ConfirmJobTool(store=store).execute(job_id=job.id, approve=True)
    assert retry.status.value == "success"
    assert store.get(job.id).plan.get("s1").confirmed is True


def test_tools_still_work_normally_when_nothing_raced():
    store = SQLiteJobStore(":memory:")
    job = store.create("goal")
    job.status = JobStatus.RUNNING
    store.save(job)

    assert PauseJobTool(store=store).execute(job_id=job.id).status.value == "success"
    assert ResumeJobTool(store=store).execute(job_id=job.id).status.value == "success"
    assert CancelJobTool(store=store).execute(job_id=job.id).status.value == "success"
    assert store.get(job.id).status == JobStatus.CANCELLED
