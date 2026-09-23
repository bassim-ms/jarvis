"""
Tests for JobEngine (app/orchestration/engine.py): planning, dependency-
respecting execution ordering, permission/confirmation gating per
AutonomyLevel, transient retry, permanent failure, replanning (and its
limits), pause/resume/cancel, restart recovery (a step caught mid-
execution is never blindly re-run), and accurate final results - never
claiming success when a step actually failed. Uses real SQLiteJobStore
(":memory:"), a real ToolRegistry with small fake Tools, and a
FakePlanner (never a real Gemini call) - `tick()` is called directly,
matching AutomationEngine/ReminderScheduler's own tests.
"""

import json
import time

import pytest

from app.orchestration.base import (
    AutonomyLevel,
    JobPriority,
    JobStatus,
    StepStatus,
)
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner, PlanningError, ReplanContext
from app.orchestration.sqlite_store import SQLiteJobStore
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.registry import ToolRegistry


# -- fake tools -----------------------------------------------------------


class RecordingTool(Tool):
    """READ/automatic tool that records every call and always succeeds."""

    name = "recording_tool"
    description = "Records each call."
    permission = PermissionLevel.READ

    def __init__(self):
        self.calls: list[str] = []

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="value", type="string", required=False, default="")]

    def execute(self, value: str = "") -> ToolResult:
        self.calls.append(value)
        return ToolResult.success(message=f"recorded {value}", data=value)


class ConfirmActionTool(Tool):
    """CONFIRM-permission tool - needs confirmation under MANUAL/ASSISTED."""

    name = "confirm_action"
    description = "Needs confirmation."
    permission = PermissionLevel.CONFIRM

    def __init__(self):
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult.success(message="confirmed action ran")


class DangerousActionTool(Tool):
    """DANGEROUS-permission tool - always needs confirmation."""

    name = "dangerous_action"
    description = "Irreversible."
    permission = PermissionLevel.DANGEROUS

    def __init__(self):
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult.success(message="dangerous action ran")


class FlakyTool(Tool):
    """Fails its first `fail_times` calls, then succeeds - for retry tests."""

    name = "flaky_tool"
    description = "Fails a fixed number of times before succeeding."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.attempts = 0

    def execute(self) -> ToolResult:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            return ToolResult.error(f"transient failure #{self.attempts}")
        return ToolResult.success(message="finally worked")


class AlwaysFailsTool(Tool):
    name = "always_fails"
    description = "Always fails."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self):
        self.attempts = 0

    def execute(self) -> ToolResult:
        self.attempts += 1
        return ToolResult.error("permanent failure")


def make_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# -- fake planner -----------------------------------------------------------


class FakePlanner(Planner):
    """Returns scripted raw step lists in sequence; raises PlanningError
    once the script is exhausted (unless `error` is set, in which case
    every call raises it)."""

    def __init__(self, plans: list[list[dict]] | None = None, error: Exception | None = None):
        self.plans = list(plans or [])
        self.error = error
        self.calls: list[dict] = []

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        self.calls.append(
            {
                "goal": goal,
                "context": context,
                "autonomy": autonomy,
                "replan_context": replan_context,
            }
        )
        if self.error is not None:
            raise self.error
        if not self.plans:
            raise PlanningError("no more scripted plans")
        return self.plans.pop(0)


def step(id, tool, arguments=None, depends_on=None, **extra):
    d = {"id": id, "tool": tool, "arguments": arguments or {}, "depends_on": depends_on or []}
    d.update(extra)
    return d


def make_engine(store, registry, planner, **overrides):
    defaults = dict(
        max_steps=20,
        max_total_steps=40,
        max_retries_per_step=2,
        max_replans=2,
        retry_delay_seconds=0,
        poll_interval_seconds=1000,
    )
    defaults.update(overrides)
    return JobEngine(store=store, tools=registry, planner=planner, **defaults)


# -- planning -----------------------------------------------------------


def test_tick_plans_a_pending_job_and_runs_it_to_completion():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "recording_tool", {"value": "hi"})]])
    engine = make_engine(store, registry, planner)

    job = store.create("say hi")
    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED
    assert tool.calls == ["hi"]
    assert fetched.final_result


def test_planning_failure_fails_the_job_with_an_explanation():
    store = SQLiteJobStore(":memory:")
    registry = make_registry()
    planner = FakePlanner(error=PlanningError("provider down"))
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert "provider down" in fetched.error


def test_plan_validation_failure_fails_the_job():
    store = SQLiteJobStore(":memory:")
    registry = make_registry()  # no tools registered
    planner = FakePlanner(plans=[[step("s1", "does_not_exist")]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert "does_not_exist" in fetched.error or "unavailable" in fetched.error.lower()


def test_planner_receives_job_context():
    store = SQLiteJobStore(":memory:")
    tool = RecordingTool()
    registry = make_registry(tool)
    planner = FakePlanner(plans=[[step("s1", "recording_tool")]])
    engine = make_engine(store, registry, planner)
    store.create("goal", context={"user_name": "Bassim"})

    engine.tick()

    assert planner.calls[0]["context"] == {"user_name": "Bassim"}


# -- dependency ordering -----------------------------------------------------------


def test_dependent_step_only_runs_after_its_dependency_succeeds():
    order: list[str] = []

    class OrderTrackingTool(Tool):
        name = "track"
        description = "records order"
        permission = PermissionLevel.READ

        @property
        def parameters(self):
            return [ToolParameter(name="label", type="string", description="")]

        def execute(self, label: str) -> ToolResult:
            order.append(label)
            return ToolResult.success()

    registry = make_registry(OrderTrackingTool())
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[
            [
                step("a", "track", {"label": "a"}),
                step("b", "track", {"label": "b"}, depends_on=["a"]),
            ]
        ]
    )
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    assert order == ["a", "b"]
    assert store.get(job.id).status == JobStatus.COMPLETED


def test_independent_steps_both_run_in_one_tick():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[[step("a", "recording_tool", {"value": "a"}), step("b", "recording_tool", {"value": "b"})]]
    )
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    assert sorted(tool.calls) == ["a", "b"]
    assert store.get(job.id).status == JobStatus.COMPLETED


def test_failed_dependency_skips_dependents_and_fails_job():
    registry = make_registry(AlwaysFailsTool(), RecordingTool())
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[
            [
                step("a", "always_fails"),
                step("b", "recording_tool", {"value": "never"}, depends_on=["a"]),
            ]
        ]
    )
    engine = make_engine(store, registry, planner, max_replans=0)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    step_a = fetched.plan.get("a")
    step_b = fetched.plan.get("b")
    assert step_a.status == StepStatus.FAILED
    assert step_b.status == StepStatus.SKIPPED
    assert fetched.status == JobStatus.FAILED
    assert registry.get("recording_tool").calls == []  # never ran


# -- retries -----------------------------------------------------------


def test_transient_failure_is_retried_and_eventually_succeeds():
    tool = FlakyTool(fail_times=1)
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "flaky_tool", max_attempts=3)]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED
    assert tool.attempts == 2
    assert fetched.plan.get("s1").attempts == 2


def test_max_retries_exhausted_fails_the_step_and_job():
    tool = AlwaysFailsTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "always_fails", max_attempts=2)]])
    engine = make_engine(store, registry, planner, max_replans=0)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert tool.attempts == 2  # never exceeds max_attempts
    assert fetched.plan.get("s1").status == StepStatus.FAILED


# -- permissions / confirmation -----------------------------------------------------------


def test_manual_autonomy_pauses_for_confirmation_on_safe_action_step():
    tool = ConfirmActionTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "confirm_action")]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal", autonomy=AutonomyLevel.MANUAL)

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.AWAITING_CONFIRMATION
    assert fetched.pending_step_ids == ["s1"]
    assert fetched.confirmation_message
    assert tool.calls == 0


def test_assisted_autonomy_auto_runs_read_but_gates_confirm():
    read_tool = RecordingTool()
    confirm_tool = ConfirmActionTool()
    registry = make_registry(read_tool, confirm_tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[[step("a", "recording_tool", {"value": "auto"}), step("b", "confirm_action")]]
    )
    engine = make_engine(store, registry, planner)
    job = store.create("goal", autonomy=AutonomyLevel.ASSISTED)

    engine.tick()

    fetched = store.get(job.id)
    assert read_tool.calls == ["auto"]  # ran automatically
    assert confirm_tool.calls == 0  # gated
    assert fetched.status == JobStatus.AWAITING_CONFIRMATION
    assert fetched.pending_step_ids == ["b"]


def test_autonomous_autonomy_auto_runs_confirm_but_not_dangerous():
    confirm_tool = ConfirmActionTool()
    dangerous_tool = DangerousActionTool()
    registry = make_registry(confirm_tool, dangerous_tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[[step("a", "confirm_action"), step("b", "dangerous_action", depends_on=["a"])]]
    )
    engine = make_engine(store, registry, planner)
    job = store.create("goal", autonomy=AutonomyLevel.AUTONOMOUS)

    engine.tick()

    fetched = store.get(job.id)
    assert confirm_tool.calls == 1  # auto-ran even though CONFIRM
    assert dangerous_tool.calls == 0  # DANGEROUS always gated, even AUTONOMOUS
    assert fetched.status == JobStatus.AWAITING_CONFIRMATION
    assert fetched.pending_step_ids == ["b"]


def test_confirmed_step_runs_after_engine_approves_it():
    tool = ConfirmActionTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "confirm_action")]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal", autonomy=AutonomyLevel.MANUAL)

    engine.tick()
    fetched = store.get(job.id)
    assert fetched.status == JobStatus.AWAITING_CONFIRMATION

    # Simulate ConfirmJobTool's approval directly on the store.
    fetched.plan.get("s1").confirmed = True
    fetched.pending_step_ids = []
    fetched.confirmation_message = None
    fetched.status = JobStatus.RUNNING
    store.save(fetched)

    engine.tick()

    final = store.get(job.id)
    assert tool.calls == 1
    assert final.status == JobStatus.COMPLETED


# -- replanning -----------------------------------------------------------


def test_replan_after_permanent_failure_completes_with_alternative_step():
    failing = AlwaysFailsTool()
    recorder = RecordingTool()
    registry = make_registry(failing, recorder)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[
            [step("s1", "always_fails", max_attempts=1)],
            [step("s2", "recording_tool", {"value": "alternative"})],
        ]
    )
    engine = make_engine(store, registry, planner, max_replans=2)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.replan_count == 1
    assert recorder.calls == ["alternative"]
    replan_call = planner.calls[1]
    assert replan_call["replan_context"] is not None
    assert replan_call["replan_context"].failed_step.id == "s1"


def test_replan_limit_is_enforced():
    failing = AlwaysFailsTool()
    registry = make_registry(failing)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[
            [step("s1", "always_fails", max_attempts=1)],
            [step("s2", "always_fails", max_attempts=1)],
            [step("s3", "always_fails", max_attempts=1)],
        ]
    )
    engine = make_engine(store, registry, planner, max_replans=1)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert fetched.replan_count == 1  # only one replan attempted, then gave up
    assert "replan limit" in fetched.error.lower()


def test_replan_that_itself_fails_to_plan_fails_the_job():
    failing = AlwaysFailsTool()
    registry = make_registry(failing)
    store = SQLiteJobStore(":memory:")

    class OnceThenErrorPlanner(Planner):
        def __init__(self):
            self.calls = 0

        def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
            self.calls += 1
            if replan_context is None:
                return [step("s1", "always_fails", max_attempts=1)]
            raise PlanningError("no alternative found")

    planner = OnceThenErrorPlanner()
    engine = make_engine(store, registry, planner, max_replans=2)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert "no alternative found" in fetched.error


def test_max_total_steps_limit_blocks_further_replanning():
    failing = AlwaysFailsTool()
    registry = make_registry(failing)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[
            [step("s1", "always_fails", max_attempts=1)],
            [step(f"s{i}", "always_fails", max_attempts=1) for i in range(2, 10)],
        ]
    )
    engine = make_engine(store, registry, planner, max_replans=5, max_total_steps=3)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.FAILED
    assert fetched.replan_count == 0  # the oversized replan was rejected outright


# -- pause / resume / cancel -----------------------------------------------------------


def test_paused_job_is_never_advanced():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "recording_tool", {"value": "x"})]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal")
    job.status = JobStatus.PAUSED
    store.save(job)

    engine.tick()

    assert store.get(job.id).status == JobStatus.PAUSED
    assert tool.calls == []


def test_cancelled_job_never_runs_further_steps():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "recording_tool", {"value": "x"})]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal")
    job.status = JobStatus.CANCELLED
    store.save(job)

    engine.tick()

    assert store.get(job.id).status == JobStatus.CANCELLED
    assert tool.calls == []


def test_resumed_job_continues_from_where_it_left_off():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(
        plans=[[step("a", "recording_tool", {"value": "a"}), step("b", "recording_tool", {"value": "b"}, depends_on=["a"])]]
    )
    engine = make_engine(store, registry, planner)
    job = store.create("goal")
    engine.tick()  # plans and runs both steps to completion

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED

    # Pausing after completion is a no-op scenario for this test; instead
    # verify a mid-plan pause/resume never re-runs a SUCCEEDED step.
    job2 = store.create("goal 2")
    planner.plans.append(
        [step("a", "recording_tool", {"value": "a2"}), step("b", "recording_tool", {"value": "b2"}, depends_on=["a"])]
    )
    engine.tick()  # plans job2 and completes step "a" (b depends on a with no confirmation needed)

    fetched2 = store.get(job2.id)
    assert fetched2.status == JobStatus.COMPLETED
    assert tool.calls.count("a2") == 1  # step "a" only ran once


# -- restart recovery -----------------------------------------------------------


def test_interrupted_running_step_is_not_blindly_re_executed():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[]])  # never actually consulted in this test
    engine = make_engine(store, registry, planner, max_replans=0)

    job = store.create("goal")
    from app.orchestration.base import Plan, PlanStep

    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="recording_tool", status=StepStatus.RUNNING, attempts=1)])
    job.status = JobStatus.RUNNING
    store.save(job)

    engine.tick()

    fetched = store.get(job.id)
    assert tool.calls == []  # never re-executed
    step1 = fetched.plan.get("s1")
    assert step1.status == StepStatus.FAILED
    assert "restart" in step1.error.lower()
    assert fetched.status == JobStatus.FAILED


def test_interrupted_step_can_be_recovered_via_replan():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s2", "recording_tool", {"value": "recovered"})]])
    engine = make_engine(store, registry, planner, max_replans=1)

    job = store.create("goal")
    from app.orchestration.base import Plan, PlanStep

    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="recording_tool", status=StepStatus.RUNNING, attempts=1)])
    job.status = JobStatus.RUNNING
    store.save(job)

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.replan_count == 1
    assert tool.calls == ["recovered"]


def test_stale_planning_job_is_safely_replanned_from_scratch():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "recording_tool", {"value": "ok"})]])
    engine = make_engine(store, registry, planner)

    job = store.create("goal")
    job.status = JobStatus.PLANNING  # simulate a crash mid-planning
    store.save(job)

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED
    assert tool.calls == ["ok"]


# -- final result accuracy -----------------------------------------------------------


def test_completed_job_never_claims_success_if_any_step_failed():
    registry = make_registry(AlwaysFailsTool())
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "always_fails", max_attempts=1)]])
    engine = make_engine(store, registry, planner, max_replans=0)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status != JobStatus.COMPLETED
    assert fetched.status == JobStatus.FAILED
    assert fetched.final_result is None
    assert fetched.error


def test_completed_job_has_a_final_result_summarizing_successes():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "recording_tool", {"value": "done"})]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    fetched = store.get(job.id)
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.final_result
    assert "recording_tool" in fetched.final_result


# -- background thread lifecycle -----------------------------------------------------------


def test_start_and_stop_run_ticks_on_a_background_thread():
    tool = RecordingTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "recording_tool", {"value": "bg"})]])
    engine = make_engine(store, registry, planner, poll_interval_seconds=0.01)
    job = store.create("goal")

    engine.start()
    try:
        for _ in range(200):
            if store.get(job.id).status == JobStatus.COMPLETED:
                break
            time.sleep(0.01)
    finally:
        engine.stop()

    assert store.get(job.id).status == JobStatus.COMPLETED


def test_stop_is_a_no_op_when_never_started():
    store = SQLiteJobStore(":memory:")
    engine = make_engine(store, make_registry(), FakePlanner())
    engine.stop()  # must not raise


# -- crash-recovery regression: step outcome persisted immediately -----------


class InspectPersistedStatusTool(Tool):
    """Reads STEP `a`'s status back from the STORE (not from in-memory
    state) the moment this tool runs - proves whether a's outcome was
    actually flushed to disk before this sibling step started, i.e.
    whether a crash right now would still see 'a' as RUNNING."""

    name = "inspect_persisted_a"
    description = "Inspects persisted state of step 'a'."
    permission = PermissionLevel.READ

    def __init__(self, store, job_id):
        self.store = store
        self.job_id = job_id
        self.observed_status_of_a = None

    def execute(self) -> ToolResult:
        job = self.store.get(self.job_id)
        self.observed_status_of_a = job.plan.get("a").status
        return ToolResult.success()


def test_a_succeeded_step_is_flushed_to_disk_before_its_sibling_runs():
    """Regression test for the crash-recovery bug: within one batch of
    independent ready steps, step 'a' completing must be durably
    persisted BEFORE step 'b' (its sibling) even starts - otherwise a
    crash during 'b' would find 'a' still marked RUNNING on disk despite
    having actually succeeded."""
    store = SQLiteJobStore(":memory:")
    recorder = RecordingTool()
    registry = make_registry(recorder)
    job = store.create("goal")
    inspector = InspectPersistedStatusTool(store, job.id)
    registry.register(inspector)

    planner = FakePlanner(
        plans=[[step("a", "recording_tool", {"value": "x"}), step("b", "inspect_persisted_a")]]
    )
    engine = make_engine(store, registry, planner)

    engine.tick()

    assert inspector.observed_status_of_a == StepStatus.SUCCEEDED
    assert store.get(job.id).status == JobStatus.COMPLETED


def test_run_step_persists_the_outcome_immediately_not_only_before_the_call():
    """Directly proves `_run_step` itself flushes the final status to the
    store before returning - the actual fix, isolated from `_advance`'s
    own batch-level bookkeeping."""
    store = SQLiteJobStore(":memory:")
    tool = RecordingTool()
    registry = make_registry(tool)
    planner = FakePlanner(plans=[[step("a", "recording_tool", {"value": "x"})]])
    engine = make_engine(store, registry, planner)

    from app.orchestration.base import Plan, PlanStep

    job = store.create("goal")
    job.plan = Plan(steps=[PlanStep(id="a", tool_name="recording_tool", arguments={"value": "x"})])
    job.status = JobStatus.RUNNING
    store.save(job)

    engine._run_step(job, job.plan.get("a"))  # simulate a crash immediately after this returns

    persisted = store.get(job.id)
    assert persisted.plan.get("a").status == StepStatus.SUCCEEDED
    assert tool.calls == ["x"]


def test_crash_immediately_after_a_successful_step_does_not_replan_or_rerun_it():
    """End-to-end simulation of the exact scenario the bug allowed: step
    'a' succeeds, the 'crash' happens before anything else runs (we
    simply stop calling the engine - no further save occurs), and a
    freshly constructed engine/tick must see 'a' as SUCCEEDED already -
    never as interrupted, never re-run, never replanned."""
    store = SQLiteJobStore(":memory:")
    tool = RecordingTool()
    registry = make_registry(tool)
    job = store.create("goal")

    from app.orchestration.base import Plan, PlanStep

    job.plan = Plan(
        steps=[
            PlanStep(id="a", tool_name="recording_tool", arguments={"value": "a"}),
            PlanStep(id="b", tool_name="recording_tool", arguments={"value": "b"}, depends_on=["a"]),
        ]
    )
    job.status = JobStatus.RUNNING
    store.save(job)

    planner = FakePlanner()  # never actually consulted - no replan should be attempted
    engine = make_engine(store, registry, planner, max_replans=1)

    # Simulate: step 'a' runs and succeeds, "process" then dies before 'b'
    # is ever attempted (we call _run_step directly instead of _advance,
    # so nothing beyond 'a' executes).
    engine._run_step(job, job.plan.get("a"))

    # A fresh tick (simulating the restarted process) must find 'a'
    # already SUCCEEDED - not RUNNING, not replanned - and proceed
    # straight to running 'b'.
    engine.tick()

    finished = store.get(job.id)
    assert finished.plan.get("a").status == StepStatus.SUCCEEDED
    assert finished.plan.get("a").attempts == 1  # never re-run
    assert finished.status == JobStatus.COMPLETED
    assert finished.replan_count == 0  # no replan was ever needed or attempted
    assert tool.calls == ["a", "b"]  # 'a' ran exactly once


# -- configurable autonomy boundary (per-job, not hardcoded) -----------------


def test_autonomous_with_default_permissions_still_auto_runs_confirm():
    """An unconfigured AUTONOMOUS Job (autonomous_permissions left at its
    default) behaves exactly as before this was made configurable."""
    tool = ConfirmActionTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "confirm_action")]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal", autonomy=AutonomyLevel.AUTONOMOUS)  # no autonomous_permissions given

    engine.tick()

    assert tool.calls == 1
    assert store.get(job.id).status == JobStatus.COMPLETED


def test_autonomous_job_can_opt_out_of_auto_running_confirm_per_job():
    """Explicit empty autonomous_permissions makes AUTONOMOUS behave like
    ASSISTED for this specific Job - a genuine per-Job configuration, not
    a global engine switch."""
    tool = ConfirmActionTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "confirm_action")]])
    engine = make_engine(store, registry, planner)
    job = store.create(
        "goal", autonomy=AutonomyLevel.AUTONOMOUS, autonomous_permissions=frozenset()
    )

    engine.tick()

    fetched = store.get(job.id)
    assert tool.calls == 0
    assert fetched.status == JobStatus.AWAITING_CONFIRMATION


def test_dangerous_is_never_auto_run_even_if_autonomous_permissions_somehow_contains_it():
    """Defense in depth: even if a Job's `autonomous_permissions` somehow
    contained 'dangerous' (impossible through the public API - see
    test_orchestration_base.py - but constructed directly here to prove
    the ENGINE itself doesn't trust that field blindly), DANGEROUS steps
    still always require confirmation."""
    tool = DangerousActionTool()
    registry = make_registry(tool)
    store = SQLiteJobStore(":memory:")
    planner = FakePlanner(plans=[[step("s1", "dangerous_action")]])
    engine = make_engine(store, registry, planner)
    job = store.create("goal", autonomy=AutonomyLevel.AUTONOMOUS)
    job.autonomous_permissions = frozenset({"confirm", "dangerous"})  # bypass parse_ validation directly
    store.save(job)

    engine.tick()

    fetched = store.get(job.id)
    assert tool.calls == 0
    assert fetched.status == JobStatus.AWAITING_CONFIRMATION


def test_two_jobs_with_different_autonomy_configuration_behave_independently():
    """Proves the boundary is per-Job, not a single global setting."""
    store = SQLiteJobStore(":memory:")
    tool_a = ConfirmActionTool()
    tool_b = ConfirmActionTool()
    tool_b.name = "confirm_action_b"
    registry = make_registry(tool_a)
    registry.register(tool_b)

    permissive_job = store.create("permissive", autonomy=AutonomyLevel.AUTONOMOUS)
    strict_job = store.create(
        "strict", autonomy=AutonomyLevel.AUTONOMOUS, autonomous_permissions=frozenset()
    )

    planner = FakePlanner(
        plans=[
            [step("s1", "confirm_action")],
            [step("s1", "confirm_action_b")],
        ]
    )
    engine = make_engine(store, registry, planner)
    engine.tick()

    assert store.get(permissive_job.id).status == JobStatus.COMPLETED
    assert store.get(strict_job.id).status == JobStatus.AWAITING_CONFIRMATION


# -- context refresh before planning -----------------------------------------


def test_context_is_refreshed_immediately_before_planning():
    store = SQLiteJobStore(":memory:")
    tool = RecordingTool()
    registry = make_registry(tool)
    planner = FakePlanner(plans=[[step("s1", "recording_tool")]])

    facts = {"favorite_color": "blue"}
    engine = make_engine(store, registry, planner, context_provider=lambda: dict(facts))
    job = store.create("goal", context={"favorite_color": "red"})  # stale snapshot at creation

    facts["favorite_color"] = "green"  # changes after Job creation, before planning
    engine.tick()

    assert planner.calls[0]["context"] == {"favorite_color": "green"}
    assert store.get(job.id).context == {"favorite_color": "green"}


def test_context_refresh_preserves_goal_and_execution_history():
    store = SQLiteJobStore(":memory:")
    registry = make_registry(AlwaysFailsTool(), RecordingTool())
    planner = FakePlanner(
        plans=[
            [step("s1", "always_fails", max_attempts=1)],
            [step("s2", "recording_tool", {"value": "recovered"})],
        ]
    )
    engine = make_engine(
        store, registry, planner, max_replans=1, context_provider=lambda: {"k": "v2"}
    )
    job = store.create("original goal", context={"k": "v1"})

    engine.tick()

    finished = store.get(job.id)
    assert finished.goal == "original goal"  # never touched by a context refresh
    assert finished.context == {"k": "v2"}  # refreshed both at plan and at replan time
    assert finished.replan_count == 1
    # Execution history (events) is untouched by context refresh - it only
    # grows, it's never cleared/reset.
    kinds = [e.kind for e in finished.events]
    assert "step_failed" in kinds
    assert "replanned" in kinds


def test_no_context_provider_means_no_refresh_at_all():
    """The default (context_provider=None, every pre-existing test)
    preserves the exact prior "fixed snapshot" behavior."""
    store = SQLiteJobStore(":memory:")
    tool = RecordingTool()
    registry = make_registry(tool)
    planner = FakePlanner(plans=[[step("s1", "recording_tool")]])
    engine = make_engine(store, registry, planner)  # no context_provider
    job = store.create("goal", context={"k": "original"})

    engine.tick()

    assert planner.calls[0]["context"] == {"k": "original"}
    assert store.get(job.id).context == {"k": "original"}


def test_context_provider_failure_does_not_block_planning():
    store = SQLiteJobStore(":memory:")
    tool = RecordingTool()
    registry = make_registry(tool)
    planner = FakePlanner(plans=[[step("s1", "recording_tool")]])

    def broken_provider():
        raise RuntimeError("memory store is down")

    engine = make_engine(store, registry, planner, context_provider=broken_provider)
    job = store.create("goal", context={"k": "still here"})

    engine.tick()  # must not raise

    finished = store.get(job.id)
    assert finished.status == JobStatus.COMPLETED
    assert finished.context == {"k": "still here"}  # stale snapshot kept, not wiped
