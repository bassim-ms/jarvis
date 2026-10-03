"""
Scheduled Jobs V1 / Slice 2 - the READ-only unattended run profile and the per-run deadline, as the EXISTING JobEngine enforces them.

The profile is a REDUCTION: these tests show it removes capability at every layer (what the Planner is offered, what plan validation
accepts, what the engine will run) and grants nothing - the ToolRegistry still executes every step, an ordinary Job (no profile) behaves
exactly as before, and a Job can never be made to do more by naming a profile. The deadline is a check inside the engine's own loop,
between steps - there is no timer, thread or second engine.
"""

import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from app.orchestration.base import (
    INTERACTIVE_PROFILE,
    TIMED_OUT_ERROR,
    AutonomyLevel,
    JobStatus,
    Plan,
    PlanStep,
    StepStatus,
)
from app.orchestration.planner import PlanningError
from app.orchestration.profiles import PROFILES, UNATTENDED_V1, UnknownProfileError, permitted_tools, profile_for
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.validator import PlanValidationError, validate_plan
from app.orchestration.base import RetryPolicy
from app.scheduled_jobs.base import KNOWN_PROFILES
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.registry import ToolRegistry
from test_orchestration_engine import FakePlanner, make_engine, make_registry, step

UTC = timezone.utc
NOW = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
READ, SAFE, CONFIRM, DANGEROUS = (
    PermissionLevel.READ,
    PermissionLevel.SAFE_ACTION,
    PermissionLevel.CONFIRM,
    PermissionLevel.DANGEROUS,
)
APPROVED = {"web_search", "observe_page", "verify_observation", "analyze", "list_tasks", "list_notes", "list_events", "list_reminders"}


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class FakeTool(Tool):
    """A tool with any name and permission; records every call. `on_execute` runs inside the call (to move the clock, cancel...)."""

    description = "fake"

    def __init__(self, name, permission=READ, *, plannable=True, on_execute=None, fail=False):
        self._name, self._permission = name, permission
        self.plannable = plannable
        self.calls: list[str] = []
        self._on_execute, self._fail = on_execute, fail

    @property
    def name(self):
        return self._name

    @property
    def permission(self):
        return self._permission

    @property
    def parameters(self):
        return [ToolParameter(name="q", type="string", required=False, default="")]

    def execute(self, q: str = "") -> ToolResult:
        self.calls.append(q)
        if self._on_execute:
            self._on_execute()
        if self._fail:
            return ToolResult.error("it did not work")
        return ToolResult.success(message=f"{self._name} ok", data=q)


class SpyRegistry(ToolRegistry):
    """The real ToolRegistry, recording which tools it was asked to execute - proof the registry is still the one executor."""

    def __init__(self):
        super().__init__()
        self.executed: list[str] = []

    def execute(self, name, **kwargs):
        self.executed.append(name)
        return super().execute(name, **kwargs)


class RecordingPlanner(FakePlanner):
    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        self.calls.append({"tools": sorted(t.name for t in available_tools), "context": context, "autonomy": autonomy, "replan": replan_context})
        if self.error is not None:
            raise self.error
        if not self.plans:
            raise PlanningError("no more scripted plans")
        return self.plans.pop(0)


class Rig:
    def __init__(self, tools, plans=(), **engine_kw):
        self.jobs = SQLiteJobStore(":memory:")
        self.interactions = SQLiteInteractionStore(self.jobs)
        self.registry = SpyRegistry()
        for tool in tools:
            self.registry.register(tool)
        self.planner = RecordingPlanner(plans=list(plans))
        self.clock = Clock()
        self.engine = make_engine(self.jobs, self.registry, self.planner, clock=self.clock, interaction_store=self.interactions, **engine_kw)

    def add(self, *, profile="unattended_v1", deadline=NOW + timedelta(minutes=15), goal="research robotics", **kw):
        job = SQLiteJobStore.build_job(
            goal, run_profile=profile, deadline_at=deadline.isoformat() if deadline else None,
            autonomous_permissions=kw.pop("autonomous_permissions", frozenset()), **kw,
        )
        if "plan" in kw:
            pass
        return self.insert(job)

    def insert(self, job):
        with self.jobs.lock:
            SQLiteJobStore.insert_job_locked(self.jobs.connection, job)
            self.jobs.connection.commit()
        return job

    def run(self, job):
        self.engine.tick()
        return self.jobs.get(job.id)


def read_tools(*names):
    return [FakeTool(n) for n in names]


# ====================================== the profile itself: a short, explicit READ-only list =======================================


def test_the_unattended_profile_is_exactly_the_approved_read_only_surface():
    assert UNATTENDED_V1.allowed_tools == APPROVED
    assert (UNATTENDED_V1.max_steps, UNATTENDED_V1.max_total_steps, UNATTENDED_V1.max_replans) == (6, 9, 1)


@pytest.mark.parametrize(
    "name",
    [
        "save_note", "create_task", "complete_task", "cancel_task", "create_reminder", "complete_reminder", "cancel_reminder",
        "pause_reminder", "resume_reminder", "create_event", "update_event", "delete_event", "open_url", "scroll", "switch_window",
        "get_active_window", "get_screen_info", "open_application", "type_text", "press_key", "move_mouse", "click_mouse",
        "double_click", "observe_screen", "remember", "set_goal", "set_current_project", "morning_brief", "create_job",
        "confirm_job", "pause_job", "resume_job", "cancel_job", "create_automation", "run_automation", "commit_memory_write",
        "commit_device_revocation",
    ],
)
def test_nothing_that_writes_acts_or_touches_the_desktop_is_on_the_list(name):
    assert name not in UNATTENDED_V1.allowed_tools


def test_the_profile_name_is_one_this_system_defines_and_the_scheduled_job_foundation_knows():
    assert profile_for(INTERACTIVE_PROFILE) is None
    assert profile_for("unattended_v1") is UNATTENDED_V1
    assert set(KNOWN_PROFILES) == set(PROFILES)  # a stored intent can never name a profile the engine does not define


@pytest.mark.parametrize("name", ["", "Unattended_V1", "unattended_v2", "admin", None, 5])
def test_an_unknown_profile_is_refused_never_treated_as_no_profile(name):
    with pytest.raises(UnknownProfileError):
        profile_for(name)


def test_the_profile_can_only_say_no():
    read = FakeTool("web_search", READ)
    assert UNATTENDED_V1.denial("web_search", read) is None
    assert "not available" in UNATTENDED_V1.denial("web_search", None)
    assert "not permitted" in UNATTENDED_V1.denial("observe_screen", FakeTool("observe_screen", READ))  # read-only, but not on the list
    assert "read-only" in UNATTENDED_V1.denial("web_search", FakeTool("web_search", SAFE))  # on the list, but no longer read-only
    assert "read-only" in UNATTENDED_V1.denial("analyze", FakeTool("analyze", CONFIRM))
    assert "cannot be used in a plan" in UNATTENDED_V1.denial("web_search", FakeTool("web_search", READ, plannable=False))


def test_the_permitted_tools_are_read_from_the_live_registry():
    registry = make_registry(FakeTool("web_search"), FakeTool("analyze", SAFE), FakeTool("save_note", SAFE), FakeTool("observe_screen"))
    assert [t.name for t in permitted_tools(UNATTENDED_V1, registry)] == ["web_search"]


# ====================================== what the Planner is offered, and what validation accepts =====================================


def test_the_planner_is_offered_only_the_profiles_tools_and_an_ordinary_job_is_offered_everything_plannable():
    tools = read_tools("web_search", "analyze", "observe_screen") + [FakeTool("save_note", SAFE), FakeTool("create_event", CONFIRM)]
    rig = Rig(tools, plans=[[step("s1", "web_search")], [step("s1", "web_search")]])  # (what the plans do is beside the point here)
    scheduled = rig.add()
    ordinary = rig.add(profile=INTERACTIVE_PROFILE, deadline=None, goal="an ordinary goal")
    rig.engine.tick()
    offered = {frozenset(call["tools"]) for call in rig.planner.calls}
    assert frozenset({"web_search", "analyze"}) in offered  # the scheduled Job: only its profile's tools that exist
    assert frozenset({"web_search", "analyze", "observe_screen", "save_note", "create_event"}) in offered  # the ordinary one: unchanged
    assert rig.jobs.get(scheduled.id).status is JobStatus.COMPLETED and rig.jobs.get(ordinary.id).status is JobStatus.COMPLETED


def test_plan_validation_rejects_a_tool_outside_the_allowed_names_and_changes_nothing_when_none_are_given():
    registry = make_registry(FakeTool("web_search"), FakeTool("save_note", SAFE))
    retry = RetryPolicy(max_attempts=1)
    plan = [step("s1", "save_note")]
    assert validate_plan(plan, registry, max_steps=5, default_retry=retry)[0].tool_name == "save_note"  # no allowed names: as before
    assert validate_plan(plan, registry, max_steps=5, default_retry=retry, allowed_tool_names=None)[0].tool_name == "save_note"
    with pytest.raises(PlanValidationError, match="not permitted in this kind of run"):
        validate_plan(plan, registry, max_steps=5, default_retry=retry, allowed_tool_names=frozenset({"web_search"}))
    assert validate_plan([step("s1", "web_search")], registry, max_steps=5, default_retry=retry, allowed_tool_names=frozenset({"web_search"}))


def test_allowed_names_never_admit_what_the_ordinary_checks_refuse():
    registry = make_registry(FakeTool("web_search", plannable=False))
    with pytest.raises(PlanValidationError, match="cannot be used in a plan"):
        validate_plan([step("s1", "web_search")], registry, max_steps=5, default_retry=RetryPolicy(), allowed_tool_names=frozenset({"web_search"}))
    with pytest.raises(PlanValidationError, match="unavailable tool"):
        validate_plan([step("s1", "analyze")], registry, max_steps=5, default_retry=RetryPolicy(), allowed_tool_names=frozenset({"analyze"}))


# ====================================== READ capabilities run; everything else is refused ==========================================


def test_a_plan_of_permitted_read_tools_runs_to_completion_and_every_call_goes_through_the_registry():
    tools = read_tools("web_search", "observe_page", "verify_observation", "analyze", "list_tasks")
    rig = Rig(tools, plans=[[
        step("s1", "web_search", {"q": "robotics"}),
        step("s2", "observe_page", {"q": "https://example.com"}, depends_on=["s1"]),
        step("s3", "verify_observation", {"q": "has title"}, depends_on=["s2"]),
        step("s4", "list_tasks", {}),
        step("s5", "analyze", {"q": "summarise"}, depends_on=["s3", "s4"]),
    ]])
    final = rig.run(rig.add())
    assert final.status is JobStatus.COMPLETED and final.error is None
    assert sorted(rig.registry.executed) == sorted(["web_search", "observe_page", "verify_observation", "list_tasks", "analyze"])
    assert rig.registry.executed.index("web_search") < rig.registry.executed.index("observe_page") < rig.registry.executed.index("verify_observation")
    assert rig.registry.executed[-1] == "analyze"  # dependencies were respected, as for any Job
    assert all(t.calls for t in tools)


@pytest.mark.parametrize(
    "name, permission",
    [
        ("save_note", SAFE), ("create_task", SAFE), ("create_reminder", SAFE), ("open_url", SAFE), ("type_text", SAFE),
        ("remember", SAFE), ("create_event", CONFIRM), ("update_event", CONFIRM), ("delete_event", DANGEROUS),
        ("observe_screen", READ), ("get_active_window", READ),
    ],
)
def test_everything_off_the_list_is_rejected_whatever_its_permission_and_never_called(name, permission):
    refused = FakeTool(name, permission)
    rig = Rig([FakeTool("web_search"), refused], plans=[[step("s1", name)]])
    final = rig.run(rig.add())
    assert final.status is JobStatus.FAILED and "not permitted in this kind of run" in final.error
    assert refused.calls == [] and rig.registry.executed == []


def test_the_same_safe_action_plan_runs_for_an_ordinary_job_so_the_profile_is_what_refused_it():
    tool = FakeTool("save_note", SAFE)
    rig = Rig([FakeTool("web_search"), tool], plans=[[step("s1", "save_note", {"q": "hello"})]])
    final = rig.run(rig.add(profile=INTERACTIVE_PROFILE, deadline=None))
    assert final.status is JobStatus.COMPLETED and tool.calls == ["hello"]


def test_a_tool_that_is_on_the_list_but_no_longer_read_only_is_not_offered_and_not_accepted():
    drifted = FakeTool("analyze", SAFE)  # on the list by name, but the live registry now says SAFE_ACTION
    rig = Rig([FakeTool("web_search"), drifted], plans=[[step("s1", "analyze")]])
    final = rig.run(rig.add())
    assert final.status is JobStatus.FAILED and drifted.calls == []
    assert rig.planner.calls[0]["tools"] == ["web_search"]


def test_with_nothing_the_profile_permits_the_planner_is_not_even_asked():
    rig = Rig([FakeTool("save_note", SAFE)], plans=[[step("s1", "save_note")]])
    final = rig.run(rig.add())
    assert final.status is JobStatus.FAILED and "no capability is available" in final.error
    assert rig.planner.calls == []


# ====================================== the engine's own check, whatever the plan says =============================================


def hand_built(rig, tool_name, permission, **job_kw):
    """A Job whose stored plan names `tool_name` directly - as if validation had been skipped or the registry had changed since."""
    plan = Plan(steps=[PlanStep(id="s1", tool_name=tool_name, permission=permission.value, retry_policy=RetryPolicy(max_attempts=1))])
    job = SQLiteJobStore.build_job("g", run_profile="unattended_v1", deadline_at=(NOW + timedelta(minutes=15)).isoformat(), **job_kw)
    job.plan, job.status, job.total_step_count = plan, JobStatus.RUNNING, 1
    return rig.insert(job)


@pytest.mark.parametrize("name, permission", [("create_event", CONFIRM), ("delete_event", DANGEROUS), ("save_note", SAFE), ("observe_screen", READ)])
def test_a_plan_that_got_past_validation_still_cannot_run_a_forbidden_tool_or_be_parked_for_confirmation(name, permission):
    forbidden = FakeTool(name, permission)
    rig = Rig([FakeTool("web_search"), forbidden])
    job = hand_built(rig, name, permission)
    final = rig.run(job)
    assert forbidden.calls == [] and rig.registry.executed == []  # the registry was never even asked
    assert final.status is JobStatus.FAILED  # failed - and NOT parked for a confirmation nobody could give
    assert final.status is not JobStatus.AWAITING_CONFIRMATION and final.pending_step_ids == []
    assert final.plan.get("s1").status is StepStatus.FAILED and "not permitted" in final.plan.get("s1").error
    assert any(e.kind == "step_blocked_by_profile" for e in final.events)


def test_naming_the_profile_cannot_be_combined_with_autonomy_to_run_a_confirm_tool():
    forbidden = FakeTool("create_event", CONFIRM)
    rig = Rig([FakeTool("web_search"), forbidden])
    job = hand_built(rig, "create_event", CONFIRM, autonomy=AutonomyLevel.AUTONOMOUS, autonomous_permissions=frozenset({"confirm"}))
    final = rig.run(job)
    assert forbidden.calls == [] and final.status is JobStatus.FAILED  # AUTONOMOUS would have auto-run it; the profile says no


def test_a_permission_that_changes_after_planning_is_honoured_at_run_time():
    tool = FakeTool("web_search", READ)
    rig = Rig([tool])
    job = hand_built(rig, "web_search", READ)
    tool._permission = CONFIRM  # the registry now says this tool needs confirmation
    final = rig.run(job)
    assert tool.calls == [] and final.status is JobStatus.FAILED and final.status is not JobStatus.AWAITING_CONFIRMATION


def test_the_registry_is_still_the_executor_and_its_own_failure_path_is_unchanged():
    flaky = FakeTool("web_search", fail=True)
    rig = Rig([flaky], plans=[[step("s1", "web_search")]])
    final = rig.run(rig.add())
    assert len(flaky.calls) == 2  # the ordinary retry policy (2 attempts) applied by the ordinary engine path
    assert rig.registry.executed == ["web_search", "web_search"]
    assert final.status is JobStatus.FAILED


def test_plan_argument_validation_still_runs_for_a_scheduled_job():
    tool = FakeTool("web_search")
    rig = Rig([tool], plans=[[step("s1", "web_search", {"not_a_parameter": "x"})]])
    final = rig.run(rig.add())
    assert final.status is JobStatus.FAILED and tool.calls == []


def test_an_ordinary_job_is_not_touched_by_any_of_this():
    confirm = FakeTool("create_event", CONFIRM)
    rig = Rig([FakeTool("web_search"), confirm], plans=[[step("s1", "create_event")]])
    final = rig.run(rig.add(profile=INTERACTIVE_PROFILE, deadline=None))
    assert final.status is JobStatus.AWAITING_CONFIRMATION and confirm.calls == []  # the ordinary confirmation gate, as always


# ====================================== bounded tighter than the engine's own limits ===========================================


def test_a_scheduled_plan_is_limited_to_six_steps_where_an_ordinary_one_may_have_twenty():
    plan = [step(f"s{i}", "web_search") for i in range(7)]
    scheduled = Rig([FakeTool("web_search")], plans=[plan])
    assert "exceeding the safety limit of 6" in scheduled.run(scheduled.add()).error
    ordinary = Rig([FakeTool("web_search")], plans=[plan])
    assert ordinary.run(ordinary.add(profile=INTERACTIVE_PROFILE, deadline=None)).status is JobStatus.COMPLETED


def test_a_scheduled_run_gets_one_replan_and_then_stops():
    failing = FakeTool("web_search", fail=True)
    rig = Rig([failing], plans=[[step("s1", "web_search")], [step("r1", "web_search")], [step("r2", "web_search")]])
    final = rig.run(rig.add())
    assert final.status is JobStatus.FAILED and "replan limit of 1 reached" in final.error
    assert len(rig.planner.calls) == 2 and final.replan_count == 1  # the plan, one replan - and no third attempt
    assert len(failing.calls) == 4  # 2 attempts per plan, bounded


def test_the_engines_own_smaller_limit_still_wins():
    rig = Rig([FakeTool("web_search")], plans=[[step(f"s{i}", "web_search") for i in range(4)]], max_steps=3)
    assert "safety limit of 3" in rig.run(rig.add()).error


def test_a_job_naming_a_profile_the_system_does_not_define_does_not_run():
    tool = FakeTool("web_search")
    rig = Rig([tool], plans=[[step("s1", "web_search")]])
    final = rig.run(rig.add(profile="unattended_v9"))
    assert final.status is JobStatus.FAILED and "cannot start" in final.error
    assert rig.planner.calls == [] and tool.calls == []


# ====================================== the per-run deadline, inside the existing loop =============================================


def test_a_job_past_its_deadline_before_planning_times_out_without_calling_the_planner():
    rig = Rig([FakeTool("web_search")], plans=[[step("s1", "web_search")]])
    job = rig.add(deadline=NOW + timedelta(minutes=15))
    rig.clock.advance(minutes=16)
    final = rig.run(job)
    assert final.status is JobStatus.FAILED and final.error == TIMED_OUT_ERROR
    assert rig.planner.calls == [] and rig.registry.executed == []
    assert any(e.kind == "job_timed_out" for e in final.events)


def test_the_deadline_is_checked_between_steps_and_stops_the_run_where_it_is():
    clock_holder = {}
    first = FakeTool("web_search", on_execute=lambda: clock_holder["clock"].advance(minutes=16))  # this step runs past the deadline
    second = FakeTool("analyze")
    rig = Rig([first, second], plans=[[step("s1", "web_search"), step("s2", "analyze", depends_on=["s1"])]])
    clock_holder["clock"] = rig.clock
    final = rig.run(rig.add())
    assert first.calls and second.calls == []  # s1 ran; s2 never started
    assert final.status is JobStatus.FAILED and final.error == TIMED_OUT_ERROR


def test_a_timeout_is_persisted_as_an_ordinary_failure_and_never_retried():
    clock_holder = {}
    first = FakeTool("web_search", on_execute=lambda: clock_holder["clock"].advance(minutes=20))
    rig = Rig([first, FakeTool("analyze")], plans=[[step("s1", "web_search"), step("s2", "analyze", depends_on=["s1"])], [step("r", "web_search")]])
    clock_holder["clock"] = rig.clock
    job = rig.add()
    rig.engine.tick()
    stored = rig.jobs.get(job.id)
    assert stored.status is JobStatus.FAILED and stored.error == TIMED_OUT_ERROR and stored.version > 0
    assert stored.deadline_at == job.deadline_at and stored.run_profile == "unattended_v1"  # fixed for the run, never rewritten
    assert stored.plan.get("s1").status is StepStatus.SUCCEEDED and stored.plan.get("s2").status is StepStatus.PENDING
    before = (len(rig.planner.calls), list(rig.registry.executed))
    for _ in range(3):
        rig.engine.tick()  # nothing re-plans, retries or runs anything
    assert (len(rig.planner.calls), rig.registry.executed) == before
    assert rig.jobs.get(job.id).version == stored.version


def test_a_failing_step_is_not_retried_past_the_deadline_and_no_replan_is_asked_for():
    clock_holder = {}
    failing = FakeTool("web_search", fail=True, on_execute=lambda: clock_holder["clock"].advance(minutes=20))
    rig = Rig([failing], plans=[[step("s1", "web_search")], [step("r1", "web_search")]])
    clock_holder["clock"] = rig.clock
    final = rig.run(rig.add())
    assert final.status is JobStatus.FAILED and final.error == TIMED_OUT_ERROR
    assert len(failing.calls) == 1  # the second attempt never started
    assert len(rig.planner.calls) == 1  # and the planner was not asked to replan


def test_independent_steps_ready_together_do_not_start_after_the_deadline():
    """Two steps are ready in the same pass; the first runs past the deadline, so the second must not start."""
    holder = {}
    first = FakeTool("web_search", on_execute=lambda: holder["clock"].advance(minutes=16))
    second = FakeTool("analyze")
    rig = Rig([first, second], plans=[[step("s1", "web_search"), step("s2", "analyze")]])  # no dependency between them
    holder["clock"] = rig.clock
    final = rig.run(rig.add())
    assert first.calls and second.calls == []
    assert final.status is JobStatus.FAILED and final.error == TIMED_OUT_ERROR


def test_no_evaluation_call_is_made_after_the_deadline():
    """A step that asks to be evaluated is judged at the start of the next pass - a (model) call that must not be made once the
    run is out of time."""
    from app.orchestration.evaluator import Evaluation, ResultEvaluator

    class CountingEvaluator(ResultEvaluator):
        def __init__(self):
            self.calls = 0

        def evaluate(self, expectation, tool_name, result):
            self.calls += 1
            return Evaluation(True, "fine")

    holder = {}
    search = FakeTool("web_search", on_execute=lambda: holder["clock"].advance(minutes=16))
    evaluator = CountingEvaluator()
    rig = Rig([search, FakeTool("analyze")], plans=[[step("s1", "web_search", expectation="has results"), step("s2", "analyze", depends_on=["s1"])]], evaluator=evaluator)
    holder["clock"] = rig.clock
    final = rig.run(rig.add())
    assert search.calls and evaluator.calls == 0  # the evaluator was never called
    assert final.status is JobStatus.FAILED and final.error == TIMED_OUT_ERROR


def test_no_replanning_call_is_made_after_the_deadline_when_a_step_finally_fails():
    holder = {}
    failing = FakeTool("web_search", fail=True, on_execute=lambda: holder["clock"].advance(minutes=20))
    rig = Rig([failing], plans=[[step("s1", "web_search")], [step("r1", "web_search")]], max_retries_per_step=1)  # one attempt: it fails at once
    holder["clock"] = rig.clock
    final = rig.run(rig.add())
    assert len(failing.calls) == 1 and final.plan.get("s1").status is StepStatus.FAILED  # a real, final step failure...
    assert len(rig.planner.calls) == 1  # ...and still no replanning call, because the run is out of time
    assert final.status is JobStatus.FAILED and final.error == TIMED_OUT_ERROR


def test_a_job_that_finished_before_its_deadline_is_not_affected_by_it():
    rig = Rig([FakeTool("web_search")], plans=[[step("s1", "web_search")]])
    job = rig.add(deadline=NOW + timedelta(seconds=1))
    final = rig.run(job)
    assert final.status is JobStatus.COMPLETED
    rig.clock.advance(hours=5)
    rig.engine.tick()
    assert rig.jobs.get(job.id).status is JobStatus.COMPLETED and rig.jobs.get(job.id).error is None


def test_the_deadline_belongs_to_the_job_instance_two_jobs_have_two_deadlines():
    rig = Rig([FakeTool("web_search")], plans=[[step("s1", "web_search")], [step("s1", "web_search")]])
    short = rig.add(deadline=NOW + timedelta(minutes=5), goal="short")
    long = rig.add(deadline=NOW + timedelta(hours=2), goal="long")
    rig.clock.advance(minutes=10)
    rig.engine.tick()
    assert rig.jobs.get(short.id).error == TIMED_OUT_ERROR
    assert rig.jobs.get(long.id).status is JobStatus.COMPLETED


def test_an_unreadable_deadline_fails_closed_and_a_job_with_none_has_no_limit():
    rig = Rig([FakeTool("web_search")], plans=[[step("s1", "web_search")]])
    broken = rig.add(deadline=None)
    with rig.jobs.lock:
        rig.jobs.connection.execute("UPDATE jobs SET deadline_at = 'not a time' WHERE id = ?", (broken.id,))
        rig.jobs.connection.commit()
    assert rig.run(broken).error == TIMED_OUT_ERROR
    unlimited = Rig([FakeTool("web_search")], plans=[[step("s1", "web_search")]])
    job = unlimited.add(deadline=None)
    unlimited.clock.advance(days=400)
    assert unlimited.run(job).status is JobStatus.COMPLETED


def test_there_is_no_timer_thread_or_second_loop_for_the_deadline():
    import inspect

    from app.orchestration.engine import JobEngine

    deadline_code = "".join(inspect.getsource(getattr(JobEngine, name)) for name in ("_enforce_deadline", "_past_deadline", "_now_dt"))
    for needle in ("Thread", "Timer", "sleep", "asyncio", "signal", "call_later", "while True", "start("):
        assert needle not in deadline_code, needle
    source = (pathlib.Path(__file__).resolve().parent.parent / "app" / "orchestration" / "engine.py").read_text(encoding="utf-8")
    for needle in ("threading.Timer", "sched.", "asyncio", "signal.alarm", "call_later"):
        assert needle not in source, needle
    assert source.count("threading.Thread(") == 2  # unchanged: the engine's poll thread and the existing notification helper


# ====================================== cancellation between steps, by the existing mechanism ======================================


def test_a_run_cancelled_between_steps_stops_cleanly_in_the_existing_cancelled_state():
    holder = {}

    def cancel_now():
        current = holder["rig"].jobs.get(holder["job"].id)
        holder["rig"].engine.cancel_job(current.id, expected_job_version=current.version, reason="dropped")

    first = FakeTool("web_search", on_execute=cancel_now)
    second = FakeTool("analyze")
    rig = Rig([first, second], plans=[[step("s1", "web_search"), step("s2", "analyze", depends_on=["s1"])]])
    holder["rig"] = rig
    holder["job"] = rig.add()
    rig.engine.tick()
    final = rig.jobs.get(holder["job"].id)
    assert final.status is JobStatus.CANCELLED  # the existing terminal state, reached through the existing cancel_job
    assert first.calls and second.calls == []  # the next step never ran
    rig.engine.tick()
    assert rig.jobs.get(holder["job"].id).status is JobStatus.CANCELLED and second.calls == []


def test_a_cancelled_run_is_never_turned_into_a_timeout_or_a_failure_later():
    rig = Rig([FakeTool("web_search")], plans=[[step("s1", "web_search")]])
    job = rig.add()
    current = rig.jobs.get(job.id)
    rig.engine.cancel_job(job.id, expected_job_version=current.version)
    rig.clock.advance(hours=3)
    rig.engine.tick()
    assert rig.jobs.get(job.id).status is JobStatus.CANCELLED and rig.jobs.get(job.id).error is None
