"""
The Slice 3 BOUNDARY, and the package's own boundaries.

Slice 2 creates no Job. What it provides is (1) `ActionProposalService.on_approved`, the one place an approved, revalidated proposal leaves
the package - unset in the application until Slice 3 provides it - and (2) `SQLiteJobStore.create_running_job`, the smallest piece of Job
infrastructure that handoff needs: a Job born RUNNING with a plan that was already validated and approved, in one atomic insert.

The composition test below shows the pieces fit (approve -> revalidate -> THEN a Job -> the existing engine runs exactly those steps) using
a handoff written inside the test; nothing of Slice 3's execution behaviour exists in the application.
"""

import ast
import pathlib
import re
import sys

import pytest

from app.action_proposal.base import ALLOWED_PERMISSIONS, ALLOWED_TOOLS
from app.action_proposal.builder import ProposalError, plan_steps_for
from app.action_proposal.base import ToolAction
from app.config import settings
from app.orchestration.base import JobStatus, Plan, PlanStep, RetryPolicy, StepStatus
from app.orchestration.sqlite_store import PREBUILT_PLAN_REPLAN_COUNT, SQLiteJobStore
from app.orchestration.validator import validate_plan
from app.tools.base import PermissionLevel, Tool, ToolResult
from test_action_proposal_door import PLAN, CoreRig
from test_interaction_resolver import World


def steps_for(world, *tools):
    raw = [{"id": f"a{i}", "tool": tool, "arguments": {}, "depends_on": []} for i, tool in enumerate(tools, start=1)]
    return validate_plan(raw, world.registry, max_steps=4, default_retry=RetryPolicy())


# =============================================== create_running_job: the atomic handoff ====================================================


def test_a_job_is_born_running_with_exactly_the_approved_plan():
    world = World()
    steps = steps_for(world, "free", "after")

    job = world.jobs.create_running_job("Approved plan", steps, conversation_id="c1", context={"source": "action_proposal", "digest": "abc"})

    stored = world.jobs.get(job.id)
    assert stored.status is JobStatus.RUNNING and world.jobs.list(JobStatus.PENDING) == []  # never PENDING, not even for a moment
    assert [(s.id, s.tool_name, s.permission, s.status) for s in stored.plan.steps] == [
        ("a1", "free", "safe_action", StepStatus.PENDING), ("a2", "after", "safe_action", StepStatus.PENDING)
    ]
    assert stored.total_step_count == 2 and stored.replan_count == PREBUILT_PLAN_REPLAN_COUNT
    assert stored.conversation_id == "c1" and stored.context == {"source": "action_proposal", "digest": "abc"}
    assert stored.version == 0 and stored.error is None and stored.final_result is None


def test_the_replan_budget_starts_spent_so_a_model_is_never_asked_for_different_steps():
    assert PREBUILT_PLAN_REPLAN_COUNT > settings.orchestration_max_replans


def test_the_engine_runs_exactly_those_steps_and_never_asks_the_planner():
    world = World()
    job = world.jobs.create_running_job("Approved plan", steps_for(world, "free", "after"))

    world.engine.tick()

    done = world.jobs.get(job.id)
    assert done.status is JobStatus.COMPLETED and world.log == ["free", "after"]
    assert world.planner.contexts == []  # the planner (the model) was never consulted: there was nothing to plan


def test_a_step_that_needs_confirmation_is_still_gated_by_the_existing_system():
    world = World()
    job = world.jobs.create_running_job("Approved plan", steps_for(world, "gated"))

    world.engine.tick()

    parked = world.jobs.get(job.id)
    assert parked.status is JobStatus.AWAITING_CONFIRMATION and world.log == []  # approving the PLAN did not approve a CONFIRM step
    assert world.planner.contexts == []


def test_a_failing_step_never_asks_a_model_for_a_different_plan():
    world = World()

    class Failing(Tool):
        name = "failing"
        description = "always fails"
        permission = PermissionLevel.SAFE_ACTION

        def execute(self) -> ToolResult:
            return ToolResult.error("it broke")

    world.registry.register(Failing())
    job = world.jobs.create_running_job("Approved plan", steps_for(world, "failing"))

    for _ in range(3):
        world.engine.tick()

    assert world.jobs.get(job.id).status is JobStatus.FAILED and world.planner.contexts == []


def test_this_is_why_the_job_is_not_created_pending_with_a_plan_the_engine_would_plan_it_over():
    """The fact the design rests on: a PENDING Job is PLANNED by the model, so a plan set on it is not what runs."""
    world = World()
    job = world.jobs.create("Approved plan")  # PENDING
    job.plan = Plan(steps=steps_for(world, "free"))
    world.jobs.save(job)

    world.engine.tick()

    assert len(world.planner.contexts) == 1  # the planner WAS asked, whatever plan the Job carried
    assert world.log == []  # ...and the carried plan did not run


@pytest.mark.parametrize("make", ["empty", "unvalidated", "already_run", "confirmed", "has_result"])
def test_anything_but_fresh_validated_steps_is_refused_and_nothing_is_inserted(make):
    world = World()
    fresh = steps_for(world, "free")[0]
    bad = {
        "empty": [],
        "unvalidated": [PlanStep(id="a1", tool_name="free")],  # no permission filled by the validator
        "already_run": [PlanStep(id="a1", tool_name="free", permission="safe_action", status=StepStatus.SUCCEEDED)],
        "confirmed": [PlanStep(id="a1", tool_name="free", permission="confirm", confirmed=True)],
        "has_result": [PlanStep(id="a1", tool_name="free", permission="safe_action", result={"status": "success"})],
    }[make]

    with pytest.raises(ValueError):
        world.jobs.create_running_job("Approved plan", bad or [])

    assert world.jobs.list() == [] and fresh.status is StepStatus.PENDING
    assert world.jobs.create_running_job("Approved plan", [fresh]).status is JobStatus.RUNNING  # and the store is unharmed


def test_a_failed_insert_leaves_no_row_and_the_store_usable(monkeypatch):
    world = World()
    steps = steps_for(world, "free")

    def boom(conn, job):
        raise RuntimeError("disk full")

    monkeypatch.setattr(SQLiteJobStore, "insert_job_locked", staticmethod(boom))
    with pytest.raises(RuntimeError):
        world.jobs.create_running_job("Approved plan", steps)
    monkeypatch.undo()

    assert world.jobs.list() == []
    assert world.jobs.create_running_job("Approved plan", steps_for(world, "free")).status is JobStatus.RUNNING


# =============================================== the composition: approval -> revalidation -> THEN a Job ==================================


def composed(tmp_path):
    rig = CoreRig(tmp_path)

    def handoff(approved):  # a stand-in for Slice 3, living only in this test
        job = rig.world.jobs.create_running_job(
            f"Approved plan: {approved.subject}", list(approved.plan_steps), conversation_id=approved.conversation_id,
            context={"source": "action_proposal", "proposal_id": approved.proposal_id, "digest": approved.digest},
        )
        return f"Started {job.id}"

    rig.service.on_approved = handoff
    return rig


def test_no_job_exists_until_the_yes_and_then_exactly_the_approved_step_runs(tmp_path):
    rig = composed(tmp_path)
    rig.planned()
    assert rig.world.jobs.list() == [] and rig.reminders.list() == []  # shown, not started

    reply = rig.say("yes")

    (job,) = rig.world.jobs.list()
    assert reply == f"Started {job.id}" and job.status is JobStatus.RUNNING
    assert [(s.tool_name, s.permission) for s in job.plan.steps] == [("create_reminder", "safe_action")]
    assert job.context["source"] == "action_proposal" and len(job.context["digest"]) == 64 and job.context["proposal_id"]
    assert rig.reminders.list() == []  # approving still ran nothing by itself...

    rig.world.engine.tick()  # ...the existing engine does, under the existing rules

    (reminder,) = rig.reminders.list()
    assert reminder.text == "Test Arduino servo wiring is due at 15:00" and rig.world.jobs.get(job.id).status is JobStatus.COMPLETED
    assert rig.world.planner.contexts == [] and rig.provider.calls == 0


@pytest.mark.parametrize("scenario", ["stale", "expired", "wrong_conversation", "already_consumed", "rejected", "unrelated", "other_message", "ok_only"])
def test_a_proposal_that_cannot_be_approved_creates_no_job_at_all(tmp_path, scenario):
    rig = composed(tmp_path)
    rig.planned()
    clock = [0.0]
    rig.service._monotonic = lambda: clock[0]
    rig.service._pending[rig.conversation].expires_at = 600.0

    if scenario == "stale":
        rig.tasks.complete(rig.second.id)
        rig.say("yes")
    elif scenario == "expired":
        clock[0] = 600.0
        rig.say("yes")
    elif scenario == "wrong_conversation":
        rig.service.note_message("another")
        rig.service.confirmation_reply("yes", "another")
    elif scenario == "already_consumed":
        rig.say("yes")  # the one approval: exactly one Job...
        assert len(rig.world.jobs.list()) == 1
        rig.say("yes")  # ...and a second yes creates nothing more
        rig.say("go ahead")
        assert len(rig.world.jobs.list()) == 1 and len(rig.handed) == 0
        return
    elif scenario == "rejected":
        rig.say("no")
        rig.say("yes")
    elif scenario == "unrelated":
        rig.say("tell me a joke")
        rig.say("yes")
    elif scenario == "other_message":
        rig.say("Search for Arduino servo wiring")
        rig.say("yes")
    elif scenario == "ok_only":
        rig.say("ok")
        rig.say("sure")

    assert rig.world.jobs.list() == [] and rig.reminders.list() == [] and rig.handed == []


def test_what_runs_is_what_was_shown_and_never_re_planned(tmp_path):
    rig = composed(tmp_path)
    reply = rig.planned()
    shown_time = "16:30" if "16:30" in reply else "14:30"

    rig.say("yes")
    rig.world.engine.tick()

    (reminder,) = rig.reminders.list()
    assert f"set a reminder for {shown_time} today" in reply and reminder.text in reply
    assert rig.world.planner.contexts == [] and rig.provider.calls == 0


# =============================================== what a proposal may ever ask for =========================================================


def test_only_a_reminder_a_note_a_task_or_one_completion_may_be_proposed_and_only_as_a_safe_action():
    assert ALLOWED_TOOLS == frozenset({"create_reminder", "save_note", "create_task", "complete_task"}) and ALLOWED_PERMISSIONS == frozenset({"safe_action"})


def test_the_registry_is_asked_every_time(tmp_path):
    rig = CoreRig(tmp_path)
    registry = rig.world.registry
    good = ToolAction("create_reminder", (("at", "14:30"), ("day", "today"), ("text", "x")))

    assert [s.tool_name for s in plan_steps_for(registry, [good])] == ["create_reminder"]
    with pytest.raises(ProposalError):
        plan_steps_for(registry, [ToolAction("cancel_task", (("task_id", "x"),))])  # real and safe - but not a tool a proposal may name
    with pytest.raises(ProposalError):
        plan_steps_for(registry, [ToolAction("create_event", (("title", "x"),))])  # needs confirmation
    with pytest.raises(ProposalError):
        plan_steps_for(registry, [ToolAction("delete_event", (("event_id", "x"),))])  # dangerous
    with pytest.raises(ProposalError):
        plan_steps_for(registry, [ToolAction("create_reminder", (("text", "x"), ("tool", "delete_event")))])  # an argument the tool does not declare
    with pytest.raises(ProposalError):
        plan_steps_for(registry, [ToolAction("not_a_tool", ())])


# =============================================== the package boundary ====================================================================

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "app" / "action_proposal"
ALLOWED_APP_IMPORTS = {
    "app.action_proposal", "app.action_proposal.base", "app.action_proposal.builder", "app.action_proposal.recognition",
    "app.action_proposal.render", "app.action_proposal.service", "app.action_proposal.handoff", "app.action_proposal.result",
    "app.focus.base", "app.focus.recognition", "app.focus.render",  # Focus is called, never changed
    "app.context_engine.scoring", "app.morning_brief.gather", "app.tasks.base",
    "app.orchestration.base", "app.orchestration.validator",  # the existing plan validator and its types - no store, no engine
}


def _modules(path):
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


def test_the_package_imports_only_the_agreed_existing_helpers_and_the_standard_library():
    files = sorted(PACKAGE.glob("*.py"))
    assert {f.name for f in files} == {"__init__.py", "base.py", "builder.py", "handoff.py", "recognition.py", "render.py", "result.py", "service.py"}
    for path in files:
        for module in _modules(path):
            top = module.split(".")[0]
            if top == "app":
                assert module in ALLOWED_APP_IMPORTS, f"{path.name} imports {module}"
            else:
                assert top in sys.stdlib_module_names or top == "__future__", f"{path.name} imports {module}"


def _without_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def test_the_package_cannot_create_a_job_run_a_tool_reach_a_model_or_touch_the_disk():
    banned_modules = {"asyncio", "subprocess", "socket", "sqlite3", "os", "pathlib", "shutil", "urllib", "http", "requests", "random", "sched"}
    banned_words = ("create_job", "create_running_job", "JobStore", "JobEngine", "generate_reply", "provider", "interaction", "resolver", ".execute(", "memory_writer")
    writes = {
        "create", "complete", "cancel", "update", "delete", "save", "set", "submit", "execute", "run", "send", "notify", "write", "add", "remove",
        "insert", "commit", "transition", "link", "put", "post", "start", "launch", "register", "unregister", "create_event", "create_task",
    }
    called = set()
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not banned_modules & {m.split(".")[0] for m in _modules(path)}, path.name
        stripped = ast.unparse(_without_docstrings(tree))  # behaviour, not the prose that explains why there is none
        for word in banned_words:
            assert word not in stripped, (path.name, word)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "exec", "eval", "compile", "__import__", "input", "print"}, path.name
    assert not called & writes, called & writes


def test_a_proposal_never_stores_anything_the_only_state_is_the_bounded_in_memory_map():
    text = (PACKAGE / "service.py").read_text(encoding="utf-8")

    assert "self._pending: dict[str, _Pending]" in text and "MAX_PENDING" in text
    assert len(re.findall(r"self\._pending\[[^\]]+\]\s*=", text)) == 2  # assigned in _hold and _skip only: one map, nothing else


def test_the_package_has_no_model_in_it():
    for path in PACKAGE.glob("*.py"):
        lowered = ast.unparse(_without_docstrings(ast.parse(path.read_text(encoding="utf-8")))).lower()
        for word in ("gemini", "openai", "anthropic", "generate_reply", "ai_provider"):
            assert word not in lowered, (path.name, word)
        assert not re.search(r"\bllm\b", lowered), path.name
