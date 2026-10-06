"""
Scheduled Jobs V1 / Slice 4 - the boundaries of the creation/approval layer (app/scheduled_operations).

This layer connects the USER to the scheduler's path and nothing more: it understands a request (through the EXISTING Handle This, handed in
as a callable), stores a PROPOSAL, asks through the EXISTING confirmation, and on an approved confirmation activates it atomically. It is
not a scheduler, an engine, a planner, a second approval system or a second Schedule store. These tests pin that, so a later change has to
loosen them deliberately.
"""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "app" / "scheduled_operations"
FILES = sorted(PACKAGE.glob("*.py"))

#: The only parts of the application this package may import.
ALLOWED_APP_IMPORTS = (
    "app.scheduled_operations", "app.scheduled_jobs", "app.scheduling", "app.reminders.language", "app.reminders.timefmt",
    "app.notifications.text", "app.notifications.base", "app.orchestration.base", "app.orchestration.interactions", "app.orchestration.sqlite_store",
    "app.tools.base", "app.config", "app.system_timezone",
)
FORBIDDEN = (
    "app.handle_this", "app.providers", "app.memory", "app.memory_policy", "app.world_model", "app.devices", "app.computer",
    "app.observation", "app.automation", "app.core", "app.main", "app.calendar", "app.search", "app.tasks", "app.notes",
    "app.context_engine", "app.context_reasoning", "app.context_planning", "app.morning_brief", "app.self_development", "app.voice",
    "app.scheduled_execution", "app.orchestration.engine", "app.orchestration.planner", "app.tools.registry",
)


def imported_modules(path: pathlib.Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def code_identifiers(path: pathlib.Path) -> set[str]:
    """Every name and attribute the CODE uses (docstrings and comments excluded)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
    return names


def test_the_package_is_what_we_expect():
    assert {f.name for f in FILES} == {"__init__.py", "language.py", "operations.py", "presenter.py", "service.py", "tools.py"}


def test_it_imports_nothing_that_could_plan_run_a_tool_reach_a_device_write_memory_or_call_a_model():
    for path in FILES:
        for module in imported_modules(path):
            if not module.startswith("app."):
                continue
            assert module.startswith(ALLOWED_APP_IMPORTS), (path.name, module)
            assert not module.startswith(FORBIDDEN), (path.name, module)


def test_handle_this_is_reached_only_through_a_callable_the_wiring_supplies():
    for path in FILES:
        assert not any(m.startswith("app.handle_this") for m in imported_modules(path)), path.name
    assert "handle_this" in code_identifiers(PACKAGE / "service.py")  # the parameter of `understander`, nothing imported


def test_there_is_no_thread_timer_loop_or_polling_here():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        for pattern in (r"^\s*import (threading|asyncio|sched|multiprocessing)\b", r"^\s*from (threading|asyncio|sched)\b", r"\.sleep\(", r"Timer\(", r"Thread\("):
            assert not re.search(pattern, text, re.MULTILINE), (path.name, pattern)


def test_nothing_here_plans_runs_a_tool_launches_work_or_settles_an_occurrence():
    used = set().union(*(code_identifiers(path) for path in FILES))
    for needle in (
        "ToolRegistry", "create_job", "JobEngine", "Planner", "confirm_job", "NotificationRouter", "generate_reply", "AIProvider",
        "insert_job_locked", "record_run", "settle_run", "list_due", "ScheduledJobScheduler", "ScheduledExecutionAdapter",
        "subprocess",
    ):
        assert needle not in used, needle


def test_the_service_builds_no_Job_at_all_and_the_confirmer_builds_only_the_waiting_confirmation_Job():
    assert not {"Job", "JobStatus", "Plan", "PlanStep", "build_job"} & code_identifiers(PACKAGE / "service.py")
    assert not {"Job", "build_job", "create_job_awaiting_confirmation"} & code_identifiers(PACKAGE / "language.py")
    operations = (PACKAGE / "operations.py").read_text(encoding="utf-8")
    assert operations.count("create_job_awaiting_confirmation(") == 1  # the ONE way it makes a Job: the existing waiting confirmation


def test_the_service_calls_only_the_stores_own_atomic_operations_and_has_no_sql_of_its_own():
    text = (PACKAGE / "service.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(sqlite3|execute\(|executescript|commit\(|rollback\()", text)
    used = code_identifiers(PACKAGE / "service.py")
    assert {"create_proposal", "activate", "transition", "resume", "expire_proposals"} <= used


def test_the_whole_layer_is_wired_through_main_alone_and_the_core_only_knows_a_duck_typed_hook():
    importers = sorted(
        str(path.relative_to(ROOT / "app")).replace("\\", "/")
        for path in (ROOT / "app").rglob("*.py")
        if "scheduled_operations" not in path.parts
        and re.search(r"^\s*(from|import) app\.scheduled_operations", path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert importers == ["main.py"]
    core = (ROOT / "app" / "core.py").read_text(encoding="utf-8")
    assert "scheduled_job_operations" in core and not re.search(r"^\s*(from|import) app\.scheduled", core, re.MULTILINE)
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert main.count("CommitScheduledJobApprovalTool(") == 1 and main.count("ScheduledJobScheduler(") == 1  # no second scheduler


def test_the_door_sits_after_the_reminder_door_and_never_serves_an_explicit_tool_status_confirmation_or_self_development_request():
    core = (ROOT / "app" / "core.py").read_text(encoding="utf-8")
    block = core[core.index("self.scheduled_job_operations is not None"):]
    block = block[: block.index("memory_reply = None\n")]
    assert "Capability.DIRECT, Capability.SEARCH, Capability.TASK" in block and "interaction_reply is None" in block
    for forbidden in ("Capability.TOOL", "Capability.TASK_STATUS", "Capability.CONFIRMATION", "Capability.SELF_DEVELOPMENT"):
        assert forbidden not in block
    assert core.index("self.reminder_operations.command_reply") < core.index("self.scheduled_job_operations.command_reply")


def test_the_only_edit_to_the_execution_adapter_is_the_shared_check_the_run_already_applied():
    adapter = (ROOT / "app" / "scheduled_execution" / "adapter.py").read_text(encoding="utf-8")
    assert adapter.count("def check_objective(") == 1
    assert "self.check_objective(scheduled_job.objective, scheduled_job.profile)" in adapter  # creation and every run use ONE check


def test_the_only_thing_this_layer_ever_writes_about_what_it_told_the_user_is_the_reported_marker():
    """`mark_reported` (the ledger's own "the user was told" flag) is used by the reporter in presenter.py and nowhere else here; the presenter
    functions are pure words and the rest of the layer cannot even name it."""
    users = {path.name for path in FILES if "mark_reported" in code_identifiers(path)}
    assert users == {"presenter.py"}
    presenter = (PACKAGE / "presenter.py").read_text(encoding="utf-8")
    present_run = presenter[presenter.index("def present_run("): presenter.index("# ====================================== telling the user, once")]
    assert not re.search(r"\b(store|_store|mark_reported|notify|transition|record_run|settle_run|activate|resume)\b", present_run.split('"""')[2])


def test_the_engine_only_learned_that_a_Job_under_a_capability_profile_is_reported_by_its_owner():
    engine = (ROOT / "app" / "orchestration" / "engine.py").read_text(encoding="utf-8")
    assert "scheduled_job" not in engine and "app.scheduled" not in engine  # still knows nothing of Scheduled Jobs (a run profile is generic)
    block = engine[engine.index("def _route_finished"): engine.index("def _notify_if_finished")]
    assert "job.run_profile != INTERACTIVE_PROFILE" in block and block.index("run_profile") < block.index("failed_event(job)")


def test_run_now_is_the_schedulers_own_method_and_no_other_package_can_launch_a_scheduled_run():
    scheduler = (ROOT / "app" / "scheduled_execution" / "scheduler.py").read_text(encoding="utf-8")
    assert scheduler.count("def run_now(") == 1 and scheduler.count("self._adapter.launch(") == 2  # the due occurrence, and run-now: ONE adapter
    for path in FILES:
        assert "launch" not in code_identifiers(path), path.name  # this layer hands over a request; it never launches
