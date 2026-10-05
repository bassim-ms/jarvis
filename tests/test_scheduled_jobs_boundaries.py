"""
Scheduled Jobs V1 / Slice 1 - the boundaries. This slice is a FOUNDATION: it must not be able to run, plan, call a tool, notify, write
memory, or reach a device, and it must not be wired into the running application yet. These tests pin that, so a later slice that
changes it has to do so deliberately.
"""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "app" / "scheduled_jobs"
FILES = sorted(PACKAGE.glob("*.py"))

#: The only parts of the application this package may import. Everything else - tools, orchestration, handle_this, providers, memory,
#: the World Model, notifications, automation, computer/observation tools, devices, core, main - is out of reach by construction.
ALLOWED_APP_IMPORTS = ("app.scheduled_jobs", "app.scheduling", "app.config")
FORBIDDEN = (
    "app.tools", "app.orchestration", "app.handle_this", "app.providers", "app.memory", "app.memory_policy", "app.world_model",
    "app.notifications", "app.automation", "app.computer", "app.observation", "app.devices", "app.core", "app.main", "app.calendar",
    "app.search", "app.tasks", "app.notes", "app.context_engine", "app.context_reasoning", "app.context_planning", "app.reminders",
    "app.morning_brief", "app.self_development", "app.voice",
)


def imported_modules(path: pathlib.Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_package_is_what_we_expect():
    assert {f.name for f in FILES} == {"__init__.py", "base.py", "policy.py", "sqlite_store.py"}


def test_it_imports_nothing_that_could_run_plan_notify_or_reach_the_world():
    for path in FILES:
        for module in imported_modules(path):
            if not module.startswith("app."):
                continue
            assert module.startswith(ALLOWED_APP_IMPORTS), (path.name, module)
            assert not module.startswith(FORBIDDEN), (path.name, module)


def test_the_model_and_the_policy_are_pure_no_clock_no_io_no_threads():
    forbidden = [
        r"\.now\(", r"\.today\(", r"utcnow", r"time\.time\(", r"time\.monotonic", r"\bopen\(", r"os\.environ", r"os\.getenv",
        r"^\s*import (os|sys|socket|subprocess|threading|sqlite3|httpx|requests|urllib|random)\b",
        r"^\s*from (os|sys|socket|subprocess|threading|sqlite3|httpx|requests|urllib|random)\b",
    ]
    for name in ("base.py", "policy.py"):
        text = (PACKAGE / name).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text, re.MULTILINE), (name, pattern)


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


def test_nothing_in_the_package_executes_tools_or_builds_jobs():
    used = set().union(*(code_identifiers(path) for path in FILES))
    for needle in (
        "ToolRegistry", "create_job", "JobEngine", "Planner", "NotificationRouter", "confirm_job", "subprocess", "Job", "JobStore",
        "build_job", "insert_job_locked", "plan_run_tool", "generate_reply", "AIProvider",
    ):
        assert needle not in used, needle


def test_the_only_way_a_job_reaches_the_database_is_a_callable_the_caller_supplies():
    source = (PACKAGE / "sqlite_store.py").read_text(encoding="utf-8")
    assert "INSERT INTO jobs" not in source and "UPDATE jobs" not in source and "FROM jobs" not in source
    assert "insert_job(self._conn)" in source  # the one place a Job is written, and it is the caller's code that does it


def test_the_foundation_is_wired_only_through_main_and_nothing_else_reaches_it():
    """Slice 3 wired the scheduler in - through ONE place, main.py, which builds it and hangs it on the JobEngine's tick. Nothing else
    in the application (the core, the default tools, the engine, the factory) names it, so a chat message or a tool cannot reach a
    Scheduled Job's store or ledger, and the engine itself still knows nothing of scheduling. (This replaces Slice 1's "not wired in".)"""
    for relative in ("app/core.py", "app/tools/defaults.py", "app/orchestration/engine.py", "app/orchestration/factory.py"):
        assert "scheduled_jobs" not in (ROOT / relative).read_text(encoding="utf-8"), relative
    importers = sorted(
        str(path.relative_to(ROOT / "app")).replace("\\", "/")
        for path in (ROOT / "app").rglob("*.py")
        if "scheduled_jobs" not in path.parts
        and re.search(r"^\s*(from|import) app\.scheduled_jobs", path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    # Slice 4 adds the creation/approval layer (app/scheduled_operations): it reads the store and the model, and is itself wired only by main.
    assert importers == [
        "main.py", "scheduled_execution/adapter.py", "scheduled_execution/context.py", "scheduled_execution/occurrence.py",
        "scheduled_execution/scheduler.py", "scheduled_operations/operations.py", "scheduled_operations/service.py",
    ]


def test_the_kill_switch_is_a_setting_the_pure_foundation_never_reads_itself():
    """The switch is read by the scheduler (injected as a callable from main.py) - never by this package's pure policy or store, so no
    rule here can depend on it and it cannot become a second permission system inside the foundation."""
    from app.config import Settings

    assert Settings.model_fields["scheduled_jobs_enabled"].default is True
    assert "scheduled_jobs_enabled" not in "\n".join(path.read_text(encoding="utf-8") for path in FILES)
