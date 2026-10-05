"""
Scheduled Jobs V1 / Slice 2 - the boundaries. The execution adapter is the one place a Scheduled Job meets the orchestration layer, and
it must stay narrow: it is INVOKED (never wakes up on its own), it creates a Job and nothing else, and it is not wired into the
application. There is no scheduler, no polling, no background thread, no notification or presenter, no chat flow, no memory write,
no tool execution, and no second engine / registry / planner / permission system.
"""

import ast
import pathlib
import re
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "app" / "scheduled_execution"
FILES = sorted(PACKAGE.glob("*.py"))

#: What the adapter may import from the application. Nothing here can run a tool, plan, notify, or write memory.
ALLOWED = (
    "app.scheduled_execution", "app.scheduled_jobs", "app.scheduling", "app.capability", "app.context_engine.request_builder",
    "app.orchestration.base", "app.orchestration.profiles", "app.orchestration.sqlite_store", "app.orchestration.tools",
    "app.handle_this.base", "app.handle_this.boundary", "app.handle_this.signals",
)
#: The ONE part of Handle This the adapter reads: its capability boundary (and the family enum and the wording detector that
#: boundary is applied with), re-run at run time. Nothing else of Handle This - not the intake, interpreter, service or factory.
HANDLE_THIS_MODULES = {"app.handle_this.base", "app.handle_this.boundary", "app.handle_this.signals"}
FORBIDDEN = (
    "app.orchestration.engine", "app.orchestration.planner", "app.orchestration.interaction", "app.orchestration.sqlite_interaction",
    "app.tools", "app.notifications", "app.memory", "app.memory_policy", "app.handle_this", "app.providers", "app.automation",
    "app.computer", "app.observation", "app.devices", "app.core", "app.main", "app.reminders", "app.world_model", "app.morning_brief",
    "app.calendar", "app.search", "app.tasks", "app.notes", "app.voice", "app.self_development", "app.config",
)


def imports(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def identifiers(path):
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
    assert {f.name for f in FILES} == {"__init__.py", "adapter.py", "context.py", "occurrence.py", "scheduler.py"}


def test_it_imports_only_what_it_needs_and_nothing_that_could_run_plan_notify_or_write_memory():
    for path in FILES:
        for module in imports(path):
            if not module.startswith("app."):
                continue
            assert module.startswith(ALLOWED), (path.name, module)
            if module in HANDLE_THIS_MODULES:
                continue
            assert not module.startswith(FORBIDDEN), (path.name, module)


def test_the_only_part_of_handle_this_it_reads_is_the_existing_capability_boundary():
    found = set()
    for path in FILES:
        found |= {m for m in imports(path) if m.startswith("app.handle_this")}
    assert found == HANDLE_THIS_MODULES
    assert "DEFAULT_BOUNDARY" in identifiers(PACKAGE / "adapter.py")  # the one existing boundary, not a copy or a new abstraction
    assert not (identifiers(PACKAGE / "adapter.py") & {"CapabilityBoundary"})  # no second boundary is constructed


#: The scheduler's ONE entry point, called by the JobEngine's own tick. It is the only name here that may mean "run periodically", and
#: it is a plain method: the loop, the thread and the timer all stay the engine's.
SCHEDULER_ENTRY_POINT = "tick"


def test_there_is_no_scheduler_loop_thread_timer_or_polling_in_the_package():
    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.While, ast.AsyncFunctionDef, ast.Await))], path.name
        for module in imports(path):
            assert module.split(".")[0] not in {"threading", "sched", "asyncio", "time", "signal", "subprocess", "multiprocessing", "socket"}, (path.name, module)
        forbidden = {"start", "stop", "tick", "poll", "run_forever", "schedule_every", "Thread", "Timer"}
        if path.name == "scheduler.py":
            forbidden -= {SCHEDULER_ENTRY_POINT}  # the one method the JobEngine's tick calls; nothing here waits, sleeps or loops
        assert not (identifiers(path) & forbidden), path.name
        if path.name != "scheduler.py":
            assert SCHEDULER_ENTRY_POINT not in identifiers(path), path.name


def test_the_adapter_never_runs_a_tool_plans_notifies_or_writes_memory():
    used = set().union(*(identifiers(path) for path in FILES))
    for needle in (
        "execute", "ToolRegistry", "JobEngine", "Planner", "plan", "NotificationRouter", "notify", "Notifier", "present", "Presenter",
        "save_fact", "delete_fact", "add_message", "create_conversation", "remember", "MemoryWriteService", "confirm_job", "cancel_job",
        "create_job", "subprocess",
    ):
        assert needle not in used, needle


def test_the_scheduler_is_wired_through_main_alone_onto_the_job_engines_own_tick():
    """Slice 3 wires the scheduler in, through ONE place: main.py builds it and hangs `tick` on the existing JobEngine's tick hook.
    Nothing else names it - not the core, the default tools, the engine, the factory, the orchestration tools or the profile - so no
    chat message or tool can invoke the adapter, and the engine knows nothing of scheduling. (This replaces Slice 2's "not wired in".)"""
    for relative in (
        "app/core.py", "app/tools/defaults.py", "app/orchestration/engine.py", "app/orchestration/factory.py",
        "app/orchestration/tools.py", "app/orchestration/profiles.py", "app/capability.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "scheduled_execution" not in text and "scheduled_jobs" not in text, relative
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "job_engine.add_tick_hook(scheduled_job_scheduler.tick)" in main
    assert "scheduled_job_scheduler.start" not in main and "scheduled_job_scheduler.stop" not in main  # no loop of its own to run or stop
    importers = sorted(
        str(path.relative_to(ROOT / "app")).replace("\\", "/")
        for path in (ROOT / "app").rglob("*.py")
        if "scheduled_execution" not in path.parts
        and re.search(r"^\s*(from|import) app\.scheduled_execution", path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert importers == ["main.py"]


def test_building_the_adapter_and_launching_an_occurrence_starts_no_thread():
    from test_scheduled_execution_adapter import Rig

    before = threading.active_count()
    rig = Rig()
    sj = rig.approved()
    rig.adapter.launch(sj, rig.occurrence(sj))
    assert threading.active_count() == before  # nothing was started in the background


def test_the_run_profile_module_stays_a_pure_reduction_with_no_clock_io_or_other_dependencies():
    path = ROOT / "app" / "orchestration" / "profiles.py"
    assert {m for m in imports(path) if m.startswith("app.")} == {"app.orchestration.base", "app.tools.base"}
    used = identifiers(path)
    for needle in ("now", "datetime", "open", "execute", "threading", "sqlite3", "environ", "getenv", "settings"):
        assert needle not in used, needle


def test_the_only_edits_to_the_job_model_are_the_two_additive_fields_with_safe_defaults():
    from app.orchestration.base import INTERACTIVE_PROFILE, Job

    fields = {f.name: f for f in __import__("dataclasses").fields(Job)}
    assert fields["run_profile"].default == INTERACTIVE_PROFILE == "interactive"
    assert fields["deadline_at"].default is None
