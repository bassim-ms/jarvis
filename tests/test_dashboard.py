"""
The local web dashboard (app/dashboard/). Real SQLite stores in pytest's temporary folders, the real task tools on a real
ToolRegistry, the real loopback Host guard - and, at the end, the real application (app.main) with its own wiring. No AI provider
is ever called: reading the dashboard must not reach one.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dashboard.api import (
    MAX_TASKS,
    SECURITY_HEADERS,
    STATIC_DIR,
    DashboardSources,
    _is_loopback_origin,
    build_dashboard_router,
)
from app.host_guard import LoopbackHostMiddleware
from app.memory.sqlite_store import SQLiteMemoryStore
from app.notes.sqlite_store import SQLiteNoteStore
from app.orchestration.interactions import InteractionType
from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.reminders.base import parse_recurrence
from app.reminders.sqlite_store import SQLiteReminderStore
from app.tasks.sqlite_store import SQLiteTaskStore
from app.tasks.tools import CancelTaskTool, CompleteTaskTool, CreateTaskTool
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.registry import ToolRegistry

DASHBOARD = {"X-Jarvis-Dashboard": "1"}
PROVIDERS = {
    "order": ["groq", "openrouter", "gemini", "cloudflare", "freellmapi"],
    "providers": [
        {"position": i, "name": name, "display_name": name.title(), "model": "m", "key_present": True,
         "key_status": "present_not_verified", "key_status_text": "key present, not verified", "enabled": True,
         "consecutive_failures": 0, "resting_seconds": 0}
        for i, name in enumerate(["groq", "openrouter", "gemini", "cloudflare", "freellmapi"], start=1)
    ],
}


class Spy(ToolRegistry):
    """The real registry, recording every tool it is asked to run."""

    def __init__(self):
        super().__init__()
        self.ran = []

    def execute(self, name, params=None, confirmed=None):
        self.ran.append((name, dict(params or {})))
        return super().execute(name, params, confirmed)


@pytest.fixture
def world(tmp_path):
    tasks = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    jobs = SQLiteJobStore(str(tmp_path / "orchestration.db"))
    tools = Spy()
    for tool in (CreateTaskTool(tasks), CompleteTaskTool(tasks), CancelTaskTool(tasks)):
        tools.register(tool)
    sources = DashboardSources(
        tasks=tasks,
        reminders=SQLiteReminderStore(str(tmp_path / "reminders.db")),
        notes=SQLiteNoteStore(str(tmp_path / "notes.db")),
        jobs=jobs,
        interactions=SQLiteInteractionStore(jobs),
        tools=tools,
        memory=SQLiteMemoryStore(str(tmp_path / "memory.db")),
        provider_status=lambda: PROVIDERS,
        health=lambda: {"status": "ok", "app": "Jarvis"},
        clock=lambda: datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc),
    )
    app = FastAPI()
    app.add_middleware(LoopbackHostMiddleware)
    app.include_router(build_dashboard_router(sources))
    return sources, TestClient(app, base_url="http://127.0.0.1:8000")


# --- the page --------------------------------------------------------------------------------------------------------------------


def test_the_page_and_its_two_files_are_served_with_strict_security_headers(world):
    _, http = world

    for path, kind in (("/dashboard", "text/html"), ("/dashboard/app.js", "text/javascript"), ("/dashboard/style.css", "text/css")):
        response = http.get(path)
        assert response.status_code == 200 and response.headers["content-type"].startswith(kind)
        for header, value in SECURITY_HEADERS.items():
            assert response.headers[header] == value
    assert "script-src 'self'" in SECURITY_HEADERS["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in SECURITY_HEADERS["Content-Security-Policy"]


def test_the_root_opens_the_dashboard(world):
    _, http = world

    response = http.get("/", follow_redirects=False)

    assert response.status_code == 307 and response.headers["location"] == "/dashboard"


@pytest.mark.parametrize("path", ["/dashboard/index.html", "/dashboard/api.py", "/dashboard/..%2Fapi.py", "/dashboard/%2e%2e%2f__init__.py",
                                  "/dashboard/missing.js"])
def test_nothing_but_the_allow_listed_files_can_be_fetched(world, path):
    _, http = world

    response = http.get(path)

    assert response.status_code == 404
    assert "build_dashboard_router" not in response.text


def test_the_page_loads_nothing_from_outside_this_computer_and_has_no_inline_code():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html, re.IGNORECASE)
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html, re.IGNORECASE)  # every script is an external, same-origin file
    assert not re.search(r"\son[a-z]+\s*=", html, re.IGNORECASE)  # no inline event handlers (the CSP would block them anyway)
    assert "<style" not in html.lower()


def test_the_script_never_writes_data_as_html_or_runs_strings_as_code():
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert forbidden not in script
    assert "textContent" in script
    assert 'rel: "noopener noreferrer"' in script  # source links open safely
    assert "/^https?:\\/\\//i.test" in script  # and only http(s) URLs become links


# --- tasks: through the existing tools, and only from the dashboard page ----------------------------------------------------------


def test_a_task_added_on_the_dashboard_goes_through_the_create_task_tool(world):
    sources, http = world

    response = http.post("/api/tasks", json={"text": "  Buy   PLA filament ", "priority": "high", "due_date": "2026-11-20"},
                         headers=DASHBOARD)

    assert response.status_code == 201
    task = response.json()["task"]
    assert task["text"] == "Buy PLA filament" and task["priority"] == "high" and task["due_date"] == "2026-11-20"
    assert sources.tools.ran == [("create_task", {"text": "Buy PLA filament", "priority": "high", "due_date": "2026-11-20"})]
    assert [t.text for t in sources.tasks.list()] == ["Buy PLA filament"]
    assert http.get("/api/tasks").json()["tasks"][0]["id"] == task["id"]


def test_the_tools_own_validation_answers(world):
    sources, http = world

    response = http.post("/api/tasks", json={"text": "x", "priority": "urgent"}, headers=DASHBOARD)

    assert response.status_code == 400 and "Unknown priority" in response.json()["detail"]
    assert sources.tasks.list() == []


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {}, {"text": "x" * 501}])
def test_a_task_needs_words(world, body):
    sources, http = world

    assert http.post("/api/tasks", json=body, headers=DASHBOARD).status_code in (400, 422)
    assert sources.tasks.list() == []


@pytest.mark.parametrize("headers", [
    {},  # no dashboard header: e.g. a plain form post from anywhere
    {"X-Jarvis-Dashboard": "0"},
    {**DASHBOARD, "Origin": "https://evil.example"},
    {**DASHBOARD, "Origin": "http://127.0.0.1.evil.example"},
    {**DASHBOARD, "Origin": "null"},
    {**DASHBOARD, "Sec-Fetch-Site": "cross-site"},
    {**DASHBOARD, "Sec-Fetch-Site": "same-site"},
])
def test_a_change_from_anywhere_but_the_dashboard_page_is_refused(world, headers):
    sources, http = world
    existing = sources.tasks.create("keep me")

    created = http.post("/api/tasks", json={"text": "planted"}, headers=headers)
    completed = http.post(f"/api/tasks/{existing.id}/complete", json={}, headers=headers)

    assert created.status_code == 403 and completed.status_code == 403
    assert [t.text for t in sources.tasks.list()] == ["keep me"] and sources.tasks.get(existing.id).status.value == "todo"
    assert sources.tools.ran == []


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"])
def test_the_dashboard_page_itself_may_change_things(world, origin):
    _, http = world

    response = http.post("/api/tasks", json={"text": "ok"}, headers={**DASHBOARD, "Origin": origin, "Sec-Fetch-Site": "same-origin"})

    assert response.status_code == 201


@pytest.mark.parametrize("value, loopback", [
    ("http://127.0.0.1:8000", True), ("http://localhost", True), ("https://localhost:8443", True), ("http://[::1]:8000", True),
    ("http://127.0.0.1.evil.example", False), ("https://evil.example", False), ("null", False), ("file://", False),
    ("http://localhost.evil.example", False), ("javascript:alert(1)", False), ("http://[::1", False),
])
def test_is_loopback_origin(value, loopback):
    assert _is_loopback_origin(value) is loopback


def test_done_and_cancel_go_through_their_tools_and_report_what_happened(world):
    sources, http = world
    first, second = sources.tasks.create("Print the bracket"), sources.tasks.create("Order bearings")

    done = http.post(f"/api/tasks/{first.id}/complete", json={}, headers=DASHBOARD)
    cancelled = http.post(f"/api/tasks/{second.id}/cancel", json={}, headers=DASHBOARD)
    again = http.post(f"/api/tasks/{first.id}/complete", json={}, headers=DASHBOARD)
    missing = http.post("/api/tasks/nope/complete", json={}, headers=DASHBOARD)

    assert done.status_code == 200 and done.json()["task"]["status"] == "done"
    assert "Print the bracket" in done.json()["message"]
    assert cancelled.status_code == 200 and cancelled.json()["task"]["status"] == "cancelled"
    assert again.status_code == 400 and "is not open" in again.json()["detail"]
    assert missing.status_code == 400 and "not found" in missing.json()["detail"]
    assert [name for name, _ in sources.tools.ran] == ["complete_task", "cancel_task", "complete_task", "complete_task"]


def test_a_task_action_also_works_without_a_body(world):
    sources, http = world
    task = sources.tasks.create("No body")

    assert http.post(f"/api/tasks/{task.id}/complete", headers=DASHBOARD).status_code == 200


class _NeedsConfirmation(Tool):
    name = "create_task"
    description = "a tool that needs the user's confirmation"
    permission = PermissionLevel.CONFIRM

    def __init__(self):
        self.ran = False

    @property
    def parameters(self):
        return [ToolParameter(name="text", type="string", description="t"),
                ToolParameter(name="priority", type="string", description="p", required=False, default="medium")]

    def execute(self, text, priority="medium"):
        self.ran = True
        return ToolResult.success(data={"id": "x"})


def test_the_dashboard_never_confirms_a_tool_for_you(world):
    sources, http = world
    guarded = _NeedsConfirmation()
    sources.tools._tools["create_task"] = guarded  # the same name, now CONFIRM-level

    response = http.post("/api/tasks", json={"text": "risky"}, headers=DASHBOARD)

    assert response.status_code == 409 and "confirmation" in response.json()["detail"]
    assert guarded.ran is False


def test_tasks_can_be_filtered(world):
    sources, http = world
    open_task, done_task = sources.tasks.create("open"), sources.tasks.create("finished")
    sources.tasks.complete(done_task.id)

    assert [t["text"] for t in http.get("/api/tasks").json()["tasks"]] == ["open"]
    assert [t["text"] for t in http.get("/api/tasks?status=done").json()["tasks"]] == ["finished"]
    assert {t["text"] for t in http.get("/api/tasks?status=all").json()["tasks"]} == {"open", "finished"}
    assert http.get("/api/tasks?status=sideways").status_code == 400


def test_a_long_task_list_is_bounded(world):
    sources, http = world
    for index in range(MAX_TASKS + 5):
        sources.tasks.create(f"task {index}")

    body = http.get("/api/tasks").json()

    assert len(body["tasks"]) == MAX_TASKS and body["total"] == MAX_TASKS + 5


# --- reading everything else ----------------------------------------------------------------------------------------------------


def test_reminders_show_the_active_ones_soonest_first(world):
    sources, http = world
    later = sources.reminders.create("Water the plants", "2026-11-20T18:00:00+04:00", recurrence=parse_recurrence("days"))
    sooner = sources.reminders.create("Call the bank", "2026-11-19T09:00:00+04:00")
    finished = sources.reminders.create("Old one", "2026-11-01T09:00:00+04:00")
    sources.reminders.complete(finished.id)
    paused = sources.reminders.create("Gym", "2026-11-21T07:00:00+04:00")
    sources.reminders.pause(paused.id)

    active = http.get("/api/reminders").json()["reminders"]

    assert [r["text"] for r in active] == ["Call the bank", "Water the plants", "Gym"]
    assert active[1]["repeats"] is True and active[0]["repeats"] is False and active[2]["status"] == "paused"
    assert len(http.get("/api/reminders?status=all").json()["reminders"]) == 4
    assert http.get("/api/reminders?status=weird").status_code == 400
    assert later.id in {r["id"] for r in active} and sooner.id == active[0]["id"]


def test_a_one_off_reminder_is_not_shown_as_repeating(world):
    """Found in the browser check: reminders V2 give a one-off reminder a ONCE schedule, which the first version called "repeats"."""
    from datetime import time

    from app.scheduling.schedule import Schedule, ScheduleKind

    sources, http = world
    once = Schedule(kind=ScheduleKind.ONCE, tz="Asia/Dubai", once_local=datetime(2026, 11, 19, 19, 0))
    daily = Schedule(kind=ScheduleKind.DAILY, tz="Asia/Dubai", at=time(7, 0))
    sources.reminders.create("study physics", "2026-11-19T15:00:00+00:00", schedule=once)
    sources.reminders.create("morning run", "2026-11-19T03:00:00+00:00", schedule=daily)

    shown = {r["text"]: r for r in http.get("/api/reminders").json()["reminders"]}

    assert shown["study physics"]["repeats"] is False and shown["study physics"]["schedule"] == "once"
    assert shown["morning run"]["repeats"] is True and shown["morning run"]["schedule"] == "daily"


def test_notes_are_listed_with_a_preview_and_opened_in_full(world):
    sources, http = world
    note = sources.notes.create("Research: PETG vs PLA", "A" * 1000)

    listed = http.get("/api/notes").json()["notes"]
    opened = http.get(f"/api/notes/{note.id}").json()

    assert listed[0]["title"] == "Research: PETG vs PLA" and len(listed[0]["preview"]) <= 240 and listed[0]["preview"].endswith("...")
    assert opened["content"] == "A" * 1000
    assert http.get("/api/notes/missing").status_code == 404


def test_jobs_are_listed_newest_first_with_their_outcome(world):
    sources, http = world
    old = sources.jobs.create("Plan my week")
    new = sources.jobs.create("Research filament prices " + "x" * 600)

    jobs = http.get("/api/jobs").json()["jobs"]

    assert {job["id"] for job in jobs} == {old.id, new.id}
    assert jobs[0]["updated_at"] >= jobs[1]["updated_at"]
    long_one = next(job for job in jobs if job["id"] == new.id)
    assert len(long_one["goal"]) <= 500 and long_one["goal"].endswith("...")
    assert long_one["status"] == "pending" and long_one["active"] is True


def test_questions_waiting_for_the_user_are_shown_with_their_answer_code(world):
    sources, http = world
    job, question = sources.interactions.create_job_awaiting_input(
        goal="Book a table", context={}, conversation_id=None, prompt="For how many people?")

    waiting = http.get("/api/interactions").json()["interactions"]

    assert len(waiting) == 1
    assert waiting[0]["prompt"] == "For how many people?" and waiting[0]["type"] == "input"
    assert waiting[0]["code"] == question.code and re.fullmatch(r"[a-z0-9]{6}", question.code)
    assert waiting[0]["job_id"] == job.id


def test_a_conversation_can_be_restored(world):
    sources, http = world
    conversation = sources.memory.create_conversation()
    sources.memory.add_message(conversation, "user", "Hello")
    sources.memory.add_message(conversation, "assistant", "Good evening, sir.")

    body = http.get(f"/api/conversations/{conversation}/messages").json()

    assert body["messages"] == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Good evening, sir."}]
    assert http.get("/api/conversations/not-a-conversation/messages").status_code == 404


def test_the_overview_counts_what_is_open_and_shows_the_provider_chain(world):
    sources, http = world
    sources.tasks.create("a")
    done = sources.tasks.create("b")
    sources.tasks.complete(done.id)
    sources.reminders.create("r", (datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    sources.jobs.create("goal")
    sources.interactions.create_job_awaiting_input(goal="g", context={}, conversation_id=None, prompt="Which one?")

    body = http.get("/api/overview").json()

    assert body["health"]["status"] == "ok" and body["generated_at"].startswith("2026-11-18T09:00")
    assert body["counts"] == {"open_tasks": 1, "active_reminders": 1, "active_jobs": 2, "waiting_for_you": 1}
    assert [p["name"] for p in body["providers"]] == ["groq", "openrouter", "gemini", "cloudflare", "freellmapi"]
    assert body["providers"][0]["key_status"] == "present_not_verified"


def test_values_come_back_as_plain_data_never_rewritten(world):
    """The page writes them with textContent; the API must not "sanitise" them into something else either."""
    sources, http = world
    sources.tasks.create('<img src=x onerror="alert(1)"> & "quotes"')

    assert http.get("/api/tasks").json()["tasks"][0]["text"] == '<img src=x onerror="alert(1)"> & "quotes"'


def test_every_dashboard_route_is_loopback_only(world):
    sources, _ = world
    app = FastAPI()
    app.add_middleware(LoopbackHostMiddleware)
    app.include_router(build_dashboard_router(sources))
    remote = TestClient(app, base_url="http://jarvis.example.com")

    for path in ("/dashboard", "/dashboard/app.js", "/api/overview", "/api/tasks", "/api/conversations/x/messages"):
        assert remote.get(path).status_code == 403
    assert remote.post("/api/tasks", json={"text": "x"}, headers=DASHBOARD).status_code == 403


# --- the real application ---------------------------------------------------------------------------------------------------------


def test_the_real_app_serves_the_dashboard_from_its_own_stores_and_tools(monkeypatch):
    from app import main
    from app.providers.base import AIProvider

    class _NeverCalled(AIProvider):
        def generate_reply(self, message: str) -> str:
            raise AssertionError("reading the dashboard must never call an AI provider")

    monkeypatch.setattr(main.core, "provider", _NeverCalled())
    http = TestClient(main.app, base_url="http://127.0.0.1:8000")

    assert http.get("/dashboard").status_code == 200
    for path in ("/api/overview", "/api/tasks", "/api/reminders", "/api/notes", "/api/jobs", "/api/interactions"):
        assert http.get(path).status_code == 200, path
    created = http.post("/api/tasks", json={"text": "dashboard integration check"}, headers=DASHBOARD)
    assert created.status_code == 201
    task_id = created.json()["task"]["id"]
    try:
        assert main._task_store.get(task_id).text == "dashboard integration check"  # the app's own store, via core.tools
        assert main.core.tools.has("create_task")
    finally:
        main._task_store.cancel(task_id)


def test_the_dashboard_files_ship_in_the_portable_package():
    import importlib.util
    import sys

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("jarvis_setup_for_dashboard", root / "installer" / "jarvis_setup.py")
    setup = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = setup
    spec.loader.exec_module(setup)

    shipped = {path.relative_to(root).as_posix() for path in setup.package_files(root)}

    assert {"app/dashboard/api.py", "app/dashboard/static/index.html", "app/dashboard/static/app.js",
            "app/dashboard/static/style.css"} <= shipped
