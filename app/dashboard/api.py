"""
The dashboard's routes (see app/dashboard/__init__.py for the list). Everything is built from the stores and the ToolRegistry the
running app already has - handed in through `DashboardSources` - so the dashboard shows exactly what chat, the schedulers and the
JobEngine see, and changes data only through the same tools chat uses.

Safety:
  * Loopback only: the app-wide LoopbackHostMiddleware (app/host_guard.py) answers only requests addressed to this computer.
  * A change (POST) must carry the `X-Jarvis-Dashboard: 1` header, and - when the browser says where it comes from - come from this
    computer's own page. A web page on another site cannot add that header without the browser first asking this server (a CORS
    preflight), which it never allows; that blocks cross-site request forgery against the local API.
  * Changes go through `ToolRegistry.execute`, so each tool's permission level applies: a CONFIRM / DANGEROUS tool is never run
    from here (it would come back "confirmation required").
  * The page sends a strict Content-Security-Policy (only its own scripts, no inline code, no framing) and its script writes
    every value with `textContent`, never as HTML.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.memory.base import MemoryStore
from app.notes.base import NoteStore
from app.orchestration.interactions import InteractionStore
from app.reminders.base import ReminderStatus, ReminderStore
from app.scheduling.schedule import ScheduleKind
from app.tasks.base import Task, TaskStatus, TaskStore
from app.tools.base import ToolResultStatus
from app.tools.registry import ToolRegistry

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: The only files the dashboard serves - an allow-list, so no request path can reach anything else on disk.
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

DASHBOARD_HEADER = "x-jarvis-dashboard"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

MAX_TASKS = 200
MAX_JOBS = 50
MAX_NOTES = 50
MAX_MESSAGES = 200
NOTE_PREVIEW_CHARS = 240
JOB_TEXT_CHARS = 500
ACTIVE_REMINDER_STATUSES = (ReminderStatus.PENDING, ReminderStatus.TRIGGERED, ReminderStatus.PAUSED)
ACTIVE_JOB_STATUSES = {"pending", "planning", "awaiting_confirmation", "awaiting_input", "running", "paused"}


@dataclass
class DashboardSources:
    """What the dashboard reads and acts through - the running app's own objects (see app/main.py)."""

    tasks: TaskStore
    reminders: ReminderStore
    notes: NoteStore
    jobs: Any  # JobStore: `.list(status=None)` and `.get(job_id)`
    interactions: InteractionStore
    tools: ToolRegistry
    memory: MemoryStore
    provider_status: Callable[[], dict]
    health: Callable[[], dict]
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


class NewTask(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    priority: str = "medium"
    due_date: str | None = Field(default=None, max_length=64)


class TaskAction(BaseModel):
    """The (empty) JSON body of a task action: a body keeps every change a JSON request."""


def _is_loopback_origin(value: str) -> bool:
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (parts.hostname or "") in _LOOPBACK_HOSTS


def require_dashboard_request(request: Request) -> None:
    """Refuse a change that does not come from the dashboard page on this computer (see the module docstring)."""
    if request.headers.get(DASHBOARD_HEADER) != "1":
        raise HTTPException(status_code=403, detail="dashboard requests only")
    origin = request.headers.get("origin")
    if origin is not None and not _is_loopback_origin(origin):
        raise HTTPException(status_code=403, detail="cross-site request refused")
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        raise HTTPException(status_code=403, detail="cross-site request refused")


def _task(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "text": task.text,
        "status": task.status.value,
        "priority": task.priority.value,
        "due_date": task.due_date,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _short(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def _tool_outcome(sources: DashboardSources, name: str, params: dict[str, Any]):
    """Run one of the EXISTING tools through the registry (its permission level applies) and turn its result into HTTP."""
    result = sources.tools.execute(name, params)
    if result.status is ToolResultStatus.SUCCESS:
        return result
    if result.status is ToolResultStatus.CONFIRMATION_REQUIRED:
        raise HTTPException(status_code=409, detail=f"'{name}' needs your confirmation in chat; the dashboard never confirms for you")
    raise HTTPException(status_code=400, detail=result.message or f"'{name}' failed")


def build_dashboard_router(sources: DashboardSources) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/dashboard", status_code=307)

    @router.get("/dashboard", include_in_schema=False)
    def dashboard_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8", headers=SECURITY_HEADERS)

    @router.get("/dashboard/{name}", include_in_schema=False)
    def dashboard_file(name: str) -> FileResponse:
        if name not in STATIC_FILES:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(STATIC_DIR / name, media_type=STATIC_FILES[name], headers=SECURITY_HEADERS)

    @router.get("/api/overview")
    def overview() -> dict:
        status = sources.provider_status()
        providers = [
            {key: entry.get(key) for key in ("position", "name", "display_name", "model", "key_status", "key_status_text",
                                              "enabled", "resting_seconds")}
            for entry in status.get("providers", [])
        ]
        jobs = sources.jobs.list()
        return {
            "health": sources.health(),
            "generated_at": sources.clock().isoformat(),
            "providers": providers,
            "counts": {
                "open_tasks": len(sources.tasks.list(status=TaskStatus.TODO)),
                "active_reminders": sum(len(sources.reminders.list(status=s)) for s in ACTIVE_REMINDER_STATUSES),
                "active_jobs": sum(1 for job in jobs if _value(job.status) in ACTIVE_JOB_STATUSES),
                "waiting_for_you": len(sources.interactions.list_pending()),
            },
        }

    @router.get("/api/tasks")
    def list_tasks(status: str = "todo") -> dict:
        if status == "all":
            tasks = sources.tasks.list()
        else:
            try:
                tasks = sources.tasks.list(status=TaskStatus(status))
            except ValueError:
                raise HTTPException(status_code=400, detail="status must be todo, done, cancelled or all")
        return {"tasks": [_task(task) for task in tasks[:MAX_TASKS]], "total": len(tasks)}

    @router.post("/api/tasks", status_code=201, dependencies=[Depends(require_dashboard_request)])
    def create_task(body: NewTask) -> dict:
        text = " ".join(body.text.split())
        if not text:
            raise HTTPException(status_code=400, detail="a task needs some words")
        params: dict[str, Any] = {"text": text, "priority": body.priority}
        if body.due_date:
            params["due_date"] = body.due_date.strip()
        result = _tool_outcome(sources, "create_task", params)
        task = sources.tasks.get(result.data["id"]) if isinstance(result.data, dict) else None
        return {"task": _task(task) if task else result.data, "message": result.message}

    def _task_action(task_id: str, tool: str) -> dict:
        result = _tool_outcome(sources, tool, {"task_id": task_id})
        task = sources.tasks.get(task_id)
        return {"task": _task(task) if task else None, "message": result.message}

    @router.post("/api/tasks/{task_id}/complete", dependencies=[Depends(require_dashboard_request)])
    def complete_task(task_id: str, body: TaskAction | None = None) -> dict:
        return _task_action(task_id, "complete_task")

    @router.post("/api/tasks/{task_id}/cancel", dependencies=[Depends(require_dashboard_request)])
    def cancel_task(task_id: str, body: TaskAction | None = None) -> dict:
        return _task_action(task_id, "cancel_task")

    @router.get("/api/reminders")
    def list_reminders(status: str = "active") -> dict:
        if status == "active":
            reminders = [r for s in ACTIVE_REMINDER_STATUSES for r in sources.reminders.list(status=s)]
        elif status == "all":
            reminders = sources.reminders.list()
        else:
            raise HTTPException(status_code=400, detail="status must be active or all")
        reminders.sort(key=lambda r: r.due_at)
        return {"reminders": [
            {
                "id": r.id,
                "text": r.text,
                "due_at": r.due_at,
                "status": r.status.value,
                # A one-off reminder has a ONCE schedule (reminders V2) - only a repeating schedule or a legacy recurrence repeats.
                "repeats": r.recurrence is not None or (r.schedule is not None and r.schedule.kind is not ScheduleKind.ONCE),
                "schedule": r.schedule.kind.value if r.schedule is not None else ("interval" if r.recurrence else "once"),
                "timezone": r.timezone,
            }
            for r in reminders
        ]}

    @router.get("/api/notes")
    def list_notes(limit: int = 20) -> dict:
        notes = sources.notes.list(limit=max(1, min(limit, MAX_NOTES)))
        return {"notes": [
            {"id": n.id, "title": n.title, "preview": _short(n.content, NOTE_PREVIEW_CHARS), "created_at": n.created_at,
             "source_job_id": n.source_job_id}
            for n in notes
        ]}

    @router.get("/api/notes/{note_id}")
    def get_note(note_id: str) -> dict:
        note = sources.notes.get(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="no such note")
        return {"id": note.id, "title": note.title, "content": note.content, "created_at": note.created_at,
                "source_job_id": note.source_job_id}

    @router.get("/api/jobs")
    def list_jobs(limit: int = 20) -> dict:
        jobs = sorted(sources.jobs.list(), key=lambda job: job.updated_at, reverse=True)[: max(1, min(limit, MAX_JOBS))]
        return {"jobs": [
            {
                "id": job.id,
                "goal": _short(job.goal, JOB_TEXT_CHARS),
                "status": _value(job.status),
                "active": _value(job.status) in ACTIVE_JOB_STATUSES,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "result": _short(job.final_result, JOB_TEXT_CHARS),
                "error": _short(job.error, JOB_TEXT_CHARS),
                "conversation_id": job.conversation_id,
            }
            for job in jobs
        ]}

    @router.get("/api/interactions")
    def list_interactions() -> dict:
        pending = sorted(sources.interactions.list_pending(), key=lambda item: item.created_at)
        return {"interactions": [
            {
                "id": item.id,
                "job_id": item.job_id,
                "type": _value(item.type),
                "prompt": item.prompt,
                "code": item.code,
                "created_at": item.created_at,
                "expires_at": item.expires_at,
            }
            for item in pending
        ]}

    @router.get("/api/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: str) -> dict:
        if not sources.memory.conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="no such conversation")
        messages = sources.memory.get_messages(conversation_id)[-MAX_MESSAGES:]
        return {"conversation_id": conversation_id,
                "messages": [{"role": m.role, "content": m.content} for m in messages]}

    return router
