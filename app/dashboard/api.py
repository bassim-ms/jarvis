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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.calendar.base import CalendarProviderError
from app.diagnostics.backup import BackupError
from app.memory.base import MemoryStore
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

#: dashboard action -> the EXISTING tool that performs it (each runs through ToolRegistry.execute, so its permission level applies).
REMINDER_ACTIONS = {"pause": "pause_reminder", "resume": "resume_reminder", "cancel": "cancel_reminder",
                    "complete": "complete_reminder"}
JOB_ACTIONS = {"pause": "pause_job", "resume": "resume_job", "cancel": "cancel_job"}
AUTOMATION_ACTIONS = {"enable": "enable_automation", "disable": "disable_automation", "cancel": "cancel_automation"}

CALENDAR_CACHE_SECONDS = 60  # the 15 s panel refresh must not ask Google every time
CALENDAR_MAX_DAYS = 31
MAX_EVENTS = 50
MAX_JOB_EVENTS = 40
STEP_TEXT_CHARS = 300
CALENDAR_NOT_CONNECTED = "Google Calendar is not connected yet - run: python -m app.calendar.authorize"
CALENDAR_UNAVAILABLE = "Google Calendar could not be read just now."


@dataclass
class DashboardSources:
    """What the dashboard reads and acts through - the running app's own objects (see app/main.py)."""

    tasks: TaskStore
    reminders: ReminderStore
    notes: Any  # NoteStore: .list(limit=) / .get(id) - read only; the dashboard never turns a note into memory
    jobs: Any  # JobStore: `.list(status=None)` and `.get(job_id)`
    interactions: InteractionStore
    tools: ToolRegistry
    memory: MemoryStore
    provider_status: Callable[[], dict]
    health: Callable[[], dict]
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    # Slice 2 - optional, so a dashboard without them still works (their panels say "not available"):
    automations: Any = None  # AutomationStore: .list() / .get_executions(id)
    calendar: Any = None  # CalendarProvider: .list_events(time_min, time_max, max_results)
    world_model: Any = None  # WorldModelService (its .store lists every entity)
    memory_candidates: Callable[[], list] | None = None  # MemoryWriteService.candidates_for_context - staged, unconfirmed
    diagnostics: Any = None  # app.diagnostics.Diagnostics - health report, verified backups (spec 452, 456-458, 460)
    # The memory write policy (MemoryWriteService): the ONLY way the dashboard changes memory - list what is kept with its
    # provenance, and forget one exact item in two confirmed steps (spec 93, 104). The dashboard never deletes memory itself.
    memory_policy: Any = None


class NewTask(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    priority: str = "medium"
    due_date: str | None = Field(default=None, max_length=64)


class TaskAction(BaseModel):
    """The (empty) JSON body of a task action: a body keeps every change a JSON request."""


class ForgetPick(BaseModel):
    kind: str = Field(min_length=1, max_length=20)
    id: str = Field(min_length=1, max_length=300)


class ForgetConfirm(BaseModel):
    token: str = Field(min_length=1, max_length=100)


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
                                              "enabled", "resting_seconds", "stats")}
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

    # --- Slice 2: acting on reminders, Jobs and automations through their EXISTING tools -------------------------------------

    def _act(actions: dict[str, str], action: str, id_param: str, item_id: str) -> dict:
        tool = actions.get(action)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"unknown action: {action}")
        result = _tool_outcome(sources, tool, {id_param: item_id})
        return {"message": result.message or "Done.", "data": result.data if isinstance(result.data, dict) else None}

    @router.post("/api/reminders/{reminder_id}/{action}", dependencies=[Depends(require_dashboard_request)])
    def reminder_action(reminder_id: str, action: str, body: TaskAction | None = None) -> dict:
        return _act(REMINDER_ACTIONS, action, "reminder_id", reminder_id)

    @router.post("/api/jobs/{job_id}/{action}", dependencies=[Depends(require_dashboard_request)])
    def job_action(job_id: str, action: str, body: TaskAction | None = None) -> dict:
        return _act(JOB_ACTIONS, action, "job_id", job_id)

    @router.post("/api/automations/{automation_id}/{action}", dependencies=[Depends(require_dashboard_request)])
    def automation_action(automation_id: str, action: str, body: TaskAction | None = None) -> dict:
        if sources.automations is None:
            raise HTTPException(status_code=404, detail="automations are not available")
        return _act(AUTOMATION_ACTIONS, action, "automation_id", automation_id)

    @router.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict:
        job = sources.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such Job")
        steps = job.plan.steps if job.plan else []
        return {
            "id": job.id,
            "goal": job.goal,
            "status": _value(job.status),
            "active": _value(job.status) in ACTIVE_JOB_STATUSES,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "result": job.final_result,
            "error": job.error,
            "confirmation_message": job.confirmation_message,
            "replan_count": job.replan_count,
            "steps": [
                {
                    "id": step.id,
                    "tool": step.tool_name,
                    "status": _value(step.status),
                    "attempts": step.attempts,
                    "depends_on": list(step.depends_on or []),
                    "arguments": _short(str(step.arguments), STEP_TEXT_CHARS) if step.arguments else "",
                    "result": _short(None if step.result is None else str(step.result), STEP_TEXT_CHARS),
                    "error": _short(step.error, STEP_TEXT_CHARS),
                }
                for step in steps
            ],
            "events": [
                {"at": event.timestamp, "kind": event.kind, "detail": _short(event.detail, STEP_TEXT_CHARS), "step_id": event.step_id}
                for event in job.events[-MAX_JOB_EVENTS:]
            ],
        }

    # --- Slice 2: automations, calendar, memory and the World Model (read-only) ---------------------------------------------

    @router.get("/api/automations")
    def list_automations() -> dict:
        if sources.automations is None:
            return {"available": False, "automations": []}
        items = []
        for automation in sources.automations.list():
            runs = sources.automations.get_executions(automation.id)
            last = max(runs, key=lambda run: run.executed_at) if runs else None
            trigger = automation.trigger
            every = None
            if trigger.recurrence is not None:
                every = f"{trigger.recurrence.interval} {_value(trigger.recurrence.unit)}"
            items.append({
                "id": automation.id,
                "name": automation.name,
                "status": _value(automation.status),
                "trigger": _value(trigger.type),
                "every": every,
                "action": automation.action.tool_name,
                "next_run_at": automation.next_run_at,
                "runs": len(runs),
                "last_run": None if last is None else {"at": last.executed_at, "outcome": _value(last.outcome),
                                                       "detail": _short(last.detail, STEP_TEXT_CHARS)},
            })
        order = {"enabled": 0, "disabled": 1, "cancelled": 2}
        items.sort(key=lambda item: (order.get(item["status"], 3), item["next_run_at"] or "~", item["name"].lower()))
        return {"available": True, "automations": items}

    calendar_cache: dict[str, Any] = {}

    @router.get("/api/calendar")
    def calendar(days: int = 7, refresh: bool = False) -> dict:
        if sources.calendar is None:
            return {"connected": False, "events": [], "message": "No calendar is configured."}
        days = max(1, min(days, CALENDAR_MAX_DAYS))
        now = sources.clock()
        cached = calendar_cache.get(days)
        if cached and not refresh and (now - cached["at"]).total_seconds() < CALENDAR_CACHE_SECONDS:
            return cached["data"]
        try:
            events = sources.calendar.list_events(time_min=now.isoformat(), time_max=(now + timedelta(days=days)).isoformat(),
                                                  max_results=MAX_EVENTS)
            data = {"connected": True, "days": days, "fetched_at": now.isoformat(), "events": [
                {"id": e.id, "title": e.title, "start": e.start, "end": e.end, "location": e.location} for e in events
            ]}
        except CalendarProviderError as exc:
            # Never the provider's own text: it can carry a vendor response. A fixed sentence says what to do.
            message = CALENDAR_NOT_CONNECTED if "not authorized" in str(exc).lower() else CALENDAR_UNAVAILABLE
            data = {"connected": False, "days": days, "fetched_at": now.isoformat(), "events": [], "message": message}
        calendar_cache[days] = {"at": now, "data": data}
        return data

    @router.get("/api/memory")
    def memory() -> dict:
        facts = sources.memory.get_facts()
        preferences = sources.world_model.get_preferences() if sources.world_model is not None else []
        candidates = sources.memory_candidates() if sources.memory_candidates is not None else []
        return {
            "facts": [{"key": key, "value": value} for key, value in sorted(facts.items())],
            "preferences": [
                {"id": p.id, "key": p.key, "value": p.value, "category": p.category, "source": _value(p.source),
                 "confidence": p.confidence, "updated_at": p.updated_at}
                for p in sorted(preferences, key=lambda p: (p.category, p.key))
            ],
            # Staged by the memory policy, NOT confirmed: shown so the user can see what Jarvis is unsure about.
            "candidates": [
                {"id": c.id, "kind": _value(c.kind), "key": c.key, "value": c.value, "confidence": c.confidence,
                 "created_at": c.created_at}
                for c in candidates
            ],
            # What the user asked Jarvis to keep, worded as Jarvis keeps it, with provenance (spec 93) - and the audit history.
            "remembered": sources.memory_policy.remembered() if sources.memory_policy is not None else [],
            "history": sources.memory_policy.history(30) if sources.memory_policy is not None else [],
        }

    # Forgetting ONE exact item, in two steps owned by the memory policy (spec 104): prepare names the item and issues a
    # single-use token; only confirm with that token forgets it - through the policy's own deletion, audit scrub and record.
    @router.post("/api/memory/forget/prepare", dependencies=[Depends(require_dashboard_request)])
    def memory_forget_prepare(body: ForgetPick) -> dict:
        if sources.memory_policy is None:
            raise HTTPException(status_code=404, detail="memory editing is not available")
        prepared = sources.memory_policy.prepare_forget(body.kind, body.id)
        if prepared is None:
            raise HTTPException(status_code=404, detail="That item is not kept - nothing to forget.")
        return prepared

    @router.post("/api/memory/forget/confirm", dependencies=[Depends(require_dashboard_request)])
    def memory_forget_confirm(body: ForgetConfirm) -> dict:
        if sources.memory_policy is None:
            raise HTTPException(status_code=404, detail="memory editing is not available")
        result = sources.memory_policy.confirm_forget(body.token)
        if not result["forgotten"]:
            raise HTTPException(status_code=409, detail=result["message"])
        return result

    @router.get("/api/world")
    def world() -> dict:
        if sources.world_model is None:
            return {"available": False, "projects": [], "commitments": [], "blockers": [], "events": []}
        store = sources.world_model.store
        project_order = {"active": 0, "paused": 1, "completed": 2, "archived": 3}
        projects = sorted(store.list_projects(), key=lambda p: (project_order.get(_value(p.status), 4), p.name.lower()))
        goal_order = {"active": 0, "paused": 1, "achieved": 2, "abandoned": 3}
        out_projects = []
        for project in projects:
            goals = sorted(store.list_goals(project_id=project.id),
                           key=lambda g: (goal_order.get(_value(g.status), 4), g.target_date or "~", g.title.lower()))
            out_projects.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": _value(project.status),
                "priority": _value(project.priority),
                "linked_tasks": len(store.list_task_links(project_id=project.id)),
                "goals": [
                    {"id": g.id, "title": g.title, "description": g.description, "status": _value(g.status),
                     "priority": _value(g.priority), "target_date": g.target_date,
                     "linked_tasks": len(store.list_task_links(goal_id=g.id))}
                    for g in goals
                ],
            })
        now = sources.clock().isoformat()
        return {
            "available": True,
            "projects": out_projects,
            "commitments": [
                {"id": c.id, "title": c.title, "status": _value(c.status), "priority": _value(c.priority), "due_at": c.due_at}
                for c in sorted(sources.world_model.get_open_commitments(), key=lambda c: c.due_at or "~")
            ],
            "blockers": [
                {"id": b.id, "title": b.title, "description": b.description, "severity": _value(b.severity),
                 "detected_at": b.detected_at}
                for b in sources.world_model.get_active_blockers()
            ],
            "events": [
                {"id": e.id, "title": e.title, "starts_at": e.starts_at, "ends_at": e.ends_at, "location": e.location}
                for e in sources.world_model.get_upcoming_events(after=now)[:MAX_EVENTS]
            ],
        }

    # --- System: health report and verified backups (app/diagnostics) ----------------------------------------------------

    def _diagnostics():
        if sources.diagnostics is None:
            raise HTTPException(status_code=404, detail="diagnostics are not available")
        return sources.diagnostics

    @router.get("/api/system")
    def system_report() -> dict:
        if sources.diagnostics is None:
            return {"available": False}
        return {"available": True, **sources.diagnostics.report()}

    @router.get("/api/system/backups")
    def system_backups() -> dict:
        return {"backups": _diagnostics().backups()}

    @router.post("/api/system/backup", dependencies=[Depends(require_dashboard_request)])
    def system_backup(body: TaskAction | None = None) -> dict:
        try:
            return _diagnostics().backup()
        except BackupError as exc:
            raise HTTPException(status_code=500, detail=f"The backup failed and was not kept: {exc}")

    @router.post("/api/system/backups/{name}/verify", dependencies=[Depends(require_dashboard_request)])
    def system_verify(name: str, body: TaskAction | None = None) -> dict:
        try:
            return _diagnostics().verify(name)
        except ValueError:
            raise HTTPException(status_code=400, detail="not a backup name")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="no such backup")

    @router.get("/api/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: str) -> dict:
        if not sources.memory.conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="no such conversation")
        messages = sources.memory.get_messages(conversation_id)[-MAX_MESSAGES:]
        return {"conversation_id": conversation_id,
                "messages": [{"role": m.role, "content": m.content} for m in messages]}

    return router
