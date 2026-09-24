"""
Rhasspy - entry point.

FastAPI server with a health check endpoint and a /chat endpoint backed by
Jarvis Core, which routes to an AI provider (Gemini for now) through the
AIProvider interface, and persists conversation history and user facts
through a MemoryStore. Since M7, it also starts a background
ReminderScheduler that polls persisted reminders and fires local
notifications for due ones. Since M8, it also starts a background
AutomationEngine that polls persisted automations and runs their
already-permission-checked tool actions when due. Since M9, it also
wires up Google Calendar (OAuth-authorized separately via
`python -m app.calendar.authorize`) and local Rhasspy Tasks - neither
needs a background process of its own. Since M15, it also starts a
background JobEngine that plans and executes goal-driven Jobs (see
app/orchestration/) - a TASK-capability chat message just creates a
PENDING Job and returns immediately; JobEngine does the actual planning/
execution on its own thread, the same way AutomationEngine/
ReminderScheduler already do their own background work. No dashboard,
browser/computer control, or hardware control yet - those come in later
milestones.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.automation.factory import create_default_automation_store
from app.calendar.factory import create_default_calendar_provider
from app.config import settings
from app.context_engine.factory import create_default_context_engine
from app.context_reasoning.factory import create_default_context_reasoner
from app.core import ConversationNotFoundError, core
from app.orchestration.factory import create_default_job_engine, create_default_job_store
from app.providers.base import ProviderError
from app.reminders.factory import create_default_reminder_store, create_default_scheduler
from app.tasks.factory import create_default_task_store
from app.tools.defaults import register_default_tools
from app.world_model.factory import (
    create_default_world_model_service,
    create_default_world_model_store,
)
from app.world_model.sqlite_store import DEFAULT_WORLD_MODEL_DB_PATH

# Application-level wiring: populate the running app's tool registry with
# Jarvis's default tools. JarvisCore's own default (when no registry is
# injected) stays empty - see app/tools/defaults.py.
#
# The reminder/automation/task stores (and the calendar provider) are
# built once here and passed into both register_default_tools (so their
# tools operate on them) and the ReminderScheduler/AutomationEngine below
# (so their background polling sees the exact same persisted rows),
# rather than each independently opening its own SQLite connection to the
# same file. `core.memory` is passed through too, so an automation's
# conditions (app/automation/base.py's FactEqualsCondition) see the same
# stored facts JarvisCore's own DIRECT/SEARCH prompts already use.
_reminder_store = create_default_reminder_store(settings.reminders_db_path)
_automation_store = create_default_automation_store(settings.automations_db_path)
_calendar_provider = create_default_calendar_provider()
_task_store = create_default_task_store(settings.tasks_db_path)
_job_store = create_default_job_store(settings.orchestration_db_path)
automation_engine = register_default_tools(
    core.tools,
    reminder_store=_reminder_store,
    automation_store=_automation_store,
    memory=core.memory,
    calendar_provider=_calendar_provider,
    task_store=_task_store,
    job_store=_job_store,
)

reminder_scheduler = create_default_scheduler(
    _reminder_store, settings.reminder_poll_interval_seconds
)

# JobEngine (M15) isn't built by register_default_tools itself (see that
# function's own docstring) - built here the same way ReminderScheduler
# is, against the exact same `_job_store`/`core.tools` the create_job/
# confirm_job/... tools above already operate on, so the background
# planner/executor and a user's tool calls always see the same
# persisted Jobs.
# `context_provider=core.memory.get_facts` refreshes a Job's context
# immediately before every planning/replanning call (see
# JobEngine._refresh_context) - the same fact source CreateJobTool's own
# `context_provider` already snapshots at creation time, so a Job never
# relies solely on that one-time snapshot going stale.
job_engine = create_default_job_engine(
    _job_store,
    core.tools,
    max_steps=settings.orchestration_max_steps,
    max_total_steps=settings.orchestration_max_total_steps,
    max_retries_per_step=settings.orchestration_max_retries_per_step,
    max_replans=settings.orchestration_max_replans,
    retry_delay_seconds=settings.orchestration_retry_delay_seconds,
    poll_interval_seconds=settings.orchestration_poll_interval_seconds,
    context_provider=core.memory.get_facts,
)

# Personal World Model + Context Engine (M16.1/M16.2) - wired onto the
# already-constructed `core` singleton from outside, the exact same
# "build it here, attach it to core" pattern every store/engine above
# already uses (see e.g. `core.tools` populated by register_default_tools).
# `core.context_engine` defaults to `None` (see app/core.py's own
# JarvisCore.__init__), so this is what actually ACTIVATES the M16.2
# integration for the real running app - without this, `_dispatch` would
# never build a RequestContext/ContextSnapshot at all. No `settings.*`
# field exists for the World Model's db path (deliberately - M16.1's own
# correction: "the World Model should continue receiving db_path
# explicitly through its store/factory APIs," never a config mechanism),
# so `DEFAULT_WORLD_MODEL_DB_PATH` (app/world_model/sqlite_store.py) - the
# one place that literal string is actually defined - is imported and
# passed here explicitly, rather than a second hardcoded copy of it
# living in this file too. `memory=core.memory` mirrors the same "share
# the one real MemoryStore, never open a second connection to the same
# file" convention every store above already follows.
_world_model_store = create_default_world_model_store(DEFAULT_WORLD_MODEL_DB_PATH)
_world_model_service = create_default_world_model_service(_world_model_store)
core.context_engine = create_default_context_engine(_world_model_service, memory=core.memory)

# Context-Aware Reasoning (M16.3) - wired onto `core` the same way M16.2
# just was, immediately above: `core.context_reasoner` defaults to `None`
# (see JarvisCore.__init__), so this is what actually ACTIVATES M16.3 for
# the real running app. Routed through `core.provider` - the same
# AIProviderManager/provider-routing chain every other AI call in this
# app already goes through (app/providers/) - never a second, separately
# configured AI client. See app/context_reasoning/reasoner.py for why a
# request whose ContextSnapshot has nothing relevant never reaches the
# provider at all, keeping this activation backward-compatible with
# every ordinary request the way M16.2's own activation above already is.
core.context_reasoner = create_default_context_reasoner(core.provider)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    reminder_scheduler.start()
    automation_engine.start()
    job_engine.start()
    yield
    job_engine.stop()
    automation_engine.stop()
    reminder_scheduler.stop()


app = FastAPI(title=settings.app_name, lifespan=_lifespan)


@app.get("/health")
def health_check() -> dict:
    """Simple proof that the server is alive and configured correctly."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatSource(BaseModel):
    """One citation backing a SEARCH-path reply - mirrors
    app.search.answer.SearchSource verbatim (title/url/query), never
    modified in transit. Always [] for a DIRECT reply."""

    title: str
    url: str
    query: str


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    sources: list[ChatSource] = []
    #: The id of the Job `handle_chat_with_job_id` just created (M15), or
    #: `None` for every capability except TASK - a real structured field
    #: instead of a client having to parse the reply's prose for it (see
    #: JarvisCore.handle_chat_with_job_id's own docstring).
    job_id: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a user message to Jarvis Core and return its reply.

    Omit conversation_id to start a new conversation; pass back the
    conversation_id from a previous response to continue it with history.
    """
    try:
        reply, conversation_id, sources, job_id = core.handle_chat_with_job_id(
            request.message, request.conversation_id
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return ChatResponse(
        reply=reply,
        conversation_id=conversation_id,
        sources=[
            ChatSource(title=s.title, url=s.url, query=s.query) for s in sources
        ],
        job_id=job_id,
    )
