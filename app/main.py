"""
Jarvis - entry point.

FastAPI server with a health check endpoint and a /chat endpoint backed by
Jarvis Core, which routes to an AI provider (Gemini for now) through the
AIProvider interface, and persists conversation history and user facts
through a MemoryStore. Since M7, it also starts a background
ReminderScheduler that polls persisted reminders and fires local
notifications for due ones. No dashboard, agents, voice, or hardware
control yet - those come in later milestones.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.core import ConversationNotFoundError, core
from app.providers.base import ProviderError
from app.reminders.factory import create_default_reminder_store, create_default_scheduler
from app.tools.defaults import register_default_tools

# Application-level wiring: populate the running app's tool registry with
# Jarvis's default tools. JarvisCore's own default (when no registry is
# injected) stays empty - see app/tools/defaults.py.
#
# The reminder store is built once here and passed into both
# register_default_tools (so create/list/complete/cancel_reminder operate
# on it) and the ReminderScheduler below (so its background polling sees
# the exact same persisted rows), rather than each independently opening
# its own SQLite connection to the same file.
_reminder_store = create_default_reminder_store(settings.reminders_db_path)
register_default_tools(core.tools, reminder_store=_reminder_store)

reminder_scheduler = create_default_scheduler(
    _reminder_store, settings.reminder_poll_interval_seconds
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    reminder_scheduler.start()
    yield
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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a user message to Jarvis Core and return its reply.

    Omit conversation_id to start a new conversation; pass back the
    conversation_id from a previous response to continue it with history.
    """
    try:
        reply, conversation_id, sources = core.handle_chat(
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
    )
