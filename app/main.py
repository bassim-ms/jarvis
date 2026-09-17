"""
Jarvis - entry point.

FastAPI server with a health check endpoint and a /chat endpoint backed by
Jarvis Core, which routes to an AI provider (Gemini for now) through the
AIProvider interface, and persists conversation history and user facts
through a MemoryStore. No dashboard, agents, voice, or hardware control yet
- those come in later milestones.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.core import ConversationNotFoundError, core
from app.providers.base import ProviderError
from app.tools.defaults import register_default_tools

app = FastAPI(title=settings.app_name)

# Application-level wiring: populate the running app's tool registry with
# Jarvis's default tools. JarvisCore's own default (when no registry is
# injected) stays empty - see app/tools/defaults.py.
register_default_tools(core.tools)


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
