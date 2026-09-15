"""
Jarvis - entry point.

FastAPI server with a health check endpoint and a /chat endpoint backed by
Gemini. No dashboard, memory, agents, voice, or hardware control yet -
those come in later milestones.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ai_service import GeminiError, ask_gemini
from app.config import settings

app = FastAPI(title=settings.app_name)


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


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a user message to Gemini and return its reply."""
    try:
        reply = ask_gemini(request.message)
    except GeminiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return ChatResponse(reply=reply)
