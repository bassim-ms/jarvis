"""
Jarvis - entry point.

This is currently just the project foundation: a FastAPI server with a
health check endpoint. No AI, no dashboard, no hardware control yet -
those come in later milestones.
"""

from fastapi import FastAPI

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
