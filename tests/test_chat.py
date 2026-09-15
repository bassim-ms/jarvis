"""
Tests for POST /chat. Gemini is monkeypatched, so no real API call is made.
"""

from fastapi.testclient import TestClient

from app import main
from app.ai_service import GeminiError

client = TestClient(main.app)


def test_chat_returns_gemini_reply(monkeypatch):
    monkeypatch.setattr(main, "ask_gemini", lambda message: f"echo: {message}")

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello"}


def test_chat_returns_502_when_gemini_fails(monkeypatch):
    def raise_error(message):
        raise GeminiError("boom")

    monkeypatch.setattr(main, "ask_gemini", raise_error)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "boom"


def test_chat_requires_message_field():
    response = client.post("/chat", json={})

    assert response.status_code == 422
