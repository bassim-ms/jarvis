"""
Tests for POST /chat. Jarvis Core's provider is monkeypatched, so no real
API call is made.
"""

from fastapi.testclient import TestClient

from app import main
from app.providers.base import ProviderError

client = TestClient(main.app)


def test_chat_returns_provider_reply(monkeypatch):
    monkeypatch.setattr(main.core, "handle_chat", lambda message: f"echo: {message}")

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello"}


def test_chat_returns_502_when_provider_fails(monkeypatch):
    def raise_error(message):
        raise ProviderError("boom")

    monkeypatch.setattr(main.core, "handle_chat", raise_error)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "boom"


def test_chat_requires_message_field():
    response = client.post("/chat", json={})

    assert response.status_code == 422
