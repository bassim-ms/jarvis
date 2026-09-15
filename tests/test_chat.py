"""
Tests for POST /chat. Jarvis Core's provider is monkeypatched, so no real
API call is made.
"""

from fastapi.testclient import TestClient

from app import main
from app.providers.base import ProviderError

client = TestClient(main.app)


def test_chat_returns_provider_reply(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return f"echo: {message}", conversation_id or "conv-1"

    monkeypatch.setattr(main.core, "handle_chat", fake_handle_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello", "conversation_id": "conv-1"}


def test_chat_returns_502_when_provider_fails(monkeypatch):
    def raise_error(message, conversation_id=None):
        raise ProviderError("boom")

    monkeypatch.setattr(main.core, "handle_chat", raise_error)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "boom"


def test_chat_requires_message_field():
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_passes_conversation_id_through(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return f"echo: {message}", conversation_id

    monkeypatch.setattr(main.core, "handle_chat", fake_handle_chat)

    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": "abc123"}
    )

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello", "conversation_id": "abc123"}


def test_chat_returns_404_for_unknown_conversation_id():
    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": "does-not-exist"}
    )

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]
