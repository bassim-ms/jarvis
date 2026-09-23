"""
Tests for POST /chat. Jarvis Core's provider is monkeypatched, so no real
API call is made.

Since M15, the endpoint calls `core.handle_chat_with_job_id` (a 4-tuple:
reply, conversation_id, sources, job_id) rather than `core.handle_chat`
(still a 3-tuple, unchanged) - see JarvisCore.handle_chat_with_job_id's
own docstring for why. Every fake here returns the 4-tuple to match.
"""

from fastapi.testclient import TestClient

from app import main
from app.providers.base import ProviderError
from app.search.answer import SearchSource

client = TestClient(main.app)


def test_chat_returns_provider_reply(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return f"echo: {message}", conversation_id or "conv-1", [], None

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", fake_handle_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "reply": "echo: hello",
        "conversation_id": "conv-1",
        "sources": [],
        "job_id": None,
    }


def test_chat_returns_502_when_provider_fails(monkeypatch):
    def raise_error(message, conversation_id=None):
        raise ProviderError("boom")

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", raise_error)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "boom"


def test_chat_requires_message_field():
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_passes_conversation_id_through(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return f"echo: {message}", conversation_id, [], None

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", fake_handle_chat)

    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": "abc123"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "echo: hello",
        "conversation_id": "abc123",
        "sources": [],
        "job_id": None,
    }


def test_chat_returns_404_for_unknown_conversation_id():
    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": "does-not-exist"}
    )

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


# --- M6.11: search sources exposed through /chat -----------------------------


def test_chat_exposes_search_sources_with_title_url_and_query_preserved(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return (
            "grounded answer",
            conversation_id or "conv-1",
            [
                SearchSource(
                    title="Python.org", url="https://python.org", query="python"
                ),
                SearchSource(
                    title="Python Docs",
                    url="https://docs.python.org",
                    query="python",
                ),
            ],
            None,
        )

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", fake_handle_chat)

    response = client.post("/chat", json={"message": "what's the latest on python?"})

    assert response.status_code == 200
    assert response.json() == {
        "reply": "grounded answer",
        "conversation_id": "conv-1",
        "sources": [
            {"title": "Python.org", "url": "https://python.org", "query": "python"},
            {
                "title": "Python Docs",
                "url": "https://docs.python.org",
                "query": "python",
            },
        ],
        "job_id": None,
    }


def test_chat_omits_sources_for_a_direct_reply(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return "direct reply", conversation_id or "conv-1", [], None

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", fake_handle_chat)

    response = client.post("/chat", json={"message": "what is quantum entanglement?"})

    assert response.status_code == 200
    assert response.json()["sources"] == []


# --- M15: job_id exposed structurally through /chat --------------------------


def test_chat_exposes_job_id_for_a_task_reply(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return "Started task 'abc-123'...", conversation_id or "conv-1", [], "abc-123"

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", fake_handle_chat)

    response = client.post("/chat", json={"message": "prepare a report"})

    assert response.status_code == 200
    assert response.json()["job_id"] == "abc-123"


def test_chat_job_id_is_none_for_a_non_task_reply(monkeypatch):
    def fake_handle_chat(message, conversation_id=None):
        return "direct reply", conversation_id or "conv-1", [], None

    monkeypatch.setattr(main.core, "handle_chat_with_job_id", fake_handle_chat)

    response = client.post("/chat", json={"message": "what is quantum entanglement?"})

    assert response.status_code == 200
    assert response.json()["job_id"] is None
