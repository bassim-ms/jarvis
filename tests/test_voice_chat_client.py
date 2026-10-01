"""
Tests for ChatClient (app/voice/chat_client.py). httpx calls are
monkeypatched - no real network call to `/chat` is ever made, the same
convention test_gemini_provider.py/test_searxng_provider.py already use.
"""

import httpx
import pytest

from app.voice import chat_client as chat_client_module
from app.voice.chat_client import ChatClient, ChatClientError


def test_send_returns_reply_and_conversation_id(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(
            200,
            json={"reply": "hi there", "conversation_id": "conv-1", "sources": []},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    client = ChatClient(base_url="http://127.0.0.1:8000")

    reply, conversation_id = client.send("hello", None)

    assert reply == "hi there"
    assert conversation_id == "conv-1"
    assert captured["url"] == "http://127.0.0.1:8000/chat"
    assert captured["json"] == {"message": "hello", "conversation_id": None}


def test_send_passes_an_existing_conversation_id_through(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return httpx.Response(
            200,
            json={"reply": "ok", "conversation_id": "conv-1"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    client = ChatClient(base_url="http://127.0.0.1:8000")

    client.send("follow up", "conv-1")

    assert captured["json"] == {"message": "follow up", "conversation_id": "conv-1"}


def test_send_strips_a_trailing_slash_from_base_url(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        return httpx.Response(
            200, json={"reply": "ok", "conversation_id": "c"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    ChatClient(base_url="http://127.0.0.1:8000/").send("hi", None)

    assert captured["url"] == "http://127.0.0.1:8000/chat"


def test_send_raises_on_connection_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    client = ChatClient(base_url="http://127.0.0.1:8000")

    with pytest.raises(ChatClientError, match="Failed to reach Jarvis"):
        client.send("hello", None)


def test_send_raises_on_timeout(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    client = ChatClient(base_url="http://127.0.0.1:8000")

    with pytest.raises(ChatClientError):
        client.send("hello", None)


def test_send_raises_on_http_error_status(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(
            502, json={"detail": "boom"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    client = ChatClient(base_url="http://127.0.0.1:8000")

    with pytest.raises(ChatClientError):
        client.send("hello", None)


def test_send_raises_on_unexpected_response_shape(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(200, json={"unexpected": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(chat_client_module.httpx, "post", fake_post)
    client = ChatClient(base_url="http://127.0.0.1:8000")

    with pytest.raises(ChatClientError, match="Unexpected /chat response"):
        client.send("hello", None)
