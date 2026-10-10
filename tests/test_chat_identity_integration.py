"""
Tests for the REAL, activated identity-context runtime flow through the
actual `/chat` route (app/main.py) and the actual `main.core` singleton -
proving personality.md/user_profile.md (repo root) are genuinely loaded
and reach a real DIRECT request, never `handle_chat` tested in isolation
with a hand-built string (see tests/test_core_identity_integration.py
for that narrower kind of test).
"""

from fastapi.testclient import TestClient

from app import main
from app.providers.base import AIProvider

client = TestClient(main.app, base_url="http://localhost")


class _RecordingProvider(AIProvider):
    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_main_core_loaded_a_real_non_empty_identity_context():
    """The real app/main.py wiring actually activated this - loaded
    from the real personality.md/user_profile.md at repo root, not a
    test double."""
    assert main.core.identity_context
    assert "Bassim" in main.core.identity_context


def test_chat_prompt_is_enriched_with_the_real_identity_context(monkeypatch):
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert "Jarvis's personality:" in prompt
    assert "About the user:" in prompt
    assert "Bassim" in prompt
    assert prompt.endswith("User: hello there")


def test_disabling_identity_context_removes_it_from_the_prompt(monkeypatch):
    """Backward-compatibility check at the real /chat level: turning
    identity_context off removes exactly that block. Compared against
    a baseline with it re-enabled (not a hardcoded literal), since
    `main.core.memory` is the real app's persistent MemoryStore and may
    already hold real stored facts from prior use - this test must not
    assume the rest of the prompt is empty, only that the identity
    block specifically is gone."""
    provider_with = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_with)
    client.post("/chat", json={"message": "hello there"})
    prompt_with_identity = provider_with.prompts[0]

    monkeypatch.setattr(main.core, "identity_context", None)
    provider_without = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_without)

    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    prompt_without_identity = provider_without.prompts[0]
    assert "Jarvis's personality:" in prompt_with_identity
    assert "Jarvis's personality:" not in prompt_without_identity
    assert prompt_without_identity.endswith("User: hello there")
