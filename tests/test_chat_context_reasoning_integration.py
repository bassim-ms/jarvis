"""
Tests for the REAL, activated M16.3 runtime flow through the actual
`/chat` route (app/main.py) and the actual `main.core` singleton - never
`handle_chat_with_reasoning()` tested in isolation (see
tests/test_core_context_reasoning_integration.py for that narrower kind
of test). These tests only monkeypatch the two things that require
network/API keys - `main.core.provider` (the DIRECT reply) and
`main.core.context_reasoner` (backed by its own fake provider, standing
in for the real Gemini-routed reasoning call) - and a fresh, temporary
World Model, and otherwise exercise the genuine `/chat` ->
JarvisCore._dispatch -> build_request_context -> ContextEngine.
build_snapshot -> ContextReasoner.reason -> _handle_direct ->
_build_prompt -> provider.generate_reply chain.
"""

import json

from fastapi.testclient import TestClient

from app import main
from app.context_engine.engine import ContextEngine
from app.context_reasoning.reasoner import LLMContextReasoner
from app.providers.base import AIProvider, ProviderError
from app.world_model.factory import (
    create_default_world_model_service,
    create_default_world_model_store,
)

client = TestClient(main.app)


class _RecordingProvider(AIProvider):
    """Backs the DIRECT reply - `call_with_requirements` treats a plain
    AIProvider as a no-op pass-through, so this records the EXACT prompt
    `_build_prompt` produced through the real dispatch chain."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _CannedReasonerProvider(AIProvider):
    """Backs ContextReasoner's own LLM call - a separate provider from
    the DIRECT reply's `_RecordingProvider`, so a test can independently
    tell whether the reasoning call happened at all."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


def _valid_payload(**overrides) -> dict:
    payload = {
        "interpretation": "The user is asking about their robotics arm project.",
        "relevant_facts": [],
        "constraints": [],
        "conflicts": [],
        "opportunities": [],
        "conclusions": [],
        "recommended_next_step": None,
        "confidence": 1.0,
        "requires_clarification": False,
        "clarification_question": None,
    }
    payload.update(overrides)
    return payload


def _swap_world_model(tmp_path, monkeypatch) -> ContextEngine:
    db_path = str(tmp_path / "world_model.db")
    store = create_default_world_model_store(db_path)
    service = create_default_world_model_service(store)
    engine = ContextEngine(service, memory=main.core.memory)
    monkeypatch.setattr(main.core, "context_engine", engine)
    return engine


def _swap_reasoner(monkeypatch, payload: dict | None = None, error: Exception | None = None):
    reasoner_provider = _CannedReasonerProvider(payload=payload, error=error)
    monkeypatch.setattr(
        main.core, "context_reasoner", LLMContextReasoner(reasoner_provider)
    )
    return reasoner_provider


# -- A: a real /chat request reaches the real ContextReasoner ---------------


def test_chat_request_reaches_the_real_context_reasoner(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    engine.world_model.create_project("Robotics arm")
    reasoner_provider = _swap_reasoner(monkeypatch, payload=_valid_payload())
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "tell me about my robotics arm project"})

    assert response.status_code == 200
    assert reasoner_provider.calls == 1
    assert len(provider.prompts) == 1


# -- B: the resulting ReasoningResult is actually threaded into the ---------
# -- downstream reasoning (prompt) that reaches the provider ---------------


def test_chat_prompt_is_enriched_with_real_reasoning_output(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    project = engine.world_model.create_project("Robotics Arm")
    payload = _valid_payload(
        conclusions=[
            {
                "text": "Nothing is currently blocking the robotics arm project.",
                "provenance": [f"project:{project.id}"],
            }
        ]
    )
    _swap_reasoner(monkeypatch, payload=payload)
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "tell me about my robotics arm project"})

    assert response.status_code == 200
    prompt = provider.prompts[0]
    assert "What Rhasspy currently knows from the user's world model" in prompt
    assert "What Rhasspy's reasoning about this request concluded" in prompt
    assert "Nothing is currently blocking the robotics arm project." in prompt
    assert prompt.endswith("User: tell me about my robotics arm project")


# -- C: no relevant context still works, unchanged --------------------------


def test_chat_with_empty_world_model_matches_no_context_reasoner_at_all(tmp_path, monkeypatch):
    """An empty World Model must produce EXACTLY the same prompt whether
    or not a context_reasoner is configured - the same backward-
    compatibility requirement M16.2's own equivalent test already
    proves one layer up (tests/test_chat_context_engine_integration.py)."""
    _swap_world_model(tmp_path, monkeypatch)  # empty store, nothing created
    monkeypatch.setattr(main.core, "context_reasoner", None)
    provider_without_reasoner = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_without_reasoner)
    client.post("/chat", json={"message": "hello there"})
    baseline_prompt = provider_without_reasoner.prompts[0]

    reasoner_provider = _swap_reasoner(monkeypatch, payload=_valid_payload())
    provider_with_reasoner = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_with_reasoner)
    client.post("/chat", json={"message": "hello there"})
    empty_reasoning_prompt = provider_with_reasoner.prompts[0]

    assert empty_reasoning_prompt == baseline_prompt
    assert "Rhasspy's reasoning" not in baseline_prompt
    # The reasoner's own empty-ContextSnapshot short circuit means the
    # provider backing it was never even called for this message.
    assert reasoner_provider.calls == 0


# -- D: World Model / reasoner remain read-only through the real path -------


def test_real_chat_request_never_mutates_the_world_model(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    project = engine.world_model.create_project("Robotics arm")
    before = engine.world_model.get_project(project.id)
    _swap_reasoner(monkeypatch, payload=_valid_payload())
    monkeypatch.setattr(main.core, "provider", _RecordingProvider())

    client.post("/chat", json={"message": "tell me about my robotics arm project"})
    client.post("/chat", json={"message": "hello again"})

    after = engine.world_model.get_project(project.id)
    assert after == before


# -- E: provider failures during reasoning degrade safely -------------------


def test_chat_still_succeeds_when_the_reasoning_provider_fails(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    engine.world_model.create_project("Robotics arm")
    _swap_reasoner(monkeypatch, error=ProviderError("boom"))
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "tell me about my robotics arm project"})

    assert response.status_code == 200
    assert response.json()["reply"] == "ok"


def test_chat_still_succeeds_when_the_reasoning_provider_returns_malformed_output(
    tmp_path, monkeypatch
):
    engine = _swap_world_model(tmp_path, monkeypatch)
    engine.world_model.create_project("Robotics arm")
    reasoner_provider = _CannedReasonerProvider(payload=None)
    reasoner_provider.generate_reply = lambda prompt: "not valid json"  # type: ignore[method-assign]
    monkeypatch.setattr(
        main.core, "context_reasoner", LLMContextReasoner(reasoner_provider)
    )
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "tell me about my robotics arm project"})

    assert response.status_code == 200
    assert response.json()["reply"] == "ok"


# -- F: existing M15 TASK behavior remains regression-free -------------------


def test_task_creation_is_unaffected_by_a_configured_context_reasoner(tmp_path, monkeypatch):
    """TASK never calls `_handle_direct`/`_build_prompt` at all, by
    construction - so a configured context_reasoner (like a configured
    context_engine before it) can never affect Job creation."""
    from app.core import JarvisCore
    from app.orchestration.sqlite_store import SQLiteJobStore
    from app.orchestration.tools import CreateJobTool, GetJobTool
    from app.tools.registry import ToolRegistry
    from app.world_model.service import WorldModelService
    from app.world_model.sqlite_store import SQLiteWorldModelStore

    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(CreateJobTool(store=job_store))
    tools.register(GetJobTool(store=job_store))

    world_model = WorldModelService(SQLiteWorldModelStore(":memory:"))
    context_engine = ContextEngine(world_model)
    reasoner_provider = _CannedReasonerProvider(payload=_valid_payload())
    context_reasoner = LLMContextReasoner(reasoner_provider)

    core = JarvisCore(
        provider=_RecordingProvider(),
        tools=tools,
        context_engine=context_engine,
        context_reasoner=context_reasoner,
    )

    reply, conversation_id, sources, job_id = core.handle_chat_with_job_id(
        "create a goal: organize my downloads"
    )

    assert job_id is not None
    assert reasoner_provider.calls == 0
    job = tools.execute("get_job", params={"job_id": job_id})
    assert job.data["goal"] == "organize my downloads"
