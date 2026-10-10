"""
Tests for the REAL, activated M16.4 runtime flow through the actual
`/chat` route (app/main.py) and the actual `main.core` singleton - never
`handle_chat_with_planning()` tested in isolation (see
tests/test_core_context_planning_integration.py for that narrower kind
of test). These tests only monkeypatch the three things that require
network/API keys - `main.core.provider` (the DIRECT reply),
`main.core.context_reasoner`, and `main.core.context_planner` (each
backed by its own fake provider, standing in for the real Gemini-routed
calls) - and a fresh, temporary World Model, and otherwise exercise the
genuine `/chat` -> JarvisCore._dispatch -> build_request_context ->
ContextEngine.build_snapshot -> ContextReasoner.reason ->
ContextPlanner.plan -> _handle_direct -> _build_prompt ->
provider.generate_reply chain.
"""

import json

from fastapi.testclient import TestClient

from app import main
from app.context_engine.engine import ContextEngine
from app.context_planning.planner import LLMContextPlanner
from app.context_reasoning.reasoner import LLMContextReasoner
from app.providers.base import AIProvider, ProviderError
from app.world_model.factory import (
    create_default_world_model_service,
    create_default_world_model_store,
)

client = TestClient(main.app, base_url="http://localhost")


class _RecordingProvider(AIProvider):
    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _CannedProvider(AIProvider):
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


def _valid_reasoning_payload(goal_id: str) -> dict:
    return {
        "interpretation": "The user wants to finish the robotics arm goal.",
        "relevant_facts": [
            {"text": "The robotics arm goal is active.", "provenance": [f"goal:{goal_id}"]}
        ],
        "constraints": [],
        "conflicts": [],
        "opportunities": [],
        "conclusions": [],
        "recommended_next_step": None,
        "confidence": 1.0,
        "requires_clarification": False,
        "clarification_question": None,
    }


def _valid_planning_payload(goal_id: str, **overrides) -> dict:
    payload = {
        "objective": "Finish the robotics arm goal.",
        "planning_type": "MULTI_STEP",
        "contextual_constraints": [],
        "relevant_resources": [
            {"text": "The robotics arm goal.", "provenance": [f"goal:{goal_id}"]}
        ],
        "dependencies": [],
        "assumptions": [],
        "suggested_approach": "Work through the remaining assembly steps around other commitments.",
        "requires_execution": True,
        "confidence": 0.85,
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
    provider = _CannedProvider(payload=payload, error=error)
    monkeypatch.setattr(main.core, "context_reasoner", LLMContextReasoner(provider))
    return provider


def _swap_planner(monkeypatch, payload: dict | None = None, error: Exception | None = None):
    provider = _CannedProvider(payload=payload, error=error)
    monkeypatch.setattr(main.core, "context_planner", LLMContextPlanner(provider))
    return provider


def _seed_project_and_goal(engine: ContextEngine) -> str:
    """A goal is in every planning-oriented intent's required_context
    (see tests/test_core_context_planning_integration.py's own fixture
    docstring for why) - returns the goal's id for grounding canned
    payloads."""
    project = engine.world_model.create_project("Robotics Arm")
    goal = engine.world_model.create_goal(project.id, "Finish robotics arm")
    return goal.id


# -- A: a real /chat request reaches the real ContextPlanner ---------------


def test_chat_request_reaches_the_real_context_planner(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    goal_id = _seed_project_and_goal(engine)
    _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload(goal_id))
    planner_provider = _swap_planner(monkeypatch, payload=_valid_planning_payload(goal_id))
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "finish my robotics arm"})

    assert response.status_code == 200
    assert planner_provider.calls == 1
    assert len(provider.prompts) == 1


# -- B: the resulting PlanningDirective is actually threaded into the -------
# -- downstream reasoning (prompt) that reaches the provider ---------------


def test_chat_prompt_is_enriched_with_real_planning_output(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    goal_id = _seed_project_and_goal(engine)
    _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload(goal_id))
    _swap_planner(monkeypatch, payload=_valid_planning_payload(goal_id))
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "finish my robotics arm"})

    assert response.status_code == 200
    prompt = provider.prompts[0]
    assert "What Jarvis's reasoning about this request concluded" in prompt
    assert "What Jarvis's planning about this request concluded" in prompt
    assert "Work through the remaining assembly steps around other commitments." in prompt
    assert prompt.endswith("User: finish my robotics arm")


# -- C: informational requests never reach the planner (though they still --
# -- reach the reasoner) -----------------------------------------------------


def test_chat_with_informational_request_never_invokes_the_planner(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    goal_id = _seed_project_and_goal(engine)
    reasoner_provider = _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload(goal_id))
    planner_provider = _swap_planner(monkeypatch, payload=_valid_planning_payload(goal_id))
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post(
        "/chat", json={"message": "what's happening with my robotics project?"}
    )

    assert response.status_code == 200
    assert reasoner_provider.calls == 1  # M16.3 still reasons about this
    assert planner_provider.calls == 0  # M16.4 never does
    assert "What Jarvis's planning about this request concluded" not in provider.prompts[0]


# -- D: no relevant context still works, unchanged -------------------------


def test_chat_with_empty_world_model_matches_no_context_planner_at_all(tmp_path, monkeypatch):
    _swap_world_model(tmp_path, monkeypatch)  # empty store, nothing created
    monkeypatch.setattr(main.core, "context_reasoner", None)
    monkeypatch.setattr(main.core, "context_planner", None)
    provider_without = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_without)
    client.post("/chat", json={"message": "finish my robotics arm"})
    baseline_prompt = provider_without.prompts[0]

    reasoner_provider = _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload("n/a"))
    planner_provider = _swap_planner(monkeypatch, payload=_valid_planning_payload("n/a"))
    provider_with = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_with)
    client.post("/chat", json={"message": "finish my robotics arm"})
    empty_context_prompt = provider_with.prompts[0]

    assert empty_context_prompt == baseline_prompt
    assert reasoner_provider.calls == 0
    assert planner_provider.calls == 0


# -- E: World Model / planner remain read-only through the real path -------


def test_real_chat_request_never_mutates_the_world_model(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    goal_id = _seed_project_and_goal(engine)
    project_before = engine.world_model.get_active_projects()[0]
    before = engine.world_model.get_project(project_before.id)
    _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload(goal_id))
    _swap_planner(monkeypatch, payload=_valid_planning_payload(goal_id))
    monkeypatch.setattr(main.core, "provider", _RecordingProvider())

    client.post("/chat", json={"message": "finish my robotics arm"})
    client.post("/chat", json={"message": "hello again"})

    after = engine.world_model.get_project(project_before.id)
    assert after == before


# -- F: provider failures during planning degrade safely -------------------


def test_chat_still_succeeds_when_the_planning_provider_fails(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    goal_id = _seed_project_and_goal(engine)
    _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload(goal_id))
    _swap_planner(monkeypatch, error=ProviderError("boom"))
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "finish my robotics arm"})

    assert response.status_code == 200
    assert response.json()["reply"] == "ok"


def test_chat_still_succeeds_when_the_planning_provider_returns_malformed_output(
    tmp_path, monkeypatch
):
    engine = _swap_world_model(tmp_path, monkeypatch)
    goal_id = _seed_project_and_goal(engine)
    _swap_reasoner(monkeypatch, payload=_valid_reasoning_payload(goal_id))
    planner_provider = _CannedProvider(payload=None)
    planner_provider.generate_reply = lambda prompt: "not valid json"  # type: ignore[method-assign]
    monkeypatch.setattr(main.core, "context_planner", LLMContextPlanner(planner_provider))
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "finish my robotics arm"})

    assert response.status_code == 200
    assert response.json()["reply"] == "ok"


# -- G: existing M15 TASK behavior remains regression-free; no duplicate ---
# -- executable planning system is ever created -----------------------------


def test_task_creation_is_unaffected_by_a_configured_context_planner():
    """TASK never calls `_handle_direct`/`_build_prompt` at all, by
    construction - so a configured context_planner (like context_engine/
    context_reasoner before it) can never affect Job creation, and
    M16.4 never itself creates a competing Job."""
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
    project = world_model.create_project("Robotics Arm")
    goal = world_model.create_goal(project.id, "Finish robotics arm")
    context_engine = ContextEngine(world_model)
    reasoner_provider = _CannedProvider(payload=_valid_reasoning_payload(goal.id))
    planner_provider = _CannedProvider(payload=_valid_planning_payload(goal.id))

    core = JarvisCore(
        provider=_RecordingProvider(),
        tools=tools,
        context_engine=context_engine,
        context_reasoner=LLMContextReasoner(reasoner_provider),
        context_planner=LLMContextPlanner(planner_provider),
    )

    reply, conversation_id, sources, job_id = core.handle_chat_with_job_id(
        "create a goal: organize my downloads"
    )

    assert job_id is not None
    assert reasoner_provider.calls == 0
    assert planner_provider.calls == 0
    job = tools.execute("get_job", params={"job_id": job_id})
    assert job.data["goal"] == "organize my downloads"
    # Exactly one Job exists - M16.4 never created a second, competing one.
    assert len(job_store.list()) == 1
