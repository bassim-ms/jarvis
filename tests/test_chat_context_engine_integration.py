"""
Tests for the REAL, activated M16.2 runtime flow through the actual
`/chat` route (app/main.py) and the actual `main.core` singleton -
never `handle_chat_with_context()` tested in isolation, and never a
monkeypatched `core.handle_chat_with_job_id` (see tests/test_chat.py for
that narrower kind of test). These tests only monkeypatch the one thing
that requires network/API keys - `main.core.provider` - and otherwise
exercise the genuine `/chat` -> JarvisCore._dispatch ->
build_request_context -> ContextEngine.build_snapshot ->
_handle_direct -> _build_prompt -> provider.generate_reply chain,
against `main.core.context_engine`, the REAL ContextEngine wired up in
app/main.py (backed by a real, temporary-file-based WorldModelStore, so
this proves the actual activation, not a fake standing in for it).
"""

from fastapi.testclient import TestClient

from app import main
from app.context_engine.engine import ContextEngine
from app.core import JarvisCore
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import CreateJobTool, GetJobTool
from app.providers.base import AIProvider
from app.tools.registry import ToolRegistry
from app.world_model.factory import create_default_world_model_service, create_default_world_model_store
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore

client = TestClient(main.app)


class _RecordingProvider(AIProvider):
    """A fake AIProvider (not an AIProviderManager) - `call_with_requirements`
    (app/providers/manager.py) treats a plain AIProvider as a no-op
    pass-through and just calls `generate_reply(prompt)` directly, so
    this records the EXACT prompt `_build_prompt` produced for real,
    through the real dispatch chain."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _swap_world_model(tmp_path, monkeypatch) -> ContextEngine:
    """Point `main.core.context_engine` at a fresh, isolated,
    file-backed WorldModelStore for this test - never the real app's
    `jarvis_world_model.db` - then restore the original afterward."""
    db_path = str(tmp_path / "world_model.db")
    store = create_default_world_model_store(db_path)
    service = create_default_world_model_service(store)
    engine = ContextEngine(service, memory=main.core.memory)
    monkeypatch.setattr(main.core, "context_engine", engine)
    return engine


# -- A: a real /chat request reaches the real ContextEngine ---------------


def test_chat_request_reaches_the_real_context_engine(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    engine.world_model.create_project("Robotics arm")
    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    assert len(provider.prompts) == 1


# -- B: the resulting ContextSnapshot is actually threaded into the -------
# -- downstream reasoning (prompt) that reaches the provider --------------


def test_chat_prompt_is_enriched_with_real_world_model_context(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    engine.world_model.create_project("Robotics Arm")
    blocker = engine.world_model.create_blocker("Printer nozzle isn't heating")
    from app.world_model.base import LinkedEntityKind

    project = engine.world_model.get_active_projects()[0]
    engine.world_model.add_blocker_link(blocker.id, LinkedEntityKind.PROJECT, project.id)

    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    prompt = provider.prompts[0]
    assert "What Jarvis currently knows from the user's world model" in prompt
    assert "Robotics Arm" in prompt
    assert "Printer nozzle isn't heating" in prompt
    assert prompt.endswith("User: hello there")


# -- C: no relevant context still works, unchanged -------------------------


def test_chat_with_empty_world_model_matches_no_context_engine_at_all(tmp_path, monkeypatch):
    """An empty World Model must produce EXACTLY the same prompt as if no
    context_engine were configured at all - the backward-compatibility
    requirement. Compared against each other (not a hardcoded literal),
    since `main.core.memory` is the real app's persistent MemoryStore and
    may already hold real stored facts from prior use - this test must
    not assume it's empty, only that the two cases agree with each other."""
    provider_without_engine = _RecordingProvider()
    monkeypatch.setattr(main.core, "context_engine", None)
    monkeypatch.setattr(main.core, "provider", provider_without_engine)
    client.post("/chat", json={"message": "hello there"})
    baseline_prompt = provider_without_engine.prompts[0]

    _swap_world_model(tmp_path, monkeypatch)  # empty store, nothing created
    provider_with_empty_world_model = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider_with_empty_world_model)
    client.post("/chat", json={"message": "hello there"})
    empty_world_model_prompt = provider_with_empty_world_model.prompts[0]

    assert empty_world_model_prompt == baseline_prompt
    assert "world model" not in baseline_prompt.lower()


# -- prompt formatter must respect ContextEngine's own budget, never --
# -- re-truncate below what was already selected -----------------------


def test_chat_prompt_includes_every_selected_item_up_to_the_real_budget_not_a_smaller_cap(
    tmp_path, monkeypatch
):
    """Blockers default to a budget of 8 (app/context_engine/base.py:
    ContextBudget.blockers). Creating 7 active blockers must show all 7
    in the prompt - proving the prompt formatter has no separate,
    smaller, hardcoded re-truncation of its own that would silently
    discard items the ContextEngine already selected as relevant."""
    engine = _swap_world_model(tmp_path, monkeypatch)
    titles = [f"blocker {i}" for i in range(7)]
    for title in titles:
        engine.world_model.create_blocker(title)

    provider = _RecordingProvider()
    monkeypatch.setattr(main.core, "provider", provider)

    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    prompt = provider.prompts[0]
    for title in titles:
        assert title in prompt


# -- D: Context Engine remains read-only through the real path ------------


def test_real_chat_request_never_mutates_the_world_model(tmp_path, monkeypatch):
    engine = _swap_world_model(tmp_path, monkeypatch)
    project = engine.world_model.create_project("Robotics arm")
    before = engine.world_model.get_project(project.id)
    monkeypatch.setattr(main.core, "provider", _RecordingProvider())

    client.post("/chat", json={"message": "tell me about my robotics arm project"})
    client.post("/chat", json={"message": "hello again"})

    after = engine.world_model.get_project(project.id)
    assert after == before  # untouched: same status/updated_at/etc.


# -- E: existing M15 TASK behavior remains regression-free -----------------
#
# Deliberately NOT run through `main.app`/`main.core` here: that registry's
# `create_job` tool is backed by the real app's persistent
# `jarvis_orchestration.db` file (app/main.py), and this test must not
# leave real rows behind in it. Instead this constructs an isolated
# JarvisCore (own ToolRegistry, own in-memory JobStore) - the exact same
# "fresh JarvisCore, real dispatch, fake/in-memory backing stores"
# convention tests/test_core_context_engine_integration.py already uses -
# with a REAL ContextEngine configured, proving `_dispatch`'s TASK branch
# (`_handle_task`) is completely unaffected by the context_snapshot it now
# also computes: TASK never calls `_handle_direct`/`_build_prompt` (the
# only methods this milestone's correction touched) at all, by construction.


def test_task_creation_is_unaffected_by_a_configured_context_engine():
    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(CreateJobTool(store=job_store))
    tools.register(GetJobTool(store=job_store))

    world_model = WorldModelService(SQLiteWorldModelStore(":memory:"))
    context_engine = ContextEngine(world_model)

    core = JarvisCore(
        provider=_RecordingProvider(), tools=tools, context_engine=context_engine
    )

    reply, conversation_id, sources, job_id = core.handle_chat_with_job_id(
        "create a goal: organize my downloads"
    )

    assert job_id is not None
    job = tools.execute("get_job", params={"job_id": job_id})
    assert job.data["goal"] == "organize my downloads"
