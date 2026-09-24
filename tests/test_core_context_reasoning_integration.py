"""
Tests for JarvisCore's M16.3 integration point (app/core.py):
`context_reasoner` is an optional, additive constructor parameter, and
`handle_chat_with_reasoning` is the one new method that exposes the
ReasoningResult `_dispatch` builds - `handle_chat`/`handle_chat_with_job_id`/
`handle_chat_with_context` must remain byte-for-byte unaffected, whether
or not a context_reasoner is configured (mirrors
tests/test_core_context_engine_integration.py's own M16.2 discipline,
one layer up).
"""

import json

from app.context_engine.engine import ContextEngine
from app.context_reasoning.reasoner import ContextReasonerError, LLMContextReasoner
from app.core import JarvisCore
from app.providers.base import AIProvider, ProviderError
from app.tools.base import PermissionLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore


class _RecordingProvider(AIProvider):
    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _CannedReasonerProvider(AIProvider):
    """Backs the ContextReasoner's own LLM call - always the SECOND
    provider call of a turn, entirely separate from the DIRECT reply's
    own `_RecordingProvider` above, so a test can tell which one was
    actually exercised."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


def _valid_payload() -> dict:
    return {
        "interpretation": "The user is asking about their robotics project.",
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


def make_world_model() -> WorldModelService:
    return WorldModelService(SQLiteWorldModelStore(":memory:"))


def make_core(
    *,
    world_model: WorldModelService | None = None,
    reasoner_payload: dict | None = None,
    reasoner_error: Exception | None = None,
    with_context_engine: bool = True,
    with_context_reasoner: bool = True,
    reply_provider: AIProvider | None = None,
    tools: ToolRegistry | None = None,
) -> tuple[JarvisCore, _RecordingProvider, _CannedReasonerProvider]:
    world_model = world_model or make_world_model()
    context_engine = ContextEngine(world_model) if with_context_engine else None
    reasoner_provider = _CannedReasonerProvider(
        payload=reasoner_payload or _valid_payload(), error=reasoner_error
    )
    context_reasoner = LLMContextReasoner(reasoner_provider) if with_context_reasoner else None
    reply_provider = reply_provider or _RecordingProvider()

    core = JarvisCore(
        provider=reply_provider,
        tools=tools,
        context_engine=context_engine,
        context_reasoner=context_reasoner,
    )
    return core, reply_provider, reasoner_provider


# -- defaults / additive presence --------------------------------------------


def test_context_reasoner_defaults_to_none():
    core = JarvisCore(provider=_RecordingProvider())
    assert core.context_reasoner is None


def test_handle_chat_with_reasoning_returns_none_when_no_context_reasoner_configured():
    core, _reply, _reasoner = make_core(with_context_reasoner=False)
    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning("hi")
    assert reasoning is None
    assert reply == "fake reply"


def test_handle_chat_with_reasoning_returns_none_when_context_engine_not_configured():
    """A ReasoningResult needs a ContextSnapshot to reason over - with no
    context_engine, `_dispatch` never builds one, so the reasoner is
    never even asked, regardless of context_reasoner being configured."""
    core, _reply, reasoner_provider = make_core(with_context_engine=False)
    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning("hi")
    assert snapshot is None
    assert reasoning is None
    assert reasoner_provider.calls == 0


def test_handle_chat_with_reasoning_returns_a_real_result_when_configured():
    world_model = make_world_model()
    world_model.create_project("Robotics arm")
    core, _reply, reasoner_provider = make_core(world_model=world_model)

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "tell me about my robotics arm project"
    )

    assert reasoning is not None
    assert reasoner_provider.calls == 1
    assert "robotics" in reasoning.interpretation.lower()


# -- existing methods stay byte-for-byte unaffected --------------------------


def test_handle_chat_is_unaffected_by_a_configured_context_reasoner():
    core_without, _r1, _x1 = make_core(with_context_reasoner=False)
    core_with, _r2, reasoner_provider = make_core()

    reply_a, conv_a, sources_a = core_without.handle_chat("hello")
    reply_b, conv_b, sources_b = core_with.handle_chat("hello")

    assert reply_a == reply_b == "fake reply"
    assert sources_a == sources_b == []


def test_handle_chat_with_job_id_is_unaffected_by_a_configured_context_reasoner():
    core_without, _r1, _x1 = make_core(with_context_reasoner=False)
    core_with, _r2, _x2 = make_core()

    result_without = core_without.handle_chat_with_job_id("hello")
    result_with = core_with.handle_chat_with_job_id("hello")

    assert len(result_without) == len(result_with) == 4
    assert result_without[0] == result_with[0] == "fake reply"


def test_handle_chat_with_context_is_unaffected_by_a_configured_context_reasoner():
    world_model = make_world_model()
    project = world_model.create_project("Robotics arm")
    core_without, _r1, _x1 = make_core(world_model=world_model, with_context_reasoner=False)
    core_with, _r2, _x2 = make_core(world_model=world_model)

    result_without = core_without.handle_chat_with_context("hello there")
    result_with = core_with.handle_chat_with_context("hello there")

    assert len(result_without) == len(result_with) == 5
    snapshot_without, snapshot_with = result_without[4], result_with[4]
    assert {p.id for p in snapshot_without.active_projects} == {
        p.id for p in snapshot_with.active_projects
    } == {project.id}


# -- read-only / no execution -------------------------------------------------


def test_context_reasoner_never_mutates_the_world_model_during_a_chat_turn():
    world_model = make_world_model()
    project = world_model.create_project("Robotics arm")
    before = world_model.get_project(project.id)
    core, _reply, _reasoner = make_core(world_model=world_model)

    core.handle_chat_with_reasoning("tell me about my robotics arm project")

    after = world_model.get_project(project.id)
    assert after == before


def test_context_reasoner_never_calls_tool_registry_execute():
    calls: list[str] = []

    class _SpyTool(Tool):
        name = "noop"
        description = "does nothing"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:  # pragma: no cover - never expected to run
            calls.append("executed")
            return ToolResult.success()

    tools = ToolRegistry()
    tools.register(_SpyTool())
    original_execute = tools.execute

    def _recording_execute(*args, **kwargs):
        calls.append("execute_called")
        return original_execute(*args, **kwargs)

    tools.execute = _recording_execute  # type: ignore[method-assign]

    world_model = make_world_model()
    world_model.create_project("Robotics arm")
    core, _reply, _reasoner = make_core(world_model=world_model, tools=tools)

    core.handle_chat_with_reasoning("tell me about my robotics arm project")

    assert calls == []


# -- provider/output failures degrade safely ---------------------------------


def test_reasoner_provider_error_still_produces_an_ordinary_reply():
    world_model = make_world_model()
    world_model.create_project("Robotics arm")
    core, reply_provider, _reasoner = make_core(
        world_model=world_model, reasoner_error=ProviderError("boom")
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "tell me about my robotics arm project"
    )

    assert reply == "fake reply"
    assert reasoning is None
    assert len(reply_provider.prompts) == 1


def test_reasoner_malformed_output_still_produces_an_ordinary_reply():
    world_model = make_world_model()
    world_model.create_project("Robotics arm")
    reasoner_provider = _CannedReasonerProvider(payload=None)
    reasoner_provider.generate_reply = lambda prompt: "not json"  # type: ignore[method-assign]
    context_engine = ContextEngine(world_model)
    reply_provider = _RecordingProvider()
    core = JarvisCore(
        provider=reply_provider,
        context_engine=context_engine,
        context_reasoner=LLMContextReasoner(reasoner_provider),
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "tell me about my robotics arm project"
    )

    assert reply == "fake reply"
    assert reasoning is None


# -- reasoning actually reaches the prompt ------------------------------------


def test_reasoning_result_reaches_the_direct_prompt():
    world_model = make_world_model()
    world_model.create_project("Robotics arm")
    payload = _valid_payload()
    payload["conclusions"] = [
        {
            "text": "The robotics arm project has nothing blocking it right now.",
            "provenance": [f"project:{world_model.get_active_projects()[0].id}"],
        }
    ]
    core, reply_provider, _reasoner = make_core(
        world_model=world_model, reasoner_payload=payload
    )

    core.handle_chat_with_reasoning("tell me about my robotics arm project")

    assert len(reply_provider.prompts) == 1
    assert "nothing blocking it right now" in reply_provider.prompts[0]
    assert "What Rhasspy's reasoning about this request concluded" in reply_provider.prompts[0]


# -- empty-context regression -------------------------------------------------


def test_empty_world_model_produces_identical_prompt_with_or_without_reasoner():
    empty_world_model = make_world_model()
    core_without, reply_without, _x1 = make_core(
        world_model=empty_world_model, with_context_reasoner=False
    )
    core_with, reply_with, reasoner_provider = make_core(world_model=make_world_model())

    core_without.handle_chat_with_reasoning("hello there")
    core_with.handle_chat_with_reasoning("hello there")

    assert reply_without.prompts[0] == reply_with.prompts[0]
    # The reasoner's own empty-ContextSnapshot short circuit means it was
    # never even called - not just that its output happened to be empty.
    assert reasoner_provider.calls == 0
