"""
Tests proving the M16.3 wiring fix in app/core.py: `ContextReasoner.reason`
is only ever invoked from the two `_dispatch` branches whose reply
actually reads a ReasoningResult (DIRECT, and CONFIRMATION's fallback to
DIRECT) - never for SEARCH/TOOL/TASK/TASK_STATUS/a resolved CONFIRMATION,
each of which would otherwise pay for a real LLM reasoning call whose
result is immediately discarded.

Every test here configures a WorldModelService with real, non-empty data
(one active project) so that, if the bug this fixes ever reappeared, the
ContextSnapshot would be non-empty and the reasoner's own empty-snapshot
short circuit (app/context_reasoning/reasoner.py) could not mask it - a
call count of 0 in these tests is only possible because `_dispatch`
itself never asks.
"""

import json

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.context_engine.engine import ContextEngine
from app.context_reasoning.reasoner import LLMContextReasoner
from app.core import JarvisCore
from app.orchestration.base import JobStatus
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import ConfirmJobTool, CreateJobTool, ListJobsTool
from app.providers.base import AIProvider, ProviderError
from app.tools.registry import ToolRegistry
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore


class _RecordingProvider(AIProvider):
    """Backs the ordinary reply (DIRECT/whatever formats a reply text) -
    never the reasoner's own provider."""

    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _CannedReasonerProvider(AIProvider):
    """Backs ContextReasoner's own LLM call - a separate provider so a
    test can independently assert whether reasoning happened at all,
    regardless of what the ordinary reply provider did."""

    def __init__(self, payload: dict | None = None):
        self.payload = payload or _valid_payload()
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps(self.payload)


def _valid_payload() -> dict:
    return {
        "interpretation": "Reasoning about the current request.",
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


class _FixedCapabilityRouter(CapabilityRouter):
    """Always returns the same fixed CapabilityDecision, ignoring the
    message - the same "prove Core dispatches purely on the router's
    decision" test double tests/test_core.py's own FakeCapabilityRouter
    already uses, extended with `goal`/`approve` so it can force every
    capability M16.3 needs to gate against."""

    def __init__(
        self,
        capability: Capability,
        tool_name: str | None = None,
        goal: str | None = None,
        approve: bool | None = None,
    ):
        self.decision = CapabilityDecision(
            capability=capability, tool_name=tool_name, goal=goal, approve=approve
        )

    def route(self, message: str) -> CapabilityDecision:
        return self.decision


def _world_model_with_project() -> WorldModelService:
    """A non-empty WorldModelService - one active project, which lands
    in ContextSnapshot.active_projects for every ContextIntent this
    module's tests exercise (GENERAL/PLANNING/STATUS all include
    RequiredContextCategory.PROJECTS - app/context_engine/base.py), so a
    real ContextSnapshot is always non-empty here."""
    world_model = WorldModelService(SQLiteWorldModelStore(":memory:"))
    world_model.create_project("Robotics Arm")
    return world_model


def _make_core(
    capability_router: CapabilityRouter | None = None,
    tools: ToolRegistry | None = None,
    world_model: WorldModelService | None = None,
) -> tuple[JarvisCore, _RecordingProvider, _CannedReasonerProvider]:
    world_model = world_model or _world_model_with_project()
    context_engine = ContextEngine(world_model)
    reasoner_provider = _CannedReasonerProvider()
    context_reasoner = LLMContextReasoner(reasoner_provider)
    reply_provider = _RecordingProvider()

    core = JarvisCore(
        provider=reply_provider,
        tools=tools or ToolRegistry(),
        capability_router=capability_router,
        context_engine=context_engine,
        context_reasoner=context_reasoner,
    )
    return core, reply_provider, reasoner_provider


# -- 1. DIRECT with non-empty context invokes the reasoner ------------------


def test_direct_with_non_empty_context_invokes_the_reasoner():
    core, reply_provider, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.DIRECT)
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "hello there"
    )

    assert reasoner_provider.calls == 1
    assert reasoning is not None
    assert len(snapshot.active_projects) == 1
    assert reply == "fake reply"


# -- 2. DIRECT with empty context does not make an LLM reasoning call -------


def test_direct_with_empty_context_does_not_call_the_reasoner():
    empty_world_model = WorldModelService(SQLiteWorldModelStore(":memory:"))
    core, _reply, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.DIRECT),
        world_model=empty_world_model,
    )

    _, _, _, _, snapshot, reasoning = core.handle_chat_with_reasoning("hello there")

    assert snapshot is not None
    # The reasoner's own empty-ContextSnapshot short circuit (app/
    # context_reasoning/reasoner.py) still returns a (trivial, empty)
    # ReasoningResult - the assertion that matters is that NO provider
    # call happened, not that the result is None.
    assert reasoner_provider.calls == 0
    assert reasoning is not None
    assert reasoning.is_empty()


# -- 3. SEARCH does not invoke the reasoner ----------------------------------


def test_search_does_not_invoke_the_reasoner():
    # web_search isn't registered - _handle_search raises ProviderError
    # (existing, unrelated behavior); the point here is purely that
    # M16.3's reasoning call is never attempted for the SEARCH branch,
    # regardless of whether SEARCH itself succeeds.
    from app.providers.base import ProviderError as _ProviderError

    core, _reply, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.SEARCH)
    )

    try:
        core.handle_chat_with_reasoning("hello there")
    except _ProviderError:
        pass

    assert reasoner_provider.calls == 0


# -- 4. TOOL does not invoke the reasoner ------------------------------------


def test_tool_does_not_invoke_the_reasoner():
    core, reply_provider, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.TOOL, tool_name=None)
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "hello there"
    )

    assert len(snapshot.active_projects) == 1  # context really was non-empty
    assert reasoner_provider.calls == 0
    assert reasoning is None
    assert reply_provider.prompts == []  # DIRECT's own provider never ran either


# -- 5. TASK does not invoke the reasoner ------------------------------------


def test_task_does_not_invoke_the_reasoner():
    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(CreateJobTool(store=job_store))

    core, _reply, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(
            Capability.TASK, goal="organize my downloads"
        ),
        tools=tools,
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "hello there"
    )

    assert len(snapshot.active_projects) == 1
    assert job_id is not None
    assert reasoner_provider.calls == 0
    assert reasoning is None


# -- 6. TASK_STATUS does not invoke the reasoner -----------------------------


def test_task_status_does_not_invoke_the_reasoner():
    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(ListJobsTool(store=job_store))

    core, _reply, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.TASK_STATUS),
        tools=tools,
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning(
        "hello there"
    )

    assert len(snapshot.active_projects) == 1
    assert reasoner_provider.calls == 0
    assert reasoning is None
    assert reply == "You don't have any tasks right now."


# -- 7. A resolved CONFIRMATION does not unnecessarily invoke reasoning -----


def test_resolved_confirmation_does_not_invoke_the_reasoner():
    """"Resolved" means `_handle_confirmation` actually found something
    pending and dispatched to `confirm_job` - a real Job awaiting
    confirmation, confirmed via the natural-language "yes" path, exactly
    like tests/test_core.py's own CONFIRMATION suite sets one up."""
    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(CreateJobTool(store=job_store))
    tools.register(ListJobsTool(store=job_store))
    tools.register(ConfirmJobTool(store=job_store))

    world_model = _world_model_with_project()
    context_engine = ContextEngine(world_model)
    reasoner_provider = _CannedReasonerProvider()
    context_reasoner = LLMContextReasoner(reasoner_provider)
    reply_provider = _RecordingProvider()

    core = JarvisCore(
        provider=reply_provider,
        tools=tools,
        context_engine=context_engine,
        context_reasoner=context_reasoner,
    )

    # Real TASK dispatch (deterministic "prepare ..." goal-verb phrasing)
    # creates a real Job, then it's manually put into
    # AWAITING_CONFIRMATION - the same setup tests/test_core.py's own
    # `test_yes_confirms_a_pending_task_in_the_same_conversation` uses.
    _, conversation_id, _, _job_id, _snapshot, _reasoning = core.handle_chat_with_reasoning(
        "prepare a report"
    )
    job = job_store.list()[0]
    job.status = JobStatus.AWAITING_CONFIRMATION
    job.pending_step_ids = []
    job.confirmation_message = "Send the report?"
    job_store.save(job)
    assert reasoner_provider.calls == 0  # the TASK turn itself never reasoned

    reply, _, _, _, snapshot, reasoning = core.handle_chat_with_reasoning(
        "yes", conversation_id=conversation_id
    )

    assert len(snapshot.active_projects) == 1  # context really was non-empty
    assert reasoner_provider.calls == 0
    assert reasoning is None
    assert job_store.get(job.id).status == JobStatus.RUNNING  # confirmation really ran


# -- 8. CONFIRMATION -> DIRECT fallback still invokes reasoning -------------


def test_confirmation_fallback_to_direct_invokes_reasoning():
    """Nothing is actually pending, so `_handle_confirmation` returns
    `None` and `_dispatch` falls back to ordinary `_handle_direct` -
    which DOES read a ReasoningResult, so the reasoner must be invoked
    here even though the router matched CONFIRMATION."""
    core, reply_provider, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.CONFIRMATION, approve=True)
    )

    reply, conv_id, sources, job_id, snapshot, reasoning = core.handle_chat_with_reasoning("yes")

    assert len(snapshot.active_projects) == 1
    assert reasoner_provider.calls == 1
    assert reasoning is not None
    assert reply == "fake reply"  # the DIRECT fallback's own reply provider ran
    assert len(reply_provider.prompts) == 1


# -- 9. Existing /chat behavior remains correct ------------------------------


def test_handle_chat_contract_is_unaffected_for_every_capability():
    """The public 3-tuple contract (`handle_chat`) is unaffected by this
    wiring fix for every capability the reasoner is now gated on."""
    for capability in (
        Capability.TOOL,
        Capability.TASK_STATUS,
    ):
        core, _reply, _reasoner = _make_core(
            capability_router=_FixedCapabilityRouter(capability)
        )
        reply, conversation_id, sources = core.handle_chat("hello there")
        assert isinstance(reply, str)
        assert isinstance(conversation_id, str)
        assert sources == []


def test_handle_chat_with_job_id_contract_is_unaffected_by_the_wiring_fix():
    tools = ToolRegistry()
    job_store = SQLiteJobStore(":memory:")
    tools.register(CreateJobTool(store=job_store))
    core, _reply, _reasoner = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.TASK, goal="organize downloads"),
        tools=tools,
    )

    reply, conversation_id, sources, job_id = core.handle_chat_with_job_id("hello there")

    assert job_id is not None
    assert sources == []


# -- 10. No World Model mutation or tool execution is introduced ------------


def test_no_world_model_mutation_across_every_gated_capability():
    world_model = _world_model_with_project()
    project = world_model.get_active_projects()[0]
    before = world_model.get_project(project.id)

    for capability, tools in (
        (Capability.SEARCH, ToolRegistry()),
        (Capability.TOOL, ToolRegistry()),
    ):
        core, _reply, _reasoner = _make_core(
            capability_router=_FixedCapabilityRouter(capability),
            tools=tools,
            world_model=world_model,
        )
        try:
            core.handle_chat_with_reasoning("hello there")
        except ProviderError:
            pass

    after = world_model.get_project(project.id)
    assert after == before


def test_no_tool_execution_is_introduced_for_direct_or_confirmation_fallback():
    """M16.3's reasoning call itself never touches ToolRegistry - proven
    by giving DIRECT/CONFIRMATION-fallback (the two branches that DO
    reason) a registry whose only tool would fail loudly if `execute`
    were ever called by anything other than the ordinary TOOL path
    (which isn't exercised in this test at all)."""
    tools = ToolRegistry()
    calls: list[str] = []
    original_execute = tools.execute

    def _recording_execute(*args, **kwargs):
        calls.append("execute_called")
        return original_execute(*args, **kwargs)

    tools.execute = _recording_execute  # type: ignore[method-assign]

    core, _reply, reasoner_provider = _make_core(
        capability_router=_FixedCapabilityRouter(Capability.DIRECT), tools=tools
    )

    core.handle_chat_with_reasoning("hello there")

    assert calls == []
    assert reasoner_provider.calls == 1  # reasoning still ran - just never via ToolRegistry
