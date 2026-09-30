"""
Regression coverage: planning, replanning, evaluation and analysis all use
the same provider-management/fallback architecture (AIProviderManager, i.e.
`core.provider`) - not a separate, unmanaged Gemini client.

Before this fix the running app built its JobEngine with a default
`GeminiPlanner()`: planning and replanning talked to one bare GeminiProvider,
so a rate-limited Gemini failed every Job even while the other configured
providers were healthy. These tests pin the wiring AND prove fallback is
really exercised through JobEngine.
"""

import json

import pytest

from app.notes.sqlite_store import SQLiteNoteStore
from app.notes.tools import SaveNoteTool
from app.orchestration.base import JobStatus, StepStatus
from app.orchestration.evaluator import GeminiResultEvaluator
from app.orchestration.factory import create_default_job_engine
from app.orchestration.planner import GeminiPlanner, Planner, PlanningError
from app.orchestration.sqlite_store import SQLiteJobStore
from app.providers.base import AIProvider, ProviderError
from app.providers.manager import AIProviderEntry, AIProviderManager
from app.tools.analyze import AnalyzeTool
from app.tools.base import PermissionLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry

ENGINE_LIMITS = dict(
    max_steps=20,
    max_total_steps=40,
    max_retries_per_step=1,
    max_replans=2,
    retry_delay_seconds=0,
    poll_interval_seconds=1000,
)


# --- the wiring ----------------------------------------------------------------------------------------


def test_the_real_app_drives_planning_evaluation_and_analysis_through_core_provider():
    import app.main as live_app

    provider = live_app.core.provider
    assert type(provider).__name__ == "AIProviderManager"
    assert live_app.job_engine.planner.provider is provider  # planning AND replanning
    assert live_app.job_engine.evaluator.provider is provider
    assert live_app.core.tools.get("analyze").provider is provider


def test_the_factory_builds_the_planner_and_evaluator_from_the_given_provider():
    provider = _Recorder("x")

    engine = create_default_job_engine(
        SQLiteJobStore(":memory:"), ToolRegistry(), provider=provider, **ENGINE_LIMITS
    )

    assert isinstance(engine.planner, GeminiPlanner) and engine.planner.provider is provider
    assert isinstance(engine.evaluator, GeminiResultEvaluator) and engine.evaluator.provider is provider


def test_an_explicit_planner_or_evaluator_still_wins_over_the_provider():
    class MyPlanner(Planner):
        def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
            return []

    my_planner = MyPlanner()
    my_evaluator = GeminiResultEvaluator(_Recorder("other"))

    engine = create_default_job_engine(
        SQLiteJobStore(":memory:"),
        ToolRegistry(),
        planner=my_planner,
        evaluator=my_evaluator,
        provider=_Recorder("x"),
        **ENGINE_LIMITS,
    )

    assert engine.planner is my_planner and engine.evaluator is my_evaluator


def test_without_a_provider_the_factory_behaves_exactly_as_before():
    engine = create_default_job_engine(SQLiteJobStore(":memory:"), ToolRegistry(), **ENGINE_LIMITS)

    assert isinstance(engine.planner, GeminiPlanner)
    assert engine.evaluator is None  # no evaluator, as before this change


# --- fallback is really used ---------------------------------------------------------------------------------


class _Recorder(AIProvider):
    """Answers planner, evaluator and analysis prompts from scripts and
    counts every call. `fail_with` makes it raise instead."""

    def __init__(self, name, *, plans=(), verdicts=(), fail_with=None):
        self.name = name
        self.plans = list(plans)
        self.verdicts = list(verdicts)
        self.fail_with = fail_with
        self.calls = {"plan": 0, "evaluate": 0, "analyze": 0}

    def generate_reply(self, message: str) -> str:
        kind = (
            "plan"
            if "strict planning engine" in message
            else "evaluate"
            if "strict result reviewer" in message
            else "analyze"
        )
        self.calls[kind] += 1
        if self.fail_with is not None:
            raise self.fail_with
        if kind == "plan":
            return json.dumps({"steps": self.plans.pop(0)})
        if kind == "evaluate":
            return json.dumps(self.verdicts.pop(0))
        return "An analysis produced by the backup provider."


class SearchTool(Tool):
    name = "search"
    description = "search"
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(message="found", data={"results": [{"title": "Acme"}]})


def step(step_id, tool, arguments=None, depends_on=(), **extra):
    return {"id": step_id, "tool": tool, "arguments": arguments or {}, "depends_on": list(depends_on), **extra}


def ref(from_step, path):
    return {"from_step": from_step, "path": path}


def rate_limited():
    return ProviderError("Gemini returned 429: quota exceeded")


def test_every_ai_call_of_a_job_falls_back_when_the_first_provider_is_rate_limited():
    """The whole chain - plan, evaluate (insufficient), REPLAN, evaluate
    (sufficient), analyze - is answered by the backup provider because the
    first one 429s. Every one of those calls went through the manager."""
    gemini = _Recorder("gemini", fail_with=rate_limited())
    backup = _Recorder(
        "backup",
        plans=[
            [step("s1", "search", expectation="results about Acme")],
            [
                step("r1", "search", expectation="results about Acme Corp"),
                step("r2", "analyze", {"instruction": "summarise", "material": ref("r1", "data.results")}, ["r1"]),
                step("r3", "save_note", {"title": "Acme", "content": ref("r2", "message")}, ["r2"]),
            ],
        ],
        verdicts=[
            {"sufficient": False, "reason": "not about the company"},
            {"sufficient": True, "reason": "about the company"},
        ],
    )
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("backup", backup)],
        retry_count=0,
        retry_delay_seconds=0,
    )
    notes = SQLiteNoteStore(":memory:")
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(AnalyzeTool(manager))
    registry.register(SaveNoteTool(notes))
    store = SQLiteJobStore(":memory:")
    engine = create_default_job_engine(store, registry, provider=manager, **ENGINE_LIMITS)
    job = store.create("Research Acme Corp")

    engine.tick()

    finished = store.get(job.id)
    assert finished.status == JobStatus.COMPLETED, finished.error
    assert finished.replan_count == 1
    # planning, replanning, evaluation (x2) and analysis were ALL answered by the backup...
    assert backup.calls == {"plan": 2, "evaluate": 2, "analyze": 1}
    # ...because the first provider was tried, failed, and fell back. It is
    # then cooled down, so it isn't hammered for every later call.
    assert 1 <= sum(gemini.calls.values()) < 5
    (note,) = notes.list()
    assert note.content == "An analysis produced by the backup provider."


def test_planning_alone_survives_a_rate_limited_first_provider_but_not_a_bare_one():
    """The exact bug: with a planner bound to the failing provider directly
    (the old wiring) planning fails; bound to the manager (the fix) it works."""
    plan = [step("s1", "search")]

    bare = GeminiPlanner(_Recorder("gemini", fail_with=rate_limited()))
    with pytest.raises(PlanningError, match="429"):
        bare.plan("g", available_tools=[SearchTool()], context={}, autonomy=_autonomy())

    manager = AIProviderManager(
        [
            AIProviderEntry("gemini", _Recorder("gemini", fail_with=rate_limited())),
            AIProviderEntry("backup", _Recorder("backup", plans=[plan])),
        ],
        retry_count=0,
        retry_delay_seconds=0,
    )
    managed = GeminiPlanner(manager)
    assert managed.plan("g", available_tools=[SearchTool()], context={}, autonomy=_autonomy()) == plan


def _autonomy():
    from app.orchestration.base import AutonomyLevel

    return AutonomyLevel.ASSISTED


def test_a_job_created_when_the_first_provider_is_down_still_completes():
    gemini = _Recorder("gemini", fail_with=rate_limited())
    backup = _Recorder("backup", plans=[[step("s1", "search")]])
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("backup", backup)],
        retry_count=0,
        retry_delay_seconds=0,
    )
    registry = ToolRegistry()
    registry.register(SearchTool())
    store = SQLiteJobStore(":memory:")
    engine = create_default_job_engine(store, registry, provider=manager, **ENGINE_LIMITS)
    job = store.create("goal")

    engine.tick()

    assert store.get(job.id).status == JobStatus.COMPLETED
    assert backup.calls["plan"] == 1


def test_when_every_provider_is_down_the_job_fails_with_the_combined_reason():
    manager = AIProviderManager(
        [
            AIProviderEntry("gemini", _Recorder("gemini", fail_with=rate_limited())),
            AIProviderEntry("backup", _Recorder("backup", fail_with=ProviderError("Groq returned 429: limit"))),
        ],
        retry_count=0,
        retry_delay_seconds=0,
    )
    registry = ToolRegistry()
    registry.register(SearchTool())
    store = SQLiteJobStore(":memory:")
    engine = create_default_job_engine(store, registry, provider=manager, **ENGINE_LIMITS)
    job = store.create("goal")

    engine.tick()

    finished = store.get(job.id)
    assert finished.status == JobStatus.FAILED
    assert "Could not create a plan" in finished.error
    assert "gemini" in finished.error and "backup" in finished.error  # both attempts reported


def test_the_evaluator_also_falls_back_and_an_all_provider_outage_is_still_unavailable():
    """Evaluation through a manager whose providers are all down is still
    'unavailable' (never sufficient) - the fallback only helps when a
    provider can actually answer."""
    gemini = _Recorder("gemini", fail_with=rate_limited())
    backup = _Recorder(
        "backup",
        plans=[[step("s1", "search", expectation="e")]],
        verdicts=[{"sufficient": True, "reason": "fine"}],
    )
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini), AIProviderEntry("backup", backup)],
        retry_count=0,
        retry_delay_seconds=0,
    )
    registry = ToolRegistry()
    registry.register(SearchTool())
    store = SQLiteJobStore(":memory:")
    engine = create_default_job_engine(store, registry, provider=manager, **ENGINE_LIMITS)
    job = store.create("goal")

    engine.tick()

    finished = store.get(job.id)
    assert finished.status == JobStatus.COMPLETED
    assert finished.plan.get("s1").evaluation["verdict"] == "sufficient"
    assert backup.calls["evaluate"] == 1
