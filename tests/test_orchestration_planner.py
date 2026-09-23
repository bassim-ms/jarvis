"""
Tests for GeminiPlanner (app/orchestration/planner.py): structured JSON
plan-proposal parsing, replan prompt construction, and strict failure on
malformed AI output - never a fallback guess. Uses a FakeProvider, never
a real Gemini call.
"""

import json

import pytest

from app.orchestration.base import AutonomyLevel, PlanStep, StepStatus
from app.orchestration.planner import GeminiPlanner, PlanningError, ReplanContext
from app.providers.base import AIProvider, ProviderError
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echoes text."
    permission = PermissionLevel.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="text", type="string", description="")]

    def execute(self, text: str = "") -> ToolResult:
        return ToolResult.success(data=text)


class FakeProvider(AIProvider):
    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.last_message: str | None = None

    def generate_reply(self, message: str) -> str:
        self.last_message = message
        if self.error is not None:
            raise self.error
        return self.reply


def test_plan_returns_parsed_steps():
    payload = {"steps": [{"id": "s1", "tool": "echo", "arguments": {"text": "hi"}, "depends_on": []}]}
    provider = FakeProvider(reply=json.dumps(payload))
    planner = GeminiPlanner(provider=provider)

    steps = planner.plan(
        "say hi", available_tools=[EchoTool()], context={}, autonomy=AutonomyLevel.ASSISTED
    )

    assert steps == payload["steps"]


def test_plan_strips_markdown_code_fence():
    payload = {"steps": [{"id": "s1", "tool": "echo", "arguments": {}, "depends_on": []}]}
    provider = FakeProvider(reply=f"```json\n{json.dumps(payload)}\n```")
    planner = GeminiPlanner(provider=provider)

    steps = planner.plan(
        "goal", available_tools=[EchoTool()], context={}, autonomy=AutonomyLevel.ASSISTED
    )

    assert steps == payload["steps"]


def test_plan_prompt_includes_goal_tools_context_and_autonomy():
    payload = {"steps": [{"id": "s1", "tool": "echo", "arguments": {}, "depends_on": []}]}
    provider = FakeProvider(reply=json.dumps(payload))
    planner = GeminiPlanner(provider=provider)

    planner.plan(
        "prepare the report",
        available_tools=[EchoTool()],
        context={"user_name": "Bassim"},
        autonomy=AutonomyLevel.AUTONOMOUS,
    )

    assert "prepare the report" in provider.last_message
    assert "echo" in provider.last_message
    assert "Bassim" in provider.last_message
    assert "autonomous" in provider.last_message


def test_plan_raises_planning_error_on_provider_error():
    provider = FakeProvider(error=ProviderError("boom"))
    planner = GeminiPlanner(provider=provider)

    with pytest.raises(PlanningError):
        planner.plan("goal", available_tools=[], context={}, autonomy=AutonomyLevel.ASSISTED)


def test_plan_raises_planning_error_on_invalid_json():
    provider = FakeProvider(reply="not json at all")
    planner = GeminiPlanner(provider=provider)

    with pytest.raises(PlanningError):
        planner.plan("goal", available_tools=[], context={}, autonomy=AutonomyLevel.ASSISTED)


def test_plan_raises_planning_error_when_steps_key_missing():
    provider = FakeProvider(reply=json.dumps({"plan": []}))
    planner = GeminiPlanner(provider=provider)

    with pytest.raises(PlanningError):
        planner.plan("goal", available_tools=[], context={}, autonomy=AutonomyLevel.ASSISTED)


def test_plan_raises_planning_error_when_steps_not_a_list():
    provider = FakeProvider(reply=json.dumps({"steps": "not a list"}))
    planner = GeminiPlanner(provider=provider)

    with pytest.raises(PlanningError):
        planner.plan("goal", available_tools=[], context={}, autonomy=AutonomyLevel.ASSISTED)


def test_plan_raises_planning_error_on_unexpected_top_level_keys():
    provider = FakeProvider(reply=json.dumps({"steps": [], "extra": 1}))
    planner = GeminiPlanner(provider=provider)

    with pytest.raises(PlanningError):
        planner.plan("goal", available_tools=[], context={}, autonomy=AutonomyLevel.ASSISTED)


def test_plan_raises_planning_error_when_response_is_not_an_object():
    provider = FakeProvider(reply=json.dumps([1, 2, 3]))
    planner = GeminiPlanner(provider=provider)

    with pytest.raises(PlanningError):
        planner.plan("goal", available_tools=[], context={}, autonomy=AutonomyLevel.ASSISTED)


def test_plan_with_replan_context_includes_failure_details():
    payload = {"steps": [{"id": "s2", "tool": "echo", "arguments": {}, "depends_on": []}]}
    provider = FakeProvider(reply=json.dumps(payload))
    planner = GeminiPlanner(provider=provider)

    completed = [PlanStep(id="s1", tool_name="echo", status=StepStatus.SUCCEEDED)]
    failed = PlanStep(id="s_fail", tool_name="echo", status=StepStatus.FAILED, error="network down")

    planner.plan(
        "goal",
        available_tools=[EchoTool()],
        context={},
        autonomy=AutonomyLevel.ASSISTED,
        replan_context=ReplanContext(
            completed_steps=completed,
            failed_step=failed,
            failure_reason="network down",
            remaining_steps=[],
        ),
    )

    assert "REPLAN" in provider.last_message
    assert "network down" in provider.last_message
    assert "s1" in provider.last_message
    assert "s_fail" in provider.last_message
