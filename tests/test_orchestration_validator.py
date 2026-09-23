"""
Tests for the Plan Validator (app/orchestration/validator.py): every raw
plan a Planner could produce must be rejected safely unless it refers
only to available tools with correct arguments, has valid/acyclic
dependency references, respects the step-count safety limit, and has
valid retry configuration.
"""

import pytest

from app.orchestration.base import RetryPolicy, StepStatus
from app.orchestration.validator import PlanValidationError, validate_plan
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echoes text."
    permission = PermissionLevel.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="text", type="string", description="")]

    def execute(self, text: str = "") -> ToolResult:
        return ToolResult.success(data=text)


class ConfirmTool(Tool):
    name = "confirm_tool"
    description = "Needs confirmation."
    permission = PermissionLevel.CONFIRM

    def execute(self) -> ToolResult:
        return ToolResult.success()


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ConfirmTool())
    return registry


DEFAULT_RETRY = RetryPolicy(max_attempts=2, delay_seconds=1.0)


def test_validate_plan_builds_steps_with_permission_from_registry():
    registry = make_registry()
    raw = [{"id": "s1", "tool": "echo", "arguments": {"text": "hi"}, "depends_on": []}]

    steps = validate_plan(raw, registry, max_steps=10, default_retry=DEFAULT_RETRY)

    assert len(steps) == 1
    assert steps[0].id == "s1"
    assert steps[0].tool_name == "echo"
    assert steps[0].permission == PermissionLevel.READ.value
    assert steps[0].status == StepStatus.PENDING
    assert steps[0].retry_policy == DEFAULT_RETRY


def test_validate_plan_fills_permission_for_confirm_tool():
    registry = make_registry()
    raw = [{"id": "s1", "tool": "confirm_tool", "arguments": {}, "depends_on": []}]

    steps = validate_plan(raw, registry, max_steps=10, default_retry=DEFAULT_RETRY)

    assert steps[0].permission == PermissionLevel.CONFIRM.value


def test_validate_plan_rejects_non_list():
    with pytest.raises(PlanValidationError):
        validate_plan({"steps": []}, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_empty_plan():
    with pytest.raises(PlanValidationError):
        validate_plan([], make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_too_many_steps():
    raw = [
        {"id": f"s{i}", "tool": "echo", "arguments": {"text": "x"}, "depends_on": []}
        for i in range(5)
    ]
    with pytest.raises(PlanValidationError, match="safety limit"):
        validate_plan(raw, make_registry(), max_steps=3, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_unknown_tool():
    raw = [{"id": "s1", "tool": "does_not_exist", "arguments": {}, "depends_on": []}]
    with pytest.raises(PlanValidationError, match="unavailable tool"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_missing_required_argument():
    raw = [{"id": "s1", "tool": "echo", "arguments": {}, "depends_on": []}]
    with pytest.raises(PlanValidationError, match="missing required"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_unexpected_argument():
    raw = [
        {
            "id": "s1",
            "tool": "echo",
            "arguments": {"text": "hi", "bogus": 1},
            "depends_on": [],
        }
    ]
    with pytest.raises(PlanValidationError, match="unexpected argument"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_wrong_argument_type():
    raw = [{"id": "s1", "tool": "echo", "arguments": {"text": 123}, "depends_on": []}]
    with pytest.raises(PlanValidationError, match="must be of type"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_duplicate_step_ids():
    raw = [
        {"id": "s1", "tool": "echo", "arguments": {"text": "a"}, "depends_on": []},
        {"id": "s1", "tool": "echo", "arguments": {"text": "b"}, "depends_on": []},
    ]
    with pytest.raises(PlanValidationError, match="Duplicate"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_self_dependency():
    raw = [{"id": "s1", "tool": "echo", "arguments": {"text": "a"}, "depends_on": ["s1"]}]
    with pytest.raises(PlanValidationError, match="cannot depend on itself"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_dangling_dependency():
    raw = [
        {"id": "s1", "tool": "echo", "arguments": {"text": "a"}, "depends_on": ["ghost"]}
    ]
    with pytest.raises(PlanValidationError, match="unknown step"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_accepts_dependency_on_known_step_id():
    raw = [
        {
            "id": "s2",
            "tool": "echo",
            "arguments": {"text": "a"},
            "depends_on": ["already_done"],
        }
    ]
    steps = validate_plan(
        raw,
        make_registry(),
        max_steps=10,
        default_retry=DEFAULT_RETRY,
        known_step_ids=frozenset({"already_done"}),
    )
    assert steps[0].depends_on == ["already_done"]


def test_validate_plan_rejects_duplicate_of_a_known_step_id():
    raw = [{"id": "already_done", "tool": "echo", "arguments": {"text": "a"}, "depends_on": []}]
    with pytest.raises(PlanValidationError, match="Duplicate"):
        validate_plan(
            raw,
            make_registry(),
            max_steps=10,
            default_retry=DEFAULT_RETRY,
            known_step_ids=frozenset({"already_done"}),
        )


def test_validate_plan_detects_direct_cycle():
    raw = [
        {"id": "a", "tool": "echo", "arguments": {"text": "x"}, "depends_on": ["b"]},
        {"id": "b", "tool": "echo", "arguments": {"text": "y"}, "depends_on": ["a"]},
    ]
    with pytest.raises(PlanValidationError, match="Circular"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_detects_longer_cycle():
    raw = [
        {"id": "a", "tool": "echo", "arguments": {"text": "x"}, "depends_on": ["c"]},
        {"id": "b", "tool": "echo", "arguments": {"text": "y"}, "depends_on": ["a"]},
        {"id": "c", "tool": "echo", "arguments": {"text": "z"}, "depends_on": ["b"]},
    ]
    with pytest.raises(PlanValidationError, match="Circular"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_accepts_valid_dependency_chain():
    raw = [
        {"id": "a", "tool": "echo", "arguments": {"text": "x"}, "depends_on": []},
        {"id": "b", "tool": "echo", "arguments": {"text": "y"}, "depends_on": ["a"]},
    ]
    steps = validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)
    assert [s.id for s in steps] == ["a", "b"]


def test_validate_plan_accepts_independent_steps_with_no_dependencies():
    raw = [
        {"id": "a", "tool": "echo", "arguments": {"text": "x"}, "depends_on": []},
        {"id": "b", "tool": "echo", "arguments": {"text": "y"}, "depends_on": []},
    ]
    steps = validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)
    assert len(steps) == 2


@pytest.mark.parametrize("step_id", [123, None, "", "has space", "semi;colon"])
def test_validate_plan_rejects_malformed_step_id(step_id):
    raw = [{"id": step_id, "tool": "echo", "arguments": {"text": "x"}, "depends_on": []}]
    with pytest.raises(PlanValidationError):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_rejects_non_dict_step():
    with pytest.raises(PlanValidationError):
        validate_plan(["not a dict"], make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_accepts_explicit_retry_policy():
    raw = [
        {
            "id": "s1",
            "tool": "echo",
            "arguments": {"text": "x"},
            "depends_on": [],
            "max_attempts": 3,
            "retry_delay_seconds": 5,
        }
    ]
    steps = validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)
    assert steps[0].retry_policy == RetryPolicy(max_attempts=3, delay_seconds=5.0)


@pytest.mark.parametrize("max_attempts", [0, -1, 6, "two", 2.5, True])
def test_validate_plan_rejects_invalid_max_attempts(max_attempts):
    raw = [
        {
            "id": "s1",
            "tool": "echo",
            "arguments": {"text": "x"},
            "depends_on": [],
            "max_attempts": max_attempts,
        }
    ]
    with pytest.raises(PlanValidationError, match="max_attempts"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


@pytest.mark.parametrize("delay", [-1, 61, "slow"])
def test_validate_plan_rejects_invalid_retry_delay(delay):
    raw = [
        {
            "id": "s1",
            "tool": "echo",
            "arguments": {"text": "x"},
            "depends_on": [],
            "retry_delay_seconds": delay,
        }
    ]
    with pytest.raises(PlanValidationError, match="retry_delay_seconds"):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)


def test_validate_plan_never_returns_partial_result_on_failure():
    """A plan where the first step is valid but the second is malformed
    must reject the whole plan - never silently return just the valid
    step."""
    raw = [
        {"id": "s1", "tool": "echo", "arguments": {"text": "x"}, "depends_on": []},
        {"id": "s2", "tool": "does_not_exist", "arguments": {}, "depends_on": []},
    ]
    with pytest.raises(PlanValidationError):
        validate_plan(raw, make_registry(), max_steps=10, default_retry=DEFAULT_RETRY)
