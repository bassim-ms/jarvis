"""
Plan Validator (M15): the one place raw planner output is turned into a
list of trustworthy PlanSteps, or rejected outright. Neither JobEngine
nor any Planner ever executes a step, or even keeps it around, without
it having passed through `validate_plan` first - "do not execute an
unvalidated plan" (see this project's M15 safety requirements) is
enforced structurally, not by convention: PlanStep.permission (the field
JobEngine gates confirmation on) is only ever set here, from the tool's
own registered PermissionLevel - never trusted from AI output.

Deliberately reusable for both a fresh plan (JobEngine._plan) and a
replan's replacement steps (JobEngine._handle_step_failure): the caller
passes `known_step_ids` for the latter case - the ids of steps that
already exist elsewhere in the Job's plan (already-completed ones a new
step is allowed to depend on) but aren't part of `raw_steps` itself.
"""

from __future__ import annotations

import re
from typing import Any

from app.orchestration.base import MAX_EXPECTATION_CHARS, PlanStep, RetryPolicy, StepStatus
from app.orchestration.references import (
    ReferenceError,
    looks_like_reference,
    validate_reference,
)
from app.tools.base import PermissionLevel, Tool
from app.tools.registry import ToolRegistry

#: A plan-local step id must be a simple token - never used as anything
#: but a dict key/dependency reference, but kept narrow so a malformed or
#: adversarial planner response can't smuggle anything odd through it.
_STEP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Hard bounds on a planner-specified retry policy - independent of
#: settings.orchestration_max_retries_per_step (the DEFAULT applied when
#: a step doesn't specify one at all), these are the absolute ceiling any
#: explicit value is checked against, so a planner can't request e.g.
#: 10,000 attempts or a multi-hour delay.
_MAX_ALLOWED_ATTEMPTS = 5
_MAX_ALLOWED_DELAY_SECONDS = 60.0

# Mirrors app.tools.decision._TYPE_CHECKS - the same best-effort mapping
# for ToolParameter's provider-agnostic plain-string types (see
# app/tools/base.py:ToolParameter). Kept as its own copy rather than
# importing a private name from app.tools.decision.
_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "structured": (dict, list),  # any JSON object or list
}


class PlanValidationError(ValueError):
    """Raised when a raw plan can't be safely validated - never silently
    repaired or partially accepted. JobEngine fails the Job with this
    error's message rather than executing anything from the rejected
    plan."""


def _matches_type(value: Any, declared_type: str) -> bool:
    expected = _TYPE_CHECKS.get(declared_type)
    if expected is None:
        return True
    if declared_type == "number" and isinstance(value, bool):
        return False  # bool is a subclass of int in Python
    return isinstance(value, expected)


def _validate_arguments(
    tool: Tool,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    deferred: frozenset[str] = frozenset(),
) -> None:
    """`deferred` names arguments whose value is a step reference: their
    presence counts (a required parameter given by a reference is
    supplied), but their TYPE can only be checked once the reference is
    resolved at execution time - see `validate_resolved_arguments`."""
    declared = {p.name: p for p in tool.parameters}

    unexpected = set(arguments) - set(declared)
    if unexpected:
        raise PlanValidationError(
            f"Step for tool '{tool_name}' has unexpected argument(s): "
            f"{sorted(unexpected)}"
        )

    missing = [
        name for name, param in declared.items() if param.required and name not in arguments
    ]
    if missing:
        raise PlanValidationError(
            f"Step for tool '{tool_name}' is missing required argument(s): "
            f"{', '.join(missing)}"
        )

    for name, value in arguments.items():
        if name in deferred:
            continue
        param = declared[name]
        if not _matches_type(value, param.type):
            raise PlanValidationError(
                f"Argument '{name}' for tool '{tool_name}' must be of type "
                f"'{param.type}', got: {value!r}"
            )


def validate_resolved_arguments(tool: Tool, tool_name: str, arguments: dict[str, Any]) -> None:
    """The full argument check (unexpected/missing/type) applied to a
    step's arguments AFTER every reference in them was resolved - the
    second half of the validation a reference-carrying step could only
    partly get at plan time. Raises PlanValidationError."""
    _validate_arguments(tool, tool_name, arguments)


def _parse_retry_policy(raw_step: dict[str, Any], tool_name: str, default: RetryPolicy) -> RetryPolicy:
    if "max_attempts" not in raw_step and "retry_delay_seconds" not in raw_step:
        return RetryPolicy(
            max_attempts=default.max_attempts, delay_seconds=default.delay_seconds
        )

    max_attempts = raw_step.get("max_attempts", default.max_attempts)
    delay_seconds = raw_step.get("retry_delay_seconds", default.delay_seconds)

    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or not (1 <= max_attempts <= _MAX_ALLOWED_ATTEMPTS)
    ):
        raise PlanValidationError(
            f"Step for tool '{tool_name}' has an invalid max_attempts: "
            f"{max_attempts!r} (must be an integer 1-{_MAX_ALLOWED_ATTEMPTS})"
        )
    if (
        not isinstance(delay_seconds, (int, float))
        or isinstance(delay_seconds, bool)
        or not (0 <= delay_seconds <= _MAX_ALLOWED_DELAY_SECONDS)
    ):
        raise PlanValidationError(
            f"Step for tool '{tool_name}' has an invalid retry_delay_seconds: "
            f"{delay_seconds!r} (must be 0-{_MAX_ALLOWED_DELAY_SECONDS})"
        )

    return RetryPolicy(max_attempts=int(max_attempts), delay_seconds=float(delay_seconds))


def _detect_cycle(steps: list[PlanStep]) -> str | None:
    """Return a description of the first dependency cycle found, or None
    if the dependency graph is acyclic. Plain DFS with a recursion stack -
    small, fixed plans (bounded by max_steps), so no need for anything
    fancier."""
    by_id = {step.id: step for step in steps}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {step.id: WHITE for step in steps}

    def visit(step_id: str, path: list[str]) -> str | None:
        color[step_id] = GRAY
        for dep in by_id[step_id].depends_on:
            if dep not in by_id:
                continue  # already reported as a dangling reference
            if color[dep] == GRAY:
                cycle = " -> ".join(path + [dep])
                return f"Circular dependency: {cycle}"
            if color[dep] == WHITE:
                found = visit(dep, path + [dep])
                if found:
                    return found
        color[step_id] = BLACK
        return None

    for step in steps:
        if color[step.id] == WHITE:
            found = visit(step.id, [step.id])
            if found:
                return found
    return None


def validate_plan(
    raw_steps: Any,
    registry: ToolRegistry,
    *,
    max_steps: int,
    default_retry: RetryPolicy,
    known_step_ids: frozenset[str] = frozenset(),
) -> list[PlanStep]:
    """Validate `raw_steps` (a planner's parsed-but-unchecked list of step
    dicts) against `registry` and return fully-populated PlanSteps, or
    raise PlanValidationError. Never returns a partial result - either
    every step is valid, or nothing is returned at all.

    Checks performed, in order: overall shape, step count safety limit,
    unique/well-formed ids, tool existence, argument schema, dependency
    references (including against `known_step_ids`), circular
    dependencies, and retry configuration. `permission` on each returned
    PlanStep comes from the tool's own registered PermissionLevel - the
    only source of truth JobEngine ever gates confirmation on.
    """
    if not isinstance(raw_steps, list):
        raise PlanValidationError(f"Plan must be a list of steps, got: {type(raw_steps).__name__}")
    if not raw_steps:
        raise PlanValidationError("Plan has no steps.")
    if len(raw_steps) > max_steps:
        raise PlanValidationError(
            f"Plan has {len(raw_steps)} steps, exceeding the safety limit of {max_steps}."
        )

    seen_ids: set[str] = set()
    parsed: list[dict[str, Any]] = []

    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise PlanValidationError(f"Each step must be an object, got: {raw_step!r}")

        step_id = raw_step.get("id")
        if not isinstance(step_id, str) or not _STEP_ID_PATTERN.match(step_id):
            raise PlanValidationError(f"Invalid or missing step id: {step_id!r}")
        if step_id in seen_ids or step_id in known_step_ids:
            raise PlanValidationError(f"Duplicate step id: '{step_id}'")
        seen_ids.add(step_id)

        tool_name = raw_step.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            raise PlanValidationError(f"Step '{step_id}' is missing a valid 'tool' name.")

        arguments = raw_step.get("arguments", {})
        if not isinstance(arguments, dict):
            raise PlanValidationError(f"Step '{step_id}' arguments must be an object.")

        depends_on = raw_step.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(isinstance(d, str) for d in depends_on):
            raise PlanValidationError(f"Step '{step_id}' depends_on must be a list of strings.")
        if step_id in depends_on:
            raise PlanValidationError(f"Step '{step_id}' cannot depend on itself.")

        for arg_name, arg_value in arguments.items():
            if not looks_like_reference(arg_value):
                continue
            try:
                from_step, _segments = validate_reference(arg_value)
            except ReferenceError as exc:
                raise PlanValidationError(
                    f"Step '{step_id}' argument '{arg_name}': {exc}."
                ) from exc
            if from_step not in depends_on:
                raise PlanValidationError(
                    f"Step '{step_id}' argument '{arg_name}' references step "
                    f"'{from_step}', which must be listed in its depends_on."
                )

        expectation = raw_step.get("expectation")
        if expectation is not None:
            if not isinstance(expectation, str) or not expectation.strip():
                raise PlanValidationError(
                    f"Step '{step_id}' expectation must be a non-empty string."
                )
            if len(expectation) > MAX_EXPECTATION_CHARS:
                raise PlanValidationError(
                    f"Step '{step_id}' expectation is longer than {MAX_EXPECTATION_CHARS} characters."
                )
            expectation = expectation.strip()

        parsed.append(
            {
                "id": step_id,
                "tool": tool_name,
                "arguments": arguments,
                "depends_on": depends_on,
                "expectation": expectation,
                "raw": raw_step,
            }
        )

    all_known_ids = seen_ids | known_step_ids
    for entry in parsed:
        for dep in entry["depends_on"]:
            if dep not in all_known_ids:
                raise PlanValidationError(
                    f"Step '{entry['id']}' depends on unknown step '{dep}'."
                )

    steps: list[PlanStep] = []
    for entry in parsed:
        tool = registry.get(entry["tool"])
        if tool is None:
            raise PlanValidationError(
                f"Step '{entry['id']}' refers to an unavailable tool: '{entry['tool']}'."
            )
        if not tool.plannable:
            raise PlanValidationError(
                f"Step '{entry['id']}' names tool '{entry['tool']}', which manages "
                "Jarvis itself and cannot be used in a plan."
            )
        _validate_arguments(
            tool,
            entry["tool"],
            entry["arguments"],
            deferred=frozenset(
                name for name, value in entry["arguments"].items() if looks_like_reference(value)
            ),
        )
        retry_policy = _parse_retry_policy(entry["raw"], entry["tool"], default_retry)
        if entry["expectation"] is not None and tool.permission is not PermissionLevel.READ:
            # An insufficient verdict fails the step and can lead a replan to
            # do it again differently; that must never mean repeating a
            # side effect, so only read-only steps can be evaluated.
            raise PlanValidationError(
                f"Step '{entry['id']}' has an expectation, but tool '{entry['tool']}' is not "
                "read-only - only read-only steps can be evaluated."
            )

        steps.append(
            PlanStep(
                id=entry["id"],
                tool_name=entry["tool"],
                arguments=entry["arguments"],
                depends_on=entry["depends_on"],
                permission=tool.permission.value,
                retry_policy=retry_policy,
                status=StepStatus.PENDING,
                expectation=entry["expectation"],
            )
        )

    cycle = _detect_cycle(steps)
    if cycle:
        raise PlanValidationError(cycle)

    return steps
