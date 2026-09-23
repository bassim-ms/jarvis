"""
Planner (M15): turns a user's goal into a structured, machine-readable
plan proposal - a list of {id, tool, arguments, depends_on, ...} step
dicts - using the existing AI provider abstraction (AIProvider/
GeminiProvider), the same "no second client, no new config, no billing"
discipline GeminiToolCallProvider (app/tools/decision.py) and
GeminiCapabilityDecisionProvider (app/capability.py) already established.

The planner NEVER executes anything and never even talks to
ToolRegistry - it's handed the already-fetched list of available Tools
by its caller (JobEngine), exactly the same shape ToolCallBuilder hands
GeminiToolCallProvider its candidate tools. What it returns is only
"proposed" - full semantic validation (tool existence given the real
registry, argument schemas, dependency cycles, safety limits) happens
entirely in app/orchestration/validator.py, a separate stage, per this
project's Plan Validator requirement. This module's own checking is
deliberately shallow: valid JSON, top-level shape, each entry looks like
a step object - "do not rely on parsing arbitrary natural-language
planner output during execution" is satisfied by structured JSON output
plus that separate validation stage, not by this module trying to be the
sole line of defense.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.orchestration.base import AutonomyLevel, PlanStep
from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.tools.base import Tool


class PlanningError(RuntimeError):
    """Raised when a Planner can't produce a plan proposal at all - a
    failed AI call, or output that isn't even well-formed JSON/shape.
    Never silently guessed around: JobEngine fails the Job with this
    error's message (or, mid-execution, treats a failed replan attempt
    the same way - see JobEngine._handle_step_failure)."""


@dataclass
class ReplanContext:
    """Everything a replan needs to know beyond the original goal -
    JobEngine builds one from the Job's own current state when a step
    fails in a way that might have an alternative path (see
    JobEngine._handle_step_failure). Never includes anything the planner
    couldn't already reconstruct from the Job it's replanning - no
    hidden state.
    """

    completed_steps: list[PlanStep]
    failed_step: PlanStep
    failure_reason: str
    remaining_steps: list[PlanStep]


def _serialize_tool(tool: Tool) -> dict[str, Any]:
    """The minimum needed to propose a call to `tool`: name, description,
    and each ToolParameter's name/type/description/required. Mirrors
    app.tools.decision._serialize_tool exactly (kept as its own copy
    rather than importing a private name from that module) - excludes
    permission, requirements, and anything about the tool's Python
    implementation; the planner decides WHAT to call, never whether it's
    allowed to run unattended (that's PermissionLevel, applied by the
    validator/JobEngine after the fact, never sent to the AI)."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
            }
            for p in tool.parameters
        ],
    }


def _serialize_step_summary(step: PlanStep) -> dict[str, Any]:
    """A compact, JSON-safe summary of one already-attempted PlanStep,
    used only to give a replan prompt honest context - never fed back as
    something to re-execute."""
    return {
        "id": step.id,
        "tool": step.tool_name,
        "arguments": step.arguments,
        "status": step.status.value,
        "error": step.error,
    }


class Planner(ABC):
    """Proposes a plan for a goal. Never executes anything, and has no
    knowledge of ToolRegistry or permissions beyond the Tool descriptions
    it's given."""

    @abstractmethod
    def plan(
        self,
        goal: str,
        *,
        available_tools: list[Tool],
        context: dict[str, Any],
        autonomy: AutonomyLevel,
        replan_context: ReplanContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of raw, shallowly-shape-checked step dicts (see
        module docstring), or raise PlanningError. `replan_context`, when
        given, means: propose ONLY the replacement steps for what's left
        to do - `replan_context.completed_steps` are already done and
        must not be repeated; new step ids must not collide with them
        (they may be named in `depends_on` instead, see
        app/orchestration/validator.py's `known_step_ids`)."""


_BASE_INSTRUCTIONS = (
    "You are a strict planning engine for a personal assistant called "
    "Jarvis. You are given a user's goal, some known context, and a "
    "fixed list of available tools with their parameter schemas. "
    "Produce a plan: an ordered list of steps, each calling exactly one "
    "of the listed tools, that together achieve the goal.\n\n"
    "Respond with ONLY one JSON object, no prose, no markdown formatting, "
    'in exactly this shape: {"steps": [{"id": "step_1", "tool": '
    '"<one of the listed tool names>", "arguments": {"<param>": '
    '<value>, ...}, "depends_on": ["<id of a step that must complete '
    'first>", ...]}, ...]}. Every "id" must be unique and match '
    '[A-Za-z0-9_-]. "depends_on" may be an empty list for a step with no '
    "prerequisites. Never invent a tool name that isn't in the list "
    "below, and never invent a parameter name that isn't declared for "
    "the chosen tool. Only include an independent step's dependencies "
    "when the goal genuinely requires one step's output before another "
    "can run - most simple goals need only one or two steps. Never "
    "include a step for anything not achievable with the listed tools; "
    "if the goal can't be achieved with them at all, still return the "
    "closest useful partial plan.\n\n"
)

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class GeminiPlanner(Planner):
    """Planner backed by the existing AIProvider/Gemini integration - the
    same free-tier client GeminiProvider (and GeminiToolCallProvider/
    GeminiCapabilityDecisionProvider) already use."""

    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or GeminiProvider()

    def plan(
        self,
        goal: str,
        *,
        available_tools: list[Tool],
        context: dict[str, Any],
        autonomy: AutonomyLevel,
        replan_context: ReplanContext | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._build_prompt(goal, available_tools, context, autonomy, replan_context)

        try:
            raw = self.provider.generate_reply(prompt)
        except ProviderError as exc:
            raise PlanningError(f"Failed to get a plan: {exc}") from exc

        return self._parse(raw)

    @staticmethod
    def _build_prompt(
        goal: str,
        available_tools: list[Tool],
        context: dict[str, Any],
        autonomy: AutonomyLevel,
        replan_context: ReplanContext | None,
    ) -> str:
        tools_json = json.dumps([_serialize_tool(t) for t in available_tools])
        parts = [
            _BASE_INSTRUCTIONS,
            "Available tools: ",
            tools_json,
            f"\n\nAutonomy level: {autonomy.value} (informational only - "
            "propose the best plan regardless; confirmation handling is "
            "not your concern).",
            f"\n\nKnown context: {json.dumps(context)}",
        ]

        if replan_context is not None:
            completed = [_serialize_step_summary(s) for s in replan_context.completed_steps]
            remaining = [_serialize_step_summary(s) for s in replan_context.remaining_steps]
            parts.append(
                "\n\nThis is a REPLAN. The original plan hit a problem "
                "partway through. Do NOT repeat steps already completed "
                "successfully - propose only the steps still needed to "
                "achieve the goal from here, using new step ids that "
                "don't collide with the completed ones listed below "
                "(you may reference a completed step's id in "
                '"depends_on" if a new step genuinely needs its result).'
                f"\n\nAlready completed: {json.dumps(completed)}"
                f"\n\nFailed step: {json.dumps(_serialize_step_summary(replan_context.failed_step))}"
                f"\n\nFailure reason: {replan_context.failure_reason}"
                f"\n\nSteps that were still pending: {json.dumps(remaining)}"
            )

        parts.append(f"\n\nUser's goal: {goal}")
        return "".join(parts)

    @staticmethod
    def _parse(raw: str) -> list[dict[str, Any]]:
        cleaned = _CODE_FENCE_PATTERN.sub("", raw.strip()).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Invalid JSON from planner: {raw!r}") from exc

        if not isinstance(payload, dict):
            raise PlanningError(f"Expected a JSON object, got: {raw!r}")

        extra_keys = set(payload) - {"steps"}
        if extra_keys:
            raise PlanningError(f"Unexpected fields in plan: {sorted(extra_keys)}")

        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise PlanningError(f"Missing or invalid 'steps' in plan: {raw!r}")

        return steps
