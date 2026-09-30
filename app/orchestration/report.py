"""
Final reporting (M15): what a COMPLETED Job tells the user.

The user-facing result is built from the plan's TERMINAL steps - the steps
nothing else depends on, i.e. where the plan ended up - not from a
transcript of every tool message. A four-step research plan therefore
reports its outcome, not "search: 10 results; ...".

Deterministic on purpose: no AI call, just a fixed rule over the persisted
plan, so the same Job always yields the same report (and a restart can
rebuild it):

  * Body: if a step whose tool declares `produces_report` (`Tool.
    produces_report`, e.g. `analyze`) succeeded, the LAST such step's output
    is the body, even if it isn't terminal - it is the synthesized answer the
    later steps (save a note, set a reminder) merely acted on. Nothing else
    from intermediate steps is shown.
  * Outcome lines: one bounded, single-line `tool: output` for each other
    terminal step (what was actually done - "save_note: Saved note ...",
    "create_reminder: Reminder set ...").
  * A note when steps were skipped as unnecessary.

Step outputs are tool results (untrusted data): they are only ever rendered
as text here, never interpreted. Sizes are bounded by the constants in
app/orchestration/base.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from app.orchestration.base import (
    MAX_FINAL_RESULT_CHARS,
    MAX_OUTCOME_LINE_CHARS,
    PlanStep,
    StepStatus,
)

_TRUNCATED = "...[report truncated]"


def render_step_output(result: dict | None) -> str:
    """The text a step's stored result stands for: its message, else its
    data (a string as-is, anything structured as compact JSON), else
    "done"."""
    if not result:
        return "done"
    message = result.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    data = result.get("data")
    if data in (None, "", [], {}):
        return "done"
    if isinstance(data, str):
        return data.strip() or "done"
    return json.dumps(data, default=str, ensure_ascii=False)


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        return collapsed[: limit - 3] + "..."
    return collapsed


def terminal_steps(steps: list[PlanStep]) -> list[PlanStep]:
    """Succeeded steps that no non-skipped step depends on, in plan order."""
    depended_on = {
        dep for step in steps if step.status != StepStatus.SKIPPED for dep in step.depends_on
    }
    return [s for s in steps if s.status == StepStatus.SUCCEEDED and s.id not in depended_on]


def build_final_report(
    steps: list[PlanStep], *, produces_report: Callable[[PlanStep], bool]
) -> str:
    """Build the user-facing result of a completed plan - see this
    module's docstring. `produces_report(step)` says whether a step's tool
    declared `produces_report` (JobEngine looks the tool up)."""
    succeeded = [s for s in steps if s.status == StepStatus.SUCCEEDED]
    skipped = [s for s in steps if s.status == StepStatus.SKIPPED]

    report_step = next((s for s in reversed(succeeded) if produces_report(s)), None)
    outcome_steps = [s for s in terminal_steps(steps) if s is not report_step]

    body = render_step_output(report_step.result) if report_step is not None else ""
    lines = [
        f"{s.tool_name}: {_one_line(render_step_output(s.result), MAX_OUTCOME_LINE_CHARS)}"
        for s in outcome_steps
    ]

    if body and lines:
        text = body + "\n\nAlso done:\n" + "\n".join(lines)
    elif body:
        text = body
    elif lines:
        text = "\n".join(lines)
    else:
        text = "All steps completed."

    if skipped:
        text += f"\n({len(skipped)} step(s) skipped as unnecessary.)"

    if len(text) > MAX_FINAL_RESULT_CHARS:
        text = text[: MAX_FINAL_RESULT_CHARS - len(_TRUNCATED)] + _TRUNCATED
    return text
