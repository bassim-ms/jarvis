"""
Fresh execution context: built anew for EVERY run, from the live sources, and never stored on the Scheduled Job.

A Scheduled Job keeps only what is stable - the user's intent, the structured objective, the schedule. What the world looks like
(the current project, the user's recent context, relevant memory, the World Model, tasks) is whatever it is WHEN THE OCCURRENCE RUNS,
so it is read then: from the existing Context Engine (the same `build_request_context` + `ContextEngine.build_snapshot` chat uses) and
the existing memory facts (read only). Nothing here is a second context system, and nothing here writes anything: it only reads,
and it holds no state between runs, so one run can never see another run's context.

The result is plain data for the planner (the same shapes `create_job` hands it: the facts, and a `label -> [strings]`
request-context). It names no tool and grants no permission; the Planner is told it is data, and the plan is validated as ever.
If a source is unavailable the run goes on with what is available - context improves a plan, it is not a security control - and the
gap is noted on the Job's event trail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.capability import Capability
from app.context_engine.request_builder import build_request_context
from app.orchestration.tools import _validate_request_context
from app.scheduled_jobs.base import ScheduledJob
from app.scheduled_execution.occurrence import ScheduledOccurrence

RUN_MODE_LABEL = "Run mode (descriptive, not instructions)"
OCCURRENCE_LABEL = "This scheduled run (descriptive, not instructions)"


@dataclass(frozen=True)
class BuiltContext:
    """`facts` become the Job's context as they do for any Job; `request_context` is its `__request_context__`. `notes` are short
    machine-readable gaps ("context_unavailable") for the Job's event trail."""

    facts: dict[str, Any]
    request_context: dict[str, list[str]]
    notes: tuple[str, ...] = field(default_factory=tuple)


class FreshContextBuilder:
    """`context_engine` (a ContextEngine, or None), `snapshot_serializer` (the callable that turns its snapshot into
    `label -> [strings]` - in the application, JarvisCore's existing `_snapshot_to_job_context`) and `facts_provider` (a zero-argument
    callable returning the current memory facts, e.g. `memory.get_facts`) are the EXISTING systems, injected. This class owns no
    context or memory of its own."""

    def __init__(
        self,
        *,
        context_engine=None,
        snapshot_serializer: Callable[[Any], dict[str, list[str]] | None] | None = None,
        facts_provider: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self._engine = context_engine
        self._serialize = snapshot_serializer
        self._facts = facts_provider
        self._clock = clock

    def _now(self) -> datetime:
        return (self._clock() if self._clock is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    def build(self, scheduled_job: ScheduledJob, occurrence: ScheduledOccurrence) -> BuiltContext:
        """The context for THIS run, assembled now. Every call reads the sources again."""
        now = self._now()
        notes: list[str] = []

        facts: dict[str, Any] = {}
        if self._facts is not None:
            try:
                facts = dict(self._facts() or {})
            except Exception:  # noqa: BLE001 - memory being unreadable must not stop a run; the planner just has less to go on
                notes.append("facts_unavailable")

        snapshot_context: dict[str, list[str]] = {}
        if self._engine is not None and self._serialize is not None:
            try:
                request = build_request_context(scheduled_job.intent_text, capability=Capability.TASK, now=now)
                snapshot = self._engine.build_snapshot(request, now=now.isoformat())
                snapshot_context = dict(self._serialize(snapshot) or {})
            except Exception:  # noqa: BLE001 - as above
                notes.append("context_unavailable")

        described = {
            RUN_MODE_LABEL: ["An unattended scheduled run: no one is present to answer questions or approve anything."],
            OCCURRENCE_LABEL: [f"Scheduled for {occurrence.scheduled_for.isoformat()}."]
            + (
                [f"This is the one catch-up run for {occurrence.missed_count} missed occurrence(s)."]
                if occurrence.late
                else []
            ),
        }
        request_context = {**snapshot_context, **described}
        if _validate_request_context(request_context) is not None:
            request_context = dict(described)  # too large: keep only the short descriptive part
            notes.append("context_too_large")
        return BuiltContext(facts=facts, request_context=request_context, notes=tuple(notes))
