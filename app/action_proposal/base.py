"""
The plain data an Action Proposal is made of. IN MEMORY ONLY - nothing here is stored, and nothing here can run anything.

A proposal is a few steps, each either THE USER'S ("Your part: ...") or JARVIS'S ("My part: ...", the only kind that has a `ToolAction`).
`action is None` is the whole distinction: only a step with an action can ever become a plan step in Slice 3, so a physical step - "connect
the joystick" - can never be offered as something Jarvis will do.

Every string that came from a stored record (a task, a project) has already been through `safe_text` and sits in a quoted slot of a
sentence Jarvis wrote: it is DATA to show, never an instruction, and nothing here parses it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MAX_STEPS = 4
TTL_SECONDS = 600.0  # an absolute lifetime: a follow-up question never extends it
MAX_PENDING = 20  # proposals held at once (one per conversation), oldest evicted
MAX_STEP_TEXT = 260
MAX_ARGUMENT_TEXT = 100

NOTE_TITLE_LIMIT = 100  # a saved plan's title, and the largest it may ever be
NOTE_CONTENT_LIMIT = 400  # ... and its text: single line, built only from what the proposal already showed

#: The only tools a proposal may ever name, and the only arguments each may carry. A tool must ALSO be registered, plannable, schema-valid
#: and a SAFE_ACTION (checked against the live registry whenever a proposal is built, again just before it is handed on, and once more by
#: the handoff that starts it).
ALLOWED_ARGUMENTS = {
    "create_reminder": frozenset({"at", "day", "text"}),
    "save_note": frozenset({"title", "content"}),
}
ALLOWED_TOOLS = frozenset({"create_reminder", "save_note"})
ALLOWED_PERMISSIONS = frozenset({"safe_action"})


@dataclass(frozen=True)
class ToolAction:
    """One registered tool call that Jarvis would make after approval. `arguments` is a sorted tuple of pairs, so it is immutable and its
    digest is stable."""

    tool_name: str
    arguments: tuple[tuple[str, Any], ...]

    def as_arguments(self) -> dict[str, Any]:
        return dict(self.arguments)


@dataclass(frozen=True)
class ProposalRef:
    """A record a step depends on: `ref` is its stable id ("task:<id>"), `stamp` fingerprints the fields the proposal was built from, and
    `label` is its text at the time (bounded, for telling the user what changed)."""

    ref: str
    stamp: str
    label: str


@dataclass(frozen=True)
class ProposalStep:
    text: str  # exactly what the user is shown for this step
    action: ToolAction | None = None  # None: the user's own step
    refs: tuple[ProposalRef, ...] = ()
    why: str = ""  # a deterministic reason, for "why" and "what does step N mean" - never model-written

    @property
    def mine(self) -> bool:
        return self.action is not None


@dataclass(frozen=True)
class ActionProposal:
    id: str  # opaque and random: names this proposal in the pending map and in provenance, never typed by the user
    conversation_id: str
    subject: str  # a label for display and the later Job's goal (untrusted-derived: bounded, shown quoted)
    steps: tuple[ProposalStep, ...]
    digest: str  # over the steps' text, actions and refs (with their stamps): what approval is bound to
    made_on: str  # the local date it was built (a "today" reminder must still be today when it is approved)

    @property
    def my_steps(self) -> tuple[int, ...]:
        """The 1-based positions of the steps Jarvis would do."""
        return tuple(i for i, step in enumerate(self.steps, start=1) if step.mine)


@dataclass(frozen=True)
class ApprovedProposal:
    """What an approval hands to Slice 3, once - never before approval, and only after revalidation. `plan_steps` are the proposal's
    action steps re-validated against the live registry (fresh PlanSteps, ready for `SQLiteJobStore.create_running_job`)."""

    proposal_id: str
    conversation_id: str
    subject: str
    steps: tuple[ProposalStep, ...]
    plan_steps: tuple[Any, ...]
    digest: str
    approved_at: datetime


def task_stamp(task: Any) -> str:
    """Fingerprint of everything about a task a proposal was built from."""
    status = getattr(task.status, "value", task.status)
    priority = getattr(task.priority, "value", task.priority)
    raw = "|".join(str(part) for part in (status, priority, task.due_date, task.updated_at, task.text))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def proposal_digest(subject: str, steps: tuple[ProposalStep, ...]) -> str:
    """A stable hash of what the user was shown and would approve: each step's text, tool call and referents (with their stamps)."""
    canonical = [
        {
            "text": step.text,
            "action": None if step.action is None else [step.action.tool_name, [[k, v] for k, v in step.action.arguments]],
            "refs": [[ref.ref, ref.stamp] for ref in step.refs],
        }
        for step in steps
    ]
    return hashlib.sha256(json.dumps([subject, canonical], sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
