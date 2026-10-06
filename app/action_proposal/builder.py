"""
Building a proposal from a Focus decision - deterministic, no model, no Planner.

The steps come from what Focus already ranked: the first thing (and the next) are THE USER'S steps - "Your part: ..." - because Jarvis
cannot do their work for them. The only step that can be Jarvis's is an assistive action built from a fixed template and checked by the
existing plan validator against the live registry; today that is one: a reminder ahead of a task's own TIMED due time. Nothing is
invented: a task with no timed deadline gets no reminder, and a proposal with no Jarvis step has nothing to approve (see service.py).

Every referent is fingerprinted (`task_stamp`) so the proposal can be recognised as stale later.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.action_proposal.base import (
    ALLOWED_PERMISSIONS,
    ALLOWED_TOOLS,
    MAX_ARGUMENT_TEXT,
    MAX_STEP_TEXT,
    MAX_STEPS,
    ActionProposal,
    ProposalRef,
    ProposalStep,
    ToolAction,
    proposal_digest,
    task_stamp,
)
from app.context_engine.scoring import (
    REASON_PRIORITY_HIGH,
    REASON_TIME_FAR_FUTURE,
    REASON_TIME_NOW,
    REASON_TIME_OVERDUE,
    REASON_TIME_THIS_WEEK,
    REASON_TIME_TODAY,
    REASON_TIME_TOMORROW,
)
from app.focus.base import FocusDecision
from app.focus.render import quoted
from app.morning_brief.gather import safe_text
from app.orchestration.base import PlanStep, RetryPolicy
from app.orchestration.validator import PlanValidationError, validate_plan
from app.tasks.base import TaskStatus

REMINDER_LEAD = timedelta(minutes=30)  # a reminder this long before a task's own timed due time
MIN_REMINDER_AHEAD = timedelta(minutes=2)  # ... and only if that moment is still comfortably ahead
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
_DUE_WORDS = {
    REASON_TIME_OVERDUE: "it's overdue",
    REASON_TIME_NOW: "it's due within the hour",
    REASON_TIME_TODAY: "it's due today",
    REASON_TIME_TOMORROW: "it's due tomorrow",
    REASON_TIME_THIS_WEEK: "it's due this week",
    REASON_TIME_FAR_FUTURE: "it's due later",
}


def reminder_zone(tools) -> ZoneInfo | None:
    """The time zone a NEW reminder is created in - the reminder service's own answer, because `at` is read in THAT zone, never in whatever
    zone the clock Focus uses happens to be. None if it cannot be determined (then no reminder is offered or approved: the tool would
    refuse it too)."""
    try:
        name = tools.get("create_reminder").service.zone_name_or_none()
        return ZoneInfo(name) if name else None
    except Exception:  # noqa: BLE001
        return None


class ProposalError(ValueError):
    """A proposed tool call is not one a proposal may make (not allowed, not valid against the live registry, or not READ / SAFE_ACTION)."""


def plan_steps_for(tools, actions: list[ToolAction]) -> list[PlanStep]:
    """The existing plan validator, applied to a proposal's tool calls against the LIVE registry: the tool exists and is plannable, the
    arguments fit its schema, the permission comes from the tool itself - and it must be READ or SAFE_ACTION. Raises ProposalError."""
    raw = [
        {"id": f"a{i}", "tool": action.tool_name, "arguments": action.as_arguments(), "depends_on": []}
        for i, action in enumerate(actions, start=1)
    ]
    try:
        steps = validate_plan(
            raw,
            tools,
            max_steps=MAX_STEPS,
            default_retry=RetryPolicy(),
            allowed_tool_names=ALLOWED_TOOLS,
        )
    except PlanValidationError as exc:
        raise ProposalError(str(exc)) from exc
    for step in steps:
        if step.permission not in ALLOWED_PERMISSIONS:
            raise ProposalError(f"'{step.tool_name}' is not a read-only or safe action.")
    return steps


def _why(candidate) -> str:
    phrases = [_DUE_WORDS[r] for r in candidate.reasons if r in _DUE_WORDS]
    if REASON_PRIORITY_HIGH in candidate.reasons:
        phrases.append("you marked it high priority")
    if candidate.matched:
        phrases.append("it mentions " + ", ".join(quoted(word) for word in candidate.matched[:3]))
    elif candidate.via:
        phrases.append(f"it shares a word with your goal {quoted(safe_text(candidate.via, MAX_ARGUMENT_TEXT))} (a looser match)")
    return "; ".join(phrases)


class ProposalBuilder:
    def __init__(self, *, tasks, tools, clock):
        self.tasks = tasks
        self.tools = tools
        self._clock = clock

    def build(self, decision: FocusDecision, conversation_id: str) -> ActionProposal | None:
        """A proposal for a PICK decision, or None if there is nothing concrete to propose. Reads the tasks it will refer to."""
        now = self._now()
        zone = reminder_zone(self.tools)
        local_now = now.astimezone(zone) if zone is not None else now  # reminder times are the reminder service's zone's
        steps: list[ProposalStep] = []
        first_task = None
        for position, pick in enumerate(decision.picks[:2]):
            task = self._task(pick.ref)
            if task is None:
                continue
            first_task = first_task or (task, pick)
            label = safe_text(task.text, MAX_ARGUMENT_TEXT) or "(untitled task)"
            lead = "work on" if not steps else "then"
            steps.append(
                ProposalStep(
                    text=safe_text(f"Your part: {lead} {quoted(label)}.", MAX_STEP_TEXT + 40),
                    refs=(ProposalRef(pick.ref, task_stamp(task), label),),
                    why=_why(pick),
                )
            )
        if not steps:
            return None
        reminder = self._reminder_step(first_task[0], local_now) if first_task and zone is not None else None
        if reminder is not None and len(steps) < MAX_STEPS:
            steps.append(reminder)
        subject = safe_text(decision.project or decision.subject, MAX_ARGUMENT_TEXT)
        frozen = tuple(steps)
        return ActionProposal(
            id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            subject=subject,
            steps=frozen,
            digest=proposal_digest(subject, frozen),
            made_on=local_now.date().isoformat(),
        )

    # -- helpers ---------------------------------------------------------------------------------------------------------------------

    def _now(self) -> datetime:
        now = self._clock()
        return now if now.tzinfo is not None else now.astimezone()

    def _task(self, ref: str):
        if not ref.startswith("task:"):
            return None
        try:
            task = self.tasks.get(ref[len("task:") :])
        except Exception:  # noqa: BLE001 - an unreadable task is simply not part of the proposal
            return None
        return task if task is not None and task.status is TaskStatus.TODO else None

    def _reminder_step(self, task, now: datetime) -> ProposalStep | None:
        """A reminder ahead of the task's own TIMED due time - or nothing. A date-only or missing due date has no time to remind before."""
        due_text = (task.due_date or "").strip()
        if not due_text or _DATE_ONLY.fullmatch(due_text):
            return None
        try:
            due = datetime.fromisoformat(due_text)
        except ValueError:
            return None
        due = due.astimezone(now.tzinfo) if due.tzinfo is not None else due.replace(tzinfo=now.tzinfo)
        remind_at = due - REMINDER_LEAD
        if remind_at < now + MIN_REMINDER_AHEAD or remind_at.date() != now.date():
            return None
        text = f"{safe_text(task.text, 60)} is due at {due:%H:%M}"
        action = ToolAction(
            "create_reminder",
            tuple(sorted({"at": f"{remind_at:%H:%M}", "day": "today", "text": text}.items())),
        )
        try:
            plan_steps_for(self.tools, [action])
        except ProposalError:
            return None  # the tool is not there or not valid right now: no reminder step, rather than one that cannot run
        label = safe_text(task.text, MAX_ARGUMENT_TEXT)
        return ProposalStep(
            text=safe_text(f"My part: I'll set a reminder for {remind_at:%H:%M} today: {quoted(text)}", MAX_STEP_TEXT + 40),
            action=action,
            refs=(ProposalRef(f"task:{task.id}", task_stamp(task), label),),
            why=f"it's {int(REMINDER_LEAD.total_seconds() // 60)} minutes before the task's own due time",
        )
