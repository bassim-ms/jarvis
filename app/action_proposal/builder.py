"""
Building a proposal from a Focus decision - deterministic, no model, no Planner.

The steps come from what Focus already ranked: the first thing (and the next) are THE USER'S steps - "Your part: ..." - because Jarvis
cannot do their work for them. The only step that can be Jarvis's is an assistive action built from a fixed template and checked by the
existing plan validator against the live registry; today there are three: a reminder ahead of a task's own TIMED due time, and - only when
the user asked for each ("... and write it in a note", "... and add a task to <what>") - saving the plan itself as a note, and adding the one
task they named. Nothing is invented: a task with no timed deadline gets no reminder, a note or a task is never offered unasked, and a
proposal with no Jarvis step has nothing to approve (see service.py).

The note's title and text are built here, once, from text the proposal ALREADY shows (the subject and the task labels), and the step shows
them in full: what the user reads is, character for character, what would be saved. They are never rebuilt after approval.

The task's text is the user's own words, taken from their request, accepted only if `task_text_ok` (one plain line, bounded, unchanged by the
safe-text handling), shown in full in a quoted slot and frozen as the step's only argument: `create_task` runs with its own defaults for
everything else. It is NOT checked against the tasks the user already has: Tasks have no uniqueness rule (two open tasks may have the same
words - a chore that comes round again), so identical text does not prove a duplicate, and nothing here decides that it does. What stops one
approval creating two tasks is the pop-once approval and the single-attempt Job (service.py, handoff.py), not a comparison of text.

Every referent is fingerprinted (`task_stamp`) so the proposal can be recognised as stale later.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.action_proposal.base import (
    ALLOWED_ARGUMENTS,
    ALLOWED_PERMISSIONS,
    ALLOWED_TOOLS,
    MAX_ARGUMENT_TEXT,
    MAX_STEP_TEXT,
    MAX_STEPS,
    NOTE_CONTENT_LIMIT,
    NOTE_TITLE_LIMIT,
    ActionProposal,
    ProposalRef,
    ProposalStep,
    ToolAction,
    proposal_digest,
    task_stamp,
    task_text_ok,
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

NOTE_TOOL = "save_note"
TASK_TOOL = "create_task"
NOTE_STEP_LIMIT = NOTE_TITLE_LIMIT + NOTE_CONTENT_LIMIT + 100  # the note step is shown IN FULL, so its text bound covers its two arguments
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
    arguments fit its schema, the permission comes from the tool itself - and it must be READ or SAFE_ACTION. A proposal may also give a tool
    ONLY the arguments base.py names for it (a tool may declare more - `create_task` takes a priority and a due date - that no proposal sets).
    Raises ProposalError."""
    for action in actions:
        extra = set(action.as_arguments()) - ALLOWED_ARGUMENTS.get(action.tool_name, frozenset(action.as_arguments()))
        if extra:
            raise ProposalError(f"'{action.tool_name}' may not be given: {', '.join(sorted(extra))}.")
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


def is_note_step(step: ProposalStep) -> bool:
    return step.action is not None and step.action.tool_name == NOTE_TOOL


def is_task_step(step: ProposalStep) -> bool:
    return step.action is not None and step.action.tool_name == TASK_TOOL


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

    def build(self, decision: FocusDecision, conversation_id: str, *, note: bool = False, task_text: str = "") -> ActionProposal | None:
        """A proposal for a PICK decision, or None if there is nothing concrete to propose. Reads the tasks it will refer to. `note` is True
        only when the user asked for the plan to be saved as a note; `task_text` is non-empty only when they asked for a task to be added -
        it is offered only if the proposal still has room (never by dropping or squeezing another step) and the text is acceptable."""
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
        made_on = local_now.date().isoformat()
        if note and len(steps) < MAX_STEPS:
            note_step = self._note_step(subject, made_on, steps)
            if note_step is not None:
                steps.append(note_step)
        if task_text and len(steps) < MAX_STEPS:  # after everything already offered: a full proposal is never squeezed or trimmed for it
            task_step = self._task_step(task_text)
            if task_step is not None:
                steps.append(task_step)
        frozen = tuple(steps)
        return ActionProposal(
            id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            subject=subject,
            steps=frozen,
            digest=proposal_digest(subject, frozen),
            made_on=made_on,
        )

    def refresh_note(self, subject: str, made_on: str, steps: tuple[ProposalStep, ...]) -> tuple[ProposalStep, ...]:
        """`steps` after one was skipped: a note step is rebuilt, in the same place, from the steps that REMAIN (its text listed the skipped
        one), or dropped if it cannot be. Every other step - a task step included - is untouched."""
        note_at = next((position for position, step in enumerate(steps) if is_note_step(step)), None)
        if note_at is None:
            return steps
        kept = [step for step in steps if not is_note_step(step)]
        note_step = self._note_step(subject, made_on, kept)
        if note_step is None:
            return tuple(kept)
        return tuple(kept[:note_at]) + (note_step,) + tuple(kept[note_at:])

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

    def _note_step(self, subject: str, made_on: str, steps: list[ProposalStep]) -> ProposalStep | None:
        """"Save this plan as a note" - its title and text built from what the user is already shown, shown in full, or no step at all. The
        text names only the user's own steps: it says nothing about what Jarvis has done, because nothing has been done yet."""
        mine = [step for step in steps if step.action is None and step.refs]
        if not mine:
            return None
        parts = [
            f"{'work on' if position == 0 else 'then'} {safe_text(step.refs[0].label, 60)}" for position, step in enumerate(mine)
        ]
        about = f" for {safe_text(subject, 60)}" if subject else ""
        title = safe_text(f"Plan: {subject}", NOTE_TITLE_LIMIT) if subject else "Plan"
        content = f"Plan{about} ({made_on}): " + "; ".join(parts) + "."
        text = f'My part: I\'ll save this plan as a note titled {quoted(title)} saying {quoted(content)}'
        if len(content) > NOTE_CONTENT_LIMIT or len(text) > NOTE_STEP_LIMIT or safe_text(content, NOTE_CONTENT_LIMIT) != content:
            return None  # never shown or saved cut short
        shown = safe_text(text, NOTE_STEP_LIMIT)
        if shown != text:
            return None
        action = ToolAction(NOTE_TOOL, tuple(sorted({"title": title, "content": content}.items())))
        try:
            plan_steps_for(self.tools, [action])
        except ProposalError:
            return None  # the notes tool is not there or not valid right now: no note step, rather than one that cannot run
        refs = tuple({ref.ref: ref for step in mine for ref in step.refs}.values())
        return ProposalStep(text=text, action=action, refs=refs, why="so you have a written copy of the plan to come back to")

    def _task_step(self, text: str) -> ProposalStep | None:
        """"Add this task" - the user's own words, shown in full in a quoted slot, exactly as they will be stored, or no step at all. The ONLY
        argument is the text: no priority, no due date, no id, and nothing that refers to a task already on the list."""
        if not task_text_ok(text):
            return None  # never shown or stored changed, cut short, or hiding something
        action = ToolAction(TASK_TOOL, (("text", text),))
        try:
            plan_steps_for(self.tools, [action])
        except ProposalError:
            return None  # the tool is not there or not valid right now: no task step, rather than one that cannot run
        shown = f"My part: I'll add a task {quoted(text)} to your list."
        if safe_text(shown, MAX_STEP_TEXT + 40) != shown:
            return None
        return ProposalStep(text=shown, action=action, why="because you asked me to add it")

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
