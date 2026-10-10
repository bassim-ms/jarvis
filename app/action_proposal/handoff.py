"""
The Slice 3 handoff: an approved, revalidated Action Proposal becomes ONE running Job - and nothing else happens here.

    ApprovedProposal  ->  checked, again  ->  the injected `start` (the store's atomic insert of a Job born RUNNING)  ->  one fixed reply

This is `ActionProposalService.on_approved`: the only place a proposal leaves the package, and the only place a Job can come from it. It is
called once, after the proposal was popped and revalidated against live data (service.py), so a stale, expired, wrong-conversation, duplicate
or otherwise invalid approval never gets here.

WHAT IS STARTED is exactly what was approved: the validated `PlanStep`s Slice 2 built for the steps marked "My part", with their exact arguments.
Nothing is reinterpreted, regenerated, expanded or re-planned - there is no model, Planner or provider anywhere in this path, and the Job is
born RUNNING (a PENDING Job would be planned by the model over the top of it). The existing JobEngine then runs each step through the existing
ToolRegistry under the existing permission rules, with the existing single attempt, recovery and reporting. A step that is "Your part" has no
`PlanStep` at all: a physical action can never become a tool call.

THE ALLOW-LIST is four tools, `create_reminder`, `save_note`, `create_task` and `complete_task`, each a SAFE_ACTION with its own fixed argument
names (base.py); everything else - any other tool (`cancel_task` and `list_tasks` among them), a read-only one, one that needs confirmation,
an unknown name - is refused here even if something upstream let it through, and nothing is created. A note is checked exactly like everything
else, and more: its title and text must be present, single-line, within their bounds and exactly what was approved, and a plan carries at
most one. A task is the same: exactly one argument, `text` (no priority, due date or id), and it must be exactly what `task_text_ok` accepts;
a plan carries at most one. Marking a task done is stricter still, because it cannot be undone: exactly one argument, `task_id`, a 32-character
hex id that is THE one task the step refers to (its single `task:<id>` referent, carrying the words the user was shown); one such step per plan,
and a plan that has one does nothing else.

FAILURE is honest and final: if the insert fails nothing was started, the user is told so, and nothing is retried or substituted. The reply is
built BEFORE the insert; after it the only thing this handoff does is pass the new Job to the injected `on_started` (Slice 4 wires it to
`ResultReports.watch`), and an exception from that callback is swallowed, so it can never turn a created Job into a failed handoff. What the
callback does, and the rest of the result lifecycle, belong to result.py, not to this module.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.action_proposal import render as words
from app.action_proposal.base import (
    ALLOWED_ARGUMENTS,
    ALLOWED_PERMISSIONS,
    ALLOWED_TOOLS,
    COMPLETION_TEXT_LIMIT,
    MAX_ARGUMENT_TEXT,
    MAX_STEPS,
    NOTE_CONTENT_LIMIT,
    NOTE_TITLE_LIMIT,
    ApprovedProposal,
    proposal_digest,
    task_id_ok,
    task_text_ok,
)
from app.morning_brief.gather import safe_text
from app.orchestration.base import AutonomyLevel, PlanStep, RetryPolicy, StepStatus

_STEP_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

REFUSED = "I couldn't start that - what I would have run doesn't match what I showed you, so nothing was started."


class ExecutionHandoff:
    """`start(goal, steps, *, context, conversation_id, autonomy)` is the store's atomic insert of a Job born RUNNING with an already-validated
    plan (injected, so this package never touches a Job store). Called with an `ApprovedProposal`, the instance returns what the user is told."""

    def __init__(self, start: Callable[..., Any], on_started: Callable[[Any], Any] | None = None):
        self._start = start
        self._on_started = on_started  # Slice 4: told about the Job once it exists (result.py); never able to change the outcome

    def __call__(self, approved: ApprovedProposal) -> str:
        if self.problem(approved) is not None:
            return REFUSED  # nothing was created
        reply = self._reply(approved)  # built first: nothing fallible happens once a Job exists
        goal = safe_text(f"Action plan: {approved.subject}", MAX_ARGUMENT_TEXT) if approved.subject else "Action plan"
        try:
            job = self._start(
                goal,
                list(approved.plan_steps),
                context={"source": "action_proposal", "proposal_id": approved.proposal_id, "digest": approved.digest},
                conversation_id=approved.conversation_id,
                autonomy=AutonomyLevel.ASSISTED,  # the existing confirmation rules, exactly as for any other Job
            )
        except Exception:  # noqa: BLE001 - one attempt, no retry, no fallback Job: nothing was started and the user is told so
            return words.FAILED
        if self._on_started is not None and job is not None:
            try:
                self._on_started(job)
            except Exception:  # noqa: BLE001 - the Job exists and runs regardless; at worst its result is not volunteered later
                pass
        return reply

    # -- the check: what runs must be exactly what was approved, and nothing the allow-list does not name -----------------------------

    @staticmethod
    def problem(approved: object) -> str | None:
        """Why this approval must not start anything, or None if it is exactly an approved, allow-listed, fresh plan."""
        if not isinstance(approved, ApprovedProposal):
            return "not an approved proposal"
        if proposal_digest(approved.subject, approved.steps) != approved.digest:
            return "the proposal does not match its own digest"
        owners = [step for step in approved.steps if step.action is not None]  # the proposal's own steps that have a tool call
        actions = [step.action for step in owners]
        plan = list(approved.plan_steps)
        if not actions or len(plan) != len(actions) or len(plan) > MAX_STEPS:
            return "the steps to run are not the steps that were approved"  # a user-only step never becomes one, and none may be added
        if len({getattr(step, "id", None) for step in plan}) != len(plan):
            return "duplicate step ids"
        for step, action, owner in zip(plan, actions, owners):
            if not isinstance(step, PlanStep):
                return "not a plan step"
            if step.tool_name not in ALLOWED_TOOLS or step.tool_name != action.tool_name:
                return f"'{step.tool_name}' is not a tool a proposal may start"
            if step.permission not in ALLOWED_PERMISSIONS:
                return "not a safe action"
            if step.arguments != action.as_arguments():
                return "the arguments are not the approved ones"
            # Only the arguments the builder gives that tool (base.py) are what was approved; anything else is not.
            if set(step.arguments) - ALLOWED_ARGUMENTS[step.tool_name] or not all(isinstance(v, str) for v in step.arguments.values()):
                return "unexpected arguments"
            if step.tool_name == "save_note" and not ExecutionHandoff._note_ok(step.arguments):
                return "the note is not a plain, bounded title and text"
            if step.tool_name == "create_task" and not ExecutionHandoff._task_ok(step.arguments):
                return "the task is not a plain, bounded line of text"
            if step.tool_name == "complete_task" and not ExecutionHandoff._completion_ok(step.arguments, owner):
                return "the task to mark done is not exactly the one that was shown"
            if not isinstance(step.id, str) or _STEP_ID.fullmatch(step.id) is None:
                return "bad step id"
            if step.depends_on or step.expectation is not None or step.evaluation is not None:
                return "a proposal's steps are independent and carry no expectation"
            if step.status is not StepStatus.PENDING or step.attempts != 0 or step.confirmed or step.result is not None or step.error is not None:
                return "not a fresh step"
            if step.retry_policy != RetryPolicy():
                return "a proposal's steps are attempted once"
        if sum(1 for step in plan if step.tool_name == "save_note") > 1:
            return "a plan saves one note"  # a note is not idempotent: never two from one approval
        if sum(1 for step in plan if step.tool_name == "create_task") > 1:
            return "a plan adds one task"  # nor is a task: never two from one approval
        if any(step.tool_name == "complete_task" for step in plan) and len(plan) != 1:
            return "marking a task done is all such a plan does"  # one task, one step, nothing alongside it: it cannot be undone
        return None

    @staticmethod
    def _note_ok(arguments: dict) -> bool:
        """Exactly a title and a text, each non-blank, one clean line, within its bound - nothing a tool would have to trim or reinterpret."""
        if set(arguments) != ALLOWED_ARGUMENTS["save_note"]:
            return False
        for name, limit in (("title", NOTE_TITLE_LIMIT), ("content", NOTE_CONTENT_LIMIT)):
            value = arguments[name]
            if not isinstance(value, str) or not value or value != value.strip() or safe_text(value, limit) != value:
                return False
        return True

    @staticmethod
    def _task_ok(arguments: dict) -> bool:
        """Exactly one argument, `text`, and exactly a line the user's own words could have made - nothing a tool would have to trim, reinterpret
        or default for us (the tool's own defaults apply to everything else)."""
        return set(arguments) == ALLOWED_ARGUMENTS["create_task"] and task_text_ok(arguments["text"])

    @staticmethod
    def _completion_ok(arguments: dict, owner: object) -> bool:
        """Exactly `{"task_id": <a 32-hex id>}`, and that id is the one task the proposal's step refers to - by the single referent that
        carries the words the user was shown. Nothing here looks a task up: it only checks that what runs is what was frozen."""
        if set(arguments) != ALLOWED_ARGUMENTS["complete_task"] or not task_id_ok(arguments["task_id"]):
            return False
        refs = getattr(owner, "refs", ())
        if len(refs) != 1 or refs[0].ref != f"task:{arguments['task_id']}":
            return False
        label = refs[0].label
        return isinstance(label, str) and 0 < len(label) <= COMPLETION_TEXT_LIMIT and label == label.strip()

    @staticmethod
    def _reply(approved: ApprovedProposal) -> str:
        mine = [i for i, step in enumerate(approved.steps, start=1) if step.action is not None]
        place = f"step {mine[0]}" if len(mine) == 1 else "steps " + ", ".join(str(n) for n in mine[:-1]) + f" and {mine[-1]}"
        return f"Okay - I've started my part ({place}). Ask me \"is that finished?\" and I'll tell you how it went."
