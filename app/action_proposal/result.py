"""
Slice 4: what a started Action Proposal Job turned out to be - checked, told ONCE, in the conversation it came from, and then left alone.

    handoff starts a Job  ->  `watch(job)`  ->  Job reaches a terminal state  ->  ONE pending report  ->  the user's NEXT message in that
    same conversation carries it (and consumes it)  ->  STOP

PULL, not push: nothing is sent unsolicited, no notification channel is added, nothing is stored (a restart forgets pending reports, and the
Job itself still answers "is that finished?"). The registry is bounded and in memory; a full registry evicts its OLDEST entry, so one chatty
conversation can never grow it without limit, and an entry is only ever shown in the conversation it is bound to.

TRUTH is deterministic. It comes from the Job's status, each step's status and the tool result the engine actually stored - never from a
model, never from the PlanStep's intended arguments. "It worked" needs the Job COMPLETED, every step SUCCEEDED and a stored result whose own
status is "success" with a message; anything less is reported as what it is (failed, interrupted, cancelled, or "I can't confirm it").

THIS MODULE ONLY READS. It never creates a Job, runs a tool, approves or builds a proposal, calls a planner or model, retries or continues
anything. A suggestion is a sentence: acting on it starts over at "make a plan" -> approval, like everything else.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from app.morning_brief.gather import safe_text
from app.orchestration.base import TERMINAL_JOB_STATUSES, JobStatus, StepStatus

MAX_WATCHES = 50  # Jobs being waited on, across all conversations; the oldest is dropped first
MAX_REPORTS = 50  # finished reports waiting for a message, across all conversations; the oldest is dropped first
MAX_REPORTS_PER_CONVERSATION = 3  # more than this and the oldest of THAT conversation is dropped
MAX_SEEN = 500  # job ids already watched or reported, so a repeat can never present twice
MAX_LINE = 240

_GOAL_PREFIX = "Action plan:"

SUCCESS_SUGGESTION = "If you'd like a next step, ask me what to focus on and I'll propose it - nothing runs unless you approve it."
FAILURE_SUGGESTION = "Nothing was retried or changed. If you still want this, ask me to make a plan again and I'll show it to you first."


# -- the check (pure) ------------------------------------------------------------------------------------------------------------------


def _line(text: Any) -> str:
    return safe_text(text if isinstance(text, str) else "", MAX_LINE)


def _subject(job: Any) -> str:
    goal = _line(getattr(job, "goal", ""))
    return (goal[len(_GOAL_PREFIX):].strip() if goal.startswith(_GOAL_PREFIX) else goal) or "that"


def _stored_message(step: Any) -> str | None:
    """The message of the tool result the engine actually stored for a step that SUCCEEDED, or None when it is not a real success result."""
    result = getattr(step, "result", None)
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    message = _line(result.get("message"))
    return message or None


def _steps(job: Any) -> list[Any]:
    plan = getattr(job, "plan", None)
    steps = getattr(plan, "steps", None)
    return list(steps) if isinstance(steps, (list, tuple)) else []


def verify(job: Any) -> tuple[str, list[str]]:
    """(kind, facts) for a TERMINAL Job. kind is "succeeded", "failed", "cancelled" or "unclear"; `facts` are one line per step, from what
    was actually stored. Never "succeeded" unless the Job and every step really succeeded."""
    status = getattr(job, "status", None)
    steps = _steps(job)
    facts: list[str] = []
    all_confirmed = bool(steps)
    for number, step in enumerate(steps, start=1):
        step_status = getattr(step, "status", None)
        message = _stored_message(step) if step_status is StepStatus.SUCCEEDED else None
        if step_status is StepStatus.SUCCEEDED and message is not None:
            facts.append(f"Step {number}: {message}")
            continue
        all_confirmed = False
        if step_status is StepStatus.SUCCEEDED:
            facts.append(f"Step {number}: marked done, but I have no result to confirm it.")
        elif step_status is StepStatus.FAILED:
            stored = getattr(step, "result", None)
            reason = _line(getattr(step, "error", None)) or _line(stored.get("message") if isinstance(stored, dict) else None)
            facts.append(f"Step {number} didn't work: {reason or 'no reason was recorded'}")
        else:
            facts.append(f"Step {number} did not run.")
    if status is JobStatus.COMPLETED:
        return ("succeeded" if all_confirmed else "unclear"), facts
    if status is JobStatus.FAILED:
        return "failed", facts
    if status is JobStatus.CANCELLED:
        return "cancelled", facts
    return "unclear", facts


def render(job: Any) -> str:
    """The one report for a terminal Job, in plain words."""
    kind, facts = verify(job)
    subject = _subject(job)
    body = "\n".join(f"- {fact}" for fact in facts)
    if kind == "succeeded":
        return f"Update on \"{subject}\": that finished, and I checked the result.\n{body}\n{SUCCESS_SUGGESTION}"
    if kind == "failed":
        reason = _line(getattr(job, "error", None))
        head = f"Update on \"{subject}\": that did not finish." + (f" {reason}" if reason else "")
        return f"{head}\n{body}\n{FAILURE_SUGGESTION}" if body else f"{head}\n{FAILURE_SUGGESTION}"
    if kind == "cancelled":
        head = f"Update on \"{subject}\": that was cancelled before it finished."
        return f"{head}\n{body}" if body else head
    head = f"Update on \"{subject}\": it is marked finished, but I can't confirm it worked, so please don't rely on it."
    return f"{head}\n{body}" if body else head


# -- the registry ----------------------------------------------------------------------------------------------------------------------


class ResultReports:
    """`get(job_id)` reads a Job (the store's own `get`, injected: this class never writes). `watch(job)` is called once, after the handoff
    inserted the Job; `poll()` turns watched Jobs that reached a terminal state into pending reports; `take(conversation_id)` returns and
    consumes that conversation's reports."""

    def __init__(self, get: Callable[[str], Any]):
        self._get = get
        self._lock = threading.Lock()
        self._watches: OrderedDict[str, str] = OrderedDict()  # job id -> conversation id
        self._reports: OrderedDict[str, tuple[str, str]] = OrderedDict()  # job id -> (conversation id, text)
        self._seen: OrderedDict[str, None] = OrderedDict()

    def watch(self, job: Any) -> bool:
        """Wait for this Job. A Job with no id or conversation, or one already watched, reported or consumed, is ignored."""
        job_id = getattr(job, "id", None)
        conversation_id = getattr(job, "conversation_id", None)
        if not isinstance(job_id, str) or not job_id or not isinstance(conversation_id, str) or not conversation_id:
            return False
        with self._lock:
            if job_id in self._seen:
                return False
            self._seen[job_id] = None
            while len(self._seen) > MAX_SEEN:
                self._seen.popitem(last=False)
            self._watches[job_id] = conversation_id
            while len(self._watches) > MAX_WATCHES:
                self._watches.popitem(last=False)
        return True

    def poll(self) -> None:
        """Move watched Jobs that are now terminal to pending reports (one each). Read-only on the store; never raises."""
        with self._lock:
            watched = list(self._watches.items())
        for job_id, conversation_id in watched:
            try:
                job = self._get(job_id)
            except Exception:  # noqa: BLE001 - an unreadable Job is simply tried again next time
                continue
            if job is None:
                with self._lock:
                    self._watches.pop(job_id, None)
                continue
            if getattr(job, "status", None) not in TERMINAL_JOB_STATUSES:
                continue
            try:
                text = render(job)
            except Exception:  # noqa: BLE001 - a Job that cannot be rendered is reported as unreadable, never as a success
                text = "Update: a task I started has finished, but I couldn't read its result, so please don't rely on it."
            with self._lock:
                if self._watches.pop(job_id, None) is None:
                    continue  # already handled
                self._reports[job_id] = (conversation_id, text)
                mine = [key for key, (conv, _) in self._reports.items() if conv == conversation_id]
                for key in mine[: max(0, len(mine) - MAX_REPORTS_PER_CONVERSATION)]:
                    del self._reports[key]
                while len(self._reports) > MAX_REPORTS:
                    self._reports.popitem(last=False)

    def take(self, conversation_id: str) -> str | None:
        """This conversation's pending reports, oldest first, removed so they cannot be shown twice. Another conversation's are never touched."""
        self.poll()
        with self._lock:
            mine = [key for key, (conv, _) in self._reports.items() if conv == conversation_id]
            if not mine:
                return None
            texts = [self._reports.pop(key)[1] for key in mine]
        return "\n\n".join(texts)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._reports)

    @property
    def watching(self) -> int:
        with self._lock:
            return len(self._watches)
