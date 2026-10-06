"""
The focus service: read what the existing systems hold, work out which project (if any) the user means, rank what
could be done first, and decide whether to answer or ask.

READ-ONLY and deterministic. Every score is a sum of named, existing weights (`RelevanceWeights`):

    score = RELATION + URGENCY + PRIORITY

  RELATION  (only when the user named something): +EXPLICIT_MATCH if the task shares a distinctive word with what
            they named or the identified project's name; +RELATIONSHIP if it shares one with one of the project's
            active goals; otherwise it is not a candidate (an unclear link is uncertainty, never a guess).
  URGENCY   by the local clock: overdue, due within the hour, today, tomorrow, this week, later; no (or unreadable)
            due date adds nothing.
  PRIORITY  the task's own field: high / medium / low. Never inferred from wording.

Ties break on the earlier due moment, then the older task, then the id - never on store order. The calendar and the
user's stated activity are shown as constraints and NEVER change a ranking (an event title can come from someone
else). Nothing is stored: the "focus" exists only inside the one reply. A source that cannot be read is said so.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Callable

from app.context_engine.scoring import (
    DEFAULT_WEIGHTS,
    REASON_EXPLICIT_MATCH,
    REASON_PRIORITY_HIGH,
    REASON_PRIORITY_LOW,
    REASON_PRIORITY_MEDIUM,
    REASON_RELATIONSHIP_GOAL,
    REASON_TIME_FAR_FUTURE,
    REASON_TIME_NOW,
    REASON_TIME_OVERDUE,
    REASON_TIME_THIS_WEEK,
    REASON_TIME_TODAY,
    REASON_TIME_TOMORROW,
    priority_relevance,
)
from app.focus.base import (
    AMBIGUOUS,
    ASK_STEP,
    NO_FOCUS,
    NO_PROJECT,
    PICK,
    RELATION_GOAL,
    RELATION_NAMED,
    RELATION_NONE,
    Candidate,
    FocusDecision,
)
from app.focus.recognition import QUESTION, FocusRequest, distinctive_tokens, parse_focus_request
from app.focus.render import quoted as _quoted, render_decision
from app.morning_brief.gather import TEXT_LIMIT, gather_calendar, parse_when, safe_text
from app.tasks.base import TaskPriority, TaskStatus

REASON_DUE_UNREADABLE = "due_unreadable"

MAX_PICKS = 3  # the first thing, then up to two more
MAX_HEADS_UP = 2
MAX_OPTIONS = 4
MAX_GOALS = 3
MAX_CONSTRAINTS = 4
URGENT_REASONS = frozenset({REASON_TIME_OVERDUE, REASON_TIME_NOW, REASON_TIME_TODAY})
WEEK_DAYS = 7

_W = DEFAULT_WEIGHTS
_TASK_PRIORITY = {
    TaskPriority.HIGH: (_W.PRIORITY_HIGH, REASON_PRIORITY_HIGH),
    TaskPriority.MEDIUM: (_W.PRIORITY_MEDIUM, REASON_PRIORITY_MEDIUM),
    TaskPriority.LOW: (_W.PRIORITY_LOW, REASON_PRIORITY_LOW),
}
_ACTIVITY_LABELS = {"class": "in class", "studying": "studying", "meeting": "in a meeting", "focused_work": "in focused work"}


# --- ranking (pure) ----------------------------------------------------------------------------------------------------------------


def urgency(value: object, now: datetime) -> tuple[float, str | None, datetime | None]:
    """(weight, reason id, due moment) for a due date against the LOCAL clock. A date-only due date is the whole day, so
    "due today" is today, not overdue. No due date adds nothing; one that cannot be read adds nothing and says so."""
    if not isinstance(value, str) or not value.strip():
        return 0.0, None, None
    due = parse_when(value, now.tzinfo)
    if due is None:
        return 0.0, REASON_DUE_UNREADABLE, None
    moment, date_only = due
    days = (moment.date() - now.date()).days
    if (date_only and days < 0) or (not date_only and moment < now):
        return _W.TIME_OVERDUE, REASON_TIME_OVERDUE, moment
    if not date_only and moment - now <= timedelta(hours=1):
        return _W.TIME_NOW, REASON_TIME_NOW, moment
    if days <= 0:
        return _W.TIME_TODAY, REASON_TIME_TODAY, moment
    if days == 1:
        return _W.TIME_TOMORROW, REASON_TIME_TOMORROW, moment
    if days <= WEEK_DAYS:
        return _W.TIME_THIS_WEEK, REASON_TIME_THIS_WEEK, moment
    return _W.TIME_FAR_FUTURE, REASON_TIME_FAR_FUTURE, moment


def _relation_weight(relation: str) -> tuple[float, tuple[str, ...]]:
    if relation == RELATION_NAMED:
        return _W.EXPLICIT_MATCH, (REASON_EXPLICIT_MATCH,)
    if relation == RELATION_GOAL:
        return _W.RELATIONSHIP, (REASON_RELATIONSHIP_GOAL,)
    return 0.0, ()


def score_task(task, now: datetime, relation: str = RELATION_NONE, via: str = "", matched: tuple[str, ...] = ()) -> Candidate:
    weight, reasons = _relation_weight(relation)
    due_weight, due_reason, moment = urgency(task.due_date, now)
    priority_weight, priority_reason = _TASK_PRIORITY.get(task.priority, _TASK_PRIORITY[TaskPriority.LOW])
    if due_reason:
        reasons += (due_reason,)
    return Candidate(
        kind="task",
        ref=f"task:{task.id}",
        text=safe_text(task.text, TEXT_LIMIT) or "(untitled task)",
        score=weight + due_weight + priority_weight,
        reasons=reasons + (priority_reason,),
        due_ts=moment.timestamp() if moment else None,
        created_at=str(task.created_at or ""),
        via=via if relation == RELATION_GOAL else "",
        matched=matched if relation == RELATION_NAMED else (),
        due_text=safe_text(task.due_date, 40) if due_reason == REASON_DUE_UNREADABLE else "",
    )


def score_goal(goal, now: datetime) -> Candidate:
    due_weight, due_reason, moment = urgency(goal.target_date, now)
    priority_weight, priority_reason = priority_relevance(goal.priority)
    return Candidate(
        kind="goal",
        ref=f"goal:{goal.id}",
        text=safe_text(goal.title, TEXT_LIMIT) or "(untitled goal)",
        score=due_weight + priority_weight,
        reasons=((due_reason,) if due_reason else ()) + (priority_reason,),
        due_ts=moment.timestamp() if moment else None,
        created_at=str(goal.created_at or ""),
    )


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Highest score first; ties by the earlier due moment (none last), the older record, then the id."""
    return sorted(candidates, key=lambda c: (-c.score, c.due_ts if c.due_ts is not None else math.inf, c.created_at, c.ref))


def match_projects(hint: tuple[str, ...], projects: list) -> list:
    """The projects whose NAME shares the most distinctive words with the hint (at least one). Whole words only; several
    sharing the same best count are all returned - the caller must ask, never pick."""
    scored = []
    for project in projects:
        overlap = len(set(hint) & set(distinctive_tokens(project.name)))
        if overlap:
            scored.append((overlap, project))
    if not scored:
        return []
    best = max(overlap for overlap, _ in scored)
    return [project for overlap, project in scored if overlap == best]


# --- the service -------------------------------------------------------------------------------------------------------------------


class FocusService:
    """Every source is optional and read on its own: one that is missing or fails is reported in the answer, never raised."""

    def __init__(
        self,
        *,
        tasks=None,
        calendar=None,
        world_model=None,
        activity=None,
        clock: Callable[[], datetime],
    ):
        self.tasks = tasks
        self.calendar = calendar
        self.world_model = world_model
        self.activity = activity
        self._clock = clock

    # -- chat door ----------------------------------------------------------------------------------------------------------------

    def command_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        """The answer if `message` is a focus request that is theirs to answer, else None (ordinary conversation). Never raises."""
        request = parse_focus_request(message)
        if request is None:
            return None
        try:
            decision = self.decide(request)
            return None if decision is None else render_decision(decision)
        except Exception:  # noqa: BLE001 - a fault here is "not a focus request"
            return None

    # -- the decision -------------------------------------------------------------------------------------------------------------

    def decide(self, request: FocusRequest) -> FocusDecision | None:
        """None means "not mine": a statement that names nothing Jarvis knows about is ordinary conversation."""
        now = self._now()
        unavailable: list[str] = []
        tasks = self._read("tasks", unavailable, lambda: list(self.tasks.list(TaskStatus.TODO)), [])

        if request.kind == QUESTION:
            ranked = rank([score_task(task, now) for task in tasks])
            if not ranked:
                projects = self._read("projects", unavailable, lambda: list(self.world_model.get_active_projects()), [])
                decision = FocusDecision(NO_FOCUS)
                decision.options, decision.more_options = self._names(projects)
            else:
                decision = FocusDecision(PICK, picks=ranked[:MAX_PICKS], candidate_count=len(ranked))
        else:
            decision = self._objective(request, tasks, now, unavailable)
            if decision is None:
                return None

        decision.unavailable = unavailable
        if decision.outcome in (PICK, ASK_STEP):
            decision.constraints = self._constraints(now, unavailable)
        return decision

    def _objective(self, request: FocusRequest, tasks: list, now: datetime, unavailable: list[str]) -> FocusDecision | None:
        subject = safe_text(request.subject, TEXT_LIMIT)
        projects = self._read("projects", unavailable, lambda: list(self.world_model.get_active_projects()), [])
        matches = match_projects(request.hint, projects)

        if len(matches) > 1:
            decision = FocusDecision(AMBIGUOUS, subject=subject)
            decision.options, decision.more_options = self._names(matches)
            return decision
        if matches:
            return self._for_project(request, matches[0], tasks, now, unavailable, subject)

        # No project by that name. Open tasks that mention it can still be ranked - and the answer says there is no project.
        hint = set(request.hint)
        related = []
        for task in tasks:
            shared = tuple(word for word in distinctive_tokens(task.text) if word in hint)
            if shared:
                related.append(score_task(task, now, RELATION_NAMED, matched=shared))
        if related:
            ranked = rank(related)
            related_refs = {c.ref for c in ranked}
            return FocusDecision(
                PICK,
                subject=subject,
                picks=ranked[:MAX_PICKS],
                candidate_count=len(ranked),
                heads_up=self._heads_up(tasks, related_refs, now),
            )
        if request.named_project:  # they said "project" and nothing matches: ask rather than guess
            decision = FocusDecision(NO_PROJECT, subject=subject)
            decision.options, decision.more_options = self._names(projects)
            return decision
        return None  # "I need to finish my coffee": nothing Jarvis knows about it - ordinary conversation

    def _for_project(self, request, project, tasks, now, unavailable, subject) -> FocusDecision:
        name = safe_text(project.name, TEXT_LIMIT)
        goals = self._read("goals", unavailable, lambda: list(self.world_model.get_active_goals(project.id)), [])
        named = set(request.hint) | set(distinctive_tokens(project.name))
        goal_tokens = [(safe_text(g.title, TEXT_LIMIT), set(distinctive_tokens(g.title))) for g in goals]

        related: list[Candidate] = []
        for task in tasks:
            ordered = distinctive_tokens(task.text)
            shared = tuple(word for word in ordered if word in named)
            if shared:
                related.append(score_task(task, now, RELATION_NAMED, matched=shared))
                continue
            words = set(ordered)
            via = next((title for title, tokens in goal_tokens if words & tokens), "")
            if via:
                related.append(score_task(task, now, RELATION_GOAL, via))
        ranked = rank(related)

        if ranked:
            return FocusDecision(
                PICK,
                subject=subject,
                project=name,
                picks=ranked[:MAX_PICKS],
                candidate_count=len(ranked),
                heads_up=self._heads_up(tasks, {c.ref for c in ranked}, now),
            )
        ordered_goals = rank([score_goal(g, now) for g in goals])
        return FocusDecision(ASK_STEP, subject=subject, project=name, goals=[c.text for c in ordered_goals[:MAX_GOALS]])

    def _heads_up(self, tasks: list, related_refs: set[str], now: datetime) -> list[Candidate]:
        """Other open tasks that are overdue or due today: surfaced so urgent work is never silently dropped - not ranked as part of the ask."""
        others = [score_task(t, now) for t in tasks]
        urgent = [c for c in others if c.ref not in related_refs and URGENT_REASONS & set(c.reasons)]
        return rank(urgent)[:MAX_HEADS_UP]

    # -- constraints (shown, never ranked) ---------------------------------------------------------------------------------------

    def _constraints(self, now: datetime, unavailable: list[str]) -> list[str]:
        """What the user's own day imposes: their stated activity first, then what is on or about to start, then any overlap."""
        lines: list[str] = []
        try:
            context = self.activity.current() if self.activity is not None else None
            label = _ACTIVITY_LABELS.get(getattr(context, "value", None))
            if label:
                until = parse_when(getattr(context, "expires_at", None), now.tzinfo)
                tail = f" until {until[0].strftime('%H:%M')}" if until and until[0].date() == now.date() else ""
                lines.append(f"You've told me you're {label}{tail}.")
        except Exception:  # noqa: BLE001 - a stated activity is only a courtesy line
            pass
        items = self._read("calendar", unavailable, lambda: gather_calendar(self.calendar, now), [])
        timed = sorted((i for i in items if i.kind == "event" and i.when is not None), key=lambda i: i.when)
        soon = [i for i in timed if {"in_progress", "starts_soon"} & set(i.reasons)]
        for item in soon[:2]:
            if "in_progress" in item.reasons:
                lines.append(f"{_quoted(item.text)} is on right now.")
            else:
                minutes = max(0, int((item.when - now).total_seconds() // 60))
                lines.append(f"{_quoted(item.text)} starts in {minutes} min ({item.when.strftime('%H:%M')}).")
        if not soon and timed:
            lines.append(f"Next on your calendar: {_quoted(timed[0].text)} at {timed[0].when.strftime('%H:%M')}.")
        for item in items:
            if item.kind == "conflict":
                lines.append(f"Calendar overlap: {_quoted(item.text)}.")
                break
        return lines[:MAX_CONSTRAINTS]

    # -- helpers --------------------------------------------------------------------------------------------------------------------

    def _now(self) -> datetime:
        now = self._clock()
        return now if now.tzinfo is not None else now.astimezone()

    @staticmethod
    def _read(label: str, unavailable: list[str], read: Callable, default):
        """One source, read on its own: missing or failing is recorded (once), never raised."""
        try:
            return read()
        except Exception:  # noqa: BLE001 - includes "no such source" (the attribute of None)
            if label not in unavailable:
                unavailable.append(label)
            return default

    @staticmethod
    def _names(projects: list) -> tuple[list[str], int]:
        names = sorted({safe_text(p.name, TEXT_LIMIT) for p in projects} - {""}, key=lambda n: (n.casefold(), n))  # never the store's order
        return names[:MAX_OPTIONS], max(0, len(names) - MAX_OPTIONS)
