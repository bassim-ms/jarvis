"""
Gathering: one small READ-ONLY function per source, each turning what an existing system already holds into
`BriefItem`s with a deterministic rank. Nothing here writes, sends, approves, executes or notifies; nothing
creates a second calendar / task / reminder / memory system - the existing stores and services are read as they
are. A function that cannot read its source raises; the service turns that into "this source is missing", never
a failed brief.

Ranking uses only STRUCTURED signals the source already carries: a due date against the clock, an explicit
priority, a Job's FAILED status, a pending interaction, a queued item's stored urgent flag / importance. The
wording of a title, task or headline is never read to decide how much it matters.

All text that reaches a `BriefItem` is untrusted display text: `safe_text` cleans it to one bounded line and
hides anything that looks like a credential. It is data - shown, never obeyed.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from typing import Callable

from app.memory_policy.policy import contains_secret_value
from app.morning_brief.base import (
    ACTIVITY,
    ATTENTION,
    BUSINESS,
    CALENDAR,
    JOBS,
    NEWS,
    RANK_BLOCKER_HIGH,
    RANK_CONFLICT,
    RANK_CONTEXT,
    RANK_DUE_TODAY,
    RANK_DUE_TODAY_HIGH,
    RANK_EVENT,
    RANK_FAILED_JOB,
    RANK_GOAL,
    RANK_HELD_HIGH,
    RANK_HIGH_PRIORITY,
    RANK_OPTIONAL,
    RANK_OVERDUE_COMMITMENT,
    RANK_OVERDUE_REMINDER,
    RANK_OVERDUE_TASK,
    RANK_PENDING_CONFIRMATION,
    RANK_PENDING_INPUT,
    RANK_REMINDER_TODAY,
    RANK_SOON,
    RANK_UPCOMING_DEADLINE,
    RANK_URGENT,
    REMINDERS,
    TASKS,
    WEATHER,
    WORLD_MODEL,
    BriefItem,
)
from app.notifications.base import NotificationKind
from app.notifications.text import clean_text

TEXT_LIMIT = 100
DETAIL_LIMIT = 90

#: Events tomorrow that start before this local hour are worth a line today (they shape tonight).
EARLY_TOMORROW_HOUR = 9
#: A HIGH-priority deadline this many days out "materially affects today".
UPCOMING_DAYS = 2
#: An event starting within this long is "coming up".
SOON = timedelta(hours=2)
#: A failed Job older than this is history, not "needs attention today".
FAILED_JOB_WINDOW = timedelta(hours=24)
HIDDEN = "(details hidden - it looks like a credential)"

_ADDRESS_PREFIX = re.compile(r"^(?:Sir|Boss),\s+", re.IGNORECASE)


# --- text and time helpers -------------------------------------------------------------------------------------------------


def safe_text(text: object, limit: int = TEXT_LIMIT) -> str:
    """One bounded line of untrusted text. Hidden only when there is reasonable evidence it holds an actual
    credential VALUE (a key/token shape, a card number, "password is hunter2"); the mere word "password",
    "token" or "key" hides nothing."""
    cleaned = clean_text(text if isinstance(text, str) else "", limit)
    if cleaned and contains_secret_value(text if isinstance(text, str) else ""):
        return HIDDEN
    return cleaned


def strip_address(text: str) -> str:
    """Drop the "Sir, " / "Boss, " a notification is addressed with; the brief has its own greeting."""
    return _ADDRESS_PREFIX.sub("", text or "", count=1)


def parse_when(value: object, tz) -> tuple[datetime, bool] | None:
    """(aware moment in `tz`, is_date_only) for an ISO date / timestamp string, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return datetime.combine(datetime.fromisoformat(raw).date(), time.min, tzinfo=tz), True
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    return moment.astimezone(tz), False


def day_bounds(now: datetime) -> tuple[datetime, datetime]:
    """[start, end) of `now`'s own local day."""
    start = datetime.combine(now.date(), time.min, tzinfo=now.tzinfo)
    return start, start + timedelta(days=1)


def _due_class(due: tuple[datetime, bool] | None, now: datetime) -> str:
    """"overdue" | "today" | "soon" (within UPCOMING_DAYS) | "later" | "none" - by the clock alone."""
    if due is None:
        return "none"
    moment, date_only = due
    start, end = day_bounds(now)
    if date_only:
        if moment < start:
            return "overdue"
        if moment < end:
            return "today"
    else:
        if moment < now:
            return "overdue"
        if moment < end:
            return "today"
    if moment < end + timedelta(days=UPCOMING_DAYS):
        return "soon"
    return "later"


# --- calendar ------------------------------------------------------------------------------------------------------------------


def event_ref(title: str, start: datetime) -> str:
    return f"event:{title.strip().casefold()}|{start.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M')}"


def gather_calendar(provider, now: datetime) -> list[BriefItem]:
    """Today's remaining / in-progress events, early-tomorrow events, and deterministic overlaps."""
    _, day_end = day_bounds(now)
    horizon = day_end + timedelta(hours=EARLY_TOMORROW_HOUR)
    events = provider.list_events(time_min=now.isoformat(), time_max=horizon.isoformat(), max_results=25)

    timed: list[tuple[datetime, datetime, str]] = []
    items: list[BriefItem] = []
    for event in events:
        start = parse_when(getattr(event, "start", None), now.tzinfo)
        end = parse_when(getattr(event, "end", None), now.tzinfo)
        title = safe_text(getattr(event, "title", ""), TEXT_LIMIT) or "(untitled event)"
        if start is None:
            continue
        begins, all_day = start
        finishes = end[0] if end is not None else begins
        if finishes <= now and not all_day:
            continue  # already over
        if all_day and finishes <= now:
            continue
        detail = safe_text(getattr(event, "location", ""), DETAIL_LIMIT)
        ref = event_ref(title, begins)
        if begins >= day_end:
            items.append(BriefItem(CALENDAR, "event_tomorrow", title, RANK_EVENT + 5, ("early_tomorrow",), begins, ref, detail=detail))
            continue
        if all_day:
            items.append(BriefItem(CALENDAR, "event_all_day", title, RANK_EVENT + 2, ("today",), begins, ref, all_day=True, detail=detail))
            continue
        timed.append((begins, finishes, title))
        if begins <= now < finishes:
            items.append(BriefItem(CALENDAR, "event", title, RANK_SOON, ("in_progress",), begins, ref, detail=detail))
        elif begins - now <= SOON:
            items.append(BriefItem(CALENDAR, "event", title, RANK_SOON, ("starts_soon",), begins, ref, detail=detail))
        else:
            items.append(BriefItem(CALENDAR, "event", title, RANK_EVENT, ("today",), begins, ref, detail=detail))

    timed.sort()
    conflicts = 0
    for i, (a_start, a_end, a_title) in enumerate(timed):
        for b_start, b_end, b_title in timed[i + 1 :]:
            if b_start >= a_end:
                break
            if conflicts >= 2:
                break
            conflicts += 1
            text = f"{a_title} overlaps {b_title}"
            items.append(
                BriefItem(CALENDAR, "conflict", text, RANK_CONFLICT, ("schedule_conflict",), b_start, f"conflict:{a_title}|{b_title}", attention=True)
            )
    return items


# --- tasks ---------------------------------------------------------------------------------------------------------------------


def gather_tasks(store, now: datetime) -> list[BriefItem]:
    """Overdue, due today, explicitly HIGH priority, and HIGH-priority deadlines within UPCOMING_DAYS.
    Priority is the task's own field; nothing is inferred from its wording."""
    from app.tasks.base import TaskPriority, TaskStatus

    items: list[BriefItem] = []
    for task in store.list(TaskStatus.TODO):
        high = task.priority is TaskPriority.HIGH
        due = parse_when(task.due_date, now.tzinfo)
        due_class = _due_class(due, now)
        text = safe_text(task.text, TEXT_LIMIT)
        when = due[0] if due else None
        date_only = bool(due and due[1])
        ref = f"task:{task.id}"
        if due_class == "overdue":
            items.append(BriefItem(TASKS, "task_overdue", text, RANK_OVERDUE_TASK - (1 if high else 0), ("overdue",) + (("priority_high",) if high else ()), when, ref, attention=True, all_day=date_only))
        elif due_class == "today":
            items.append(BriefItem(TASKS, "task_due_today", text, RANK_DUE_TODAY_HIGH if high else RANK_DUE_TODAY, ("due_today",) + (("priority_high",) if high else ()), when, ref, all_day=date_only))
        elif due_class == "soon" and high:
            items.append(BriefItem(TASKS, "task_upcoming", text, RANK_UPCOMING_DEADLINE, ("deadline_soon", "priority_high"), when, ref, all_day=date_only))
        elif high:
            items.append(BriefItem(TASKS, "task_high", text, RANK_HIGH_PRIORITY, ("priority_high",), when, ref, all_day=date_only))
        # anything else (not due soon, not high priority) is not part of today's brief
    return items


# --- reminders -----------------------------------------------------------------------------------------------------------------


def reminder_ref(reminder) -> str:
    return f"reminder:{reminder.id}:{reminder.due_at}"  # the key the reminder's own notification uses


def gather_reminders(store, now: datetime) -> list[BriefItem]:
    from app.reminders.base import ReminderStatus

    _, day_end = day_bounds(now)
    items: list[BriefItem] = []
    for reminder in store.list(ReminderStatus.PENDING):
        due = parse_when(reminder.due_at, now.tzinfo)
        if due is None:
            continue
        moment = due[0]
        text = safe_text(reminder.text, TEXT_LIMIT)
        if moment < now:
            items.append(BriefItem(REMINDERS, "reminder_overdue", text, RANK_OVERDUE_REMINDER, ("overdue",), moment, reminder_ref(reminder), attention=True))
        elif moment < day_end:
            items.append(BriefItem(REMINDERS, "reminder_today", text, RANK_REMINDER_TODAY, ("due_today",), moment, reminder_ref(reminder)))
    return items


# --- jobs and interactions -----------------------------------------------------------------------------------------------------


def gather_failed_jobs(jobs, now: datetime) -> list[BriefItem]:
    """FAILED Jobs from the last day. The system already knows they failed; no model is asked."""
    from app.orchestration.base import JobStatus

    items: list[BriefItem] = []
    for job in jobs.list(JobStatus.FAILED):
        updated = parse_when(job.updated_at, now.tzinfo)
        if updated is None or now - updated[0] > FAILED_JOB_WINDOW:
            continue
        items.append(
            BriefItem(JOBS, "job_failed", safe_text(job.goal, TEXT_LIMIT), RANK_FAILED_JOB, ("failed",), updated[0], f"job:{job.id}", attention=True,
                      detail=safe_text(job.error, DETAIL_LIMIT))
        )
    return items


def gather_pending_interactions(interactions, now: datetime) -> list[BriefItem]:
    """Questions and confirmations still waiting on the user. Reading them changes nothing: this does not
    present, resolve, decline or expire any of them."""
    from app.orchestration.interactions import InteractionType, is_expired

    utc_now = now.astimezone(timezone.utc)
    items: list[BriefItem] = []
    for request in interactions.list_pending():
        try:
            if is_expired(request, utc_now):
                continue
        except ValueError:
            continue
        confirm = request.type is InteractionType.CONFIRMATION
        items.append(
            BriefItem(JOBS, "pending_confirmation" if confirm else "pending_input", safe_text(request.prompt, 140),
                      RANK_PENDING_CONFIRMATION if confirm else RANK_PENDING_INPUT, ("pending_confirmation" if confirm else "pending_input",),
                      None, f"job:{request.job_id}", attention=True)
        )
    return items


# --- attention queue -----------------------------------------------------------------------------------------------------------


def gather_attention(
    queue, now: datetime, *, relevance_check: Callable | None, reminders=None, skip_job_ids: frozenset[str] = frozenset()
) -> tuple[list[BriefItem], int]:
    """What the Attention system is holding, without touching it (no release, no presenting, no status change).

    A held item is "urgent" only if the queue itself says so (status `escalation` / the stored urgent
    flag); a merely queued item is never promoted. HIGH-importance items are listed as held-for-later;
    everything else is only counted. Returns (items, how many lower-importance items were only counted)."""
    from app.notifications.attention import Importance

    items: list[BriefItem] = []
    other = 0
    for held in queue.waiting():
        event = held.event
        if event.job_id and event.job_id in skip_job_ids:
            continue  # the Job / interaction itself is already in the brief
        if relevance_check is not None:
            try:
                if not relevance_check(event):
                    continue  # stale: e.g. the question it announces was answered meanwhile
            except Exception:  # noqa: BLE001 - unsure: keep it
                pass
        if event.kind is NotificationKind.REMINDER:
            text = _held_reminder_text(event, reminders)
            items.append(BriefItem(ATTENTION, "reminder_held", text, RANK_REMINDER_TODAY, ("held",), None, event.key))
            continue
        text = safe_text(strip_address(event.text), 140)
        if held.status == "escalation" or held.signal.urgent:
            items.append(BriefItem(ATTENTION, "urgent_unreached", text, RANK_URGENT, ("urgent", "not_delivered"), None, event.key, attention=True))
        elif held.signal.importance >= Importance.HIGH:
            items.append(BriefItem(ATTENTION, "held_high", text, RANK_HELD_HIGH, ("held", "importance_high"), None, event.key, attention=True))
        else:
            other += 1
    return items, other


def _held_reminder_text(event, reminders) -> str:
    parts = event.key.split(":")
    if reminders is not None and len(parts) >= 3:
        try:
            reminder = reminders.get(parts[1])
        except Exception:  # noqa: BLE001
            reminder = None
        if reminder is not None:
            return safe_text(reminder.text, TEXT_LIMIT)
    text = strip_address(event.text)
    return safe_text(re.sub(r"^a reminder:\s*", "", text, flags=re.IGNORECASE), TEXT_LIMIT)


# --- World Model / Context Engine, memory --------------------------------------------------------------------------------------


def request_context_for_today(now: datetime):
    """A RequestContext for "today": PLANNING intent, today's window, and only the categories a brief uses.
    Built directly (not by classifying text) so no message wording can widen what is retrieved."""
    from app.context_engine.base import (
        ContextIntent,
        RequestContext,
        RequiredContextCategory as Cat,
        TimeScope,
        TimeScopeKind,
    )

    start, end = day_bounds(now)
    scope = TimeScope(TimeScopeKind.TODAY, start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat())
    return RequestContext(
        raw_request="morning brief",
        context_intent=ContextIntent.PLANNING,
        time_scope=scope,
        required_context=frozenset({Cat.PROJECTS, Cat.GOALS, Cat.COMMITMENTS, Cat.EVENTS, Cat.BLOCKERS, Cat.MEMORY}),
    )


def gather_context(engine, now: datetime):
    """(world-model items, raw memory lines, unconfirmed candidates) from the Context Engine snapshot. Memory
    lines and candidates are returned RAW: which of them matter today is decided against the rest of the
    brief (see prioritize.py), not here."""
    from app.world_model.base import BlockerSeverity, CommitmentStatus, GoalStatus, Priority, ProjectStatus

    snapshot = engine.build_snapshot(request_context_for_today(now), now=now.astimezone(timezone.utc).isoformat())
    _, day_end = day_bounds(now)
    items: list[BriefItem] = []

    for commitment in snapshot.upcoming_commitments:
        if commitment.status is not CommitmentStatus.OPEN:
            continue
        high = commitment.priority is Priority.HIGH
        due_class = _due_class(parse_when(commitment.due_at, now.tzinfo), now)
        when = parse_when(commitment.due_at, now.tzinfo)
        text = safe_text(commitment.title, TEXT_LIMIT)
        ref = f"commitment:{commitment.id}"
        if due_class == "overdue":
            items.append(BriefItem(WORLD_MODEL, "commitment_overdue", text, RANK_OVERDUE_COMMITMENT, ("overdue",), when[0] if when else None, ref, attention=True))
        elif due_class == "today":
            items.append(BriefItem(WORLD_MODEL, "commitment_due", text, RANK_DUE_TODAY_HIGH if high else RANK_DUE_TODAY, ("due_today",), when[0] if when else None, ref))
        elif due_class == "soon" and high:
            items.append(BriefItem(WORLD_MODEL, "commitment_due", text, RANK_UPCOMING_DEADLINE, ("deadline_soon", "priority_high"), when[0] if when else None, ref))

    for blocker in snapshot.active_blockers:
        if blocker.severity in (BlockerSeverity.HIGH, BlockerSeverity.CRITICAL):
            items.append(BriefItem(WORLD_MODEL, "blocker", safe_text(blocker.title, TEXT_LIMIT), RANK_BLOCKER_HIGH, ("blocker", blocker.severity.value), None, f"blocker:{blocker.id}", attention=True))

    for goal in snapshot.relevant_goals[:6]:
        if goal.status is not GoalStatus.ACTIVE:
            continue
        target = parse_when(goal.target_date, now.tzinfo)
        due_class = _due_class(target, now)
        if due_class in ("overdue", "today") or goal.priority is Priority.HIGH:
            items.append(BriefItem(WORLD_MODEL, "goal", safe_text(goal.title, TEXT_LIMIT), RANK_GOAL, ("active_goal",) + ((due_class,) if due_class in ("overdue", "today") else ()), target[0] if target else None, f"goal:{goal.id}"))

    for project in snapshot.active_projects[:6]:
        if project.status is ProjectStatus.ACTIVE and project.priority is Priority.HIGH:
            items.append(BriefItem(WORLD_MODEL, "project", safe_text(project.name, TEXT_LIMIT), RANK_GOAL + 2, ("active_project", "priority_high"), None, f"project:{project.id}"))

    for event in snapshot.upcoming_events:
        start = parse_when(event.starts_at, now.tzinfo)
        end = parse_when(event.ends_at, now.tzinfo)
        if start is None or start[0] >= day_end or (end is not None and end[0] <= now):
            continue
        title = safe_text(event.title, TEXT_LIMIT)
        rank = RANK_SOON if start[0] - now <= SOON else RANK_EVENT
        items.append(BriefItem(WORLD_MODEL, "event", title, rank, ("today",), start[0], event_ref(title, start[0]), detail=safe_text(event.location, DETAIL_LIMIT)))

    return items, list(snapshot.relevant_memory), list(snapshot.unconfirmed_candidates)


# --- temporary activity context ------------------------------------------------------------------------------------------------

_ACTIVITY_LABELS = {"class": "in class", "studying": "studying", "meeting": "in a meeting", "focused_work": "in focused work"}


def gather_activity(activity, now: datetime) -> list[BriefItem]:
    """What the user themselves said they are doing right now (temporary, from their own statement)."""
    context = activity.current()
    if context is None:
        return []
    end = parse_when(context.expires_at, now.tzinfo)
    label = _ACTIVITY_LABELS.get(context.value, safe_text(context.value, 30))
    return [BriefItem(ACTIVITY, "activity", label, RANK_CONTEXT, ("user_statement",), end[0] if end else None, "activity:current")]


# --- optional sources ----------------------------------------------------------------------------------------------------------


def gather_weather(provider) -> list[BriefItem]:
    report = provider.today()
    if report is None or not safe_text(report.summary, 160):
        return []
    return [BriefItem(WEATHER, "weather", safe_text(report.summary, 160), RANK_OPTIONAL, ("provider",), None, "weather:today", detail=safe_text(report.source, 30))]


def gather_news(provider, limit: int) -> list[BriefItem]:
    """A very small, bounded set of headlines - each kept with where it came from."""
    items: list[BriefItem] = []
    for headline in provider.headlines(limit)[:limit]:
        title = safe_text(headline.title, 120)
        if title:
            items.append(BriefItem(NEWS, "news", title, RANK_OPTIONAL, ("headline",), None, f"news:{title.casefold()}", detail=safe_text(headline.source, 40)))
    return items


def gather_business(source, limit: int = 3) -> list[BriefItem]:
    items: list[BriefItem] = []
    for update in source.updates()[: limit * 2]:
        text = safe_text(update.text, 120)
        if not text:
            continue
        important = update.important is True  # a structured flag, never inferred from text
        items.append(BriefItem(BUSINESS, "business", text, RANK_HELD_HIGH if important else RANK_OPTIONAL, ("important",) if important else ("update",), None, f"business:{text.casefold()}", attention=important, detail=safe_text(update.source, 30)))
    items.sort(key=lambda i: i.rank)
    return items[:limit]
