"""
Wording the brief. The DETERMINISTIC rendering here is complete on its own - the brief never needs a model.
It is short by construction: empty sections are left out, each section is capped, and if the whole still runs
over `MAX_CHARS` the caps tighten step by step (and optional sections go first) until it fits.

It is plain text a person reads in a chat reply or hears aloud - no markup - and only ever repeats what the
items say; it adds no fact of its own (counts are counts of items, times are the items' own times).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.morning_brief.base import BriefItem, MorningBriefData
from app.notifications.text import address_for

MAX_CHARS = 1500


@dataclass(frozen=True)
class Caps:
    schedule: int = 6
    tasks: int = 5
    reminders: int = 4
    know: int = 6
    business: int = 3
    news: int = 3
    weather: bool = True


_CAP_PROFILES = (
    Caps(),
    Caps(schedule=4, tasks=3, reminders=3, know=4, business=2, news=2),
    Caps(schedule=3, tasks=3, reminders=2, know=3, business=1, news=0, weather=False),
    Caps(schedule=2, tasks=2, reminders=1, know=2, business=0, news=0, weather=False),
)

_LABELS = {
    "job_failed": "Failed",
    "pending_confirmation": "Waiting on your confirmation",
    "pending_input": "Waiting on your answer",
    "urgent_unreached": "Urgent, and I couldn't reach you",
    "held_high": "Held for later",
    "task_overdue": "Overdue",
    "task_due_today": "Due today",
    "task_high": "High priority",
    "task_upcoming": "Deadline coming up",
    "commitment_overdue": "Overdue commitment",
    "commitment_due": "Commitment due",
    "reminder_overdue": "Overdue reminder",
    "reminder_held": "Held reminder",
    "conflict": "Conflict",
    "blocker": "Blocker",
    "goal": "Goal",
    "project": "Active project",
    "memory": "Worth remembering",
    "candidate": "Unconfirmed",
    "business": "Business",
}


def _greeting(now: datetime) -> str:
    hour = now.hour
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    return f"Good {part}, {address_for(now.date().isoformat())}."


def _clock(moment: datetime) -> str:
    return moment.strftime("%H:%M")


def _day(moment: datetime) -> str:
    return moment.strftime("%a %d %b")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _line(item: BriefItem, now: datetime) -> str:
    kind = item.kind
    text = item.text
    detail = f" ({item.detail})" if item.detail else ""
    when = item.when
    if kind == "event":
        return f"{_clock(when)} {text}{detail}" if when else f"{text}{detail}"
    if kind == "event_all_day":
        return f"All day: {text}{detail}"
    if kind == "event_tomorrow":
        return f"Tomorrow {_clock(when)} {text}{detail}"
    if kind == "task_overdue":
        return f"Overdue: {text}" + (f" (was due {_day(when)})" if when else "")
    if kind == "task_due_today":
        return f"Due today: {text}" + ("" if item.all_day or not when else f" (by {_clock(when)})")
    if kind == "task_upcoming":
        return f"Due {_day(when)}: {text} (high priority)" if when else f"Deadline coming up: {text}"
    if kind == "task_high":
        return f"High priority: {text}"
    if kind == "reminder_today":
        return f"{_clock(when)} {text}" if when else text
    if kind == "reminder_overdue":
        return f"Overdue reminder: {text}" + (f" (was due {_clock(when)})" if when else "")
    if kind == "activity":
        return f"You told me you're {text}" + (f" until {_clock(when)}" if when else "")
    if kind == "weather":
        return text
    if kind == "news":
        return f"{text}" + (f" ({item.detail})" if item.detail else "")
    if kind == "job_failed":
        return f"Failed: {text}" + (f" - {item.detail}" if item.detail else "")
    if kind == "candidate":
        return f"Unconfirmed: {text} (not confirmed - I may have this wrong)"
    if kind == "business":
        return f"{text}" + (f" ({item.detail})" if item.detail else "")
    label = _LABELS.get(kind)
    return f"{label}: {text}" if label else text


def _bullets(items: list[BriefItem], cap: int, now: datetime) -> list[str]:
    shown = [f"- {_line(item, now)}" for item in items[:cap]]
    if len(items) > cap:
        shown.append(f"- ...and {len(items) - cap} more")
    return shown


def _overview(data: MorningBriefData) -> str:
    items = data.all_items()
    count = lambda *kinds: sum(1 for i in items if i.kind in kinds)  # noqa: E731
    parts = []
    events = count("event", "event_all_day")
    if events:
        parts.append(_plural(events, "event"))
    overdue = count("task_overdue")
    if overdue:
        parts.append(f"{overdue} overdue task{'' if overdue == 1 else 's'}")
    due = count("task_due_today")
    if due:
        parts.append(f"{_plural(due, 'task')} due today")
    reminders = count("reminder_today", "reminder_overdue", "reminder_held")
    if reminders:
        parts.append(_plural(reminders, "reminder"))
    needing = sum(1 for i in items if i.attention and i.kind not in ("task_overdue", "reminder_overdue"))
    if needing:
        parts.append(f"{needing} thing{'' if needing == 1 else 's'} needing your attention")
    return ", ".join(parts)


def _render_once(data: MorningBriefData, now: datetime, caps: Caps) -> str:
    lines = [_greeting(now)]
    if data.is_empty():
        lines.append("Nothing is scheduled and nothing needs your attention right now.")
    else:
        overview = _overview(data)
        if overview:
            lines += ["", f"Today: {overview}."]
        if data.priorities:
            lines += ["", "Main priorities:"] + [f"- {_line(i, now)}" for i in data.priorities]
        if data.calendar_items:
            lines += ["", "Schedule:"] + _bullets(data.calendar_items, caps.schedule, now)
        if data.tasks or data.reminders:
            lines += ["", "Tasks and reminders:"]
            lines += _bullets(data.tasks, caps.tasks, now) + _bullets(data.reminders, caps.reminders, now)
        know = [*data.attention_items, *data.important_notifications, *data.important_context]
        if know or data.other_held_count:
            lines += ["", "Need to know:"] + _bullets(know, caps.know, now)
            if data.other_held_count:
                lines.append(f"- {_plural(data.other_held_count, 'other notification')} held for later (nothing urgent)")
        if caps.business and data.business_updates:
            lines += ["", "Business:"] + _bullets(data.business_updates, caps.business, now)
        if caps.weather and data.weather is not None:
            lines += ["", f"Weather: {data.weather.text}"]
        if caps.news and data.news:
            lines += ["", "In the news:"] + _bullets(data.news, caps.news, now)
    if data.unavailable:
        names = " and ".join(sorted(data.unavailable)) if len(data.unavailable) <= 2 else ", ".join(sorted(data.unavailable))
        lines += ["", f"(I couldn't read your {names}, so that part may be missing.)"]
    if not data.is_empty():
        needing = any(i.attention for i in data.all_items())
        lines += ["", "That's what needs you today." if needing else "Nothing else urgent."]
    return "\n".join(lines)


def render_brief(data: MorningBriefData, now: datetime, max_chars: int = MAX_CHARS) -> str:
    """The deterministic brief, never longer than `max_chars`."""
    text = ""
    for caps in _CAP_PROFILES:
        text = _render_once(data, now, caps)
        if len(text) <= max_chars:
            return text
    return text[: max_chars - 3].rstrip() + "..."
