"""
Deterministic filtering, de-duplication and prioritisation of gathered items - the step that decides what
actually matters BEFORE any wording (and before any model). Pure functions over `BriefItem`s: no I/O, no
clock, no model.

  * the same thing arriving from two sources (a World Model event that is also a calendar event; a held
    reminder that is also a pending reminder; a queued "job failed" notice for a Job the brief already lists)
    appears once - the more important copy wins;
  * memory and unconfirmed candidates are only kept if they touch something that IS in today's brief (they are
    never listed for their own sake), and anything sensitive or credential-like is dropped;
  * items are ordered by (rank, time, text); the few with the strongest signals are lifted into "main
    priorities" and removed from their own sections so nothing is said twice;
  * unconfirmed items stay marked unconfirmed all the way through.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone

from app.memory_policy.base import Sensitivity
from app.memory_policy.policy import classify_sensitivity
from app.morning_brief.base import (
    ACTIVITY,
    ATTENTION,
    BUSINESS,
    CALENDAR,
    JOBS,
    MEMORY,
    NEWS,
    PRIORITY_MAX_RANK,
    RANK_CONTEXT,
    REMINDERS,
    TASKS,
    WEATHER,
    WORLD_MODEL,
    BriefItem,
    MorningBriefData,
)
from app.morning_brief.gather import safe_text

MAX_PRIORITIES = 3
#: Events are ordered by their timing (an imminent one ranks first) but stay in the schedule, so the schedule
#: is never left with a hole; only something that needs ACTING on (overdue, failed, waiting, conflict...) is lifted.
_STAYS_IN_SCHEDULE = frozenset({"event", "event_all_day", "event_tomorrow"})
MAX_MEMORY_LINES = 2
MAX_CANDIDATES = 2

_STOPWORDS = frozenset(
    "with that this from have will your about after before today tomorrow meeting call task work "
    "what when where which their them then there these those into over under".split()
)
_TOKEN = re.compile(r"[a-z0-9]{4,}")


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").casefold().replace("_", " ")) if t not in _STOPWORDS}


def dedupe(items: list[BriefItem]) -> list[BriefItem]:
    """One copy per identity. By `ref` first (a stable id), then by identical text within the same kind; the
    lower rank (more important) copy is kept, and an `attention` flag is never lost."""
    best: dict[str, BriefItem] = {}
    order: list[str] = []
    for item in items:
        key = item.ref or f"{item.source}:{item.kind}:{item.text.casefold()}"
        current = best.get(key)
        if current is None:
            best[key] = item
            order.append(key)
        elif item.rank < current.rank:
            best[key] = replace(item, attention=item.attention or current.attention)
        elif item.attention and not current.attention:
            best[key] = replace(current, attention=True)
    by_text: dict[tuple[str, str], BriefItem] = {}
    result: list[BriefItem] = []
    for key in order:
        item = best[key]
        text_key = (item.kind, item.text.casefold())
        if text_key in by_text:
            continue
        by_text[text_key] = item
        result.append(item)
    return result


def relevant_memory(memory_lines: list[str], corpus: set[str]) -> list[BriefItem]:
    """Stored facts that touch something in today's brief. Never sensitive or credential-like ones; never
    more than MAX_MEMORY_LINES; read only."""
    items: list[BriefItem] = []
    for line in memory_lines:
        key, _, value = line.partition(":")
        if classify_sensitivity(key, value) is not Sensitivity.NONE:
            continue
        if not tokens(f"{key} {value}") & corpus:
            continue
        text = safe_text(f"{key.strip().replace('_', ' ')}: {value.strip()}", 120)
        items.append(BriefItem(MEMORY, "memory", text, RANK_CONTEXT + 1, ("relevant_to_today",), None, f"memory:{key.strip()}"))
    return items[:MAX_MEMORY_LINES]


def relevant_candidates(candidates: list, corpus: set[str]) -> list[BriefItem]:
    """Staged, UNCONFIRMED candidates that touch today's brief - always kept marked unconfirmed."""
    items: list[BriefItem] = []
    for candidate in candidates:
        if getattr(candidate, "status", "unconfirmed") != "unconfirmed":
            continue
        if classify_sensitivity(candidate.key, candidate.value) is not Sensitivity.NONE:
            continue
        if not tokens(f"{candidate.key} {candidate.value}") & corpus:
            continue
        text = safe_text(f"{candidate.key.replace('_', ' ')}: {candidate.value}", 120)
        items.append(BriefItem(MEMORY, "candidate", text, RANK_CONTEXT + 2, ("unconfirmed",), None, f"candidate:{candidate.id}", unconfirmed=True))
    return items[:MAX_CANDIDATES]


def order_key(item: BriefItem):
    when = item.when.astimezone(timezone.utc) if item.when is not None else datetime.max.replace(tzinfo=timezone.utc)
    return (item.rank, when, item.text.casefold())


def assemble(
    items: list[BriefItem],
    *,
    date: str,
    generated_at: datetime,
    other_held_count: int,
    unavailable: list[str],
) -> MorningBriefData:
    """Sort into `MorningBriefData` sections; lift the strongest few into `priorities`."""
    ordered = sorted(dedupe(items), key=order_key)

    priorities: list[BriefItem] = []
    for item in ordered:
        if item.rank <= PRIORITY_MAX_RANK and item.kind not in _STAYS_IN_SCHEDULE and len(priorities) < MAX_PRIORITIES:
            priorities.append(item)
    lifted = {id(item) for item in priorities}
    rest = [item for item in ordered if id(item) not in lifted]

    data = MorningBriefData(date=date, generated_at=generated_at, other_held_count=other_held_count, unavailable=list(unavailable))
    data.priorities = priorities
    for item in rest:
        if item.source == CALENDAR or (item.source == WORLD_MODEL and item.kind == "event"):
            data.calendar_items.append(item)
        elif item.source == TASKS:
            data.tasks.append(item)
        elif item.source == REMINDERS or item.kind == "reminder_held":
            data.reminders.append(item)
        elif item.source == JOBS:
            data.attention_items.append(item)
        elif item.source == ATTENTION:
            data.important_notifications.append(item)
        elif item.source in (WORLD_MODEL, MEMORY, ACTIVITY):
            data.important_context.append(item)
        elif item.source == BUSINESS:
            data.business_updates.append(item)
        elif item.source == NEWS:
            data.news.append(item)
        elif item.source == WEATHER:
            data.weather = item

    data.sources_used = sorted({item.source for item in data.all_items()})
    return data
