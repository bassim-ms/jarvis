"""
The Morning Brief service: gather (existing systems, read-only) -> filter/prioritise (deterministic) -> word it
(deterministic; optionally reworded by a bounded, validated model call). This is NOT a planner, executor,
scheduler or automation: it never creates a Job, calls a tool, changes a task / calendar event / reminder /
memory / World Model / notification state, approves anything or sends anything. It holds references to the
existing services only to READ them, and each read is isolated: one source failing leaves the rest of the
brief intact.

Entry points: `build()` -> `MorningBriefData`, `render(data)` -> text, `brief()` -> text, and
`command_reply(message, conversation_id)` - the same duck-typed shape the activity-context service has, called
by JarvisCore for a message that is, in its entirety, a request for the brief (see recognition.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from app.morning_brief import gather
from app.morning_brief.base import (
    ATTENTION,
    BUSINESS,
    CALENDAR,
    CORE_SOURCES,
    JOBS,
    NEWS,
    REMINDERS,
    TASKS,
    WEATHER,
    WORLD_MODEL,
    ACTIVITY,
    BriefItem,
    BusinessUpdateSource,
    MorningBriefData,
    NewsProvider,
    WeatherProvider,
)
from app.morning_brief.prioritize import assemble, dedupe, relevant_candidates, relevant_memory, tokens
from app.morning_brief.recognition import is_morning_brief_request
from app.morning_brief.render import MAX_CHARS, render_brief

log = logging.getLogger(__name__)

NEWS_LIMIT = 3
FAILURE_REPLY = "I couldn't put your brief together just now. Please try again in a moment."


@dataclass
class BriefSources:
    """The existing services a brief READS. Every one is optional: an absent source is simply not in the brief."""

    calendar: object | None = None  # CalendarProvider
    tasks: object | None = None  # TaskStore
    reminders: object | None = None  # ReminderStore
    context_engine: object | None = None  # ContextEngine (World Model + memory + unconfirmed candidates)
    attention_queue: object | None = None  # AttentionQueue
    jobs: object | None = None  # JobStore
    interactions: object | None = None  # InteractionStore
    activity: object | None = None  # ActivityContextService
    weather: WeatherProvider | None = None
    news: NewsProvider | None = None
    business: BusinessUpdateSource | None = None
    relevance_check: Callable | None = None  # NotificationEvent -> bool ("still worth telling")


class MorningBriefService:
    def __init__(
        self,
        sources: BriefSources,
        *,
        clock: Callable[[], datetime],
        synthesizer=None,
        max_chars: int = MAX_CHARS,
        news_limit: int = NEWS_LIMIT,
    ):
        self.sources = sources
        self._clock = clock
        self.synthesizer = synthesizer
        self.max_chars = max_chars
        self.news_limit = news_limit

    # -- gathering ---------------------------------------------------------------------------------------------------------

    def build(self) -> MorningBriefData:
        now = self._clock()
        src = self.sources
        items: list[BriefItem] = []
        unavailable: list[str] = []

        def read(source: str, fn, *args) -> list[BriefItem]:
            """One source, isolated: a failure is recorded (for core sources) and never propagates."""
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001 - one source failing must not fail the brief
                log.info("morning brief: source %s unavailable (%s)", source, type(exc).__name__)
                if source in CORE_SOURCES:
                    unavailable.append(source)
                return []

        if src.calendar is not None:
            items += read(CALENDAR, gather.gather_calendar, src.calendar, now)
        if src.tasks is not None:
            items += read(TASKS, gather.gather_tasks, src.tasks, now)
        if src.reminders is not None:
            items += read(REMINDERS, gather.gather_reminders, src.reminders, now)

        job_items: list[BriefItem] = []
        if src.jobs is not None:
            job_items += read(JOBS, gather.gather_failed_jobs, src.jobs, now)
        if src.interactions is not None:
            job_items += read(JOBS, gather.gather_pending_interactions, src.interactions, now)
        items += job_items
        skip_job_ids = frozenset(i.ref.split(":", 1)[1] for i in job_items if i.ref.startswith("job:"))

        other_held = 0
        if src.attention_queue is not None:
            try:
                held, other_held = gather.gather_attention(
                    src.attention_queue, now, relevance_check=src.relevance_check, reminders=src.reminders, skip_job_ids=skip_job_ids
                )
                items += held
            except Exception as exc:  # noqa: BLE001
                log.info("morning brief: attention state unavailable (%s)", type(exc).__name__)

        memory_lines: list[str] = []
        candidates: list = []
        if src.context_engine is not None:
            try:
                world_items, memory_lines, candidates = gather.gather_context(src.context_engine, now)
                items += world_items
            except Exception as exc:  # noqa: BLE001
                log.info("morning brief: context unavailable (%s)", type(exc).__name__)
        if src.activity is not None:
            items += read(ACTIVITY, gather.gather_activity, src.activity, now)

        if src.business is not None:
            items += read(BUSINESS, gather.gather_business, src.business)
        if src.weather is not None:
            items += read(WEATHER, gather.gather_weather, src.weather)
        if src.news is not None:
            items += read(NEWS, gather.gather_news, src.news, self.news_limit)

        # Memory and unconfirmed candidates: only what touches something already in today's brief.
        items = dedupe(items)
        corpus: set[str] = set()
        for item in items:
            if item.source not in (NEWS, WEATHER, BUSINESS, ACTIVITY):
                corpus |= tokens(item.text)
        items += relevant_memory(memory_lines, corpus)
        items += relevant_candidates(candidates, corpus)

        return assemble(
            items,
            date=now.date().isoformat(),
            generated_at=now,
            other_held_count=other_held,
            unavailable=unavailable,
        )

    # -- wording -----------------------------------------------------------------------------------------------------------

    def render(self, data: MorningBriefData) -> str:
        """The brief as text: deterministic, unless a synthesizer is configured and its output validates."""
        text = render_brief(data, data.generated_at, self.max_chars)
        if self.synthesizer is not None and not data.is_empty():
            reworded = self.synthesizer.synthesize(data, text)
            if reworded:
                data.generated_summary = reworded
                return reworded
        return text

    def brief(self) -> str:
        return self.render(self.build())

    def command_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        """The brief, if `message` is (entirely) a request for it; else None. Never raises."""
        if not is_morning_brief_request(message):
            return None
        try:
            return self.brief()
        except Exception:  # noqa: BLE001
            log.exception("morning brief failed")
            return FAILURE_REPLY
