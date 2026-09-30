"""
Morning Brief data model: what one brief is made of, before it is worded.

A brief is READ-ONLY aggregation of what the existing systems already know (calendar, tasks, reminders,
World Model / Context Engine, memory, Attention queue, Jobs and interactions, plus optional weather, news and
business sources). Nothing here decides or does anything: every `BriefItem` records which source it came from
(`source`), what kind of thing it is (`kind`), a deterministic `rank` (lower = matters more today) and the
plain `reasons` that put it there. Ranks come from structured signals the source systems already hold
(overdue, due today, explicit priority, failed, pending, urgent flag) - never from the wording of the item.

Every `text` is UNTRUSTED display text (a calendar title, a task, a headline): it is cleaned and bounded on
the way in (see gather.py) and is only ever shown - it is never read as an instruction by anything.

The optional provider interfaces (weather, news, business) exist so those sources can be plugged in without
touching the brief; none is required, and none is built here beyond a news adapter over the EXISTING search
provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

# Sources. A brief keeps them distinguishable end to end.
CALENDAR = "calendar"
TASKS = "tasks"
REMINDERS = "reminders"
WORLD_MODEL = "world_model"
MEMORY = "memory"
ATTENTION = "attention"
JOBS = "jobs"
ACTIVITY = "activity"
WEATHER = "weather"
NEWS = "news"
BUSINESS = "business"

#: The sources whose absence would make "nothing scheduled / nothing due" a misleading thing to say. When one
#: of these cannot be read the brief says so; an optional source (weather, news, business) is just left out.
CORE_SOURCES = frozenset({CALENDAR, TASKS, REMINDERS})

# Ranks (lower = more important today). The gaps are deliberate: they are the whole prioritisation policy.
RANK_URGENT = 0
RANK_FAILED_JOB = 10
RANK_PENDING_CONFIRMATION = 12
RANK_PENDING_INPUT = 14
RANK_OVERDUE_TASK = 20
RANK_OVERDUE_COMMITMENT = 22
RANK_OVERDUE_REMINDER = 24
RANK_CONFLICT = 26
RANK_DUE_TODAY_HIGH = 30
RANK_HELD_HIGH = 32
RANK_BLOCKER_HIGH = 34
RANK_SOON = 36  # a calendar event starting within the next couple of hours
RANK_DUE_TODAY = 40
RANK_HIGH_PRIORITY = 44
RANK_UPCOMING_DEADLINE = 46
RANK_REMINDER_TODAY = 50
RANK_EVENT = 60
RANK_GOAL = 70
RANK_CONTEXT = 80
RANK_OPTIONAL = 90

#: Items at or below this rank may be lifted into the "main priorities" block.
PRIORITY_MAX_RANK = RANK_SOON


@dataclass(frozen=True)
class BriefItem:
    source: str
    kind: str
    text: str
    rank: int
    reasons: tuple[str, ...] = ()
    #: When the item happens / is due (aware), if it has a time. Display and ordering only.
    when: datetime | None = None
    #: Stable identity, used to remove the same thing arriving from two sources.
    ref: str = ""
    #: True for a staged candidate or anything else that is NOT a confirmed fact: it is shown as such.
    unconfirmed: bool = False
    #: True when the item is "needs attention" (overdue, failed, waiting on the user, urgent...), as opposed
    #: to plain information about the day.
    attention: bool = False
    #: Whole-day event, or a date-only due date.
    all_day: bool = False
    #: Secondary text shown after the item (e.g. a location), already cleaned.
    detail: str = ""


@dataclass(frozen=True)
class WeatherReport:
    summary: str
    source: str = "weather"


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str = ""  # provenance: where the headline came from (a site name / domain)
    url: str = ""


@dataclass(frozen=True)
class BusinessUpdate:
    text: str
    #: A STRUCTURED flag set by the integration - never inferred from the text.
    important: bool = False
    source: str = "business"


class WeatherProvider(ABC):
    @abstractmethod
    def today(self) -> WeatherReport | None:
        """Today's weather, or None if there is none. May raise: the brief then omits weather."""


class NewsProvider(ABC):
    @abstractmethod
    def headlines(self, limit: int) -> list[NewsItem]:
        """At most `limit` current items, each with its source. May raise: the brief then omits news."""


class BusinessUpdateSource(ABC):
    @abstractmethod
    def updates(self) -> list[BusinessUpdate]:
        """Meaningful current business items (pending work, deadlines, blocked work...). May raise: omitted."""


@dataclass
class MorningBriefData:
    """Everything gathered for one brief, by source, before it is worded. Sections hold what survived
    filtering and prioritisation, in presentation order."""

    date: str
    generated_at: datetime
    calendar_items: list[BriefItem] = field(default_factory=list)
    tasks: list[BriefItem] = field(default_factory=list)
    reminders: list[BriefItem] = field(default_factory=list)
    important_context: list[BriefItem] = field(default_factory=list)  # World Model, memory, activity context
    important_notifications: list[BriefItem] = field(default_factory=list)
    business_updates: list[BriefItem] = field(default_factory=list)
    weather: BriefItem | None = None
    news: list[BriefItem] = field(default_factory=list)
    attention_items: list[BriefItem] = field(default_factory=list)  # failed jobs, pending interactions, urgent
    priorities: list[BriefItem] = field(default_factory=list)  # the few that matter most, lifted out of the rest
    #: Held notifications that are NOT important (normal/low): only counted, never listed.
    other_held_count: int = 0
    #: Core sources that could not be read (so the brief does not imply "nothing there").
    unavailable: list[str] = field(default_factory=list)
    generated_summary: str = ""
    #: Every source that contributed at least one item - the provenance of the whole brief.
    sources_used: list[str] = field(default_factory=list)

    def all_items(self) -> list[BriefItem]:
        items = [
            *self.calendar_items, *self.tasks, *self.reminders, *self.important_context,
            *self.important_notifications, *self.business_updates, *self.news, *self.attention_items,
            *self.priorities,
        ]
        if self.weather is not None:
            items.append(self.weather)
        return items

    def is_empty(self) -> bool:
        return not self.all_items() and not self.other_held_count
