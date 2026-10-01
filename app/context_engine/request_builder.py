"""
Deterministic RequestContext construction (M16.2).

`build_request_context` is the ONLY place a RequestContext gets built -
no AI call, no randomness, matching the exact "small, fixed, explicit
patterns" discipline every other deterministic classifier in this
codebase already uses (SearchRouter, DeterministicCapabilityRouter,
classify_request in app/providers/requirements.py). It deliberately does
NOT create a second general-purpose request classifier: `capability`,
when given, is the SAME CapabilityDecision.capability
JarvisCore._dispatch already computed via CapabilityRouter
(app/capability.py) - this module only adds a second, orthogonal axis
(ContextIntent - "what does the World Model need to explain this?") on
top of it, exactly the way classify_request already adds TaskType
("what does the provider need to execute this?") on top of the same
capability.

Entity-mention extraction and temporal-phrase parsing are both narrow,
fixed-pattern heuristics - no fuzzy matching, no embeddings, no NLP
library. Nothing in this project has a name/entity-matching utility to
reuse (checked app/search/router.py, app/tools/decision.py,
app/search/query_builder.py - none exist), so this is new, but kept to
the same "explicit, bounded, documented trade-offs" shape those modules
already use, and it only ever produces TEXT HINTS - resolving a hint
against real World Model entities is the Context Engine's job (see
app/context_engine/engine.py), never this module's.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from app.capability import Capability
from app.context_engine.base import (
    ContextIntent,
    RequestContext,
    TimeScope,
    TimeScopeKind,
    default_required_context,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Time scope
# ---------------------------------------------------------------------------

# Checked in this exact order (most explicit/specific first) - the first
# pattern to match wins, exactly like DeterministicCapabilityRouter's own
# ordered checks (app/capability.py).
_DATE_RANGE_PATTERN = re.compile(
    r"\b(?:from|between)\s+(\d{4}-\d{2}-\d{2})\s+(?:to|and)\s+(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_OVERDUE_PATTERN = re.compile(r"\boverdue\b|\bpast[\s-]due\b", re.IGNORECASE)
_TODAY_PATTERN = re.compile(r"\btoday\b", re.IGNORECASE)
_TOMORROW_PATTERN = re.compile(r"\btomorrow\b", re.IGNORECASE)
_THIS_WEEK_PATTERN = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_UPCOMING_PATTERN = re.compile(
    r"\bupcoming\b|\bcoming\s+up\b|\bwhat'?s?\s+next\b", re.IGNORECASE
)
_NOW_PATTERN = re.compile(
    r"\bright\s+now\b|\bcurrently\b|\bat\s+the\s+moment\b", re.IGNORECASE
)


def _day_bounds(day: datetime) -> tuple[datetime, datetime]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _week_bounds(reference: datetime) -> tuple[datetime, datetime]:
    """ISO calendar week (Monday 00:00 UTC through the following Monday
    00:00 UTC) containing `reference` - a fixed, deterministic
    definition of "this week" rather than a rolling N-day window."""
    start_of_today, _ = _day_bounds(reference)
    monday = start_of_today - timedelta(days=reference.weekday())
    return monday, monday + timedelta(days=7)


def extract_time_scope(message: str, *, now: datetime | None = None) -> TimeScope:
    """Deterministically parse a small, fixed set of temporal phrases
    into a TimeScope. Returns TimeScopeKind.NONE (no bounds) when
    nothing recognizable is found - never a guess. `now` is injectable
    for tests; defaults to the real current UTC time (this codebase's
    existing timezone convention - every timestamp elsewhere is
    `datetime.now(timezone.utc).isoformat()`, see e.g.
    app/orchestration/sqlite_store.py, app/world_model/service.py)."""
    reference = now or _now()

    date_range_match = _DATE_RANGE_PATTERN.search(message)
    if date_range_match:
        return TimeScope(
            kind=TimeScopeKind.DATE_RANGE,
            start=date_range_match.group(1) + "T00:00:00+00:00",
            end=date_range_match.group(2) + "T00:00:00+00:00",
        )

    if _OVERDUE_PATTERN.search(message):
        return TimeScope(kind=TimeScopeKind.OVERDUE, start=None, end=_iso(reference))

    if _TODAY_PATTERN.search(message):
        start, end = _day_bounds(reference)
        return TimeScope(kind=TimeScopeKind.TODAY, start=_iso(start), end=_iso(end))

    if _TOMORROW_PATTERN.search(message):
        start, end = _day_bounds(reference + timedelta(days=1))
        return TimeScope(kind=TimeScopeKind.TOMORROW, start=_iso(start), end=_iso(end))

    if _THIS_WEEK_PATTERN.search(message):
        start, end = _week_bounds(reference)
        return TimeScope(kind=TimeScopeKind.THIS_WEEK, start=_iso(start), end=_iso(end))

    if _UPCOMING_PATTERN.search(message):
        return TimeScope(kind=TimeScopeKind.UPCOMING, start=_iso(reference), end=None)

    if _NOW_PATTERN.search(message):
        return TimeScope(kind=TimeScopeKind.NOW)

    return TimeScope(kind=TimeScopeKind.NONE)


# ---------------------------------------------------------------------------
# Context intent
# ---------------------------------------------------------------------------

# Checked in this exact order - the same "most explicit/distinctive
# signal first" discipline as time-scope parsing above. Each pattern set
# is small and fixed, matching this project's existing keyword-list
# philosophy (see e.g. _CODING_KEYWORDS/_REASONING_KEYWORDS in
# app/providers/requirements.py).
_SCHEDULING_PATTERN = re.compile(
    r"\bcalendar\b|\bschedule[ds]?\b|\bwhat'?s?\s+on\s+my\b|\bwhen\s+is\b|\bwhen'?s\b",
    re.IGNORECASE,
)
_STATUS_PATTERN = re.compile(
    r"\bstatus\b|\bprogress\b|\bblocking\b|"
    r"\bstuck\b|\bhow'?s?\s+.*\s+going\b",
    re.IGNORECASE,
)
_PLANNING_PATTERN = re.compile(
    r"\bplan(?:ning)?\b|\bwhat\s+should\s+i\s+(?:do|work\s+on)\b|"
    r"\bprioriti[sz]e\b|\bnext\s+steps?\b",
    re.IGNORECASE,
)
_DECISION_SUPPORT_PATTERN = re.compile(
    r"\bshould\s+i\b|\bwhich\s+(?:one|project|task|goal)\b|"
    r"\bdecide\s+between\b|\bor\s+should\s+i\b",
    re.IGNORECASE,
)
_PROJECT_FOCUS_PATTERN = re.compile(r"\bproject\b", re.IGNORECASE)
_TASK_EXECUTION_PATTERN = re.compile(
    r"\bfinish\b|\bcomplete\b|\bwork\s+on\b|\bcontinue\b|\bresume\b", re.IGNORECASE
)


def classify_context_intent(
    message: str, *, capability: Capability | None = None
) -> ContextIntent:
    """Deterministically derive a ContextIntent from `message` and
    whatever Capability the existing M13/M14 router already decided, if
    any - reusing that decision rather than re-classifying from scratch.

    `capability` is checked first when it's a strong, unambiguous
    signal: Capability.TASK_STATUS (app/capability.py) IS a status
    question by construction, so it maps straight to
    ContextIntent.STATUS with no further keyword parsing. Every other
    capability (including TASK, which only means "this is a goal to
    plan/execute," not any particular ContextIntent) falls through to
    the keyword classifier below, so a TASK message that also happens to
    mention "project"/"schedule"/etc. still gets the more specific
    intent its own wording implies.

    Falls back to ContextIntent.GENERAL when nothing matches - never a
    speculative guess at intent (M16.2's own explicit requirement)."""
    if capability is Capability.TASK_STATUS:
        return ContextIntent.STATUS

    if _SCHEDULING_PATTERN.search(message):
        return ContextIntent.SCHEDULING
    if _STATUS_PATTERN.search(message):
        return ContextIntent.STATUS
    if _PLANNING_PATTERN.search(message):
        return ContextIntent.PLANNING
    if _DECISION_SUPPORT_PATTERN.search(message):
        return ContextIntent.DECISION_SUPPORT
    if _PROJECT_FOCUS_PATTERN.search(message):
        return ContextIntent.PROJECT_FOCUS
    if _TASK_EXECUTION_PATTERN.search(message):
        return ContextIntent.TASK_EXECUTION

    if capability is Capability.TASK:
        # A goal-driven request with no more specific wording match is
        # still inherently about deciding/sequencing work, not idle
        # conversation - PLANNING, not GENERAL.
        return ContextIntent.PLANNING

    return ContextIntent.GENERAL


# ---------------------------------------------------------------------------
# Entity mention extraction (text hints only - never ids)
# ---------------------------------------------------------------------------

#: A quoted phrase is treated as an explicit mention for EVERY category -
#: the message doesn't say which kind of entity it names, so every
#: category the Context Engine actually retrieves gets a chance to
#: resolve it (see app/context_engine/engine.py's entity resolution).
_QUOTED_PHRASE_PATTERN = re.compile(r"[\"']([^\"']{2,60})[\"']")

#: A phrase word, but never "my"/"the" themselves - blocks the phrase
#: group from swallowing a SECOND "my"/"the" that starts a closer,
#: correct match (e.g. "the status of the gripper goal": without this
#: exclusion, greedy backtracking captures "status of the gripper"
#: starting from the FIRST "the" instead of just "gripper" starting from
#: the second one - the negative lookahead makes that first attempt fail
#: outright, so the regex engine retries from the second "the" instead).
_PHRASE_WORD = r"(?:(?!\b(?:my|the)\b)[A-Za-z][\w\-]*)"
_PHRASE = rf"({_PHRASE_WORD}(?:\s+{_PHRASE_WORD}){{0,4}})"

#: One (before, after) pattern pair per category - "my/the <phrase>
#: <keyword>" and "<keyword> called/named <phrase>". Small and fixed,
#: the same discipline as every other pattern in this file.
_CATEGORY_KEYWORDS = {
    "mentioned_projects": "project",
    "mentioned_goals": "goal",
    "mentioned_tasks": "task",
    "mentioned_commitments": "commitment",
    "mentioned_resources": "resource",
}

_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    field_name: [
        re.compile(rf"\b(?:my|the)\s+{_PHRASE}\s+{keyword}\b", re.IGNORECASE),
        re.compile(
            rf"\b{keyword}\s+(?:called|named)\s+{_PHRASE}\b", re.IGNORECASE
        ),
    ]
    for field_name, keyword in _CATEGORY_KEYWORDS.items()
}

#: Fallback for a bare "my <phrase>" with no trailing category keyword -
#: e.g. "what is blocking my robotics arm?" (M16.2's own worked example).
#: Treated as a PROJECT hint by default, since Project is this codebase's
#: top-level World Model container (app/world_model/base.py's own
#: hierarchy docstring: Project -> Goal -> linked Task/Job) - the most
#: natural reading of an otherwise-uncategorized "my X" reference.
_GENERIC_MY_PATTERN = re.compile(rf"\bmy\s+{_PHRASE}\b", re.IGNORECASE)


def _extract_mentions(message: str) -> dict[str, list[str]]:
    mentions: dict[str, list[str]] = {field_name: [] for field_name in _CATEGORY_KEYWORDS}

    for phrase in _QUOTED_PHRASE_PATTERN.findall(message):
        cleaned = phrase.strip()
        for field_name in mentions:
            if cleaned not in mentions[field_name]:
                mentions[field_name].append(cleaned)

    matched_spans: list[tuple[int, int]] = []
    for field_name, patterns in _CATEGORY_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(message):
                phrase = match.group(1).strip()
                if phrase and phrase not in mentions[field_name]:
                    mentions[field_name].append(phrase)
                matched_spans.append(match.span(1))

    for match in _GENERIC_MY_PATTERN.finditer(message):
        span = match.span(1)
        # Skip a "my <phrase>" that a category-specific pattern already
        # captured (e.g. "my robotics goal" is a goal mention, not also
        # a project mention) - overlap check against the captured
        # group's own span.
        if any(start <= span[0] < end for start, end in matched_spans):
            continue
        phrase = match.group(1).strip()
        if phrase and phrase not in mentions["mentioned_projects"]:
            mentions["mentioned_projects"].append(phrase)

    return mentions


# ---------------------------------------------------------------------------
# RequestContext construction
# ---------------------------------------------------------------------------


def build_request_context(
    message: str,
    *,
    capability: Capability | None = None,
    now: datetime | None = None,
    device_context=None,
) -> RequestContext:
    """Deterministically build a RequestContext from a raw message and
    (optionally) the Capability the existing M13/M14 CapabilityRouter
    already decided for it - no AI call, no second classifier.

    `required_context` defaults to `default_required_context(intent)`
    (app/context_engine/base.py) - the centralized per-intent category
    set - and `context_budget` defaults to DEFAULT_CONTEXT_BUDGET; a
    caller building a RequestContext some other way (or overriding
    either field afterward) is free to, this function just supplies the
    deterministic defaults every ordinary call site needs.

    `device_context` is the Device Gateway's authenticated DeviceContext for this request (or None). It is
    carried as-is: never derived from `message`, and never used to classify the intent.
    """
    intent = classify_context_intent(message, capability=capability)
    time_scope = extract_time_scope(message, now=now)
    mentions = _extract_mentions(message)

    return RequestContext(
        raw_request=message,
        context_intent=intent,
        mentioned_projects=mentions["mentioned_projects"],
        mentioned_goals=mentions["mentioned_goals"],
        mentioned_tasks=mentions["mentioned_tasks"],
        mentioned_commitments=mentions["mentioned_commitments"],
        mentioned_resources=mentions["mentioned_resources"],
        time_scope=time_scope,
        required_context=default_required_context(intent),
        device_context=device_context,
    )
