"""
Wording a focus decision: fixed templates only, bounded, deterministic.

Stored text (a task, a project, a goal, a calendar title) is placed into a quoted slot of a template that Jarvis
wrote; it is never interpreted. No model is asked to word anything.
"""

from __future__ import annotations

from app.context_engine.scoring import (
    REASON_EXPLICIT_MATCH,
    REASON_PRIORITY_HIGH,
    REASON_RELATIONSHIP_GOAL,
    REASON_TIME_FAR_FUTURE,
    REASON_TIME_NOW,
    REASON_TIME_OVERDUE,
    REASON_TIME_THIS_WEEK,
    REASON_TIME_TODAY,
    REASON_TIME_TOMORROW,
)
from app.focus.base import AMBIGUOUS, ASK_STEP, NO_FOCUS, NO_PROJECT, PICK, Candidate, FocusDecision

MAX_CHARS = 1500
MAX_MATCHED_WORDS = 3

_DUE_WORDS = {
    REASON_TIME_OVERDUE: "overdue",
    REASON_TIME_NOW: "due within the hour",
    REASON_TIME_TODAY: "due today",
    REASON_TIME_TOMORROW: "due tomorrow",
    REASON_TIME_THIS_WEEK: "due this week",
    REASON_TIME_FAR_FUTURE: "due later",
}
_UNAVAILABLE_WORDS = {
    "tasks": "I couldn't read your tasks, so I can't see what's open.",
    "calendar": "I couldn't read your calendar, so I can't see what's on it or how much time you have.",
    "projects": "I couldn't read your projects.",
    "goals": "I couldn't read your goals.",
}


def quoted(text: str) -> str:
    """Stored text goes into the reply only like this: a quoted slot of a sentence Jarvis wrote - data, never Jarvis's own words."""
    return f'"{text}"'


def _why(pick: Candidate, decision: FocusDecision) -> str:
    phrases: list[str] = []
    for reason in pick.reasons:
        if reason == REASON_EXPLICIT_MATCH:
            # All the evidence there is: the task's own words overlap what was named. It says that, and no more.
            words = ", ".join(quoted(word) for word in pick.matched[:MAX_MATCHED_WORDS])
            phrases.append(f"it mentions {words}" if words else "it shares a word with what you named")
        elif reason == REASON_RELATIONSHIP_GOAL:
            phrases.append(f"it shares a word with your goal {quoted(pick.via)} (a looser match)")
        elif reason in _DUE_WORDS:
            phrases.append(f"it's {_DUE_WORDS[reason]}")
        elif reason == REASON_PRIORITY_HIGH:
            phrases.append("you marked it high priority")
        elif reason == "due_unreadable":
            phrases.append(f"its due date ({quoted(pick.due_text)}) isn't one I can read")
    if decision.candidate_count == 1:
        phrases.append("it's the only open task I found for this")
    if not phrases:
        return "Why: no deadline or priority is set on any of them, so I went by what has been waiting longest."
    return "Why: " + "; ".join(phrases) + "."


def _tags(candidate: Candidate) -> str:
    tags = [_DUE_WORDS[r] for r in candidate.reasons if r in _DUE_WORDS]
    if REASON_PRIORITY_HIGH in candidate.reasons:
        tags.append("high priority")
    return ", ".join(tags)


def _names(decision: FocusDecision) -> str:
    names = ", ".join(quoted(name) for name in decision.options)
    return names + (f" and {decision.more_options} more" if decision.more_options else "")


def _notes(decision: FocusDecision) -> list[str]:
    return [_UNAVAILABLE_WORDS.get(label, f"I couldn't read {label}.") for label in decision.unavailable]


def render_decision(decision: FocusDecision) -> str:
    """The reply for a decision. The first thing (and any question) always survive the length bound; extras are dropped from the end."""
    must: list[str] = []
    extras: list[str] = []

    if decision.outcome == PICK:
        first = decision.picks[0]
        if decision.project:
            must.append(f"This sounds like your project {quoted(decision.project)}. Start with: {quoted(first.text)}.")
        elif decision.subject:
            must.append(
                f"I don't have a project called {quoted(decision.subject)} on record, but open tasks mention it. "
                f"Start with: {quoted(first.text)}."
            )
        else:
            must.append(f"Start with: {quoted(first.text)}.")
        must.append(_why(first, decision))
        for c in decision.picks[1:]:
            tags = _tags(c)
            extras.append(f"Then: {quoted(c.text)}" + (f" ({tags})." if tags else "."))
        for item in decision.heads_up:
            extras.append(f"Not part of this, but {_tags(item)}: {quoted(item.text)}.")
        extras.extend(decision.constraints)
    elif decision.outcome == AMBIGUOUS:
        must.append(f"More than one project could be {quoted(decision.subject)}: {_names(decision)}. Which one do you mean?")
        must.append("Tell me which, and I'll work out where to start.")
    elif decision.outcome == NO_PROJECT:
        if decision.options:
            must.append(f"I don't have a project matching {quoted(decision.subject)}. Your active projects: {_names(decision)}. Which one did you mean?")
        elif "tasks" in decision.unavailable:
            must.append(f"I don't have a project matching {quoted(decision.subject)}, and I couldn't read your tasks to check. What are you trying to get done?")
        else:
            must.append(f"I don't have a project matching {quoted(decision.subject)}, and no open task mentions it. What are you trying to get done?")
    elif decision.outcome == ASK_STEP:
        if "tasks" in decision.unavailable:
            must.append(f"This sounds like your project {quoted(decision.project)}, but I can't see its tasks.")
        else:
            must.append(f"This sounds like your project {quoted(decision.project)}, but I don't see an open task for it.")
        if decision.goals:
            must.append("Its active goals: " + "; ".join(quoted(g) for g in decision.goals) + ".")
        must.append("What's the next step you want to take on it?")
        extras.extend(decision.constraints)
    elif decision.outcome == NO_FOCUS:
        if "tasks" in decision.unavailable:
            must.append("I can't tell what to start with without your tasks.")
        else:
            must.append("I don't have any open tasks to rank.")
        if decision.options:
            must.append(f"Your active projects: {_names(decision)}.")
        must.append("Tell me what you're working on - for example, name a project - and I'll work out where to start.")

    must.extend(_notes(decision))  # a source that could not be read is always said
    while extras and len("\n".join(must + extras)) > MAX_CHARS:
        extras.pop()
    return "\n".join(must + extras)
