"""
The plain data a focus answer is made of. Nothing here reads or writes anything.

Every string held here that came from a stored record (a task, a project, a goal, a calendar title) has already
been through `safe_text` (one bounded line, credential-looking values hidden): it is DATA to show, never an
instruction, and nothing in this package parses it for commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Outcomes
PICK = "pick"  # a concrete first thing, with why
AMBIGUOUS = "ambiguous"  # several projects could be what the user means: ask, rank nothing
NO_PROJECT = "no_project"  # the user said "project" and none matches: ask
ASK_STEP = "ask_step"  # a project was identified but nothing open mentions it: ask for the next step
NO_FOCUS = "no_focus"  # nothing to rank and nothing named: ask what they are working on

# How a candidate relates to what the user named (Objective requests only)
RELATION_NONE = ""
RELATION_NAMED = "named"  # shares a distinctive word with what the user said / the project's name
RELATION_GOAL = "goal"  # shares a distinctive word with one of the project's active goals


@dataclass(frozen=True)
class Candidate:
    """One thing that could be done first, with the score and the named reasons it was given."""

    kind: str  # "task" | "goal"
    ref: str
    text: str
    score: float
    reasons: tuple[str, ...]
    due_ts: float | None = None  # the due moment as an epoch time, for the tie-break only
    created_at: str = ""
    via: str = ""  # the goal a goal-related task was tied through
    matched: tuple[str, ...] = ()  # the distinctive words the task shares with what was named (the whole of the evidence for a "named" link)
    due_text: str = ""  # the user's own words for a due date that could not be read


@dataclass
class FocusDecision:
    outcome: str
    subject: str = ""  # what the user called it
    project: str = ""  # the identified project's name, if any
    picks: list[Candidate] = field(default_factory=list)  # the first thing, then up to two more
    candidate_count: int = 0  # how many candidates were ranked
    heads_up: list[Candidate] = field(default_factory=list)  # urgent but not part of what was named
    goals: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)  # project names to choose from
    more_options: int = 0
    constraints: list[str] = field(default_factory=list)  # calendar / activity facts, never ranked
    unavailable: list[str] = field(default_factory=list)  # sources that could not be read
