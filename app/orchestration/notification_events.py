"""
The notifications the Job/Interaction system sends, built as deterministic,
engine-authored templates (never model-generated) into channel-neutral
`NotificationEvent`s (app/notifications/base.py).

Content rules, enforced here:

- Addressed only as "Sir" or "Boss".
- Short and understandable when heard aloud: the actual question is IN the
  notification, so nobody has to open anything to know what is being asked.
- Untrusted text (a Job's goal, an interaction's prompt) is cleaned and
  bounded on the way in; nothing from a step's error or a tool's output is
  ever included, so no stack traces, tool names or internals leak.
- An interaction reply code appears only for background Jobs (which have no
  conversation to answer in). A Job that belongs to a conversation is
  answered there in plain words.
- Each event has a stable key naming the LOGICAL notification, which is what
  makes delivery at-most-once (see NotificationRouter).

Building an event never changes, resolves or executes anything.
"""

from __future__ import annotations

from app.notifications.base import NotificationEvent, NotificationKind, NotificationPriority
from app.notifications.text import (
    address_for,
    clean_text,
    humanize_remaining,
    sentence,
)
from app.orchestration.base import Job
from app.orchestration.interactions import InteractionRequest, InteractionType

_PROMPT_LIMIT = 200
_GOAL_LIMIT = 60


# --- keys -------------------------------------------------------------------------------------------------


def waiting_key(interaction_id: str) -> str:
    return f"interaction:{interaction_id}:waiting"


def reminder_key(interaction_id: str, ordinal: int) -> str:
    return f"interaction:{interaction_id}:reminder:{ordinal}"


def warning_key(interaction_id: str) -> str:
    return f"interaction:{interaction_id}:warning"


def expired_key(interaction_id: str) -> str:
    return f"interaction:{interaction_id}:expired"


def reminders_prefix(interaction_id: str) -> str:
    """Prefix shared by the reminder keys of one interaction: used to enforce
    the hard maximum on how many it can ever be sent."""
    return f"interaction:{interaction_id}:reminder:"


def job_key(job_id: str, outcome: str) -> str:
    return f"job:{job_id}:{outcome}"


# --- text -------------------------------------------------------------------------------------------------


def _is_dangerous(interaction: InteractionRequest) -> bool:
    return bool(interaction.detail.get("dangerous"))


def _kind_word(interaction: InteractionRequest) -> str:
    return "input" if interaction.type is InteractionType.INPUT else "confirmation"


def _reply_hint(interaction: InteractionRequest, job: Job) -> str:
    """How to answer from another conversation - only for a background Job."""
    if job.conversation_id is not None:
        return ""
    code = interaction.code
    if _is_dangerous(interaction):
        return (
            " This is irreversible, so a plain yes can't approve it. "
            f"To decline, say: answer {code}: no."
        )
    if interaction.type is InteractionType.CONFIRMATION:
        return f" To reply, say: answer {code}: yes, or answer {code}: no."
    return f" To reply, say: answer {code}: and then your answer."


def _question(interaction: InteractionRequest) -> str:
    return sentence(clean_text(interaction.prompt, _PROMPT_LIMIT))


def _event(
    key: str,
    kind: NotificationKind,
    priority: NotificationPriority,
    text: str,
    job: Job,
    interaction: InteractionRequest | None = None,
) -> NotificationEvent:
    return NotificationEvent(
        key=key,
        kind=kind,
        priority=priority,
        text=text,
        conversation_id=job.conversation_id,
        job_id=job.id,
        interaction_code=interaction.code if interaction is not None and job.conversation_id is None else None,
    )


# --- interaction events -----------------------------------------------------------------------------------


def waiting_event(interaction: InteractionRequest, job: Job) -> NotificationEvent:
    key = waiting_key(interaction.id)
    text = (
        f"{address_for(key)}, I need your {_kind_word(interaction)}: {_question(interaction)}"
        + _reply_hint(interaction, job)
    )
    return _event(key, NotificationKind.WAITING, NotificationPriority.NORMAL, text, job, interaction)


def reminder_event(interaction: InteractionRequest, job: Job, ordinal: int) -> NotificationEvent:
    key = reminder_key(interaction.id, ordinal)
    text = (
        f"{address_for(key)}, I'm still waiting for your {_kind_word(interaction)}: "
        f"{_question(interaction)}" + _reply_hint(interaction, job)
    )
    return _event(key, NotificationKind.REMINDER, NotificationPriority.NORMAL, text, job, interaction)


def expiry_warning_event(
    interaction: InteractionRequest, job: Job, remaining_seconds: float
) -> NotificationEvent:
    key = warning_key(interaction.id)
    who = address_for(key)
    when = humanize_remaining(remaining_seconds)
    if interaction.type is InteractionType.INPUT:
        body = f"{who}, I still need your answer: {_question(interaction)} This request expires {when}."
    else:
        body = f"{who}, your confirmation expires {when}: {_question(interaction)}"
    return _event(
        key,
        NotificationKind.EXPIRY_WARNING,
        NotificationPriority.HIGH,
        body + _reply_hint(interaction, job),
        job,
        interaction,
    )


def expired_event(interaction: InteractionRequest, job: Job) -> NotificationEvent:
    key = expired_key(interaction.id)
    who = address_for(key)
    question = clean_text(interaction.prompt, _PROMPT_LIMIT)
    if interaction.type is InteractionType.INPUT:
        text = f"{who}, I didn't get your answer in time, so I've stopped working on it: {question}"
    else:
        text = f"{who}, I didn't get your confirmation in time, so I did not go ahead: {question}"
    return _event(key, NotificationKind.EXPIRED, NotificationPriority.NORMAL, text, job)


# --- job outcome events -----------------------------------------------------------------------------------


def completed_event(job: Job) -> NotificationEvent:
    key = job_key(job.id, "completed")
    text = f'{address_for(key)}, I finished "{clean_text(job.goal, _GOAL_LIMIT)}".'
    return _event(key, NotificationKind.COMPLETED, NotificationPriority.LOW, text, job)


def failed_event(job: Job) -> NotificationEvent:
    """Deliberately no reason: a Job's error text is tool/model-derived
    (names of tools, steps, raw messages), which does not belong in something
    that may be spoken aloud or shown on a lock screen."""
    key = job_key(job.id, "failed")
    text = f'{address_for(key)}, I couldn\'t finish "{clean_text(job.goal, _GOAL_LIMIT)}".'
    return _event(key, NotificationKind.FAILED, NotificationPriority.HIGH, text, job)
