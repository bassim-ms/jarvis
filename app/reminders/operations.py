"""
The user-facing reminder operations - through the EXISTING request path, not a new one.

JarvisCore hands a message to `command_reply(message, conversation_id)` (the same duck-typed hook the activity context, the
Morning Brief and the device operations use) on the ordinary conversation path only, and only when the WHOLE message is a
reminder request:

  CREATE   "Remind me tomorrow at 7 PM to study" / "every day at 7 PM" / "every weekday at 6 PM" / "every Monday at 5 PM" /
           "in 10 minutes" / "every 2 hours" / "on 5 October at 9 AM"
  LIST     "What reminders do I have?"
  PAUSE / RESUME / DELETE   "pause my study reminder", "resume ...", "delete ..."

This module reads the words (app/reminders/language.py - a fixed grammar; no model, no Planner, no Job) and ACTS only by calling
the registered reminder tools through the ToolRegistry (`create_reminder`, `list_reminders`, `pause_reminder`,
`resume_reminder`, `cancel_reminder`), so their permission levels and ReminderService's rules apply exactly as they do for any
other caller. The application wiring (app/main.py - the only place that may import Handle This) hands it Handle This's
capability-boundary check as `family_on_offer`: asked whether the ORGANIZATION family (reminders) is available with the tools
registered, a "no" means it does nothing and the message is ordinary conversation. Multi-step goals that merely CONTAIN a reminder ("research X
and remind me tomorrow") never reach this door - they are delegated objectives and go to Handle This as before.

Asking, not guessing. At most ONE question is held per conversation, for a few minutes, in memory only:
  * a request with no time ("remind me to study") asks WHEN and keeps what it understood;
  * a bare 12-hour time with an explicit day or repetition ("every day at 7") asks AM or PM;
  * a pause/resume/delete that matches several reminders lists them and asks which ONE.
Only a reply that is nothing but the answer ("7 pm", "the second one") completes it; anything else drops the question and is
treated as an ordinary message. A request is never carried out on several reminders at once: "delete all my reminders" is
declined, and an ambiguous "delete my study reminder" changes nothing until the user picks one.

The reminder's own words are DATA. They are stored and shown as plain text; nothing in them is ever parsed as a command,
looked up as a tool, or fed to a model.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import time as dtime

from app.notifications.text import clean_text
from app.reminders.language import (
    ClockSpec,
    Kind,
    Request,
    When,
    is_drop,
    parse_choice,
    parse_meridiem_answer,
    parse_request,
    parse_when_answer,
    topic_words,
)
from app.reminders.service import ask_meridiem, needs_meridiem
from app.tools.base import ToolResultStatus

PENDING_TTL_SECONDS = 600.0
MAX_PENDING = 50
MAX_CHOICES = 5
MAX_LISTED = 15
_ITEM_CHARS = 80
_DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_NEEDED = {
    Kind.CREATE: ("create_reminder",),
    Kind.LIST: ("list_reminders",),
}
_ACTION_TOOL = {"pause": "pause_reminder", "resume": "resume_reminder", "delete": "cancel_reminder"}
_ACTION_STATES = {"pause": ("pending",), "resume": ("paused",), "delete": ("pending", "paused")}

_BULK = (
    "I won't change several reminders at once. Tell me which one - for example \"delete my study reminder\" - "
    "or ask \"what reminders do I have?\" to see them."
)
_NO_TIME_HINT = "For example: \"tomorrow at 7 PM\"."


@dataclass
class _Option:
    reminder_id: str
    text: str
    line: str


@dataclass
class _Pending:
    kind: str  # "when" | "meridiem" | "choose"
    created: float
    text: str = ""
    when: When = field(default_factory=When)
    clock: ClockSpec | None = None
    action: str = ""
    options: tuple[_Option, ...] = ()


class ReminderOperations:
    def __init__(self, tools, *, family_on_offer=None, monotonic=time.monotonic):
        """`family_on_offer(tool_names) -> bool` is Handle This's boundary check for the reminders family, injected by the
        application wiring. None (a bare registry in a test): only the tools this request needs are checked."""
        self.tools = tools
        self._family_on_offer = family_on_offer
        self._monotonic = monotonic
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    # -- the chat hook ------------------------------------------------------------------------------------------------------

    def command_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        if not isinstance(message, str) or not message.strip():
            return None
        key = conversation_id or ""
        pending = self._take(key)
        if pending is not None:
            answered = self._answer(pending, message, key)
            if answered is not None:
                return answered
        request = parse_request(message)
        if request is None or not self._on_offer(request):
            return None
        if request.kind is Kind.LIST:
            return self._list()
        if request.kind is Kind.MANAGE:
            return self._manage(request, key)
        return self._create(request.text, request.when, key)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- is this door open --------------------------------------------------------------------------------------------------

    def _on_offer(self, request: Request) -> bool:
        """Handle This's boundary says reminders (the ORGANIZATION family) are on offer, and the tools this request needs exist."""
        names = {tool.name for tool in self.tools.list_tools()}
        if self._family_on_offer is not None and not self._family_on_offer(names):
            return False
        if request.kind is Kind.MANAGE:
            return {"list_reminders", _ACTION_TOOL[request.action]} <= names
        return set(_NEEDED[request.kind]) <= names

    # -- creating -----------------------------------------------------------------------------------------------------------

    def _create(self, text: str, when: When, key: str) -> str:
        shown = clean_text(text, _ITEM_CHARS)
        if not text:
            return "What should I remind you about? For example: \"Remind me tomorrow at 7 PM to study\"."
        if when.relative_minutes is not None or (when.relative_days is not None and when.clock is None):
            minutes = when.relative_minutes if when.relative_minutes is not None else when.relative_days * 24 * 60
            return self._run_create({"text": text, "due_in_minutes": minutes})
        if when.every is not None:
            unit, count = when.every
            minutes = count * {"minutes": 1, "hours": 60, "days": 1440, "weeks": 10080}[unit]
            return self._run_create(
                {"text": text, "due_in_minutes": minutes, "recurrence_unit": unit, "recurrence_interval": count}
            )
        if when.clock is None:
            self._hold(key, _Pending("when", self._monotonic(), text=text, when=when))
            return f"When should I remind you about \"{shown}\"? {_NO_TIME_HINT}"
        if needs_meridiem(when.clock, when.day, when.repeat):
            self._hold(key, _Pending("meridiem", self._monotonic(), text=text, when=when, clock=when.clock))
            return ask_meridiem(when.clock)
        params: dict = {"text": text, "at": _at_text(when.clock)}
        if when.day:
            params["day"] = when.day
        if when.repeat:
            params["repeat"] = when.repeat
        if when.weekdays:
            params["weekdays"] = ",".join(_DAY_NAMES[d - 1] for d in when.weekdays)
        return self._run_create(params)

    def _run_create(self, params: dict) -> str:
        result = self.tools.execute("create_reminder", params=params)
        return result.message or "I couldn't set that reminder."

    # -- listing ------------------------------------------------------------------------------------------------------------

    def _active(self) -> list[dict] | None:
        found: list[dict] = []
        for status in ("pending", "paused"):
            result = self.tools.execute("list_reminders", params={"status": status})
            if result.status is not ToolResultStatus.SUCCESS or not isinstance(result.data, list):
                return None
            found.extend(result.data)
        return sorted(found, key=lambda d: d.get("due_at") or "")

    def _list(self) -> str:
        active = self._active()
        if active is None:
            return "I couldn't read your reminders just now."
        if not active:
            return "You don't have any reminders set."
        lines = [f"You have {len(active)} reminder{'' if len(active) == 1 else 's'}:"]
        for number, item in enumerate(active[:MAX_LISTED], start=1):
            lines.append(f"{number}. {_line(item)}")
        if len(active) > MAX_LISTED:
            lines.append(f"...and {len(active) - MAX_LISTED} more.")
        return "\n".join(lines)

    # -- pause / resume / delete --------------------------------------------------------------------------------------------

    def _manage(self, request: Request, key: str) -> str:
        if request.everything or (request.plural and not request.topic):
            return _BULK
        active = self._active()
        if active is None:
            return "I couldn't read your reminders just now."
        words = topic_words(request.topic)
        about = [item for item in active if _matches(words, item)]
        states = _ACTION_STATES[request.action]
        eligible = [item for item in about if item.get("status") in states]
        if not eligible:
            return self._nothing_to_do(request.action, about, request.topic, active)
        if len(eligible) == 1:
            return self._apply(request.action, eligible[0])
        if len(eligible) > MAX_CHOICES:
            return f"That matches {len(eligible)} reminders - tell me more of what it says so I can pick the right one."
        options = tuple(_Option(item["id"], item.get("text", ""), _line(item)) for item in eligible)
        self._hold(key, _Pending("choose", self._monotonic(), action=request.action, options=options))
        verb = {"pause": "pause", "resume": "resume", "delete": "delete"}[request.action]
        listing = "\n".join(f"{number}. {option.line}" for number, option in enumerate(options, start=1))
        return f"That matches more than one reminder:\n{listing}\nWhich one should I {verb}? Say the number."

    @staticmethod
    def _nothing_to_do(action: str, about: list[dict], topic: str, active: list[dict]) -> str:
        if about:
            name = clean_text(about[0].get("text", ""), _ITEM_CHARS)
            if action == "pause":
                return f"\"{name}\" is already paused."
            if action == "resume":
                return f"\"{name}\" isn't paused."
        if not active:
            return "You don't have any reminders set."
        about_what = f" about \"{clean_text(topic, 40)}\"" if topic else ""
        return f"I couldn't find a reminder{about_what}. Ask \"what reminders do I have?\" to see them."

    def _apply(self, action: str, item: dict) -> str:
        result = self.tools.execute(_ACTION_TOOL[action], params={"reminder_id": item["id"]})
        if result.status is not ToolResultStatus.SUCCESS:
            return result.message or "I couldn't do that."
        if action == "delete":
            return f"Deleted: \"{clean_text(item.get('text', ''), _ITEM_CHARS)}\"."
        return result.message

    # -- the one question that may be waiting ---------------------------------------------------------------------------------

    def _hold(self, key: str, pending: _Pending) -> None:
        with self._lock:
            now = self._monotonic()
            for stale in [k for k, p in self._pending.items() if now - p.created > PENDING_TTL_SECONDS]:
                del self._pending[stale]
            while len(self._pending) >= MAX_PENDING and key not in self._pending:
                del self._pending[min(self._pending, key=lambda k: self._pending[k].created)]
            self._pending[key] = pending

    def _take(self, key: str) -> _Pending | None:
        with self._lock:
            pending = self._pending.pop(key, None)
        if pending is None or self._monotonic() - pending.created > PENDING_TTL_SECONDS:
            return None
        return pending

    def _answer(self, pending: _Pending, message: str, key: str) -> str | None:
        """The reply to the question that was asked, or None if `message` is not an answer to it (the question is then dropped)."""
        if pending.kind == "when":
            answer = parse_when_answer(message)
            if answer is None:
                return None
            return self._create(pending.text, _overlay(pending.when, answer), key)
        if pending.kind == "meridiem":
            clock = pending.clock
            at = parse_meridiem_answer(message, clock.hour_text, clock.candidates[0].minute)
            if at is None:
                return None
            return self._create(pending.text, replace(pending.when, clock=ClockSpec((at,), clock.hour_text)), key)
        return self._choose(pending, message, key)

    def _choose(self, pending: _Pending, message: str, key: str) -> str | None:
        if is_drop(message):
            return "Okay - I've left them all as they are."
        number = parse_choice(message)
        if number is not None:
            if not 1 <= number <= len(pending.options):
                self._hold(key, _Pending("choose", self._monotonic(), action=pending.action, options=pending.options))
                return f"Please pick a number from 1 to {len(pending.options)}."
            return self._apply(pending.action, {"id": pending.options[number - 1].reminder_id, "text": pending.options[number - 1].text})
        words = topic_words(message)
        if words and len(message) <= 60:
            narrowed = [o for o in pending.options if _matches(words, {"text": o.text})]
            if len(narrowed) == 1:
                return self._apply(pending.action, {"id": narrowed[0].reminder_id, "text": narrowed[0].text})
        return None


# -- small pure helpers -------------------------------------------------------------------------------------------------------


def _at_text(clock: ClockSpec) -> str:
    """The `at` argument for the tool: exact times as 24-hour HH:MM; a bare 12-hour time as "H:MM" (which the tool reads as
    morning-or-evening and ReminderService resolves)."""
    if clock.ambiguous:
        return f"{int(clock.hour_text)}:{clock.candidates[0].minute:02d}"
    value: dtime = clock.exact
    return f"{value.hour:02d}:{value.minute:02d}"


def _overlay(base: When, answer: When) -> When:
    """What the user said first, with whatever the answer adds or changes."""
    changes = {name: getattr(answer, name) for name in ("clock", "day", "repeat", "every", "relative_minutes", "relative_days")
               if getattr(answer, name) is not None}
    if answer.weekdays:
        changes["weekdays"] = answer.weekdays
    return replace(base, **changes)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _matches(words: tuple[str, ...], item: dict) -> bool:
    """Whether every word of the topic appears in the reminder's text (a plural matches its singular)."""
    if not words:
        return True
    have = {_stem(w) for w in re.findall(r"[a-z0-9']+", (item.get("text") or "").lower())}
    return all(_stem(w) in have for w in words)


def _line(item: dict) -> str:
    text = clean_text(item.get("text", ""), _ITEM_CHARS)
    parts = [f"{text} - {item.get('description') or 'scheduled'}"]
    if item.get("status") == "paused":
        parts.append("(paused)")
    elif item.get("next"):
        parts.append(f"- next: {item['next']}")
    return " ".join(parts)
