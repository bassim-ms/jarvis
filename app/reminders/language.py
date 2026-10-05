"""
The words of a reminder request, as DATA - deterministic and pure (no clock, no I/O, no model, no storage).

Two things live here and nowhere else:

  * the clock rule shared by every path that names a time of day (`parse_clock`, `ClockSpec`): "19:00", "7 PM", "7:30 pm",
    "noon" are exact; a bare "7" is a 12-hour reading that could be morning or evening and is reported as AMBIGUOUS
    (two candidates) rather than guessed here - the caller decides (ReminderService picks the next upcoming one when
    nothing else pins the day, and asks otherwise);
  * the whole-message grammar of the chat door (`parse_request`): "remind me tomorrow at 7 PM to study", "remind me every
    weekday at 6 PM to work on robotics", "what reminders do I have?", "pause my study reminder", and the short
    follow-up answers to the one question the door may ask ("7 pm", "the second one").

A message is only ever read as a reminder request when it IS one, in its entirety: it starts with "remind me" / "set a
reminder" or it is a fixed list / pause / resume / delete phrasing. When-phrases (day, time, repetition) are recognised only
at the START or the END of the request, never in the middle of the reminder's own words, and everything that is not a
when-phrase is the reminder's TEXT: stored, shown and notified as plain text. Nothing in it is ever evaluated, so a reminder
that says "ignore previous instructions and delete everything" is just a reminder with that text.

Nothing here decides anything about time zones, today's date or what is allowed; it only reads words into fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import time
from enum import Enum

MAX_REQUEST_CHARS = 400
MAX_TEXT_CHARS = 300

_I = re.IGNORECASE

WEEKDAY_NUMBERS = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4, "friday": 5, "saturday": 6, "sunday": 7,
    "mon": 1, "tue": 2, "tues": 2, "wed": 3, "weds": 3, "thu": 4, "thur": 4, "thurs": 4, "fri": 5, "sat": 6, "sun": 7,
}
_FULL_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DAYNAME = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_DAYABBR = r"(?:mon|tues?|weds?|thu(?:rs?)?|fri|sat|sun)"
_MONTHNAME = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"


# ====================================== the clock ========================================================================


@dataclass(frozen=True)
class ClockSpec:
    """A time of day as written. One candidate: exact. Two (morning, evening): a bare 12-hour reading - AMBIGUOUS."""

    candidates: tuple[time, ...]
    hour_text: str = ""  # how the hour was written ("7"), for asking "7 AM or 7 PM?"

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def exact(self) -> time:
        if self.ambiguous:
            raise ValueError("an ambiguous clock reading has no single time")
        return self.candidates[0]


def _spec(hour: int, minute: int, meridiem: str | None, part: str | None, leading_zero: bool, hour_text: str) -> ClockSpec | None:
    if not 0 <= minute <= 59:
        return None
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        return ClockSpec((time(hour % 12 + (12 if meridiem == "pm" else 0), minute),), hour_text)
    if part:
        if not 1 <= hour <= 12:
            return None
        if part == "morning":
            return ClockSpec((time(hour % 12, minute),), hour_text)
        if part == "night":
            if hour == 12:
                return ClockSpec((time(0, minute),), hour_text)
            return ClockSpec((time(hour if hour < 6 else hour % 12 + 12, minute),), hour_text)
        return ClockSpec((time(hour % 12 + 12, minute),), hour_text)  # afternoon / evening
    if hour > 23:
        return None
    if hour == 0 or hour >= 13 or leading_zero:
        return ClockSpec((time(hour, minute),), hour_text)  # a 24-hour reading: "19:00", "07:30", "0:30"
    return ClockSpec((time(hour % 12, minute), time(hour % 12 + 12, minute)), hour_text)  # 1-12, bare: morning or evening


_WORD_CLOCKS = {"noon": time(12, 0), "midday": time(12, 0), "midnight": time(0, 0)}
_CLOCK_BODY = (
    r"(?:(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?(?:\s*(?P<ap>am|pm|a\.m\.|p\.m\.)(?![a-z]))?"
    r"(?:\s+(?:in\s+the\s+)?(?P<part>morning|afternoon|evening|night)\b)?"
    r"(?!\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?|st\b|nd\b|rd\b|th\b))(?![\d:/-])"
    r"|(?P<word>noon|midday|midnight)\b)"
)
_CLOCK_STRICT = re.compile(rf"^\s*{_CLOCK_BODY}\s*$", _I)


def _clock_from(match: re.Match) -> ClockSpec | None:
    word = match.group("word")
    if word:
        return ClockSpec((_WORD_CLOCKS[word.lower()],), word.lower())
    digits = match.group("h")
    ap = (match.group("ap") or "").replace(".", "").lower() or None
    part = (match.group("part") or "").lower() or None
    return _spec(int(digits), int(match.group("mi") or 0), ap, part, digits.startswith("0") and len(digits) == 2, digits)


def parse_clock(value: object) -> ClockSpec | None:
    """A time of day written on its own ("19:00", "7 pm", "7:30", "noon"), or None if it isn't one. Used for the `at`
    argument of the reminder tool, so a Planner and a person are held to exactly the same reading."""
    if not isinstance(value, str) or len(value) > 40:
        return None
    match = _CLOCK_STRICT.match(value)
    return _clock_from(match) if match else None


# ====================================== what a request says ==============================================================


class Kind(Enum):
    CREATE = "create"
    LIST = "list"
    MANAGE = "manage"


@dataclass(frozen=True)
class When:
    """Everything the timing words said. All fields are optional; the caller decides what is enough."""

    clock: ClockSpec | None = None
    day: str | None = None  # "today" | "tonight" | "tomorrow" | "monday" | "next monday" | "YYYY-MM-DD" | "MM-DD" | "+N"
    repeat: str | None = None  # "daily" | "weekdays" | "weekly"
    weekdays: tuple[int, ...] = ()  # ISO weekdays, for "weekly"
    every: tuple[str, int] | None = None  # ("hours", 2): a repeat measured in elapsed time
    relative_minutes: int | None = None  # "in 10 minutes"
    relative_days: int | None = None  # "in 3 days" (a date, so it may be combined with a time of day)

    @property
    def empty(self) -> bool:
        return self == When()


@dataclass(frozen=True)
class Request:
    kind: Kind
    text: str = ""
    when: When = field(default_factory=When)
    action: str = ""  # MANAGE: "pause" | "resume" | "delete"
    topic: str = ""  # MANAGE: what the user called it ("study")
    plural: bool = False  # MANAGE: asked about reminders in general ("my reminders") rather than one
    everything: bool = False  # MANAGE: "all"/"every" - never honoured, only recognised so it can be declined


_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis|please)[,.!\s]+)*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
_CREATE = re.compile(
    _LEAD
    + r"(?:remind\s+me|(?:set|add|create|make)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:new\s+)?reminder)\b[\s,:]*(?P<rest>.*)$",
    _I | re.S,
)
_END_PUNCT = re.compile(r"[\s.!?]+$")

_N = r"(?P<n>\d{1,4})"
_UNIT = r"(?P<u>minutes?|mins?|hours?|hrs?|days?|weeks?)"
_REL = re.compile(rf"^in\s+(?:half\s+an?\s+(?P<half>hour)|(?:an?|{_N})\s+{_UNIT})\b", _I)
_EVERY_N = re.compile(rf"^every\s+(?:(?:{_N}|(?P<other>other))\s+)?{_UNIT}\b", _I)
_DAILY = re.compile(r"^(?:every\s*day|each\s+day|daily)\b", _I)
_WEEKDAYS = re.compile(r"^(?:every\s+week\s*day|each\s+week\s*day|(?:on\s+)?week\s*days)\b", _I)
_DAYLIST = rf"(?:{_DAYNAME}|{_DAYABBR})(?:\s*(?:,|and|&)\s*(?:{_DAYNAME}|{_DAYABBR}))*"
_WEEKLY = re.compile(rf"^(?:every|each)\s+(?P<days>{_DAYLIST})\b", _I)
_WEEKLY_PLURAL = re.compile(rf"^on\s+(?P<days>(?:{_DAYNAME})s(?:\s*(?:,|and|&)\s*(?:{_DAYNAME})s)*)\b", _I)
_DAYWORD = re.compile(r"^(?P<w>today|tonight|tomorrow)\b", _I)
_WEEKDAY = re.compile(rf"^(?:(?P<o>on|this|next)\s+)?(?P<d>{_DAYNAME}|(?<=on )(?:{_DAYABBR}))\b", _I)
_ISO_DATE = re.compile(r"^(?:on\s+)?(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})\b", _I)
_DAY_MONTH = re.compile(rf"^(?:on\s+)?(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mo>{_MONTHNAME})(?:\s+(?P<y>\d{{4}}))?\b", _I)
_MONTH_DAY = re.compile(rf"^(?:on\s+)?(?P<mo>{_MONTHNAME})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?\b", _I)
_AT_CLOCK = re.compile(rf"^(?:at|@)\s*{_CLOCK_BODY}", _I)
_BARE_CLOCK = re.compile(rf"^{_CLOCK_BODY}", _I)
_SEPARATOR = re.compile(r"^[\s,;]*(?:and\s+)?", _I)
_CONNECTIVE = re.compile(r"^[\s,;:]*(?:(?:to|that|about|of|for)\s+)?", _I)

_UNIT_NAMES = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def _unit(raw: str) -> str:
    return _UNIT_NAMES[raw[0].lower()]


def _iso(year: str | None, month: int, day: int) -> str | None:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{int(year):04d}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"


def _days(raw: str) -> tuple[int, ...]:
    """ISO weekday numbers named in "monday and wednesday" / "mondays, fridays"."""
    found = set()
    for word in re.findall(r"[a-z]+", raw.lower()):
        word = word if word in WEEKDAY_NUMBERS else word.rstrip("s")
        if word in WEEKDAY_NUMBERS:
            found.add(WEEKDAY_NUMBERS[word])
    return tuple(sorted(found))


def _take(text: str, when: When) -> tuple[When, str] | None:
    """Read ONE when-phrase from the start of `text` into `when`. Returns the updated When and the remaining text, or None
    if `text` does not start with one (or it would contradict what is already known)."""
    m = _REL.match(text)
    if m:
        if not when.empty:
            return None
        if m.group("half"):
            return replace(when, relative_minutes=30), text[m.end():]
        n = int(m.group("n") or 1)
        unit = _unit(m.group("u"))
        if unit == "minutes":
            return replace(when, relative_minutes=n), text[m.end():]
        if unit == "hours":
            return replace(when, relative_minutes=n * 60), text[m.end():]
        return replace(when, relative_days=n * (7 if unit == "weeks" else 1)), text[m.end():]
    for pattern, repeat in ((_DAILY, "daily"), (_WEEKDAYS, "weekdays")):
        m = pattern.match(text)
        if m:
            if when.repeat or when.every or when.day or when.relative_days or when.relative_minutes:
                return None
            return replace(when, repeat=repeat), text[m.end():]
    m = _WEEKLY.match(text) or _WEEKLY_PLURAL.match(text)
    if m:
        days = _days(m.group("days"))
        if not days or when.repeat or when.every or when.day or when.relative_days or when.relative_minutes:
            return None
        return replace(when, repeat="weekly", weekdays=days), text[m.end():]
    m = _EVERY_N.match(text)
    if m:
        if when.repeat or when.every or when.day or when.relative_days or when.relative_minutes:
            return None
        n = 2 if m.group("other") else int(m.group("n") or 1)
        if n < 1:
            return None
        return replace(when, every=(_unit(m.group("u")), n)), text[m.end():]
    if when.day is None and not when.repeat and when.every is None and when.relative_minutes is None and when.relative_days is None:
        m = _DAYWORD.match(text)
        if m:
            return replace(when, day=m.group("w").lower()), text[m.end():]
        m = _ISO_DATE.match(text)
        if m:
            value = _iso(m.group("y"), int(m.group("mo")), int(m.group("d")))
            if value:
                return replace(when, day=value), text[m.end():]
        m = _DAY_MONTH.match(text) or _MONTH_DAY.match(text)
        if m:
            value = _iso(m.group("y"), MONTH_NUMBERS[m.group("mo").lower()], int(m.group("d")))
            if value:
                return replace(when, day=value), text[m.end():]
        m = _WEEKDAY.match(text)
        if m:
            name = m.group("d").lower()
            number = WEEKDAY_NUMBERS.get(name)
            if number:
                full = _FULL_DAYS[number - 1]
                return replace(when, day=(f"next {full}" if (m.group("o") or "").lower() == "next" else full)), text[m.end():]
    if when.clock is None:
        m = _AT_CLOCK.match(text)
        if m:
            clock = _clock_from(m)
            return (replace(when, clock=clock), text[m.end():]) if clock else None
        m = _BARE_CLOCK.match(text)
        if m and (m.group("ap") or m.group("mi") or m.group("part") or m.group("word")):
            clock = _clock_from(m)
            return (replace(when, clock=clock), text[m.end():]) if clock else None
    return None


def _consume(text: str, when: When) -> tuple[When, str]:
    """Take as many when-phrases as there are from the start of `text`."""
    for _ in range(8):
        text = _SEPARATOR.sub("", text, count=1)
        taken = _take(text, when)
        if taken is None:
            break
        when, text = taken
    return when, text


def _finish(when: When) -> When:
    """Fold "in 3 days" into a date offset when a time of day goes with it, and read "tonight at 8" as 8 PM."""
    if when.relative_days is not None and when.clock is not None and when.day is None:
        when = replace(when, day=f"+{when.relative_days}", relative_days=None)
    if when.day == "tonight" and when.clock is not None and when.clock.ambiguous and when.clock.candidates[0].hour != 0:
        # "tonight at 8" is the evening: narrow a bare 1-11 to its PM reading ("tonight at 12" stays a question)
        return replace(when, clock=replace(when.clock, candidates=(when.clock.candidates[1],)))
    return when


def split_when(rest: str) -> tuple[When, str]:
    """(the timing words, the reminder's own text) for what follows "remind me": when-phrases at the start, then an optional
    "to"/"about"/"that", then the text, then when-phrases at the very end."""
    rest = re.sub(r"^for\s+", "", _END_PUNCT.sub("", rest.strip()), count=1, flags=_I)
    when, rest = _consume(rest, When())
    rest = _CONNECTIVE.sub("", rest, count=1).strip()
    # when-phrases at the END: the longest tail that is entirely timing words (so "on friday at 5" loses its "on")
    starts = [m.start() for m in re.finditer(r"(?<!\S)\S", rest)]
    for start in starts[1:]:  # never the whole text: a request must keep some words of its own
        tail_when, left = _consume(rest[start:], when if when.empty else When())
        if left.strip() == "" and not tail_when.empty:
            merged = _merge(when, tail_when)
            if merged is not None:
                return _finish(merged), _tidy(rest[:start], cut=True)
    return _finish(when), _tidy(rest, cut=False)


def split_leading_when(rest: str) -> tuple[When, str]:
    """Like `split_when`, but ONLY the timing words at the START are read; everything after them (and any "to"/"that"/"about") is the
    text, whatever it ends with. For callers whose text is a whole instruction ("research X and summarize tomorrow"): its own closing
    words must never be taken for a second timing."""
    rest = re.sub(r"^for\s+", "", _END_PUNCT.sub("", rest.strip()), count=1, flags=_I)
    when, rest = _consume(rest, When())
    rest = _CONNECTIVE.sub("", rest, count=1).strip()
    return _finish(when), _tidy(rest, cut=False)


_DANGLING = re.compile(r"\s+(?:by|on|at|before|until|till|for|every|each)$", _I)


def _tidy(text: str, *, cut: bool) -> str:
    """The reminder's words with surrounding punctuation taken off - and, only when a time phrase was just cut from the
    end, the preposition that led into it ("submit the report by" + "friday at 5 pm" -> "submit the report")."""
    text = text.strip(" ,;:-")
    return (_DANGLING.sub("", text) if cut else text).strip(" ,;:-")


def _merge(first: When, second: When) -> When | None:
    """Combine timing words from both ends of a request; None if they contradict each other."""
    if first.empty:
        return second
    for name in ("clock", "day", "repeat", "every", "relative_minutes", "relative_days"):
        if getattr(first, name) is not None and getattr(second, name) is not None:
            return None
    if first.weekdays and second.weekdays:
        return None
    return When(
        clock=first.clock or second.clock,
        day=first.day or second.day,
        repeat=first.repeat or second.repeat,
        weekdays=first.weekdays or second.weekdays,
        every=first.every or second.every,
        relative_minutes=first.relative_minutes if first.relative_minutes is not None else second.relative_minutes,
        relative_days=first.relative_days if first.relative_days is not None else second.relative_days,
    )


# ====================================== management and listing ===========================================================

_END = r"\s*[.!?]*\s*$"
_LIST = re.compile(
    _LEAD
    + r"(?:(?:what|which)\s+reminders\s+do\s+i\s+have(?:\s+(?:set|now|right\s+now|coming\s+up))?|"
    r"what\s+(?:are|is)\s+my\s+(?:active\s+|upcoming\s+|pending\s+)?reminders?|"
    r"what'?s\s+my\s+(?:active\s+|upcoming\s+)?reminders?|"
    r"(?:show|list|tell)(?:\s+me)?\s+(?:all\s+)?(?:of\s+)?(?:my\s+)?(?:active\s+|upcoming\s+|pending\s+)?reminders|"
    r"do\s+i\s+have\s+any\s+(?:active\s+|upcoming\s+)?reminders(?:\s+set)?|"
    r"my\s+(?:active\s+|upcoming\s+)?reminders)" + _END,
    _I,
)
_VERBS = {"pause": "pause", "suspend": "pause", "hold": "pause", "resume": "resume", "unpause": "resume",
          "restart": "resume", "delete": "delete", "remove": "delete", "cancel": "delete", "clear": "delete"}
_MANAGE = re.compile(
    _LEAD
    + r"(?P<verb>pause|suspend|hold|resume|unpause|restart|delete|remove|cancel|clear)\s+(?:(?:my|the|that|this|a|an)\s+)?"
    r"(?P<body>.+?)" + _END,
    _I,
)
_REMINDER_NOUN = re.compile(r"\breminders?\b", _I)
_TOPIC_NOISE = frozenset(
    "my the a an that this about for to of on regarding please reminder reminders called named titled one ones".split()
)


def topic_words(topic: str) -> tuple[str, ...]:
    return tuple(w for w in re.findall(r"[a-z0-9']+", topic.lower()) if w not in _TOPIC_NOISE)


def _manage(match: re.Match) -> Request | None:
    body = match.group("body").strip()
    noun = _REMINDER_NOUN.search(body)
    if noun is None:
        return None  # "delete my files", "cancel the meeting": not about reminders at all
    before, after = body[: noun.start()].strip(), body[noun.end():].strip()
    topic = (before if before else after).strip()
    everything = bool(re.match(r"^(?:all|every|everything|each)\b", before, _I)) or bool(re.search(r"\b(?:all|every|everything)\b", before, _I))
    plural = noun.group(0).lower().endswith("s")
    stripped = " ".join(w for w in topic.split() if w.lower() not in ("all", "every", "of", "them", "my", "the"))
    if everything:
        stripped = ""
    return Request(Kind.MANAGE, action=_VERBS[match.group("verb").lower()], topic=stripped, plural=plural, everything=everything)


def parse_request(message: object) -> Request | None:
    """The reminder request a message is, in its entirety - or None (ordinary conversation, left to the rest of Jarvis)."""
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_REQUEST_CHARS:
        return None
    if _LIST.match(message):
        return Request(Kind.LIST)
    manage = _MANAGE.match(message)
    if manage:
        return _manage(manage)
    create = _CREATE.match(message)
    if create:
        when, text = split_when(create.group("rest"))
        return Request(Kind.CREATE, text=" ".join(text.split()), when=when)
    return None


# ====================================== follow-up answers ================================================================

_ORDINALS = {"first": 1, "1st": 1, "one": 1, "second": 2, "2nd": 2, "two": 2, "third": 3, "3rd": 3, "three": 3,
             "fourth": 4, "4th": 4, "four": 4, "fifth": 5, "5th": 5, "five": 5}
_CHOICE = re.compile(r"^\s*(?:(?:the|number|no\.?|#)\s*)?(?P<n>\d|first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|one|two|three|four|five)(?:\s+one)?\s*[.!]*\s*$", _I)
_MERIDIEM = re.compile(
    r"^\s*(?:at\s+)?(?:(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?\s*)?(?:(?P<ap>am|pm|a\.m\.|p\.m\.)|(?:in\s+the\s+|this\s+)?(?P<part>morning|afternoon|evening|night))\s*[.!]*\s*$",
    _I,
)
_DROP = re.compile(r"^\s*(?:never\s*mind|forget\s+it|cancel(?:\s+that)?|none|neither|no\s+thanks?|stop|nothing)\s*[.!]*\s*$", _I)


def parse_choice(message: object) -> int | None:
    """"2", "the second one", "number 3" -> the 1-based number; anything else None."""
    if not isinstance(message, str):
        return None
    match = _CHOICE.match(message)
    if not match:
        return None
    token = match.group("n").lower()
    return int(token) if token.isdigit() else _ORDINALS.get(token)


def is_drop(message: object) -> bool:
    return isinstance(message, str) and bool(_DROP.match(message))


def parse_when_answer(message: object) -> When | None:
    """A reply that is NOTHING BUT timing words ("7 pm", "tomorrow at 9", "every day at 7 pm"), or None."""
    if not isinstance(message, str) or not message.strip() or len(message) > 80:
        return None
    text = _END_PUNCT.sub("", message.strip())
    when, left = _consume(text, When())
    if left.strip() or when.empty:
        return None
    return _finish(when)


def parse_meridiem_answer(message: object, hour_text: str, minute: int) -> time | None:
    """The answer to "7 AM or 7 PM?": "pm", "7 pm", "in the evening" -> the time asked about. A number that is not the hour
    asked about is not an answer."""
    if not isinstance(message, str) or len(message) > 40:
        return None
    match = _MERIDIEM.match(message)
    if not match:
        return None
    if match.group("h") and match.group("h").lstrip("0") != hour_text.lstrip("0"):
        return None
    if match.group("mi") and int(match.group("mi")) != minute:
        return None
    hour = int(hour_text)
    ap = (match.group("ap") or "").replace(".", "").lower() or None
    part = (match.group("part") or "").lower() or None
    spec = _spec(hour, minute, ap, part, False, hour_text)
    return spec.candidates[0] if spec is not None and not spec.ambiguous else None
