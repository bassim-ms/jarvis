"""
Changing facts are DERIVED, never re-stored.

Jarvis keeps the stable thing - a date of birth, a start date, a deadline, a recurring date - and works out the changing
thing (age, time since, time remaining, next occurrence) from it and today's date WHEN IT IS ASKED. A derived value is never
written as an authoritative fact, so nothing has to be edited every year:

    date of birth -> current age          start date -> time since
    deadline      -> time remaining       recurring date -> next occurrence

Pure functions only: every one takes `today` explicitly (callers pass the real date; tests pass a fixed one). No clock, no
store, no model. This is deliberately NOT a general temporal-reasoning system - just the few derivations above.

Dates are only ever parsed from UNAMBIGUOUS forms (ISO "2009-03-14", "14 March 2009", "March 14, 2009"). A numeric form such
as 03/04/2009 could be either 3 April or 4 March, so it is never guessed: `parse_date` returns None and the caller asks the
user to write the month in words.
"""

from __future__ import annotations

import re
from datetime import date

#: The one stable fact key a date of birth is stored under (Memory facts). The value is an ISO date.
DOB_KEY = "date_of_birth"
#: The read-time-only key a derived age is shown under to the model. Never written to a store.
DERIVED_AGE_KEY = "age"

MAX_PLAUSIBLE_AGE_YEARS = 130
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
            ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_WORDS = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_WORDS})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_DAY_YEAR = re.compile(rf"\b({_MONTH_WORDS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)


def parse_date(text: object) -> date | None:
    """The date `text` unambiguously states, or None (never a guess)."""
    if not isinstance(text, str):
        return None
    try:
        if m := _ISO.search(text):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m := _DAY_MONTH_YEAR.search(text):
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        if m := _MONTH_DAY_YEAR.search(text):
            return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
    except ValueError:  # 31 February, month 13 ...
        return None
    return None


def plausible_birth_date(born: date, today: date) -> bool:
    """A birth date that is not in the future and not more than MAX_PLAUSIBLE_AGE_YEARS ago."""
    return born <= today and age_on(born, today) <= MAX_PLAUSIBLE_AGE_YEARS


def _anniversary(origin: date, year: int) -> date:
    """`origin`'s month and day in `year`. A 29 February is observed on 1 March in a year that has no 29 February."""
    try:
        return date(year, origin.month, origin.day)
    except ValueError:
        return date(year, 3, 1)


def age_on(born: date, today: date) -> int:
    """Whole years completed between `born` and `today` (never negative)."""
    years = today.year - born.year
    if today < _anniversary(born, today.year):
        years -= 1
    return max(years, 0)


def next_occurrence(origin: date, today: date) -> date:
    """The next date on or after `today` on which `origin`'s month and day recur (a birthday, an annual date)."""
    this_year = _anniversary(origin, today.year)
    return this_year if this_year >= today else _anniversary(origin, today.year + 1)


def days_until(target: date, today: date) -> int:
    return (target - today).days


def time_remaining(deadline: date, today: date) -> int:
    """Days left until `deadline` (0 on the day; negative once it has passed)."""
    return (deadline - today).days


def elapsed_since(start: date, today: date) -> tuple[int, int, int]:
    """(years, months, days) completed between `start` and `today`; (0, 0, 0) if `start` is in the future."""
    if start > today:
        return (0, 0, 0)
    years, months, days = today.year - start.year, today.month - start.month, today.day - start.day
    if days < 0:
        months -= 1
        previous_month_end = date(today.year, today.month, 1).toordinal() - 1
        days += date.fromordinal(previous_month_end).day
    if months < 0:
        years -= 1
        months += 12
    return (years, months, days)


def describe_span(years: int, months: int, days: int) -> str:
    """"2 years, 3 months and 4 days" - parts that are zero are left out."""
    parts = [f"{n} {unit}{'' if n == 1 else 's'}" for n, unit in ((years, "year"), (months, "month"), (days, "day")) if n]
    if not parts:
        return "0 days"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def format_date(value: date) -> str:
    return f"{value.day} {value.strftime('%B %Y')}"


def stored_birth_date(facts: dict[str, str], today: date) -> date | None:
    """The date of birth held in `facts`, if there is one that is a valid, plausible ISO date."""
    value = facts.get(DOB_KEY)
    born = parse_date(value) if isinstance(value, str) and _ISO.fullmatch(value.strip()) else None
    return born if born is not None and plausible_birth_date(born, today) else None


def with_derived_facts(facts: dict[str, str], today: date) -> dict[str, str]:
    """A COPY of `facts` with the changing values worked out for `today` (currently: `age`, from the stored date of
    birth), so a prompt never shows the model a stale or hand-edited age. Nothing is written back; a stored `age`
    fact, if one exists, is replaced in the copy by the derived one."""
    shown = dict(facts)
    born = stored_birth_date(facts, today)
    if born is not None:
        shown[DERIVED_AGE_KEY] = str(age_on(born, today))
    return shown
