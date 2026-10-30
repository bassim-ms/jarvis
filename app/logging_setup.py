"""
Logging for the running server: what Jarvis did, written somewhere a person can read it.

Without this, no logging configuration exists at all, so every INFO record (the per-request trace in app/core.py, the computer-input audit in
app/computer/gate.py) is silently dropped and only WARNING and above reach the console. `configure_logging` is called once, when the server
starts (app/main.py's lifespan) - never on import, so tests and tools that merely import the app write no file.

WHAT IS WRITTEN. Only Jarvis's own loggers: "jarvis.*" and "app.*" (modules that log under their own `__name__`). Nothing is attached to the
root logger, so third-party records stay out - in particular httpx's INFO request lines, which can carry a provider's API key in the URL.
What each record says is up to its caller; the request trace deliberately records no message text (see JarvisCore._dispatch).

EXCEPTIONS ARE WRITTEN WITHOUT THEIR TEXT. A record logged with `exc_info` (log.exception, exc_info=True, the trace's DEBUG traceback) is
rendered by PayloadFreeFormatter as each exception's TYPE and WHERE it was raised - file, line, function - for the whole chain, and never
`str(exc)`. An exception's text can quote whatever it was handling: a user's message, a Job's objective, a tool argument, a provider's
response, or a request URL - and the Gemini provider sends its key in the URL, which an HTTP error's text (or the original exception
chained behind a cleaned-up one) would repeat. Where it happened and what kind of failure it was stay in the log; the payload never does.
"""

from __future__ import annotations

import logging
import os
import traceback
from logging.handlers import RotatingFileHandler

#: The logger names this module configures; every Jarvis module logs under one of them.
JARVIS_LOGGERS: tuple[str, ...] = ("jarvis", "app")
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3
#: Bounds on what one exception renders as: at most this many chained exceptions, and this many frames of each.
MAX_CHAIN = 5
MAX_FRAMES = 30

#: Marks the handlers this module added, so a second call replaces them instead of stacking duplicates.
_MARK = "_jarvis_logging_setup"


def describe_exception(exc: BaseException | None) -> str:
    """An exception and everything chained to it (its __cause__ / __context__), as types and code locations only - never its text."""
    lines: list[str] = []
    seen: set[int] = set()
    current, label = exc, ""
    while current is not None and id(current) not in seen and len(seen) < MAX_CHAIN:
        seen.add(id(current))
        if label:
            lines.append(label)
        lines.append(f"{type(current).__module__}.{type(current).__qualname__} (message withheld)")
        for frame in traceback.extract_tb(current.__traceback__)[-MAX_FRAMES:]:
            lines.append(f"    at {os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}")
        if current.__cause__ is not None:
            current, label = current.__cause__, "  which was raised from:"
        elif current.__context__ is not None and not current.__suppress_context__:
            current, label = current.__context__, "  which happened while handling:"
        else:
            current = None
    return "\n".join(lines)


class PayloadFreeFormatter(logging.Formatter):
    """A formatter whose exception section names types and code locations only (see describe_exception). It never reuses a record's
    `exc_text` cached by some other handler's formatter, which could hold the full text."""

    def formatException(self, ei) -> str:  # noqa: N802 - logging's own name
        return describe_exception(ei[1] if ei else None)

    def format(self, record: logging.LogRecord) -> str:
        cached = record.exc_text
        record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_text = cached


def configure_logging(log_file: str = "", level: str = "INFO") -> list[logging.Handler]:
    """Send Jarvis's own records at `level` and above to the console and, if `log_file` is given, to a size-bounded rotating file.

    Safe to call again: the handlers it added before are removed first. An unknown level name falls back to INFO. Returns the handlers
    it installed (for tests)."""
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    formatter = PayloadFreeFormatter(FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(RotatingFileHandler(log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _MARK, True)

    for name in JARVIS_LOGGERS:
        logger = logging.getLogger(name)
        for old in [h for h in logger.handlers if getattr(h, _MARK, False)]:
            logger.removeHandler(old)
            old.close()
        logger.setLevel(numeric_level)
        logger.propagate = False  # never also to the root logger: no duplicates, and nothing of Jarvis's mixed into others' handlers
        for handler in handlers:
            logger.addHandler(handler)
    return handlers
