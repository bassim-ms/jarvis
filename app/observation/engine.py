"""
ObservationEngine: picks the best observation source and normalises the result. That is all it does.

    1. structured page      (observe_page)   - when a URL is given
    2. window metadata      (observe_window) - the active window's application and title
    3. screenshot (+ vision) (observe_screen) - only when asked for AND nothing structured was enough

What it is NOT: it does not plan, does not act (the only backend method it ever calls is the READ
`get_active_window`), does not decide or grant a permission, does not confirm anything, does not touch Memory or
the World Model, and does not loop. Every call is one bounded observation, rate-limited per minute; callers
(ToolRegistry tools, run by the existing JobEngine) decide what to do with the result.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace

from app.computer.base import ComputerBackend, ComputerControlError
from app.observation.base import (
    MAX_CONTROLS,
    MAX_LABEL_CHARS,
    MAX_OBSERVATIONS_PER_MINUTE,
    MAX_SCREEN_CAPTURES_PER_MINUTE,
    MAX_TITLE_CHARS,
    MAX_VISION_CALLS_PER_MINUTE,
    MAX_VISION_CANDIDATES,
    PAGE_SOURCE_NOTE,
    Control,
    Observation,
    ObservationError,
    scrub,
)
from app.observation.page import PageFetcher, parse_page
from app.observation.screen import CapturedImage, ScreenCapturer, ScreenshotStore, VisionCandidate, VisionInterpreter

_VISION_KINDS = frozenset({"button", "link", "input", "text", "other"})
_DEFAULT_QUESTION = "List the clickable controls and the main visible text."


class _RateLimit:
    """At most `limit` events in any sliding minute."""

    def __init__(self, limit: int, clock: Callable[[], float]):
        self._limit, self._clock, self._events = limit, clock, deque()

    def take(self, what: str) -> None:
        now = self._clock()
        while self._events and now - self._events[0] > 60.0:
            self._events.popleft()
        if len(self._events) >= self._limit:
            raise ObservationError(f"{what} rate limit reached ({self._limit} per minute); try again shortly")
        self._events.append(now)


class ObservationEngine:
    def __init__(
        self,
        *,
        fetcher: PageFetcher,
        backend: ComputerBackend,
        capturer: ScreenCapturer | None = None,
        vision: VisionInterpreter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._fetcher = fetcher
        self._backend = backend
        self._capturer = capturer
        self._vision = vision
        self._observations = _RateLimit(MAX_OBSERVATIONS_PER_MINUTE, clock)
        self._captures = _RateLimit(MAX_SCREEN_CAPTURES_PER_MINUTE, clock)
        self._vision_calls = _RateLimit(MAX_VISION_CALLS_PER_MINUTE, clock)
        self.screenshots = ScreenshotStore(clock)

    # --- the three sources -----------------------------------------------------------------------------------

    def observe_page(self, url: str) -> Observation:
        """Structured observation of a public page: URL, title, visible text, links, buttons, fields."""
        self._observations.take("observation")
        fetched = self._fetcher.fetch(url)
        return replace(parse_page(fetched.html, fetched.url), note=PAGE_SOURCE_NOTE)

    def observe_window(self) -> Observation:
        """The active window's application and title (OS metadata). READ-only."""
        self._observations.take("observation")
        try:
            window = self._backend.get_active_window()
        except ComputerControlError as exc:
            raise ObservationError(f"could not read the active window: {exc}") from exc
        return Observation(
            source="window",
            application=scrub(window.process_name, 80),
            title=scrub(window.title, MAX_TITLE_CHARS),
        )

    def observe_screen(self, question: str | None = None) -> Observation:
        """One bounded screenshot (kept in memory behind a handle) and, if a vision interpreter is configured, its
        UNTRUSTED candidate targets. At most one capture and one vision call per observation."""
        if self._capturer is None:
            raise ObservationError("screen capture is not available")
        self._observations.take("observation")
        self._captures.take("screen capture")
        image = self._capturer.capture()
        handle = self.screenshots.put(image)
        if self._vision is None:
            return Observation(source="screen", screenshot_ref=handle, note="captured; no vision interpreter is configured")
        self._vision_calls.take("vision")
        try:
            candidates = self._vision.interpret(image, scrub(question, 200) or _DEFAULT_QUESTION)
        except Exception as exc:  # noqa: BLE001 - a vision failure is "no candidates", never a crash
            return Observation(source="screen", screenshot_ref=handle, note=f"vision unavailable ({type(exc).__name__})")
        return Observation(
            source="screen",
            screenshot_ref=handle,
            controls=tuple(self._candidates(candidates, image)),
            note="candidates come from a vision model: untrusted suggestions, not facts or permissions",
        )

    # --- selection ---------------------------------------------------------------------------------------------

    def observe(self, *, url: str | None = None, allow_screen: bool = False, question: str | None = None) -> Observation:
        """The best available observation, structured first: the page (when `url` is given), then window metadata,
        then - only if `allow_screen` and nothing structured was sufficient - the screen. A fallback records why
        in `note`; it never silently substitutes one source for another."""
        reason = ""
        if url:
            try:
                page = self.observe_page(url)
                if page.is_sufficient:
                    return page
                reason = "the page had no readable text or controls"
            except ObservationError as exc:
                reason = f"page structure unavailable: {exc}"
        try:
            window = self.observe_window()
            if window.is_sufficient:
                return _with_note(window, reason)
            reason = reason or "the active window had no title"
        except ObservationError as exc:
            reason = reason or str(exc)
        if allow_screen:
            return _with_note(self.observe_screen(question), reason)
        raise ObservationError(reason or "nothing could be observed")

    # --- vision output is validated, bounded and labelled ----------------------------------------------------------

    @staticmethod
    def _candidates(candidates: object, image: CapturedImage) -> list[Control]:
        controls: list[Control] = []
        for candidate in candidates if isinstance(candidates, list) else []:
            if len(controls) >= min(MAX_VISION_CANDIDATES, MAX_CONTROLS):
                break
            if not isinstance(candidate, VisionCandidate):
                continue
            x, y = candidate.x, candidate.y
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y)):
                continue
            if not (0 <= x < image.width and 0 <= y < image.height):
                continue  # outside the image it came from: dropped, not clamped
            kind = candidate.kind if candidate.kind in _VISION_KINDS else "other"
            controls.append(
                Control(
                    f"vision:{kind}",
                    scrub(candidate.label, MAX_LABEL_CHARS),
                    x=round(x * image.screen_width / image.width),
                    y=round(y * image.screen_height / image.height),
                )
            )
        return controls


def _with_note(observation: Observation, note: str) -> Observation:
    if not note:
        return observation
    return replace(observation, note=scrub(f"{note}; {observation.note}" if observation.note else note, 300))
