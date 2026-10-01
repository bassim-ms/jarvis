"""
Deterministic checks of an Observation: "did the expected state appear?" - no model, no I/O, no side effect.

A failed check is a plain, recoverable failure: the verification TOOL turns it into a ToolResult.error, and the
existing JobEngine does the rest (its bounded per-step retry, then its bounded replan). Nothing here loops, retries
or recovers on its own. What is compared is the scrubbed Observation, so a check can never be used to probe for
content the Observation does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.observation.base import Observation, scrub
from app.observation.screen import screen_changed

PAGE_CHECKS = frozenset({"url_contains", "title_contains", "text_contains"})
WINDOW_CHECKS = frozenset({"window_title_contains", "window_process_is"})
SCREEN_CHECKS = frozenset({"screen_changed"})
CHECKS = PAGE_CHECKS | WINDOW_CHECKS | SCREEN_CHECKS

MAX_EXPECTED_CHARS = 200


@dataclass(frozen=True)
class Verification:
    verified: bool
    check: str
    expected: str
    observed: str  # a short, scrubbed excerpt of what was actually seen


def _observed_for(check: str, observation: Observation) -> str:
    return {
        "url_contains": observation.url,
        "title_contains": observation.title,
        "text_contains": observation.text,
        "window_title_contains": observation.title,
        "window_process_is": observation.application,
    }[check]


def check_observation(observation: Observation, check: str, expected: str) -> Verification:
    """Whether `observation` satisfies `check` against `expected` (case-insensitive; `window_process_is` compares
    the whole process name, e.g. "chrome.exe"). Unknown checks and empty expectations never verify."""
    if check == "screen_changed":
        # `expected` is the BASELINE fingerprint (from an earlier observe_screen); verified only if the screen now
        # differs from it by a real margin. Never scrubbed (it is a thumbnail, not text) and never shown back.
        changed, distance = screen_changed(expected, observation.fingerprint)
        measured = "unmeasurable" if distance is None else f"{distance * 100:.2f}% difference"
        return Verification(changed, check, "a changed screen", measured)
    expected = scrub(expected, MAX_EXPECTED_CHARS)
    observed = _observed_for(check, observation) if check in CHECKS else ""
    if check not in CHECKS or not expected:
        matched = False
    elif check == "window_process_is":
        matched = observed.lower() == expected.lower()
    else:
        matched = expected.lower() in observed.lower()
    excerpt = observed if len(observed) <= 160 else observed[:157] + "..."
    return Verification(matched, check, expected, excerpt)
