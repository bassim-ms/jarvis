"""
The observed-anchor cache: which same-page section anchors (#workflow, #savings ...) an observation of a page actually
exposed, so a Job may only open `page#anchor` when that exact anchor was SEEN on that exact page.

Why: the Planner writes its plan before it has seen the page, so left alone it guesses an anchor from a label
("How It Works" -> #how-it-works) that the page may not have. A guessed anchor opens the page at the top while the Job
reports success. This cache lets the Job's `open_url` refuse an anchor that was never observed (deterministically, with
a message naming the anchors that WERE observed), which the existing retry / replan then recovers from.

What it is NOT: not a planner, not an executor, not persistent. It holds only anchor NAMES that passed
`safe_anchor` (a plain section identifier - never a token, route or key=value pair), keyed by a scrubbed page URL
(`safe_url`: no user-info, no sensitive query parameters, no fragment). It is bounded three ways - at most
MAX_ANCHORS_PER_PAGE names per page, MAX_ANCHOR_PAGES pages (least recently used goes first), and each page's entry
lives ANCHOR_TTL_SECONDS - and lives only in process memory. An anchor name is inert data: it is compared to a stored
set, never read as an instruction.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

from app.observation.base import (
    ANCHOR_TTL_SECONDS,
    MAX_ANCHOR_PAGES,
    MAX_ANCHORS_PER_PAGE,
    safe_anchor,
    safe_url,
)


def page_key(url: object) -> str:
    """The identity of the PAGE a URL points at: scrubbed, lower-cased host, an empty path read as "/", the (scrubbed)
    query, and NO fragment. "" for anything that is not an http(s) URL."""
    cleaned = safe_url(url)
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    host = (parts.hostname or "").lower() + (f":{parts.port}" if parts.port else "")
    return f"{parts.scheme}://{host}{parts.path or '/'}" + (f"?{parts.query}" if parts.query else "")


class AnchorCache:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        *,
        ttl: float = ANCHOR_TTL_SECONDS,
        max_pages: int = MAX_ANCHOR_PAGES,
        max_anchors: int = MAX_ANCHORS_PER_PAGE,
    ):
        self._clock, self._ttl, self._max_pages, self._max_anchors = clock, ttl, max_pages, max_anchors
        self._pages: OrderedDict[str, tuple[float, tuple[str, ...]]] = OrderedDict()

    def remember(self, page_urls: Iterable[str], anchors: Iterable[str]) -> None:
        """Record that the page(s) at `page_urls` (the requested URL and the one it ended up at) exposed `anchors`.
        Every name is re-checked with `safe_anchor`; duplicates and anything past the cap are dropped. An empty set is
        remembered too: "observed, and it has none" is different from "never observed"."""
        names: list[str] = []
        for anchor in anchors:
            if anchor and safe_anchor(anchor) == anchor and anchor not in names and len(names) < self._max_anchors:
                names.append(anchor)
        stamp = self._clock()
        for url in page_urls:
            key = page_key(url)
            if not key:
                continue
            self._pages[key] = (stamp, tuple(names))
            self._pages.move_to_end(key)
        while len(self._pages) > self._max_pages:
            self._pages.popitem(last=False)

    def observed(self, url: object) -> tuple[str, ...] | None:
        """The anchors observed for the page `url` points at (its own fragment ignored): a tuple (possibly empty),
        or None when that page was never observed or its entry has expired."""
        self._expire()
        entry = self._pages.get(page_key(url))
        return entry[1] if entry else None

    def __len__(self) -> int:
        self._expire()
        return len(self._pages)

    def _expire(self) -> None:
        now = self._clock()
        for key in [k for k, (stamp, _a) in self._pages.items() if now - stamp > self._ttl]:
            del self._pages[key]
