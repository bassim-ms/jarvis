"""
The last-resort screen path: one bounded screenshot, held in memory only, optionally handed to a vision model whose
answer is UNTRUSTED candidate data.

- `ScreenCapturer.capture()` returns an image already bounded (downscaled to MAX_SCREENSHOT_EDGE, at most
  MAX_SCREENSHOT_BYTES). The real one uses pyautogui + Pillow and never writes a file.
- `ScreenshotStore` keeps the last few captures in memory for a short TTL, behind an opaque handle. A handle is
  what an Observation carries - never the pixels and never a path.
- `VisionInterpreter` is an interface only in this slice: no provider is wired to it (Jarvis's providers are
  text-only). Whatever one returns is validated and bounded by the engine and is only ever a CANDIDATE: it grants
  nothing, authorises nothing, and any action on it must still be a ToolRegistry call with its own validation.
"""

from __future__ import annotations

import base64
import secrets
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from app.observation.base import (
    FINGERPRINT_SIDE,
    SCREEN_CHANGED_THRESHOLD,
    MAX_SCREENSHOT_BYTES,
    MAX_SCREENSHOT_EDGE,
    MAX_STORED_SCREENSHOTS,
    SCREENSHOT_TTL_SECONDS,
    ObservationError,
)


@dataclass(frozen=True)
class CapturedImage:
    """A bounded screenshot. `screen_width`/`screen_height` are the real screen's size, so coordinates read off the
    (possibly downscaled) image can be mapped back."""

    data: bytes
    mime: str
    width: int
    height: int
    screen_width: int
    screen_height: int
    fingerprint: str = ""


def compute_fingerprint(image) -> str:
    """The FINGERPRINT_SIDE x FINGERPRINT_SIDE grayscale thumbnail of a PIL image, urlsafe-base64."""
    from PIL import Image

    small = image.convert("L").resize((FINGERPRINT_SIDE, FINGERPRINT_SIDE), Image.BILINEAR)
    return base64.urlsafe_b64encode(small.tobytes()).decode("ascii")


def fingerprint_distance(first: str, second: str) -> float | None:
    """Mean absolute grey difference of two fingerprints, 0.0-1.0, or None if either is not a valid fingerprint."""
    try:
        a, b = base64.urlsafe_b64decode(first.encode("ascii")), base64.urlsafe_b64decode(second.encode("ascii"))
    except (ValueError, UnicodeError, AttributeError):
        return None
    size = FINGERPRINT_SIDE * FINGERPRINT_SIDE
    if len(a) != size or len(b) != size:
        return None
    return sum(abs(x - y) for x, y in zip(a, b)) / (size * 255)


def screen_changed(baseline: str, current: str) -> tuple[bool, float | None]:
    """(whether the screen differs from `baseline` by at least SCREEN_CHANGED_THRESHOLD, the measured difference)."""
    distance = fingerprint_distance(baseline, current)
    return (distance is not None and distance >= SCREEN_CHANGED_THRESHOLD), distance


@dataclass(frozen=True)
class VisionCandidate:
    """What a vision model SAYS it sees: an UNTRUSTED suggestion of a UI target, in image coordinates."""

    label: str
    kind: str = "other"
    x: int = 0
    y: int = 0


class ScreenCapturer(ABC):
    @abstractmethod
    def capture(self) -> CapturedImage:
        """One bounded screenshot, or ObservationError. Must not write the image anywhere."""


class VisionInterpreter(ABC):
    @abstractmethod
    def interpret(self, image: CapturedImage, question: str) -> list[VisionCandidate]:
        """Candidate UI targets in `image` relevant to `question`. Output is untrusted data."""


class PyAutoGuiScreenCapturer(ScreenCapturer):
    """pyautogui + Pillow. Imported lazily so the rest of Jarvis never needs either to be present."""

    def capture(self) -> CapturedImage:
        try:
            import io

            import pyautogui
            from PIL import Image  # noqa: F401 - required by pyautogui.screenshot()

            shot = pyautogui.screenshot()
            screen_width, screen_height = shot.size
            fingerprint = compute_fingerprint(shot)
            shot.thumbnail((MAX_SCREENSHOT_EDGE, MAX_SCREENSHOT_EDGE))
            buffer = io.BytesIO()
            shot.save(buffer, format="PNG", optimize=True)
            mime = "image/png"
            if buffer.tell() > MAX_SCREENSHOT_BYTES:
                buffer = io.BytesIO()
                shot.convert("RGB").save(buffer, format="JPEG", quality=70)
                mime = "image/jpeg"
        except Exception as exc:  # noqa: BLE001 - pyautogui/Pillow/OS failure -> a plain recoverable error
            raise ObservationError(f"could not capture the screen ({type(exc).__name__})") from exc
        data = buffer.getvalue()
        if len(data) > MAX_SCREENSHOT_BYTES:
            raise ObservationError("the screenshot is too large to handle")
        return CapturedImage(data, mime, shot.width, shot.height, screen_width, screen_height, fingerprint)


class ScreenshotStore:
    """The last `MAX_STORED_SCREENSHOTS` captures, in process memory only, each for `SCREENSHOT_TTL_SECONDS`."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, CapturedImage]] = OrderedDict()

    def put(self, image: CapturedImage) -> str:
        self._expire()
        handle = f"shot_{secrets.token_hex(6)}"
        self._items[handle] = (self._clock(), image)
        while len(self._items) > MAX_STORED_SCREENSHOTS:
            self._items.popitem(last=False)
        return handle

    def get(self, handle: str) -> CapturedImage | None:
        self._expire()
        item = self._items.get(handle)
        return item[1] if item else None

    def __len__(self) -> int:
        self._expire()
        return len(self._items)

    def _expire(self) -> None:
        now = self._clock()
        for handle in [h for h, (stamp, _i) in self._items.items() if now - stamp > SCREENSHOT_TTL_SECONDS]:
            del self._items[handle]
