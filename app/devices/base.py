"""
Device model: which devices Jarvis knows, what each declares it can do, and whether it is trusted.

This is IDENTITY and METADATA only. A device's capabilities are declarations drawn from a closed vocabulary
(`Capability`) - they describe the hardware, they are never instructions and never permissions: what Jarvis may
DO is decided, unchanged, by the ToolRegistry / capability boundary / confirmation rules, whichever device asks.

A `Device` never carries its credential (only a hash lives in the store), so nothing built from one - a
context, a log line, a prompt - can leak it. `DeviceContext` is what an AUTHENTICATED request carries into
Jarvis; it is produced only by the `DeviceGateway` (gateway.py) and is transient: never written to Memory or
the World Model. The registry is the one source of truth for device identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DeviceType(Enum):
    COMPUTER = "computer"
    PHONE = "phone"
    TABLET = "tablet"
    SMARTWATCH = "smartwatch"
    GLASSES = "glasses"
    EARBUDS = "earbuds"
    GATEWAY = "gateway"
    OTHER = "other"


class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"  # never seen


class TrustState(Enum):
    PENDING = "pending"  # registered, not yet trusted: cannot authenticate
    TRUSTED = "trusted"
    REVOKED = "revoked"  # final: cannot authenticate, cannot be re-trusted (register a new device)


class AuthenticationState(Enum):
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"


class Capability(Enum):
    """What a device DECLARES it has. A closed vocabulary: anything else is rejected at registration."""

    CHAT = "chat"
    DISPLAY = "display"
    AUDIO = "audio"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    BROWSER = "browser"
    NOTIFICATIONS = "notifications"
    VIBRATION = "vibration"
    BUTTONS = "buttons"
    SENSORS = "sensors"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SPEAKER = "speaker"


_C = Capability
DEFAULT_CAPABILITIES: dict[DeviceType, tuple[Capability, ...]] = {
    DeviceType.COMPUTER: (_C.CHAT, _C.DISPLAY, _C.AUDIO, _C.KEYBOARD, _C.MOUSE, _C.BROWSER),
    DeviceType.PHONE: (_C.CHAT, _C.DISPLAY, _C.AUDIO, _C.NOTIFICATIONS),
    DeviceType.TABLET: (_C.CHAT, _C.DISPLAY, _C.AUDIO, _C.NOTIFICATIONS),
    DeviceType.SMARTWATCH: (_C.DISPLAY, _C.VIBRATION, _C.BUTTONS, _C.SENSORS),
    DeviceType.GLASSES: (_C.CAMERA, _C.MICROPHONE, _C.SPEAKER, _C.DISPLAY),
    DeviceType.EARBUDS: (_C.MICROPHONE, _C.SPEAKER),
    DeviceType.GATEWAY: (),
    DeviceType.OTHER: (),
}


class DeviceError(ValueError):
    """A device operation was refused (bad input, unknown device, duplicate name, illegal transition)."""


class DeviceAuthenticationError(Exception):
    """Authentication failed. Deliberately says nothing about WHY (unknown, pending, revoked and a wrong
    credential look identical to a caller); `reason` is for the server's own log only and never contains a
    credential."""

    def __init__(self, reason: str = "failed"):
        super().__init__("device authentication failed")
        self.reason = reason


@dataclass(frozen=True)
class Device:
    device_id: str
    name: str
    device_type: DeviceType
    capabilities: tuple[Capability, ...]
    status: DeviceStatus
    trust_state: TrustState
    last_seen: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DeviceContext:
    """Who is knocking, as established by the Gateway - never by the request's own words. It is information
    for Jarvis's context, and it grants nothing: `capabilities` describe the device, not what it may do."""

    device_id: str
    device_type: DeviceType
    capabilities: tuple[Capability, ...]
    trust_state: TrustState
    authentication_state: AuthenticationState

    @property
    def is_authenticated(self) -> bool:
        return self.authentication_state is AuthenticationState.AUTHENTICATED and self.trust_state is TrustState.TRUSTED
