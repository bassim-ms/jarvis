"""
The Device Registry: the one source of truth for which devices exist, what they declare and whether they are
trusted. It registers, trusts and revokes devices; it does nothing else (it never plans, runs a tool, or
touches a Job).

CREDENTIALS. Registering a device issues a random credential ONCE (returned to the admin who registered it,
never stored): only its SHA-256 hash is kept. The credential is 256 bits of randomness, so a plain hash is
enough to store it safely (there is nothing to brute-force); comparison is constant-time (see gateway.py).

TRUST. A new device is PENDING and cannot authenticate. Only `trust()` - reachable from the admin command, never
from chat or from a device itself - makes it TRUSTED, and only from PENDING. `revoke()` is final: the credential
hash is wiped, so the device can never authenticate again; a replacement is a NEW registration. Nothing is ever
trusted from a name, an address, "localhost" or anything the device or a message says about itself.

ONLINE / OFFLINE is deterministic: a device is ONLINE only while its last authenticated check-in is newer than
`stale_after`; otherwise it reads as OFFLINE (or UNKNOWN if it has never checked in). No daemon is involved.
"""

from __future__ import annotations

import hashlib
import secrets
import unicodedata
import uuid
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from app.devices.base import (
    DEFAULT_CAPABILITIES,
    Capability,
    Device,
    DeviceError,
    DeviceStatus,
    DeviceType,
    TrustState,
)
from app.devices.store import DeviceStore

MAX_NAME_CHARS = 60
STALE_AFTER = timedelta(minutes=5)


def hash_credential(credential: str) -> str:
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()


def _clean_name(name: object) -> str:
    if not isinstance(name, str):
        raise DeviceError("a device needs a name")
    cleaned = " ".join("".join(" " if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") else c for c in name).split())
    if not cleaned or len(cleaned) > MAX_NAME_CHARS:
        raise DeviceError(f"a device name must be 1-{MAX_NAME_CHARS} characters")
    return cleaned


def _device_type(value: object) -> DeviceType:
    if isinstance(value, DeviceType):
        return value
    try:
        return DeviceType(value)
    except ValueError:
        raise DeviceError("unknown device type") from None


def _capabilities(values: Iterable[object] | None, device_type: DeviceType) -> tuple[Capability, ...]:
    if values is None:
        return DEFAULT_CAPABILITIES[device_type]
    found: list[Capability] = []
    for value in values:
        try:
            capability = value if isinstance(value, Capability) else Capability(value)
        except ValueError:
            raise DeviceError("unknown capability") from None
        if capability not in found:
            found.append(capability)
    return tuple(found)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DeviceRegistry:
    def __init__(self, store: DeviceStore, *, clock: Callable[[], datetime] = utc_now, stale_after: timedelta = STALE_AFTER):
        self.store = store
        self._clock = clock
        self.stale_after = stale_after

    # -- registering / trusting / revoking -----------------------------------------------------------------------------

    def register(
        self,
        name: str,
        device_type: DeviceType | str,
        capabilities: Iterable[Capability | str] | None = None,
        *,
        status: DeviceStatus = DeviceStatus.UNKNOWN,
    ) -> tuple[Device, str]:
        """Register a PENDING device. Returns (device, credential); the credential is shown here and never again.
        The id is generated here - a device never chooses its own identity."""
        now = self._clock()
        kind = _device_type(device_type)
        device = Device(
            device_id="dev_" + uuid.uuid4().hex[:16],
            name=_clean_name(name),
            device_type=kind,
            capabilities=_capabilities(capabilities, kind),
            status=status,
            trust_state=TrustState.PENDING,
            last_seen=None,
            created_at=now,
            updated_at=now,
        )
        credential = secrets.token_urlsafe(32)
        self.store.add(device, hash_credential(credential))
        return device, credential

    def trust(self, device_id: str) -> Device:
        device = self._require(device_id)
        if device.trust_state is not TrustState.PENDING:
            raise DeviceError(f"only a pending device can be trusted (this one is {device.trust_state.value})")
        self.store.set_trust(device_id, TrustState.TRUSTED, self._clock())
        return self.view(self._require(device_id))

    def revoke(self, device_id: str) -> Device:
        device = self._require(device_id)
        if device.trust_state is TrustState.REVOKED:
            raise DeviceError("that device is already revoked")
        self.store.set_trust(device_id, TrustState.REVOKED, self._clock())
        return self.view(self._require(device_id))

    # -- reading ---------------------------------------------------------------------------------------------------------------

    def get(self, device_id: str) -> Device | None:
        device = self.store.get(device_id)
        return self.view(device) if device else None

    def find(self, name_or_id: str) -> Device | None:
        device = self.store.get(name_or_id) or self.store.get_by_name(name_or_id)
        return self.view(device) if device else None

    def list_devices(self) -> list[Device]:
        return [self.view(d) for d in self.store.list()]

    def view(self, device: Device) -> Device:
        """The device as it reads NOW: ONLINE only while its last check-in is fresh."""
        if device.status is DeviceStatus.ONLINE:
            fresh = device.last_seen is not None and self._clock() - device.last_seen <= self.stale_after
            if not fresh or device.trust_state is not TrustState.TRUSTED:
                return replace(device, status=DeviceStatus.OFFLINE)
        return device

    def _require(self, device_id: str) -> Device:
        device = self.store.get(device_id)
        if device is None:
            raise DeviceError("unknown device")
        return device
