"""
The Device Gateway: IDENTITY -> AUTHENTICATION -> DEVICE CONTEXT. Nothing else.

    device presents (device_id, credential)
      -> look the device up in the registry      (the registry is the source of truth)
      -> compare the credential with the stored hash, in constant time
      -> require TRUST_STATE == TRUSTED
      -> hand back a DeviceContext, or fail closed

It has no business logic: it never plans, runs a tool, creates a Job, approves a confirmation, or touches
memory, goals or tasks. A `DeviceContext` is information for Jarvis's context; it carries no permission - the
ToolRegistry, the capability boundary and the confirmation rules decide what may be done, exactly as before,
whichever device asked. Who the device IS comes only from the credential it proves, never from what a message
says ("I am the trusted PC" is just text).

Every failure looks the same to the caller (`DeviceAuthenticationError`); the reason goes to the server log
only, and never contains a credential. An unknown id costs the same comparison as a known one.
"""

from __future__ import annotations

import hmac
import logging

from app.devices.base import (
    AuthenticationState,
    Device,
    DeviceAuthenticationError,
    DeviceContext,
    TrustState,
)
from app.devices.registry import DeviceRegistry, hash_credential

logger = logging.getLogger(__name__)

MAX_ID_CHARS = 64
MAX_CREDENTIAL_CHARS = 256
_DUMMY_HASH = hash_credential("no such device")


class DeviceGateway:
    def __init__(self, registry: DeviceRegistry):
        self.registry = registry

    def authenticate(self, device_id: object, credential: object) -> DeviceContext:
        """A DeviceContext for a TRUSTED device that proved its credential; DeviceAuthenticationError otherwise."""
        if (
            not isinstance(device_id, str) or not isinstance(credential, str)
            or not device_id or not credential
            or len(device_id) > MAX_ID_CHARS or len(credential) > MAX_CREDENTIAL_CHARS
        ):
            self._fail("malformed", None)
        stored = self.registry.store.credential_hash(device_id)
        matches = hmac.compare_digest(hash_credential(credential), stored or _DUMMY_HASH)
        device = self.registry.store.get(device_id)
        if stored is None or device is None or not matches:
            self._fail("unknown device or wrong credential", device_id)
        if device.trust_state is not TrustState.TRUSTED:
            self._fail(f"device is {device.trust_state.value}", device_id)
        return self._context(device)

    def check_in(self, device_id: object, credential: object) -> Device:
        """An authenticated "I am alive": ONLINE, last_seen = now. No background process is involved."""
        context = self.authenticate(device_id, credential)
        self.registry.store.record_check_in(context.device_id, self.registry._clock())
        return self.registry.get(context.device_id)

    @staticmethod
    def _context(device: Device) -> DeviceContext:
        return DeviceContext(
            device_id=device.device_id,
            device_type=device.device_type,
            capabilities=device.capabilities,
            trust_state=device.trust_state,
            authentication_state=AuthenticationState.AUTHENTICATED,
        )

    @staticmethod
    def _fail(reason: str, device_id: str | None):
        # The id is not a secret; the credential is never logged.
        logger.warning("device authentication failed: %s (device_id=%r)", reason, (device_id or "-")[:MAX_ID_CHARS])
        raise DeviceAuthenticationError(reason)
