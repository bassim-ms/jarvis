"""
The small set of user-facing device operations - through the EXISTING request path, not a new one.

JarvisCore hands a message to `command_reply(message, conversation_id)` (the same duck-typed hook the activity
context and the Morning Brief use) only when the WHOLE message is one of a few fixed phrasings. There is no
intent engine and no model call here.

  READ     "what devices do you have?" / "is my phone connected?" / "what can my PC do?" - answered straight
           from the registry, read-only. "What device am I using?" / "what can this device do?" are answered
           from the AUTHENTICATED DeviceContext of the request (looked up in the registry), or - on the
           anonymous /chat, which is never proof of a device - an honest "I can't tell which device this is".
  REVOKE   "revoke my phone" - CONSEQUENTIAL, so it is not done: it is ASKED, through the same Job / interaction /
           ToolRegistry confirmation the memory policy uses (a waiting Job whose one step is the CONFIRM-level,
           non-plannable, non-automatable `commit_device_revocation`). Only an approved confirmation of THAT Job
           revokes anything; a "no", an expiry or silence changes nothing. Revoking only ever REMOVES trust.
  REGISTER "register a device" - not done from chat: registering issues a secret credential, which must never pass
           through a conversation, a prompt or a log. The reply says to use the admin command on this machine.
  TRUST    There is no chat path at all. A device becomes TRUSTED only through the admin command; "make this
           device trusted" is ordinary conversation.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.devices.base import Device, DeviceError, DeviceStatus, DeviceType, TrustState
from app.devices.registry import DeviceRegistry
from app.orchestration.base import Job, JobStatus, Plan, PlanStep
from app.orchestration.interactions import InteractionStatus, InteractionType, default_ttl_seconds
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

COMMIT_TOOL_NAME = "commit_device_revocation"
ORIGIN = "device_gateway"
GOAL = "Revoke a device"
PENDING_TTL_SECONDS = 3600.0
MAX_PENDING = 20
_STEP_ID = "s1"
_NO_REPLANS = 10_000

_I = re.IGNORECASE
_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis|please)[,.!\s]+)*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
_END = r"\s*[.!?]*\s*$"
_WHO = r"(?:my\s+|the\s+)?(?P<who>[a-z0-9][a-z0-9 _'-]{0,40}?)"


def _rx(body: str) -> re.Pattern:
    return re.compile(_LEAD + body + _END, _I)


_LIST = _rx(
    r"(?:(?:what|which)\s+devices\s+(?:do\s+(?:you|i)\s+have|are\s+(?:registered|connected|known))|"
    r"what\s+are\s+my\s+(?:registered\s+)?devices|"
    r"(?:show|list)(?:\s+me)?\s+(?:all\s+)?(?:my\s+)?(?:registered\s+)?devices|my\s+devices)"
)
_CONNECTED = _rx(r"is\s+" + _WHO + r"\s+(?:connected|online|offline|available|on)")
_CAPABILITIES = _rx(r"what\s+can\s+" + _WHO + r"\s+do")
_REVOKE = _rx(r"revoke\s+" + _WHO)
_THIS_DEVICE = _rx(
    r"(?:(?:what|which)\s+(?:device|machine|computer)\s+(?:am\s+i|are\s+we)\s+(?:using|on)|"
    r"(?:what|which)\s+device\s+is\s+this|what\s+is\s+this\s+(?:device|machine)|what'?s\s+this\s+(?:device|machine))"
)
_THIS_CAPABILITIES = _rx(r"what\s+can\s+this\s+(?:device|machine|computer)\s+do")
_REGISTER = _rx(r"register\s+(?:a\s+|another\s+)?(?:new\s+)?device")

#: Words that name a KIND of device ("my phone"). A name that is not one of these must match a registered name.
_TYPE_WORDS = {
    "pc": DeviceType.COMPUTER, "computer": DeviceType.COMPUTER, "laptop": DeviceType.COMPUTER, "desktop": DeviceType.COMPUTER,
    "phone": DeviceType.PHONE, "mobile": DeviceType.PHONE, "tablet": DeviceType.TABLET,
    "watch": DeviceType.SMARTWATCH, "smartwatch": DeviceType.SMARTWATCH, "glasses": DeviceType.GLASSES,
    "smart glasses": DeviceType.GLASSES, "earbuds": DeviceType.EARBUDS, "earphones": DeviceType.EARBUDS,
    "headphones": DeviceType.EARBUDS,
}

REGISTER_REPLY = (
    "Registering a device issues a secret credential, so it is not done in chat. On this machine run "
    "\"python -m app.devices.admin register --help\"; a new device starts untrusted until you trust it there."
)


@dataclass
class _Pending:
    device_id: str
    created: float
    job_id: str | None = None


class DeviceRevocationConfirmer:
    """Asks the user before a device is revoked, through the existing Job / interaction machinery (see
    app/memory_policy/confirmation.py, whose shape this follows)."""

    def __init__(self, interactions, *, mark_surfaced: Callable[[str, str], object] | None = None):
        self.interactions = interactions
        self.mark_surfaced = mark_surfaced

    def approved(self, job_id: str) -> bool:
        """Whether the user APPROVED this Job's confirmation - read from the interaction store, so a forged call
        (or the step of a declined Job) can never revoke."""
        return any(
            i.type is InteractionType.CONFIRMATION
            and i.status is InteractionStatus.RESOLVED
            and i.detail.get("origin") == ORIGIN
            for i in self.interactions.list_for_job(job_id)
        )

    def ask(self, pending_id: str, prompt: str, conversation_id: str | None) -> str:
        from app.orchestration.sqlite_store import SQLiteJobStore

        job: Job = SQLiteJobStore.build_job(GOAL, conversation_id=conversation_id, status=JobStatus.AWAITING_CONFIRMATION)
        job.plan = Plan(
            steps=[PlanStep(id=_STEP_ID, tool_name=COMMIT_TOOL_NAME, arguments={"pending_id": pending_id}, permission=PermissionLevel.CONFIRM.value)]
        )
        job.total_step_count = 1
        job.pending_step_ids = [_STEP_ID]
        job.replan_count = _NO_REPLANS  # never re-planned by a model
        job.confirmation_message = prompt
        created, interaction = self.interactions.create_job_awaiting_confirmation(
            job, prompt=prompt, step_ids=[_STEP_ID], detail={"origin": ORIGIN, "dangerous": False},
            ttl_seconds=default_ttl_seconds(InteractionType.CONFIRMATION),
        )
        if self.mark_surfaced is not None and conversation_id:
            try:
                self.mark_surfaced(interaction.id, conversation_id)
            except Exception:  # noqa: BLE001 - best effort: the ordinary "waiting on you" notice shows it next
                pass
        return created.id


class DeviceOperations:
    def __init__(self, registry: DeviceRegistry, confirmer: DeviceRevocationConfirmer | None = None):
        self.registry = registry
        self.confirmer = confirmer
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    # -- the chat hook ------------------------------------------------------------------------------------------------

    def command_reply(self, message: str, conversation_id: str | None = None, device_context=None) -> str | None:
        """`device_context` is the Gateway-authenticated DeviceContext of THIS request (None on the anonymous
        /chat). It is only ever used to answer "this device" questions - never to change anything."""
        if not isinstance(message, str) or not message.strip() or len(message) > 160 or "\n" in message:
            return None
        if _THIS_DEVICE.match(message):
            return self._this_device(device_context, capabilities=False)
        if _THIS_CAPABILITIES.match(message):
            return self._this_device(device_context, capabilities=True)
        if _LIST.match(message):
            return self._list()
        if _REGISTER.match(message):
            return REGISTER_REPLY
        for pattern, handler in ((_CONNECTED, self._connected), (_CAPABILITIES, self._capabilities), (_REVOKE, self._revoke)):
            m = pattern.match(message)
            if m:
                resolved = self._resolve(m.group("who"))
                if resolved is None:
                    return None  # not a device: ordinary conversation
                return handler(resolved, conversation_id)
        return None

    # -- reading ------------------------------------------------------------------------------------------------------

    def _this_device(self, device_context, *, capabilities: bool) -> str:
        """The sending device, from the authenticated context and the registry (the source of truth). Read-only."""
        if device_context is None or not getattr(device_context, "is_authenticated", False):
            return (
                "I can't tell which device this is: this request didn't come through an authenticated device. "
                "(The plain chat interface isn't proof of a device.)"
            )
        device = self.registry.get(device_context.device_id)
        if device is None:
            return "This request says it came from a device I no longer have registered, so I can't vouch for it."
        if device.trust_state is not TrustState.TRUSTED:
            return f"{device.name} is {device.trust_state.value}, so I'm not treating this request as coming from a trusted device."
        if capabilities:
            return self._capabilities(device, None)
        status = "connected" if device.status is DeviceStatus.ONLINE else device.status.value
        return f"You're on {device.name}, a {device.device_type.value} - trusted and {status}."

    def _list(self) -> str:
        devices = self.registry.list_devices()
        if not devices:
            return "I don't have any devices registered yet."
        lines = [f"I have {len(devices)} device{'' if len(devices) == 1 else 's'} registered:"]
        lines += [f"- {_line(d)}" for d in devices]
        return "\n".join(lines)

    def _connected(self, resolved, conversation_id) -> str:
        device = self._one(resolved)
        if isinstance(device, str):
            return device
        if device.trust_state is TrustState.REVOKED:
            return f"{device.name} has been revoked, so it can't connect."
        if device.trust_state is TrustState.PENDING:
            return f"{device.name} is registered but not trusted yet, so it isn't connected."
        if device.status is DeviceStatus.ONLINE:
            return f"Yes - {device.name} is connected (last seen {_ago(device, self.registry)})."
        if device.last_seen is None:
            return f"No - {device.name} hasn't connected yet."
        return f"No - {device.name} is offline (last seen {_ago(device, self.registry)})."

    def _capabilities(self, resolved, conversation_id) -> str:
        device = self._one(resolved)
        if isinstance(device, str):
            return device
        caps = ", ".join(c.value for c in device.capabilities) or "nothing declared"
        return f"{device.name} declares: {caps}. Those describe the device; they don't change what I'm allowed to do."

    # -- revoking (asked, never just done) --------------------------------------------------------------------------------

    def _revoke(self, resolved, conversation_id) -> str:
        device = self._one(resolved)
        if isinstance(device, str):
            return device
        if device.trust_state is TrustState.REVOKED:
            return f"{device.name} is already revoked."
        if self.confirmer is None:
            return "I can't ask for that confirmation right now, so nothing was changed."
        pending_id = _new_id()
        with self._lock:
            self._prune_locked()
            self._pending[pending_id] = _Pending(device.device_id, time.monotonic())
        prompt = f"Shall I permanently revoke {device.name}? It will no longer be able to connect to me."
        try:
            job_id = self.confirmer.ask(pending_id, prompt, conversation_id)
        except Exception:  # noqa: BLE001 - it cannot be asked: nothing is revoked, nothing stays held
            logger.warning("device revocation confirmation failed")
            with self._lock:
                self._pending.pop(pending_id, None)
            return "I couldn't ask for that confirmation, so nothing was changed."
        with self._lock:
            held = self._pending.get(pending_id)
            if held is not None:
                held.job_id = job_id
        return f"{prompt} Please confirm - say yes or no."

    def commit_revocation(self, pending_id: str, job_id: str | None) -> str:
        """Revoke the device a confirmation was asked about - called ONLY by the CONFIRM-level tool, i.e. only when
        the user approved the confirmation of THAT Job (`job_id` comes from the engine, never from a plan)."""
        with self._lock:
            self._prune_locked()
            held = self._pending.get(pending_id)
            approved = getattr(self.confirmer, "approved", None)
            if held is None:
                raise DeviceError("that request has expired, so nothing was changed")
            if not job_id or held.job_id != job_id or approved is None or not approved(job_id):
                raise DeviceError("that can only be done by the confirmation that was asked")
            del self._pending[pending_id]
        device = self.registry.revoke(held.device_id)
        return f"{device.name} has been revoked."

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, p in self._pending.items() if now - p.created > PENDING_TTL_SECONDS]:
            del self._pending[key]
        while len(self._pending) >= MAX_PENDING:
            del self._pending[min(self._pending, key=lambda k: self._pending[k].created)]

    # -- resolving "my phone" -----------------------------------------------------------------------------------------------

    def _resolve(self, who: str):
        """A device name, or a kind of device ("phone"), or None if it is neither (then it is not about devices)."""
        named = self.registry.store.get_by_name(who.strip())
        if named is not None:
            return self.registry.view(named)
        kind = _TYPE_WORDS.get(" ".join(who.casefold().split()))
        return kind

    def _one(self, resolved) -> Device | str:
        if isinstance(resolved, Device):
            return resolved
        matches = [d for d in self.registry.list_devices() if d.device_type is resolved and d.trust_state is not TrustState.REVOKED]
        if not matches:
            matches = [d for d in self.registry.list_devices() if d.device_type is resolved]
        if not matches:
            return f"I don't have a {resolved.value} registered."
        if len(matches) > 1:
            names = ", ".join(d.name for d in matches)
            return f"I have more than one {resolved.value}: {names}. Tell me which one by name."
        return matches[0]


def _line(device: Device) -> str:
    status = device.status.value if device.trust_state is TrustState.TRUSTED else "not connected"
    return f"{device.name} ({device.device_type.value}): {device.trust_state.value}, {status}"


def _ago(device: Device, registry: DeviceRegistry) -> str:
    if device.last_seen is None:
        return "never"
    seconds = max(0, int((registry._clock() - device.last_seen).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex


class CommitDeviceRevocationTool(Tool):
    """Revoke a device the user has just CONFIRMED revoking. Not for models, plans or automations: CONFIRM-level
    (the engine runs it only after the user approved the confirmation that gates it, through the existing
    ToolRegistry check), non-plannable, non-automatable, and it refuses any call that does not come from the very
    Job that asked (`source_job_id` is added by JobEngine and cannot be supplied by a plan or a caller). It can only
    REMOVE trust, never grant it."""

    name = COMMIT_TOOL_NAME
    description = "Revoke a device the user has just confirmed revoking."
    permission = PermissionLevel.CONFIRM
    plannable = False
    automatable = False
    wants_job_provenance = True

    def __init__(self, operations: DeviceOperations):
        self.operations = operations

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="pending_id", type="string", description="The held request's id")]

    def execute(self, pending_id: str, source_job_id: str | None = None, source_step_id: str | None = None) -> ToolResult:
        if not source_job_id:
            return ToolResult.error("This can only run as an approved confirmation.")
        try:
            return ToolResult.success(message=self.operations.commit_revocation(pending_id, source_job_id))
        except DeviceError as exc:
            return ToolResult.error(str(exc))
