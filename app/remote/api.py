"""
The remote API (`/remote/v1/...`) and the dashboard's view of remote access - see app/remote/__init__.py for the whole design.

    POST   /remote/v1/session   device id + credential (+ fresh timestamp and unused nonce) -> a session: id, key, expiry
    GET    /remote/v1/status    signed: who you are, when the session ends, the server's time
    POST   /remote/v1/chat      signed: the ordinary Jarvis chat, as the authenticated device (the same path as /device/chat)
    DELETE /remote/v1/session   signed: disconnect

Every route answers 403 while remote access is off (the default). Every authentication failure is the same 401 with the same
text - nothing tells a caller WHY (unknown device, wrong credential, revoked, expired, bad signature, replay).
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.devices.base import DeviceAuthenticationError, DeviceContext
from app.remote import signing
from app.remote.sessions import NonceCache, RemoteSessions, ReplyCache, Session

logger = logging.getLogger("jarvis.remote")

MAX_CLOCK_SKEW = timedelta(seconds=60)
REPLY_CACHE_TTL = timedelta(minutes=10)
AUTH_FAILED = "remote authentication failed"
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)+")


def parse_remote_hosts(text: str) -> frozenset[str]:
    """"jarvis-pc.tail1234.ts.net, other.example" -> the host names remote requests may be addressed to. Only plain DNS names
    with at least one dot are accepted - never "localhost", an IP address, a wildcard, a port or a path."""
    hosts = set()
    for part in (text or "").split(","):
        host = part.strip().lower().rstrip(".")
        if _HOST.fullmatch(host) and not host.replace(".", "").isdigit() and not host.endswith("localhost"):
            hosts.add(host)
    return frozenset(hosts)


class RemoteChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=100)


@dataclass
class RemoteAccess:
    """Remote access state for this Jarvis: the live sessions and the checks every remote request goes through."""

    gateway: object  # app.devices.gateway.DeviceGateway
    registry: object  # app.devices.registry.DeviceRegistry
    enabled: Callable[[], bool]
    hosts: Callable[[], frozenset[str]]
    session_ttl: timedelta = timedelta(hours=12)
    max_sessions_per_device: int = 5
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    sessions: RemoteSessions = field(init=False)
    nonces: NonceCache = field(init=False)
    replies: ReplyCache = field(init=False)

    def __post_init__(self):
        self.sessions = RemoteSessions(self.session_ttl, self.max_sessions_per_device)
        self.nonces = NonceCache(2 * MAX_CLOCK_SKEW)
        self.replies = ReplyCache(REPLY_CACHE_TTL)

    # -- checks -----------------------------------------------------------------------------------------------------

    def fresh(self, timestamp: object, nonce: object, scope: str, now: datetime) -> bool:
        if not isinstance(timestamp, str) or not timestamp.isdigit() or len(timestamp) > 12:
            return False
        if not isinstance(nonce, str) or not signing.NONCE_PATTERN.fullmatch(nonce):
            return False
        sent = datetime.fromtimestamp(int(timestamp), timezone.utc)
        if abs(now - sent) > MAX_CLOCK_SKEW:
            return False
        return self.nonces.first_use(scope, nonce, now)

    # -- the dashboard's view (local only) -----------------------------------------------------------------------

    def overview(self) -> dict:
        now = self.clock()
        sessions = self.sessions.active(now)
        devices = []
        for device in self.registry.list_devices():
            mine = [s for s in sessions if s.device_id == device.device_id]
            devices.append({
                "device_id": device.device_id,
                "name": device.name,
                "type": device.device_type.value,
                "trust": device.trust_state.value,
                "status": device.status.value,
                "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                "sessions": [
                    {"session": s.session_id[:10] + "...", "session_id": s.session_id, "created_at": s.created_at.isoformat(),
                     "expires_at": s.expires_at.isoformat(), "last_used": s.last_used.isoformat(), "requests": s.requests}
                    for s in mine
                ],
            })
        return {
            "enabled": bool(self.enabled()),
            "hosts": sorted(self.hosts()),
            "session_hours": round(self.session_ttl.total_seconds() / 3600, 2),
            "active_sessions": len(sessions),
            "devices": devices,
        }

    def end_session(self, session_id: str) -> bool:
        return self.sessions.end(session_id)

    def revoke_device(self, device_id: str) -> dict:
        """Revoke a device NOW: its trust becomes REVOKED (final - its credential can never authenticate again) and every
        session it has ends. Revoking only ever takes access away."""
        device = self.registry.revoke(device_id)
        ended = self.sessions.end_device(device_id)
        return {"device_id": device.device_id, "name": device.name, "trust": device.trust_state.value, "sessions_ended": ended}


def build_remote_router(access: RemoteAccess, chat: Callable[[str, str | None, DeviceContext], dict]) -> APIRouter:
    router = APIRouter(prefix="/remote/v1")

    def require_enabled() -> None:
        if not access.enabled():
            raise HTTPException(status_code=403, detail="remote access is disabled")

    def refuse(reason: str, device_id: object = None) -> HTTPException:
        shown = device_id[:64] if isinstance(device_id, str) else "-"
        logger.warning("remote request refused: %s (device_id=%r)", reason, shown)
        return HTTPException(status_code=401, detail=AUTH_FAILED)

    async def signed(request: Request) -> tuple[Session, DeviceContext]:
        """A session's signed request, or a 401. Order: shape, session, freshness and nonce, signature, device trust."""
        now = access.clock()
        headers = request.headers
        session_id = headers.get(signing.SESSION_HEADER)
        signature = headers.get(signing.SIGNATURE_HEADER)
        timestamp = headers.get(signing.TIMESTAMP_HEADER)
        nonce = headers.get(signing.NONCE_HEADER)
        if not session_id or len(session_id) > 64 or not signature or not signing.SIGNATURE_PATTERN.fullmatch(signature):
            raise refuse("malformed session headers")
        session = access.sessions.get(session_id, now)
        if session is None:
            raise refuse("unknown or expired session")
        body = await request.body()
        path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        if not signing.signature_matches(session.key, signature, request.method, path, timestamp or "", nonce or "", body):
            raise refuse("bad signature", session.device_id)
        if not access.fresh(timestamp, nonce, "session:" + session.session_id, now):
            raise refuse("stale timestamp or reused nonce (replay)", session.device_id)
        try:
            context = access.gateway.session_context(session.device_id)
        except DeviceAuthenticationError:
            access.sessions.end_device(session.device_id)  # the device is no longer trusted: none of its sessions survive
            raise refuse("device no longer trusted", session.device_id)
        access.sessions.touch(session, now)
        return session, context

    @router.post("/session", dependencies=[Depends(require_enabled)])
    def create_session(request: Request) -> dict:
        now = access.clock()
        headers = request.headers
        device_id = headers.get(signing.DEVICE_ID_HEADER)
        credential = headers.get(signing.DEVICE_TOKEN_HEADER)
        if not isinstance(device_id, str) or not device_id or len(device_id) > 64:
            raise refuse("malformed device id")
        if not access.fresh(headers.get(signing.TIMESTAMP_HEADER), headers.get(signing.NONCE_HEADER), "device:" + device_id, now):
            raise refuse("stale timestamp or reused nonce (replay)", device_id)
        try:
            context = access.gateway.authenticate(device_id, credential)
            access.gateway.check_in(device_id, credential)
        except DeviceAuthenticationError:
            raise refuse("device authentication failed", device_id)
        session = access.sessions.create(context.device_id, now)
        logger.info("remote session started for %s", context.device_id)
        return {
            "session_id": session.session_id,
            "session_key": base64.urlsafe_b64encode(session.key).decode("ascii"),  # returned ONCE; never stored on disk
            "expires_at": session.expires_at.isoformat(),
            "device": {"device_id": context.device_id, "name": context.name},
        }

    @router.get("/status", dependencies=[Depends(require_enabled)])
    def status(auth: tuple[Session, DeviceContext] = Depends(signed)) -> dict:
        session, context = auth
        return {"ok": True, "device": {"device_id": context.device_id, "name": context.name},
                "session_expires_at": session.expires_at.isoformat(), "server_time": access.clock().isoformat()}

    @router.post("/chat", dependencies=[Depends(require_enabled)])
    def remote_chat(body: RemoteChatRequest, request: Request, auth: tuple[Session, DeviceContext] = Depends(signed)) -> dict:
        session, context = auth
        request_id = request.headers.get(signing.REQUEST_ID_HEADER)
        if request_id is not None and not signing.REQUEST_ID_PATTERN.fullmatch(request_id):
            raise HTTPException(status_code=400, detail="bad request id")
        now = access.clock()
        if request_id:
            earlier = access.replies.get(context.device_id, request_id, now)
            if earlier is not None:
                return {**earlier, "repeated": True}  # a retry after a lost connection: the same reply, done once
        reply = chat(body.message, body.conversation_id, context)
        if request_id:
            access.replies.put(context.device_id, request_id, reply, now)
        return reply

    @router.delete("/session", dependencies=[Depends(require_enabled)])
    def end_session(auth: tuple[Session, DeviceContext] = Depends(signed)) -> dict:
        session, _ = auth
        access.sessions.end(session.session_id)
        return {"ended": True}

    return router
