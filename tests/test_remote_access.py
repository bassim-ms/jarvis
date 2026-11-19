"""
Secure remote access (app/remote/): off by default; a configured private host reaches /remote/ only; sessions only for devices
registered AND trusted with the local admin command; signed requests checked for replay, tampering, expiry and revocation; the
same permission levels as local chat. Everything here runs in-process against isolated temporary databases with a fake clock -
nothing listens on a network interface and the real machine is never exposed.
"""

import base64
import json
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.devices.gateway import DeviceGateway
from app.devices.registry import DeviceRegistry
from app.devices.store import SQLiteDeviceStore
from app.host_guard import LoopbackHostMiddleware
from app.orchestration.sqlite_store import SQLiteJobStore
from app.remote import signing
from app.remote.api import AUTH_FAILED, RemoteAccess, build_remote_router, parse_remote_hosts
from app.remote.client import RemoteClient, RemoteError

HOST = "jarvis-pc.tail1234.ts.net"
T0 = datetime(2026, 11, 19, 9, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


class World:
    def __init__(self, tmp_path, enabled=True):
        self.clock = Clock()
        self.registry = DeviceRegistry(SQLiteDeviceStore.for_job_store(SQLiteJobStore(str(tmp_path / "orch.db"))), clock=self.clock)
        self.gateway = DeviceGateway(self.registry)
        self.enabled = enabled
        self.access = RemoteAccess(gateway=self.gateway, registry=self.registry, enabled=lambda: self.enabled,
                                   hosts=lambda: frozenset({HOST}) if self.enabled else frozenset(),
                                   session_ttl=timedelta(hours=12), max_sessions_per_device=3, clock=self.clock)
        self.chats = []
        app = FastAPI()
        app.add_middleware(LoopbackHostMiddleware, remote_hosts=self.access.hosts)
        app.include_router(build_remote_router(self.access, self.chat))

        @app.get("/dashboard")
        def dashboard():
            return {"local": True}

        @app.post("/chat")
        def local_chat():
            return {"local": True}

        self.remote = TestClient(app, base_url=f"https://{HOST}")
        self.local = TestClient(app, base_url="http://127.0.0.1:8000")

    def chat(self, message, conversation_id, device):
        self.chats.append((message, device.device_id))
        return {"reply": f"echo: {message}", "conversation_id": conversation_id or "c1", "device": device.name}

    def device(self, name="Laptop", trust=True):
        device, credential = self.registry.register(name, "computer")
        if trust:
            device = self.registry.trust(device.device_id)
        return device, credential

    def session(self, device, credential, nonce=None):
        headers = {"X-Jarvis-Device-Id": device.device_id, "X-Jarvis-Device-Token": credential,
                   "X-Jarvis-Timestamp": str(int(self.clock.now.timestamp())), "X-Jarvis-Nonce": nonce or signing.new_nonce()}
        return self.remote.post("/remote/v1/session", headers=headers)

    def signed(self, session, method, path, payload=None, *, nonce=None, at=None, key=None, extra=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        key = key or base64.urlsafe_b64decode(session["session_key"])
        stamp = int((at or self.clock.now).timestamp())
        headers = signing.signed_headers(key, session["session_id"], method, path, body, stamp, nonce)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        headers.update(extra or {})
        return self.remote.request(method, path, content=body, headers=headers)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def connected(world):
    device, credential = world.device()
    response = world.session(device, credential)
    assert response.status_code == 200, response.text
    return device, credential, response.json()


# --- off by default, and only /remote/ through the private host -------------------------------------------------------------


def test_remote_access_is_off_by_default_in_settings():
    from app.config import Settings

    fields = Settings.model_fields
    assert fields["remote_access_enabled"].default is False and fields["remote_access_hosts"].default == ""


def test_while_off_every_remote_route_refuses_and_the_private_host_reaches_nothing(tmp_path):
    w = World(tmp_path, enabled=False)
    device, credential = w.device()

    assert w.local.post("/remote/v1/session").status_code == 403
    for path in ("/remote/v1/session", "/dashboard", "/chat"):
        assert w.remote.post(path).status_code == 403
    assert w.local.post("/chat").status_code == 200 and w.local.get("/dashboard").status_code == 200  # local still works


@pytest.mark.parametrize("path", ["/dashboard", "/chat", "/providers/status", "/api/remote", "/docs", "/openapi.json", "/dev/status",
                                  "/remote", "/remotex/v1/session"])
def test_the_private_host_reaches_only_remote_routes(world, path):
    assert world.remote.get(path).status_code == 403
    assert world.remote.post(path).status_code == 403


@pytest.mark.parametrize("host", ["evil.example", f"{HOST}.evil.example", "tail1234.ts.net", "jarvis-pc", "192.168.1.5"])
def test_any_other_host_is_refused_even_on_remote_routes(world, host):
    assert TestClient(world.remote.app, base_url=f"https://{host}").post("/remote/v1/session").status_code == 403


def test_the_private_host_may_carry_a_port(world):
    device, credential = world.device()
    client = TestClient(world.remote.app, base_url=f"https://{HOST}:443")
    headers = {"X-Jarvis-Device-Id": device.device_id, "X-Jarvis-Device-Token": credential,
               "X-Jarvis-Timestamp": str(int(T0.timestamp())), "X-Jarvis-Nonce": signing.new_nonce()}

    assert client.post("/remote/v1/session", headers=headers).status_code == 200


@pytest.mark.parametrize("text, expected", [
    ("jarvis-pc.tail1234.ts.net", {"jarvis-pc.tail1234.ts.net"}),
    (" Jarvis-PC.Tail1234.ts.net. , other.example ", {"jarvis-pc.tail1234.ts.net", "other.example"}),
    ("localhost, 127.0.0.1, *.ts.net, host:8000, jarvis-pc, a.localhost, http://x.example", set()),
    ("", set()),
])
def test_only_plain_dns_names_can_be_configured_as_remote_hosts(text, expected):
    assert parse_remote_hosts(text) == frozenset(expected)


# --- sessions: only for trusted devices, never replayable ------------------------------------------------------------------------


def test_a_trusted_device_opens_a_session(world):
    device, credential = world.device()

    body = world.session(device, credential).json()

    assert body["session_id"].startswith("rs_") and len(base64.urlsafe_b64decode(body["session_key"])) == 32
    assert body["device"] == {"device_id": device.device_id, "name": "Laptop"}
    assert body["expires_at"] == (T0 + timedelta(hours=12)).isoformat()


@pytest.mark.parametrize("case", ["pending", "revoked", "wrong credential", "unknown device", "no credential", "long id"])
def test_no_session_for_an_untrusted_or_unproven_device_and_every_refusal_looks_the_same(world, case):
    device, credential = world.device(trust=case != "pending")
    if case == "revoked":
        world.registry.revoke(device.device_id)
    if case == "wrong credential":
        credential = credential[:-2] + "xx"
    headers = {"X-Jarvis-Device-Id": device.device_id, "X-Jarvis-Device-Token": credential,
               "X-Jarvis-Timestamp": str(int(T0.timestamp())), "X-Jarvis-Nonce": signing.new_nonce()}
    if case == "unknown device":
        headers["X-Jarvis-Device-Id"] = "dev_0000000000000000"
    if case == "no credential":
        del headers["X-Jarvis-Device-Token"]
    if case == "long id":
        headers["X-Jarvis-Device-Id"] = "d" * 300

    response = world.remote.post("/remote/v1/session", headers=headers)

    assert response.status_code == 401 and response.json() == {"detail": AUTH_FAILED}
    assert world.access.sessions.active(T0) == []


def test_a_captured_session_request_cannot_be_replayed(world):
    device, credential = world.device()
    nonce = signing.new_nonce()

    assert world.session(device, credential, nonce=nonce).status_code == 200
    assert world.session(device, credential, nonce=nonce).status_code == 401


@pytest.mark.parametrize("skew", [timedelta(minutes=-2), timedelta(minutes=2)])
def test_a_stale_or_future_session_request_is_refused(world, skew):
    device, credential = world.device()
    headers = {"X-Jarvis-Device-Id": device.device_id, "X-Jarvis-Device-Token": credential,
               "X-Jarvis-Timestamp": str(int((T0 + skew).timestamp())), "X-Jarvis-Nonce": signing.new_nonce()}

    assert world.remote.post("/remote/v1/session", headers=headers).status_code == 401


@pytest.mark.parametrize("nonce", ["short", "x" * 200, "has spaces in it!!", ""])
def test_a_malformed_nonce_is_refused(world, nonce):
    device, credential = world.device()

    assert world.session(device, credential, nonce=nonce or None).status_code in (200, 401)
    if nonce:
        assert world.session(device, credential, nonce=nonce).status_code == 401


# --- signed requests ---------------------------------------------------------------------------------------------------------------


def test_a_signed_request_is_answered_as_the_authenticated_device(world, connected):
    device, _, session = connected

    status = world.signed(session, "GET", "/remote/v1/status").json()
    chat = world.signed(session, "POST", "/remote/v1/chat", {"message": "hello"}).json()

    assert status["ok"] is True and status["device"]["device_id"] == device.device_id
    assert chat["reply"] == "echo: hello" and world.chats == [("hello", device.device_id)]


def test_a_replayed_signed_request_is_refused_and_does_nothing(world, connected):
    _, _, session = connected
    nonce = signing.new_nonce()

    assert world.signed(session, "POST", "/remote/v1/chat", {"message": "pay the bill"}, nonce=nonce).status_code == 200
    replay = world.signed(session, "POST", "/remote/v1/chat", {"message": "pay the bill"}, nonce=nonce)

    assert replay.status_code == 401 and replay.json() == {"detail": AUTH_FAILED}
    assert len(world.chats) == 1


def test_a_tampered_request_is_refused(world, connected):
    _, _, session = connected
    key = base64.urlsafe_b64decode(session["session_key"])
    body = json.dumps({"message": "hello"}).encode()
    headers = signing.signed_headers(key, session["session_id"], "POST", "/remote/v1/chat", body, int(T0.timestamp()))
    headers["Content-Type"] = "application/json"

    changed = world.remote.post("/remote/v1/chat", content=json.dumps({"message": "delete everything"}).encode(), headers=headers)

    assert changed.status_code == 401 and world.chats == []


def test_a_signature_for_another_path_or_method_is_refused(world, connected):
    _, _, session = connected
    key = base64.urlsafe_b64decode(session["session_key"])
    headers = signing.signed_headers(key, session["session_id"], "GET", "/remote/v1/status", b"", int(T0.timestamp()))

    assert world.remote.delete("/remote/v1/session", headers=headers).status_code == 401
    assert len(world.access.sessions.active(T0)) == 1


def test_someone_with_only_the_session_id_cannot_use_it(world, connected):
    _, _, session = connected

    assert world.signed(session, "GET", "/remote/v1/status", key=b"\x00" * 32).status_code == 401


@pytest.mark.parametrize("missing", ["X-Jarvis-Session", "X-Jarvis-Timestamp", "X-Jarvis-Nonce", "X-Jarvis-Signature"])
def test_a_request_missing_any_signing_header_is_refused(world, connected, missing):
    _, _, session = connected
    key = base64.urlsafe_b64decode(session["session_key"])
    headers = signing.signed_headers(key, session["session_id"], "GET", "/remote/v1/status", b"", int(T0.timestamp()))
    del headers[missing]

    assert world.remote.get("/remote/v1/status", headers=headers).status_code == 401


@pytest.mark.parametrize("skew", [timedelta(seconds=-61), timedelta(seconds=61), timedelta(days=1)])
def test_a_stale_or_future_signed_request_is_refused(world, connected, skew):
    _, _, session = connected

    assert world.signed(session, "GET", "/remote/v1/status", at=T0 + skew).status_code == 401


@pytest.mark.parametrize("body", [b"not json", json.dumps({"message": ""}).encode(), json.dumps({"message": "x" * 9000}).encode(),
                                  json.dumps({"text": "hi"}).encode()])
def test_a_malformed_body_is_rejected_without_reaching_chat(world, connected, body):
    _, _, session = connected
    key = base64.urlsafe_b64decode(session["session_key"])
    headers = signing.signed_headers(key, session["session_id"], "POST", "/remote/v1/chat", body, int(T0.timestamp()))
    headers["Content-Type"] = "application/json"

    response = world.remote.post("/remote/v1/chat", content=body, headers=headers)

    assert response.status_code == 422 and world.chats == []


# --- expiry, disconnection, revocation ------------------------------------------------------------------------------------------


def test_a_session_expires(world, connected):
    _, _, session = connected
    world.clock.now = T0 + timedelta(hours=12, seconds=1)

    assert world.signed(session, "GET", "/remote/v1/status").status_code == 401


def test_disconnecting_ends_the_session(world, connected):
    _, _, session = connected

    assert world.signed(session, "DELETE", "/remote/v1/session").json() == {"ended": True}
    assert world.signed(session, "GET", "/remote/v1/status").status_code == 401


def test_revoking_on_the_dashboard_cuts_a_device_off_at_once(world, connected):
    device, credential, session = connected

    result = world.access.revoke_device(device.device_id)

    assert result["trust"] == "revoked" and result["sessions_ended"] == 1
    assert world.signed(session, "GET", "/remote/v1/status").status_code == 401
    assert world.session(device, credential).status_code == 401  # and can never open another


def test_revoking_with_the_admin_command_also_stops_live_sessions(world, connected):
    device, _, session = connected

    world.registry.revoke(device.device_id)  # what `python -m app.devices.admin revoke` does

    assert world.signed(session, "GET", "/remote/v1/status").status_code == 401
    assert world.access.sessions.active(T0) == []


def test_ending_one_session_from_the_dashboard(world, connected):
    _, _, session = connected

    assert world.access.end_session(session["session_id"]) is True
    assert world.signed(session, "GET", "/remote/v1/status").status_code == 401
    assert world.access.end_session(session["session_id"]) is False


# --- concurrent sessions and reconnection -------------------------------------------------------------------------------------------


def test_a_device_may_hold_several_sessions_up_to_the_limit(world):
    device, credential = world.device()
    sessions = [world.session(device, credential).json() for _ in range(4)]  # limit 3: the oldest is ended

    assert world.signed(sessions[0], "GET", "/remote/v1/status").status_code == 401
    assert all(world.signed(s, "GET", "/remote/v1/status").status_code == 200 for s in sessions[1:])


def test_two_devices_have_separate_sessions(world):
    a, a_credential = world.device("Laptop")
    b, b_credential = world.device("Tablet")
    sa, sb = world.session(a, a_credential).json(), world.session(b, b_credential).json()

    world.access.revoke_device(a.device_id)

    assert world.signed(sa, "GET", "/remote/v1/status").status_code == 401
    assert world.signed(sb, "GET", "/remote/v1/status").json()["device"]["name"] == "Tablet"


def test_concurrent_signed_requests_all_succeed_and_none_is_mistaken_for_a_replay(world, connected):
    _, _, session = connected
    results = []

    def worker():
        for _ in range(15):
            results.append(world.signed(session, "POST", "/remote/v1/chat", {"message": "ping"}).status_code)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(200) == 90 and len(world.chats) == 90
    assert world.access.sessions.active(T0)[0].requests == 90


def test_a_retried_request_after_a_lost_connection_is_answered_once(world, connected):
    device, credential, session = connected
    request_id = {"X-Jarvis-Request-Id": "req-12345678"}

    first = world.signed(session, "POST", "/remote/v1/chat", {"message": "add a task"}, extra=request_id).json()
    again = world.signed(session, "POST", "/remote/v1/chat", {"message": "add a task"}, extra=request_id).json()
    fresh = world.session(device, credential).json()  # the device reconnected with a new session
    later = world.signed(fresh, "POST", "/remote/v1/chat", {"message": "add a task"}, extra=request_id).json()

    assert len(world.chats) == 1 and first["reply"] == again["reply"] == later["reply"]
    assert again["repeated"] is True and later["repeated"] is True


def test_a_malformed_request_id_is_refused(world, connected):
    _, _, session = connected

    response = world.signed(session, "POST", "/remote/v1/chat", {"message": "x"}, extra={"X-Jarvis-Request-Id": "../bad id"})

    assert response.status_code == 400 and world.chats == []


# --- the client --------------------------------------------------------------------------------------------------------------------


class InProcessTransport:
    """The real RemoteClient talking to the in-process app through TestClient; can drop connections on demand."""

    def __init__(self, world, drop=0):
        self.world, self.drop, self.sent = world, drop, []

    def __call__(self, method, url, headers, body):
        self.sent.append((method, url))
        if self.drop > 0:
            self.drop -= 1
            raise ConnectionResetError("connection lost")
        path = url.split(HOST, 1)[1]
        response = self.world.remote.request(method, path, content=body, headers=headers)
        return response.status_code, response.json()


def client_for(world, device, credential, transport):
    return RemoteClient(f"https://{HOST}", device.device_id, credential, transport=transport,
                        clock=lambda: world.clock.now.timestamp(), sleep=lambda s: None)


def test_the_client_connects_signs_and_chats(world):
    device, credential = world.device()
    client = client_for(world, device, credential, InProcessTransport(world))

    assert client.chat("hello")["reply"] == "echo: hello"
    assert client.status()["device"]["name"] == "Laptop"
    client.disconnect()
    assert world.access.sessions.active(T0) == []


def test_the_client_reconnects_when_its_session_expired(world):
    device, credential = world.device()
    client = client_for(world, device, credential, InProcessTransport(world))
    client.connect()
    first_session = client.session_id
    world.clock.now = T0 + timedelta(hours=13)

    assert client.chat("still there?")["reply"] == "echo: still there?" and client.session_id != first_session


def test_the_client_retries_a_lost_connection_without_doing_it_twice(world):
    device, credential = world.device()
    transport = InProcessTransport(world)
    client = client_for(world, device, credential, transport)
    client.connect()
    transport.drop = 1  # the next send is lost

    assert client.chat("note this")["reply"] == "echo: note this" and len(world.chats) == 1


def test_the_client_gives_up_after_its_retries(world):
    device, credential = world.device()
    transport = InProcessTransport(world)
    client = client_for(world, device, credential, transport)
    client.connect()
    transport.drop = 10

    with pytest.raises(OSError):
        client.chat("anyone?")


def test_the_client_refuses_plain_http_to_a_remote_address_and_reports_a_revoked_device(world):
    with pytest.raises(ValueError):
        RemoteClient(f"http://{HOST}", "dev_x", "c")
    device, credential = world.device()
    world.registry.revoke(device.device_id)

    with pytest.raises(RemoteError) as refused:
        client_for(world, device, credential, InProcessTransport(world)).status()
    assert refused.value.status == 401


# --- the real application: permissions, dashboard, local use with remote off -----------------------------------------------------


@pytest.fixture
def live(monkeypatch):
    """The REAL app with remote access switched on for HOST, its own isolated device database (the test suite's database
    redirect), a stub AI provider and four tools - one per permission level - that record whether they ran."""
    from app import main
    from app.tools.base import PermissionLevel, Tool, ToolResult

    ran = []

    def make(level):
        class Probe(Tool):
            name = f"probe_{level.value}"
            description = f"records that it ran ({level.value})"
            permission = level

            @property
            def parameters(self):
                return []

            def execute(self):
                ran.append(self.name)
                return ToolResult.success(message=f"{self.name} ran")
        return Probe()

    for level in PermissionLevel:
        tool = make(level)
        monkeypatch.setitem(main.core.tools._tools, tool.name, tool)
    monkeypatch.setattr(main.settings, "remote_access_enabled", True)
    monkeypatch.setattr(main.settings, "remote_access_hosts", HOST)
    monkeypatch.setattr(main.core.provider, "generate_reply", lambda message, requirements=None: "stub reply")
    device, credential = main._device_registry.register(f"Remote test {len(ran)}-{id(ran)}", "computer")
    main._device_registry.trust(device.device_id)
    remote = TestClient(main.app, base_url=f"https://{HOST}")
    yield main, remote, device, credential, ran
    try:
        main._device_registry.revoke(device.device_id)
    except Exception:
        pass


def live_client(live):
    main, remote, device, credential, _ = live

    def transport(method, url, headers, body):
        response = remote.request(method, url.split(HOST, 1)[1], content=body, headers=headers)
        return response.status_code, response.json()

    return RemoteClient(f"https://{HOST}", device.device_id, credential, transport=transport)


@pytest.mark.parametrize("level, executes", [("read", True), ("safe_action", True), ("confirm", False), ("dangerous", False)])
def test_every_permission_level_applies_to_remote_requests_exactly_as_locally(live, level, executes):
    _, _, _, _, ran = live

    reply = live_client(live).chat(f"use tool probe_{level}")["reply"]

    assert (f"probe_{level}" in ran) is executes, reply
    if not executes:
        assert "confirm" in reply.lower()


def test_the_real_app_keeps_everything_else_loopback_only_for_the_remote_host(live):
    _, remote, _, _, _ = live

    for path in ("/dashboard", "/api/remote", "/api/memory", "/providers/status", "/dev/status", "/docs", "/openapi.json", "/health"):
        assert remote.get(path).status_code == 403, path
    assert remote.post("/chat", json={"message": "hi"}).status_code == 403
    assert remote.post("/device/chat", json={"message": "hi"}).status_code == 403


def test_the_dashboard_shows_the_remote_device_and_can_revoke_it(live):
    main, _, device, _, _ = live
    client = live_client(live)
    client.connect()
    local = TestClient(main.app, base_url="http://127.0.0.1:8000")
    dashboard = {"X-Jarvis-Dashboard": "1"}

    overview = local.get("/api/remote").json()
    mine = next(d for d in overview["devices"] if d["device_id"] == device.device_id)
    assert overview["enabled"] is True and overview["hosts"] == [HOST] and len(mine["sessions"]) == 1 and mine["trust"] == "trusted"

    assert local.post(f"/api/remote/devices/{device.device_id}/revoke", json={}).status_code == 403  # dashboard page only
    assert local.post(f"/api/remote/devices/{device.device_id}/revoke", json={},
                      headers={**dashboard, "Origin": "https://evil.example"}).status_code == 403
    revoked = local.post(f"/api/remote/devices/{device.device_id}/revoke", json={}, headers=dashboard)
    assert revoked.status_code == 200 and revoked.json()["sessions_ended"] == 1
    with pytest.raises(RemoteError):
        client.status()


def test_with_remote_access_off_the_real_app_works_locally_and_refuses_remote(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "remote_access_enabled", False)
    monkeypatch.setattr(main.settings, "remote_access_hosts", HOST)  # configured but OFF: still nothing remote
    monkeypatch.setattr(main.core.provider, "generate_reply", lambda message, requirements=None: "stub reply")
    local = TestClient(main.app, base_url="http://127.0.0.1:8000")

    assert local.get("/dashboard").status_code == 200 and local.get("/health").status_code == 200
    assert local.post("/chat", json={"message": "hello"}).status_code == 200
    assert local.post("/remote/v1/session").status_code == 403
    assert TestClient(main.app, base_url=f"https://{HOST}").post("/remote/v1/session").status_code == 403
    assert local.get("/api/remote").json()["enabled"] is False


def test_the_app_still_never_registers_or_trusts_a_device():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "app"
    for path in list((root / "remote").glob("*.py")) + [root / "dashboard" / "api.py", root / "main.py"]:
        text = path.read_text(encoding="utf-8")
        assert ".trust(" not in text and "registry.register(" not in text, path.name
