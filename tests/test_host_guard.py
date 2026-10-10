"""
Loopback-only Host check (app/host_guard.py): a request whose `Host` header is not localhost / a 127.x.x.x address / [::1] (each with an optional port) is
answered 403 before it reaches Jarvis - on EVERY route - which closes the DNS-rebinding route to the anonymous /chat. It is an additional safeguard only;
binding to the loopback interface and authentication remain what actually protect the server.

Real: the pure function, the ASGI middleware, and the production FastAPI app in app/main.py. Faked: nothing about the check itself; the production core is
swapped for a tripwire wherever a refused request must prove it never reached Jarvis.
"""

import asyncio
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main
from app.host_guard import LoopbackHostMiddleware, is_loopback_host

ACCEPTED = [
    "localhost", "LOCALHOST", "LocalHost", "localhost:8000", "localhost:1", "localhost:65535",
    "127.0.0.1", "127.0.0.1:8000", "127.0.0.1:80", "127.1.2.3", "127.255.255.254:9",
    "[::1]", "[::1]:8000", "[::1]:65535", "[0:0:0:0:0:0:0:1]", "[0:0:0:0:0:0:0:1]:8000", "[0000:0000:0000:0000:0000:0000:0000:0001]",
]

REFUSED = [
    # not this machine
    "evil.example", "evil.example:8000", "example.com", "192.168.1.5", "10.0.0.1:8000", "0.0.0.0", "0.0.0.0:8000", "8.8.8.8", "[2001:db8::1]", "[::]", "[::2]",
    # look-alikes and suffix/prefix tricks
    "localhost.evil.example", "evil-localhost", "127.0.0.1.evil.example", "localhost.", "localhost..", "xlocalhost", "localhostx", "127.0.0.1.", "128.0.0.1",
    # IPv4 spellings that are not four plain decimal fields
    "127.1", "127.0.1", "0x7f.0.0.1", "0177.0.0.1", "2130706433", "127.000.000.001", "127.0.0.01", "127.0.0.1.1", "127.0.0",
    # IPv6 that is not exactly ::1, or not bracketed
    "::1", "::1:8000", "[::1", "::1]", "[::1]]", "[::ffff:127.0.0.1]", "[::ffff:7f00:1]", "[::1%25lo]", "[::1%lo]", "[]", "[localhost]", "[127.0.0.1]", "[::1]x", "[::1]:", "[::1]:0",
    "[::1]:65536", "[::1]:-1", "[::1] ", "[::1]/", "[fe80::1]",
    # ports
    "localhost:", "127.0.0.1:", "127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:99999", "127.0.0.1:123456", "127.0.0.1:-1", "127.0.0.1:+80", "127.0.0.1:8o", "127.0.0.1:8000:9",
    "127.0.0.1: 80", "127.0.0.1:٣٤", "127.0.0.1:８０", ":8000", ":",
    # characters that have no business in a Host
    "", " ", " localhost", "localhost ", "local host", "user@localhost", "user:pw@localhost", "localhost@evil.example", "localhost/", "localhost/x", "localhost\\x", "localhost?x",
    "localhost#x", "localhost%2e", "localhost%00", "localhost\n", "localhost\r\nX-Injected: 1", "localhost\t", "localhost\x00", "locаlhost", "löcalhost", "localhost\x7f",
    "a" * 300, "localhost" + ":8000" * 3,
]


@pytest.mark.parametrize("host", ACCEPTED)
def test_a_loopback_host_with_or_without_a_port_is_accepted(host):
    assert is_loopback_host(host) is True


@pytest.mark.parametrize("host", REFUSED, ids=[repr(h)[:34] for h in REFUSED])
def test_anything_else_is_refused(host):
    assert is_loopback_host(host) is False


@pytest.mark.parametrize("value", [None, 5, 5.5, b"localhost", ["localhost"], {"host": "localhost"}, True])
def test_a_non_string_is_never_a_host(value):
    assert is_loopback_host(value) is False


def test_localhost_is_matched_exactly_not_by_resolution():
    """Names are never resolved: only the literal `localhost` is trusted, never a name that merely RESOLVES to 127.0.0.1 (which is the rebinding trick)."""
    assert is_loopback_host("localhost") and not is_loopback_host("rebind.attacker.example") and not is_loopback_host("localtest.me")


# ===================================================== the middleware on a throwaway app =====================================================


def tiny_app():
    app = FastAPI()
    calls = []

    @app.get("/ping")
    def ping():
        calls.append("ping")
        return {"ok": True}

    @app.post("/echo")
    def echo(body: dict):
        calls.append("echo")
        return body

    app.add_middleware(LoopbackHostMiddleware)
    return app, calls


@pytest.mark.parametrize("host", ["localhost", "localhost:8000", "127.0.0.1", "127.0.0.1:8000", "[::1]", "[::1]:8000", "LOCALHOST:8000"])
def test_a_loopback_host_reaches_the_app(host):
    app, calls = tiny_app()

    response = TestClient(app, base_url="http://localhost").get("/ping", headers={"Host": host})

    assert response.status_code == 200 and calls == ["ping"]


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "localhost.evil.example", "127.0.0.1.evil.example", "192.168.1.5", "0.0.0.0:8000", "[::ffff:127.0.0.1]:8000", "testserver"])
def test_any_other_host_is_refused_with_403_and_the_app_is_never_called(host):
    app, calls = tiny_app()
    client = TestClient(app, base_url="http://localhost")

    get = client.get("/ping", headers={"Host": host})
    post = client.post("/echo", json={"message": "hello"}, headers={"Host": host})
    preflight = client.options("/echo", headers={"Host": host, "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})

    assert get.status_code == post.status_code == preflight.status_code == 403
    assert "loopback" in get.text and calls == []


def test_the_refusal_is_plain_text_and_says_nothing_about_the_app():
    app, _ = tiny_app()

    response = TestClient(app, base_url="http://localhost").get("/ping", headers={"Host": "evil.example"})

    assert response.headers["content-type"].startswith("text/plain") and int(response.headers["content-length"]) == len(response.content)
    assert "evil.example" not in response.text  # the attacker's own text is not echoed back


def run_asgi(middleware, scope):
    """Drive the middleware directly with a hand-built scope; returns (messages sent, whether the inner app ran)."""
    sent, ran = [], []

    async def inner(scope, receive, send):
        ran.append(scope["type"])

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(LoopbackHostMiddleware(inner)(scope, receive, send))
    return sent, ran


def http_scope(*hosts):
    return {"type": "http", "method": "GET", "path": "/x", "headers": [(b"host", h.encode("latin-1")) for h in hosts] + [(b"accept", b"*/*")]}


def test_a_request_with_no_host_header_is_refused():
    sent, ran = run_asgi(LoopbackHostMiddleware, {"type": "http", "method": "GET", "path": "/x", "headers": [(b"accept", b"*/*")]})

    assert sent[0]["status"] == 403 and ran == []


def test_a_request_with_two_host_headers_is_refused_even_if_both_are_loopback():
    sent, ran = run_asgi(LoopbackHostMiddleware, http_scope("localhost", "localhost"))

    assert sent[0]["status"] == 403 and ran == []  # ambiguous: which one would the app (or a proxy in front of it) believe?


def test_header_names_are_matched_case_insensitively_and_the_value_exactly():
    scope = {"type": "http", "method": "GET", "path": "/x", "headers": [(b"HOST", b"localhost:8000")]}
    sent, ran = run_asgi(LoopbackHostMiddleware, scope)
    assert ran == ["http"] and sent == []

    for hostile in ("localhost:8000 ", " localhost:8000", "localhost:8000\r\nX: y", "localhost\x00"):
        sent, ran = run_asgi(LoopbackHostMiddleware, http_scope(hostile))
        assert sent[0]["status"] == 403 and ran == [], repr(hostile)


def test_a_forwarded_host_header_cannot_stand_in_for_the_real_one():
    scope = {"type": "http", "method": "GET", "path": "/x", "headers": [(b"host", b"evil.example"), (b"x-forwarded-host", b"localhost")]}

    sent, ran = run_asgi(LoopbackHostMiddleware, scope)

    assert sent[0]["status"] == 403 and ran == []


def test_a_websocket_is_checked_the_same_way():
    bad = {"type": "websocket", "path": "/ws", "headers": [(b"host", b"evil.example")]}
    good = {"type": "websocket", "path": "/ws", "headers": [(b"host", b"[::1]:8000")]}

    sent, ran = run_asgi(LoopbackHostMiddleware, bad)
    assert sent == [{"type": "websocket.close", "code": 1008}] and ran == []
    sent, ran = run_asgi(LoopbackHostMiddleware, good)
    assert sent == [] and ran == ["websocket"]


def test_other_scope_types_such_as_lifespan_pass_through_untouched():
    sent, ran = run_asgi(LoopbackHostMiddleware, {"type": "lifespan"})

    assert sent == [] and ran == ["lifespan"]


def test_a_refusal_is_logged_with_a_bounded_host_and_no_body(caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis.host_guard")
    app, _ = tiny_app()

    TestClient(app, base_url="http://localhost").post("/echo", json={"secret": "BODY-SECRET-123"}, headers={"Host": "e" * 500 + ".example"})

    (record,) = caplog.records
    assert record.levelno == logging.WARNING and len(record.getMessage()) < 200 and "BODY-SECRET-123" not in record.getMessage()


# ===================================================== the production app =====================================================================

ROUTES = [
    ("GET", "/health", None), ("GET", "/dev/ping", None), ("GET", "/dev/status", None),
    ("POST", "/chat", {"message": "hello"}), ("POST", "/device/chat", {"message": "hello"}), ("POST", "/device/check-in", None),
]
HOSTILE = ["evil.example", "evil.example:8000", "localhost.evil.example", "192.168.1.5:8000", "0.0.0.0:8000", "testserver"]


def test_the_production_app_has_the_guard_and_not_starlettes_port_stripping_one():
    names = {m.cls.__name__ for m in main.app.user_middleware}

    assert "LoopbackHostMiddleware" in names
    assert "TrustedHostMiddleware" not in names  # it splits on ':' and so mishandles a bracketed IPv6 Host; ours is exact


@pytest.fixture
def tripwire(monkeypatch):
    """Jarvis itself: any request that reaches it fails the test."""

    class Tripwire:
        def handle_chat_with_job_id(self, *args, **kwargs):
            raise AssertionError("a refused request reached Jarvis")

    monkeypatch.setattr(main, "core", Tripwire())


@pytest.mark.parametrize("host", HOSTILE)
@pytest.mark.parametrize("method, path, body", ROUTES, ids=[f"{m} {p}" for m, p, _ in ROUTES])
def test_every_production_route_refuses_a_foreign_host_and_never_reaches_jarvis(tripwire, method, path, body, host):
    client = TestClient(main.app, base_url="http://localhost")

    response = client.request(method, path, json=body, headers={"Host": host})

    assert response.status_code == 403


@pytest.mark.parametrize("host", ["localhost", "localhost:8000", "127.0.0.1:8000", "[::1]:8000"])
def test_production_routes_still_answer_a_loopback_host(host):
    client = TestClient(main.app, base_url="http://localhost")
    headers = {"Host": host}

    assert client.get("/health", headers=headers).status_code == 200
    assert client.get("/dev/ping", headers=headers).json() == "pong"
    assert client.post("/chat", json={}, headers=headers).status_code == 422  # reached the app (bad body), not the guard
    assert client.post("/device/chat", json={"message": "x"}, headers=headers).status_code == 401  # reached device authentication
    assert client.post("/device/check-in", headers=headers).status_code == 401


def test_the_guard_does_not_replace_authentication():
    """A loopback Host proves nothing about WHO is calling: /device/chat still needs its credential, and /chat is still the anonymous interface."""
    client = TestClient(main.app, base_url="http://127.0.0.1")

    assert client.post("/device/chat", json={"message": "x"}).status_code == 401


def test_the_guard_covers_every_route_the_application_actually_has_including_docs_and_unknown_paths(monkeypatch):
    """Routes are read from the live application, so a route added later is covered without anyone remembering to list it here."""

    class Tripwire:
        def handle_chat_with_job_id(self, *args, **kwargs):
            raise AssertionError("a refused request reached Jarvis")

    client = TestClient(main.app, base_url="http://localhost")
    routes = {}
    for route in main.app.routes:
        methods = sorted((getattr(route, "methods", None) or {"GET"}) - {"HEAD", "OPTIONS"}) or ["GET"]
        routes[route.path] = methods[0]
    assert {"/chat", "/device/chat", "/device/check-in", "/health", "/dev/ping", "/dev/status", "/docs", "/redoc", "/openapi.json"} <= set(routes)

    real_core = main.core
    monkeypatch.setattr(main, "core", Tripwire())  # hostile phase: if any request got through, the core would blow up
    for path, method in routes.items():
        for host in HOSTILE:
            body = {"message": "x"} if method == "POST" else None
            assert client.request(method, path, json=body, headers={"Host": host}).status_code == 403, (method, path, host)
    for method in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
        assert client.request(method, "/no/such/route", headers={"Host": "evil.example"}).status_code == 403  # refused before routing: nothing can be probed

    monkeypatch.setattr(main, "core", real_core)  # loopback phase: the real application, with POST bodies that stop at validation or authentication
    for path, method in routes.items():
        body = {} if method == "POST" else None
        assert client.request(method, path, json=body, headers={"Host": "localhost:8000"}).status_code != 403, (method, path)  # not caught by the guard
    for method in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
        assert client.request(method, "/no/such/route", headers={"Host": "[::1]:8000"}).status_code in (404, 405)


def test_the_guard_judges_only_the_host_header_so_it_is_not_network_isolation_or_authentication():
    """Pinned on purpose, so nobody mistakes it for more than it is: a client at a NON-loopback address that simply sends `Host: localhost` is let through.
    What keeps such a client out is binding the server to the loopback interface (uvicorn's own default is 127.0.0.1), and what makes a caller trusted is
    authentication - not this check, which stops a BROWSER being steered at the local server under an attacker's hostname (DNS rebinding)."""
    scope = {"type": "http", "method": "GET", "path": "/x", "client": ("203.0.113.9", 54321), "headers": [(b"host", b"localhost:8000")]}

    sent, ran = run_asgi(LoopbackHostMiddleware, scope)

    assert ran == ["http"] and sent == []
