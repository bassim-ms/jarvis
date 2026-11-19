"""
Loopback-only Host check: refuse an HTTP request whose `Host` header is not a loopback address.

WHY. Jarvis's `/chat` is the anonymous LOCAL interface and has no authentication. A web page can reach a local server through "DNS rebinding": the
attacker's own domain is made to resolve to 127.0.0.1, the browser then treats the local server as that page's own origin, and the request arrives
with `Host: attacker.example`. A genuine local client always names the machine itself, so refusing every other Host closes that route for EVERY
endpoint, including the ones that can reach the desktop.

WHAT IS ACCEPTED (the whole of it; the header is judged exactly as received, nothing is trimmed or normalised):

    localhost            127.0.0.1 (any 127.x.x.x)            [::1]        each optionally followed by  :<port>   (1-65535)

  * `localhost` is matched case-insensitively and EXACTLY: not `localhost.` and not `localhost.evil.example`.
  * An IPv4 literal must be four plain decimal fields (`127.1`, `0x7f.0.0.1`, `2130706433` and leading zeros are not accepted) and loopback.
  * IPv6 must be in brackets (as the Host header requires) and be exactly `::1` in any spelling (`[0:0:0:0:0:0:0:1]` too); an IPv4-mapped address,
    a zone id, or a bare `::1` without brackets is refused.
  * A port is one to five ASCII digits with a value from 1 to 65535: an empty port (`127.0.0.1:`), a sign, a space or a Unicode digit is refused.
  * No whitespace, `@`, `/`, `\\`, `?`, `#`, `%` or control character anywhere; a missing Host header, or more than one, is refused.

WHAT IT IS NOT. This is an additional safeguard, not a replacement for the two things that actually protect a local server: binding it to the
loopback interface (start it with `uvicorn app.main:app --host 127.0.0.1`; the `host` default in config.py is not read by anything) and, for anything that must be
trusted, authentication (`/device/chat`). It does not look at the client's address, `Origin` or `X-Forwarded-Host`, and it does not make `/chat`
authenticated. A refused request gets `403` and never reaches Jarvis.
"""

from __future__ import annotations

import ipaddress
import logging
import re

logger = logging.getLogger("jarvis.host_guard")

_PORT = re.compile(r"[0-9]{1,5}")
_FORBIDDEN = frozenset(" \t\r\n\x00\\/@?#%")
_IPV6_LOOPBACK = ipaddress.IPv6Address("::1")
MAX_HOST_CHARS = 255
#: The only path prefix a remote (non-loopback, configured) host may reach - see LoopbackHostMiddleware.remote_hosts.
REMOTE_PREFIX = "/remote/"


def _port_ok(port: str) -> bool:
    return _PORT.fullmatch(port) is not None and 1 <= int(port) <= 65535


def is_loopback_host(value: object) -> bool:
    """True only for a Host header value that names this machine's loopback address - see the module docstring for exactly what that means."""
    if not isinstance(value, str) or not value or len(value) > MAX_HOST_CHARS:
        return False
    if any(ch in _FORBIDDEN or ord(ch) < 32 or ord(ch) > 126 for ch in value):
        return False
    if value.startswith("["):  # bracketed IPv6, optionally with a port
        end = value.find("]")
        if end == -1:
            return False
        literal, rest = value[1:end], value[end + 1 :]
        if rest and not (rest.startswith(":") and _port_ok(rest[1:])):
            return False
        try:
            address = ipaddress.ip_address(literal)
        except ValueError:
            return False
        return isinstance(address, ipaddress.IPv6Address) and address == _IPV6_LOOPBACK
    if value.count(":") > 1:  # an IPv6 literal must be bracketed
        return False
    host, separator, port = value.partition(":")
    if separator and not _port_ok(port):
        return False
    host = host.lower()
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return isinstance(address, ipaddress.IPv4Address) and address.is_loopback


class LoopbackHostMiddleware:
    """Pure ASGI middleware: HTTP and WebSocket requests without exactly one loopback Host header are refused (HTTP 403 / WebSocket close)."""

    def __init__(self, app, remote_hosts=None):
        self.app = app
        # Remote access (app/remote): a callable returning the host names remote requests may be addressed to - EMPTY while
        # remote access is off (the default). Such a host is let through ONLY for an HTTP request under /remote/; every other
        # route (the dashboard, the anonymous /chat, /providers/status, /dev, the docs) stays loopback-only.
        self.remote_hosts = remote_hosts

    def _remote_allowed(self, scope, host: str) -> bool:
        if self.remote_hosts is None or scope["type"] != "http":
            return False
        path = scope.get("path", "")
        if not path.startswith(REMOTE_PREFIX) or ".." in path:
            return False
        name = host.lower().rstrip(".")
        name = name.rsplit(":", 1)[0] if name.count(":") == 1 and _port_ok(name.rsplit(":", 1)[1]) else name
        try:
            allowed = self.remote_hosts()
        except Exception:  # noqa: BLE001 - a broken setting means "not allowed", never "allowed"
            return False
        return bool(allowed) and name in allowed

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        hosts = [value for name, value in scope.get("headers", []) if name.lower() == b"host"]
        host = hosts[0].decode("latin-1") if len(hosts) == 1 else None
        if host is not None and (is_loopback_host(host) or self._remote_allowed(scope, host)):
            await self.app(scope, receive, send)
            return
        shown = "(missing or repeated)" if host is None else repr(host[:80])
        logger.warning("host_guard refused a request: Host=%s", shown)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = b"Forbidden: this server only answers requests addressed to a loopback Host."
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(body)).encode("ascii"))],
            }
        )
        await send({"type": "http.response.body", "body": body})
