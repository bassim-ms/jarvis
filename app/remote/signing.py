"""
Request signing for remote access - shared by the server (app/remote/api.py) and the client (app/remote/client.py).

Every call made with a session carries four headers:

    X-Jarvis-Session    the session id (not secret on its own)
    X-Jarvis-Timestamp  Unix seconds when the request was signed
    X-Jarvis-Nonce      16-128 random URL-safe characters, never reused
    X-Jarvis-Signature  hex HMAC-SHA256(session key, canonical string)

The canonical string binds the signature to everything that matters: METHOD, path (with query), timestamp, nonce and the
SHA-256 of the exact body bytes. So a captured request cannot be replayed (the nonce is remembered and the timestamp must be
fresh), changed (any byte of method, path or body breaks the signature) or reused by someone holding only the session id
(the key never travels after the session is created and is never stored on disk).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

SESSION_HEADER = "x-jarvis-session"
TIMESTAMP_HEADER = "x-jarvis-timestamp"
NONCE_HEADER = "x-jarvis-nonce"
SIGNATURE_HEADER = "x-jarvis-signature"
REQUEST_ID_HEADER = "x-jarvis-request-id"
DEVICE_ID_HEADER = "x-jarvis-device-id"
DEVICE_TOKEN_HEADER = "x-jarvis-device-token"

NONCE_PATTERN = re.compile(r"[A-Za-z0-9_\-]{16,128}")
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_\-]{8,64}")
SIGNATURE_PATTERN = re.compile(r"[0-9a-f]{64}")


def body_digest(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def canonical(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bytes:
    return "\n".join([method.upper(), path, timestamp, nonce, body_digest(body)]).encode("utf-8")


def sign(key: bytes, method: str, path: str, timestamp: str, nonce: str, body: bytes) -> str:
    return hmac.new(key, canonical(method, path, timestamp, nonce, body), hashlib.sha256).hexdigest()


def signature_matches(key: bytes, signature: str, method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bool:
    return hmac.compare_digest(sign(key, method, path, timestamp, nonce, body), signature)


def new_nonce() -> str:
    return secrets.token_urlsafe(24)


def signed_headers(key: bytes, session_id: str, method: str, path: str, body: bytes, timestamp: int, nonce: str | None = None) -> dict:
    """The four headers for one request (the client side)."""
    nonce = nonce or new_nonce()
    stamp = str(int(timestamp))
    return {
        "X-Jarvis-Session": session_id,
        "X-Jarvis-Timestamp": stamp,
        "X-Jarvis-Nonce": nonce,
        "X-Jarvis-Signature": sign(key, method, path, stamp, nonce, body),
    }
