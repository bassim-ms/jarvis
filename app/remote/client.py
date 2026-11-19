"""
A small client for Jarvis's remote API, for a SEPARATE device (another laptop) - standard library only, so it runs with any
Python 3.11+ even where Jarvis itself is not installed (copy this file and signing.py, or the whole app/remote folder).

    python -m app.remote.client --url https://jarvis-pc.tail1234.ts.net --credential-file laptop.cred status
    python -m app.remote.client --url ... --credential-file laptop.cred chat "What's on my list today?"

The credential file is the one `python -m app.devices.admin register ... --credential-file` wrote on the Jarvis computer
({"device_id", "credential"}); move it to the device over a channel you trust and keep it private. The client opens a session
with it, then signs every request with the session key (never sending the credential again), and:
  * re-opens the session once if it has expired or was ended (401), then repeats the request;
  * on a lost connection retries the SAME request with the SAME request id, so Jarvis answers it once, never twice.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

try:
    from app.remote import signing
except ImportError:  # running from a copied folder
    import signing  # type: ignore[no-redef]


class RemoteError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class Transport:
    """Sends one HTTP request: (method, url, headers, body) -> (status, json). Replaceable in tests."""

    timeout: float = 60.0

    def __call__(self, method: str, url: str, headers: dict, body: bytes) -> tuple[int, dict]:
        request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read() or b"{}").get("detail", "")
            except ValueError:
                detail = ""
            return error.code, {"detail": detail}


class RemoteClient:
    def __init__(self, url: str, device_id: str, credential: str, transport=None, clock=time.time, retries: int = 2,
                 retry_delay: float = 1.0, sleep=time.sleep):
        parts = urlsplit(url)
        if parts.scheme != "https" and parts.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("remote access needs an https:// address (only a local test may use plain http)")
        self.base = url.rstrip("/")
        self.device_id, self.credential = device_id, credential
        self.transport = transport or Transport()
        self.clock, self.retries, self.retry_delay, self.sleep = clock, retries, retry_delay, sleep
        self.session_id: str | None = None
        self.key: bytes | None = None

    @classmethod
    def from_credential_file(cls, url: str, path: str, **kwargs) -> "RemoteClient":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(url, data["device_id"], data["credential"], **kwargs)

    def connect(self) -> dict:
        headers = {
            "X-Jarvis-Device-Id": self.device_id,
            "X-Jarvis-Device-Token": self.credential,
            "X-Jarvis-Timestamp": str(int(self.clock())),
            "X-Jarvis-Nonce": signing.new_nonce(),
        }
        status, body = self.transport("POST", self.base + "/remote/v1/session", headers, b"")
        if status != 200:
            raise RemoteError(status, body.get("detail", ""))
        self.session_id = body["session_id"]
        self.key = base64.urlsafe_b64decode(body["session_key"])
        return body

    def disconnect(self) -> None:
        if self.session_id:
            try:
                self._signed("DELETE", "/remote/v1/session", None, reconnect=False)
            finally:
                self.session_id, self.key = None, None

    def status(self) -> dict:
        return self._signed("GET", "/remote/v1/status", None)

    def chat(self, message: str, conversation_id: str | None = None) -> dict:
        request_id = secrets.token_urlsafe(12)
        return self._signed("POST", "/remote/v1/chat", {"message": message, "conversation_id": conversation_id},
                            request_id=request_id)

    def _signed(self, method: str, path: str, payload: dict | None, *, request_id: str | None = None, reconnect: bool = True):
        if self.session_id is None:
            self.connect()
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        attempt = 0
        reconnected = False
        while True:
            headers = signing.signed_headers(self.key, self.session_id, method, path, body, int(self.clock()))
            if payload is not None:
                headers["Content-Type"] = "application/json"
            if request_id:
                headers["X-Jarvis-Request-Id"] = request_id
            try:
                status, answer = self.transport(method, self.base + path, headers, body)
            except OSError:  # the connection was lost: try again, same request id (Jarvis answers it once)
                attempt += 1
                if attempt > self.retries:
                    raise
                self.sleep(self.retry_delay * attempt)
                continue
            if status == 401 and reconnect and not reconnected:  # session expired or ended: open a new one, once
                reconnected = True
                self.connect()
                continue
            if status != 200:
                raise RemoteError(status, answer.get("detail", ""))
            return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Talk to your Jarvis from another device (remote access).")
    parser.add_argument("--url", required=True, help="https://<your computer's tailnet name>")
    parser.add_argument("--credential-file", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    chat = sub.add_parser("chat")
    chat.add_argument("message")
    chat.add_argument("--conversation")
    args = parser.parse_args(argv)
    client = RemoteClient.from_credential_file(args.url, args.credential_file)
    try:
        if args.command == "status":
            print(json.dumps(client.status(), indent=2))
        else:
            answer = client.chat(args.message, args.conversation)
            print(answer.get("reply", ""))
            print(f"(conversation {answer.get('conversation_id')})", file=sys.stderr)
    except RemoteError as error:
        print(f"Jarvis refused: {error.detail or error.status}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Could not reach Jarvis: {type(error).__name__}", file=sys.stderr)
        return 2
    finally:
        try:
            client.disconnect()
        except (RemoteError, OSError):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
