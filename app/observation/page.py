"""
Structured page observation: a guarded, cookie-less, credential-less HTTP GET of a public http(s) page, parsed with
the standard library into an Observation (URL, title, visible text, links, buttons, form fields).

This reads what a stranger would be served. It is NOT the user's logged-in browser tab: no cookies are sent or kept,
no Authorization header exists, `trust_env` is off (no proxy / .netrc credentials), and a URL carrying user-info is
refused. Authenticated browsing is out of scope for this slice.

The fetch guard (`validate_fetch_url`) refuses non-http(s), user-info, unusual ports, and any host that resolves to
a non-public address (loopback, private, link-local, reserved ...), and re-checks every redirect hop - so a page
cannot bounce Jarvis into the local network. Known limit: the check resolves the name before connecting, so a DNS
rebinding race is not closed here.

The parser is a best-effort extractor, not a browser: it ignores script / style / template / svg / head (except
the title) and elements marked hidden. It NEVER reads a form field's value or a hidden input (prefilled secrets,
CSRF tokens) - only a field's type, name and label.
"""

from __future__ import annotations

import ipaddress
import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from http.cookiejar import DefaultCookiePolicy
from urllib.parse import urljoin, urlsplit

import httpx

from app.observation.base import (
    FETCH_TIMEOUT_SECONDS,
    MAX_CONTROLS,
    MAX_FETCH_BYTES,
    MAX_LABEL_CHARS,
    MAX_LINKS,
    MAX_REDIRECTS,
    MAX_TEXT_CHARS,
    MAX_TITLE_CHARS,
    Control,
    Observation,
    ObservationError,
    safe_url,
    scrub,
)

ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})
USER_AGENT = "Jarvis-Observer/1.0 (read-only page observation)"
_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


# --- fetch guard --------------------------------------------------------------------------------------------------


def _resolve(host: str) -> list[str]:
    try:
        return [info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)]
    except OSError as exc:
        raise ObservationError(f"could not resolve host '{host}'") from exc


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_fetch_url(url: object, resolver: Callable[[str], list[str]] = _resolve) -> str:
    """The URL if it may be fetched, else ObservationError. Called for the requested URL and every redirect hop."""
    if not isinstance(url, str) or not url.strip():
        raise ObservationError("url must be a non-empty string")
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        raise ObservationError("url is not valid") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ObservationError("url must be an http or https URL with a host")
    if parts.username or parts.password:
        raise ObservationError("url must not contain credentials")
    if port is not None and port not in ALLOWED_PORTS:
        raise ObservationError(f"port {port} is not allowed")
    addresses = [parts.hostname] if _looks_like_ip(parts.hostname) else resolver(parts.hostname)
    if not addresses or not all(_is_public(a) for a in addresses):
        raise ObservationError("that address is not a public web host, so it is not fetched")
    return url.strip()


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class FetchedPage:
    url: str  # the final URL after redirects
    status: int
    html: str


class PageFetcher(ABC):
    """Fetches one page's HTML. The seam tests replace; the real one is HttpPageFetcher."""

    @abstractmethod
    def fetch(self, url: str) -> FetchedPage:
        """Return the page or raise ObservationError."""


class HttpPageFetcher(PageFetcher):
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], list[str]] = _resolve,
        timeout: float = FETCH_TIMEOUT_SECONDS,
        max_bytes: int = MAX_FETCH_BYTES,
        max_redirects: int = MAX_REDIRECTS,
    ):
        self._transport = transport
        self._resolver = resolver
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def fetch(self, url: str) -> FetchedPage:
        current = validate_fetch_url(url, self._resolver)
        # A new client per fetch, with cookies switched off: nothing is kept between observations.
        with httpx.Client(
            transport=self._transport,
            timeout=self._timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.5"},
        ) as client:
            client.cookies.jar.set_policy(DefaultCookiePolicy(allowed_domains=[]))
            for _hop in range(self._max_redirects + 1):
                try:
                    with client.stream("GET", current) as response:
                        if response.is_redirect:
                            location = response.headers.get("location", "")
                            current = validate_fetch_url(urljoin(current, location), self._resolver)
                            continue
                        return self._read(current, response)
                except httpx.HTTPError as exc:
                    raise ObservationError(f"could not fetch the page ({type(exc).__name__})") from exc
        raise ObservationError(f"too many redirects (more than {self._max_redirects})")

    def _read(self, url: str, response: httpx.Response) -> FetchedPage:
        if response.status_code >= 400:
            raise ObservationError(f"the page answered HTTP {response.status_code}")
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type and content_type not in _HTML_TYPES:
            raise ObservationError(f"the page is '{content_type}', not HTML or text")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) >= self._max_bytes:
                del body[self._max_bytes :]  # a bounded read: what fits is what is observed
                break
        return FetchedPage(url=url, status=response.status_code, html=body.decode(response.encoding or "utf-8", "replace"))


# --- parsing ------------------------------------------------------------------------------------------------------

_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head", "iframe", "object", "canvas"})
#: Form widgets whose CONTENT is a value (a prefilled textarea, the options of a select): the control is listed, its
#: content is never read into the text.
_FORM_CONTENT = frozenset({"textarea", "select", "option", "optgroup", "datalist"})
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})
_BLOCK = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "section", "article", "header", "footer", "nav",
     "main", "h1", "h2", "h3", "h4", "h5", "h6", "form", "label", "blockquote", "pre", "dt", "dd", "hr", "aside",
     "a", "button", "select", "textarea"}
)
_BUTTON_INPUTS = frozenset({"submit", "button", "reset", "image"})
_NEVER_READ_INPUTS = frozenset({"hidden"})


def _hidden(attrs: dict) -> bool:
    style = (attrs.get("style") or "").replace(" ", "").lower()
    return (
        "hidden" in attrs
        or (attrs.get("aria-hidden") or "").lower() == "true"
        or "display:none" in style
        or "visibility:hidden" in style
    )


class _PageParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[Control] = []
        self.controls: list[Control | dict] = []
        self._labels: dict[str, str] = {}  # <label for=id> -> text
        self._stack: list[tuple[str, bool]] = []  # (tag, hides content)
        self._in_title = False
        self._link: dict | None = None
        self._button: dict | None = None
        self._label_for: str | None = None

    # --- helpers ---
    @property
    def _visible(self) -> bool:
        return not any(hides for _tag, hides in self._stack)

    def _add_text(self, text: str) -> None:
        if self._visible:
            self.text_parts.append(text)

    # --- HTMLParser hooks ---
    def handle_starttag(self, tag: str, attrs_list: list) -> None:
        attrs = {k.lower(): (v if v is not None else "") for k, v in attrs_list}
        hides = tag in _SKIP or _hidden(attrs)
        if tag == "title":
            self._in_title = True
        was_visible = self._visible
        if tag not in _VOID:
            self._stack.append((tag, hides or tag in _FORM_CONTENT))
        if tag in _BLOCK:
            self._add_text(" ")
        if hides or not was_visible:
            return
        if tag == "a" and attrs.get("href"):
            href = safe_url(urljoin(self.base_url, attrs["href"].strip()))
            self._link = {"href": href, "label": attrs.get("aria-label") or attrs.get("title") or "", "parts": []}
        elif tag == "button":
            self._button = {"label": attrs.get("aria-label") or attrs.get("title") or "", "parts": [], "type": attrs.get("type", "")}
        elif tag == "input":
            self._input(attrs)
        elif tag in ("select", "textarea"):
            self._field(tag, tag, attrs)
        elif tag == "label" and attrs.get("for"):
            self._label_for = attrs["for"]
            self._labels.setdefault(self._label_for, "")

    def _input(self, attrs: dict) -> None:
        input_type = (attrs.get("type") or "text").lower()
        if input_type in _NEVER_READ_INPUTS:
            return  # CSRF tokens and other hidden values are never observed
        if input_type in _BUTTON_INPUTS:
            self._append_control(Control("button", scrub(attrs.get("value") or attrs.get("aria-label") or attrs.get("alt") or input_type, MAX_LABEL_CHARS)))
            return
        self._field("input", input_type, attrs)

    def _field(self, tag: str, input_type: str, attrs: dict) -> None:
        label = attrs.get("aria-label") or attrs.get("placeholder") or ""
        control = {
            "kind": tag,
            "name": scrub(attrs.get("name") or attrs.get("id") or "", 60),
            "input_type": input_type if tag == "input" else "",
            "label": label,
            "id": attrs.get("id") or "",
        }
        # NOTE: the field's `value` is deliberately never read.
        self._append_control(control)

    def _append_control(self, control: Control | dict) -> None:
        if len(self.controls) < MAX_CONTROLS:
            self.controls.append(control)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._link is not None:
            label = scrub(" ".join(self._link["parts"]) or self._link["label"], MAX_LABEL_CHARS)
            if self._link["href"] and len(self.links) < MAX_LINKS:
                same_host = urlsplit(self._link["href"]).hostname == urlsplit(self.base_url).hostname
                self.links.append(Control("link", label, href=self._link["href"], external=not same_host))
            self._link = None
        elif tag == "button" and self._button is not None:
            label = scrub(" ".join(self._button["parts"]) or self._button["label"] or self._button["type"] or "button", MAX_LABEL_CHARS)
            self._append_control(Control("button", label))
            self._button = None
        elif tag == "label":
            self._label_for = None
        if tag in _BLOCK:
            self._add_text(" ")
        for index in range(len(self._stack) - 1, -1, -1):  # close the nearest open `tag`
            if self._stack[index][0] == tag:
                del self._stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if not self._visible:
            return
        self.text_parts.append(data)
        if self._link is not None:
            self._link["parts"].append(data)
        if self._button is not None:
            self._button["parts"].append(data)
        if self._label_for is not None:
            self._labels[self._label_for] = (self._labels[self._label_for] + " " + data).strip()

    def result(self) -> tuple[str, str, list[Control], bool]:
        controls: list[Control] = []
        for item in self.controls:
            if isinstance(item, Control):
                controls.append(item)
                continue
            label = item["label"] or self._labels.get(item["id"], "") or item["name"]
            controls.append(Control(item["kind"], scrub(label, MAX_LABEL_CHARS), name=item["name"], input_type=item["input_type"]))
        text = " ".join("".join(self.text_parts).split())
        title = scrub(" ".join("".join(self.title_parts).split()), MAX_TITLE_CHARS)
        return title, text, controls, False


def parse_page(html: str, url: str) -> Observation:
    """The Observation of `html` as served at `url`: bounded and scrubbed. Malformed or empty markup yields an
    empty (insufficient) Observation rather than an error."""
    parser = _PageParser(url)
    try:
        parser.feed(html if isinstance(html, str) else "")
        parser.close()
    except Exception:  # noqa: BLE001 - html.parser can raise on pathological input; that is "nothing usable"
        pass
    title, text, controls, _ = parser.result()
    truncated = len(text) > MAX_TEXT_CHARS
    return Observation(
        source="page",
        url=safe_url(url),
        title=title,
        text=scrub(text, MAX_TEXT_CHARS),
        controls=tuple(parser.links) + tuple(controls),
        truncated=truncated
        or len(parser.links) >= MAX_LINKS
        or len(parser.controls) >= MAX_CONTROLS,
    )
