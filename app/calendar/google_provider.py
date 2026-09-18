"""
Minimal Google Calendar client, behind the CalendarProvider interface.

Talks to the Google Calendar REST v3 API directly over HTTP using httpx
(already a project dependency), so no extra SDK is needed - the same
pattern GeminiProvider (app/providers/gemini.py) and SearXNGProvider
(app/search/searxng.py) already use. Every method refreshes the stored
OAuth token first if it's expired/near-expiry (see
app/calendar/oauth.py::needs_refresh), transparently to the caller.

Google's event JSON shape (`summary`/`start.dateTime`/...) never leaks
past `_event_from_json`/`_event_to_json` - callers only ever see
`CalendarEvent` (app/calendar/base.py).
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from app.calendar.base import (
    CalendarEvent,
    CalendarProvider,
    CalendarProviderError,
    TokenStore,
)
from app.calendar.oauth import GoogleOAuthClient, OAuthError, needs_refresh

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


class GoogleCalendarError(CalendarProviderError):
    """Raised when the Google Calendar API can't be reached, isn't
    authorized, the stored OAuth token can't be refreshed, or the
    response is unusable."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_datetime(value: dict) -> str:
    """A Google event's start/end is `{"dateTime": ...}` for a timed
    event or `{"date": ...}` for an all-day one - either way, the raw
    string is passed through unchanged, same convention as every other
    timestamp in this codebase."""
    if "dateTime" in value:
        return value["dateTime"]
    if "date" in value:
        return value["date"]
    raise KeyError("dateTime/date")


class GoogleCalendarProvider(CalendarProvider):
    """CalendarProvider implementation backed by a single Google Calendar
    (`calendar_id`, defaulting to the account's primary calendar)."""

    def __init__(
        self,
        oauth_client: GoogleOAuthClient,
        token_store: TokenStore,
        calendar_id: str = "primary",
    ):
        self.oauth_client = oauth_client
        self.token_store = token_store
        self.calendar_id = calendar_id

    def _access_token(self) -> str:
        token = self.token_store.get()
        if token is None:
            raise GoogleCalendarError(
                "Calendar is not authorized yet - run the OAuth setup flow "
                "first (see app/calendar/authorize.py)."
            )
        if needs_refresh(token):
            try:
                token = self.oauth_client.refresh(token.refresh_token)
            except OAuthError as exc:
                raise GoogleCalendarError(
                    f"Failed to refresh calendar access: {exc}"
                ) from exc
            self.token_store.save(token)
        return token.access_token

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        access_token = self._access_token()
        url = f"{CALENDAR_API_BASE}/calendars/{self.calendar_id}{path}"
        try:
            response = httpx.request(
                method,
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=30,
                **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GoogleCalendarError(f"Failed to reach Google Calendar: {exc}") from exc
        return response

    @staticmethod
    def _event_from_json(data: dict) -> CalendarEvent:
        try:
            return CalendarEvent(
                id=data["id"],
                title=data.get("summary", ""),
                start=_extract_datetime(data["start"]),
                end=_extract_datetime(data["end"]),
                description=data.get("description", ""),
                location=data.get("location", ""),
            )
        except (KeyError, TypeError) as exc:
            raise GoogleCalendarError(f"Unexpected Google Calendar event: {data}") from exc

    @staticmethod
    def _event_to_json(
        title: str, start: str, end: str, description: str, location: str
    ) -> dict:
        return {
            "summary": title,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
            "description": description,
            "location": location,
        }

    def list_events(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 20,
    ) -> list[CalendarEvent]:
        params: dict[str, Any] = {
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
            "timeMin": time_min or _now_iso(),
        }
        if time_max is not None:
            params["timeMax"] = time_max

        response = self._request("GET", "/events", params=params)
        data = response.json()
        try:
            items = data["items"]
        except (KeyError, TypeError) as exc:
            raise GoogleCalendarError(f"Unexpected Google Calendar response: {data}") from exc

        return [self._event_from_json(item) for item in items]

    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
    ) -> CalendarEvent:
        body = self._event_to_json(title, start, end, description, location)
        response = self._request("POST", "/events", json=body)
        return self._event_from_json(response.json())

    def update_event(
        self,
        event_id: str,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
    ) -> CalendarEvent:
        body: dict[str, Any] = {}
        if title is not None:
            body["summary"] = title
        if start is not None:
            body["start"] = {"dateTime": start}
        if end is not None:
            body["end"] = {"dateTime": end}
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location

        response = self._request("PATCH", f"/events/{event_id}", json=body)
        return self._event_from_json(response.json())

    def delete_event(self, event_id: str) -> None:
        self._request("DELETE", f"/events/{event_id}")
