"""
Calendar abstraction (M9): read/write access to a single external
calendar, kept fully decoupled from any specific vendor - the same
"interface + swappable backend" shape as AIProvider (app/providers/base.py)
and SearchProvider (app/search/base.py).

`CalendarProvider` is the only thing calendar Tools (app/calendar/tools.py)
depend on; `GoogleCalendarProvider` (app/calendar/google_provider.py) is
the only implementation for now, but nothing above this boundary ever
sees a Google API response shape, a Google client type, or Google-specific
error - only `CalendarEvent`/`CalendarProviderError`. A future provider
(CalDAV, Outlook, ...) plugs in here without touching Tools, ToolRegistry,
or JarvisCore.

`OAuthToken`/`TokenStore` are provider-agnostic too (not Google-specific)
even though only a Google OAuth client exists yet - see
app/calendar/oauth.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class CalendarProviderError(RuntimeError):
    """Raised when the calendar provider can't be reached, isn't
    authorized yet, or returns something unusable - never a raw
    vendor-specific exception past this boundary. Callers (Tools) turn
    this into a normal ToolResult.error, the same convention
    SearchProviderError/ProviderError already use at their own
    boundaries."""


@dataclass
class CalendarEvent:
    """One calendar event, in a shape common to every provider.

    `id` is always populated on an event returned by `list_events`/
    `create_event`/`update_event` - there is no "event under
    construction" variant of this type; `create_event`/`update_event`
    take the fields they need as plain arguments instead (see
    `CalendarProvider` below), so this dataclass never has to represent
    a partially-filled state.

    `start`/`end` are ISO 8601 timestamps (with timezone), the same
    string convention every other timestamp in this codebase already
    uses (Reminder.due_at, Automation.next_run_at, ...).
    """

    id: str
    title: str
    start: str
    end: str
    description: str = ""
    location: str = ""


class CalendarProvider(ABC):
    """A single external calendar backend - always a single primary
    calendar for now (see the concrete provider's own `calendar_id`)."""

    @abstractmethod
    def list_events(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 20,
    ) -> list[CalendarEvent]:
        """Return events in [time_min, time_max), soonest first.
        `time_min`/`time_max` default to "now" / "no upper bound" when
        omitted - a provider-level convenience, not AI-driven."""

    @abstractmethod
    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
    ) -> CalendarEvent:
        """Create a new event and return it (with its provider-assigned
        `id`)."""

    @abstractmethod
    def update_event(
        self,
        event_id: str,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
    ) -> CalendarEvent:
        """Update only the given (non-None) fields of an existing event
        and return the event as it now stands."""

    @abstractmethod
    def delete_event(self, event_id: str) -> None:
        """Delete an event. Raises CalendarProviderError if it doesn't
        exist or the delete otherwise fails - never a silent no-op."""


@dataclass
class OAuthToken:
    """A single OAuth 2.0 token set - provider-agnostic (see this
    module's docstring). `expires_at` is an ISO 8601 UTC timestamp, the
    same string convention as every other timestamp in this codebase.
    `refresh_token` is treated as a sensitive credential everywhere it
    flows - never logged, never included in a ToolResult, never returned
    to JarvisCore (see app/calendar/oauth.py and
    app/calendar/token_store.py)."""

    access_token: str
    refresh_token: str
    expires_at: str
    scope: str
    token_type: str = "Bearer"


class TokenStore(ABC):
    """Persistent storage for a single OAuth token set. Deliberately
    single-token (no keying by user/account) - matches this milestone's
    "single primary calendar only" scope; a future multi-account
    milestone would extend this, not work around it."""

    @abstractmethod
    def get(self) -> OAuthToken | None:
        """Return the stored token, or None if the OAuth flow has never
        been completed."""

    @abstractmethod
    def save(self, token: OAuthToken) -> None:
        """Persist `token`, replacing whatever was stored before."""

    @abstractmethod
    def clear(self) -> None:
        """Remove any stored token (e.g. after the user revokes access)."""
