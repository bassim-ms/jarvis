"""
Run this once to interactively authorize Rhasspy's access to your Google
Calendar:

    python -m app.calendar.authorize

Opens your browser for Google's consent screen, waits for the local
redirect (see app/calendar/oauth.py::CalendarAuthFlow), and persists the
resulting refresh token to JARVIS_CALENDAR_TOKEN_DB_PATH (default
jarvis_calendar_token.db). After this, GoogleCalendarProvider refreshes
the access token automatically on every call - you never need to run
this again unless you revoke access or delete the token database.

Requires JARVIS_GOOGLE_CLIENT_ID/JARVIS_GOOGLE_CLIENT_SECRET to already
be set (a Google Cloud "Desktop app" OAuth client) - see .env.example.
"""

from app.calendar.factory import create_default_oauth_client, create_default_token_store
from app.calendar.oauth import CalendarAuthFlow
from app.config import settings


def main() -> None:
    token_store = create_default_token_store(settings.calendar_token_db_path)
    oauth_client = create_default_oauth_client()
    flow = CalendarAuthFlow(
        oauth_client=oauth_client,
        token_store=token_store,
        redirect_port=settings.google_oauth_redirect_port,
    )
    flow.run()
    print("Calendar authorization complete.")


if __name__ == "__main__":
    main()
