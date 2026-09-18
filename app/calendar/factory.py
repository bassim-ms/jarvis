"""
Builds the application's default TokenStore/GoogleOAuthClient/
CalendarProvider from config - the one place construction happens, same
role app/reminders/factory.py plays for the reminder store/scheduler.
"""

from app.calendar.base import CalendarProvider, TokenStore
from app.calendar.google_provider import GoogleCalendarProvider
from app.calendar.oauth import CALENDAR_EVENTS_SCOPE, GoogleOAuthClient
from app.calendar.token_store import SQLiteTokenStore
from app.config import settings


def create_default_token_store(db_path: str) -> TokenStore:
    return SQLiteTokenStore(db_path)


def create_default_oauth_client() -> GoogleOAuthClient:
    return GoogleOAuthClient(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        redirect_uri=f"http://localhost:{settings.google_oauth_redirect_port}/",
        scope=CALENDAR_EVENTS_SCOPE,
    )


def create_default_calendar_provider(
    token_store: TokenStore | None = None,
) -> CalendarProvider:
    token_store = token_store or create_default_token_store(
        settings.calendar_token_db_path
    )
    return GoogleCalendarProvider(
        oauth_client=create_default_oauth_client(),
        token_store=token_store,
        calendar_id=settings.calendar_id,
    )
