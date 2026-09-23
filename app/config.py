"""
Central configuration for Rhasspy.

All settings are read from environment variables (or a local .env file).
Nothing sensitive is hard-coded here — see .env.example for the list of
variables Rhasspy understands.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Basic app info
    app_name: str = "Rhasspy"
    environment: str = "development"  # development | production

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Gemini (AI provider)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # Additional AI providers (M11 foundation - see app/providers/factory.py
    # for the registry these are selected through). Each is independently
    # optional: leaving a provider's API key blank just means that
    # provider raises a clear config error if it's ever actually invoked -
    # it never affects importing, testing, or starting Rhasspy, and Core
    # still defaults to Gemini alone (no provider selection/fallback yet).
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/auto"

    # Cloudflare Workers AI needs both a token and an account id, since its
    # API URL is per-account (.../accounts/{account_id}/ai/run/{model}).
    cloudflare_api_key: str = ""
    cloudflare_account_id: str = ""
    cloudflare_model: str = "@cf/meta/llama-3.1-8b-instruct"

    # AI provider manager (M12 - see app/providers/manager.py and
    # app/providers/factory.py:create_default_ai_provider_manager) - ordered
    # fallback with retry/cooldown across the providers above. Local AI is
    # planned as a future addition to this same order, not built yet.
    # Ordered, comma-separated list of provider names to try, with automatic
    # fallback to the next on failure; a name left out of this list is
    # effectively disabled. Same convention as JARVIS_SEARCH_PROVIDER_ORDER.
    ai_provider_order: str = "gemini,groq,openrouter,cloudflare"
    # Extra attempts against the SAME provider for a transient (5xx) error
    # before falling back to the next provider - 2 means up to 3 attempts
    # total. Rate-limit/network/auth errors never retry the same provider.
    ai_provider_retry_count: int = 2
    ai_provider_retry_delay_seconds: float = 0.5
    # How long a provider that returned a rate-limit (429) error is skipped
    # for, in seconds, before the manager will try it again.
    ai_provider_cooldown_seconds: float = 60.0

    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"

    xai_api_key: str = ""
    xai_model: str = "grok-2-latest"

    # Memory (local SQLite persistence for conversations and user facts)
    memory_db_path: str = "jarvis_memory.db"

    # SearXNG (web search provider) - base URL of a running SearXNG instance
    searxng_url: str = ""

    # Ordered, comma-separated list of search provider names to try, with
    # automatic fallback to the next on failure (see app/search/manager.py
    # and app/search/factory.py). Only "searxng" is implemented so far.
    search_provider_order: str = "searxng"

    # Search result cache (SQLite-backed, see app/search/sqlite_cache.py) -
    # how long a cached web_search result stays valid, in seconds, and
    # where the cache file lives. Expired entries are treated as misses.
    search_cache_ttl: int = 600
    search_cache_db_path: str = "jarvis_search_cache.db"

    # Reminders (persistent tasks/reminders + background scheduler,
    # see app/reminders/) - local SQLite file, and how often the
    # background scheduler polls for due reminders, in seconds.
    reminders_db_path: str = "jarvis_reminders.db"
    reminder_poll_interval_seconds: float = 30.0

    # Automations (persistent automation engine, see app/automation/) -
    # local SQLite file, and how often the background engine polls for
    # due automations, in seconds.
    automations_db_path: str = "jarvis_automations.db"
    automation_poll_interval_seconds: float = 30.0

    # Google Calendar (OAuth 2.0 desktop/localhost flow, see
    # app/calendar/) - client_id/client_secret come from a Google Cloud
    # "Desktop app" OAuth client; no default and no key is hard-coded
    # anywhere. A single primary calendar only (see app/calendar/base.py).
    google_client_id: str = ""
    google_client_secret: str = ""
    google_oauth_redirect_port: int = 8765
    calendar_id: str = "primary"
    calendar_token_db_path: str = "jarvis_calendar_token.db"

    # Rhasspy Tasks (local-only todo items, see app/tasks/) - independent
    # of Reminders; no scheduling, no notifications, no external sync.
    tasks_db_path: str = "jarvis_tasks.db"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="JARVIS_")


settings = Settings()
