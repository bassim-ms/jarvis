"""
Central configuration for Jarvis.

All settings are read from environment variables (or a local .env file).
Nothing sensitive is hard-coded here — see .env.example for the list of
variables Jarvis understands.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Basic app info
    app_name: str = "Jarvis"
    environment: str = "development"  # development | production

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Gemini (AI provider)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

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

    model_config = SettingsConfigDict(env_file=".env", env_prefix="JARVIS_")


settings = Settings()
