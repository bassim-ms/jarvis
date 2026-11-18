"""
Builds an AIProvider by stable name - the one place a provider name is
ever checked against a string, mirroring the role app/search/factory.py
plays for search providers. JarvisCore is untouched by M11: it still
constructs its own default provider directly (GeminiProvider()) and has
no notion of provider selection - this registry exists so a future
provider-manager milestone (fallback/rotation/health scoring) has
somewhere to select from, without Core ever needing to know provider
names itself. Adding a new provider means one entry in
PROVIDER_FACTORIES/describe_providers - no changes needed anywhere else.

Deliberately exactly four cloud providers for now: Gemini, Groq,
OpenRouter, Cloudflare Workers AI. A dedicated local model (hosted on a
separate machine, reached over the local network) is planned as the
eventual last-resort fallback after these four, and a browser/computer-use
fallback after that - both future milestones, not built yet. This
registry's by-name construction is exactly the extension point either
will need: a local/browser provider is still just another AIProvider
implementation registered here.
"""

import logging

from app.config import settings
from app.providers.base import AIProvider, ProviderInfo
from app.providers.cloudflare import CLOUDFLARE_URL, CloudflareProvider
from app.providers.freellmapi import FreeLLMAPIProvider, freellmapi_url
from app.providers.gemini import GEMINI_URL, GeminiProvider
from app.providers.groq import GROQ_URL, GroqProvider
from app.providers.manager import AIProviderEntry, AIProviderManager
from app.providers.openrouter import OPENROUTER_URL, OpenRouterProvider
from app.providers.stats import ProviderStatsStore, SQLiteProviderStatsStore

PROVIDER_FACTORIES: dict[str, type[AIProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
    "freellmapi": FreeLLMAPIProvider,
}


class UnknownProviderError(ValueError):
    """Raised when a name outside PROVIDER_FACTORIES is requested."""


def create_provider(name: str) -> AIProvider:
    """Instantiate the AIProvider registered under `name`. Construction
    never requires an API key - each provider only checks its own
    settings once `generate_reply` is actually called (see e.g.
    GeminiProvider.generate_reply), so building a provider that's never
    invoked is always safe, key or no key."""
    try:
        factory = PROVIDER_FACTORIES[name]
    except KeyError:
        raise UnknownProviderError(f"Unknown AI provider '{name}'.") from None
    return factory()


def describe_providers() -> list[ProviderInfo]:
    """Static, non-secret metadata for every registered provider - base
    URL, which env vars it needs, and whether those are currently set.
    Never touches an API key's value, only its presence, so this is
    always safe to log/display (e.g. a future `/providers` status
    endpoint or CLI diagnostic)."""
    return [
        ProviderInfo(
            name="gemini",
            display_name="Gemini",
            base_url=GEMINI_URL,
            required_env=["JARVIS_GEMINI_API_KEY"],
            model_env="JARVIS_GEMINI_MODEL",
            configured=bool(settings.gemini_api_key),
        ),
        ProviderInfo(
            name="groq",
            display_name="Groq",
            base_url=GROQ_URL,
            required_env=["JARVIS_GROQ_API_KEY"],
            model_env="JARVIS_GROQ_MODEL",
            configured=bool(settings.groq_api_key),
        ),
        ProviderInfo(
            name="openrouter",
            display_name="OpenRouter",
            base_url=OPENROUTER_URL,
            required_env=["JARVIS_OPENROUTER_API_KEY"],
            model_env="JARVIS_OPENROUTER_MODEL",
            configured=bool(settings.openrouter_api_key),
        ),
        ProviderInfo(
            name="cloudflare",
            display_name="Cloudflare Workers AI",
            base_url=CLOUDFLARE_URL,
            required_env=["JARVIS_CLOUDFLARE_API_KEY", "JARVIS_CLOUDFLARE_ACCOUNT_ID"],
            model_env="JARVIS_CLOUDFLARE_MODEL",
            configured=bool(
                settings.cloudflare_api_key and settings.cloudflare_account_id
            ),
        ),
        ProviderInfo(
            name="freellmapi",
            display_name="FreeLLMAPI",
            base_url=freellmapi_url(),
            required_env=["JARVIS_FREELLMAPI_API_KEY"],
            # No model_env: FreeLLMAPI always uses "auto" - see
            # app/providers/freellmapi.py. ProviderInfo.model_env already
            # defaults to "", so it's simply left unset here.
            configured=bool(settings.freellmapi_api_key),
        ),
    ]


def _effective_provider_order() -> list[str]:
    """The actual provider order build_ai_provider_entries() uses - almost
    always just settings.ai_provider_order, parsed the same way it always
    has been, with one automatic adjustment: "freellmapi" (see
    app/providers/freellmapi.py) is appended as the final entry whenever
    JARVIS_FREELLMAPI_API_KEY is configured, even if
    JARVIS_AI_PROVIDER_ORDER predates FreeLLMAPI and never mentions it -
    the only manual configuration step FreeLLMAPI needs is its own API
    key, never a second edit to JARVIS_AI_PROVIDER_ORDER (an existing,
    explicit override in .env is not exempt from this). If "freellmapi" is
    already named anywhere in JARVIS_AI_PROVIDER_ORDER - auto-appended
    here on a previous call, or typed in by hand - it's always moved to
    the end: FreeLLMAPI must never be attempted before any other
    configured provider. If the key is blank and "freellmapi" isn't
    explicitly named, nothing changes: exactly today's parsed list, in
    today's order - existing behavior for anyone not using FreeLLMAPI is
    completely unaffected.

    This derives the effective order in application logic rather than
    ever writing to settings.ai_provider_order or .env itself - no second
    routing system, no change to M13's ProviderRouter (which only ever
    sees whatever ordered entries build_ai_provider_entries() below
    produces, exactly as it always has)."""
    names = [
        name.strip() for name in settings.ai_provider_order.split(",") if name.strip()
    ]

    if "freellmapi" in names:
        names = [name for name in names if name != "freellmapi"] + ["freellmapi"]
    elif settings.freellmapi_api_key:
        names = [*names, "freellmapi"]

    return names


def build_ai_provider_entries() -> list[AIProviderEntry]:
    """Instantiate the providers named in _effective_provider_order(), in
    order, each wrapped in an AIProviderEntry under its registry name -
    mirrors app/search/factory.py:build_search_providers for AI providers."""
    return [
        AIProviderEntry(name=name, provider=create_provider(name))
        for name in _effective_provider_order()
    ]


def create_default_ai_provider_manager() -> AIProviderManager:
    """Build the application's default AIProviderManager: an ordered
    fallback chain, with retry/cooldown from settings, over every provider
    named in JARVIS_AI_PROVIDER_ORDER. Not used by JarvisCore yet (M12 only
    establishes the manager) - a later milestone wires Core to this."""
    return AIProviderManager(
        build_ai_provider_entries(),
        retry_count=settings.ai_provider_retry_count,
        retry_delay_seconds=settings.ai_provider_retry_delay_seconds,
        cooldown_seconds=settings.ai_provider_cooldown_seconds,
        max_cooldown_seconds=settings.ai_provider_max_cooldown_seconds,
        stats_store=create_default_stats_store(),
        daily_request_limits=parse_daily_limits(settings.ai_provider_daily_request_limits),
    )


def create_default_stats_store() -> ProviderStatsStore | None:
    """The provider statistics database (settings.provider_stats_db_path, placed in JARVIS_DATA_DIR when that is set). If it
    cannot be opened, statistics are kept in memory only - Jarvis still starts."""
    try:
        return SQLiteProviderStatsStore(settings.provider_stats_db_path)
    except Exception as exc:  # noqa: BLE001 - statistics must never stop Jarvis from starting
        logging.getLogger("jarvis.providers").warning(
            "provider statistics database could not be opened (%s); statistics will not survive a restart", type(exc).__name__
        )
        return None


def parse_daily_limits(text: str) -> dict[str, int]:
    """"groq=1000, gemini=250" -> {"groq": 1000, "gemini": 250}. A malformed or non-positive entry is ignored."""
    limits: dict[str, int] = {}
    for part in (text or "").split(","):
        name, separator, value = part.partition("=")
        name, value = name.strip().lower(), value.strip()
        if separator and name and value.isdigit() and int(value) > 0:
            limits[name] = int(value)
    return limits
