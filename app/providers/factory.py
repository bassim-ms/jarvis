"""
Builds an AIProvider by stable name - the one place a provider name is
ever checked against a string, mirroring the role app/search/factory.py
plays for search providers. JarvisCore is untouched by M11: it still
constructs its own default provider directly (GeminiProvider()) and has
no notion of provider selection - this registry exists so a future
provider-manager milestone (fallback/rotation/health scoring) has
somewhere to select from, without Core ever needing to know provider
names itself. Adding a new provider means one entry in
PROVIDER_FACTORIES/_PROVIDER_INFO - no changes needed anywhere else.
"""

from app.config import settings
from app.providers.base import AIProvider, ProviderInfo
from app.providers.cloudflare import CLOUDFLARE_URL, CloudflareProvider
from app.providers.gemini import GEMINI_URL, GeminiProvider
from app.providers.groq import GROQ_URL, GroqProvider
from app.providers.manager import AIProviderEntry, AIProviderManager
from app.providers.mistral import MISTRAL_URL, MistralProvider
from app.providers.openrouter import OPENROUTER_URL, OpenRouterProvider
from app.providers.xai import XAI_URL, XAIProvider

PROVIDER_FACTORIES: dict[str, type[AIProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
    "mistral": MistralProvider,
    "xai": XAIProvider,
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
            name="mistral",
            display_name="Mistral",
            base_url=MISTRAL_URL,
            required_env=["JARVIS_MISTRAL_API_KEY"],
            model_env="JARVIS_MISTRAL_MODEL",
            configured=bool(settings.mistral_api_key),
        ),
        ProviderInfo(
            name="xai",
            display_name="xAI",
            base_url=XAI_URL,
            required_env=["JARVIS_XAI_API_KEY"],
            model_env="JARVIS_XAI_MODEL",
            configured=bool(settings.xai_api_key),
        ),
    ]


def build_ai_provider_entries() -> list[AIProviderEntry]:
    """Instantiate the providers named in settings.ai_provider_order, in
    order, each wrapped in an AIProviderEntry under its registry name -
    mirrors app/search/factory.py:build_search_providers for AI providers."""
    names = [
        name.strip() for name in settings.ai_provider_order.split(",") if name.strip()
    ]
    return [AIProviderEntry(name=name, provider=create_provider(name)) for name in names]


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
    )
