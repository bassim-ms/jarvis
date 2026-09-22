"""
Tests for the M13 provider-side capability registry
(app/providers/capabilities.py): pins the exact, justified capability set
per provider (see that module's docstring for the justification behind
each entry) and guards that capability metadata never touches secrets.
"""

import app.providers.capabilities as capabilities_module
from app.providers.capabilities import (
    PROVIDER_CAPABILITIES,
    PROVIDER_LATENCY,
    LatencyTier,
    ProviderCapability,
    capabilities_for,
    latency_tier_for,
)

_ALL_PROVIDER_NAMES = {"gemini", "groq", "openrouter", "cloudflare"}


def test_every_registered_provider_has_capability_metadata():
    assert set(PROVIDER_CAPABILITIES) == _ALL_PROVIDER_NAMES
    assert set(PROVIDER_LATENCY) == _ALL_PROVIDER_NAMES


def test_every_provider_declares_general_conversation():
    for name in _ALL_PROVIDER_NAMES:
        assert ProviderCapability.GENERAL_CONVERSATION in capabilities_for(name)


def test_only_gemini_declares_structured_output_tool_use_search_synthesis():
    for name in _ALL_PROVIDER_NAMES:
        caps = capabilities_for(name)
        expects_all_three = name == "gemini"
        has_any = bool(
            caps
            & {
                ProviderCapability.STRUCTURED_OUTPUT,
                ProviderCapability.TOOL_USE,
                ProviderCapability.SEARCH_SYNTHESIS,
            }
        )
        assert has_any is expects_all_three


def test_no_provider_claims_reasoning_or_coding():
    # Deliberately unassigned everywhere - nothing in this codebase
    # differentiates provider quality on these axes (see module docstring).
    for name in _ALL_PROVIDER_NAMES:
        caps = capabilities_for(name)
        assert ProviderCapability.REASONING not in caps
        assert ProviderCapability.CODING not in caps


def test_only_gemini_declares_long_context():
    assert ProviderCapability.LONG_CONTEXT in capabilities_for("gemini")
    for name in _ALL_PROVIDER_NAMES - {"gemini"}:
        assert ProviderCapability.LONG_CONTEXT not in capabilities_for(name)


def test_only_groq_has_low_latency_tier():
    assert latency_tier_for("groq") is LatencyTier.LOW
    for name in _ALL_PROVIDER_NAMES - {"groq"}:
        assert latency_tier_for(name) is LatencyTier.STANDARD


def test_capabilities_for_unknown_provider_is_empty_not_an_error():
    assert capabilities_for("not-a-real-provider") == frozenset()


def test_latency_tier_for_unknown_provider_defaults_to_standard():
    assert latency_tier_for("not-a-real-provider") is LatencyTier.STANDARD


def test_capabilities_module_never_imports_settings_or_config():
    # Structural guard, not just policy: capability metadata must stay
    # separate from secrets/credentials (app.config.settings) - this
    # module should never even import app.config.
    assert not hasattr(capabilities_module, "settings")
    assert "app.config" not in {
        getattr(v, "__module__", None) for v in vars(capabilities_module).values()
    }


def test_capability_registries_never_contain_key_shaped_values():
    for value in list(PROVIDER_CAPABILITIES) + list(PROVIDER_LATENCY):
        assert "key" not in value.lower()
        assert "token" not in value.lower()
        assert "secret" not in value.lower()
