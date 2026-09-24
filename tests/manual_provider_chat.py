"""
Manual, network-touching diagnostic for Jarvis's individual AI providers
(Gemini, Groq, OpenRouter, Cloudflare Workers AI - see
app/providers/factory.py). "Rhasspy" is only the local wake-word name (see
app/voice/); the assistant itself is Jarvis, which is why this utility's own
banner reads "Jarvis AI Provider Chat Test" - nothing here renames anything.

This is NOT part of the pytest suite (see tests/test_manual_provider_chat.py
for the mocked unit tests that do run under pytest) - it is a standalone
script you run by hand. Unlike tests/manual_provider_fallback.py (which
exercises the AIProviderManager fallback chain end to end), this utility
asks the SAME question directly to each provider independently, through
app.providers.factory.create_provider - never AIProviderManager - so every
configured provider is actually invoked and one provider's failure never
skips or masks another's result.

Nothing here modifies .env, .env.example, API keys, model names, the
provider manager, wake-word code, or any production code - this only calls
the existing AIProvider/factory building blocks from a separate process.

Usage:
    python -m tests.manual_provider_chat --provider gemini
    python -m tests.manual_provider_chat --provider groq
    python -m tests.manual_provider_chat --provider openrouter
    python -m tests.manual_provider_chat --provider cloudflare
    python -m tests.manual_provider_chat --provider all
    python -m tests.manual_provider_chat --provider all --question "..."
"""

import argparse
import sys
from dataclasses import dataclass

from app.providers import factory
from app.providers.base import ProviderError

# The four registered providers (app/providers/factory.py: PROVIDER_FACTORIES),
# in the same order as the production fallback chain - not read from
# settings.ai_provider_order, so `--provider all` always covers every
# provider this project supports regardless of local .env overrides.
PROVIDERS = ["gemini", "groq", "openrouter", "cloudflare"]
DISPLAY_NAMES = {
    "gemini": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "cloudflare": "Cloudflare",
}
DEFAULT_QUESTION = "What is the capital of Japan? Answer in one sentence."


@dataclass
class ChatResult:
    """Outcome of asking one provider the question. `answer`/`reason` are
    mutually exclusive with `success`. Never holds an API key, header, or
    request URL - only what came back from AIProvider.generate_reply (an
    answer string) or a ProviderError's already secret-safe message (see
    app/providers/_openai_compatible.py, app/providers/gemini.py,
    app/providers/cloudflare.py - none of them build their error message
    from a key/header/URL)."""

    name: str
    display_name: str
    success: bool
    answer: str | None = None
    reason: str | None = None


def ask_provider(name: str, question: str) -> ChatResult:
    """Build the named provider through the existing factory and ask it
    `question` directly - one real HTTP request if run outside a test.
    Construction never requires an API key (see factory.create_provider);
    a missing key surfaces as an ordinary ProviderError from generate_reply
    ("JARVIS_..._API_KEY is not set."), handled the same as any other
    provider failure below - no separate "missing credentials" code path
    needed."""
    display_name = DISPLAY_NAMES[name]
    provider = factory.create_provider(name)

    try:
        answer = provider.generate_reply(question)
    except ProviderError as exc:
        return ChatResult(name, display_name, False, reason=str(exc))
    except Exception as exc:
        # A provider is only ever supposed to raise ProviderError (see
        # app/providers/base.py). This is a safety net for anything else
        # (e.g. a raw httpx timeout) - only the exception's type name is
        # recorded, never str(exc), since a non-ProviderError could be
        # holding anything, including a URL/header (mirrors
        # UnexpectedProviderError in app/providers/manager.py).
        return ChatResult(
            name,
            display_name,
            False,
            reason=f"unexpected {type(exc).__name__} (not a ProviderError)",
        )

    return ChatResult(name, display_name, True, answer=answer)


def run_providers(names: list[str], question: str) -> list[ChatResult]:
    """Ask every provider in `names` the same question, independently.
    Each call is isolated in its own try/except inside ask_provider, so one
    provider failing can never prevent the rest from being tested."""
    return [ask_provider(name, question) for name in names]


def format_result(result: ChatResult) -> str:
    status = "SUCCESS" if result.success else "FAILURE"
    lines = [f"{result.display_name}: {status}"]
    if result.success:
        lines.append(f"Answer: {result.answer}")
    else:
        lines.append(f"Reason: {result.reason}")
    return "\n".join(lines)


def print_report(names: list[str], question: str, results: list[ChatResult]) -> bool:
    print("=== Jarvis AI Provider Chat Test ===")
    print()
    print(f"Question: {question}")
    print()
    for result in results:
        print(format_result(result))
        print()

    passed = all(result.success for result in results)
    print("Provider chat test: PASS" if passed else "Provider chat test: FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manual diagnostic: ask each configured Jarvis AI provider the "
            "same question directly (makes real API requests)."
        )
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=[*PROVIDERS, "all"],
        help="Which provider to ask, or 'all' to ask every provider.",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to send (default: capital-of-Japan sanity check).",
    )
    args = parser.parse_args(argv)

    names = PROVIDERS if args.provider == "all" else [args.provider]
    results = run_providers(names, args.question)
    passed = print_report(names, args.question, results)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
