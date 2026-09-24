"""
Manual, network-touching integration test for the real AI provider fallback
chain (Gemini -> Groq -> OpenRouter -> Cloudflare - see
app/providers/factory.py and app/providers/manager.py).

This is NOT part of the pytest suite (see tests/test_manual_provider_fallback.py
for the mocked unit tests that do run under pytest) - it is a standalone
script you run by hand, because it makes one real HTTP request against
whichever provider you pick as `--target`. Every provider ahead of the
target in the chain is replaced, only inside this process, with a tiny fake
AIProvider that deterministically raises ProviderError - so exactly one real
network call happens per run, and everything before it is guaranteed to
never touch the network. Providers after the target are wired to the real
AIProviderManager exactly as production builds them (via
app.providers.factory.create_provider), but since the manager stops at the
first success, they're never actually invoked when the target succeeds; if
the target unexpectedly fails, the report below reflects whatever actually
happened rather than assuming the plan held.

Nothing here modifies .env, API keys, model names, or any production code -
this only composes the existing AIProviderManager/AIProviderEntry/
create_provider building blocks from a separate process.

Usage:
    python -m tests.manual_provider_fallback --target groq
    python -m tests.manual_provider_fallback --target openrouter
    python -m tests.manual_provider_fallback --target cloudflare
    python -m tests.manual_provider_fallback --target none
"""

import argparse
import sys

from app.config import settings
from app.providers import factory
from app.providers.base import AIProvider, ProviderError
from app.providers.manager import (
    AIProviderEntry,
    AIProviderManager,
    AllProvidersFailedError,
)

# The production fallback order (app/config.py: ai_provider_order default).
# Hardcoded here rather than read from settings.ai_provider_order so this
# tool always exercises the documented Gemini -> Groq -> OpenRouter ->
# Cloudflare chain regardless of local .env overrides.
ORDER = ["gemini", "groq", "openrouter", "cloudflare"]
DISPLAY_NAMES = {
    "gemini": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "cloudflare": "Cloudflare",
}
TEST_PROMPT = "Reply with exactly: RHASSPY_PROVIDER_TEST_OK"
EXPECTED_MARKER = "RHASSPY_PROVIDER_TEST_OK"


class FakeFailingProvider(AIProvider):
    """Deterministically-failing stand-in for a provider this run should
    never actually call. Never touches the network, settings, or any real
    provider class."""

    def __init__(self, name: str):
        self.name = name

    def generate_reply(self, message: str) -> str:
        raise ProviderError(
            f"{self.name}: intentionally failed (manual fallback test)"
        )


class CallRecorder(AIProvider):
    """Wraps another AIProvider only to record whether/how it was actually
    invoked, so the report below reflects real execution - not just the
    plan - even if a "real" provider unexpectedly fails."""

    def __init__(self, name: str, inner: AIProvider, kind: str):
        self.name = name
        self.inner = inner
        self.kind = kind  # "real" or "fake"
        self.invoked = False
        self.succeeded = False
        self.response: str | None = None
        self.error: BaseException | None = None

    def generate_reply(self, message: str) -> str:
        self.invoked = True
        try:
            reply = self.inner.generate_reply(message)
        except Exception as exc:
            self.error = exc
            raise
        self.succeeded = True
        self.response = reply
        return reply


def build_recorders(target: str) -> list[CallRecorder]:
    """One CallRecorder per provider in ORDER: the target gets a real
    provider built through the existing factory (app.providers.factory);
    every other provider gets a FakeFailingProvider. `target == "none"`
    means every provider is faked."""
    target_index = None if target == "none" else ORDER.index(target)

    recorders = []
    for index, name in enumerate(ORDER):
        if index == target_index:
            inner: AIProvider = factory.create_provider(name)
            kind = "real"
        else:
            inner = FakeFailingProvider(name)
            kind = "fake"
        recorders.append(CallRecorder(name, inner, kind))
    return recorders


def run_manager(recorders: list[CallRecorder]):
    """Build a real AIProviderManager (production class, production retry/
    cooldown settings) over the recorders and run one generate_reply call.
    Returns (reply, error) - exactly one of which is None."""
    entries = [AIProviderEntry(name=r.name, provider=r) for r in recorders]
    manager = AIProviderManager(
        entries,
        retry_count=settings.ai_provider_retry_count,
        retry_delay_seconds=settings.ai_provider_retry_delay_seconds,
        cooldown_seconds=settings.ai_provider_cooldown_seconds,
    )
    try:
        return manager.generate_reply(TEST_PROMPT), None
    except AllProvidersFailedError as exc:
        return None, exc


def evaluate(target: str, recorders: list[CallRecorder], error: BaseException | None):
    """Pass/fail verdict for this run. Returns (passed, reason_if_failed)."""
    if target == "none":
        if error is None:
            return False, "Expected every provider to fail, but the manager returned a reply."
        if not isinstance(error, AllProvidersFailedError):
            return False, f"Expected AllProvidersFailedError, got {type(error).__name__}."
        for r in recorders:
            if not (r.invoked and r.kind == "fake" and not r.succeeded):
                return False, f"{DISPLAY_NAMES[r.name]} did not fail as expected."
        return True, None

    target_index = ORDER.index(target)
    target_recorder = recorders[target_index]

    for r in recorders[:target_index]:
        if not (r.invoked and r.kind == "fake" and not r.succeeded):
            return False, f"{DISPLAY_NAMES[r.name]} (before target) did not fail as expected."

    if error is not None:
        return False, f"All providers failed instead of {DISPLAY_NAMES[target]} succeeding: {error}"
    if not (target_recorder.invoked and target_recorder.kind == "real" and target_recorder.succeeded):
        return False, f"{DISPLAY_NAMES[target]} did not return a successful real response."

    return True, None


def status_line(recorder: CallRecorder) -> str:
    label = DISPLAY_NAMES[recorder.name]
    if not recorder.invoked:
        return f"{label}: NOT ATTEMPTED"
    if recorder.kind == "fake":
        return f"{label}: INTENTIONALLY FAILED"
    if recorder.succeeded:
        return f"{label}: REAL REQUEST -> SUCCESS"
    # Safe to print: every ProviderError message in this codebase is
    # deliberately built without keys/URLs/prompts (see
    # app/providers/manager.py's own docstrings/prints for the same
    # convention this line follows).
    return f"{label}: REAL REQUEST -> FAILURE ({recorder.error})"


def print_report(target: str, recorders: list[CallRecorder], reply, error) -> bool:
    print("=== Rhasspy AI Provider Fallback Integration Test ===")
    print()
    print(f"Target: {target}")
    print()
    for recorder in recorders:
        print(status_line(recorder))
    print()

    if target == "none":
        if isinstance(error, AllProvidersFailedError):
            print("AllProvidersFailedError: CONFIRMED")
        else:
            print("AllProvidersFailedError: NOT RAISED")
    elif reply is not None:
        print(f"Response: {reply!r}")
        if reply.strip() != EXPECTED_MARKER:
            print(
                "Note: response did not exactly match the expected marker - "
                "this is informational only, not a failure, since model "
                "wording can vary."
            )

    passed, reason = evaluate(target, recorders, error)
    print()
    if passed:
        print("Fallback test: PASS")
        if target != "none":
            print(f"Successful provider: {target}")
    else:
        print("Fallback test: FAIL")
        print(f"Reason: {reason}")

    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual integration test for the real AI provider fallback chain."
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=[*ORDER, "none"],
        help="Which provider should succeed (or 'none' to fail every provider).",
    )
    args = parser.parse_args(argv)

    recorders = build_recorders(args.target)
    reply, error = run_manager(recorders)
    passed = print_report(args.target, recorders, reply, error)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
