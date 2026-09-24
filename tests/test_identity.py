"""
Tests for app/identity.py: loading personality.md/user_profile.md as
plain, static prompt context - never executable instructions.

Covers: loading real repo files, missing files failing safely,
malformed files failing safely, and that secrets are never expected/
required. See tests/test_core_identity_integration.py for how this
context actually reaches JarvisCore's DIRECT prompt, and for the
permission/confirmation/tool-execution security-boundary tests.
"""

import os

from app.identity import load_identity_context

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_PERSONALITY_PATH = os.path.join(_REPO_ROOT, "personality.md")
_REAL_USER_PROFILE_PATH = os.path.join(_REPO_ROOT, "user_profile.md")


# -- 1/2: personality.md and user_profile.md load correctly -----------------


def test_real_personality_file_loads_and_is_included():
    context = load_identity_context(_REAL_PERSONALITY_PATH, _REAL_USER_PROFILE_PATH)
    assert "Jarvis's personality:" in context
    assert "helpful" in context.lower()


def test_real_user_profile_file_loads_and_is_included():
    context = load_identity_context(_REAL_PERSONALITY_PATH, _REAL_USER_PROFILE_PATH)
    assert "About the user:" in context
    assert "Bassim" in context
    assert "robotics" in context.lower()


def test_real_files_produce_a_single_combined_string_with_the_preamble():
    context = load_identity_context(_REAL_PERSONALITY_PATH, _REAL_USER_PROFILE_PATH)
    assert context.startswith("The following is Jarvis's own background personality")
    # Personality comes before the user profile, matching this module's
    # own fixed section order.
    assert context.index("Jarvis's personality:") < context.index("About the user:")


def test_loading_is_a_pure_function_of_its_inputs(tmp_path):
    personality_path = tmp_path / "personality.md"
    profile_path = tmp_path / "user_profile.md"
    personality_path.write_text("Be concise.", encoding="utf-8")
    profile_path.write_text("Name: Test User", encoding="utf-8")

    first = load_identity_context(str(personality_path), str(profile_path))
    second = load_identity_context(str(personality_path), str(profile_path))
    assert first == second
    assert "Be concise." in first
    assert "Name: Test User" in first


# -- 3: missing files fail safely --------------------------------------------


def test_missing_personality_file_degrades_to_only_user_profile(tmp_path):
    profile_path = tmp_path / "user_profile.md"
    profile_path.write_text("Name: Test User", encoding="utf-8")

    context = load_identity_context(str(tmp_path / "does_not_exist.md"), str(profile_path))

    assert "Jarvis's personality:" not in context
    assert "About the user:" in context
    assert "Name: Test User" in context


def test_missing_user_profile_file_degrades_to_only_personality(tmp_path):
    personality_path = tmp_path / "personality.md"
    personality_path.write_text("Be concise.", encoding="utf-8")

    context = load_identity_context(
        str(personality_path), str(tmp_path / "does_not_exist.md")
    )

    assert "About the user:" not in context
    assert "Jarvis's personality:" in context


def test_both_files_missing_returns_empty_string(tmp_path):
    context = load_identity_context(
        str(tmp_path / "no_personality.md"), str(tmp_path / "no_profile.md")
    )
    assert context == ""


def test_missing_files_never_raise(tmp_path):
    # Never raises - a completely bogus path is still handled safely.
    load_identity_context("/definitely/does/not/exist.md", "/also/missing.md")


# -- 4: malformed files fail safely ------------------------------------------


def test_directory_instead_of_file_fails_safely(tmp_path):
    """A misconfigured path pointing at a directory (not a file) must
    degrade to no content, never raise (IsADirectoryError on most
    platforms, PermissionError on Windows - both are plain OSError
    subclasses, caught the same way)."""
    directory_path = tmp_path / "a_directory"
    directory_path.mkdir()

    context = load_identity_context(str(directory_path), str(tmp_path / "missing.md"))
    assert context == ""


def test_invalid_utf8_bytes_fail_safely(tmp_path):
    personality_path = tmp_path / "personality.md"
    personality_path.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")

    context = load_identity_context(str(personality_path), str(tmp_path / "missing.md"))
    assert context == ""


def test_one_malformed_file_does_not_block_the_other(tmp_path):
    bad_personality_path = tmp_path / "personality.md"
    bad_personality_path.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")
    good_profile_path = tmp_path / "user_profile.md"
    good_profile_path.write_text("Name: Test User", encoding="utf-8")

    context = load_identity_context(str(bad_personality_path), str(good_profile_path))

    assert "Jarvis's personality:" not in context
    assert "About the user:" in context
    assert "Name: Test User" in context


def test_empty_file_contributes_nothing(tmp_path):
    empty_path = tmp_path / "personality.md"
    empty_path.write_text("   \n\n  ", encoding="utf-8")
    profile_path = tmp_path / "user_profile.md"
    profile_path.write_text("Name: Test User", encoding="utf-8")

    context = load_identity_context(str(empty_path), str(profile_path))
    assert "Jarvis's personality:" not in context
    assert "About the user:" in context


# -- 8: secrets are not expected or required ---------------------------------


def test_loading_never_requires_any_secret_or_credential(monkeypatch, tmp_path):
    """No env var, API key, or credential is read by this module at
    all - proven by clearing the environment entirely and confirming
    loading still works from plain files."""
    monkeypatch.setattr(os, "environ", {})
    personality_path = tmp_path / "personality.md"
    personality_path.write_text("Be concise.", encoding="utf-8")

    context = load_identity_context(str(personality_path), str(tmp_path / "missing.md"))
    assert "Be concise." in context


def test_real_files_never_leak_any_configured_secret_value():
    """Defensive content check on the actual repo files (not just the
    loader) - guards against a future edit accidentally pasting a real
    configured secret VALUE into either Markdown file. Deliberately
    checks for leaked secret *values*, not the word "secret" itself -
    user_profile.md legitimately tells future editors not to add
    secrets, which would false-positive on a naive keyword ban."""
    from app.config import settings

    context = load_identity_context(_REAL_PERSONALITY_PATH, _REAL_USER_PROFILE_PATH)
    configured_secrets = [
        settings.gemini_api_key,
        settings.groq_api_key,
        settings.openrouter_api_key,
        settings.cloudflare_api_key,
        settings.freellmapi_api_key,
        settings.google_client_secret,
    ]
    for secret in configured_secrets:
        if secret:  # only a real, non-empty configured value is meaningful to check
            assert secret not in context
