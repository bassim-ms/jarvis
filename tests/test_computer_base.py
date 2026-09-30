"""
Tests for the computer-control validation helpers/constants
(app/computer/base.py): `validate_url`/`validate_key`/`validate_key_combo`/
`normalize_key_input`/`validate_text`/`validate_scroll_amount`,
`ALLOWED_KEYS`/`MODIFIER_KEYS`/`ALLOWED_COMBO_TARGETS`, and
`AllowedApplication`. These are the SAME functions/constants both
app/computer/tools.py and WindowsComputerBackend use - proving them here
proves the one shared definition of "safe input" for every layer that
calls them.

No OS interaction anywhere in this file - these are pure functions over
plain Python values.
"""

import pytest

from app.computer.base import (
    ALLOWED_COMBO_TARGETS,
    ALLOWED_KEYS,
    MAX_COMBO_TOKENS,
    MAX_SCROLL_AMOUNT,
    MAX_TEXT_LENGTH,
    MODIFIER_KEYS,
    AllowedApplication,
    ComputerControlError,
    ScreenInfo,
    normalize_key_input,
    validate_coordinates,
    validate_key,
    validate_key_combo,
    validate_scroll_amount,
    validate_text,
    validate_url,
)


# --- validate_url --------------------------------------------------------


def test_validate_url_accepts_http():
    assert validate_url("http://example.com") == "http://example.com"


def test_validate_url_accepts_https():
    assert validate_url("https://example.com/path?q=1") == "https://example.com/path?q=1"


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/System32/cmd.exe",
        "javascript:alert(1)",
        "ftp://example.com",
        "ms-settings:",
        "custom-protocol://launch",
        "",
        "not a url",
        "example.com",  # no scheme
    ],
)
def test_validate_url_rejects_unsafe_or_malformed_urls(url):
    with pytest.raises(ComputerControlError):
        validate_url(url)


def test_validate_url_rejects_non_string():
    with pytest.raises(ComputerControlError):
        validate_url(None)  # type: ignore[arg-type]


# --- validate_key -----------------------------------------------------


def test_validate_key_accepts_every_allowed_key():
    for key in ALLOWED_KEYS:
        assert validate_key(key) == key


@pytest.mark.parametrize(
    "key",
    ["", "rm -rf /", "alt+f4", "ctrl+alt+delete", "a", "1", None, 123],
)
def test_validate_key_rejects_anything_not_in_the_allowlist(key):
    with pytest.raises(ComputerControlError):
        validate_key(key)


# --- validate_key: case-insensitivity (Fix A) -----------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ENTER", "enter"),
        ("Enter", "enter"),
        ("eNtEr", "enter"),
        ("TAB", "tab"),
        ("Esc", "esc"),
        ("ESCAPE", "escape"),
        ("F5", "f5"),
        ("Up", "up"),
        ("CAPSLOCK", "capslock"),
    ],
)
def test_validate_key_is_case_insensitive_and_returns_the_canonical_lowercase_form(
    raw, expected
):
    assert validate_key(raw) == expected


def test_validate_key_case_insensitivity_covers_the_whole_allowlist():
    """Every member of ALLOWED_KEYS, in every case variant, still
    validates and normalizes to the same canonical lowercase form."""
    for key in ALLOWED_KEYS:
        assert validate_key(key.upper()) == key
        assert validate_key(key.capitalize()) == key


@pytest.mark.parametrize("key", ["ENTERX", "TABS", "NOTAKEY", "CTRL+L", "ALT+F4"])
def test_validate_key_case_insensitivity_does_not_widen_the_allowlist(key):
    """Normalizing case must never accept something that wasn't already
    a valid key name - it only folds case, it doesn't fuzzy-match, and a
    combination string is still never a valid STANDALONE key."""
    with pytest.raises(ComputerControlError):
        validate_key(key)


# --- validate_key_combo / normalize_key_input (Fix B) ---------------------


def test_modifier_keys_and_combo_targets_are_the_approved_closed_sets():
    """Locks in the exact sets approved for Fix B - a change to either
    must be a deliberate, reviewed decision, not an accidental drift."""
    assert MODIFIER_KEYS == {"ctrl", "alt", "shift", "win"}
    assert MAX_COMBO_TOKENS == 3
    # Every lowercase letter/digit, plus every ALLOWED_KEYS name EXCEPT
    # the four modifiers themselves (a modifier is never a valid final
    # target - see test_validate_key_combo_rejects_a_combo_of_only_modifiers).
    assert ALLOWED_COMBO_TARGETS == frozenset("abcdefghijklmnopqrstuvwxyz0123456789") | (
        ALLOWED_KEYS - MODIFIER_KEYS
    )
    assert not (MODIFIER_KEYS & ALLOWED_COMBO_TARGETS)


@pytest.mark.parametrize(
    "combo,expected_tokens",
    [
        ("ctrl+l", ("ctrl", "l")),
        ("CTRL+L", ("ctrl", "l")),
        ("Ctrl+L", ("ctrl", "l")),
        ("ctrl+c", ("ctrl", "c")),
        ("ctrl+shift+esc", ("ctrl", "shift", "esc")),
        ("alt+tab", ("alt", "tab")),
        ("win+d", ("win", "d")),
        ("ctrl+enter", ("ctrl", "enter")),
        ("ctrl+0", ("ctrl", "0")),
        ("shift+f5", ("shift", "f5")),
    ],
)
def test_validate_key_combo_accepts_the_approved_examples(combo, expected_tokens):
    assert validate_key_combo(combo) == expected_tokens


@pytest.mark.parametrize(
    "combo",
    [
        "l+ctrl",  # no leading modifier
        "ctrl+",  # trailing '+' -> empty token
        "+l",  # leading '+' -> empty token
        "ctrl++l",  # doubled '+' -> empty token
        "ctrl+shift+alt+l",  # 4 tokens, exceeds MAX_COMBO_TOKENS
        "banana+l",  # unknown modifier token
        "ctrl+alt",  # two modifiers, no real target
        "ctrl+ctrl",  # target token is itself a modifier
        "ctrl+banana",  # unknown target token
        "ctrl+/bin/sh",  # path-shaped target token
        "ctrl+powershell",  # command-shaped target token (>1 char, not in ALLOWED_KEYS)
        "",  # empty string
        "+",  # bare separator
        "ctrl",  # no separator at all - not a combo (single token)
    ],
)
def test_validate_key_combo_rejects_every_disallowed_example(combo):
    with pytest.raises(ComputerControlError):
        validate_key_combo(combo)


def test_validate_key_combo_rejects_non_string():
    with pytest.raises(ComputerControlError):
        validate_key_combo(None)  # type: ignore[arg-type]


def test_validate_key_combo_never_accepts_shell_path_or_url_like_strings():
    """A combo can only ever carry a modifier prefix plus one
    single-character/named-key target - it can never smuggle an
    executable command, shell metacharacter sequence, filesystem path,
    or URL, because every token is checked against a closed allowlist,
    never treated as free text."""
    dangerous = [
        "ctrl+rm -rf /",
        "ctrl+C:\\Windows\\System32\\cmd.exe",
        "ctrl+http://evil.example.com",
        "ctrl+; shutdown /s",
        "ctrl+$(whoami)",
        "ctrl+`whoami`",
    ]
    for combo in dangerous:
        with pytest.raises(ComputerControlError):
            validate_key_combo(combo)


# --- normalize_key_input: the single/combo dispatcher ---------------------


def test_normalize_key_input_dispatches_standalone_keys():
    assert normalize_key_input("enter") == ("enter",)
    assert normalize_key_input("ENTER") == ("enter",)


def test_normalize_key_input_dispatches_combinations():
    assert normalize_key_input("ctrl+l") == ("ctrl", "l")
    assert normalize_key_input("CTRL+SHIFT+ESC") == ("ctrl", "shift", "esc")


def test_normalize_key_input_rejects_invalid_standalone_keys():
    with pytest.raises(ComputerControlError):
        normalize_key_input("banana")


def test_normalize_key_input_rejects_invalid_combinations():
    with pytest.raises(ComputerControlError):
        normalize_key_input("ctrl+alt")


# --- validate_text -----------------------------------------------------


def test_validate_text_accepts_ordinary_text():
    assert validate_text("hello world") == "hello world"


def test_validate_text_accepts_text_at_the_max_length():
    text = "a" * MAX_TEXT_LENGTH
    assert validate_text(text) == text


def test_validate_text_rejects_empty_string():
    with pytest.raises(ComputerControlError):
        validate_text("")


def test_validate_text_rejects_text_over_the_max_length():
    with pytest.raises(ComputerControlError):
        validate_text("a" * (MAX_TEXT_LENGTH + 1))


def test_validate_text_rejects_non_string():
    with pytest.raises(ComputerControlError):
        validate_text(12345)  # type: ignore[arg-type]


# --- validate_scroll_amount -------------------------------------------


@pytest.mark.parametrize("amount", [0, 1, -1, MAX_SCROLL_AMOUNT, -MAX_SCROLL_AMOUNT])
def test_validate_scroll_amount_accepts_in_range_integers(amount):
    assert validate_scroll_amount(amount) == amount


@pytest.mark.parametrize(
    "amount", [MAX_SCROLL_AMOUNT + 1, -MAX_SCROLL_AMOUNT - 1, 1.5, "10", None, True, False]
)
def test_validate_scroll_amount_rejects_out_of_range_or_non_integers(amount):
    with pytest.raises(ComputerControlError):
        validate_scroll_amount(amount)


# --- validate_coordinates -----------------------------------------------


_SCREEN = ScreenInfo(width=1920, height=1080)


@pytest.mark.parametrize("x,y", [(0, 0), (1919, 1079), (960, 540)])
def test_validate_coordinates_accepts_in_bounds_points(x, y):
    assert validate_coordinates(x, y, _SCREEN) == (x, y)


@pytest.mark.parametrize(
    "x,y",
    [
        (-1, 0),
        (0, -1),
        (1920, 0),  # width itself is out of bounds (0-indexed)
        (0, 1080),  # height itself is out of bounds
        (999999, 999999),
        (-999999, -999999),
    ],
)
def test_validate_coordinates_rejects_out_of_bounds_points(x, y):
    with pytest.raises(ComputerControlError):
        validate_coordinates(x, y, _SCREEN)


@pytest.mark.parametrize(
    "x,y",
    [(1.5, 100), (100, 1.5), ("100", 100), (100, None), (True, 100), (100, False)],
)
def test_validate_coordinates_rejects_non_integer_values(x, y):
    with pytest.raises(ComputerControlError):
        validate_coordinates(x, y, _SCREEN)


# --- AllowedApplication ---------------------------------------------------


def test_allowed_application_is_a_small_fixed_set():
    values = {app.value for app in AllowedApplication}
    assert values == {
        "notepad",
        "calculator",
        "file_explorer",
        "paint",
        "chrome",
        "edge",
        "firefox",
    }


def test_allowed_application_rejects_arbitrary_names():
    with pytest.raises(ValueError):
        AllowedApplication("C:/Windows/System32/cmd.exe")


def test_allowed_application_rejects_shell_metacharacters():
    with pytest.raises(ValueError):
        AllowedApplication("notepad.exe & calc.exe")
