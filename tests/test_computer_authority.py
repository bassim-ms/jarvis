"""
Authorization grants, the DATA half (app/computer/authority.py): the canonical form of an input action, the keyed (HMAC) match, and the grant object itself.
Pure and offline: nothing here touches a gate, a backend, a desktop or a clock.
"""

import copy
import pickle

import pytest

from app.computer.authority import (
    GRANTABLE_ACTIONS,
    MAX_GRANT_ENTRIES,
    MAX_GRANT_SECONDS,
    AuthorityError,
    InputGrant,
    InputStep,
    canonical_input,
    check_label,
)
from app.computer.base import AllowedApplication, MouseButton, normalize_key_input
from app.computer.gate import GATED_METHODS

# One valid example of every grantable action, exactly as a backend presents it after validating and normalizing its arguments.
EXAMPLES = {
    "open_application": (AllowedApplication.NOTEPAD,),
    "type_text": ("hello",),
    "press_key": (normalize_key_input("ctrl+n"),),
    "move_mouse": (10, 20),
    "click_mouse": (10, 20, MouseButton.LEFT),
    "double_click": (10, 20, MouseButton.LEFT),
    "scroll": (-300,),
    "open_url": ("https://example.com/a?x=1#top",),  # the exact URL string
    "switch_window": ("Untitled - Notepad", "notepad.exe"),  # the exact window title and the exact (lower-cased) process file name
}


def test_every_grantable_action_has_an_example_and_the_set_equals_the_gates_gated_methods():
    assert set(EXAMPLES) == set(GRANTABLE_ACTIONS) == set(GATED_METHODS)


@pytest.mark.parametrize("action", sorted(EXAMPLES))
def test_the_canonical_form_is_deterministic_and_tied_to_the_action(action):
    first = canonical_input(action, *EXAMPLES[action])

    assert canonical_input(action, *EXAMPLES[action]) == first and isinstance(first, bytes)
    assert len({canonical_input(a, *args) for a, args in EXAMPLES.items()}) == len(EXAMPLES)  # no two actions share a form


@pytest.mark.parametrize(
    "action, a, b",
    [
        ("type_text", ("hello",), ("hellp",)),
        ("type_text", ("hello",), ("hello ",)),
        ("type_text", ("hello",), ("Hello",)),
        ("press_key", (("ctrl", "n"),), (("ctrl", "s"),)),
        ("press_key", (("ctrl", "n"),), (("n", "ctrl"),)),
        ("press_key", (("enter",),), (("enter", "enter"),)),
        ("move_mouse", (10, 20), (10, 21)),
        ("move_mouse", (10, 20), (20, 10)),
        ("click_mouse", (10, 20, MouseButton.LEFT), (10, 20, MouseButton.RIGHT)),
        ("click_mouse", (10, 20, MouseButton.LEFT), (None, None, MouseButton.LEFT)),
        ("double_click", (10, 20, MouseButton.LEFT), (10, 20, MouseButton.MIDDLE)),
        ("scroll", (5,), (-5,)),
        ("open_application", (AllowedApplication.NOTEPAD,), (AllowedApplication.CALCULATOR,)),
    ],
)
def test_any_change_to_any_argument_changes_the_canonical_form(action, a, b):
    assert canonical_input(action, *a) != canonical_input(action, *b)


def test_the_encoding_is_typed_and_length_prefixed_so_fields_cannot_be_confused_or_merged():
    assert canonical_input("press_key", ("ab", "c")) != canonical_input("press_key", ("a", "bc"))  # where a field ends matters
    assert canonical_input("type_text", "1") != canonical_input("scroll", 1)
    assert canonical_input("type_text", "1") != canonical_input("type_text", ("1",))  # a string is not a one-key list
    assert canonical_input("click_mouse", None, None, MouseButton.LEFT) != canonical_input("click_mouse", "none", "none", MouseButton.LEFT)
    assert canonical_input("open_application", AllowedApplication.NOTEPAD) != canonical_input("open_application", "notepad")  # the enum is not the bare string
    assert canonical_input("press_key", ["ctrl", "n"]) == canonical_input("press_key", ("ctrl", "n"))  # a list and a tuple are the same key sequence


def test_unicode_text_is_encoded_exactly_and_never_normalized_or_dropped():
    assert canonical_input("type_text", "café") != canonical_input("type_text", "café")  # precomposed vs combining: different keystrokes, different form
    assert canonical_input("type_text", "‮") != canonical_input("type_text", "")


@pytest.mark.parametrize(
    "bad",
    [
        ("sudo_type", ("x",)),
        ("switch_window", ("Notes",)),  # a window is bound by its title AND its process: one argument is the wrong number
        ("switch_window", ("Notes", "notes.exe", "extra")),
        ("switch_window", ("Notes", True)),  # a bool is not a process name
        ("open_url", ()),
        ("open_url", ("https://example.com", "https://example.org")),
        ("type_text", ()),
        ("type_text", ("a", "b")),
        ("move_mouse", (1,)),
        ("click_mouse", (1, 2)),
        ("scroll", (True,)),  # a bool is not a number here
        ("move_mouse", (1.5, 2)),
        ("type_text", (b"bytes",)),
        ("type_text", (object(),)),
        ("press_key", ([1, 2],)),
    ],
)
def test_anything_that_cannot_be_authorized_is_refused_not_guessed(bad):
    action, args = bad

    with pytest.raises(AuthorityError):
        canonical_input(action, *args)


def test_an_input_step_is_just_the_canonical_form_of_its_action_and_arguments():
    step = InputStep("type_text", ("hello",))

    assert step.canonical() == canonical_input("type_text", "hello")
    with pytest.raises(AuthorityError):
        InputStep("nope", ()).canonical()
    with pytest.raises(Exception):
        step.action = "press_key"  # frozen


def test_labels_are_short_lowercase_vocabulary_so_a_log_line_cannot_be_injected_through_them():
    assert check_label("notepad_demo", "the purpose") == "notepad_demo" and check_label("cli", "the actor") == "cli"
    for bad in ("", "Has Upper", "two words", "line\nbreak", "x" * 41, "-leading", "tab\t", "secret=abc def", None, 5):
        with pytest.raises(AuthorityError):
            check_label(bad, "the label")


def test_the_limits_are_what_the_milestone_says():
    assert MAX_GRANT_SECONDS == 300.0 and MAX_GRANT_ENTRIES == 32


# ======================================================== the grant object ====================================================================


def test_a_grant_has_an_unguessable_identity_and_key_that_are_never_reused():
    grants = [InputGrant() for _ in range(500)]

    assert all(len(g.id) == 32 and int(g.id, 16) >= 0 for g in grants)  # 128 random bits, hex
    assert len({g.id for g in grants}) == 500 and len({g._key for g in grants}) == 500
    assert all(len(g._key) == 32 for g in grants)  # 256 random bits
    assert grants[0].short_id == grants[0].id[:8]


def test_the_keyed_match_is_deterministic_per_grant_and_different_across_grants_and_inputs():
    one, two = InputGrant(), InputGrant()
    text = canonical_input("type_text", "hello")

    assert one._mac(text) == one._mac(text) and len(one._mac(text)) == 32
    assert one._mac(text) != two._mac(text)  # the key differs per grant, so a digest from one grant means nothing for another
    assert one._mac(text) != one._mac(canonical_input("type_text", "hellp"))


def test_a_grants_repr_and_str_show_only_a_short_prefix_never_the_key_or_the_full_id():
    grant = InputGrant()
    shown = repr(grant) + str(grant) + f"{grant}" + format(grant, "")

    assert grant.short_id in shown and grant.id not in shown and grant._key.hex() not in shown
    assert not hasattr(grant, "__dict__")  # no attribute bag to dump, and nothing is added to it by accident


def test_a_grant_cannot_be_copied_or_pickled_so_a_duplicate_credential_cannot_be_made_by_accident():
    grant = InputGrant()

    for attempt in (lambda: copy.copy(grant), lambda: copy.deepcopy(grant), lambda: pickle.dumps(grant), lambda: pickle.dumps(grant, protocol=0)):
        with pytest.raises(TypeError):
            attempt()


# ============================================ text that cannot be encoded is an AuthorityError, never a raw UnicodeEncodeError ===========================

LONE_SURROGATES = [
    ("type_text", ("\ud800",)),
    ("type_text", ("abc\udfffdef",)),  # buried in the middle of otherwise ordinary text
    ("type_text", ("ok\udc00",)),
    ("open_url", ("https://example.com/a\ud800",)),
    ("switch_window", ("\ud800", "notepad.exe")),
    ("switch_window", ("Untitled - Notepad", "\udc00.exe")),
    ("press_key", (("ctrl", "\ud800"),)),
    ("press_key", (["\ud800"],)),
]


@pytest.mark.parametrize("action, args", LONE_SURROGATES, ids=lambda value: ascii(value)[:40])
def test_text_that_cannot_be_encoded_is_the_same_authority_error_as_any_other_unauthorizable_argument(action, args):
    with pytest.raises(AuthorityError) as caught:
        canonical_input(action, *args)

    assert type(caught.value) is AuthorityError  # exactly the expected type, not a UnicodeEncodeError (which is not an AuthorityError)
    assert not isinstance(caught.value, UnicodeError)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__ is True  # the original error (which names the character) is not chained on
    message = str(caught.value)
    assert not any(0xD800 <= ord(character) <= 0xDFFF for character in message) and "ud800" not in message.lower() and "udc00" not in message.lower()  # the value is never echoed


def test_an_input_step_with_unencodable_text_cannot_be_turned_into_an_entry():
    with pytest.raises(AuthorityError):
        InputStep("type_text", ("a\ud800b",)).canonical()


@pytest.mark.parametrize(
    "action, args, digest",
    [
        ("type_text", ("caf\u00e9 \U0001F600",), "6a61727669732e636f6d70757465722e696e7075742e76310000000009747970655f74657874730000000a636166c3a920f09f9880"),
        ("press_key", (("ctrl", "\u00e9"),), "6a61727669732e636f6d70757465722e696e7075742e7631000000000970726573735f6b6579740002000000046374726c00000002c3a9"),
        (
            "open_url", ("https://ex\u00e4mple.com/\u00fc?q=\U0001F600#top",),
            "6a61727669732e636f6d70757465722e696e7075742e763100000000086f70656e5f75726c730000002268747470733a2f2f6578c3a46d706c652e636f6d2fc3bc3f713df09f988023746f70",
        ),
        (
            "switch_window", ("Fen\u00eatre \u2014 Bloc-notes \U0001F600", "notepad.exe"),
            "6a61727669732e636f6d70757465722e696e7075742e7631000000000d7377697463685f77696e646f77730000001c46656ec3aa74726520e2809420426c6f632d6e6f74657320f09f9880730000000b6e6f74657061642e657865",
        ),
    ],
    ids=["type_text", "press_key", "open_url", "switch_window"],
)
def test_valid_non_ascii_text_still_encodes_to_exactly_the_bytes_it_always_did(action, args, digest):
    """Captured from the code BEFORE the fix: accents, an em dash and an astral-plane character (one code point, two UTF-16 units) are valid text and are unchanged."""
    assert canonical_input(action, *args).hex() == digest
