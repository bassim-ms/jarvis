"""
The write policy: a PURE, deterministic decision - no store, no I/O, no AI.

    EXPLICIT + LOW RISK            -> may persist automatically
    UNCERTAIN / INFERRED           -> candidate or temporary (never a fact)
    SENSITIVE / CONSEQUENTIAL /
    AMBIGUOUS                      -> ask (nothing persisted until the user says more)
    SECRET                         -> never stored, whoever asks
    EXTERNAL CONTENT               -> data, never authority: it cannot write

The order below is the whole policy, and the first rule that applies wins, so
a stronger protection can never be overridden by a weaker signal (an explicit
"remember" cannot store a password; a confident inference cannot contradict
what the user said; text from a web page cannot write at all).

Whether something is a secret or sensitive is decided by patterns over the
words, not by asking a model. It is deliberately conservative: a false positive
costs the user one sentence ("remember anyway: ..."); a false negative would
persist something that must not be.
"""

from __future__ import annotations

import re

from app.memory_policy.base import (
    CANDIDATE_CONFIDENCE_CEILING,
    COMMAND_CONFIDENCE,
    DEFAULT_CANDIDATE_CONFIDENCE,
    MAX_KEY_CHARS,
    MAX_VALUE_CHARS,
    STATEMENT_CONFIDENCE,
    Destination,
    Kind,
    Origin,
    Sensitivity,
    WriteDecision,
    WriteProposal,
)
from app.world_model.base import EntitySource

_I = re.IGNORECASE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


# --- secrets: NEVER stored ------------------------------------------------------------------------------------------------------

_SECRET_WORDS = _rx(
    r"\b(?:password|passwd|passcode|passphrase|pin\s+(?:is|code|number)|api[\s_-]?key|secret(?:\s+key)?|"
    r"(?:access|auth(?:entication)?|refresh|bearer|api|secret)[\s_-]?token|bearer\s+\S|private[\s_-]?key|"
    r"seed\s+phrase|recovery\s+(?:code|phrase|key)|cvv|cvc|card\s+number|credit\s+card|debit\s+card|iban|"
    r"routing\s+number|social\s+security|ssn|otp|one[\s-]time\s+(?:code|password)|2fa\s+code|"
    r"login\s+credentials?|credentials?)\b"
)
_SECRET_SHAPES = (
    _rx(r"\bsk-[A-Za-z0-9_-]{16,}"),
    _rx(r"\bAKIA[0-9A-Z]{16}\b"),
    _rx(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    _rx(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    _rx(r"\bAIza[0-9A-Za-z_-]{20,}"),
    _rx(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\."),
    _rx(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_=-]{32,}")
_DIGIT_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        n = int(char)
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _looks_like_secret(text: str) -> bool:
    if _SECRET_WORDS.search(text) or any(p.search(text) for p in _SECRET_SHAPES):
        return True
    for token in _LONG_TOKEN.findall(text):
        if re.search(r"\d", token) and re.search(r"[A-Za-z]", token):
            return True  # a long unbroken mixed token: a key or a hash, not a word
    for run in _DIGIT_RUN.findall(text):
        digits = re.sub(r"\D", "", run)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True  # a card number
    return False


# A secret WORD followed by what is plainly a VALUE: "wifi password is hunter2", "PIN: 4821", "api key = sk9...".
_ASSIGNED_SECRET = _rx(
    r"\b(?P<word>password|passwd|passcode|passphrase|pin(?:\s+code)?|otp|cvv|cvc|token|credentials?|secret(?:\s+key)?|"
    r"api[\s_-]?key|(?:access|auth|refresh|bearer|api|secret)[\s_-]?token|private[\s_-]?key)"
    r"\s*(?:is|=|:)\s*[\"'`]?(?P<value>[^\s\"'`,;]{4,})"
)
_NUMERIC_CODE_WORDS = frozenset({"pin", "pin code", "otp", "passcode", "cvv", "cvc"})
_BEARER_VALUE = _rx(r"\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")


def _looks_like_secret_value(word: str, value: str) -> bool:
    if re.fullmatch(r"\d{4,8}", value) and re.sub(r"\s+", " ", word.lower()) in _NUMERIC_CODE_WORDS:
        return True
    if re.fullmatch(r"\d{6,}", value):
        return True
    has_digit, has_alpha = bool(re.search(r"\d", value)), bool(re.search(r"[A-Za-z]", value))
    if len(value) >= 6 and has_digit and has_alpha:
        return True  # "hunter2", "Summer2024!"
    return len(value) >= 8 and has_alpha and bool(re.search(r"[!@#$%^&*+=]", value))


def contains_secret_value(text: str) -> bool:
    """Reasonable evidence that `text` holds an actual credential VALUE - not merely a word like "password".

    For DISPLAYING ordinary text (a calendar title, a task, a reminder) where hiding everything that mentions
    "password" would make it useless. Deliberately narrower than `classify_sensitivity` (which keeps memory
    conservative): true for the known key/token shapes (sk-..., AKIA..., ghp_..., JWTs, private-key blocks,
    bearer tokens), a long unbroken mixed token, a card number, and a secret word directly followed by a
    value-shaped string ("password is hunter2", "PIN: 4821"). "Reset the wifi password", "Password policy
    review", "Rotate the API key on Friday" are not."""
    if not isinstance(text, str) or not text:
        return False
    if any(p.search(text) for p in _SECRET_SHAPES) or _BEARER_VALUE.search(text):
        return True
    for token in _LONG_TOKEN.findall(text):
        if re.search(r"\d", token) and re.search(r"[A-Za-z]", token):
            return True
    for run in _DIGIT_RUN.findall(text):
        digits = re.sub(r"\D", "", run)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return any(_looks_like_secret_value(m.group("word"), m.group("value")) for m in _ASSIGNED_SECRET.finditer(text))


# --- sensitive: only with the user's say-so ------------------------------------------------------------------------------------

_SENSITIVE = _rx(
    r"\b(?:diagnos\w*|medical|medication|prescription|illness|disease|cancer|diabet\w*|depress\w*|anxiety|"
    r"therapy|therapist|pregnan\w*|hiv|surgery|disabilit\w*|"
    r"religio\w*|muslim|christian|jewish|hindu|buddhist|atheist|"
    r"political\w*|voted\s+for|"
    r"sexual\w*|gay|lesbian|bisexual|transgender|"
    r"immigration|visa\s+status|passport|national\s+id|emirates\s+id|driver'?s?\s+licen[cs]e|tax\s+id|"
    r"salary|income|net\s+worth|debt|loan|mortgage|bank\s+account|account\s+balance|savings|credit\s+score|"
    r"home\s+address|street\s+address|phone\s+number|email\s+address|"
    r"criminal|arrest\w*|convict\w*)\b"
)


def classify_sensitivity(*texts: str) -> Sensitivity:
    """SECRET beats SENSITIVE beats NONE, over every text given."""
    joined = " ".join(t for t in texts if t).replace("_", " ")  # a key is made of words joined by "_"
    if _looks_like_secret(joined):
        return Sensitivity.SECRET
    if _SENSITIVE.search(joined):
        return Sensitivity.SENSITIVE
    return Sensitivity.NONE


# --- the decision --------------------------------------------------------------------------------------------------------------

_BLANKS = re.compile(r"\s+")


def normalize(value: str | None) -> str:
    """The form two values are compared in: case-folded, spaces collapsed."""
    return _BLANKS.sub(" ", (value or "").strip()).casefold()


def _is_clean_line(text: str, limit: int) -> bool:
    return (
        isinstance(text, str)
        and bool(text.strip())
        and len(text) <= limit
        and "\n" not in text
        and "\r" not in text
        and "\x00" not in text
    )


class WritePolicy:
    """See the module docstring. Stateless."""

    def decide(self, proposal: WriteProposal, existing_value: str | None = None) -> WriteDecision:
        """`existing_value`: what is already recorded for this proposal's key
        (None if nothing), looked up by the caller."""
        p = proposal
        # 1. shape: one short clean line, or nothing
        if not _is_clean_line(p.value, MAX_VALUE_CHARS) or not _is_clean_line(p.key, MAX_KEY_CHARS):
            return WriteDecision(Destination.REJECT, "the text is empty, too long, or not a single line")

        # 2. secrets: never, by anyone, through any door
        sensitivity = classify_sensitivity(p.key, p.value, p.evidence, p.project)
        if sensitivity is Sensitivity.SECRET:
            return WriteDecision(Destination.REJECT, "it looks like a credential or secret", sensitivity)

        # 3. external content is data, never authority
        if p.origin is Origin.EXTERNAL:
            return WriteDecision(
                Destination.REJECT, "text from an external source cannot write to memory", sensitivity
            )

        # 4. sensitive: never inferred; from the user, only when they said to keep it anyway
        if sensitivity is Sensitivity.SENSITIVE:
            if p.origin is Origin.INFERENCE:
                return WriteDecision(Destination.REJECT, "sensitive information is never inferred", sensitivity)
            if not (p.origin is Origin.USER_COMMAND and p.acknowledged_sensitive):
                return WriteDecision(
                    Destination.ASK, "it looks sensitive, so it needs the user's say-so", sensitivity
                )

        same = existing_value is not None and normalize(existing_value) == normalize(p.value)
        conflicts = existing_value is not None and not same

        # 5. an inference is only ever a candidate - and never against what the user said
        if p.origin is Origin.INFERENCE:
            if conflicts:
                return WriteDecision(
                    Destination.REJECT,
                    "an inference cannot contradict what the user has said",
                    sensitivity,
                    existing_value=existing_value,
                )
            if same:
                return WriteDecision(Destination.TEMPORARY, "already known", sensitivity, existing_value=existing_value)
            return WriteDecision(
                Destination.CANDIDATE,
                "inferred, not stated",
                sensitivity,
                source=EntitySource.INFERENCE,
                confidence=self._candidate_confidence(p),
            )

        # 6. said uncertainly: staged as a candidate, not persisted
        if p.hedged:
            return WriteDecision(
                Destination.CANDIDATE,
                "said uncertainly",
                sensitivity,
                source=EntitySource.USER,
                confidence=self._candidate_confidence(p),
                existing_value=existing_value,
            )

        # 7. about now, not a lasting truth: usable this turn, not worth keeping
        if p.temporary and p.origin is Origin.USER_STATEMENT:
            return WriteDecision(Destination.TEMPORARY, "it is about right now", sensitivity)

        # 8. it contradicts what is recorded
        if conflicts and p.origin is Origin.USER_STATEMENT and not p.supersede:
            return WriteDecision(
                Destination.ASK,
                "it differs from what was said before",
                sensitivity,
                existing_value=existing_value,
            )

        # 9. clear, low-risk, explicit: persist, with provenance
        command = p.origin is Origin.USER_COMMAND
        destination = Destination.MEMORY if p.kind is Kind.FACT else Destination.WORLD_MODEL
        return WriteDecision(
            destination,
            "already recorded" if same else ("updated" if conflicts else "stated by the user"),
            sensitivity,
            source=EntitySource.USER,
            confidence=COMMAND_CONFIDENCE if command else STATEMENT_CONFIDENCE,
            existing_value=existing_value,
            updates_existing=conflicts,
        )

    @staticmethod
    def _candidate_confidence(p: WriteProposal) -> float:
        value = DEFAULT_CANDIDATE_CONFIDENCE if p.confidence is None else p.confidence
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = DEFAULT_CANDIDATE_CONFIDENCE
        return max(0.0, min(float(value), CANDIDATE_CONFIDENCE_CEILING))
