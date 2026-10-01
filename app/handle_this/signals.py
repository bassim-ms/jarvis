"""
Deterministic text signals for Handle This: small, fixed regular expressions
that read a request's OWN words. No AI, no I/O, no state.

They answer narrow questions - which broad families of work does this wording
point at, does it ask for something outside the boundary, which constraints
does it state, is it about choosing something to buy, does it name a subject,
does a reminder in it have a time - and nothing else. They never produce a
tool name, a step or an action.

Two deliberate biases, both toward safety:

- The UNSUPPORTED scan is conservative. It would rather refuse "remind me to
  email the professor" (the user can rephrase) than miss "email the professor".
- A request with no recognizable signal is reported as such (an empty family
  set), never guessed into a family.
"""

from __future__ import annotations

import re

from app.handle_this.base import (
    Constraint,
    ConstraintSource,
    GoalFamily,
    Topic,
    UnsupportedCategory,
    clean_line,
    MAX_ITEM_CHARS,
)

_I = re.IGNORECASE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


# --- families ---------------------------------------------------------------------------------------------------------

_RESEARCH = _rx(
    r"\b(research|investigate|look\s+up|look\s+into|look\s+for|find(?:\s+out)?|search(?:\s+for|\s+the\s+web)?|"
    r"gather\s+(?:information|info|sources|evidence|data)|current\s+options|latest|sources?)\b"
)
#: Wording that is unambiguously about REASONING over what is already known.
#: ("compare"/"decide" are not here: comparing candidates is part of research,
#: and on their own they are weaker signals - see `_ANALYSIS_WEAK`.)
_ANALYSIS_STRONG = _rx(
    r"\b(analy[sz]e|analysis|evaluate|assess|weigh|pros\s+and\s+cons|synthesi[sz]e|summari[sz]e|"
    r"trade-?offs?|break\s+(?:it\s+)?down|think\s+(?:it\s+)?through|reason\s+(?:about|through)|"
    r"what\s+do\s+you\s+think)\b"
)
_ANALYSIS_WEAK = _rx(r"\b(compare|comparison|decide|help\s+me\s+(?:decide|choose)|which\s+(?:one|is)|recommend)\b")

_ORGANIZATION = _rx(
    r"\b(remind(?:er)?s?(?:\s+me)?|to-?do|task\s+list|(?:create|add|new)\s+(?:a\s+)?task|"
    r"(?:save|store|keep)\s+(?:it|them|this|these|those|the\s+(?:best|top|results?|findings?|options?|"
    r"info(?:rmation)?|summary|answer)|a\s+(?:copy|note)|notes?)\b|"
    r"write\s+down|note\s+down|jot\s+down|make\s+a\s+note|take\s+a\s+note|"
    r"set\s+(?:a\s+)?(?:timer|alarm)|calendar|schedule|appointment|organi[sz]e|plan\s+my\s+(?:day|week))"
)

_COMPUTER = _rx(
    r"\b((?:open|launch)\s+(?:the\s+|my\s+)?(?:app(?:lication)?|browser|chrome|edge|firefox|notepad|"
    r"calculator|explorer|website|site|https?://\S+|\S+\.(?:com|org|net|io)\b)|"
    r"click|double[-\s]click|scroll|type\s+(?:in|into)|press\s+(?:the\s+)?key|move\s+the\s+mouse|"
    r"(?:go|navigate|browse)\s+to\s+\S+|on\s+my\s+computer|in\s+(?:the|my)\s+browser)"
)


# --- computer actions ------------------------------------------------------------------------------------------------------
#
# WHICH computer actions the wording asks for - not whether they are allowed. What is allowed is decided by the capability
# boundary against the tools a Job is actually offered (app/handle_this/boundary.py); this only names the kinds so the boundary
# can compare. Fixed patterns, no AI, never a tool name here.

_URLISH = (
    r"(?:https?://\S+|www\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|app|dev|ai|co|me|edu|gov|info|biz|xyz|site|"
    r"online|tech|cloud|page|ae|uk|us|de|fr|in)\b(?:/\S*)?)"
)
_NAVIGATE = _rx(
    r"\b(?:open|visit|go\s+to|navigate\s+to|browse\s+to|load|pull\s+up|head\s+to)\s+(?:(?:the|my|this)\s+)?"
    rf"(?:{_URLISH}|(?:web\s?)?(?:site|page)\b|browser\b)"
)
_HAS_URL = _rx(_URLISH)
_LAUNCH_APP = _rx(
    r"\b(?:open|launch|start)\s+(?:(?:the|my|a)\s+)?(?:app(?:lication)?|notepad|calculator|(?:file\s+)?explorer|paint|"
    r"chrome|edge|firefox|browser|word|excel|outlook|spotify|terminal|settings)\b"
)
_SCROLL = _rx(r"\bscroll(?:ing|ed)?\b")
_SWITCH_WINDOW = _rx(
    r"\bswitch\s+(?:to|over\s+to|between)\s+[\w\s\-'\"]{0,40}?(?:window|app|application|program)s?\b|\bswitch\s+windows?\b|"
    r"\balt[-\s]?tab\b|\b(?:bring|pull)\s+up\s+(?:the\s+)?[\w\s\-]{0,30}?window\b"
)
_TAB = _rx(
    r"\b(?:switch|change|go)\s+(?:to|between|over\s+to)\s+(?:the\s+)?(?:\w+\s+)?tabs?\b|"
    r"\b(?:new|next|previous|other|another|second|first)\s+tab\b"
)
_CLOSE = _rx(r"\bclose\s+(?:the\s+|this\s+|that\s+|my\s+)?(?:tab|window|browser|app|application)s?\b")
_CLICK = _rx(
    r"\b(?:double[-\s]|right[-\s])?click(?:ing|ed)?\b|\btap\s+(?:on|the)\b|\bpress\s+(?:the\s+)?[\w\s]{0,20}?button\b"
)
_TYPE = _rx(
    r"\btype\s+(?:in|into|out|the\s+following|this|that|my|your|\"|“|')|"
    r"\benter\s+(?:my|your|the|a|an|this|that)\b[^.,;]{0,40}?\b(?:in|into|on)\b"
)
_KEY = _rx(
    r"\bpress\s+(?:the\s+)?(?:\w+\s+)?(?:key|enter|tab|escape|esc|ctrl|control|alt|shift|f\d{1,2})\b|"
    r"\bhit\s+(?:the\s+)?(?:enter|return|tab|escape|esc)\b|\bkeyboard\s+shortcut\b"
)
_MOUSE = _rx(r"\bmove\s+(?:the\s+)?(?:mouse|cursor)\b|\bdrag\b|\bhover\b")
_FORM = _rx(
    r"\bfill\s+(?:in|out)\b|\bsubmit\b|\bsign\s?(?:up|in)\b|\blog\s?in(?:to)?\b|\bjoin\s+(?:the\s+)?wait\s?list\b|\bregister\b|"
    r"\bcheck\s?out\b"
)
_OPEN_VERB = _rx(r"\b(?:open|launch|visit|go\s+to|navigate\s+to|browse\s+to)\b")

#: Kinds, in a fixed order. `form` is a click-and-type workflow and is only treated as computer wording next to an
#: open/visit verb, because "submit", "register" and "log in" are ordinary words in non-computer requests.
_STRONG_COMPUTER_KINDS = ("navigate", "scroll", "switch_window", "tab", "close", "click", "type", "key", "mouse", "launch_app")


def detect_computer_actions(message: str) -> tuple[str, ...]:
    """The kinds of computer action the wording asks for, in a fixed order: navigate, scroll, switch_window, tab,
    close, click, type, key, mouse, launch_app, form. A launch of a browser/app is `launch_app` only when no URL or
    site is named (with one, opening it IS navigation)."""
    kinds: list[str] = []
    has_url = bool(_HAS_URL.search(message))
    if _NAVIGATE.search(message):
        kinds.append("navigate")
    if _SCROLL.search(message):
        kinds.append("scroll")
    if _SWITCH_WINDOW.search(message):
        kinds.append("switch_window")
    if _TAB.search(message):
        kinds.append("tab")
    if _CLOSE.search(message):
        kinds.append("close")
    if _CLICK.search(message):
        kinds.append("click")
    if _TYPE.search(message):
        kinds.append("type")
    if _KEY.search(message):
        kinds.append("key")
    if _MOUSE.search(message):
        kinds.append("mouse")
    if _LAUNCH_APP.search(message) and not has_url:
        kinds.append("launch_app")
    if _FORM.search(message):
        kinds.append("form")
    return tuple(kinds)


def is_computer_workflow(message: str) -> bool:
    """Whether the wording is a request to operate the computer / a browser (as opposed to merely mentioning one)."""
    kinds = detect_computer_actions(message)
    if any(k in _STRONG_COMPUTER_KINDS for k in kinds):
        return True
    return "form" in kinds and bool(_OPEN_VERB.search(message))


def detect_families(message: str) -> tuple[GoalFamily, ...]:
    """The supported families the wording points at, in a fixed order (empty
    when there is no signal). Comparing candidates counts as RESEARCH when a
    research signal is also present, and as ANALYSIS only on its own."""
    found: list[GoalFamily] = []
    research = bool(_RESEARCH.search(message))
    if research:
        found.append(GoalFamily.RESEARCH)
    if _ANALYSIS_STRONG.search(message) or (not research and _ANALYSIS_WEAK.search(message)):
        found.append(GoalFamily.ANALYSIS)
    if _ORGANIZATION.search(message):
        found.append(GoalFamily.ORGANIZATION)
    if _COMPUTER.search(message) or is_computer_workflow(message):
        found.append(GoalFamily.COMPUTER_WORKFLOW)
    return tuple(found)


# --- outside the boundary -----------------------------------------------------------------------------------------------

_CLAUSE_START = r"(?:^\s*|[,.;:!\n]\s*|\b(?:and|then|also|please|just|now)\s+)(?:go\s+ahead\s+and\s+)?"

#: Checked in this order; the first that matches names the category.
_UNSUPPORTED: tuple[tuple[UnsupportedCategory, re.Pattern], ...] = (
    (
        UnsupportedCategory.PERMISSION_CHANGE,
        _rx(
            r"\b(?:grant|give)\s+(?:yourself|jarvis|you)\b.*\b(?:permission|access|autonomy|rights)\b|"
            r"\b(?:ignore|bypass|disable|skip|override|turn\s+off)\s+(?:all\s+|your\s+|the\s+|any\s+|of\s+your\s+)*"
            r"(?:rules|restrictions|safety|permissions?|confirmations?|guardrails|limits|checks|instructions)\b|"
            r"\bwithout\s+(?:asking|confirmation|confirming|permission|approval)\b|"
            r"\b(?:full|unrestricted|root|admin(?:istrator)?)\s+(?:access|autonomy|control|rights)\b|"
            r"\b(?:autonomous\s+mode|unattended)\b|"
            r"\bincluding\s+(?:any\s+|the\s+|all\s+)?(?:dangerous|destructive|risky|irreversible)\b|"
            r"\b(?:dangerous|destructive)\s+(?:tools?|actions?|ones|commands?)\b"
        ),
    ),
    (
        UnsupportedCategory.SELF_MODIFICATION,
        _rx(
            r"\b(?:modify|change|edit|rewrite|update|fix|improve|patch|delete)\s+"
            r"(?:your(?:\s+own)?|jarvis(?:'s)?(?:\s+own)?|the\s+jarvis)\s+"
            r"(?:code|source|codebase|prompts?|instructions|rules|tools?|memory|personality)\b|"
            r"\b(?:modify|rewrite|improve|update)\s+yourself\b|\bself[-\s]?modify\b"
        ),
    ),
    (
        UnsupportedCategory.FILE_OR_SYSTEM_CHANGE,
        _rx(
            r"\b(?:delete|remove|erase|wipe|format|uninstall|install|overwrite|rename)\s+"
            r"(?:all\s+|the\s+|my\s+|this\s+|these\s+|every\s+)*"
            r"(?:files?|folders?|directory|directories|drive|disk|system|registry|programs?|software|apps?)\b|"
            r"\b(?:run|running|execute|executing)\s+"
            r"(?:(?:a|an|the|this|that|my|any|some|arbitrary|shell|system|raw|random)\s+)*"
            r"(?:scripts?|commands?|shell|powershell|cmd|terminal|code|programs?|executables?)\b|"
            r"\bsudo\b|\brm\s+-rf\b|\bdownload\s+and\s+(?:run|install|execute)\b"
        ),
    ),
    (
        UnsupportedCategory.HARDWARE_CONTROL,
        _rx(
            r"\b(?:control|operate|start|stop|power\s+(?:on|off)|turn\s+(?:on|off)|move|drive|fly)\s+"
            r"(?:the\s+|my\s+)?(?:3d\s+)?(?:printer|robot(?:ic)?\s+arm|robot|drone|motors?|servos?|arduino|gpio)\b|"
            r"\bstart\s+(?:the\s+|a\s+)?print\b"
        ),
    ),
    (
        UnsupportedCategory.TRANSACTION,
        _rx(
            _CLAUSE_START
            + r"(?:buy|purchase|order|pay(?:\s+for)?|check\s*out|subscribe\s+to|donate|transfer)\b|"
            r"\bbook\s+(?:me\s+)?(?:a|an|the)\s+(?:flight|hotel|ticket|table|room|car)\b|"
            r"\bplace\s+(?:an|the)\s+order\b"
        ),
    ),
    (
        UnsupportedCategory.COMMUNICATION,
        _rx(
            r"\b(?:send|write|draft|compose|forward|reply\s+to)\s+(?:an?\s+|the\s+|my\s+|them\s+|him\s+|her\s+|it\s+)?"
            r"(?:e-?mail|message|text|dm|whatsapp)\b|"
            r"\be-?mail\s+(?:him|her|them|my|the|it|me|[a-z]+@)|"
            r"\b(?:text|call|dm|message|whatsapp)\s+(?:him|her|them|my|the)\b|"
            r"\b(?:tweet|publish|post\s+(?:it\s+)?(?:on|to))\b"
        ),
    ),
)


def detect_unsupported(message: str) -> UnsupportedCategory | None:
    """The first out-of-boundary category the wording asks for, or None."""
    for category, pattern in _UNSUPPORTED:
        if pattern.search(message):
            return category
    return None


# --- constraints ---------------------------------------------------------------------------------------------------------

_NUMBER_WORD = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_AMOUNT = r"[$€£]\s?\d[\d,]*(?:\.\d+)?\s?(?:k\b)?|\d[\d,]*(?:\.\d+)?\s?(?:k\b)?\s?(?:dollars|usd|eur|euros|gbp|pounds|bucks|aed|dirhams?)"
_BUDGET = _rx(
    rf"(?:(?:under|below|less\s+than|up\s+to|within|around|about|at\s+most|no\s+more\s+than|"
    rf"max(?:imum)?(?:\s+of)?|budget\s+(?:of|is)?)\s*(?:{_AMOUNT})|(?:{_AMOUNT}))"
)
_BUDGET_REFERENCE = _rx(
    r"\b(?:within|on|under|inside)\s+(?:my|our|the)\s+budget\b|\bmy\s+budget\b|\bon\s+a\s+budget\b|"
    r"\bbudget[-\s]friendly\b|\baffordable\b|\bcheap(?:est)?\b"
)
_QUANTITY = _rx(rf"\b(?:top|best|compare|research|find|save|pick|shortlist|choose|list|give\s+me)\s+(?:the\s+)?({_NUMBER_WORD})\b")
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_TIMEFRAME = _rx(
    rf"\b(?:today|tonight|tomorrow|this\s+(?:morning|afternoon|evening|week|weekend|month)|"
    rf"next\s+(?:week|month|{_WEEKDAY})|by\s+(?:{_WEEKDAY}|\d{{1,2}}(?:st|nd|rd|th)?)|"
    rf"in\s+(?:\d+|an?|a\s+few)\s+(?:minutes?|hours?|days?|weeks?)|on\s+{_WEEKDAY}|at\s+\d{{1,2}}(?::\d{{2}})?\s?(?:am|pm)?)\b"
)
_USE_CASE = _rx(
    r"\bfor\s+(?:my|our)\s+((?:[a-z0-9\-]+)(?:\s+(?!within\b|under\b|and\b|that\b|which\b|with\b|to\b|but\b|then\b)"
    r"[a-z0-9\-]+){0,3})"
)


def extract_constraints(message: str) -> tuple[Constraint, ...]:
    """The constraints the request itself states (an amount, a count, a time,
    a purpose). "within my budget" is NOT a constraint - it points at a budget
    that must come from somewhere else (see `mentions_budget_reference`)."""
    found: list[Constraint] = []

    def add(topic: Topic, text: str) -> None:
        text = clean_line(text, MAX_ITEM_CHARS)
        if text and not any(c.topic is topic and c.text.lower() == text.lower() for c in found):
            found.append(Constraint(topic, text, ConstraintSource.REQUEST))

    if (m := _BUDGET.search(message)) and re.search(r"\d", m.group(0)):
        add(Topic.BUDGET, m.group(0))
    if m := _QUANTITY.search(message):
        add(Topic.QUANTITY, m.group(1))
    if m := _TIMEFRAME.search(message):
        add(Topic.TIMEFRAME, m.group(0))
    if m := _USE_CASE.search(message):
        add(Topic.USE_CASE, m.group(1))
    return tuple(found)


def mentions_budget_reference(message: str) -> bool:
    return bool(_BUDGET_REFERENCE.search(message))


# --- what the request is about ---------------------------------------------------------------------------------------------

_COMMERCE = _rx(
    r"\b(buy|buying|purchase|price|prices|pricing|cost|costs|cheap|cheapest|affordable|budget|"
    r"worth\s+(?:buying|getting)|should\s+i\s+(?:get|buy)|shopping|deals?)\b"
)
_PRODUCT = _rx(
    r"\b(printers?|laptops?|notebooks?|phones?|smartphones?|cameras?|monitors?|keyboards?|mice|drones?|"
    r"motors?|servos?|sensors?|arduinos?|raspberry\s+pi|chairs?|desks?|headphones|earbuds|tablets?|gpus?|"
    r"cpus?|routers?|microphones?|batter(?:y|ies)|kits?|soldering\s+irons?|oscilloscopes?|multimeters?|"
    r"watch(?:es)?|speakers?|tvs?)\b"
)
_SELECTING = _rx(r"\b(research|find|look\s+for|compare|recommend|best|top|which|choose|pick|shortlist|options?)\b")
_INFORMATIONAL = _rx(
    r"\b(history\s+of|how\s+(?:does|do|to|is|are)|how\s+[\w\-]+(?:\s+[\w\-]+){0,3}?\s+works?|explain|"
    r"what\s+is|what\s+are\s+the\s+parts|why)\b"
)


def mentions_product(message: str) -> bool:
    """Whether the request names a kind of thing one acquires (a printer, a
    laptop, a sensor...)."""
    return bool(_PRODUCT.search(message))


def is_informational(message: str) -> bool:
    """Whether the wording asks to be TOLD something ("what is", "how does ...
    work", "explain") rather than to have work done."""
    return bool(_INFORMATIONAL.search(message))


def is_purchase_oriented(message: str) -> bool:
    """Whether the request is about CHOOSING something to acquire, where price
    would materially change the answer. A heuristic on purpose: commerce words,
    or a product noun together with wording about selecting one."""
    if _INFORMATIONAL.search(message):
        return False
    if _COMMERCE.search(message):
        return True
    return bool(_PRODUCT.search(message) and _SELECTING.search(message))


_STOPWORDS = frozenset(
    """a an the and or but so then to of in on at for from with without by about into over under it its this that
    these those them they he she him her me my mine our we you your i is are was were be been being do does did
    please just some any all also again now will would can could should shall may might must need want like
    something anything everything stuff things thing one ones up out off as if not no yes""".split()
)
STOPWORDS = _STOPWORDS  # public name for the same set (used by app/handle_this/continuation.py)
_ACTION_WORDS = frozenset(
    """research investigate look lookup find search gather compare analyze analyse evaluate assess weigh summarize
    summarise synthesize synthesise decide recommend remind reminder save store keep write jot schedule
    open launch click scroll type press move go navigate browse tell show give help organize organise plan
    create add make handle sort check tomorrow today tonight tomorrow week weekend month latest current options
    option best top""".split()
)


def has_subject(message: str) -> bool:
    """Whether, beyond the action words, the request names anything at all it
    is about ("research a laptop" does; "research it" / "look into this" do
    not)."""
    words = re.findall(r"[a-z0-9][a-z0-9'\-]*", message.lower())
    return any(w not in _STOPWORDS and w not in _ACTION_WORDS and not w.isdigit() for w in words)


_NEEDS_TIME = _rx(r"\b(?:remind(?:er)?s?(?:\s+me)?|schedule|appointment|put\s+.+\s+on\s+my\s+calendar|add\s+.+\s+to\s+my\s+calendar)\b")


def needs_time(message: str) -> bool:
    """Whether the wording asks for something that only makes sense at a
    particular time (a reminder, a scheduled event)."""
    return bool(_NEEDS_TIME.search(message))


def has_time(message: str) -> bool:
    return bool(_TIMEFRAME.search(message))


# --- success criteria ----------------------------------------------------------------------------------------------------------

_CRITERIA: tuple[tuple[re.Pattern, str], ...] = (
    (_rx(r"\b(research|find|look\s+up|look\s+into|look\s+for|investigate|search)\b"), "Relevant options or information are found"),
    (_rx(r"\b(compare|comparison)\b"), "The options are compared"),
    (_rx(r"\b(analy[sz]e|evaluate|assess|weigh|pros\s+and\s+cons|trade-?offs?)\b"), "The information is analysed"),
    (_rx(r"\b(what\s+i\s+should\s+consider|considerations?|tell\s+me\s+what)\b"), "The key considerations are reported"),
    (_rx(r"\b(summari[sz]e|summary)\b"), "A summary is provided"),
    (
        _rx(r"\b(?:save|store|keep)\s+(?:it|them|this|these|those|the\s+\w+)|write\s+down|note\s+down|jot\s+down"),
        "The chosen results are saved",
    ),
    (_rx(r"\bremind(?:er)?s?\b"), "A reminder is set"),
    (_rx(r"\b(task|to-?do)\b"), "The task is recorded"),
    (_rx(r"\b(schedule|calendar|appointment)\b"), "The calendar is updated"),
    (_COMPUTER, "The requested computer workflow is carried out"),
)


#: What "done" means for each computer action: never "the command ran" - the resulting state is OBSERVED and verified.
#: Fixed phrases (no user text); they reach the Planner as part of the objective, which is data about the request.
_COMPUTER_CRITERIA = {
    "navigate": "The requested page is confirmed open by observing the window or page afterwards, not assumed",
    "scroll": "After scrolling (at most 2000 per step) the screen is confirmed to have changed: capture it before, verify it after",
    "switch_window": "The requested window is confirmed to be the active one afterwards",
}


def derive_success_criteria(message: str) -> tuple[str, ...]:
    """What the wording says "done" looks like - fixed phrases keyed to the
    deliverable words the user used, never the user's text itself."""
    general = [text for pattern, text in _CRITERIA if pattern.search(message)]
    computer = [_COMPUTER_CRITERIA[k] for k in detect_computer_actions(message) if k in _COMPUTER_CRITERIA]
    return tuple((general[: 8 - len(computer)] + computer)[:8])
