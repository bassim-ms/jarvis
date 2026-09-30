"""
Parser/planner tests for explicit tool-command understanding
(app/tools/command.py).

The runtime path (/chat -> JarvisCore -> planner -> ToolRegistry) is covered
in tests/test_tool_command_runtime.py; this file pins the deterministic
edge cases of the command layer itself, against small fake tools that stand
for the SHAPES real tools have (no params, one single-token string, one
free-text string, numbers, booleans, optional params, prose-argument tools).

Nothing here needs a real backend, and none of it may need an AI call: an
ExplodingProvider fails the test if one is ever asked to construct a call.
"""

import pytest

from app.tools.base import (
    PermissionLevel,
    Tool,
    ToolArgumentError,
    ToolParameter,
    ToolResult,
)
from app.tools.command import (
    CommandRejection,
    ExplicitToolCommandPlanner,
    ParsedCommand,
    ToolSequence,
    match_invocation_head,
    parse_explicit_command,
)
from app.tools.decision import (
    ToolCall,
    ToolCallBuilder,
    ToolCallError,
    ToolCallProvider,
)
from app.tools.registry import ToolRegistry


# --- fake tools ---------------------------------------------------------------


class PingTool(Tool):
    name = "ping"
    description = "pong"
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(message="pong")


class ListTasksLikeTool(PingTool):
    """A no-argument tool (like list_tasks with no filter)."""

    name = "list_tasks"


class OpenLikeTool(Tool):
    """One single-token string (a URL) with a value-level validator."""

    name = "open_url"
    description = "open"
    permission = PermissionLevel.SAFE_ACTION

    @property
    def parameters(self):
        return [ToolParameter(name="url", type="string")]

    def validate_arguments(self, url: str) -> None:
        if not url.startswith(("http://", "https://")):
            raise ToolArgumentError(f"url must use http or https, got: {url!r}")

    def execute(self, url: str) -> ToolResult:
        return ToolResult.success(message=f"opened {url}")


class ScrollLikeTool(Tool):
    name = "scroll"
    description = "scroll"
    permission = PermissionLevel.SAFE_ACTION

    @property
    def parameters(self):
        return [ToolParameter(name="amount", type="number")]

    def validate_arguments(self, amount) -> None:
        if not isinstance(amount, int) or abs(amount) > 2000:
            raise ToolArgumentError(f"scroll amount must be an integer within 2000, got: {amount!r}")

    def execute(self, amount) -> ToolResult:
        return ToolResult.success(message=f"scrolled {amount}")


class MoveLikeTool(Tool):
    name = "move_mouse"
    description = "move"
    permission = PermissionLevel.SAFE_ACTION

    @property
    def parameters(self):
        return [ToolParameter(name="x", type="number"), ToolParameter(name="y", type="number")]

    def execute(self, x, y) -> ToolResult:
        return ToolResult.success(message=f"moved {x},{y}")


class ClickLikeTool(Tool):
    """Two required numbers and one OPTIONAL string (like click_mouse)."""

    name = "click_mouse"
    description = "click"
    permission = PermissionLevel.SAFE_ACTION

    @property
    def parameters(self):
        return [
            ToolParameter(name="x", type="number"),
            ToolParameter(name="y", type="number"),
            ToolParameter(name="button", type="string", required=False, default="left"),
        ]

    def execute(self, x, y, button="left") -> ToolResult:
        return ToolResult.success(message=f"clicked {button}")


class EchoTool(Tool):
    """One FREE-TEXT string (prose)."""

    name = "echo"
    description = "echo"
    permission = PermissionLevel.READ

    @property
    def parameters(self):
        return [ToolParameter(name="message", type="string", free_text=True)]

    def execute(self, message: str) -> ToolResult:
        return ToolResult.success(message=message)


class WordTool(Tool):
    """One single-token string with NO free_text (the default)."""

    name = "word"
    description = "word"
    permission = PermissionLevel.READ

    @property
    def parameters(self):
        return [ToolParameter(name="text", type="string")]

    def execute(self, text: str) -> ToolResult:
        return ToolResult.success(message=text)


class FlagTool(Tool):
    name = "flag"
    description = "flag"
    permission = PermissionLevel.READ

    @property
    def parameters(self):
        return [
            ToolParameter(name="enabled", type="boolean"),
            ToolParameter(name="note", type="string", required=False),
        ]

    def execute(self, enabled: bool, note: str = "") -> ToolResult:
        return ToolResult.success(message=f"{enabled}/{note}")


class FreeThenRequiredTool(Tool):
    """Ambiguous by construction: free text FOLLOWED by a required arg."""

    name = "free_then_required"
    description = "ambiguous"
    permission = PermissionLevel.READ

    @property
    def parameters(self):
        return [
            ToolParameter(name="text", type="string", free_text=True),
            ToolParameter(name="count", type="number"),
        ]

    def execute(self, text: str, count) -> ToolResult:
        return ToolResult.success(message=text)


class ConfirmTool(Tool):
    name = "confirm_me"
    description = "needs confirmation"
    permission = PermissionLevel.CONFIRM

    def execute(self) -> ToolResult:
        return ToolResult.success(message="ran")


class ProseTool(Tool):
    """A prose-argument tool: the ONLY kind an AI may be asked about."""

    name = "remind"
    description = "prose"
    permission = PermissionLevel.SAFE_ACTION
    natural_language_arguments = True

    @property
    def parameters(self):
        return [
            ToolParameter(name="text", type="string"),
            ToolParameter(name="minutes", type="number"),
        ]

    def execute(self, text: str, minutes) -> ToolResult:
        return ToolResult.success(message=f"{text}@{minutes}")


class ExplodingProvider(ToolCallProvider):
    def __init__(self):
        self.calls: list[str] = []

    def decide(self, message, candidate_tools):
        self.calls.append(message)
        raise AssertionError(f"an AI call must not be made, got: {message!r}")


def make_registry(*extra: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        PingTool(),
        ListTasksLikeTool(),
        OpenLikeTool(),
        ScrollLikeTool(),
        MoveLikeTool(),
        ClickLikeTool(),
        EchoTool(),
        WordTool(),
        FlagTool(),
        FreeThenRequiredTool(),
        ConfirmTool(),
        *extra,
    ):
        registry.register(tool)
    return registry


REGISTRY = make_registry()


def parse(message: str):
    return parse_explicit_command(message, REGISTRY.get)


def steps_of(message: str) -> list[tuple[str, dict]]:
    parsed = parse(message)
    assert isinstance(parsed, ParsedCommand), getattr(parsed, "issues", parsed)
    return [(step.tool.name, step.arguments) for step in parsed.steps]


def rejection_of(message: str) -> CommandRejection:
    parsed = parse(message)
    assert isinstance(parsed, CommandRejection), parsed
    return parsed


# --- 1-3: one, two, three explicit tools ---------------------------------------


def test_a_single_explicit_tool():
    assert steps_of("use tool ping") == [("ping", {})]


def test_two_explicit_tools():
    assert steps_of("use tool ping, then use tool list_tasks") == [
        ("ping", {}),
        ("list_tasks", {}),
    ]


def test_three_explicit_tools_keep_their_order():
    assert steps_of(
        "use tool scroll 1, then use tool scroll 2, then use tool scroll 3"
    ) == [("scroll", {"amount": 1}), ("scroll", {"amount": 2}), ("scroll", {"amount": 3})]


# --- 4-6: the separators ---------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "use tool ping, then use tool list_tasks",  # A, then use tool B
        "use tool ping, then list_tasks",  # A, then B
        "use tool ping then use tool list_tasks",  # no comma
        "use tool ping and then use tool list_tasks",  # A and then use tool B
        "use tool ping and then list_tasks",  # A and then B
        "use tool ping, and then list_tasks",  # comma before "and then"
        "use tool ping followed by list_tasks",  # A followed by B
        "use tool ping followed by use tool list_tasks",
        "USE TOOL ping, THEN list_tasks",  # case-insensitive keywords
        "use the ping tool, then use the list_tasks tool",  # name-before-tool phrasing
    ],
)
def test_every_supported_separator_form_yields_the_same_ordered_steps(message):
    assert steps_of(message) == [("ping", {}), ("list_tasks", {})]


def test_and_alone_is_not_a_separator():
    issues = rejection_of("use tool ping and use tool list_tasks").issues
    assert "unexpected text" in issues[0].message


# --- 7: URLs are preserved exactly ------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://luniti.app",
        "https://luniti.app/",
        "https://luniti.app/path/to/page.html",
        "https://luniti.app:8443/x",
        "http://example.com/a?b=1&c=2#frag",
        "https://example.com/?ids=1,2,3",  # comma INSIDE the token
        "https://example.com/then",  # separator-lookalike inside the token
        "https://example.com/a-b_c~d%20e",
    ],
)
def test_https_urls_are_preserved_exactly(url):
    assert steps_of(f"use tool open_url {url}") == [("open_url", {"url": url})]
    assert steps_of(f"use tool open_url {url}, then use tool scroll 5") == [
        ("open_url", {"url": url}),
        ("scroll", {"amount": 5}),
    ]


def test_a_url_never_swallows_a_trailing_clause():
    """THE root-cause regression: open_url used to receive "https://luniti.app,
    wait for 5 seconds"."""
    rejection = rejection_of(
        "use tool open_url https://luniti.app, wait for 5 seconds, then use tool scroll -600"
    )

    assert len(rejection.issues) == 1
    issue = rejection.issues[0]
    assert issue.index == 1 and issue.tool_name == "open_url"
    assert '"wait for 5 seconds"' in issue.message
    assert '"wait"' in issue.message and "not an available tool" in issue.message


def test_a_url_followed_by_words_without_a_comma_is_also_rejected():
    rejection = rejection_of("use tool open_url https://luniti.app wait for 5 seconds")

    assert '"wait for 5 seconds"' in rejection.issues[0].message


# --- 8/9: numbers -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("-600", -600), ("600", 600), ("+5", 5), ("0", 0), ("-0", 0), ("3.5", 3.5), ("-.25", -0.25)],
)
def test_numeric_arguments_bind_to_a_number(text, expected):
    (name, arguments), = steps_of(f"use tool scroll {text}")
    assert arguments == {"amount": expected}
    assert type(arguments["amount"]) is type(expected)


def test_negative_number_followed_by_another_step():
    assert steps_of("use tool scroll -600, then use tool scroll 600") == [
        ("scroll", {"amount": -600}),
        ("scroll", {"amount": 600}),
    ]


@pytest.mark.parametrize("bad", ["abc", "1e5", "0x10", "--5", "5-", "1,000", "٣", "1 2 3"])
def test_non_numeric_text_is_rejected_for_a_number(bad):
    rejection = rejection_of(f"use tool scroll {bad}")
    assert rejection.issues[0].index == 1


def test_a_quoted_value_is_not_a_number():
    rejection = rejection_of('use tool scroll "5"')
    assert "must be a number" in rejection.issues[0].message


# --- 10-13: strings, multiple arguments, punctuation -------------------------------


def test_quoted_strings_double_single_and_curly():
    assert steps_of('use tool word "hello there"') == [("word", {"text": "hello there"})]
    assert steps_of("use tool word 'hello there'") == [("word", {"text": "hello there"})]
    assert steps_of("use tool word \u201chello there\u201d") == [("word", {"text": "hello there"})]


def test_quoted_string_escapes_its_own_quote_and_backslash():
    assert steps_of(r'use tool word "say \"hi\""') == [("word", {"text": 'say "hi"'})]
    assert steps_of(r'use tool word "back\\slash"') == [("word", {"text": "back\\slash"})]


def test_unquoted_single_token_string():
    assert steps_of("use tool word hello") == [("word", {"text": "hello"})]


def test_an_unquoted_multi_word_value_is_rejected_for_a_single_token_string():
    """Fail closed by default: the extra words are not silently absorbed."""
    rejection = rejection_of("use tool word hello world")
    assert '"world"' in rejection.issues[0].message


def test_free_text_string_spans_the_rest_of_its_step_verbatim():
    assert steps_of("use tool echo hello   big world!") == [
        ("echo", {"message": "hello   big world!"})
    ]
    assert steps_of("use the echo tool to say hello") == [("echo", {"message": "to say hello"})]


def test_free_text_stops_at_the_step_separator():
    assert steps_of("use tool echo hello world, then use tool ping") == [
        ("echo", {"message": "hello world"}),
        ("ping", {}),
    ]
    assert steps_of("use tool echo hello world followed by ping") == [
        ("echo", {"message": "hello world"}),
        ("ping", {}),
    ]


def test_free_text_may_contain_commas_that_are_not_separators():
    assert steps_of("use tool echo one, two, three") == [("echo", {"message": "one, two, three"})]


def test_multiple_arguments_bind_positionally():
    assert steps_of("use tool move_mouse 500 300") == [("move_mouse", {"x": 500, "y": 300})]
    assert steps_of("use tool click_mouse 10 20 right") == [
        ("click_mouse", {"x": 10, "y": 20, "button": "right"})
    ]


def test_optional_arguments_may_be_omitted():
    assert steps_of("use tool click_mouse 10 20") == [("click_mouse", {"x": 10, "y": 20})]
    assert steps_of("use tool flag true") == [("flag", {"enabled": True})]


def test_a_missing_required_argument_is_named():
    issue = rejection_of("use tool move_mouse 500").issues[0]
    assert "missing required argument(s): y" in issue.message


def test_boolean_arguments():
    assert steps_of("use tool flag yes note") == [("flag", {"enabled": True, "note": "note"})]
    assert steps_of("use tool flag FALSE") == [("flag", {"enabled": False})]
    assert "must be true or false" in rejection_of("use tool flag maybe").issues[0].message


@pytest.mark.parametrize(
    "token",
    ["ctrl+shift+esc", "a.b/c?d=e&f=g#h", "C:\\temp\\x.txt", "user@example.com", "50%", "it's", "a=1,2"],
)
def test_arguments_containing_punctuation_stay_one_token(token):
    assert steps_of(f"use tool word {token}") == [("word", {"text": token})]
    assert steps_of(f"use tool word {token}, then use tool ping") == [
        ("word", {"text": token}),
        ("ping", {}),
    ]


# --- 14: commas / separators inside quotes -------------------------------------------


def test_commas_inside_quoted_strings_are_preserved():
    assert steps_of('use tool word "a, b, c"') == [("word", {"text": "a, b, c"})]


def test_separator_words_inside_quotes_never_split():
    assert steps_of('use tool word "first, then second and then third followed by fourth"') == [
        ("word", {"text": "first, then second and then third followed by fourth"})
    ]
    assert steps_of('use tool echo "wait, then go", then use tool ping') == [
        ("echo", {"message": "wait, then go"}),
        ("ping", {}),
    ]


def test_free_text_with_a_leading_quote_takes_only_the_quoted_value():
    issue = rejection_of('use tool echo "hi" there').issues[0]
    assert '"there"' in issue.message


# --- 15-17: unknown tools -----------------------------------------------------------------


def test_unknown_tool_as_the_first_step():
    issues = rejection_of("use tool fake_tool, then use tool scroll -600").issues
    assert [(i.index, i.tool_name) for i in issues] == [(1, "fake_tool")]
    assert "Unknown tool: 'fake_tool'" in issues[0].message


def test_unknown_tool_as_a_middle_step():
    issues = rejection_of(
        "use tool ping, then use tool fake_tool, then use tool scroll -600"
    ).issues
    assert [(i.index, i.tool_name) for i in issues] == [(2, "fake_tool")]


def test_unknown_tool_as_the_final_step():
    issues = rejection_of("use tool open_url https://luniti.app, then use tool fake_tool").issues
    assert [(i.index, i.tool_name) for i in issues] == [(2, "fake_tool")]


def test_every_invalid_step_is_reported_not_just_the_first():
    issues = rejection_of("use tool nope1, then use tool ping, then use tool nope2").issues
    assert [(i.index, i.tool_name) for i in issues] == [(1, "nope1"), (3, "nope2")]


def test_a_bare_step_after_a_separator_must_still_be_a_registered_tool():
    issues = rejection_of("use tool ping, then definitely_not_a_real_tool").issues
    assert "Unknown tool: 'definitely_not_a_real_tool'" in issues[0].message


def test_a_non_identifier_tool_name_is_rejected_not_guessed():
    for message in ("use tool 5", "use tool https://luniti.app", "use tool 'ping'"):
        issues = rejection_of(message).issues
        assert "not a valid tool name" in issues[0].message


# --- 18: unsupported "wait" ----------------------------------------------------------------


def test_wait_is_not_an_available_tool_anywhere():
    """There is no wait/sleep/delay tool - so it fails closed like any
    unknown tool, in every position, with a useful explanation."""
    assert REGISTRY.get("wait") is None

    as_a_step = rejection_of("use tool ping, then wait for 5 seconds, then use tool scroll -600")
    assert "Unknown tool: 'wait'" in as_a_step.issues[0].message

    as_a_tool = rejection_of("use tool wait 5")
    assert "Unknown tool: 'wait'" in as_a_tool.issues[0].message

    after_args = rejection_of("use tool ping, wait for 5 seconds, then use tool scroll -600")
    assert '"wait" was meant as another step' in after_args.issues[0].message


# --- 19/20: malformed and ambiguous sequences ------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "use tool ping, then",  # trailing separator
        "use tool ping and then",
        "use tool ping followed by",
        "use tool ping, then, then use tool ping",  # doubled separator
        "use tool ping then then ping",
        'use tool word "unterminated',  # unterminated quote
        "use tool word 'unterminated, then use tool ping",
        'use tool word "closed"glued',  # text glued to a closing quote
        "use tool ping, then use tool",  # step names nothing
        "use tool, then use tool ping",
        "use tool ping, then use",
    ],
)
def test_malformed_sequences_are_rejected(message):
    assert isinstance(parse(message), CommandRejection)


def test_a_bare_use_tool_is_not_an_explicit_command_and_keeps_its_existing_handling():
    assert parse("use tool") is None
    assert parse("use tool   ") is None


def test_free_text_followed_by_then_and_an_unknown_word_is_ambiguous_and_rejected_with_a_hint():
    issue = rejection_of("use tool echo buy milk then go home").issues[0]
    assert issue.index == 2
    assert "Unknown tool: 'go'" in issue.message
    assert "put that text in quotes" in issue.message


def test_quoting_resolves_that_ambiguity():
    assert steps_of('use tool echo "buy milk then go home"') == [
        ("echo", {"message": "buy milk then go home"})
    ]


def test_free_text_before_a_required_argument_is_ambiguous_and_rejected():
    issue = rejection_of("use tool free_then_required buy milk 3").issues[0]
    assert "ambiguous" in issue.message
    assert steps_of('use tool free_then_required "buy milk" 3') == [
        ("free_then_required", {"text": "buy milk", "count": 3})
    ]


# --- 21/22/24: leftover text is never silently ignored ---------------------------------------


def test_a_no_argument_tool_followed_by_unexpected_text_is_rejected():
    issue = rejection_of("use tool list_tasks please").issues[0]
    assert issue.tool_name == "list_tasks"
    assert '"please"' in issue.message


def test_a_tool_name_followed_directly_by_a_comma_is_not_part_of_the_name():
    assert steps_of("use tool ping, then use tool list_tasks") == [("ping", {}), ("list_tasks", {})]
    assert isinstance(parse("use tool ping, please"), CommandRejection)
    assert isinstance(parse("use tool ping,"), CommandRejection)
    assert isinstance(parse("use tool ping ,"), CommandRejection)


def test_a_comma_directly_after_the_tool_name_never_stands_in_for_an_argument():
    issue = rejection_of("use tool open_url, https://luniti.app").issues[0]
    assert issue.tool_name == "open_url"


def test_text_glued_to_the_tool_name_by_a_comma_is_rejected():
    for message in ("use tool ping,then use tool ping", "use tool echo,hi"):
        issue = rejection_of(message).issues[0]
        assert "directly attached to the tool name" in issue.message


@pytest.mark.parametrize(
    "message",
    [
        "use tool ping please",
        "use tool ping now, then use tool list_tasks",
        "use tool scroll 5 pixels",
        "use tool scroll 5 6",
        "use tool move_mouse 1 2 3",
        "use tool word a b",
        "use tool ping. Thanks!",
    ],
)
def test_extra_words_that_used_to_be_ignored_are_now_rejected(message):
    assert isinstance(parse(message), CommandRejection)


# --- 23: prose is not a tool command -----------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "my tool is great",
        "the tool works, then it broke",
        "I want to use tool ping",  # not at the start of the message
        "please use tool ping",
        "use a tool to check the weather",
        "what tool should I use, then?",
        "then use tool ping",
        "tool ping",
        "",
        "   ",
    ],
)
def test_prose_is_not_an_explicit_tool_command(message):
    assert parse(message) is None


# --- the head matcher itself --------------------------------------------------------------------


def test_match_invocation_head_cleans_the_name():
    assert match_invocation_head("use tool echo").tool_name == "echo"
    assert match_invocation_head("run tool: echo").tool_name == "echo"
    assert match_invocation_head("run tool echo: hello").tool_name == "echo"
    assert match_invocation_head("use tool echo, then").tool_name == "echo"
    assert match_invocation_head("use the echo tool to say hi").tool_name == "echo"
    assert match_invocation_head("use tool").tool_name is None
    assert match_invocation_head("use a tool to x") is None


def test_a_colon_after_the_tool_name_is_accepted():
    assert steps_of("run tool echo: hello") == [("echo", {"message": "hello"})]


# --- the planner: resolve + validate the WHOLE sequence, never execute --------------------------


def make_planner(registry=None, provider=None):
    registry = registry or REGISTRY
    provider = provider or ExplodingProvider()
    return ExplicitToolCommandPlanner(lambda name: registry, ToolCallBuilder(provider)), provider


def test_planner_returns_an_immutable_tool_sequence_of_resolved_steps():
    planner, provider = make_planner()

    sequence = planner.plan("use tool open_url https://luniti.app, then use tool scroll -600")

    assert isinstance(sequence, ToolSequence)
    assert [(s.index, s.tool_name, s.parameters) for s in sequence.steps] == [
        (1, "open_url", {"url": "https://luniti.app"}),
        (2, "scroll", {"amount": -600}),
    ]
    assert provider.calls == []  # -600 needed no AI
    with pytest.raises(Exception):
        sequence.steps = ()  # frozen


def test_planner_returns_none_for_prose_and_for_a_bare_use_tool():
    planner, _ = make_planner()
    assert planner.plan("my tool is great") is None
    assert planner.plan("use tool") is None


def test_planner_validates_argument_values_not_just_shapes():
    planner, _ = make_planner()

    rejection = planner.plan("use tool open_url ftp://example.com/file")
    assert isinstance(rejection, CommandRejection)
    assert "must use http or https" in rejection.issues[0].message

    rejection = planner.plan("use tool scroll 999999")
    assert "within 2000" in rejection.issues[0].message

    rejection = planner.plan("use tool scroll 1.5")
    assert isinstance(rejection, CommandRejection)


def test_planner_rejects_the_whole_sequence_when_any_value_is_invalid():
    planner, _ = make_planner()

    rejection = planner.plan(
        "use tool open_url https://luniti.app, then use tool scroll 999999, then use tool ping"
    )

    assert isinstance(rejection, CommandRejection)
    assert [i.index for i in rejection.issues] == [2]


def test_planner_checks_every_steps_permission_before_any_can_run():
    """27/28: a CONFIRM-tier tool anywhere rejects the whole sequence."""
    planner, _ = make_planner()

    for message in (
        "use tool confirm_me, then use tool ping",
        "use tool ping, then use tool confirm_me",
        "use tool ping, then use tool confirm_me, then use tool ping",
    ):
        rejection = planner.plan(message)
        assert isinstance(rejection, CommandRejection), message
        (issue,) = rejection.issues
        assert issue.tool_name == "confirm_me"
        assert "requires confirmation" in issue.message


def test_planner_uses_the_registry_that_owns_each_tool():
    """Each step is resolved through `resolve_registry(name)` - the seam
    JarvisCore uses to keep computer tools out of M15's registry."""
    computer, ordinary = ToolRegistry(), ToolRegistry()
    computer.register(ScrollLikeTool())
    ordinary.register(PingTool())
    seen: list[str | None] = []

    def resolve(name):
        seen.append(name)
        return computer if computer.has(name) else ordinary

    planner = ExplicitToolCommandPlanner(resolve, ToolCallBuilder(ExplodingProvider()))

    sequence = planner.plan("use tool ping, then use tool scroll 5")

    assert isinstance(sequence, ToolSequence)
    assert {"ping", "scroll"} <= set(seen)

    # A name that only the OTHER registry has is simply unknown to this one.
    only_ordinary = ExplicitToolCommandPlanner(lambda n: ordinary, ToolCallBuilder(ExplodingProvider()))
    rejection = only_ordinary.plan("use tool ping, then use tool scroll 5")
    assert "Unknown tool: 'scroll'" in rejection.issues[0].message


# --- AI use is opt-in and never for deterministic arguments -------------------------------------


class RecordingProvider(ToolCallProvider):
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def decide(self, message, candidate_tools):
        self.calls.append((message, [t.name for t in candidate_tools]))
        if self.error:
            raise self.error
        return self.result


def test_deterministic_arguments_never_reach_the_ai_even_when_a_provider_is_available():
    registry = make_registry(ProseTool())
    provider = RecordingProvider(result=ToolCall("scroll", {"amount": 12345}))
    planner, _ = make_planner(registry, provider)

    sequence = planner.plan("use tool scroll -600, then use tool move_mouse 1 2")

    assert [s.parameters for s in sequence.steps] == [{"amount": -600}, {"x": 1, "y": 2}]
    assert provider.calls == []


def test_a_tool_that_cannot_be_bound_fails_closed_instead_of_asking_the_ai():
    registry = make_registry()
    provider = RecordingProvider(result=ToolCall("move_mouse", {"x": 5, "y": 5}))
    planner, _ = make_planner(registry, provider)

    rejection = planner.plan("use tool move_mouse somewhere")

    assert isinstance(rejection, CommandRejection)
    assert provider.calls == []


def test_only_a_prose_tool_is_offered_to_the_ai_and_only_with_its_own_clause():
    registry = make_registry(ProseTool())
    provider = RecordingProvider(result=ToolCall("remind", {"text": "stretch", "minutes": 10}))
    planner, _ = make_planner(registry, provider)

    sequence = planner.plan("use tool ping, then use tool remind stretch in 10 minutes, then use tool ping")

    assert [s.tool_name for s in sequence.steps] == ["ping", "remind", "ping"]
    assert sequence.steps[1].parameters == {"text": "stretch", "minutes": 10}
    assert provider.calls == [("use tool remind stretch in 10 minutes", ["remind"])]


def test_ai_output_for_a_prose_tool_is_still_validated():
    registry = make_registry(ProseTool())
    provider = RecordingProvider(result=ToolCall("remind", {"text": "stretch"}))  # minutes missing
    planner, _ = make_planner(registry, provider)

    rejection = planner.plan("use tool remind stretch")

    # The planner never repairs an AI result: the missing parameter rejects
    # the whole command.
    assert isinstance(rejection, CommandRejection)
    assert "minutes" in rejection.issues[0].message


def test_a_failed_ai_call_propagates_so_the_caller_can_apply_its_502_convention():
    registry = make_registry(ProseTool())
    provider = RecordingProvider(error=ToolCallError("provider down"))
    planner, _ = make_planner(registry, provider)

    with pytest.raises(ToolCallError, match="provider down"):
        planner.plan("use tool remind stretch in 10 minutes")
