"""
Tests for the Tool Decision Layer (app/tools/decision.py): ToolCall
representation, GeminiToolCallProvider's strict validation, and
ToolCallBuilder's deterministic-first construction. All AI calls are
faked - no real Gemini call is made. ToolRegistry is only ever used here
as a lookup table (register/get/list_tools); actual execution through it
is covered by tests/test_core.py.
"""

import pytest

from app.providers.base import AIProvider, ProviderError
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.decision import (
    GeminiToolCallProvider,
    ToolCall,
    ToolCallBuilder,
    ToolCallError,
    ToolCallIncompleteError,
    ToolCallProvider,
)
from app.tools.registry import ToolRegistry


class FakeAIProvider(AIProvider):
    """Fake AIProvider for testing GeminiToolCallProvider without a real
    Gemini call. Records the prompt it was given."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.last_message: str | None = None

    def generate_reply(self, message: str) -> str:
        self.last_message = message
        if self.error:
            raise self.error
        return self.reply


class FakeToolCallProvider(ToolCallProvider):
    """Records whether/how it was called and returns a fixed ToolCall, or
    raises - decoupled from GeminiToolCallProvider's own prompt/JSON
    logic, which is covered separately below."""

    def __init__(
        self, tool_call: ToolCall | None = None, error: Exception | None = None
    ):
        self.tool_call = tool_call
        self.error = error
        self.call_count = 0
        self.last_message: str | None = None
        self.last_candidate_tools: list[Tool] | None = None

    def decide(self, message: str, candidate_tools: list[Tool]) -> ToolCall:
        self.call_count += 1
        self.last_message = message
        self.last_candidate_tools = candidate_tools
        if self.error:
            raise self.error
        return self.tool_call


class EchoTool(Tool):
    """Fake READ tool with one required string parameter."""

    name = "echo"
    description = "Echoes back a message."
    permission = PermissionLevel.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="message", type="string", description="Text to echo")
        ]

    def execute(self, message: str = "") -> ToolResult:
        return ToolResult.success(data=message)


class AddTool(Tool):
    """Fake READ tool with two required number parameters."""

    name = "add"
    description = "Adds two numbers."
    permission = PermissionLevel.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="a", type="number", description="First number"),
            ToolParameter(name="b", type="number", description="Second number"),
        ]

    def execute(self, a: float = 0, b: float = 0) -> ToolResult:
        return ToolResult.success(data=a + b)


class PingTool(Tool):
    """Fake READ tool with no parameters at all."""

    name = "ping"
    description = "Does nothing, returns pong."
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(data="pong")


# --- ToolCall representation ---------------------------------------------------


def test_tool_call_holds_name_and_parameters():
    call = ToolCall(tool_name="echo", parameters={"message": "hi"})

    assert call.tool_name == "echo"
    assert call.parameters == {"message": "hi"}


def test_tool_call_parameters_default_to_an_empty_dict():
    call = ToolCall(tool_name="ping")

    assert call.parameters == {}


def test_tool_call_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ToolCallProvider()


# --- GeminiToolCallProvider: prompt/request shape -------------------------------


def test_decide_sends_the_message_and_candidate_tool_schemas():
    provider = FakeAIProvider(
        reply='{"tool_name": "echo", "parameters": {"message": "hi"}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    tool_call_provider.decide("say hi with echo", [EchoTool()])

    prompt = provider.last_message
    assert "say hi with echo" in prompt
    assert "echo" in prompt
    assert "message" in prompt


def test_decide_does_not_mention_permission_or_requirements_internals():
    provider = FakeAIProvider(
        reply='{"tool_name": "echo", "parameters": {"message": "hi"}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    tool_call_provider.decide("say hi with echo", [EchoTool()])

    prompt = provider.last_message
    assert "READ" not in prompt
    assert "permission" not in prompt.lower()
    assert "requirement" not in prompt.lower()


def test_decide_only_offers_the_given_candidate_tools():
    """A tool not in the candidate list is never mentioned - the provider
    can't even see it, let alone select it."""
    provider = FakeAIProvider(
        reply='{"tool_name": "echo", "parameters": {"message": "hi"}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    tool_call_provider.decide("say hi with echo", [EchoTool()])

    assert '"add"' not in provider.last_message


# --- GeminiToolCallProvider: valid structured output ----------------------------


def test_valid_natural_language_decision_becomes_a_tool_call():
    provider = FakeAIProvider(
        reply='{"tool_name": "echo", "parameters": {"message": "hello"}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    call = tool_call_provider.decide(
        "use the echo tool to say hello", [EchoTool(), AddTool()]
    )

    assert call == ToolCall(tool_name="echo", parameters={"message": "hello"})


def test_valid_parameters_for_a_two_parameter_tool():
    provider = FakeAIProvider(reply='{"tool_name": "add", "parameters": {"a": 2, "b": 3}}')
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    call = tool_call_provider.decide("add 2 and 3", [EchoTool(), AddTool()])

    assert call == ToolCall(tool_name="add", parameters={"a": 2, "b": 3})


def test_markdown_code_fence_is_tolerated_before_parsing():
    provider = FakeAIProvider(
        reply='```json\n{"tool_name": "echo", "parameters": {"message": "hi"}}\n```'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    call = tool_call_provider.decide("echo hi", [EchoTool()])

    assert call == ToolCall(tool_name="echo", parameters={"message": "hi"})


def test_parameters_key_may_be_omitted_for_a_no_parameter_tool():
    provider = FakeAIProvider(reply='{"tool_name": "ping"}')
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    call = tool_call_provider.decide("ping", [PingTool()])

    assert call == ToolCall(tool_name="ping", parameters={})


# --- GeminiToolCallProvider: strict validation, fails safely --------------------


def test_unknown_tool_name_is_rejected():
    provider = FakeAIProvider(
        reply='{"tool_name": "delete_everything", "parameters": {}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("do something", [EchoTool()])


def test_missing_tool_name_raises():
    provider = FakeAIProvider(reply='{"parameters": {"message": "hi"}}')
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("echo hi", [EchoTool()])


def test_missing_required_parameter_raises_incomplete_error():
    provider = FakeAIProvider(reply='{"tool_name": "echo", "parameters": {}}')
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallIncompleteError):
        tool_call_provider.decide("use the echo tool", [EchoTool()])


def test_invalid_parameter_type_raises():
    provider = FakeAIProvider(
        reply='{"tool_name": "add", "parameters": {"a": "two", "b": 3}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("add two and 3", [AddTool()])


def test_unexpected_parameter_key_raises():
    provider = FakeAIProvider(
        reply='{"tool_name": "echo", "parameters": {"message": "hi", "volume": "loud"}}'
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("echo hi loudly", [EchoTool()])


def test_extra_top_level_field_raises():
    provider = FakeAIProvider(
        reply=(
            '{"tool_name": "echo", "parameters": {"message": "hi"}, '
            '"confidence": 0.9}'
        )
    )
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("echo hi", [EchoTool()])


def test_malformed_json_raises():
    provider = FakeAIProvider(reply="sure! use the echo tool.")
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("echo hi", [EchoTool()])


def test_non_object_json_raises():
    provider = FakeAIProvider(reply='"echo"')
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("echo hi", [EchoTool()])


def test_non_object_parameters_raises():
    provider = FakeAIProvider(reply='{"tool_name": "echo", "parameters": "hi"}')
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError):
        tool_call_provider.decide("echo hi", [EchoTool()])


# --- GeminiToolCallProvider: provider failure -----------------------------------


def test_underlying_provider_failure_raises_tool_call_error():
    provider = FakeAIProvider(error=ProviderError("Gemini is down"))
    tool_call_provider = GeminiToolCallProvider(provider=provider)

    with pytest.raises(ToolCallError, match="Gemini is down"):
        tool_call_provider.decide("echo hi", [EchoTool()])


# --- Dependency injection / defaults --------------------------------------------


def test_default_provider_is_gemini_provider():
    from app.providers.gemini import GeminiProvider

    tool_call_provider = GeminiToolCallProvider()

    assert isinstance(tool_call_provider.provider, GeminiProvider)


def test_gemini_tool_call_provider_is_a_tool_call_provider():
    assert isinstance(GeminiToolCallProvider(), ToolCallProvider)


# --- ToolCallBuilder: deterministic fast path -----------------------------------


def test_zero_parameter_tool_is_built_deterministically_without_an_ai_call():
    registry = ToolRegistry()
    registry.register(PingTool())
    call_provider = FakeToolCallProvider()
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build("use tool ping", registry, known_tool_name="ping")

    assert call == ToolCall(tool_name="ping", parameters={})
    assert call_provider.call_count == 0


def test_single_required_string_parameter_is_filled_deterministically():
    registry = ToolRegistry()
    registry.register(EchoTool())
    call_provider = FakeToolCallProvider()
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build("use tool echo hello world", registry, known_tool_name="echo")

    assert call == ToolCall(tool_name="echo", parameters={"message": "hello world"})
    assert call_provider.call_count == 0


def test_deterministic_fill_strips_a_colon_separator():
    registry = ToolRegistry()
    registry.register(EchoTool())
    builder = ToolCallBuilder(call_provider=FakeToolCallProvider())

    call = builder.build("run tool: echo: hello", registry, known_tool_name="echo")

    assert call.parameters == {"message": "hello"}


def test_deterministic_fill_works_for_the_name_before_tool_phrasing():
    """M6.16: "use the echo tool ..." names the tool BEFORE the literal
    word "tool" (see app/capability.py's _TOOL_MENTION_PATTERN) - leftover
    extraction must use the text after "tool", not after the name, for
    this phrasing to fill deterministically at all."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    call_provider = FakeToolCallProvider()
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build(
        "use the echo tool to say hello world", registry, known_tool_name="echo"
    )

    assert call == ToolCall(
        tool_name="echo", parameters={"message": "to say hello world"}
    )
    assert call_provider.call_count == 0


def test_single_required_non_string_parameter_falls_back_to_ai():
    """A non-string required parameter isn't deterministically filled from
    raw leftover text - too risky a coercion - so it still goes to AI."""

    class DoubleTool(Tool):
        name = "double"
        description = "Doubles a number."
        permission = PermissionLevel.READ

        @property
        def parameters(self) -> list[ToolParameter]:
            return [ToolParameter(name="n", type="number")]

        def execute(self, n: float = 0) -> ToolResult:
            return ToolResult.success(data=n * 2)

    registry = ToolRegistry()
    registry.register(DoubleTool())
    call_provider = FakeToolCallProvider(
        tool_call=ToolCall(tool_name="double", parameters={"n": 4})
    )
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build("use tool double 4", registry, known_tool_name="double")

    assert call_provider.call_count == 1
    assert call == ToolCall(tool_name="double", parameters={"n": 4})


def test_missing_leftover_text_raises_incomplete_error_without_an_ai_call():
    registry = ToolRegistry()
    registry.register(EchoTool())
    call_provider = FakeToolCallProvider()
    builder = ToolCallBuilder(call_provider=call_provider)

    with pytest.raises(ToolCallIncompleteError):
        builder.build("use tool echo", registry, known_tool_name="echo")

    assert call_provider.call_count == 0


def test_multiple_required_parameters_go_to_ai_when_leftover_text_exists():
    registry = ToolRegistry()
    registry.register(AddTool())
    call_provider = FakeToolCallProvider(
        tool_call=ToolCall(tool_name="add", parameters={"a": 2, "b": 3})
    )
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build("use tool add 2 and 3", registry, known_tool_name="add")

    assert call_provider.call_count == 1
    assert call_provider.last_candidate_tools == [registry.get("add")]
    assert call == ToolCall(tool_name="add", parameters={"a": 2, "b": 3})


def test_unregistered_known_tool_name_is_passed_through_without_an_ai_call():
    """Deterministically constructing a ToolCall for a tool_name that
    isn't registered needs no special handling: ToolRegistry.execute's
    own existing "Unknown tool" error will report it."""
    registry = ToolRegistry()
    call_provider = FakeToolCallProvider()
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build(
        "use tool does_not_exist", registry, known_tool_name="does_not_exist"
    )

    assert call == ToolCall(tool_name="does_not_exist", parameters={})
    assert call_provider.call_count == 0


# --- ToolCallBuilder: natural-language / unknown-tool-name path -----------------


def test_unknown_tool_name_asks_ai_to_select_from_the_full_registry():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(AddTool())
    call_provider = FakeToolCallProvider(
        tool_call=ToolCall(tool_name="echo", parameters={"message": "hello"})
    )
    builder = ToolCallBuilder(call_provider=call_provider)

    call = builder.build("say hello using the echo tool", registry, known_tool_name=None)

    assert call_provider.call_count == 1
    assert {t.name for t in call_provider.last_candidate_tools} == {"echo", "add"}
    assert call == ToolCall(tool_name="echo", parameters={"message": "hello"})


def test_empty_registry_with_no_known_tool_name_raises_incomplete_error_without_an_ai_call():
    registry = ToolRegistry()
    call_provider = FakeToolCallProvider()
    builder = ToolCallBuilder(call_provider=call_provider)

    with pytest.raises(ToolCallIncompleteError):
        builder.build("use tool", registry, known_tool_name=None)

    assert call_provider.call_count == 0


def test_default_call_provider_is_gemini_backed():
    builder = ToolCallBuilder()

    assert isinstance(builder.call_provider, GeminiToolCallProvider)


def test_tool_call_provider_failure_propagates_from_the_builder():
    registry = ToolRegistry()
    registry.register(AddTool())
    call_provider = FakeToolCallProvider(error=ToolCallError("provider is down"))
    builder = ToolCallBuilder(call_provider=call_provider)

    with pytest.raises(ToolCallError, match="provider is down"):
        builder.build("use tool add 2 and 3", registry, known_tool_name="add")
