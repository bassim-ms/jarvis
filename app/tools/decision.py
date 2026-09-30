"""
Tool Decision Layer (M6.16): turns a TOOL capability decision (M6.13/M6.14)
into a concrete, validated ToolCall - which registered tool to run, and
with what parameters - without ever executing it. ToolRegistry
(app/tools/registry.py) remains the sole executor and the sole authority
on tool existence and permissions; nothing here calls `Tool.execute`.

Separate from CapabilityRouter (app/capability.py): that layer only ever
decides WHICH capability, and for TOOL may already know a concrete
tool_name from its own explicit invocation syntax. This layer picks up
from there and never re-decides the capability.

Deterministic-first, like every other decision layer in this codebase
(SearchRouter, DeterministicCapabilityRouter, SearchQueryBuilder): a tool
with no required parameters, or exactly one required string parameter the
message already spells out, is built with zero AI calls.
GeminiToolCallProvider (built entirely on the existing AIProvider/
GeminiProvider - no second client, no new config, no paid API) is only
consulted when parameter construction is genuinely ambiguous or the tool
itself isn't already known, and is given only each candidate tool's
name/description/parameter schema - never permissions, source code, or
conversation history. It never selects a tool outside what it was
offered, and never executes anything.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.tools.base import Tool
from app.tools.registry import ToolRegistry


@dataclass
class ToolCall:
    """A concrete request to run one registered tool - only ever
    constructed after validation (see ToolCallBuilder/
    GeminiToolCallProvider), never built directly from raw AI output."""

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)


class ToolCallError(RuntimeError):
    """Raised when a ToolCall can't be safely constructed - a failed AI
    call, or AI output that fails validation. Never silently repaired:
    JarvisCore re-raises this as ProviderError, the same convention
    CapabilityDecisionError/SearchAnswerError already use."""


class ToolCallIncompleteError(ToolCallError):
    """A TOOL request is understood but there isn't enough information to
    build a ToolCall yet - a required parameter is missing, or no tools
    are registered to choose from. Kept distinct from ToolCallError so
    JarvisCore can treat it as a normal "need more information" reply,
    not a 502 - the seam a future milestone can turn into an actual
    clarification workflow, without one existing yet."""


class ToolCallProvider(ABC):
    """Asks an AI to construct a ToolCall for a message ToolCallBuilder's
    deterministic fast path couldn't resolve on its own. Never executes
    anything, and has no knowledge of ToolRegistry or permissions."""

    @abstractmethod
    def decide(self, message: str, candidate_tools: list[Tool]) -> ToolCall:
        """Return a validated ToolCall choosing only from
        `candidate_tools` (never a tool outside this list), or raise
        ToolCallError/ToolCallIncompleteError."""


def _serialize_tool(tool: Tool) -> dict[str, Any]:
    """The minimum needed to select `tool` and build its parameters:
    name, description, and each ToolParameter's name/type/description/
    required. Excludes permission, requirements, and anything about the
    tool's Python implementation - none of that is ever sent."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
            }
            for p in tool.parameters
        ],
    }


# Deliberately small and fixed - only the candidate tool list and the
# user's message vary per call.
_TOOL_CALL_INSTRUCTIONS = (
    "You are a strict tool-call constructor for a personal assistant. "
    "You are given the user's message and a fixed list of available "
    "tools with their parameter schemas. Choose exactly one tool from "
    "the list and construct its parameters from the message.\n\n"
    "Respond with ONLY one JSON object, no prose, no markdown formatting, "
    'in exactly this shape: {"tool_name": "<one of the listed tool '
    'names>", "parameters": {"<param name>": <value>, ...}}. Never choose '
    "a tool that is not in the list below, and never invent a parameter "
    "name that isn't declared for the chosen tool. If a required "
    "parameter's value isn't present in the message, omit it entirely "
    "rather than inventing one.\n\n"
)

# Gemini often wraps JSON in a markdown code fence even when told not to;
# stripping it is tolerant formatting cleanup, not guessing at content -
# the JSON underneath still has to pass full validation below.
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_VALID_TOOL_CALL_KEYS = {"tool_name", "parameters"}

# ToolParameter.type is deliberately simple/provider-agnostic (plain
# strings, see app/tools/base.py) - this is a best-effort mapping for the
# common cases, not a full type system. An unrecognized type string is
# left unchecked rather than rejected.
_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "structured": (dict, list),  # any JSON object or list
}


def _matches_type(value: Any, declared_type: str) -> bool:
    expected = _TYPE_CHECKS.get(declared_type)
    if expected is None:
        return True
    if declared_type == "number" and isinstance(value, bool):
        return False  # bool is a subclass of int in Python
    return isinstance(value, expected)


def _validate_parameters(tool: Tool, parameters: dict[str, Any]) -> None:
    """Raise ToolCallError/ToolCallIncompleteError if `parameters` doesn't
    conform to `tool`'s own ToolParameter schema. Never drops, coerces, or
    repairs an invalid or missing value."""
    declared = {p.name: p for p in tool.parameters}

    unexpected = set(parameters) - set(declared)
    if unexpected:
        raise ToolCallError(
            f"Unexpected parameter(s) for tool '{tool.name}': {sorted(unexpected)}"
        )

    missing = [
        name
        for name, param in declared.items()
        if param.required and name not in parameters
    ]
    if missing:
        raise ToolCallIncompleteError(
            f"Missing required parameter(s) for tool '{tool.name}': "
            f"{', '.join(missing)}"
        )

    for name, value in parameters.items():
        param = declared[name]
        if not _matches_type(value, param.type):
            raise ToolCallError(
                f"Parameter '{name}' for tool '{tool.name}' must be of "
                f"type '{param.type}', got: {value!r}"
            )


class GeminiToolCallProvider(ToolCallProvider):
    """ToolCallProvider backed by the existing AIProvider abstraction - not
    tied to any one vendor despite the name (kept for backward
    compatibility with every existing caller/test). Defaults to a plain
    GeminiProvider() - the same free-tier client GeminiProvider (and
    GeminiSearchAnswerGenerator/GeminiCapabilityDecisionProvider) already
    use - when constructed with no arguments, exactly as before. Since
    M17.1-E, JarvisCore's own default `tool_call_builder` instead passes
    its own `self.provider` explicitly (see app/core.py), which in the
    real running app is an AIProviderManager - so `self.provider.
    generate_reply(prompt)` below already gets AIProviderManager's
    existing ordered retry/cooldown/fallback across every configured
    provider (Gemini -> Groq -> OpenRouter -> Cloudflare -> FreeLLMAPI) on
    a ProviderError, with no changes needed here: this class has always
    only ever depended on the generic AIProvider interface, never on
    GeminiProvider specifically.

    Strictly validates the response before constructing a ToolCall: valid
    JSON object with only tool_name/parameters, tool_name matching one of
    the offered candidates exactly, parameters a JSON object conforming
    to that tool's own schema (see _validate_parameters). Any failure
    raises ToolCallError/ToolCallIncompleteError - never a fallback guess,
    and never retried against another provider: `self.provider.
    generate_reply` returning successfully - Gemini, or the provider
    AIProviderManager fell back to - ends this call's only attempt at
    parsing/validation; a malformed reply from whichever provider
    answered is a ToolCallError, exactly as before M17.1-E.
    """

    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or GeminiProvider()

    def decide(self, message: str, candidate_tools: list[Tool]) -> ToolCall:
        tools_json = json.dumps([_serialize_tool(t) for t in candidate_tools])
        prompt = (
            _TOOL_CALL_INSTRUCTIONS
            + "Available tools: "
            + tools_json
            + "\n\nUser message: "
            + message
        )

        try:
            raw = self.provider.generate_reply(prompt)
        except ProviderError as exc:
            raise ToolCallError(f"Failed to construct a tool call: {exc}") from exc

        return self._parse_and_validate(raw, candidate_tools)

    @staticmethod
    def _parse_and_validate(raw: str, candidate_tools: list[Tool]) -> ToolCall:
        cleaned = _CODE_FENCE_PATTERN.sub("", raw.strip()).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ToolCallError(
                f"Invalid JSON from tool call provider: {raw!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise ToolCallError(f"Expected a JSON object, got: {raw!r}")

        extra_keys = set(payload) - _VALID_TOOL_CALL_KEYS
        if extra_keys:
            raise ToolCallError(f"Unexpected fields in tool call: {sorted(extra_keys)}")

        tool_name = payload.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolCallError(f"Missing or invalid 'tool_name' in tool call: {raw!r}")

        candidates_by_name = {tool.name: tool for tool in candidate_tools}
        tool = candidates_by_name.get(tool_name)
        if tool is None:
            raise ToolCallError(
                f"Tool '{tool_name}' is not one of the offered candidates: "
                f"{sorted(candidates_by_name)}"
            )

        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ToolCallError(f"'parameters' must be a JSON object, got: {parameters!r}")

        _validate_parameters(tool, parameters)

        return ToolCall(tool_name=tool_name, parameters=parameters)


_TOOL_WORD_PATTERN = re.compile(r"\btool\b", re.IGNORECASE)


def _leftover_text(message: str, tool_name: str) -> str:
    """Text remaining in `message` after the invocation phrase that named
    `tool_name` - used only to deterministically fill a tool's single
    required string parameter. The name can appear either right after the
    literal word "tool" ("use tool echo hello") or right before it ("use
    the echo tool to say hello") - see app/capability.py's TOOL detection
    - so this takes the text after whichever of the two appears LAST in
    the message, stripped of surrounding whitespace and a leading ':'/'-'
    separator. Returns "" if neither appears."""
    ends = [m.end() for m in re.finditer(rf"\b{re.escape(tool_name)}\b", message)]
    ends += [m.end() for m in _TOOL_WORD_PATTERN.finditer(message)]
    if not ends:
        return ""
    return message[max(ends) :].strip().lstrip(":-").strip()


class ToolCallBuilder:
    """Deterministic-first construction of a ToolCall from a
    TOOL-capability message. Mirrors SearchQueryBuilder: handle the
    common case with zero AI calls, and fall back to `call_provider`
    (defaulting to GeminiToolCallProvider()) only when genuinely
    necessary - never to re-discover a tool name CapabilityRouter already
    identified, and never more than once per message.

    `known_tool_name` is `decision.tool_name`. If it names an unregistered
    tool, that's passed through unchanged - ToolRegistry.execute's own
    "Unknown tool" handling reports it. If the tool needs no parameters,
    or exactly one required "string" parameter that the message spells
    out as trailing text, the ToolCall is built directly. Anything else
    (multiple required parameters, a non-string type, or no trailing text
    at all) either falls back to `call_provider` or - if there's nothing
    in the message to extract - raises ToolCallIncompleteError without
    even trying the AI.

    If `known_tool_name` is `None`, the full ToolRegistry contents are
    offered to `call_provider` for real tool selection, unless the
    registry is empty, in which case ToolCallIncompleteError is raised
    immediately.
    """

    def __init__(self, call_provider: ToolCallProvider | None = None):
        self.call_provider = call_provider or GeminiToolCallProvider()

    def build(
        self, message: str, registry: ToolRegistry, known_tool_name: str | None
    ) -> ToolCall:
        if known_tool_name is not None:
            return self._build_for_known_tool(message, registry, known_tool_name)

        candidates = registry.list_tools()
        if not candidates:
            raise ToolCallIncompleteError("No tools are registered to choose from.")

        return self.call_provider.decide(message, candidates)

    def _build_for_known_tool(
        self, message: str, registry: ToolRegistry, tool_name: str
    ) -> ToolCall:
        tool = registry.get(tool_name)
        if tool is None:
            return ToolCall(tool_name=tool_name, parameters={})

        required = [p for p in tool.parameters if p.required]
        if not required:
            return ToolCall(tool_name=tool_name, parameters={})

        leftover = _leftover_text(message, tool_name)
        if len(required) == 1 and required[0].type == "string" and leftover:
            return ToolCall(tool_name=tool_name, parameters={required[0].name: leftover})

        if not leftover:
            missing = ", ".join(p.name for p in required)
            raise ToolCallIncompleteError(
                f"Missing required parameter(s) for tool '{tool_name}': {missing}"
            )

        return self.call_provider.decide(message, [tool])
