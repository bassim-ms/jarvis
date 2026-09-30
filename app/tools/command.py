"""
Explicit tool-command understanding.

Turns an explicit, command-like message ("use tool open_url https://luniti.app,
then use tool scroll -600") into a structured, fully validated, immutable
`ToolSequence` - or into a `CommandRejection` explaining why nothing may run.
JarvisCore executes the ToolSequence and never re-reads the sentence.

Pipeline (all of it is PHASE 1 - nothing here ever executes a tool):

    message
      -> tokenize            word / quoted / comma tokens, with source offsets
      -> split into steps    "then" / "and then" / "followed by" (optional
                             leading comma), recognized only as unquoted
                             tokens - never inside a quoted string
      -> resolve each tool   registry lookup by name; unknown -> issue
      -> bind each argument  positionally, against THAT tool's own
                             ToolParameter schema (see `_bind_arguments`)
      -> validate each step  schema (types/missing/unexpected), the tool's
                             own side-effect-free `validate_arguments`, and
                             the registry's permission gate
      -> ToolSequence        only if EVERY step passed; otherwise the whole
                             command is rejected and NOTHING executes

Argument boundaries come from the schema, not from the sentence: a number
is one numeric token; a boolean is one true/false token; a string is ONE
token or ONE quoted string unless its ToolParameter declares
`free_text=True` (prose), in which case it may span the rest of its step.
Whatever a step's arguments do not consume is not ignored - it is a
rejection. There are no per-tool or per-URL rules anywhere in this module:
adding a tool means declaring its ToolParameters, nothing else.

An LLM is never asked to resolve a deterministic argument. The only AI use
is for a tool that explicitly declares `natural_language_arguments=True`
(prose tools such as create_reminder), via the existing ToolCallBuilder,
and its answer goes through the same validation as everything else.

This module is registry-agnostic: which registry holds a given tool
(interactive computer tools vs M15's registry) is decided by the caller's
`resolve_registry`, exactly as before - nothing here can widen what an
explicit command may reach.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.tools.base import Tool, ToolArgumentError
from app.tools.decision import (
    ToolCallBuilder,
    ToolCallError,
    ToolCallIncompleteError,
    _validate_parameters,
)
from app.tools.registry import ToolRegistry

# --- invocation head ("use tool X" / "use the X tool") ----------------------

_VERBS = r"(?:use|run|call|invoke)"

#: "use tool [:] NAME" - NAME is captured raw (anything up to whitespace or
#: a comma) and judged in Python, so "use tool 5" is an INVALID NAME to
#: reject, not "nothing named" to hand to an AI.
_HEAD_TOOL_FIRST = re.compile(
    rf"^\s*{_VERBS}\s+tool(?![\w-])\s*:?\s*(?P<name>[^\s,]+)?", re.IGNORECASE
)
#: "use [the|a|an] NAME tool" - the mirror phrasing. The lookahead stops the
#: optional article from backtracking into NAME ("use a tool to..." is not
#: a tool named "a").
_HEAD_NAME_FIRST = re.compile(
    rf"^\s*{_VERBS}\s+(?:(?:the|a|an)\s+)?(?!(?:the|a|an)\b)"
    r"(?P<name>[^\s,]+)\s+tool(?![\w-])",
    re.IGNORECASE,
)
#: Steps after the first may omit the invocation words: "..., then scroll -600".
_HEAD_BARE = re.compile(r"^\s*(?P<name>[^\s,]+)")

_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class InvocationHead:
    """The "use tool NAME" part of a step. `tool_name` is the cleaned name
    (trailing ':' removed), or None when the phrasing named nothing.
    `end` is the offset in the matched text where arguments begin."""

    tool_name: str | None
    end: int


def _clean_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw[:-1] if raw.endswith(":") else raw
    return cleaned or None


def match_invocation_head(text: str) -> InvocationHead | None:
    """Return the InvocationHead if `text` STARTS with an explicit tool
    invocation phrasing, else None. The one definition of that phrasing -
    the capability router uses it too, so router and parser can't drift."""
    match = _HEAD_TOOL_FIRST.match(text) or _HEAD_NAME_FIRST.match(text)
    if match is None:
        return None
    return InvocationHead(tool_name=_clean_name(match.group("name")), end=match.end())


# --- tokenizer ---------------------------------------------------------------

_WORD, _QUOTED, _COMMA = "word", "quoted", "comma"
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str  # word text, unquoted inner text, or ","
    start: int
    end: int  # offsets into the original message


class _Malformed(ValueError):
    pass


def _tokenize(message: str) -> list[_Token]:
    """Split into word / quoted / comma tokens. A quote only OPENS a quoted
    token at the start of a token ("don't" is one word). A comma is its own
    token only when it ends a clause (followed by whitespace/end, or at a
    token start); a comma inside a word ("1,000", "a=1,2") stays in it.
    Raises _Malformed for an unterminated quote or text glued to a closing
    quote - never guesses."""
    tokens: list[_Token] = []
    n = len(message)
    i = 0
    while i < n:
        ch = message[i]
        if ch.isspace():
            i += 1
        elif ch == ",":
            tokens.append(_Token(_COMMA, ",", i, i + 1))
            i += 1
        elif ch in _QUOTE_PAIRS:
            close = _QUOTE_PAIRS[ch]
            j = i + 1
            chars: list[str] = []
            while j < n:
                c = message[j]
                if c == "\\" and j + 1 < n and message[j + 1] in (close, "\\"):
                    chars.append(message[j + 1])
                    j += 2
                    continue
                if c == close:
                    break
                chars.append(c)
                j += 1
            else:
                raise _Malformed(f"unterminated quote starting at: {message[i:i + 20]!r}")
            j += 1  # past the closing quote
            if j < n and not message[j].isspace() and message[j] != ",":
                raise _Malformed(
                    f"unexpected text directly after a closing quote: {message[i:j + 10]!r}"
                )
            tokens.append(_Token(_QUOTED, "".join(chars), i, j))
            i = j
        else:
            j = i
            while j < n and not message[j].isspace():
                if message[j] == "," and (j + 1 >= n or message[j + 1].isspace()):
                    break  # clause-ending comma
                j += 1
            tokens.append(_Token(_WORD, message[i:j], i, j))
            i = j
    return tokens


# --- step separators ---------------------------------------------------------


def _separator_length(tokens: list[_Token], i: int) -> int:
    """Number of tokens, starting at `i`, that form a step separator
    ([","] then | and then | followed by), or 0. Only UNQUOTED word tokens
    can be a separator."""
    j = i
    if j < len(tokens) and tokens[j].kind == _COMMA:
        j += 1

    def words(*expected: str) -> bool:
        return j + len(expected) <= len(tokens) and all(
            tokens[j + k].kind == _WORD and tokens[j + k].text.lower() == word
            for k, word in enumerate(expected)
        )

    if words("then"):
        return j + 1 - i
    if words("and", "then") or words("followed", "by"):
        return j + 2 - i
    return 0


def _split_steps(tokens: list[_Token]) -> tuple[list[list[_Token]], list[str]]:
    """Split on separators. Returns (steps, problems); an empty step (a
    leading/trailing/doubled separator) is a problem, never dropped."""
    steps: list[list[_Token]] = [[]]
    problems: list[str] = []
    i = 0
    while i < len(tokens):
        length = _separator_length(tokens, i)
        if length:
            steps.append([])
            i += length
        else:
            steps[-1].append(tokens[i])
            i += 1
    for index, step in enumerate(steps, start=1):
        if not step:
            problems.append(
                f"step {index} is empty - each 'then' / 'and then' / "
                "'followed by' must be followed by a tool"
            )
    return steps, problems


# --- argument binding --------------------------------------------------------

_NUMBER = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)$")
_BOOLEANS = {"true": True, "yes": True, "false": False, "no": False}
_MAX_NUMBER_CHARS = 30


class _Unbindable(ValueError):
    pass


def _bind_value(param, token: _Token) -> Any:
    if token.kind == _COMMA:
        raise _Unbindable(f"expected a value for '{param.name}' but found a comma")

    if param.type == "number":
        if token.kind != _WORD or not _NUMBER.match(token.text) or len(token.text) > _MAX_NUMBER_CHARS:
            raise _Unbindable(
                f"'{param.name}' must be a number, got: {token.text!r}"
            )
        text = token.text
        return float(text) if "." in text else int(text)

    if param.type == "boolean":
        if token.kind != _WORD or token.text.lower() not in _BOOLEANS:
            raise _Unbindable(
                f"'{param.name}' must be true or false, got: {token.text!r}"
            )
        return _BOOLEANS[token.text.lower()]

    if param.type == "string":
        return token.text

    raise _Unbindable(
        f"'{param.name}' (type '{param.type}') can't be given in an explicit command"
    )


def _bind_arguments(
    tool: Tool,
    tokens: list[_Token],
    message: str,
    lookup: Callable[[str], Tool | None],
) -> dict[str, Any]:
    """Bind a step's argument tokens to `tool.parameters`, positionally.

    Raises _Unbindable with a precise reason. Never invents a value,
    never drops text: whatever the declared parameters don't consume is an
    error."""
    declared = list(tool.parameters)
    params: dict[str, Any] = {}
    i = 0

    for pos, param in enumerate(declared):
        if i >= len(tokens):
            break
        token = tokens[i]

        if param.type == "string" and param.free_text and token.kind == _WORD:
            if any(later.required for later in declared[pos + 1 :]):
                raise _Unbindable(
                    f"ambiguous: '{param.name}' is free text but more required "
                    "arguments follow it - put its value in quotes"
                )
            params[param.name] = message[token.start : tokens[-1].end]
            i = len(tokens)
            break

        params[param.name] = _bind_value(param, token)
        i += 1

    missing = [p.name for p in declared if p.required and p.name not in params]
    if missing:
        raise _Unbindable("missing required argument(s): " + ", ".join(missing))

    if i < len(tokens):
        raise _Unbindable(_leftover_message(tool, tokens[i:], message, lookup))

    return params


def _leftover_message(
    tool: Tool,
    leftover: list[_Token],
    message: str,
    lookup: Callable[[str], Tool | None],
) -> str:
    shown = message[leftover[0].start : leftover[-1].end].strip(" ,")
    shown = shown or message[leftover[0].start : leftover[-1].end]
    first_word = next((t.text for t in leftover if t.kind == _WORD), None)

    hint = ""
    if first_word is not None:
        if lookup(first_word) is None:
            hint = (
                f' If "{first_word}" was meant as another step, it is not an '
                "available tool."
            )
        else:
            hint = f' "{first_word}" is a tool, but a step needs a separator before it.'
    return (
        f'unexpected text after the arguments of {tool.name}: "{shown}".{hint} '
        'Steps are joined with "then", "and then" or "followed by"; otherwise '
        "remove the extra text, or quote a multi-word value"
    )


# --- public result types -----------------------------------------------------


@dataclass(frozen=True)
class StepIssue:
    """Why one step (or, with `index=None`, the whole command) is invalid."""

    index: int | None
    tool_name: str | None
    message: str

    def render(self) -> str:
        message = self.message if self.message.endswith((".", "!", "?")) else self.message + "."
        if self.index is None:
            return message
        label = f"Step {self.index}" + (f" ({self.tool_name})" if self.tool_name else "")
        return f"{label}: {message}"


@dataclass(frozen=True)
class CommandRejection:
    """The command was recognized as an explicit tool command but is not
    valid as a whole. NOTHING was executed."""

    issues: tuple[StepIssue, ...]

    def render(self) -> str:
        if len(self.issues) == 1:
            return f"{self.issues[0].render()} Nothing was executed."
        lines = "\n".join(f"- {issue.render()}" for issue in self.issues)
        return f"I couldn't run that tool command, so nothing was executed:\n{lines}"


@dataclass(frozen=True)
class ParsedStep:
    """A step whose tool exists. `arguments` is None only for a
    `natural_language_arguments` tool, whose arguments the planner still
    has to extract from `clause` (this step's own text, never another's)."""

    index: int
    tool: Tool
    arguments: dict[str, Any] | None
    clause: str


@dataclass(frozen=True)
class ToolStep:
    """One fully resolved, validated step: exactly what will be executed."""

    index: int
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    clause: str = ""


@dataclass(frozen=True)
class ToolSequence:
    """The validated, ordered source of truth for execution."""

    steps: tuple[ToolStep, ...]


@dataclass(frozen=True)
class ParsedCommand:
    steps: tuple[ParsedStep, ...]


def parse_explicit_command(
    message: str, lookup: Callable[[str], Tool | None]
) -> ParsedCommand | CommandRejection | None:
    """Parse `message` if - and only if - it is an explicit tool command.

    Returns None when it is not one (ordinary prose, "use a tool to ...",
    or a bare "use tool" that names nothing - which keeps its existing
    handling); a CommandRejection when it is one but any step is invalid;
    otherwise a ParsedCommand. Purely textual plus registry LOOKUPS: never
    executes anything and never calls an AI."""
    if match_invocation_head(message) is None:
        return None

    try:
        tokens = _tokenize(message)
    except _Malformed as exc:
        return CommandRejection((StepIssue(None, None, f"Malformed tool command: {exc}"),))

    steps, problems = _split_steps(tokens)
    issues: list[StepIssue] = [
        StepIssue(None, None, f"Malformed tool sequence: {p}") for p in problems
    ]
    parsed: list[ParsedStep] = []

    for index, step_tokens in enumerate(steps, start=1):
        if not step_tokens:
            continue
        text = message[step_tokens[0].start : step_tokens[-1].end]

        head = match_invocation_head(text)
        if head is None and index > 1:
            match = _HEAD_BARE.match(text)
            head = InvocationHead(tool_name=_clean_name(match.group("name")), end=match.end()) if match else None
        if head is None:  # first step always matches; defensive
            issues.append(StepIssue(index, None, "didn't start with a tool invocation."))
            continue

        if head.tool_name is None:
            if len(steps) == 1:
                return None  # a bare "use tool": existing handling
            issues.append(StepIssue(index, None, "didn't name a tool."))
            continue
        name = head.tool_name
        if not _TOOL_NAME.match(name):
            issues.append(StepIssue(index, None, f"{name!r} is not a valid tool name."))
            continue

        tool = lookup(name)
        if tool is None:
            hint = ""
            if (
                parsed
                and parsed[-1].index == index - 1
                and any(p.free_text for p in parsed[-1].tool.parameters)
            ):
                hint = (
                    f" If it was part of the text for step {parsed[-1].index} "
                    f"({parsed[-1].tool.name}), put that text in quotes."
                )
            issues.append(
                StepIssue(
                    index, name, f"Unknown tool: '{name}'. It is not an available tool.{hint}"
                )
            )
            continue

        abs_head_end = step_tokens[0].start + head.end
        if any(t.start < abs_head_end < t.end for t in step_tokens):
            issues.append(
                StepIssue(index, name, "unexpected text directly attached to the tool name.")
            )
            continue
        arg_tokens = [t for t in step_tokens if t.start >= abs_head_end]

        if tool.natural_language_arguments and (
            arg_tokens or any(p.required for p in tool.parameters)
        ):
            parsed.append(ParsedStep(index, tool, None, text))
            continue

        try:
            arguments = _bind_arguments(tool, arg_tokens, message, lookup)
        except _Unbindable as exc:
            issues.append(StepIssue(index, name, f"{exc}"))
            continue
        parsed.append(ParsedStep(index, tool, arguments, text))

    if issues:
        return CommandRejection(tuple(issues))
    return ParsedCommand(tuple(parsed))


# --- planner: parse + resolve + validate the WHOLE sequence -------------------


def _check_step(
    tool: Tool, parameters: dict[str, Any], registry: ToolRegistry
) -> str | None:
    """Everything that must hold before this step may run, except running
    it. Returns the reason it may not, or None."""
    try:
        _validate_parameters(tool, parameters)
    except ToolCallError as exc:
        return str(exc)

    try:
        tool.validate_arguments(**parameters)
    except ToolArgumentError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - fail closed on any validator fault
        return f"its arguments could not be validated ({exc})"

    blocked = registry.check_permission(tool.name)
    if blocked is not None:
        return blocked.message
    return None


class ExplicitToolCommandPlanner:
    """PHASE 1 for an explicit tool command: parse, resolve every tool,
    resolve every argument, validate every step (schema, tool-level value
    check, permission) - and only if ALL of it holds, return a ToolSequence.

    `resolve_registry(name)` is JarvisCore._resolve_tool_registry: which
    registry owns a tool is application-controlled, never derived from the
    message. `tool_call_builder` is consulted ONLY for
    `natural_language_arguments` tools.

    `plan` never executes a tool. A ToolCallError from the AI path (a
    failed provider call, malformed AI output) propagates unchanged so the
    caller can apply its existing 502 convention - safe here because
    nothing has executed yet."""

    def __init__(
        self,
        resolve_registry: Callable[[str | None], ToolRegistry],
        tool_call_builder: ToolCallBuilder,
    ):
        self.resolve_registry = resolve_registry
        self.tool_call_builder = tool_call_builder

    def _lookup(self, name: str) -> Tool | None:
        return self.resolve_registry(name).get(name)

    def plan(self, message: str) -> ToolSequence | CommandRejection | None:
        parsed = parse_explicit_command(message, self._lookup)
        if parsed is None or isinstance(parsed, CommandRejection):
            return parsed

        steps: list[ToolStep] = []
        issues: list[StepIssue] = []
        for step in parsed.steps:
            registry = self.resolve_registry(step.tool.name)
            parameters = step.arguments
            if parameters is None:
                try:
                    call = self.tool_call_builder.build(step.clause, registry, step.tool.name)
                except ToolCallIncompleteError as exc:
                    issues.append(StepIssue(step.index, step.tool.name, f"{exc}"))
                    continue
                parameters = call.parameters

            reason = _check_step(step.tool, parameters, registry)
            if reason is not None:
                issues.append(StepIssue(step.index, step.tool.name, reason))
                continue
            steps.append(ToolStep(step.index, step.tool.name, dict(parameters), step.clause))

        if issues:
            return CommandRejection(tuple(issues))
        return ToolSequence(tuple(steps))
