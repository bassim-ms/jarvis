"""
Step references (M15): how one PlanStep's argument takes its value from an
earlier step's stored result - deterministically, with no AI in the loop.

A reference is a plain JSON object with exactly two keys:

    {"from_step": "s1", "path": "data.results.0.title"}

Two stages, both in this module so they can't drift apart:

  * `validate_reference` (plan time, called by
    app/orchestration/validator.py): shape, the source step's id, and the
    path's syntax. The validator additionally requires `from_step` to be
    listed in the referring step's own `depends_on`.
  * `resolve_reference` (execution time, called by JobEngine right before a
    step runs): walks the path through the source step's stored result
    exactly as written. A missing key, an out-of-range index, a path that
    steps into a scalar, or a source with no usable result is a
    `ReferenceError` - never a guess, never a default, never a retry.

The path grammar is deliberately tiny: dot-separated segments made of
letters, digits, `_` and `-`; the first segment must be one of the fields a
stored result actually has (`status`, `message`, `data`); a segment indexes
a list when the value is a list (a non-negative integer) and a dict
otherwise. There is no wildcard, filter, expression or function syntax.

Resolved values are BOUNDED (`bound_value`): a huge tool result can never be
blindly injected into another tool's arguments. A string or list/dict larger
than `MAX_REFERENCE_CHARS` is NOT truncated or pruned - cutting research data
could silently change its meaning - the reference simply fails, with a clear
size error, and the step fails through the normal bounded failure/replan
path. The limit is this one constant.

Referenced data is untrusted (it is whatever a tool returned): a reference
only ever supplies an argument VALUE. It can't name a tool, add a step,
change a dependency or grant any permission - those are fixed by the
validated plan and enforced by ToolRegistry as always.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Fields of a step's stored result (see app/orchestration/engine.py:
#: `_serialize_tool_result`) a path may start from.
RESULT_ROOTS = frozenset({"status", "message", "data"})

MAX_PATH_CHARS = 200
MAX_PATH_SEGMENTS = 8

#: Upper bound on a resolved value: the string length, or the serialized
#: JSON length of a list/dict. A larger value fails the reference (it is
#: never truncated). The single place this limit is defined.
MAX_REFERENCE_CHARS = 12000

_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
_STEP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_INDEX = re.compile(r"^[0-9]+$")

_REFERENCE_KEYS = frozenset({"from_step", "path"})


class ReferenceError(ValueError):
    """A step reference is malformed (plan time) or can't be resolved
    (execution time). Always a hard failure of the step - never guessed
    around. (Shadows the builtin deliberately: it IS this module's error.)"""


def looks_like_reference(value: Any) -> bool:
    """True for any dict that uses the reference key `from_step` - so a
    half-formed or padded reference is caught as malformed instead of
    silently passing through as an ordinary object argument."""
    return isinstance(value, dict) and "from_step" in value


def validate_reference(value: Any) -> tuple[str, tuple[str, ...]]:
    """Return (source step id, path segments) for a well-formed reference,
    or raise ReferenceError. Syntax only - whether the source step exists,
    is a declared dependency, or has a result is checked elsewhere."""
    if not isinstance(value, dict) or set(value) != _REFERENCE_KEYS:
        raise ReferenceError(
            'a step reference must be exactly {"from_step": "<step id>", "path": "<path>"}'
        )

    from_step, path = value["from_step"], value["path"]
    if not isinstance(from_step, str) or not _STEP_ID.match(from_step):
        raise ReferenceError(f"invalid from_step in reference: {from_step!r}")
    if not isinstance(path, str) or not path or len(path) > MAX_PATH_CHARS:
        raise ReferenceError(
            f"reference path must be a non-empty string of at most {MAX_PATH_CHARS} characters"
        )

    segments = tuple(path.split("."))
    if len(segments) > MAX_PATH_SEGMENTS:
        raise ReferenceError(f"reference path has more than {MAX_PATH_SEGMENTS} segments: {path!r}")
    if any(not _SEGMENT.match(segment) for segment in segments):
        raise ReferenceError(
            f"reference path {path!r} may only contain letters, digits, '_' and '-', "
            "separated by '.'"
        )
    if segments[0] not in RESULT_ROOTS:
        raise ReferenceError(
            f"reference path {path!r} must start with one of {sorted(RESULT_ROOTS)}"
        )
    return from_step, segments


def resolve_path(result: Any, segments: tuple[str, ...], path: str) -> Any:
    """Walk `segments` through `result` exactly as written."""
    current = result
    walked: list[str] = []
    for segment in segments:
        walked.append(segment)
        where = ".".join(walked)
        if isinstance(current, dict):
            if segment not in current:
                raise ReferenceError(f"reference path {path!r}: no key '{segment}' (at '{where}')")
            current = current[segment]
        elif isinstance(current, list):
            if not _INDEX.match(segment):
                raise ReferenceError(
                    f"reference path {path!r}: '{segment}' is not a list index (at '{where}')"
                )
            index = int(segment)
            if index >= len(current):
                raise ReferenceError(
                    f"reference path {path!r}: index {index} is out of range for a list of "
                    f"{len(current)} (at '{where}')"
                )
            current = current[index]
        else:
            raise ReferenceError(
                f"reference path {path!r}: cannot step into a "
                f"{type(current).__name__} with '{segment}' (at '{where}')"
            )
    return current


def bound_value(value: Any, limit: int = MAX_REFERENCE_CHARS) -> Any:
    """Return `value` unchanged if it fits within `limit`, else raise
    ReferenceError. Never truncates, prunes or otherwise alters a value: a
    string is measured by length, a list/dict by its serialized JSON length,
    and scalars other than strings always fit."""
    if isinstance(value, str):
        size = len(value)
    elif isinstance(value, (dict, list)):
        size = len(json.dumps(value, default=str))
    else:
        return value
    if size > limit:
        raise ReferenceError(
            f"the referenced value is too large to pass on ({size} characters; "
            f"the limit is {limit}) - it is not truncated"
        )
    return value


def resolve_reference(reference: dict, source_result: dict[str, Any] | None) -> Any:
    """Resolve a validated reference against its source step's stored
    result and return the value, exactly as stored. Raises ReferenceError if
    there is no result, the path doesn't lead to a value, or the value
    exceeds `MAX_REFERENCE_CHARS`. `source_result` is the
    `{status, message, data}` dict a SUCCEEDED step stores."""
    _from_step, segments = validate_reference(reference)
    path = reference["path"]
    if not source_result:
        raise ReferenceError(
            f"step '{reference['from_step']}' has no stored result to reference"
        )
    return bound_value(resolve_path(source_result, segments, path))
