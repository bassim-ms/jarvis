"""
SelfDevelopmentEngine (V1) - the ONLY thing in this package that calls
its own ToolRegistry. Drives a DevelopmentSession through its bounded
state machine (app/self_development/base.py:SessionStage): propose a
plan -> wait for approval -> create a branch -> [if the plan touches a
protected path, wait for a SEPARATE SECURITY_EDIT approval] -> edit ->
test -> a SMALL, BOUNDED number of repair attempts -> summarize -> wait
for a SEPARATE commit approval -> commit (or rollback).

This is a controlled development SUBSYSTEM, not a second M15
(app/orchestration/): there is no Plan/PlanStep/Job here, no
dependency graph, no background poller. Every actual action is one
call to `self.tools.execute(name, params=..., confirmed=...)` - the
EXACT SAME PermissionLevel/confirmation enforcement every other tool
in this codebase already goes through (app/tools/registry.py). This
engine decides WHEN to pass `confirmed=True` - only in direct response
to a real approval/confirmation call this class exposes (`approve_plan`,
`approve_security_edit`, `approve_commit`, `rollback`) - it never
grants itself permission or fabricates an approval that didn't happen.

SECURITY_EDIT boundary (hardening pass, added after a security review
found nothing protected this engine's OWN source from being an
ordinary edit target): approving a plan's ordinary CODE_EDIT via
`approve_plan` NEVER authorizes touching a protected system path
(app/self_development/safety.py:is_protected_system_path). If a plan's
`files_affected` includes one, `approve_plan` creates the branch and
STOPS at SessionStage.AWAITING_SECURITY_APPROVAL - `_apply_plan` is
never called until a completely separate `approve_security_edit(
confirmed=True)` call resolves it. The LLM that PROPOSED the plan has
no path to satisfy this itself: `required_permissions` in a
DevelopmentPlan is validated (app/self_development/base.py) but is
descriptive of what the plan needs, never an approval grant, and
`approve_security_edit` only ever proceeds on an explicit `confirmed`
argument supplied by whatever is driving the engine - never inferred
from the plan, the request, or any AI output.

`self.tools` is a SEPARATE ToolRegistry from JarvisCore's own
`self.tools` (app/core.py) - these development tools are never
reachable through the ordinary TOOL capability/ToolCallBuilder path,
and JarvisCore's own tools are never reachable from here either. The
only bridge between the two systems is JarvisCore calling
`SelfDevelopmentEngine.start()`/`approve_plan()`/etc. directly (see
app/core.py's own self-development wiring).
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import uuid
from pathlib import Path

from app.self_development.base import (
    ABANDONED_SESSION_STAGES,
    CheckRun,
    CommandResult,
    DevelopmentPlan,
    DevelopmentSession,
    DevelopmentSessionError,
    DevelopmentSummary,
    SessionStage,
)
from app.self_development.git import GitClient, branch_name_for
from app.self_development.planner import DevelopmentPlanner, DevelopmentPlanningError
from app.self_development.safety import (
    PathSafetyError,
    PathTraversalError,
    is_protected_system_path,
    resolve_within_root,
)
from app.tools.base import ToolResultStatus
from app.tools.registry import ToolRegistry

#: Failure output bounded before it's ever handed to the planner for a
#: repair attempt or included in a summary - never an unbounded dump.
_MAX_FAILURE_TEXT_CHARS = 8_000

# ---------------------------------------------------------------------------
# Plan grounding (hardening pass 2, revised) - a live test showed the real
# planner proposing a plan that invented a whole new file (app/status.py)
# AND hedged with "if the engine has no status attribute, return a generic
# placeholder" for a request whose real answer already existed in the
# repository (DevelopmentSession.stage). The FIRST grounding pass
# (`_find_ungrounded_file_paths`'s own "a new file's parent directory must
# exist" check) doesn't catch this - `app/` genuinely exists.
#
# Deliberately NOT a full-repository text scan (an earlier version of this
# fix read every non-secret file's content on every `start()` call - slow,
# imprecise, and still easily fooled by generic English words appearing
# unrelated to what's actually there). Instead, a cheap two-stage funnel:
#
# 1. `_find_candidate_files` - a PATH-only filter (no file content read at
#    all) over the SAME file listing `_build_repo_digest` already uses,
#    narrowing to the small set of .py files whose own path already
#    relates to the request's own words - e.g. "...self-development
#    engine status..." naturally narrows to app/self_development/*.py,
#    the same way "add a voice command" would narrow to app/voice/*.py.
#    Bounded by `_MAX_CANDIDATE_FILES`; reads NOTHING when no path
#    matches at all.
# 2. `_discover_symbols_in_file` - for ONLY that narrow candidate set,
#    parses real Python source with `ast` (never a raw-text substring
#    search) and extracts the REAL classes/functions/methods/attributes
#    actually defined there - e.g. DevelopmentSession's own `stage`
#    field. This is the "relevant existing symbols/classes/methods"
#    evidence handed to the planner, and the same evidence
#    `_find_ungrounded_file_paths` uses afterward to verify a proposed
#    new file's own description names something REAL, never merely
#    restates the request in isolation.
#
# `_find_hedged_language` remains as a SEPARATE, general backstop (not the
# primary mechanism) - a plan can still fail this even when it happens to
# reuse a real symbol name, if it also admits to guessing about it.
# ---------------------------------------------------------------------------

#: A repo-relative path/file-content keyword must be at least this long
#: to count - short, generic fragments ("dev", "get", "app") match far
#: too much to mean anything.
_MIN_KEYWORD_LENGTH = 4

#: Words about the ACT of asking (verbs/pronouns/framing), never
#: domain-specific - excluded so the keyword set stays focused on what
#: the request is actually ABOUT. Deliberately small and reviewable, the
#: same "narrow, explicit, no fuzzy matching" discipline every other
#: pattern in this package already uses. Domain nouns ("self,"
#: "development," "engine," ...) are deliberately NOT stopworded here -
#: they're exactly what makes e.g. "...self-development engine
#: status..." path-match app/self_development/*.py in the first place.
_KEYWORD_STOPWORDS = frozenset({
    "your", "code", "codebase", "this", "that", "with", "from", "into",
    "file", "files", "please", "harmless", "current", "returns",
    "return", "endpoint", "about", "what", "does", "have", "should",
    "would", "could", "make", "need", "want", "then", "when", "where",
    "will", "task", "request",
})

#: How many keyword-matched candidate files are actually read/parsed -
#: bounded so a very generic keyword can't blow this up into "most of
#: the repository," while comfortably covering one real package/module.
_MAX_CANDIDATE_FILES = 12

#: Identifier word-fragments too generic, standing ALONE, to count as
#: meaningful grounding evidence for a brand-new file in
#: `_find_ungrounded_file_paths` - a live test showed a real, UNRELATED
#: method (GitClient's own `status_porcelain`, plain git-status
#: plumbing) "grounding" an invented, unrelated `/dev/status` endpoint
#: file purely because both happened to share the bare word "status."
#: This is the SYMBOL side of the same problem `_KEYWORD_STOPWORDS`
#: (the REQUEST side) already guards against - deliberately small,
#: fixed, and reviewable, never a semantic/fuzzy judgment.
_GENERIC_IDENTIFIER_FRAGMENTS = frozenset({
    "status", "engine", "test", "self", "development", "sync", "run",
    "data", "value", "manager", "handler", "service", "system",
    "module", "helper", "utility", "base", "core", "main", "file",
    "path", "name", "type", "result", "error", "event", "log",
})

# ---------------------------------------------------------------------------
# Edit-stage bounding (hardening pass 3, revised) - a live test showed an
# approved, well-grounded plan still corrupting app/main.py at the EDIT
# stage: the model's full-file-rewrite response dumped hundreds of lines
# of its own reasoning into the file, made unrelated changes elsewhere,
# and produced a much larger rewrite than requested. A first fix (syntax
# validation + a similarity-ratio backstop, still below) caught THAT
# shape of failure but not a subtler one a SECOND live test then showed:
# a well-formed, syntactically-VALID rewrite that still deleted an
# unrelated existing endpoint, reformatted unrelated code, and invented
# unrequested models/exception handling - similar enough overall to stay
# above the similarity threshold, while still touching code nobody asked
# to change. Grounding (which files/attributes a PLAN may name) and
# edit-bounding (whether a PROPOSED EDIT's actual content is trustworthy)
# remain separate concerns - this section is the second one.
#
# The actual fix: `propose_file_edits` (planner.py) no longer asks for
# "the complete new file" at all - it asks for a small set of EXACT
# find-and-replace edits against the real current content. `_apply_file_
# edits` (below) applies them itself, against the REAL current content
# this engine already has in hand - never the model's own claim of what
# the "old" text was. An edit whose `old_text` doesn't match verbatim, or
# matches more than once (ambiguous), is rejected outright; unrelated
# code the model never quotes can never be touched AT ALL, structurally,
# not merely by being asked nicely not to. This is what "structured and
# bounded, not an unrestricted full-file rewrite" (V1's own explicit
# requirement) actually means in practice.
#
# `_content_is_syntactically_valid` and `_is_change_proportionate` remain
# as a SECOND, independent layer, applied to the FINAL content after
# edits are applied - still real, still useful (an edit set can still
# reconstruct something syntactically broken, or - bounded per-edit size
# aside - collectively add up to something disproportionate), just no
# longer the primary defense.
#
# No check here ever silently repairs or trims a bad response - every one
# REJECTS it outright (no file write happens at all, the ORIGINAL file is
# untouched) and logs why, the same fail-closed discipline every other
# structural check in this package already uses.
# ---------------------------------------------------------------------------

#: An individual edit's own old_text/new_text is bounded - never large
#: enough, by itself, to plausibly BE "most of the file" dressed up as a
#: single "exact match" quote. Generous enough for a real multi-line
#: function (existing or new), firmly smaller than a substantial file
#: like the real app/main.py (255 lines, ~7,000 characters).
_MAX_EDIT_TEXT_CHARS = 4_000

#: How many separate edits a single file-edit proposal may contain -
#: bounded so "many small edits" can't be used to add up to a de facto
#: full rewrite either.
_MAX_EDITS_PER_FILE = 6

#: The COMBINED old_text across every edit for one existing file must
#: not exceed this fraction of the file's own current size (with a
#: small floor for tiny files) - the per-edit cap above stops any ONE
#: edit from being a disguised bulk rewrite, but several individually-
#: small edits could otherwise still add up to touching most of a large
#: file; this bounds the total, matching "surgical" in aggregate, not
#: just per edit.
_MAX_TOTAL_OLD_TEXT_RATIO = 0.3
_MIN_TOTAL_OLD_TEXT_FLOOR = 300

#: How much of a file's own original content a proposed edit must still
#: resemble (0.0-1.0, via difflib.SequenceMatcher's own line-based
#: similarity ratio) - never applied to a genuinely NEW file (nothing to
#: stay proportionate to). Deliberately generous (a real, legitimate
#: multi-part edit can still change a meaningful fraction of a file) -
#: this exists to catch a full, unrelated rewrite, not to police normal
#: editing.
_MIN_UNCHANGED_SIMILARITY = 0.5

#: A file this small is normally a deliberate stub/placeholder - fully
#: replacing it (e.g. a 1-line "# empty" file becoming a real function)
#: is the ORDINARY, expected shape of a first edit, not a runaway
#: rewrite, so the similarity check below only applies once a file is
#: already substantial enough that "barely resembles the original"
#: actually means something.
_MIN_LINES_FOR_PROPORTIONALITY_CHECK = 15


def _content_is_syntactically_valid(path: str, content: str) -> bool:
    """True for any non-Python path (nothing here to check), or for
    Python source that actually parses. False - never an exception -
    for anything else, including reasoning/prose text a model wrote
    instead of code."""
    if not path.endswith(".py"):
        return True
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True


def _is_change_proportionate(old_content: str, new_content: str) -> bool:
    """A bounded, structured edit still closely resembles the file it
    started from - it does NOT replace most of the file just to make
    one small, requested change (a live test showed exactly this: a
    proposed edit that kept almost none of app/main.py's own original
    content, instead dumping hundreds of unrelated lines - including
    the model's own reasoning text - into it). Measured with the
    stdlib's own difflib, never a new dependency.

    Skipped (always True) for a file smaller than `_MIN_LINES_FOR_
    PROPORTIONALITY_CHECK` - fully replacing a small stub/placeholder
    file is the ORDINARY, expected shape of a first edit, not a
    runaway rewrite."""
    old_lines = old_content.splitlines()
    if len(old_lines) < _MIN_LINES_FOR_PROPORTIONALITY_CHECK:
        return True
    new_lines = new_content.splitlines()
    similarity = difflib.SequenceMatcher(None, old_lines, new_lines).ratio()
    return similarity >= _MIN_UNCHANGED_SIMILARITY


def _apply_file_edits(current_content: str, edits: tuple[dict, ...]) -> str | None:
    """Applies a small set of EXACT find-and-replace edits against the
    REAL current content this engine already read - never the model's
    own claim of what was there. Returns the resulting content, or
    `None` (never a partial result, never a guess) if:
    - any edit's `old_text` doesn't appear in the content AT ALL
      (hallucinated - the model quoted text that isn't real), or
      appears MORE THAN ONCE (ambiguous - which occurrence?);
    - any edit's `old_text`/`new_text` exceeds `_MAX_EDIT_TEXT_CHARS`,
      there are more than `_MAX_EDITS_PER_FILE` edits, or the edits'
      COMBINED `old_text` exceeds `_MAX_TOTAL_OLD_TEXT_RATIO` of the
      file's own current size (several small edits adding up to a de
      facto bulk rewrite is rejected too, not just one large one);
    - `current_content` is empty (a brand-new file) and the edits
      AREN'T exactly the single documented "whole new file" form
      (`old_text == ""`); or that form is used for a file that already
      has real content (it must never silently blow away an existing
      file this way).

    Edits are applied in order, each against the content the PREVIOUS
    edit already produced - so a later edit's `old_text` must match
    what's there at that point, not the original file, the same way a
    person editing a file one change at a time would experience it."""
    if len(edits) > _MAX_EDITS_PER_FILE:
        return None

    if current_content == "":
        if (
            len(edits) == 1
            and edits[0]["old_text"] == ""
            and len(edits[0]["new_text"]) <= _MAX_EDIT_TEXT_CHARS
        ):
            return edits[0]["new_text"]
        return None

    max_total_old_text = max(
        _MIN_TOTAL_OLD_TEXT_FLOOR, int(len(current_content) * _MAX_TOTAL_OLD_TEXT_RATIO)
    )
    total_old_text = sum(len(edit["old_text"]) for edit in edits)
    if total_old_text > max_total_old_text:
        return None  # touches too much of the file in aggregate, even if no single edit does

    content = current_content
    for edit in edits:
        old_text = edit["old_text"]
        new_text = edit["new_text"]
        if not old_text:
            return None  # empty old_text is only ever valid for a brand-new file
        if len(old_text) > _MAX_EDIT_TEXT_CHARS or len(new_text) > _MAX_EDIT_TEXT_CHARS:
            return None
        if content.count(old_text) != 1:
            return None  # not found, or matches more than once - never guess
        content = content.replace(old_text, new_text, 1)
    return content


#: A fixed, general set of hedge/uncertainty/placeholder phrases - never
#: specific to one request or feature. A plan genuinely grounded in real,
#: inspected code never needs to say any of this about something it's
#: proposing to touch or expose; a plan that does is describing what it
#: WOULD do if reality turned out a certain way, i.e. guessing.
_HEDGE_PHRASES = (
    "placeholder",
    "fallback value",
    "fallback implementation",
    "generic fallback",
    "dummy value",
    "dummy implementation",
    "stub implementation",
    "best guess",
    "hypothetical",
    "made up",
    "made-up",
    "not sure if",
    "may not exist",
    "might not exist",
    "doesn't exist yet",
    "does not exist yet",
    "assume it exists",
    "assuming it exists",
)

#: The other common shape the same hedging takes - a conditional on
#: something not actually being there ("if the engine has no status
#: attribute...", "if it doesn't have a config value...").
_HEDGE_CONDITIONAL_PATTERN = re.compile(
    r"\bif\s+(?:the\s+\w+|it|this|there)\s+(?:has|have|had|is|are)\s+no\b",
    re.IGNORECASE,
)


def _extract_keywords(text: str) -> frozenset[str]:
    """Pure, deterministic, no-LLM keyword extraction - lowercase
    alphabetic runs of `_MIN_KEYWORD_LENGTH`+ characters, minus the
    small fixed stopword list above. Used identically to build the
    search terms for `_inspect_relevant_existing_code` and to check a
    proposed new file's own description against what that search
    actually found - the SAME function, so the two can never drift
    apart into inconsistent notions of "keyword."."""
    words = re.findall(r"[a-z]{%d,}" % _MIN_KEYWORD_LENGTH, text.lower())
    return frozenset(w for w in words if w not in _KEYWORD_STOPWORDS)


def _extract_words(text: str) -> frozenset[str]:
    """Same tokenization as `_extract_keywords`, WITHOUT the stopword
    filter - used as the search TARGET (a proposed file's own
    description), never as the query side, so a stopword that happens
    to also be a real discovered symbol name is never unfairly
    excluded."""
    return frozenset(re.findall(r"[a-z]{%d,}" % _MIN_KEYWORD_LENGTH, text.lower()))


def _split_identifier(name: str) -> frozenset[str]:
    """Splits a CamelCase or snake_case identifier into its real word
    parts (`DevelopmentSession` -> {"development", "session"},
    `approve_commit` -> {"approve", "commit"}), in addition to the
    whole name itself - so a plain-English description ("expose the
    session's stage") can match a real symbol (`DevelopmentSession`,
    `stage`) without needing the exact, squished identifier spelling."""
    parts = {p.lower() for p in re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", name)}
    parts.add(name.lower())
    return frozenset(p for p in parts if len(p) >= _MIN_KEYWORD_LENGTH)


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Minimal crash-recovery metadata (hardening pass) - deliberately NOT a
# persistence subsystem: one small JSON file per session, written
# best-effort after each meaningful state change, read only to REPORT
# abandoned sessions - never to resume, reset, or delete anything
# automatically. Stored under `.git/jarvis_selfdev_sessions/` - inside
# Git's own metadata directory, so it's never listed/read/searched by
# this engine's own repo tools, never shown in `git status`/diffs, and
# never mistaken for tracked project content.
# ---------------------------------------------------------------------------


def _session_records_dir(project_root: str) -> Path:
    return Path(project_root) / ".git" / "jarvis_selfdev_sessions"


def _session_record_path(project_root: str, session_id: str) -> Path:
    return _session_records_dir(project_root) / f"{session_id}.json"


def _persist_session_record(project_root: str, session: DevelopmentSession) -> None:
    """Best-effort only - a persistence failure must never break the
    actual workflow. Called AFTER a state change already happened, not
    as part of deciding whether it's allowed to."""
    try:
        directory = _session_records_dir(project_root)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "id": session.id,
            "branch_name": session.dev_branch,
            "original_branch": session.original_branch,
            "modified_files": list(session.edited_files),
            "stage": session.stage.value,
        }
        _session_record_path(project_root, session.id).write_text(
            json.dumps(record), encoding="utf-8"
        )
    except OSError:
        pass


def _delete_session_record(project_root: str, session_id: str) -> None:
    """Called only once a session reaches a fully resolved stage
    (COMMITTED/ROLLED_BACK) - nothing left to recover, so nothing left
    to report either."""
    try:
        _session_record_path(project_root, session_id).unlink(missing_ok=True)
    except OSError:
        pass


def scan_for_abandoned_sessions(project_root: str) -> list[dict]:
    """Read-only: reports every persisted session record still sitting
    in a non-terminal stage (app/self_development/base.py:
    ABANDONED_SESSION_STAGES) - a real branch exists, with nothing
    committed and nothing rolled back. NEVER deletes, resets, restores,
    or otherwise acts on anything found - purely informational, safe
    to call at any time, including application startup. A missing or
    unreadable record is silently skipped, never raised."""
    directory = _session_records_dir(project_root)
    if not directory.is_dir():
        return []

    abandoned = []
    for record_path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            stage = SessionStage(record.get("stage"))
        except ValueError:
            continue
        if stage in ABANDONED_SESSION_STAGES:
            abandoned.append(record)
    return abandoned


class SelfDevelopmentEngine:
    def __init__(
        self,
        project_root: str,
        tools: ToolRegistry,
        git_client: GitClient,
        planner: DevelopmentPlanner,
        *,
        max_repair_iterations: int,
    ):
        self.project_root = project_root
        self.tools = tools
        self.git = git_client
        self.planner = planner
        self.max_repair_iterations = max_repair_iterations
        self.sessions: dict[str, DevelopmentSession] = {}

    def scan_for_abandoned_sessions(self) -> list[dict]:
        """See the module-level `scan_for_abandoned_sessions` docstring
        - read-only, reports only, bound to this engine's own
        `project_root`."""
        return scan_for_abandoned_sessions(self.project_root)

    def _persist(self, session: DevelopmentSession) -> None:
        _persist_session_record(self.project_root, session)

    # -- 1-3: understand request -> inspect repo -> create change plan ------

    def start(
        self, request: str, *, conversation_id: str | None = None
    ) -> DevelopmentSession:
        """Refuses to start at all if the working tree already has
        uncommitted changes - "detect and report rather than blindly
        resetting" (V1's own explicit requirement). Otherwise proposes
        a DevelopmentPlan and stops there - no branch is created and no
        file is touched until `approve_plan` explicitly says so.

        `conversation_id` is purely additive bookkeeping (see
        DevelopmentSession.conversation_id's own docstring) - it never
        affects anything this method decides."""
        original_branch = self.git.current_branch()

        if self.git.has_uncommitted_changes():
            session = DevelopmentSession(
                id=_new_session_id(),
                request=request,
                stage=SessionStage.BLOCKED_DIRTY_TREE,
                original_branch=original_branch,
                conversation_id=conversation_id,
            )
            session.log(
                "Uncommitted changes already exist in the working tree - "
                "refusing to start a self-development session. Commit or "
                "stash your own changes first."
            )
            self.sessions[session.id] = session
            return session

        session = DevelopmentSession(
            id=_new_session_id(),
            request=request,
            stage=SessionStage.STARTED,
            original_branch=original_branch,
            conversation_id=conversation_id,
        )
        self.sessions[session.id] = session

        # BEFORE proposing anything - a real, deterministic inspection of
        # the repository's own file names AND contents relevant to this
        # request, never a guess from a bare list of paths. See this
        # module's own "Plan grounding (hardening pass 2)" section. One
        # listing call, reused for both the digest and candidate search.
        file_list = self._list_repo_files()
        inspected_text, discovered_symbols = self._inspect_relevant_existing_code(
            request, file_list
        )

        try:
            digest = self._build_repo_digest(file_list)
            plan = self.planner.propose_plan(request, f"{digest}\n\n{inspected_text}")
        except DevelopmentPlanningError as exc:
            session.stage = SessionStage.PLAN_REJECTED
            session.log(f"Could not produce a development plan: {exc}")
            return session

        hedges = self._find_hedged_language(plan)
        if hedges:
            session.stage = SessionStage.PLAN_REJECTED
            session.log(
                "Rejected a plan that hedges on missing information "
                "instead of grounding the request in real, inspected "
                f"code (found: {', '.join(hedges)}). Inspect the code "
                "yourself and ask again once what's actually needed is "
                "confirmed to exist."
            )
            return session

        ungrounded = self._find_ungrounded_file_paths(plan, discovered_symbols)
        if ungrounded:
            session.stage = SessionStage.PLAN_REJECTED
            session.log(
                "Rejected an ungrounded development plan - proposed "
                f"file(s) not grounded in the actual repository: "
                f"{', '.join(ungrounded)}. Ask again, or narrow the "
                "request to files/directories that already exist."
            )
            return session

        session.plan = plan
        session.stage = SessionStage.PLAN_PROPOSED
        session.log("Development plan proposed - awaiting approval.")

        protected = plan.protected_file_paths()
        if protected:
            session.log(
                "This plan touches protected security-critical file(s): "
                f"{', '.join(sorted(protected))}. Approving the plan will NOT "
                "authorize editing them - a separate SECURITY_EDIT approval "
                "(approve_security_edit) will be required before those "
                "specific files can be touched."
            )
        return session

    def _find_ungrounded_file_paths(
        self, plan: DevelopmentPlan, discovered_symbols: frozenset[str]
    ) -> tuple[str, ...]:
        """A proposed EDIT to a real, already-existing file is always
        grounded. A proposed BRAND-NEW file must clear TWO bars: its
        parent directory must already exist in this repository (self-
        development may add a new file, but never invents an entirely
        new directory/subsystem out of thin air just because the
        planner's own LLM call happened to mention one), AND - when
        `_discover_relevant_symbols` actually found real symbols related
        to this request - its own description must name one of them,
        never merely restate the request in isolation (a live test
        surfaced a plan inventing a whole new file, `app/status.py`,
        with no such connection to anything real at all). Returns every
        offending path, so the caller rejects the WHOLE plan - never
        silently drops just the offending file(s), the same fail-closed
        discipline every other structural check in this package already
        uses (unexpected JSON fields, unknown permissions, protected/
        secret paths, ...)."""
        ungrounded: list[str] = []
        for change in plan.files_affected:
            if is_protected_system_path(change.path):
                # A SEPARATE, already fully-enumerated fixed allowlist
                # (PROTECTED_SYSTEM_PATH_PREFIXES) - never a free-form
                # path the LLM could hallucinate into existing, so
                # neither grounding check applies to it; SECURITY_EDIT's
                # own separate approval gate is what actually protects
                # these paths.
                continue
            try:
                resolved = resolve_within_root(self.project_root, change.path)
            except (PathSafetyError, PathTraversalError):
                ungrounded.append(change.path)
                continue
            if not resolved.parent.is_dir():
                ungrounded.append(change.path)
                continue
            if not resolved.exists() and discovered_symbols:
                description_words = _extract_words(change.description)
                matched = description_words & discovered_symbols
                # A match on a word that's ALSO one of the overly-generic
                # fragments above never counts by itself - a live test
                # showed exactly this (GitClient.status_porcelain's own
                # bare "status" "grounding" an invented, unrelated
                # /dev/status endpoint file). Only a SPECIFIC match - a
                # whole compound identifier, or a real word that isn't
                # one of the common CS/English fragments - counts as
                # evidence. Never widened by simply combining several
                # generic matches either: the live plan matched FIVE
                # generic words at once ("status," "engine,"
                # "development," "self," "test") and would still have
                # been wrongly accepted under a "2+ matches" rule.
                if not (matched - _GENERIC_IDENTIFIER_FRAGMENTS):
                    ungrounded.append(change.path)
        return tuple(ungrounded)

    def _find_candidate_files(
        self, keywords: frozenset[str], file_list: list[dict] | None = None
    ) -> list[str]:
        """PATH-ONLY filter (never reads a single byte of file content) -
        the real repository file listing `_build_repo_digest` already
        computes, narrowed to real `.py` files whose own path already
        relates to one of the request's own keywords. Deliberately
        cheap: reads nothing at all when nothing matches, so a request
        with no distinctive domain word never touches the filesystem
        beyond the one listing call every `start()` already makes.

        Ranked by HOW MANY distinct keywords a path matches, most first,
        before the `_MAX_CANDIDATE_FILES` cap is applied - a generic
        single-word match (e.g. "engine," which many unrelated
        subsystems each have their own engine.py for) must never crowd
        out a path matching several of the request's own words at once
        (e.g. "self_development" matching "self"+"development"+
        "engine" together)."""
        if not keywords:
            return []
        if file_list is None:
            file_list = self._list_repo_files()
        if not file_list:
            return []

        scored: list[tuple[int, str]] = []
        for entry in file_list:
            path = entry["path"]
            if entry["secret"] or not path.endswith(".py") or path.startswith("tests/"):
                # Test files verify capabilities, they aren't capabilities
                # themselves - never useful "evidence" for what a plan
                # should build on.
                continue
            hits = sum(1 for k in keywords if k in path.lower())
            if hits:
                scored.append((hits, path))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [path for _hits, path in scored[:_MAX_CANDIDATE_FILES]]

    @staticmethod
    def _discover_symbols_in_file(source: str) -> frozenset[str]:
        """Real static inspection, never a raw-text substring search:
        parses `source` with `ast` and collects the names of every
        class, function/method, and attribute (dataclass field,
        annotated or plain assignment, or `self.x = ...` store)
        ACTUALLY defined in it - e.g. DevelopmentSession's own `stage`
        field. Malformed/unparseable source yields no symbols, never
        raises - this is evidence-gathering, not a correctness check on
        the file itself."""
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return frozenset()

        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(item.name)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        names.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                names.add(target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                names.add(node.attr)  # e.g. `self.stage = ...`
        return frozenset(names)

    def _inspect_relevant_existing_code(
        self, request: str, file_list: list[dict] | None = None
    ) -> tuple[str, frozenset[str]]:
        """Discovers relevant existing CAPABILITIES for `request` -
        general-purpose, driven entirely by words already in the
        request itself, never hardcoded to any specific feature, file,
        class, or attribute. Two cheap stages: `_find_candidate_files`
        (path-only, no content read) narrows to a small set of real
        `.py` files, then `_discover_symbols_in_file` (real `ast`
        parsing) extracts the classes/functions/methods/attributes
        ACTUALLY defined in just that narrow set - never every file in
        the repository, and never a raw-text keyword scan of file
        content.

        Returns (a text block to append to the planner's own prompt,
        and) the set of discovered symbol names - used afterward by
        `_find_ungrounded_file_paths`/`start()` to structurally verify
        the resulting plan names something that ACTUALLY exists, never
        merely what the LLM claims. Never reads a secret-named file."""
        keywords = _extract_keywords(request)
        candidates = self._find_candidate_files(keywords, file_list)
        if not candidates:
            return (
                "(no existing file's path relates to any of this "
                f"request's own keywords: {', '.join(sorted(keywords)) or '(none)'} - "
                "nothing was read)"
            ), frozenset()

        symbols_by_file: dict[str, frozenset[str]] = {}
        for path in candidates:
            read_result = self.tools.execute("self_dev_read_file", params={"path": path})
            if read_result.status is not ToolResultStatus.SUCCESS or not read_result.data:
                continue
            symbols = self._discover_symbols_in_file(read_result.data.get("content", ""))
            if symbols:
                symbols_by_file[path] = symbols

        if not symbols_by_file:
            return (
                "(the matched file(s) defined no discoverable classes, "
                "functions, or attributes)"
            ), frozenset()

        lines = [
            "RELEVANT EXISTING CAPABILITIES (discovered just now via real "
            "static inspection of the matched file(s) below - classes, "
            "functions/methods, and attributes ACTUALLY defined there) - "
            "ground your plan in these REAL names; never invent a class, "
            "method, attribute, or file that isn't one of them."
        ]
        matchable_tokens: set[str] = set()
        for path in sorted(symbols_by_file):
            names = symbols_by_file[path]
            lines.append(f"### {path}\nnames: {', '.join(sorted(names))}")
            for name in names:
                matchable_tokens |= _split_identifier(name)
        return "\n\n".join(lines), frozenset(matchable_tokens)

    def _find_hedged_language(self, plan: DevelopmentPlan) -> tuple[str, ...]:
        """A plan genuinely grounded in real, inspected code never needs
        to hedge about whether something it's proposing to touch or
        expose actually exists (a live test surfaced exactly this: "if
        the engine has no status attribute, return a generic
        placeholder" - proof the planner never actually confirmed that
        against real code). Checked against the plan's own objective,
        every file's description, and every risk - a fixed, general set
        of phrases, never anything specific to one request, so this
        generalizes to any future hallucinated plan with the same tell.
        Returns every matched phrase (for the rejection message), never
        silently ignored."""
        texts = [plan.objective, *(c.description for c in plan.files_affected), *plan.risks]
        found: list[str] = []
        for text in texts:
            lowered = text.lower()
            for phrase in _HEDGE_PHRASES:
                if phrase in lowered:
                    found.append(phrase)
            conditional_match = _HEDGE_CONDITIONAL_PATTERN.search(text)
            if conditional_match:
                found.append(conditional_match.group(0))
        return tuple(dict.fromkeys(found))

    def _list_repo_files(self) -> list[dict] | None:
        """The ONE place this module calls `self_dev_list_files` -
        `_build_repo_digest` and `_find_candidate_files` both accept an
        already-fetched listing so `start()` can fetch it exactly once
        and reuse it for both, never two separate listing calls for one
        session."""
        result = self.tools.execute("self_dev_list_files")
        if result.status is not ToolResultStatus.SUCCESS or not result.data:
            return None
        return result.data

    def _build_repo_digest(self, file_list: list[dict] | None = None) -> str:
        if file_list is None:
            file_list = self._list_repo_files()
        if not file_list:
            return "(no files listed)"
        return "\n".join(
            f"{entry['path']}" + (" [secret - contents never shown]" if entry["secret"] else "")
            for entry in file_list
        )

    # -- 4-6: ask user for approval -> create branch -> edit files ----------

    def approve_plan(self, session: DevelopmentSession, approved: bool) -> DevelopmentSession:
        """Authorizes ordinary CODE_EDIT for this session's bounded
        editing/repair loop - NEVER SECURITY_EDIT (see this module's
        own docstring). If the plan touches no protected path, this
        single call runs the entire rest of the workflow synchronously
        (edit -> test -> bounded repair -> finalize). If it does, this
        call stops right after branch creation and
        `approve_security_edit` must be called separately."""
        if session.stage is not SessionStage.PLAN_PROPOSED:
            raise DevelopmentSessionError(
                f"Cannot approve a plan from stage '{session.stage.value}'."
            )
        if not approved:
            session.stage = SessionStage.PLAN_REJECTED
            session.log("Plan rejected - no branch created, no files touched.")
            return session

        branch = branch_name_for(session.id)
        result = self.tools.execute("self_dev_create_branch", params={"name": branch})
        if result.status is not ToolResultStatus.SUCCESS:
            session.stage = SessionStage.PLAN_REJECTED
            session.log(f"Could not create a development branch: {result.message}")
            return session

        session.dev_branch = branch
        session.stage = SessionStage.BRANCH_CREATED
        session.log(f"Created development branch '{branch}'.")
        self._persist(session)

        if session.plan.protected_file_paths():
            session.stage = SessionStage.AWAITING_SECURITY_APPROVAL
            session.log(
                "Branch created. Waiting for a separate SECURITY_EDIT approval "
                "(approve_security_edit) before any protected file is edited."
            )
            self._persist(session)
            return session

        self._proceed_with_editing(session)
        return session

    def approve_security_edit(
        self, session: DevelopmentSession, *, confirmed: bool | None
    ) -> DevelopmentSession:
        """The ONLY way protected-path edits proceed. `confirmed` must
        be explicitly `True` - never inferred from the plan's own
        `required_permissions`, the request text, or any AI output.
        `None` (not yet asked) or `False` (declined) both leave the
        session's protected files untouched; declining marks the
        session SECURITY_EDIT_DECLINED rather than silently retrying."""
        if session.stage is not SessionStage.AWAITING_SECURITY_APPROVAL:
            raise DevelopmentSessionError(
                f"Cannot approve SECURITY_EDIT from stage '{session.stage.value}'."
            )
        if confirmed is not True:
            if confirmed is False:
                session.stage = SessionStage.SECURITY_EDIT_DECLINED
                session.log(
                    "SECURITY_EDIT declined - protected files were not edited. "
                    "The session's branch still exists for manual review or rollback."
                )
            else:
                session.log("SECURITY_EDIT requires an explicit confirmation.")
            self._persist(session)
            return session

        session.log("SECURITY_EDIT approved - proceeding to edit protected file(s).")
        self._persist(session)
        self._proceed_with_editing(session)
        return session

    def _proceed_with_editing(self, session: DevelopmentSession) -> None:
        """Shared tail of `approve_plan` (plans with no protected
        paths) and `approve_security_edit` (plans that needed it) -
        edit -> test -> bounded repair -> finalize."""
        self._apply_plan(session)
        self._persist(session)
        self._run_checks_with_bounded_repair(session)
        self._persist(session)
        self._finalize(session)
        self._persist(session)

    def _apply_plan(self, session: DevelopmentSession) -> None:
        session.stage = SessionStage.EDITING
        edited = list(session.edited_files)
        for change in session.plan.files_affected:
            if self._edit_one_file(session, change.path, change.description, session.plan.objective):
                if change.path not in edited:
                    edited.append(change.path)
        session.edited_files = tuple(edited)

    def _edit_one_file(
        self, session: DevelopmentSession, path: str, description: str, objective: str
    ) -> bool:
        """Routes to the CODE_EDIT tool for an ordinary path, or the
        SEPARATE SECURITY_EDIT tool for a protected one - never the
        other way around (see app/self_development/editor.py, where
        each tool refuses the other's path category outright). Both
        are only ever called with `confirmed=True` here, and this
        method is only ever reached via `_proceed_with_editing`, which
        is only ever reached after the matching real approval
        (`approve_plan` for ordinary paths, `approve_security_edit` for
        protected ones) already happened."""
        read_result = self.tools.execute("self_dev_read_file", params={"path": path})
        current_content = (
            read_result.data["content"]
            if read_result.status is ToolResultStatus.SUCCESS
            else ""
        )
        try:
            edits = self.planner.propose_file_edits(
                objective, description, path, current_content
            )
        except DevelopmentPlanningError as exc:
            session.log(f"Could not generate an edit for '{path}': {exc}")
            return False

        new_content = _apply_file_edits(current_content, edits)
        if new_content is None:
            session.log(
                f"Edit rejected for '{path}': the proposed edit(s) didn't "
                "match the file's real current content exactly (not "
                "found, matched more than once, or too large/too many) "
                "- refusing an ambiguous, hallucinated, or oversized edit."
            )
            return False

        if not _content_is_syntactically_valid(path, new_content):
            session.log(
                f"Edit rejected for '{path}': the proposed content is not "
                "syntactically valid - the self-development engine never "
                "writes malformed or non-code output (e.g. reasoning text) "
                "to a source file."
            )
            return False

        if current_content and not _is_change_proportionate(current_content, new_content):
            session.log(
                f"Edit rejected for '{path}': the proposed change replaces "
                "far more of the file than the requested edit needed - "
                "refusing an unbounded rewrite of an existing file."
            )
            return False

        tool_name = (
            "self_dev_security_edit_file" if is_protected_system_path(path) else "self_dev_edit_file"
        )
        edit_result = self.tools.execute(
            tool_name, params={"path": path, "content": new_content}, confirmed=True
        )
        if edit_result.status is not ToolResultStatus.SUCCESS:
            session.log(f"Edit rejected for '{path}': {edit_result.message}")
            return False
        session.log(f"Edited '{path}'.")
        return True

    # -- 7-9: run tests/checks -> inspect failures -> bounded fixes ---------

    def _run_checks_with_bounded_repair(self, session: DevelopmentSession) -> None:
        while True:
            check = self._run_tests(session)
            if check.result.succeeded:
                session.stage = SessionStage.CHECKS_PASSED
                session.log("Checks passed.")
                return

            if session.repair_attempts >= self.max_repair_iterations:
                session.stage = SessionStage.CHECKS_FAILED
                session.log(
                    f"Checks still failing after {session.repair_attempts} bounded "
                    "repair attempt(s) - stopping (max_repair_iterations reached)."
                )
                return

            session.repair_attempts += 1
            session.log(
                f"Checks failed - bounded repair attempt "
                f"{session.repair_attempts}/{self.max_repair_iterations}."
            )
            self._attempt_repair(session, check.result)

    def _run_tests(self, session: DevelopmentSession) -> CheckRun:
        result = self.tools.execute("self_dev_run_tests")
        if result.status is ToolResultStatus.SUCCESS:
            data = result.data
            command_result = CommandResult(
                argv=(), exit_code=data["exit_code"], stdout=data["stdout"],
                stderr=data["stderr"], timed_out=data["timed_out"],
            )
        else:
            command_result = CommandResult(
                argv=(), exit_code=None, stdout="", stderr=result.message, timed_out=False
            )
        check = CheckRun(command="self_dev_run_tests", result=command_result)
        session.checks_run = session.checks_run + (check,)
        return check

    def _attempt_repair(self, session: DevelopmentSession, failed: CommandResult) -> None:
        failure_text = (failed.stdout + "\n" + failed.stderr)[:_MAX_FAILURE_TEXT_CHARS]
        try:
            digest = self._build_repo_digest()
            fix_plan: DevelopmentPlan = self.planner.propose_fix(
                session.plan, failure_text, digest
            )
        except DevelopmentPlanningError as exc:
            session.log(f"Could not produce a repair fix: {exc}")
            return

        hedges = self._find_hedged_language(fix_plan)
        if hedges:
            session.log(
                "Ignoring a repair fix that hedges on missing "
                f"information instead of grounding itself in real code "
                f"(found: {', '.join(hedges)})."
            )
            return

        allowed_paths = session.plan.file_paths()
        edited = list(session.edited_files)
        for change in fix_plan.files_affected:
            if change.path not in allowed_paths:
                session.log(
                    f"Ignoring out-of-scope repair edit to '{change.path}' - not "
                    "in the originally approved plan."
                )
                continue
            if self._edit_one_file(session, change.path, change.description, fix_plan.objective):
                if change.path not in edited:
                    edited.append(change.path)
        session.edited_files = tuple(edited)

    # -- 10-11: inspect diff -> show summary ---------------------------------

    def _finalize(self, session: DevelopmentSession) -> None:
        diff_result = self.tools.execute("self_dev_git_diff")
        diff_text = diff_result.data["diff"] if diff_result.status is ToolResultStatus.SUCCESS else ""

        remaining_failures = ()
        if session.stage is SessionStage.CHECKS_FAILED and session.checks_run:
            last = session.checks_run[-1].result
            remaining_failures = ((last.stdout + "\n" + last.stderr)[:_MAX_FAILURE_TEXT_CHARS],)

        session.summary = DevelopmentSummary(
            files_changed=session.edited_files,
            change_summary=session.plan.objective if session.plan else "",
            checks_run=session.checks_run,
            remaining_failures=remaining_failures,
            git_diff_summary=diff_text,
            rollback_available=session.rollback_available(),
        )
        if session.stage is SessionStage.CHECKS_PASSED:
            session.stage = SessionStage.AWAITING_COMMIT_APPROVAL
            session.log("Ready for commit approval.")

    # -- 12: require a SEPARATE explicit approval before committing --------

    def approve_commit(
        self, session: DevelopmentSession, *, confirmed: bool | None, message: str
    ) -> DevelopmentSession:
        if session.stage is not SessionStage.AWAITING_COMMIT_APPROVAL:
            raise DevelopmentSessionError(
                f"Cannot commit from stage '{session.stage.value}' - checks must "
                "have passed and produced a summary first."
            )
        result = self.tools.execute(
            "self_dev_commit", params={"message": message}, confirmed=confirmed
        )
        if result.status is ToolResultStatus.SUCCESS:
            session.stage = SessionStage.COMMITTED
            session.log("Committed.")
            _delete_session_record(self.project_root, session.id)  # fully resolved
        elif result.status is ToolResultStatus.DENIED:
            session.stage = SessionStage.COMMIT_DECLINED
            session.log("Commit declined.")
            self._persist(session)
        else:
            session.log(f"Commit not completed: {result.message}")
            self._persist(session)
        return session

    # -- rollback -------------------------------------------------------------

    def rollback(self, session: DevelopmentSession, *, confirmed: bool | None) -> DevelopmentSession:
        if not session.rollback_available():
            raise DevelopmentSessionError(
                f"Nothing to roll back from stage '{session.stage.value}'."
            )
        result = self.tools.execute(
            "self_dev_rollback",
            params={
                "edited_files": ",".join(session.edited_files),
                "original_branch": session.original_branch,
            },
            confirmed=confirmed,
        )
        if result.status is ToolResultStatus.SUCCESS:
            session.stage = SessionStage.ROLLED_BACK
            session.log(f"Rolled back to '{session.original_branch}'.")
            _delete_session_record(self.project_root, session.id)  # fully resolved
        elif result.status is ToolResultStatus.DENIED:
            session.log("Rollback declined.")
            self._persist(session)
        else:
            session.log(f"Rollback not completed: {result.message}")
            self._persist(session)
        return session
