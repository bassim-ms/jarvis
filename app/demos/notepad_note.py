"""
The Notepad demonstration: open Notepad, write one short note, save it to a NEW file in a dedicated test directory, and prove from the disk that the
file exists and holds the note.

    python -m app.demos.notepad_note --dir C:\\...\\jarvis_demo_run1            # a dry run: checks everything, touches nothing
    python -m app.demos.notepad_note --dir C:\\...\\jarvis_demo_run1 --run      # the real desktop run

It is plumbing between parts that already exist, not a new capability:

    preflight (refuse early)  ->  a fixed, validated plan  ->  a Job born RUNNING  ->  the existing JobEngine  ->  the existing tools  ->  the gate
                                                                                                                  ->  an independent look at the disk

  * THE GATE IS NEVER BYPASSED OR CONFIGURED HERE. The input tools are the real ones over the real backend, which asks the process-wide Computer Input
    Gate before every OS call. This module never switches input on, never changes the limit and never builds a gate of its own. With input switched
    off the first input step is refused by the backend and nothing else runs. The owner enables input for one run (`JARVIS_COMPUTER_INPUT_ENABLED=true`
    in that process's environment).
  * THE SWITCH IS NOT ENOUGH: THE RUN CARRIES ITS OWN GRANT. `--run` is the explicit authorization. It derives the seven exact input actions (with their exact
    arguments - for the window switch, the exact title and the exact process name) from the SAME plan the Job then runs, mints one grant for exactly that ordered list (valid 120 s, single-use entries, app/computer/authority.py),
    hands it to ONE backend instance, and revokes it when the run ends, however it ends. A different action, different text or a different order is refused AND ends
    the grant; the shared backend the rest of Jarvis uses carries no grant and admits nothing. Nothing here authorizes chat, Jobs, schedulers or automations.
  * ONE ATTEMPT PER INPUT ACTION. Opening Notepad, bringing it forward, each keystroke and typed text, the save: `max_attempts=1`, so nothing is ever repeated
    blindly. Only READ steps (is the right window in front? has the dialog appeared? is the file there yet?) are retried, and that is how the demo waits.
  * BRINGING NOTEPAD FORWARD IS AN AUTHORIZED ACTION TOO (`front_notepad`, tool `demo_front_notepad`). Windows does not bring a launched Notepad forward, and the
    window needs a moment to exist, so this one step first LOOKS for it - up to five read-only looks, two seconds apart, that change no focus and send no input and
    never touch the gated action - and only when exactly one window has Notepad's exact title and process does it make the ONE gated `switch_window` call. The grant binds
    that exact title (`--notepad-title`, default "Untitled - Notepad") and the exact process `notepad.exe`; the backend admits first and then looks for an exact match
    ONCE. So: no window after five looks, or more than one, fails the step WITHOUT calling the backend - that entry is not spent and nothing is typed; once the backend
    is called, its entry is spent whatever the lookup finds, and it is never retried. The failure report lists the expected title and the titles of the visible
    notepad.exe windows, read-only. A grant binds a title and a process, not a window instance: two matching windows are an error, never a choice.
  * FOCUS IS VERIFIED IMMEDIATELY BEFORE EVERY KEYSTROKE AFTER THE LAUNCH: the step right before ctrl+n, the note, ctrl+s, the file name and the final
    Enter is a read of the foreground window - its process must be notepad.exe, and before the file name and the Enter the Save As dialog's title must
    be in front too (otherwise the path would land in the note). Each read is one existing, fail-closed check (a process that cannot be resolved is
    "unknown", which never matches); if the window cannot be established, nothing is typed. There is no "assume it is fine" path.
  * THE NOTE IS READ BACK BEFORE ANYTHING IS SAVED. `type_text` succeeding only means the keystrokes were queued; Notepad's editor has dropped some of
    them in live runs. So right after the note is typed, `verify_document_text` reads the visible editor of the window in front (notepad_reader.py: read-only,
    bounded window messages, hidden tabs ignored) and compares the WHOLE text with the note. Any difference - or any failure to read - stops the plan before
    Ctrl+S: nothing is repaired, retyped or retried, no Save As is opened, no file is created. That step is also the last foreground check before Ctrl+S.
  * NO OVERWRITE, AND NO ANSWERING A PROMPT. The directory must be fresh (new or empty) and the file name unique. Right before the Enter that saves,
    `verify_demo_target_free` re-checks from the disk that the directory is still an empty, real directory and the target still does not exist. After the
    Enter, the save only counts if the file on disk holds the note AND Notepad's own title now shows the new file's name. Jarvis never presses a key to
    answer an overwrite (or any other) prompt - the plan has exactly one Enter - and never relies on which button such a prompt defaults to: if one
    appears, the title check fails, the Job fails, and the prompt is left on the screen unanswered for the owner.
  * RESIDUAL RACE (cannot be removed with this architecture). Notepad, not Jarvis, creates the file at the instant of the Enter, and Jarvis cannot reserve
    the name without creating a file itself. So another process that can write to the fresh directory could still create the target between the last
    check and Notepad's save. The effects are bounded: the one Enter would raise an overwrite prompt (left unanswered, so nothing is replaced by Jarvis),
    the title check fails, and the report says so. The directory is brand new and created by this run, so this needs something else on the machine
    deliberately writing there in that window. Do not point the demo at a shared, synced or watched folder.
  * NO MODEL, NO REPLANNING, NO CLEAN-UP. The plan is fixed; the planner is never asked; a failed step fails the Job and its dependants are skipped.
    Jarvis does not close Notepad, delete anything or try again - the report says what was left on the screen.
  * CONFINED: the directory rules and the one file tool are in file_check.py. Nothing is read, changed or deleted anywhere else.

The Job allowlist (`JOB_COMPUTER_TOOL_NAMES`) is untouched: the registry built here is private to this module.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.computer.authority import AuthorityError, InputStep
from app.computer.base import (
    AllowedApplication,
    ComputerBackend,
    ComputerControlError,
    exact_window_matches,
    normalize_key_input,
    validate_exact_window_title,
    validate_process_name,
    windows_of_process,
)
from app.computer.gate import GATED_METHODS, InputGate, default_input_gate
from app.computer.tools import OpenApplicationTool, PressKeyTool, TypeTextTool
from app.demos.file_check import (
    CheckDemoTargetFreeTool,
    DemoPathError,
    VerifyDemoFileTool,
    check_target,
    new_target,
    validate_directory,
    validate_note,
)
from app.demos.notepad_reader import DocumentReadError, read_visible_editor_text
from app.observation.base import ObservationError, scrub
from app.observation.engine import ObservationEngine
from app.observation.page import FetchedPage, PageFetcher
from app.observation.tools import VerifyObservationTool
from app.orchestration.base import TERMINAL_JOB_STATUSES, AutonomyLevel, Job, JobStatus, RetryPolicy, StepStatus
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.validator import validate_plan
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolRequirements, ToolResult
from app.tools.registry import ToolRegistry

#: open_application, switch_window (Notepad, by its exact title and process), ctrl+n, the note, ctrl+s, the file name, enter - the gated actions one successful run takes.
INPUT_ACTIONS_NEEDED = 7
NOTEPAD_PROCESS = "notepad.exe"
#: The EXACT title the freshly opened Notepad window is expected to show (Windows 11 Notepad, English). The grant binds this exact string, so a Notepad that shows anything
#: else - a restored session, another display language - is never switched to; pass --notepad-title with what your Notepad really shows.
DEFAULT_NOTEPAD_TITLE = "Untitled - Notepad"
DEFAULT_NOTE = "Jarvis demo note: written by Jarvis."
DEFAULT_SAVE_DIALOG_TITLE = "Save As"  # Windows shows this title in English; pass --save-dialog-title for another display language
MAX_STEPS = 20
#: How long the run's authorization grant lives: the demo takes well under a minute; the library caps any grant at 300 s.
DEMO_GRANT_SECONDS = 120.0
#: Each settling pause, and the ceiling `SettleTool` enforces. Three pauses (after the new tab, after the Save As dialog appears, after the file name is
#: typed) give Notepad time to finish handling the keys just sent before the next one arrives.
SETTLE_SECONDS = 1.5
MAX_SETTLE_SECONDS = 5.0
#: Reading the disk for the saved file. The shared plan validator allows at most 5 attempts per step (validator.py `_MAX_ALLOWED_ATTEMPTS`), and this demo
#: does not raise that shared limit; the window is widened with a longer delay instead: 5 looks, 3 s apart = a wait of about 12 s (it was 4 looks, 1.5 s
#: apart = about 4.5 s).
FILE_CHECK_ATTEMPTS = 5
FILE_CHECK_DELAY_SECONDS = 3.0
MAX_WAIT_SECONDS = 240.0
MAX_TICKS = 400
#: The step that brings Notepad forward: a read-only wait for the window, then ONE gated switch_window. It is an input step like the others (one attempt, in the grant).
FRONT_TOOL_NAME = "demo_front_notepad"
FRONT_CHECKS = 5  # up to five read-only looks for the window ...
FRONT_CHECK_DELAY_SECONDS = 2.0  # ... two seconds apart: a wait of at most 8 s, which is what the old five-attempt retry of this step amounted to
MAX_CANDIDATES = 5  # the Notepad window titles a failure report lists
INPUT_TOOLS = frozenset({"open_application", "type_text", "press_key", FRONT_TOOL_NAME})


class DemoRefused(Exception):
    """The demo will not start (nothing was done to the desktop or the disk). The message says why."""


@dataclass(frozen=True)
class DemoSetup:
    directory: Path
    path: Path
    note: str
    save_dialog_title: str = DEFAULT_SAVE_DIALOG_TITLE
    notepad_title: str = DEFAULT_NOTEPAD_TITLE


@dataclass(frozen=True)
class DemoReport:
    succeeded: bool
    text: str
    job: Job | None = None


# ============================================================ preflight: refuse early, change nothing ================================================


def real_window_list() -> list[tuple[int, str, str]]:
    """(hwnd, title, process) of the visible top-level windows, read-only - the same list `switch_window` chooses from."""
    from app.computer.windows_backend import _list_switchable_windows

    return _list_switchable_windows()


def check_ready(
    directory: object,
    note: str = DEFAULT_NOTE,
    *,
    save_dialog_title: str = DEFAULT_SAVE_DIALOG_TITLE,
    notepad_title: str = DEFAULT_NOTEPAD_TITLE,
    gate: InputGate | None = None,
    list_windows: Callable[[], list[tuple[int, str, str]]] | None = None,
    now: datetime | None = None,
    token: str | None = None,
) -> DemoSetup:
    """Everything that can be checked WITHOUT touching the desktop or the disk. Raises DemoRefused (with the reason) or returns what a run would do."""
    try:
        note = validate_note(note)
        folder = validate_directory(directory)
        if not isinstance(save_dialog_title, str) or not save_dialog_title.strip() or len(save_dialog_title) > 60 or not save_dialog_title.isprintable():
            raise DemoPathError("The Save As dialog title must be a short printable string.")
        path = new_target(folder, now=now, token=token)
    except DemoPathError as exc:
        raise DemoRefused(str(exc)) from None
    try:
        notepad_title = validate_exact_window_title(notepad_title)  # exactly as it will be requested and bound: no padding, no repeated spaces, at most 100 characters
    except ComputerControlError as exc:
        raise DemoRefused(f"The Notepad window title is not usable ({exc})") from None

    gate = gate if gate is not None else default_input_gate()
    try:
        gate.check_budget()  # non-consuming: why input would be refused right now (switched off, limit reached); the grant is minted only at --run
    except ComputerControlError as exc:
        raise DemoRefused(str(exc)) from None
    remaining = gate.budget_remaining()
    if remaining < INPUT_ACTIONS_NEEDED:
        raise DemoRefused(
            f"The demo takes {INPUT_ACTIONS_NEEDED} input actions in a row but only {remaining} remain in the current minute "
            f"(or the per-minute limit is lower than {INPUT_ACTIONS_NEEDED}); it will not start a task its budget cannot finish."
        )

    try:
        windows = (list_windows or real_window_list)()
    except Exception:  # noqa: BLE001 - if the open windows cannot be listed, "none of them is Notepad" cannot be shown
        raise DemoRefused("Could not list the open windows to confirm Notepad is not already running, so nothing was started.") from None
    in_the_way = [w for w in windows if "notepad" in str(w[1]).lower() or str(w[2]).lower() == NOTEPAD_PROCESS]
    if in_the_way:
        raise DemoRefused(
            f"{len(in_the_way)} open window(s) look like Notepad (or have 'Notepad' in their title). Close them first (including on other desktops): "
            "the demo must not type into, or save over the work in, a window that is not its own."
        )
    return DemoSetup(directory=folder, path=path, note=note, save_dialog_title=save_dialog_title, notepad_title=notepad_title)


# ================================================================== the plan =======================================================================


def plan_input_steps(plan: list[dict[str, Any]]) -> list[InputStep]:
    """The exact, ordered input actions this plan will perform, in the canonical form the backend will present: the ONE source of what the run's grant covers, taken from
    the same plan the Job runs. `press_key` uses the normalized key tokens and `open_application` the enum member, exactly as the backend does."""
    steps: list[InputStep] = []
    for raw in plan:
        tool, arguments = raw["tool"], raw["arguments"]
        if tool == "open_application":
            steps.append(InputStep("open_application", (AllowedApplication(arguments["app"]),)))
        elif tool == "press_key":
            steps.append(InputStep("press_key", (normalize_key_input(arguments["key"]),)))
        elif tool == "type_text":
            steps.append(InputStep("type_text", (arguments["text"],)))
        elif tool == FRONT_TOOL_NAME:
            steps.append(InputStep("switch_window", (validate_exact_window_title(arguments["title"]), validate_process_name(arguments["process"]))))
        elif tool in INPUT_TOOLS or tool in GATED_METHODS:  # any input action this function does not know how to describe cannot be authorized
            raise AuthorityError("the plan contains an input action that cannot be authorized.")
    return steps


def _step(step_id: str, tool: str, arguments: dict[str, Any], after: str | None, *, attempts: int = 1, delay: float = 0.0) -> dict[str, Any]:
    raw: dict[str, Any] = {"id": step_id, "tool": tool, "arguments": arguments, "depends_on": [after] if after else []}
    raw["max_attempts"] = attempts
    raw["retry_delay_seconds"] = delay
    return raw


def build_raw_plan(setup: DemoSetup) -> list[dict[str, Any]]:
    """The fixed plan. Each step depends on the one before it, so the first failure skips everything after it. Input steps: one attempt. READ steps: a
    few bounded attempts, which is how the demo waits for Notepad, the dialog and the file."""

    def in_front() -> dict[str, Any]:
        return {"check": "window_process_is", "value": NOTEPAD_PROCESS}

    chain = [
        ("open_notepad", "open_application", {"app": "notepad"}, 1, 0.0),
        ("front_notepad", FRONT_TOOL_NAME, {"title": setup.notepad_title, "process": NOTEPAD_PROCESS}, 1, 0.0),  # waits read-only INSIDE the step; ONE gated switch
        ("verify_notepad", "verify_observation", in_front(), 3, 1.0),
        ("new_document", "press_key", {"key": "ctrl+n"}, 1, 0.0),
        ("settle_after_new_tab", SETTLE_TOOL_NAME, {}, 1, 0.0),  # the new tab must be ready before the note arrives (the first live run lost characters here)
        ("verify_focus_for_note", "verify_observation", in_front(), 2, 1.0),
        ("type_note", "type_text", {"text": setup.note}, 1, 0.0),
        # Read the document back BEFORE any save. This step is also the last foreground check before Ctrl+S (the reader refuses unless the window in front is
        # notepad.exe, and it reads THAT window's visible editor), so it takes the place of the old process check at this spot. ONE attempt: never re-read, never repair.
        ("verify_document_text", DOCUMENT_TOOL_NAME, {}, 1, 0.0),
        ("open_save_as", "press_key", {"key": "ctrl+s"}, 1, 0.0),
        ("verify_save_dialog", "verify_observation", {"check": "window_title_contains", "value": setup.save_dialog_title}, 5, 1.0),
        ("settle_after_dialog", SETTLE_TOOL_NAME, {}, 1, 0.0),  # the dialog has a title before its file name box is ready for keys
        ("verify_focus_for_name", "verify_observation", in_front(), 2, 1.0),
        ("type_file_name", "type_text", {"text": str(setup.path)}, 1, 0.0),
        ("settle_before_save", SETTLE_TOOL_NAME, {}, 1, 0.0),  # let the dialog finish taking the typed path (and its suggestion list settle) before Enter
        # Everything that can be checked, checked again right before the one Enter that saves - the process (the read nearest the keystroke) last.
        ("verify_target_free", "verify_demo_target_free", {}, 1, 0.0),
        ("verify_dialog_before_save", "verify_observation", {"check": "window_title_contains", "value": setup.save_dialog_title}, 2, 1.0),
        ("verify_focus_before_save", "verify_observation", in_front(), 2, 1.0),
        ("confirm_save", "press_key", {"key": "enter"}, 1, 0.0),
        ("verify_file", "verify_demo_file", {}, FILE_CHECK_ATTEMPTS, FILE_CHECK_DELAY_SECONDS),
        # Notepad itself must now show the new file: an overwrite or error prompt in front (whose title is not the file's) fails here and is left alone.
        ("verify_saved_title", "verify_observation", {"check": "window_title_contains", "value": setup.path.stem}, 3, 1.0),
    ]
    steps: list[dict[str, Any]] = []
    previous: str | None = None
    for step_id, tool, arguments, attempts, delay in chain:
        steps.append(_step(step_id, tool, arguments, previous, attempts=attempts, delay=delay))
        previous = step_id
    return steps


class _NoPages(PageFetcher):
    """The demo only looks at windows; it never fetches a page."""

    def fetch(self, url: str) -> FetchedPage:
        raise ObservationError("this demo does not read pages")


class _NoPlanner(Planner):
    """The plan is fixed. If anything ever asked a model to change it, that is a bug, and it stops here."""

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        raise RuntimeError("the Notepad demo has a fixed plan; nothing may re-plan it")


SETTLE_TOOL_NAME = "demo_settle"


def _checked_seconds(seconds: object) -> float:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= MAX_SETTLE_SECONDS:
        raise ValueError(f"A settling pause must be more than 0 and at most {MAX_SETTLE_SECONDS:g} seconds.")
    return float(seconds)


class SettleTool(Tool):
    """READ-only: wait a short, FIXED time so the application can finish handling the keys just sent. It sends no input, reads nothing and takes no arguments,
    so a plan cannot choose how long to wait; the duration is fixed when the tool is built and may never exceed MAX_SETTLE_SECONDS. The sleep is injected so
    tests never really wait. Private to this module's registry - it is in no shared registry and no Job allowlist."""

    name = SETTLE_TOOL_NAME
    description = "Wait a short, fixed time so the application can finish handling the keys just sent. Takes no arguments. Sends no input and reads nothing."
    permission = PermissionLevel.READ
    requirements = ToolRequirements()
    automatable = False

    def __init__(self, seconds: float = SETTLE_SECONDS, *, sleep: Callable[[float], None] = time.sleep):
        self._seconds = _checked_seconds(seconds)
        self._sleep = sleep

    def execute(self) -> ToolResult:
        self._sleep(self._seconds)
        return ToolResult.success(data={"waited_seconds": self._seconds}, message=f"Waited {self._seconds:g} s for the application to settle.")


class DemoFrontNotepadTool(Tool):
    """Bring the demo's own Notepad window forward: a bounded READ-only wait for it, then ONE gated `switch_window`. Private to this module's registry - it is in no shared
    registry and no Job allowlist, and the shared `SwitchWindowTool` is neither used nor changed.

    It adds no authority and no matching rule of its own:
      * The ONLY gated call it makes is `backend.switch_window(title, process)` - the same backend method every caller goes through. The backend asks the gate (the exact
        title and process must be what the run's grant binds, in order) BEFORE it lists any window, then looks for an exact match once.
      * What counts as a match is `exact_window_matches` (app/computer/base.py), the very function the backend uses; this tool compares nothing itself.
      * It is bound at construction to ONE title and ONE process; asked for any other, it refuses before the backend is touched. It calls the backend at most once.

    The wait is the part that is not gated: up to FRONT_CHECKS looks through `list_windows` (a read-only listing), FRONT_CHECK_DELAY_SECONDS apart, through an injected
    sleep. It changes no focus and sends no input. No exact match after the last look, or more than one at any look, fails the step WITHOUT calling the backend, so the
    grant entry is not spent and nothing is typed. Once the backend is called (exactly one match was seen), its entry is spent whatever it then finds - a window that vanished,
    two windows, Windows refusing - and the step is never retried. A failure says what was expected and which notepad.exe windows are visible, read-only."""

    name = FRONT_TOOL_NAME
    description = "Bring the demo's Notepad window forward by its exact title and process: a short read-only wait for it, then one authorized switch. Sends no input."
    permission = PermissionLevel.SAFE_ACTION
    requirements = ToolRequirements()
    automatable = False

    def __init__(
        self,
        backend: ComputerBackend,
        title: str,
        process: str,
        list_windows: Callable[[], list[tuple[int, str, str]]],
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._backend = backend
        self._title = validate_exact_window_title(title)
        self._process = validate_process_name(process)
        self._list_windows = list_windows
        self._sleep = sleep
        self._called_backend = False

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="title", type="string", description="The exact title of the Notepad window", free_text=True),
            ToolParameter(name="process", type="string", description="The exact process file name, notepad.exe"),
        ]

    def _expected(self) -> str:
        return f"Expected a window titled exactly {scrub(self._title, 100)!r} of {self._process}"

    def _candidates(self, windows: list[tuple[int, str, str]]) -> str:
        titles = [scrub(str(w[1]), 100) for w in windows_of_process(windows, self._process)][:MAX_CANDIDATES]
        return "Visible " + self._process + " windows: " + ("; ".join(repr(t) for t in titles) if titles else "none") + "."

    def _listing(self) -> list[tuple[int, str, str]] | None:
        try:
            return list(self._list_windows())
        except Exception:  # noqa: BLE001 - a listing that fails is "cannot tell", and nothing is switched
            return None

    def execute(self, title: str, process: str) -> ToolResult:
        if self._called_backend:
            return ToolResult.error("This step already asked for the window once; it is never asked again, and no further entry of the grant is used.")
        try:
            asked = (validate_exact_window_title(title), validate_process_name(process))
        except ComputerControlError as exc:
            return ToolResult.error(f"Not asked: {exc}")
        if asked != (self._title, self._process):
            return ToolResult.error("This step can only bring its own Notepad window forward; it was asked for a different title or process, so nothing was done.")
        windows: list[tuple[int, str, str]] = []
        matches: list = []
        for look in range(FRONT_CHECKS):  # the READ-only wait: no focus change, no input, no gated call
            listed = self._listing()
            if listed is None:
                return ToolResult.error("The open windows could not be listed, so no window was brought forward and no entry of the grant was used.")
            windows = listed
            matches = exact_window_matches(windows, self._title, self._process)
            if len(matches) > 1:
                return ToolResult.error(
                    f"{self._expected()}, but {len(matches)} windows have exactly that title and process, so none was chosen and no entry of the grant was used. "
                    f"{self._candidates(windows)}"
                )
            if matches:
                break
            if look < FRONT_CHECKS - 1:
                self._sleep(FRONT_CHECK_DELAY_SECONDS)
        if not matches:
            return ToolResult.error(
                f"{self._expected()}, and none appeared in {FRONT_CHECKS} looks, so nothing was brought forward and no entry of the grant was used. "
                f"{self._candidates(windows)} If your Notepad shows another title, pass it with --notepad-title."
            )
        self._called_backend = True  # from here the gated action is in play: whatever it does, it is not repeated
        try:
            window = self._backend.switch_window(self._title, self._process)
        except ComputerControlError as exc:
            after = self._listing()
            return ToolResult.error(f"{self._expected()}. The authorized switch failed: {scrub(str(exc), 160)} {self._candidates(after) if after is not None else ''}".strip())
        shown = {"title": scrub(window.title, 120), "process_name": scrub(window.process_name, 60)}
        return ToolResult.success(data=shown, message=f"Brought '{shown['title']}' ({shown['process_name']}) forward.")


DOCUMENT_TOOL_NAME = "verify_document_text"


class VerifyDocumentTextTool(Tool):
    """READ-only, run right after the note is typed and BEFORE any save: does the visible Notepad document hold exactly the expected note? It waits one fixed
    settling pause (so Notepad can finish handling the keys just sent), then reads the editor ONCE through `read_document` and compares the whole text. Any
    difference, and any failure to read (no editor, ambiguous editor, not Notepad, no answer in time, an API error), is a failure; a failure here stops the plan
    before Ctrl+S. It never repairs, retypes or re-reads. It takes no arguments, sends no input, and reports only match / mismatch and the two LENGTHS - never
    the document's text, which can hold anything that was already in Notepad. Private to this module's registry."""

    name = DOCUMENT_TOOL_NAME
    description = "Check that the visible Notepad document holds exactly the expected note, before saving. Takes no arguments. Read-only; reports lengths, not text."
    permission = PermissionLevel.READ
    requirements = ToolRequirements()
    automatable = False

    def __init__(self, expected: str, read_document: Callable[[], str], *, sleep: Callable[[float], None] = time.sleep, settle_seconds: float = SETTLE_SECONDS):
        self._expected = validate_note(expected)
        self._read = read_document
        self._sleep = sleep
        self._settle = _checked_seconds(settle_seconds)

    def execute(self) -> ToolResult:
        self._sleep(self._settle)
        try:
            actual = self._read()
        except DocumentReadError as exc:
            return ToolResult.error(f"Document check failed: {exc}. Save As was not opened and nothing was saved.")
        except Exception as exc:  # noqa: BLE001 - never echo the exception's own text: it could carry window content
            return ToolResult.error(f"Document check failed: the Notepad editor could not be read ({exc.__class__.__name__}). Save As was not opened and nothing was saved.")
        if not isinstance(actual, str):
            return ToolResult.error("Document check failed: the editor did not return text. Save As was not opened and nothing was saved.")
        if actual != self._expected:
            return ToolResult.error(
                f"Document check failed: the document does not hold the expected note (expected {len(self._expected)} characters, found {len(actual)}). "
                "Nothing was repaired or retyped, Save As was not opened and nothing was saved."
            )
        return ToolResult.success(data={"verified": True, "characters": len(actual)}, message=f"Verified: the document holds exactly the expected {len(actual)} characters.")


def build_registry(
    backend: ComputerBackend,
    setup: DemoSetup,
    *,
    sleep: Callable[[float], None] = time.sleep,
    read_document: Callable[[], str] | None = None,
    list_windows: Callable[[], list[tuple[int, str, str]]] | None = None,
) -> ToolRegistry:
    """A PRIVATE registry: the three input tools and the demo's own window step over `backend`, window verification, the settling pause, and the two file checks bound to
    this run's directory and file. It is registered nowhere else, so it widens nothing. `read_document` is how the document is read back before the save;
    the default is the real read-only reader, and tests always pass a fake."""
    registry = ToolRegistry()
    registry.register(SettleTool(sleep=sleep))
    registry.register(VerifyDocumentTextTool(setup.note, read_document if read_document is not None else (lambda: read_visible_editor_text()), sleep=sleep))
    observer = ObservationEngine(fetcher=_NoPages(), backend=backend)
    registry.register(OpenApplicationTool(backend))
    registry.register(PressKeyTool(backend))
    registry.register(TypeTextTool(backend))
    registry.register(DemoFrontNotepadTool(backend, setup.notepad_title, NOTEPAD_PROCESS, list_windows if list_windows is not None else real_window_list, sleep=sleep))
    registry.register(VerifyObservationTool(observer))
    registry.register(CheckDemoTargetFreeTool(setup.directory, setup.path))
    registry.register(VerifyDemoFileTool(setup.directory, setup.path, setup.note))
    return registry


# ================================================================== running it =====================================================================


def run_demo(
    setup: DemoSetup,
    backend: ComputerBackend,
    *,
    sleep: Callable[[float], None] = time.sleep,
    read_document: Callable[[], str] | None = None,
    list_windows: Callable[[], list[tuple[int, str, str]]] | None = None,
) -> DemoReport:
    """Run the plan through the existing JobEngine and report what really happened. `backend` must be the AUTHORIZED one (`create_authorized_computer_backend(grant)`): the
    default backend carries no grant and admits nothing."""
    try:
        directory = validate_directory(setup.directory)
        if not directory.exists():
            directory.mkdir()  # one level; the parent was checked to exist
        check_target(directory, setup.path, must_not_exist=True)  # re-checked now: never save over a file that appeared since the preflight
    except (DemoPathError, OSError) as exc:
        raise DemoRefused(str(exc) if isinstance(exc, DemoPathError) else f"The directory could not be prepared ({exc.__class__.__name__}).") from None

    registry = build_registry(backend, setup, sleep=sleep, read_document=read_document, list_windows=list_windows)
    steps = validate_plan(build_raw_plan(setup), registry, max_steps=MAX_STEPS, default_retry=RetryPolicy())
    for step in steps:
        if step.tool_name in INPUT_TOOLS and step.retry_policy.max_attempts != 1:
            raise DemoRefused(f"Internal check failed: input step '{step.id}' would be attempted more than once.")

    store = SQLiteJobStore(":memory:")  # nothing is persisted: the run lives and dies with this process
    job = store.create_running_job("Notepad demo: write a short note and save it to a new file", steps, autonomy=AutonomyLevel.ASSISTED)
    engine = JobEngine(
        store=store, tools=registry, planner=_NoPlanner(), max_steps=MAX_STEPS, max_total_steps=2 * MAX_STEPS, max_retries_per_step=1,
        max_replans=0, retry_delay_seconds=0.0, poll_interval_seconds=1000.0, sleep=sleep,
    )
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    for _ in range(MAX_TICKS):
        engine.tick()
        job = store.get(job.id)
        if job.status in TERMINAL_JOB_STATUSES or time.monotonic() > deadline:
            break
    return build_report(job, setup)


# ================================================================== the report =====================================================================


def build_report(job: Job, setup: DemoSetup) -> DemoReport:
    steps = list(job.plan.steps) if job.plan else []
    by_id = {s.id: s for s in steps}

    def status(step_id: str) -> StepStatus | None:
        return by_id[step_id].status if step_id in by_id else None

    ok = job.status is JobStatus.COMPLETED and status("verify_file") is StepStatus.SUCCEEDED and status("verify_saved_title") is StepStatus.SUCCEEDED
    lines = [f"Notepad demo: {'SUCCEEDED' if ok else 'DID NOT SUCCEED'}", f"Target file: {setup.path}", "Steps:"]
    for number, step in enumerate(steps, start=1):
        detail = ""
        if step.status is StepStatus.FAILED and step.error:
            detail = f" - {scrub(step.error, 600 if step.id == 'front_notepad' else 220)}"  # the window step lists the expected title and what is visible
        lines.append(f"  {number:>2}. {step.id} ({step.tool_name}): {step.status.value}, {step.attempts} attempt(s){detail}")
    if ok:
        lines.append("Checked from the disk, not from the keystrokes: " + scrub((by_id["verify_file"].result or {}).get("message", ""), 200))
        return DemoReport(True, "\n".join(lines), job)

    failed = next((s for s in steps if s.status is StepStatus.FAILED), None)
    skipped = [s.id for s in steps if s.status is StepStatus.SKIPPED]
    if failed is not None:
        lines.append(f"Stopped at '{failed.id}'." + (f" Not run: {', '.join(skipped)}." if skipped else ""))
    else:
        lines.append(f"The Job ended as '{job.status.value}' without finishing the plan.")
    left: list[str] = []
    if status("open_notepad") is StepStatus.SUCCEEDED:
        left.append("Notepad was opened and is still open (Jarvis does not close it)")
    if status("type_note") is StepStatus.SUCCEEDED:
        left.append("the note text was typed into a Notepad document that may be unsaved")
    if status("verify_document_text") is StepStatus.FAILED:
        left.append(
            "the typed note was NOT confirmed in the Notepad document (the step's error gives the lengths), so Save As was never opened and nothing was saved - "
            "look at the document yourself, then close it or fix it by hand"
        )
    if status("open_save_as") is StepStatus.SUCCEEDED and status("confirm_save") is not StepStatus.SUCCEEDED:
        left.append(
            "the Save As dialog may still be open with the note unsaved"
            + (" and the file path typed into it - Enter was NOT pressed, so nothing was saved" if status("type_file_name") is StepStatus.SUCCEEDED else "")
        )
    if status("confirm_save") is StepStatus.SUCCEEDED and status("verify_file") is not StepStatus.SUCCEEDED:
        left.append(
            "Enter was pressed in the Save As dialog, but the file was not confirmed on disk - a pressed Enter is not a saved file: the dialog may still be open "
            "with the text unsaved, and if you finish the save yourself the file will hold whatever Notepad's document contains, which may differ from the note"
            " - look at Notepad for a message or a prompt"
        )
    if status("confirm_save") is StepStatus.SUCCEEDED and status("verify_saved_title") is not StepStatus.SUCCEEDED:
        left.append(
            "after the one Enter Notepad's title did not show the new file's name, so a prompt (for example an overwrite confirmation) or an error may be open - "
            "Jarvis pressed nothing to answer it and will not; deal with it yourself and check the directory before trusting any file in it"
        )
    lines.append("Left on the screen: " + ("; ".join(left) + "." if left else "nothing was started."))
    lines.append("Nothing was retried, closed or deleted by Jarvis. Input actions are never repeated; review the screen and the directory yourself.")
    lines.append("A typed path or a pressed Enter is not a saved file - only the check of the disk counts. Check the text in Notepad before saving anything by hand.")
    return DemoReport(False, "\n".join(lines), job)


# ====================================================================== CLI =====================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.demos.notepad_note",
        description="Open Notepad, write a note, save it to a NEW file, verify it from the disk.",
        allow_abbrev=False,  # "--r" must never be read as "--run": the real desktop run needs the whole flag, spelled out
    )
    parser.add_argument("--dir", required=True, help="the dedicated directory (absolute; its own name must start with 'jarvis_demo')")
    parser.add_argument("--note", default=DEFAULT_NOTE, help="the one-line ASCII note to write")
    parser.add_argument("--save-dialog-title", default=DEFAULT_SAVE_DIALOG_TITLE, help="the Save As window title in your Windows display language")
    parser.add_argument(
        "--notepad-title", default=DEFAULT_NOTEPAD_TITLE,
        help="the EXACT title of the Notepad window a fresh launch shows (case-sensitive, at most 100 characters, no padding); the run's grant binds this exact string",
    )
    parser.add_argument("--run", action="store_true", help="really do it on the desktop (without this, only the dry run happens)")
    args = parser.parse_args(argv)

    try:
        setup = check_ready(args.dir, args.note, save_dialog_title=args.save_dialog_title, notepad_title=args.notepad_title)
    except DemoRefused as exc:
        print(f"Not started: {exc}")
        return 2
    plan = build_raw_plan(setup)
    print(
        f"Ready. Directory: {setup.directory}\nNew file: {setup.path.name}\nNote: {setup.note}\n"
        f"Notepad window it will bring forward: exactly {setup.notepad_title!r} of {NOTEPAD_PROCESS}\n"
        f"Plan: {len(plan)} steps, {INPUT_ACTIONS_NEEDED} input actions."
    )
    if not args.run:
        print("Dry run only: nothing was opened, typed, created or saved. Add --run to do it for real.")
        return 0

    from app.computer.factory import create_authorized_computer_backend
    from app.computer.issuer import default_issuer

    try:
        issuer = default_issuer()
        grant = issuer.issue(plan_input_steps(plan), ttl_seconds=DEMO_GRANT_SECONDS, purpose="notepad_demo", actor="cli")
    except (AuthorityError, ComputerControlError) as exc:  # switched off, or an action that cannot be authorized: nothing was minted, nothing was started
        print(f"Not started: {exc}")
        return 2
    try:
        report = run_demo(setup, create_authorized_computer_backend(grant))
    except DemoRefused as exc:
        print(f"Not started: {exc}")
        return 2
    finally:
        issuer.gate.revoke(grant)  # the grant never outlives the run, whatever happened in it
    print(report.text)
    return 0 if report.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
