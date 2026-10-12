"""
The Notepad demonstration, end to end, WITHOUT a real desktop.

Real: the plan, the validator, `create_running_job`, the JobEngine, the ToolRegistry, the real input/observation/file tools, the real Computer Input Gate
(`GatedComputerBackend` over a fake desktop; and, where it matters, the real `WindowsComputerBackend` with its OS libraries replaced by tripwires) and the
real disk (pytest's temp directory). Stand-ins: the desktop itself (a small Notepad simulation that can fail in each way a real one can) and the clock/sleep.
Nothing here ever touches the real desktop: the Windows backend's `pyautogui`, `subprocess` and `webbrowser` are tripwires for every test in this file.
"""

import ast
import os
import re
from pathlib import Path

import pytest

from app.computer import windows_backend
from app.computer.base import AllowedApplication, ComputerBackend, ComputerControlError, ScreenInfo, WindowInfo
from app.computer.defaults import JOB_COMPUTER_TOOL_NAMES
from app.computer.gate import GatedComputerBackend, InputGate, default_input_gate
from app.computer.windows_backend import WindowsComputerBackend
from app.config import settings
from app.demos import file_check, notepad_note, notepad_reader
from app.demos.notepad_reader import DocumentReadError
from app.demos.file_check import DemoPathError, VerifyDemoFileTool, check_target, new_target, validate_directory, validate_note
from app.demos.notepad_note import (
    INPUT_ACTIONS_NEEDED,
    DemoRefused,
    DemoSetup,
    build_raw_plan,
    build_registry,
    check_ready,
    run_demo,
)
from app.orchestration.base import JobStatus, RetryPolicy, StepStatus
from app.orchestration.validator import PlanValidationError, validate_plan
from app.tools.base import ToolResultStatus

NOTE = "Jarvis demo note: written by Jarvis."
APP_DIR = Path(__file__).resolve().parent.parent / "app"
REAL_WIN32_WINDOW_API = notepad_reader.Win32WindowApi  # captured at import, before the autouse fixture swaps in a tripwire; only the constructor-only test uses it


class _NoDesktop:
    def __getattr__(self, name):
        raise AssertionError(f"the real desktop was touched: {name}")


@pytest.fixture(autouse=True)
def no_real_desktop(monkeypatch):
    for attribute in ("pyautogui", "subprocess", "webbrowser"):
        monkeypatch.setattr(windows_backend, attribute, _NoDesktop())


@pytest.fixture(autouse=True)
def no_real_document_read(monkeypatch):
    """The real document reader asks Windows about the real foreground window (read-only, but still the desktop): no test may use it, by either door -
    the name the demo calls (`notepad_note.read_visible_editor_text`) or the reader module's own `read_visible_editor_text()` called without a fake API, which
    would build the real `Win32WindowApi`. Both are replaced by tripwires that record the attempt and fail the test at teardown. A test that reaches one ON
    PURPOSE (to prove the wiring) takes this fixture by name and clears the list after checking it. The real class is kept in REAL_WIN32_WINDOW_API for the
    one constructor-only test."""
    used = []

    def tripwire(*args, **kwargs):
        used.append("notepad_note.read_visible_editor_text")
        raise DocumentReadError("tripwire: a test reached the real document reader")

    class WindowApiTripwire:
        def __init__(self, *args, **kwargs):
            used.append("notepad_reader.Win32WindowApi")
            raise RuntimeError("tripwire: a test tried to build the real Windows API")

    monkeypatch.setattr(notepad_note, "read_visible_editor_text", tripwire)
    monkeypatch.setattr(notepad_reader, "Win32WindowApi", WindowApiTripwire)
    yield used
    assert used == [], f"a test used the real Notepad document reader or the real Windows API: {used}"


# ============================================================ a Notepad that can fail the ways a real one can ======================================


class FakeDesktop(ComputerBackend):
    """A tiny Notepad. The terminal has focus; `open_application` starts Notepad but (like Windows' foreground lock) does not bring it forward;
    `switch_window` does. Ctrl+S opens a Save As dialog, a name typed there is kept, and Enter writes the document to that path on the REAL disk."""

    def __init__(self, *, appear_after=0, process="notepad.exe", dialog=True, mistype=False, save_works=True, override_from=None, after_name_typed=None, before_enter=None, title_after_save=None, drop_index=None, save_text=None, initial_document="", ignore_new_tab=False, swallow_typing=False):
        self.drop_index, self.save_text, self.ignore_new_tab, self.swallow_typing = drop_index, save_text, ignore_new_tab, swallow_typing  # how typing / saving can go wrong
        self.title_after_save = title_after_save  # what Notepad's title says after a save (default: the new file's name)
        self.appear_after, self.process, self.dialog_works, self.mistype, self.save_works = appear_after, process, dialog, mistype, save_works
        self.override_from = override_from  # (n, WindowInfo): from the n-th read of the foreground window on, a different window is in front
        self.after_name_typed = after_name_typed  # called with the typed path right after it was typed into the Save As dialog
        self.before_enter = before_enter  # called with the path at the instant Enter is pressed in the Save As dialog
        self.reads = 0
        self.prompt_open = False
        self.window = WindowInfo("Terminal", "terminal.exe", False)
        self.log: list[tuple] = []
        self.typed: list[str] = []
        self.misdirected: list[str] = []
        self.switches = 0
        self.opened = False
        self.document = initial_document
        self.dialog_open = False
        self.name = ""

    def _in_notepad(self):
        return self.window.process_name == "notepad.exe"

    def get_active_window(self):
        self.reads += 1
        if self.override_from is not None and self.reads >= self.override_from[0]:
            return self.override_from[1]
        return self.window

    def get_screen_info(self):
        return ScreenInfo(1920, 1080)

    def open_url(self, url):
        raise AssertionError("the demo never opens a URL")

    def scroll(self, amount):
        raise AssertionError("the demo never scrolls")

    def move_mouse(self, x, y):
        raise AssertionError("the demo never moves the mouse")

    def click_mouse(self, x, y, button):
        raise AssertionError("the demo never clicks")

    def double_click(self, x, y, button):
        raise AssertionError("the demo never double-clicks")

    def open_application(self, app):
        assert app is AllowedApplication.NOTEPAD
        self.log.append(("open", app.value))
        self.opened = True

    def switch_window(self, title):
        self.switches += 1
        if not self.opened or self.switches <= self.appear_after:
            raise ComputerControlError(f"No visible window title contains {title!r}.")
        self.window = WindowInfo("Untitled - Notepad", self.process, False)
        return self.window

    def type_text(self, text):
        self.log.append(("type", text))
        self.typed.append(text)
        if self.dialog_open:
            self.name += text
            if self.after_name_typed:
                self.after_name_typed(Path(self.name))
        elif self._in_notepad():
            typed = ("X" + text[1:]) if self.mistype else text
            if self.drop_index is not None:
                typed = typed[: self.drop_index] + typed[self.drop_index + 1 :]  # one keystroke lost, like the live runs
            if not self.swallow_typing:
                self.document += typed
        else:
            self.misdirected.append(text)

    def press_key(self, key):
        self.log.append(("key", key))
        if key == "ctrl+n" and self._in_notepad():
            if not self.ignore_new_tab:
                self.document = ""
        elif key == "ctrl+s" and self._in_notepad() and self.dialog_works:
            self.dialog_open = True
            self.window = WindowInfo("Save As", self.process, False)
        elif key == "enter" and self.dialog_open:
            if self.before_enter:
                self.before_enter(Path(self.name))
            if Path(self.name).exists():
                # Windows' overwrite confirmation: it waits for an answer; this fake (like Jarvis) gives none, so nothing is replaced.
                self.prompt_open = True
                self.window = WindowInfo("Confirm Save As", self.process, False)
                return
            self.dialog_open = False
            self.window = WindowInfo(self.title_after_save or f"{Path(self.name).name} - Notepad", self.process, False)
            if self.save_works:
                Path(self.name).write_text(self.document if self.save_text is None else self.save_text, encoding="utf-8")

    def read_document(self):
        """What the real reader does: refuse unless the window in front is notepad.exe (one foreground read, like every other check), else return the document."""
        window = self.get_active_window()
        if window.process_name != "notepad.exe":
            raise DocumentReadError("the window in front does not belong to notepad.exe")
        return self.document


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def gate(limit=10, enabled=True):
    return InputGate(enabled=lambda: enabled, limit=lambda: limit, clock=Clock())


def setup_for(tmp_path, *, gate_=None, name="jarvis_demo_run"):
    return check_ready(tmp_path / name, NOTE, gate=gate_ or gate(), list_windows=lambda: [], token="abc123")


def run(tmp_path, desktop=None, *, limit=10, enabled=True, name="jarvis_demo_run", read_document=None):
    desktop = desktop or FakeDesktop()
    the_gate = gate(limit, enabled)
    setup = check_ready(tmp_path / name, NOTE, gate=gate(), list_windows=lambda: [], token="abc123")  # preflight against a healthy gate ...
    report = run_demo(setup, GatedComputerBackend(desktop, the_gate), sleep=lambda seconds: None, read_document=read_document or desktop.read_document)  # ... the run against the gate under test
    return setup, desktop, report, the_gate


def steps_of(report):
    return {s.id: s for s in report.job.plan.steps}


# ================================================================== the happy path =================================================================


def test_the_whole_task_runs_and_the_file_is_verified_from_the_disk(tmp_path):
    setup, desktop, report, the_gate = run(tmp_path)

    assert report.succeeded and report.job.status is JobStatus.COMPLETED
    assert desktop.log == [
        ("open", "notepad"),
        ("key", "ctrl+n"),
        ("type", NOTE),
        ("key", "ctrl+s"),
        ("type", str(setup.path)),
        ("key", "enter"),
    ]
    assert desktop.misdirected == []
    assert setup.path.read_text(encoding="utf-8") == NOTE  # the disk, not the keystrokes
    assert sorted(p.name for p in setup.directory.iterdir()) == [setup.path.name]
    assert the_gate.remaining() == 10 - INPUT_ACTIONS_NEEDED  # six gated actions, no more
    assert all(s.status is StepStatus.SUCCEEDED for s in report.job.plan.steps)
    assert "SUCCEEDED" in report.text and "Checked from the disk" in report.text


def test_every_input_step_has_exactly_one_attempt_and_only_reads_are_retried(tmp_path):
    _, _, report, _ = run(tmp_path)

    for step in report.job.plan.steps:
        if step.tool_name in notepad_note.INPUT_TOOLS:
            assert step.retry_policy.max_attempts == 1 and step.attempts == 1, step.id


def test_a_notepad_that_appears_late_is_waited_for_by_the_read_step_without_repeating_any_input(tmp_path):
    desktop = FakeDesktop(appear_after=2)

    _, _, report, _ = run(tmp_path, desktop)

    steps = steps_of(report)
    assert report.succeeded and steps["front_notepad"].attempts == 3
    assert desktop.log.count(("open", "notepad")) == 1 and [k for k in desktop.log if k[0] == "key"] == [("key", "ctrl+n"), ("key", "ctrl+s"), ("key", "enter")]


# ================================================================== the plan itself ================================================================


def test_the_plan_is_valid_chained_and_verifies_focus_before_every_typing_step(tmp_path):
    setup = setup_for(tmp_path)
    registry = build_registry(FakeDesktop(), setup)
    raw = build_raw_plan(setup)

    steps = validate_plan(raw, registry, max_steps=20, default_retry=RetryPolicy())

    ids = [s.id for s in steps]
    assert [s.depends_on for s in steps] == [[]] + [[previous] for previous in ids[:-1]]  # one chain: a failure skips everything after it
    in_front = {"check": "window_process_is", "value": "notepad.exe"}
    for keystroke in ("new_document", "type_note", "type_file_name", "confirm_save"):  # EVERY keystroke after the launch ...
        before = steps[ids.index(keystroke) - 1]  # ... has a read of the foreground window as the step right before it
        assert before.tool_name == "verify_observation" and before.arguments in (in_front,), keystroke
    assert steps[ids.index("open_save_as") - 1].tool_name == "verify_document_text"  # ... and Ctrl+S is preceded by the document check, which also checks the foreground
    assert ids.index("verify_save_dialog") < ids.index("type_file_name") and ids.index("open_save_as") < ids.index("verify_save_dialog")
    assert ids[ids.index("confirm_save") - 3 : ids.index("confirm_save")] == ["verify_target_free", "verify_dialog_before_save", "verify_focus_before_save"]
    assert sum(1 for s in steps if s.tool_name in notepad_note.INPUT_TOOLS) == INPUT_ACTIONS_NEEDED
    assert ids[-2:] == ["verify_file", "verify_saved_title"] and steps[-2].arguments == {}


def test_the_file_tool_accepts_no_arguments_so_a_plan_cannot_name_a_path(tmp_path):
    setup = setup_for(tmp_path)
    registry = build_registry(FakeDesktop(), setup)
    tool = registry.get("verify_demo_file")

    assert tool.parameters == []
    for forbidden in ({"path": str(tmp_path / "other.txt")}, {"file": "x"}, {"directory": str(tmp_path)}):
        raw = [{"id": "s1", "tool": "verify_demo_file", "arguments": forbidden, "depends_on": []}]
        with pytest.raises(PlanValidationError, match="unexpected argument"):
            validate_plan(raw, registry, max_steps=20, default_retry=RetryPolicy())
    with pytest.raises(TypeError):
        tool.execute(path=str(tmp_path / "other.txt"))


def test_nothing_is_widened_the_job_allowlist_is_unchanged_and_the_file_tool_exists_nowhere_else():
    assert JOB_COMPUTER_TOOL_NAMES == frozenset({"open_url", "scroll", "get_active_window", "switch_window"})
    users = [p.relative_to(APP_DIR).as_posix() for p in APP_DIR.rglob("*.py") if "verify_demo_file" in p.read_text(encoding="utf-8")]
    assert sorted(users) == ["demos/file_check.py", "demos/notepad_note.py"]
    for p in APP_DIR.rglob("*.py"):
        if p.relative_to(APP_DIR).parts[0] != "demos":
            assert "app.demos" not in p.read_text(encoding="utf-8"), p  # nothing in the application imports the demo


# ============================================================ the gate: disabled, exhausted, never bypassed ========================================


def test_with_input_switched_off_nothing_reaches_the_desktop_and_the_job_fails_at_the_first_input(tmp_path):
    setup, desktop, report, _ = run(tmp_path, enabled=False)

    steps = steps_of(report)
    assert not report.succeeded and report.job.status is JobStatus.FAILED
    assert desktop.log == [] and not setup.path.exists()
    assert steps["open_notepad"].status is StepStatus.FAILED and steps["open_notepad"].attempts == 1
    assert "switched off" in steps["open_notepad"].error
    assert all(s.status is StepStatus.SKIPPED for sid, s in steps.items() if sid != "open_notepad")
    assert "nothing was started" in report.text


def test_the_real_windows_backend_with_input_off_is_refused_by_its_own_gate_and_no_os_call_happens(tmp_path, monkeypatch):
    process_gate = default_input_gate()
    monkeypatch.setattr(process_gate, "_events", type(process_gate._events)())
    monkeypatch.setattr(settings, "computer_input_enabled", False)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10)
    setup = DemoSetup(tmp_path / "jarvis_demo_real", tmp_path / "jarvis_demo_real" / "jarvis_demo_20260101T000000Z_abc123.txt", NOTE)

    report = run_demo(setup, WindowsComputerBackend(), sleep=lambda seconds: None)  # the tripwires would raise on any OS call

    assert not report.succeeded and steps_of(report)["open_notepad"].status is StepStatus.FAILED
    assert "switched off" in steps_of(report)["open_notepad"].error


def test_when_the_budget_runs_out_mid_task_the_refused_action_is_not_repeated_and_nothing_later_runs(tmp_path):
    setup, desktop, report, _ = run(tmp_path, limit=4)  # open, ctrl+n, the note and ctrl+s fit; the file name does not

    steps = steps_of(report)
    assert not report.succeeded
    assert steps["type_file_name"].status is StepStatus.FAILED and steps["type_file_name"].attempts == 1
    assert "limited to 4 actions per minute" in steps["type_file_name"].error
    assert steps["confirm_save"].status is StepStatus.SKIPPED and steps["verify_file"].status is StepStatus.SKIPPED
    assert desktop.log[-1] == ("key", "ctrl+s") and not setup.path.exists()
    assert desktop.misdirected == []  # the path was never typed anywhere, least of all into the note
    assert "Save As dialog may still be open" in report.text


# ===================================================== preflight: refuse early, and say so, before anything is touched =============================


def test_preflight_refuses_when_input_is_switched_off_and_creates_nothing(tmp_path):
    with pytest.raises(DemoRefused, match="switched off"):
        check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=gate(enabled=False), list_windows=lambda: [])
    assert not (tmp_path / "jarvis_demo_x").exists()


def test_preflight_refuses_a_limit_below_what_the_task_needs(tmp_path):
    with pytest.raises(DemoRefused, match="only 5 remain"):
        check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=gate(limit=5), list_windows=lambda: [])


def test_preflight_counts_recent_actions_against_the_rolling_minute(tmp_path):
    clock = Clock()
    busy = InputGate(enabled=lambda: True, limit=lambda: 10, clock=clock)
    for _ in range(5):
        busy.admit("press_key")  # five slots already used in the last minute
    with pytest.raises(DemoRefused, match="only 5 remain"):
        check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=busy, list_windows=lambda: [])

    clock.now += 60  # they age out
    assert check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=busy, list_windows=lambda: []).note == NOTE


def test_preflight_refuses_when_the_limit_is_already_reached(tmp_path):
    full = gate(limit=1)
    full.admit("press_key")
    with pytest.raises(DemoRefused, match="limited to 1 actions per minute"):
        check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=full, list_windows=lambda: [])


@pytest.mark.parametrize(
    "windows",
    [
        [(1, "Untitled - Notepad", "notepad.exe")],
        [(1, "notes.txt - Notepad", "Notepad.exe")],
        [(1, "my notepad things - Editor", "editor.exe")],  # switch_window("Notepad") would match this one
        [(1, "Inbox", "mail.exe"), (2, "x", "notepad.exe")],
    ],
)
def test_preflight_refuses_when_a_notepad_like_window_is_already_open(tmp_path, windows):
    with pytest.raises(DemoRefused, match="look like Notepad") as refused:
        check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=gate(), list_windows=lambda: windows)
    assert "Inbox" not in str(refused.value) and "notes.txt" not in str(refused.value)  # window titles are the user's; they are counted, not quoted
    assert not (tmp_path / "jarvis_demo_x").exists()


def test_preflight_refuses_when_the_open_windows_cannot_be_listed(tmp_path):
    def broken():
        raise OSError("no")

    with pytest.raises(DemoRefused, match="Could not list"):
        check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=gate(), list_windows=broken)


def test_preflight_passes_when_nothing_is_in_the_way_and_changes_nothing(tmp_path):
    setup = check_ready(tmp_path / "jarvis_demo_x", NOTE, gate=gate(), list_windows=lambda: [(1, "Inbox", "mail.exe")], token="abc123")
    assert setup.path.name.startswith("jarvis_demo_") and setup.path.name.endswith("_abc123.txt") and setup.path.parent == setup.directory
    assert not setup.directory.exists()


# ===================================================== paths: unsafe directories, existing targets, links ========================================


@pytest.mark.parametrize("bad", ["relative/jarvis_demo", "", "C:\\", "C:/", "jarvis_demo", None, 42])
def test_unsafe_directories_are_refused(bad):
    with pytest.raises(DemoPathError):
        validate_directory(bad)


def test_the_directory_must_be_one_made_for_the_demo(tmp_path):
    for bad in ("Documents", "my_demo", "demo_jarvis"):
        (tmp_path / bad).mkdir()
        with pytest.raises(DemoPathError, match="must start with 'jarvis_demo'"):
            validate_directory(tmp_path / bad)
    with pytest.raises(DemoPathError, match="must start with 'jarvis_demo'"):
        validate_directory(Path.home())


def test_dot_dot_non_ascii_and_overlong_paths_are_refused(tmp_path):
    with pytest.raises(DemoPathError, match=r"'\.\.'"):
        validate_directory(str(tmp_path / "other" / ".." / "jarvis_demo_x"))
    with pytest.raises(DemoPathError, match="plain ASCII"):
        validate_directory(str(tmp_path / "jarvis_demo_\u00e9"))
    with pytest.raises(DemoPathError, match="plain ASCII"):
        validate_directory(str(tmp_path / ("jarvis_demo_" + "a" * 300)))


def test_a_missing_parent_is_refused_the_demo_creates_one_level_only(tmp_path):
    with pytest.raises(DemoPathError, match="parent does not exist"):
        validate_directory(tmp_path / "nope" / "jarvis_demo_x")
    assert not (tmp_path / "nope").exists()


def test_a_file_where_the_directory_should_be_is_refused(tmp_path):
    (tmp_path / "jarvis_demo_file").write_text("x")
    with pytest.raises(DemoPathError, match="not a plain directory"):
        validate_directory(tmp_path / "jarvis_demo_file")


def test_a_directory_holding_anything_at_all_is_not_fresh_and_is_left_alone(tmp_path):
    folder = tmp_path / "jarvis_demo_mixed"
    folder.mkdir()
    (folder / "taxes.xlsx").write_text("private")
    with pytest.raises(DemoPathError, match="already holds 1 item"):
        validate_directory(folder)
    assert (folder / "taxes.xlsx").read_text() == "private"


def test_even_an_earlier_demo_file_makes_a_directory_stale_every_run_needs_a_new_or_empty_one(tmp_path):
    folder = tmp_path / "jarvis_demo_again"
    folder.mkdir()
    (folder / "jarvis_demo_20260101T000000Z_aaaaaa.txt").write_text("old")
    with pytest.raises(DemoPathError, match="Use a new directory, or an empty one"):
        validate_directory(folder)
    assert (folder / "jarvis_demo_20260101T000000Z_aaaaaa.txt").read_text() == "old"
    with pytest.raises(DemoRefused, match="already holds 1 item"):
        check_ready(folder, NOTE, gate=gate(), list_windows=lambda: [])


def test_a_new_directory_and_an_existing_empty_one_are_both_fresh(tmp_path):
    empty = tmp_path / "jarvis_demo_empty"
    empty.mkdir()
    assert validate_directory(empty) == empty
    assert validate_directory(tmp_path / "jarvis_demo_new") == tmp_path / "jarvis_demo_new"


def test_a_directory_reached_through_a_link_is_refused(tmp_path, monkeypatch):
    real = tmp_path / "jarvis_demo_real"
    real.mkdir()
    monkeypatch.setattr(file_check.os.path, "realpath", lambda p: str(tmp_path / "elsewhere"))  # what a link/junction anywhere in the path looks like
    with pytest.raises(DemoPathError, match="link or junction"):
        validate_directory(real)


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are a Windows feature")
def test_a_real_junction_is_refused_as_the_directory_and_anywhere_above_it(tmp_path):
    import subprocess  # the test's own (the tripwires replace only the Windows backend's)

    target = tmp_path / "somewhere"
    target.mkdir()
    junction = tmp_path / "jarvis_demo_junction"
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip("could not create a junction here")
    inner = target / "jarvis_demo_inside"
    inner.mkdir()
    with pytest.raises(DemoPathError):
        validate_directory(junction)
    with pytest.raises(DemoPathError, match="link or junction"):
        validate_directory(tmp_path / "jarvis_demo_junction" / "jarvis_demo_inside")  # a junction ABOVE the directory is just as bad


def test_an_actual_symlink_is_refused_where_the_platform_allows_creating_one(tmp_path):
    real = tmp_path / "somewhere"
    real.mkdir()
    link = tmp_path / "jarvis_demo_link"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symbolic links")
    with pytest.raises(DemoPathError):
        validate_directory(link)


def test_an_existing_target_is_never_reused_and_is_left_untouched(tmp_path):
    folder = tmp_path / "jarvis_demo_run"
    folder.mkdir()
    existing = folder / "jarvis_demo_20260101T000000Z_abc123.txt"
    existing.write_text("precious")

    with pytest.raises(DemoPathError, match="already exists"):
        new_target(folder, now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc), token="abc123")
    assert existing.read_text() == "precious"


def test_a_target_that_appears_between_the_preflight_and_the_run_stops_the_run_before_the_desktop_is_touched(tmp_path):
    setup = setup_for(tmp_path)
    setup.directory.mkdir()
    setup.path.write_text("someone else's file")
    desktop = FakeDesktop()

    with pytest.raises(DemoRefused, match="already holds 1 item"):
        run_demo(setup, GatedComputerBackend(desktop, gate()), sleep=lambda seconds: None)
    assert desktop.log == [] and setup.path.read_text() == "someone else's file"


@pytest.mark.parametrize("name", ["note.txt", "jarvis_demo_x.txt", "jarvis_demo_20260101T000000Z_abc123.md", "../jarvis_demo_20260101T000000Z_abc123.txt"])
def test_only_a_demo_named_file_directly_inside_the_directory_is_a_valid_target(tmp_path, name):
    folder = tmp_path / "jarvis_demo_run"
    with pytest.raises(DemoPathError):
        check_target(folder, folder / name, must_not_exist=False)
    with pytest.raises(DemoPathError):
        check_target(folder, tmp_path / "jarvis_demo_20260101T000000Z_abc123.txt", must_not_exist=False)  # right name, wrong directory


@pytest.mark.parametrize("bad", ["", " leading", "trailing ", "two\nlines", "caf\u00e9", "x" * 201, None, 5])
def test_the_note_must_be_one_plain_ascii_line(bad):
    with pytest.raises(DemoPathError):
        validate_note(bad)


# ============================================================ the file tool: independent verification ===============================================


def make_tool(tmp_path, expected=NOTE):
    folder = tmp_path / "jarvis_demo_tool"
    folder.mkdir()
    path = folder / "jarvis_demo_20260101T000000Z_abc123.txt"
    return VerifyDemoFileTool(folder, path, expected), path


def test_the_tool_verifies_a_file_that_holds_exactly_the_expected_text(tmp_path):
    tool, path = make_tool(tmp_path)
    path.write_bytes(NOTE.encode())
    result = tool.execute()
    assert result.status is ToolResultStatus.SUCCESS and result.data["verified"] is True


def test_a_byte_order_mark_does_not_matter_but_any_other_difference_does(tmp_path):
    tool, path = make_tool(tmp_path)
    path.write_bytes(b"\xef\xbb\xbf" + NOTE.encode())
    assert tool.execute().status is ToolResultStatus.SUCCESS
    for wrong in (NOTE + "\n", NOTE + "\r\n", NOTE + " ", " " + NOTE, NOTE.lower(), NOTE[:-1], NOTE + NOTE):
        path.write_bytes(wrong.encode())
        assert tool.execute().status is ToolResultStatus.ERROR, wrong


def test_a_missing_file_fails_verification(tmp_path):
    tool, path = make_tool(tmp_path)
    result = tool.execute()
    assert result.status is ToolResultStatus.ERROR and "does not exist" in result.message


def test_wrong_content_fails_with_a_short_excerpt_and_the_lengths(tmp_path):
    tool, path = make_tool(tmp_path)
    path.write_text("Xarvis demo note: written by Jarvis.")
    result = tool.execute()
    assert result.status is ToolResultStatus.ERROR and "does not hold the expected text" in result.message and "Xarvis demo note" in result.message


def test_an_empty_file_a_directory_an_oversized_file_and_binary_junk_all_fail(tmp_path):
    tool, path = make_tool(tmp_path)
    path.write_bytes(b"")
    assert tool.execute().status is ToolResultStatus.ERROR
    path.write_bytes(b"a" * 100_000)
    assert "bytes, far more" in tool.execute().message
    path.write_bytes(b"\xff\xfe\x00\x01")
    assert "not UTF-8" in tool.execute().message
    path.unlink()
    path.mkdir()
    assert tool.execute().status is ToolResultStatus.ERROR


def test_a_different_file_with_the_right_text_does_not_count(tmp_path):
    tool, path = make_tool(tmp_path)
    (path.parent / "jarvis_demo_20260101T000000Z_zzzzzz.txt").write_text(NOTE)  # same directory, same text, not THE file
    assert tool.execute().status is ToolResultStatus.ERROR


def test_the_tool_cannot_be_built_for_any_other_file_or_directory_or_text(tmp_path):
    folder = tmp_path / "jarvis_demo_tool"
    folder.mkdir()
    good = folder / "jarvis_demo_20260101T000000Z_abc123.txt"
    for bad_path in (tmp_path / "notes.txt", folder / "notes.txt", tmp_path / good.name, folder / "sub" / good.name):
        with pytest.raises(DemoPathError):
            VerifyDemoFileTool(folder, bad_path, NOTE)
    with pytest.raises(DemoPathError):
        VerifyDemoFileTool(folder, good, "two\nlines")


def test_the_tool_is_read_only_non_automatable_and_its_source_never_writes(tmp_path):
    tool, _ = make_tool(tmp_path)
    assert tool.permission.value == "read" and tool.automatable is False


def test_the_tool_re_checks_its_binding_when_it_runs(tmp_path, monkeypatch):
    tool, path = make_tool(tmp_path)
    path.write_text(NOTE)
    monkeypatch.setattr(file_check, "is_link", lambda p: Path(p) == path)  # the target turned into a link after the tool was built
    result = tool.execute()
    assert result.status is ToolResultStatus.ERROR and "cannot be read" in result.message


# ============================================================ wrong focus, failed save, content mismatch =============================================


def test_when_the_wrong_window_has_focus_nothing_is_typed_and_everything_after_is_skipped(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(process="editor.exe"))

    steps = steps_of(report)
    assert not report.succeeded and steps["verify_notepad"].status is StepStatus.FAILED and steps["verify_notepad"].attempts == 3
    assert "window_process_is" in steps["verify_notepad"].error
    assert desktop.log == [("open", "notepad")] and desktop.typed == []
    assert not setup.path.exists()
    assert steps["new_document"].status is StepStatus.SKIPPED and steps["type_note"].status is StepStatus.SKIPPED
    assert "Notepad was opened and is still open" in report.text and "Jarvis does not close it" in report.text


def test_when_notepad_never_comes_forward_the_wait_is_bounded_and_nothing_is_typed(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(appear_after=99))

    steps = steps_of(report)
    assert steps["front_notepad"].status is StepStatus.FAILED and steps["front_notepad"].attempts == 5
    assert desktop.log == [("open", "notepad")] and desktop.typed == []


def test_when_the_save_dialog_never_appears_the_file_name_is_never_typed(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(dialog=False))

    steps = steps_of(report)
    assert not report.succeeded
    assert steps["verify_save_dialog"].status is StepStatus.FAILED and steps["verify_save_dialog"].attempts == 5
    assert desktop.typed == [NOTE]  # the path did not go into the note
    assert desktop.document == NOTE
    assert desktop.log[-1] == ("key", "ctrl+s") and not setup.path.exists()
    assert steps["type_file_name"].status is StepStatus.SKIPPED and steps["confirm_save"].status is StepStatus.SKIPPED
    assert "may be unsaved" in report.text


def test_when_the_save_does_nothing_the_missing_file_fails_the_job_and_enter_is_not_pressed_again(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(save_works=False))

    steps = steps_of(report)
    assert not report.succeeded and report.job.status is JobStatus.FAILED
    assert steps["confirm_save"].status is StepStatus.SUCCEEDED and steps["confirm_save"].attempts == 1  # the keystroke "worked" ...
    assert steps["verify_file"].status is StepStatus.FAILED and steps["verify_file"].attempts == notepad_note.FILE_CHECK_ATTEMPTS  # ... the disk disagreed, after bounded waiting
    assert "does not exist" in steps["verify_file"].error
    assert desktop.log.count(("key", "enter")) == 1 and not setup.path.exists()
    assert "was not confirmed on disk" in report.text and "DID NOT SUCCEED" in report.text


def test_when_the_keystrokes_land_wrong_the_run_stops_before_the_save_and_no_file_exists(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(mistype=True))

    steps = steps_of(report)
    assert not setup.path.exists() and list(setup.directory.iterdir()) == []  # nothing was saved
    assert not report.succeeded and steps["verify_document_text"].status is StepStatus.FAILED and steps["verify_document_text"].attempts == 1
    assert steps["open_save_as"].status is StepStatus.SKIPPED and ("key", "ctrl+s") not in desktop.log
    assert desktop.log.count(("type", NOTE)) == 1  # the typing was not repeated to "fix" it


def test_the_disk_check_still_catches_a_save_that_wrote_something_else(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(save_text="Xarvis demo note: written by Jarvis."))  # the document looked right; the saved file is not

    steps = steps_of(report)
    assert setup.path.exists() and steps["verify_document_text"].status is StepStatus.SUCCEEDED
    assert not report.succeeded and steps["verify_file"].status is StepStatus.FAILED
    assert "does not hold the expected text" in steps["verify_file"].error and "Xarvis" in steps["verify_file"].error


def test_a_failed_run_never_asks_a_model_for_a_new_plan_and_never_cleans_up(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(save_works=False))

    assert "replanned" not in [event.kind for event in report.job.events]
    assert report.job.replan_count >= 10_000  # the pre-spent budget: the planner (which raises if called) was never reached
    assert desktop.opened  # nothing closed Notepad
    assert setup.directory.exists()


def test_the_report_names_the_failed_step_and_what_was_not_run(tmp_path):
    _, _, report, _ = run(tmp_path, FakeDesktop(dialog=False))

    assert re.search(r"Stopped at 'verify_save_dialog'\. Not run: .*type_file_name.*confirm_save.*verify_file", report.text)
    assert "Nothing was retried, closed or deleted by Jarvis" in report.text


# ============================================================ the demo cannot reach beyond itself ====================================================


def _calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names, imports = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return names, imports


def test_the_demo_code_never_deletes_renames_or_writes_files_and_never_touches_the_os_directly():
    forbidden_calls = {"remove", "unlink", "rmdir", "rmtree", "rename", "replace", "write_text", "write_bytes", "truncate", "symlink_to", "system", "popen", "Popen", "run"}
    forbidden_imports = {"subprocess", "shutil", "pyautogui", "win32api", "win32gui", "win32con", "ctypes", "webbrowser", "tempfile"}
    for module in sorted((APP_DIR / "demos").glob("*.py")):
        calls, imports = _calls(module)
        allowed_here = {"ctypes", "win32gui"} if module.name == "notepad_reader.py" else set()  # the read-only reader, pinned call by call in its own test below
        assert not (calls & forbidden_calls), (module.name, calls & forbidden_calls)
        assert not ((imports & forbidden_imports) - allowed_here), (module.name, (imports & forbidden_imports) - allowed_here)
    source = (APP_DIR / "demos" / "file_check.py").read_text(encoding="utf-8")
    opens = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open"]
    assert opens == []  # no builtin open(): the one file the tool opens is opened by os.open ...
    assert source.count("os.open(") == 1 and "os.O_RDONLY | getattr(os, \"O_BINARY\", 0)" in source  # ... exactly once, read-only
    assert (APP_DIR / "demos" / "notepad_note.py").read_text(encoding="utf-8").count(".mkdir(") == 1  # the one directory it may create


def test_the_demo_never_configures_builds_or_replaces_the_gate():
    for module in sorted((APP_DIR / "demos").glob("*.py")):
        source = module.read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        assert "computer_input_enabled" not in code and "computer_input_actions_per_minute" not in code, module.name
        assert "InputGate(" not in code and "create_default_input_gate" not in code and "GatedComputerBackend(" not in code, module.name
        assert "_events" not in code and ".admit(" not in code, module.name


# ====================================================================== the command line =================================================================


@pytest.fixture
def process_gate(monkeypatch):
    process = default_input_gate()
    monkeypatch.setattr(process, "_events", type(process._events)())
    monkeypatch.setattr(settings, "computer_input_enabled", True)
    monkeypatch.setattr(settings, "computer_input_actions_per_minute", 10)
    monkeypatch.setattr(notepad_note, "real_window_list", lambda: [])
    return process


def test_without_run_the_command_only_checks_and_changes_nothing(tmp_path, capsys, process_gate):
    folder = tmp_path / "jarvis_demo_cli"

    code = notepad_note.main(["--dir", str(folder)])

    assert code == 0 and not folder.exists()
    assert "Dry run only" in capsys.readouterr().out


def test_the_command_reports_a_refusal_with_exit_code_2_and_touches_nothing(tmp_path, capsys, process_gate, monkeypatch):
    monkeypatch.setattr(settings, "computer_input_enabled", False)
    folder = tmp_path / "jarvis_demo_cli"

    code = notepad_note.main(["--dir", str(folder), "--run"])

    assert code == 2 and not folder.exists()
    assert "Not started" in capsys.readouterr().out


def test_with_run_the_command_does_the_task_through_the_gated_backend(tmp_path, capsys, process_gate, monkeypatch):
    desktop = FakeDesktop()
    monkeypatch.setattr(notepad_note, "read_visible_editor_text", desktop.read_document)
    monkeypatch.setattr("app.computer.factory.create_default_computer_backend", lambda: GatedComputerBackend(desktop, process_gate_for(desktop)))
    folder = tmp_path / "jarvis_demo_cli"

    code = notepad_note.main(["--dir", str(folder), "--run", "--note", "Hello from the test."])

    out = capsys.readouterr().out
    assert code == 0 and "SUCCEEDED" in out
    (written,) = list(folder.iterdir())
    assert written.read_text(encoding="utf-8") == "Hello from the test."


def process_gate_for(_desktop):
    return gate()


def test_the_command_exits_1_when_the_task_fails_and_says_what_was_left_behind(tmp_path, capsys, process_gate, monkeypatch):
    desktop = FakeDesktop(dialog=False)
    monkeypatch.setattr(notepad_note, "read_visible_editor_text", desktop.read_document)
    monkeypatch.setattr("app.computer.factory.create_default_computer_backend", lambda: GatedComputerBackend(desktop, gate()))

    code = notepad_note.main(["--dir", str(tmp_path / "jarvis_demo_cli"), "--run"])

    assert code == 1
    assert "DID NOT SUCCEED" in capsys.readouterr().out


# ===================================== the run re-checks for itself; it does not trust what it was handed ============================================


def test_a_run_handed_an_unsafe_directory_refuses_before_the_desktop_is_touched(tmp_path):
    unsafe = tmp_path / "Documents"
    unsafe.mkdir()
    (unsafe / "taxes.xlsx").write_text("private")
    desktop = FakeDesktop()
    setup = DemoSetup(unsafe, unsafe / "jarvis_demo_20260101T000000Z_abc123.txt", NOTE)  # built by hand, skipping the preflight

    with pytest.raises(DemoRefused, match="must start with 'jarvis_demo'"):
        run_demo(setup, GatedComputerBackend(desktop, gate()), sleep=lambda seconds: None)
    assert desktop.log == [] and sorted(p.name for p in unsafe.iterdir()) == ["taxes.xlsx"]


def test_a_plan_that_would_repeat_an_input_action_is_refused_before_anything_runs(tmp_path, monkeypatch):
    setup = setup_for(tmp_path)
    plan = build_raw_plan(setup)
    for step in plan:
        if step["id"] == "type_note":
            step["max_attempts"] = 2
    monkeypatch.setattr(notepad_note, "build_raw_plan", lambda _setup: plan)
    desktop = FakeDesktop()

    with pytest.raises(DemoRefused, match="more than once"):
        run_demo(setup, GatedComputerBackend(desktop, gate()), sleep=lambda seconds: None)
    assert desktop.log == []


def test_a_completed_job_without_the_disk_check_having_succeeded_is_never_reported_as_success(tmp_path):
    from datetime import datetime, timezone

    from app.orchestration.base import AutonomyLevel, Job, JobPriority, Plan, PlanStep

    setup = setup_for(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    only_typing = Plan(steps=[PlanStep(id="type_note", tool_name="type_text", status=StepStatus.SUCCEEDED, attempts=1)])  # no disk check, no title check at all
    job = Job(id="j", goal="g", status=JobStatus.COMPLETED, priority=JobPriority.MEDIUM, autonomy=AutonomyLevel.ASSISTED, created_at=now, updated_at=now, plan=only_typing)

    report = notepad_note.build_report(job, setup)

    assert not report.succeeded and "DID NOT SUCCEED" in report.text


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are a Windows feature")
def test_is_link_sees_a_junction_which_the_standard_symlink_test_does_not(tmp_path):
    import subprocess

    target = tmp_path / "somewhere"
    target.mkdir()
    junction = tmp_path / "junction"
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip("could not create a junction here")
    assert file_check.is_link(junction) is True and file_check.is_link(target) is False and file_check.is_link(tmp_path / "missing") is False



# ===================================== the command line: --run must be spelled out, never inferred from an abbreviation =============================


@pytest.fixture
def never_runs(monkeypatch, tmp_path):
    """Anything that would start the real thing fails the test."""

    def boom(*args, **kwargs):
        raise AssertionError("the desktop workflow was started")

    monkeypatch.setattr(notepad_note, "run_demo", boom)
    monkeypatch.setattr("app.computer.factory.create_default_computer_backend", boom)
    folder = tmp_path / "jarvis_demo_x"
    monkeypatch.setattr(notepad_note, "check_ready", lambda *a, **k: DemoSetup(folder, folder / "jarvis_demo_20260101T000000Z_abc123.txt", NOTE))
    return folder


@pytest.mark.parametrize("abbreviation", ["--r", "--ru", "--run=1", "--runn"])
def test_an_abbreviated_or_malformed_run_flag_is_rejected_and_nothing_starts(never_runs, abbreviation):
    with pytest.raises(SystemExit) as stopped:
        notepad_note.main(["--dir", str(never_runs), abbreviation])
    assert stopped.value.code == 2
    assert not never_runs.exists()


@pytest.mark.parametrize(
    "argv",
    [["--d", "X"], ["--dir", "X", "--no", "n"], ["--dir", "X", "--sav", "t"]],
    ids=["--d", "--no", "--sav"],
)
def test_no_other_option_can_be_abbreviated_either(never_runs, argv):
    with pytest.raises(SystemExit) as stopped:
        notepad_note.main([str(never_runs) if part == "X" else part for part in argv])
    assert stopped.value.code == 2


def test_the_parser_is_built_with_abbreviations_off_and_a_bare_dir_is_still_only_a_dry_run(never_runs, capsys):
    assert "allow_abbrev=False" in (APP_DIR / "demos" / "notepad_note.py").read_text(encoding="utf-8")
    assert notepad_note.main(["--dir", str(never_runs)]) == 0  # never_runs would have failed this had it started anything
    assert "Dry run only" in capsys.readouterr().out
    assert not never_runs.exists()


# =============================================== focus is re-read immediately before every keystroke; if it is wrong, nothing is sent ==================

# The reads of the foreground window, in order: 1 verify_notepad, 2 before the note, 3 the document check before ctrl+s, 4 the Save As title, 5 before the file name,
# 6 the Save As title again before Enter, 7 the process right before Enter, 8 the title after Enter.
WRONG_APP = WindowInfo("Some other app", "other.exe", False)
FOCUS_CASES = [
    (1, "verify_notepad", [("open", "notepad")]),
    (2, "verify_focus_for_note", [("open", "notepad"), ("key", "ctrl+n")]),
    (3, "verify_document_text", [("open", "notepad"), ("key", "ctrl+n"), ("type", NOTE)]),
    (4, "verify_save_dialog", [("open", "notepad"), ("key", "ctrl+n"), ("type", NOTE), ("key", "ctrl+s")]),
    (5, "verify_focus_for_name", [("open", "notepad"), ("key", "ctrl+n"), ("type", NOTE), ("key", "ctrl+s")]),
]


@pytest.mark.parametrize("from_read, failing_step, expected_log", FOCUS_CASES)
def test_when_the_wrong_window_is_in_front_at_any_check_the_next_keystroke_is_never_sent(tmp_path, from_read, failing_step, expected_log):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(override_from=(from_read, WRONG_APP)))

    steps = steps_of(report)
    assert not report.succeeded and steps[failing_step].status is StepStatus.FAILED
    assert desktop.log == expected_log  # not one keystroke more than the checks allowed
    assert desktop.misdirected == [] and not setup.path.exists()


def test_the_window_in_front_right_before_enter_is_not_the_save_dialog_so_enter_is_not_pressed(tmp_path):
    still_notepad_but_not_the_dialog = WindowInfo("Untitled - Notepad", "notepad.exe", False)

    setup, desktop, report, _ = run(tmp_path, FakeDesktop(override_from=(6, still_notepad_but_not_the_dialog)))

    steps = steps_of(report)
    assert steps["verify_dialog_before_save"].status is StepStatus.FAILED and steps["verify_dialog_before_save"].attempts == 2
    assert steps["confirm_save"].status is StepStatus.SKIPPED
    assert ("key", "enter") not in desktop.log and desktop.log[-1] == ("type", str(setup.path)) and not setup.path.exists()


def test_a_process_that_cannot_be_identified_is_not_notepad_and_nothing_is_sent(tmp_path):
    unidentified = WindowInfo("Save As", "unknown", False)  # what the backend reports when it cannot resolve the owning process

    setup, desktop, report, _ = run(tmp_path, FakeDesktop(override_from=(7, unidentified)))

    steps = steps_of(report)
    assert steps["verify_focus_before_save"].status is StepStatus.FAILED and steps["confirm_save"].status is StepStatus.SKIPPED
    assert ("key", "enter") not in desktop.log and not setup.path.exists()


# ================================ the last-moment re-check before saving, and what happens if something appears anyway ================================


def test_something_created_in_the_directory_after_the_file_name_was_typed_stops_the_save_before_enter(tmp_path):
    def intruder(typed_path):
        (typed_path.parent / "other.txt").write_text("not ours")

    setup, desktop, report, _ = run(tmp_path, FakeDesktop(after_name_typed=intruder))

    steps = steps_of(report)
    assert steps["verify_target_free"].status is StepStatus.FAILED and steps["verify_target_free"].attempts == 1
    assert "no longer empty" in steps["verify_target_free"].error
    assert steps["confirm_save"].status is StepStatus.SKIPPED and ("key", "enter") not in desktop.log
    assert (setup.directory / "other.txt").read_text() == "not ours" and not setup.path.exists()


def test_the_target_itself_appearing_after_the_file_name_was_typed_stops_the_save_before_enter(tmp_path):
    desktop = FakeDesktop(after_name_typed=lambda typed_path: typed_path.write_text("someone else's file"))

    setup, desktop, report, _ = run(tmp_path, desktop)

    assert steps_of(report)["verify_target_free"].status is StepStatus.FAILED
    assert ("key", "enter") not in desktop.log and setup.path.read_text() == "someone else's file"


def test_the_directory_swapped_for_a_link_after_validation_stops_the_save_before_enter(tmp_path, monkeypatch):
    swap = lambda typed_path: monkeypatch.setattr(file_check.os.path, "realpath", lambda p: str(tmp_path / "elsewhere"))  # noqa: E731

    setup, desktop, report, _ = run(tmp_path, FakeDesktop(after_name_typed=swap))

    steps = steps_of(report)
    assert steps["verify_target_free"].status is StepStatus.FAILED and "link or junction" in steps["verify_target_free"].error
    assert ("key", "enter") not in desktop.log


@pytest.mark.parametrize("foreign_text", ["someone else's file", NOTE], ids=["different content", "content identical to the note"])
def test_a_file_created_at_the_last_instant_raises_an_overwrite_prompt_that_is_refused_and_the_run_fails(tmp_path, foreign_text):
    """The residual race: the file appears AFTER the last check, at the instant of the one Enter. The prompt must be left unanswered and the run must fail -
    even when the stranger's file happens to hold exactly the note, which the disk check alone would accept."""
    desktop = FakeDesktop(before_enter=lambda typed_path: typed_path.write_text(foreign_text))

    setup, desktop, report, _ = run(tmp_path, desktop)

    steps = steps_of(report)
    assert not report.succeeded and report.job.status is JobStatus.FAILED
    assert steps["confirm_save"].status is StepStatus.SUCCEEDED and steps["confirm_save"].attempts == 1
    if foreign_text == NOTE:
        assert steps["verify_file"].status is StepStatus.SUCCEEDED  # the disk check alone WOULD have said yes ...
        assert steps["verify_saved_title"].status is StepStatus.FAILED and "Confirm Save As" in steps["verify_saved_title"].error  # ... Notepad's title says a prompt is in front
    else:
        assert steps["verify_file"].status is StepStatus.FAILED and steps["verify_saved_title"].status is StepStatus.SKIPPED
    assert desktop.prompt_open and desktop.log[-1] == ("key", "enter") and desktop.log.count(("key", "enter")) == 1  # nothing answered it
    assert setup.path.read_text() == foreign_text  # the stranger's file is untouched
    if foreign_text == NOTE:
        assert "pressed nothing to answer it" in report.text


def test_no_input_step_follows_the_one_enter_and_it_is_the_only_enter(tmp_path):
    plan = build_raw_plan(setup_for(tmp_path))
    ids = [step["id"] for step in plan]
    after = plan[ids.index("confirm_save") + 1 :]
    assert [step["id"] for step in after] == ["verify_file", "verify_saved_title"]
    assert all(step["tool"] in ("verify_demo_file", "verify_observation") for step in after)  # only reads: nothing can answer a prompt
    assert len([step for step in plan if step["tool"] == "press_key" and step["arguments"]["key"] == "enter"]) == 1


def test_success_needs_both_the_disk_check_and_the_title_check_and_is_still_six_gated_actions(tmp_path):
    setup, desktop, report, the_gate = run(tmp_path)

    steps = steps_of(report)
    assert report.succeeded and steps["verify_saved_title"].status is StepStatus.SUCCEEDED and steps["verify_file"].status is StepStatus.SUCCEEDED
    assert the_gate.remaining() == 10 - INPUT_ACTIONS_NEEDED  # the extra checks are reads


def test_the_residual_race_and_the_refusal_to_answer_a_prompt_are_documented_in_the_module():
    doc = " ".join(notepad_note.__doc__.split())
    assert "RESIDUAL RACE" in doc and "cannot be removed" in doc
    assert "never presses a key to answer" in doc and "never relies on which button" in doc


# ============================================== the new tool, on its own =====================================================================


def make_free_tool(tmp_path):
    from app.demos.file_check import CheckDemoTargetFreeTool

    folder = tmp_path / "jarvis_demo_free"
    folder.mkdir()
    path = folder / "jarvis_demo_20260101T000000Z_abc123.txt"
    return CheckDemoTargetFreeTool(folder, path), folder, path


def test_the_target_free_tool_passes_only_for_an_empty_real_directory_without_the_target(tmp_path):
    tool, folder, path = make_free_tool(tmp_path)
    assert tool.parameters == [] and tool.permission.value == "read" and tool.automatable is False
    assert tool.execute().status is ToolResultStatus.SUCCESS
    assert list(folder.iterdir()) == []  # it created nothing

    (folder / "x.txt").write_text("1")
    assert "no longer empty" in tool.execute().message
    (folder / "x.txt").unlink()
    path.write_text("1")
    assert "not free" in tool.execute().message and path.read_text() == "1"


def test_the_target_free_tool_refuses_a_missing_or_replaced_directory_and_a_link(tmp_path, monkeypatch):
    tool, folder, path = make_free_tool(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(file_check.os.path, "realpath", lambda p: str(tmp_path / "elsewhere"))
        assert "link or junction" in tool.execute().message
    folder.rmdir()
    assert tool.execute().status is ToolResultStatus.ERROR
    folder.write_text("now a file")
    assert "no longer a plain directory" in tool.execute().message


def test_the_target_free_tool_cannot_be_built_for_any_other_file_and_a_plan_cannot_give_it_a_path(tmp_path):
    from app.demos.file_check import CheckDemoTargetFreeTool

    folder = tmp_path / "jarvis_demo_free"
    folder.mkdir()
    with pytest.raises(DemoPathError):
        CheckDemoTargetFreeTool(folder, folder / "notes.txt")
    registry = build_registry(FakeDesktop(), setup_for(tmp_path, name="jarvis_demo_registry"))
    raw = [{"id": "s1", "tool": "verify_demo_target_free", "arguments": {"path": "x"}, "depends_on": []}]
    with pytest.raises(PlanValidationError, match="unexpected argument"):
        validate_plan(raw, registry, max_steps=20, default_retry=RetryPolicy())


def test_the_file_tool_now_also_refuses_a_directory_that_was_swapped(tmp_path, monkeypatch):
    tool, path = make_tool(tmp_path)
    path.write_text(NOTE)
    assert tool.execute().status is ToolResultStatus.SUCCESS
    monkeypatch.setattr(file_check.os.path, "realpath", lambda p: str(tmp_path / "elsewhere"))
    result = tool.execute()
    assert result.status is ToolResultStatus.ERROR and "not trusted" in result.message


def test_a_save_that_leaves_notepad_not_showing_the_new_file_name_fails_even_though_the_file_is_on_disk(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(title_after_save="Untitled - Notepad"))

    steps = steps_of(report)
    assert setup.path.read_text(encoding="utf-8") == NOTE and steps["verify_file"].status is StepStatus.SUCCEEDED  # the disk check alone would pass ...
    assert steps["verify_saved_title"].status is StepStatus.FAILED and steps["verify_saved_title"].attempts == 3
    assert not report.succeeded and report.job.status is JobStatus.FAILED  # ... but the run does not claim success


@pytest.mark.parametrize(
    "statuses, expected",
    [
        ({"verify_file": StepStatus.SUCCEEDED, "verify_saved_title": StepStatus.SUCCEEDED}, True),
        ({"verify_file": StepStatus.SUCCEEDED}, False),  # no title check ran at all
        ({"verify_saved_title": StepStatus.SUCCEEDED}, False),  # no disk check ran at all
        ({"verify_file": StepStatus.SUCCEEDED, "verify_saved_title": StepStatus.FAILED}, False),
        ({"verify_file": StepStatus.FAILED, "verify_saved_title": StepStatus.SUCCEEDED}, False),
    ],
)
def test_the_report_calls_a_run_a_success_only_when_a_completed_job_passed_both_the_disk_check_and_the_title_check(tmp_path, statuses, expected):
    from datetime import datetime, timezone

    from app.orchestration.base import AutonomyLevel, Job, JobPriority, Plan, PlanStep

    setup = setup_for(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    plan = Plan(steps=[PlanStep(id=step_id, tool_name="verify_observation", status=status, attempts=1, result={"message": "ok"}) for step_id, status in statuses.items()])
    job = Job(id="j", goal="g", status=JobStatus.COMPLETED, priority=JobPriority.MEDIUM, autonomy=AutonomyLevel.ASSISTED, created_at=now, updated_at=now, plan=plan)

    assert notepad_note.build_report(job, setup).succeeded is expected


# ============================== after the first live run: settling pauses, a longer disk wait, a plainer failure report ===============================
# The first live run lost 15 of the note's 36 characters and its Enter did not save. The fixes are a paced `type_text` (tested in
# test_computer_backend_windows.py), three bounded settling pauses, a longer wait for the file, and a report that does not call a typed path a saved file.

SETTLE_STEPS = ("settle_after_new_tab", "settle_after_dialog", "settle_before_save")


def run_recording_sleeps(tmp_path, desktop=None, *, limit=10):
    """Like `run`, but every sleep (the engine's retry delays AND the settling pauses) is recorded - in order, in the desktop's own log."""
    desktop = desktop or FakeDesktop()
    the_gate = gate(limit)
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        desktop.log.append(("sleep", seconds))

    setup = check_ready(tmp_path / "jarvis_demo_run", NOTE, gate=gate(), list_windows=lambda: [], token="abc123")
    report = run_demo(setup, GatedComputerBackend(desktop, the_gate), sleep=sleep, read_document=desktop.read_document)
    return setup, desktop, report, the_gate, sleeps


def test_the_settling_pause_is_fixed_bounded_read_only_and_takes_no_arguments():
    waits = []
    tool = notepad_note.SettleTool(sleep=waits.append)

    result = tool.execute()

    assert result.status is ToolResultStatus.SUCCESS and waits == [notepad_note.SETTLE_SECONDS] == [1.5]
    assert tool.parameters == [] and tool.permission.value == "read" and tool.automatable is False
    with pytest.raises(TypeError):
        tool.execute(seconds=60)  # a plan cannot choose how long to wait


@pytest.mark.parametrize("bad", [0, -1, -0.5, notepad_note.MAX_SETTLE_SECONDS + 0.1, 3600, float("inf"), float("nan"), True, False, "1", None])
def test_a_settling_pause_outside_zero_to_the_ceiling_cannot_be_built(bad):
    with pytest.raises(ValueError):
        notepad_note.SettleTool(bad, sleep=lambda seconds: None)


def test_a_pause_at_the_ceiling_is_allowed_and_the_ceiling_is_short():
    waits = []
    notepad_note.SettleTool(notepad_note.MAX_SETTLE_SECONDS, sleep=waits.append).execute()

    assert waits == [notepad_note.MAX_SETTLE_SECONDS] and notepad_note.MAX_SETTLE_SECONDS <= 5.0
    assert notepad_note.SETTLE_SECONDS <= notepad_note.MAX_SETTLE_SECONDS


def test_the_three_pauses_sit_exactly_where_the_first_live_run_needed_them(tmp_path):
    ids = [s["id"] for s in build_raw_plan(setup_for(tmp_path))]

    assert [i for i in ids if i.startswith("settle_")] == list(SETTLE_STEPS)
    assert ids.index("settle_after_new_tab") == ids.index("new_document") + 1 and ids.index("settle_after_new_tab") < ids.index("verify_focus_for_note") < ids.index("type_note")
    assert ids.index("verify_save_dialog") < ids.index("settle_after_dialog") < ids.index("verify_focus_for_name") < ids.index("type_file_name")
    assert ids.index("settle_before_save") == ids.index("type_file_name") + 1 and ids.index("settle_before_save") < ids.index("verify_target_free") < ids.index("confirm_save")


def test_the_pauses_are_single_attempt_argument_free_reads_and_the_plan_still_fits_its_limits(tmp_path):
    setup = setup_for(tmp_path)
    raw = build_raw_plan(setup)
    steps = validate_plan(raw, build_registry(FakeDesktop(), setup), max_steps=notepad_note.MAX_STEPS, default_retry=RetryPolicy())

    for step in steps:
        if step.id in SETTLE_STEPS:
            assert step.tool_name == "demo_settle" and step.arguments == {} and step.retry_policy.max_attempts == 1 and step.permission == "read"
        if step.tool_name in notepad_note.INPUT_TOOLS:
            assert step.retry_policy.max_attempts == 1, step.id  # still ONE attempt per input action
    assert len(raw) <= notepad_note.MAX_STEPS
    assert sum(1 for s in steps if s.tool_name in notepad_note.INPUT_TOOLS) == INPUT_ACTIONS_NEEDED == 6  # the pauses added no input action
    assert "demo_settle" not in notepad_note.INPUT_TOOLS


def test_during_a_run_each_pause_happens_between_the_right_two_actions_and_nothing_really_waits(tmp_path):
    setup, desktop, report, the_gate, sleeps = run_recording_sleeps(tmp_path)

    assert report.succeeded
    assert desktop.log == [
        ("open", "notepad"),
        ("key", "ctrl+n"),
        ("sleep", 1.5),
        ("type", NOTE),
        ("sleep", 1.5),  # the document check waits once, then reads once
        ("key", "ctrl+s"),
        ("sleep", 1.5),
        ("type", str(setup.path)),
        ("sleep", 1.5),
        ("key", "enter"),
    ]
    assert sleeps == [1.5, 1.5, 1.5, 1.5]  # injected: the happy path has no retry delay, so these are the only waits
    assert the_gate.remaining() == 10 - INPUT_ACTIONS_NEEDED


def test_the_pauses_use_no_gate_slot_so_a_limit_of_exactly_six_still_finishes(tmp_path):
    _, _, report, the_gate, _ = run_recording_sleeps(tmp_path, limit=INPUT_ACTIONS_NEEDED)

    assert report.succeeded and the_gate.remaining() == 0


def test_a_failure_after_the_pauses_never_repeats_an_input_action(tmp_path):
    setup, desktop, report, _, _ = run_recording_sleeps(tmp_path, FakeDesktop(save_works=False))

    assert not report.succeeded
    assert [k for k in desktop.log if k[0] in ("key", "type", "open")] == [("open", "notepad"), ("key", "ctrl+n"), ("type", NOTE), ("key", "ctrl+s"), ("type", str(setup.path)), ("key", "enter")]


def test_the_wait_for_the_file_is_longer_than_before_but_within_the_validators_limits(tmp_path):
    steps = {s["id"]: s for s in build_raw_plan(setup_for(tmp_path))}
    verify_file = steps["verify_file"]
    wait = (verify_file["max_attempts"] - 1) * verify_file["retry_delay_seconds"]

    assert verify_file["max_attempts"] == notepad_note.FILE_CHECK_ATTEMPTS <= 5  # the shared validator's own ceiling (validator._MAX_ALLOWED_ATTEMPTS)
    assert 10.0 <= wait <= 30.0  # it was 3 x 1.5 = 4.5 s
    setup, desktop, report, _, sleeps = run_recording_sleeps(tmp_path, FakeDesktop(save_works=False))
    assert sleeps.count(notepad_note.FILE_CHECK_DELAY_SECONDS) == notepad_note.FILE_CHECK_ATTEMPTS - 1
    assert steps_of(report)["verify_file"].attempts == notepad_note.FILE_CHECK_ATTEMPTS and desktop.log.count(("key", "enter")) == 1


def test_the_pause_tool_exists_in_no_shared_registry_and_no_other_module(tmp_path):
    from app.computer.defaults import create_computer_tools_registry

    assert "demo_settle" not in JOB_COMPUTER_TOOL_NAMES
    assert create_computer_tools_registry(backend=FakeDesktop()).get("demo_settle") is None
    users = [p.relative_to(APP_DIR).as_posix() for p in APP_DIR.rglob("*.py") if "demo_settle" in p.read_text(encoding="utf-8") or "SettleTool" in p.read_text(encoding="utf-8")]
    assert users == ["demos/notepad_note.py"]
    assert build_registry(FakeDesktop(), setup_for(tmp_path)).get("demo_settle") is not None  # only the demo's own private registry has it


def _failed_job(statuses):
    from datetime import datetime, timezone

    from app.orchestration.base import AutonomyLevel, Job, JobPriority, Plan, PlanStep

    now = datetime.now(timezone.utc).isoformat()
    plan = Plan(steps=[PlanStep(id=step_id, tool_name="x", status=status, attempts=1, result={"message": "m"}, error="boom" if status is StepStatus.FAILED else None) for step_id, status in statuses.items()])
    return Job(id="j", goal="g", status=JobStatus.FAILED, priority=JobPriority.MEDIUM, autonomy=AutonomyLevel.ASSISTED, created_at=now, updated_at=now, plan=plan)


OK, FAILED, SKIPPED = StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.SKIPPED


def test_a_typed_path_is_not_reported_as_a_save_and_the_open_dialog_with_unsaved_text_is_warned_about(tmp_path):
    job = _failed_job({"open_notepad": OK, "type_note": OK, "open_save_as": OK, "type_file_name": OK, "verify_target_free": FAILED, "confirm_save": SKIPPED})

    text = notepad_note.build_report(job, setup_for(tmp_path)).text

    assert "the Save As dialog may still be open with the note unsaved and the file path typed into it" in text
    assert "Enter was NOT pressed, so nothing was saved" in text
    assert "A typed path or a pressed Enter is not a saved file - only the check of the disk counts." in text


def test_when_no_path_was_typed_the_report_still_warns_the_dialog_may_be_open_with_unsaved_text(tmp_path):
    job = _failed_job({"open_notepad": OK, "type_note": OK, "open_save_as": OK, "verify_save_dialog": FAILED, "type_file_name": SKIPPED})

    text = notepad_note.build_report(job, setup_for(tmp_path)).text

    assert "the Save As dialog may still be open with the note unsaved" in text and "file path typed into it" not in text


def test_a_pressed_enter_without_a_confirmed_file_is_not_called_a_save_and_warns_about_finishing_it_by_hand(tmp_path):
    job = _failed_job({"open_notepad": OK, "type_note": OK, "open_save_as": OK, "type_file_name": OK, "confirm_save": OK, "verify_file": FAILED})

    text = notepad_note.build_report(job, setup_for(tmp_path)).text

    assert "was not confirmed on disk" in text and "a pressed Enter is not a saved file" in text
    assert "may still be open with the text unsaved" in text and "which may differ from the note" in text


def test_the_real_failed_run_report_carries_the_new_warnings(tmp_path):
    _, _, report, _ = run(tmp_path, FakeDesktop(save_works=False))

    assert not report.succeeded
    assert "a pressed Enter is not a saved file" in report.text and "only the check of the disk counts" in report.text


def test_a_successful_report_has_none_of_the_failure_warnings(tmp_path):
    _, _, report, _ = run(tmp_path)

    assert report.succeeded and "not a saved file" not in report.text and "Left on the screen" not in report.text



# ===================== the note is read back from Notepad's visible editor BEFORE anything is saved (after the second live run lost a "d") ===================


class FakeWindowApi:
    """Windows, as the reader asks about them: `windows` maps a handle to {"class", "visible", "text", "process", "children", "hang", "hang_text"}."""

    def __init__(self, windows, foreground=1, fail_in=None):
        self.windows, self.foreground, self.fail_in, self.calls = windows, foreground, fail_in, []

    def _maybe_fail(self, where):
        if self.fail_in == where:
            raise RuntimeError("SECRET-WINDOW-CONTENT")

    def foreground_window(self):
        self._maybe_fail("foreground_window")
        return self.foreground

    def process_name(self, hwnd):
        return self.windows[hwnd].get("process", "notepad.exe")

    def child_windows(self, hwnd):
        self._maybe_fail("child_windows")
        return list(self.windows[hwnd].get("children", []))

    def class_name(self, hwnd):
        return self.windows[hwnd]["class"]

    def is_visible(self, hwnd):
        return self.windows[hwnd].get("visible", True)

    def text_length(self, hwnd, timeout_ms):
        self.calls.append(("length", hwnd, timeout_ms))
        window = self.windows[hwnd]
        return None if window.get("hang") else len(window["text"])

    def text(self, hwnd, length, timeout_ms):
        self.calls.append(("text", hwnd, timeout_ms))
        window = self.windows[hwnd]
        return None if window.get("hang_text") else window["text"][:length]


def notepad_windows(visible_text, *, hidden_text="Jarvis emo note: written by Jarvis.", hidden_first=True):
    """The window tree the probe found: the top-level window, a HIDDEN tab's editor holding old text, and the VISIBLE editor holding the document."""
    hidden = {"class": "RichEditD2DPT", "visible": False, "text": hidden_text}
    shown = {"class": "RichEditD2DPT", "visible": True, "text": visible_text}
    order = [10, 11] if hidden_first else [11, 10]
    return {
        1: {"class": "Notepad", "text": "Untitled - Notepad", "children": order + [12]},
        10: hidden,
        11: shown,
        12: {"class": "InputNonClientPointerSource", "visible": True, "text": "tab strip"},
    }


def test_the_reader_returns_the_visible_editors_text_and_every_read_is_bounded_to_500_ms():
    api = FakeWindowApi(notepad_windows(NOTE))

    assert notepad_reader.read_visible_editor_text(api) == NOTE
    assert notepad_reader.READ_TIMEOUT_MS == 500
    assert api.calls == [("length", 11, 500), ("text", 11, 500)]  # the visible editor only, two bounded reads


@pytest.mark.parametrize("hidden_first", [True, False])
def test_a_hidden_tabs_stale_text_is_never_read_whichever_editor_comes_first(hidden_first):
    api = FakeWindowApi(notepad_windows(NOTE, hidden_text="stale text from an old tab", hidden_first=hidden_first))

    assert notepad_reader.read_visible_editor_text(api) == NOTE
    assert all(call[1] == 11 for call in api.calls)  # the hidden editor (10) was never even asked


def test_when_the_visible_editor_is_the_wrong_text_the_hidden_one_does_not_rescue_it():
    api = FakeWindowApi(notepad_windows("Jarvis emo note: written by Jarvis.", hidden_text=NOTE))  # the hidden tab holds exactly the note; the document does not

    assert notepad_reader.read_visible_editor_text(api) != NOTE


def test_two_visible_editors_are_refused_not_guessed_between():
    windows = notepad_windows(NOTE)
    windows[10]["visible"] = True

    with pytest.raises(DocumentReadError, match="2 visible editor controls"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))


def test_no_editor_or_only_hidden_editors_is_a_failure():
    windows = notepad_windows(NOTE)
    windows[11]["visible"] = False
    with pytest.raises(DocumentReadError, match="no visible Notepad editor"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))

    windows = notepad_windows(NOTE)
    windows[1]["children"] = [12]
    with pytest.raises(DocumentReadError, match="no visible Notepad editor"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))


def test_an_editor_of_the_wrong_class_is_not_taken_for_the_document():
    windows = notepad_windows(NOTE)
    windows[11]["class"] = "SomethingElse"
    windows[10]["visible"] = True  # the other editor is visible but not a RichEditD2DPT either: no control of the right class is left
    windows[10]["class"] = "NotAnEditor"

    with pytest.raises(DocumentReadError, match="no visible Notepad editor"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))


def test_the_window_in_front_must_be_notepad_exe_and_must_exist():
    windows = notepad_windows(NOTE)
    windows[1]["process"] = "Code.exe"
    with pytest.raises(DocumentReadError, match="does not belong to notepad.exe"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))
    windows[1]["process"] = "unknown"  # what the backend reports when it cannot resolve the owner
    with pytest.raises(DocumentReadError, match="does not belong to notepad.exe"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))
    with pytest.raises(DocumentReadError, match="no window is in front"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(notepad_windows(NOTE), foreground=0))


def test_a_notepad_editor_that_is_not_under_the_window_in_front_is_not_read():
    windows = notepad_windows(NOTE)
    windows[2] = {"class": "Notepad", "text": "Other - Notepad", "children": [20]}  # a second Notepad window
    windows[20] = {"class": "RichEditD2DPT", "visible": True, "text": "the other window's document"}

    api = FakeWindowApi(windows, foreground=1)

    assert notepad_reader.read_visible_editor_text(api) == NOTE and all(call[1] == 11 for call in api.calls)


@pytest.mark.parametrize("hang_field", ["hang", "hang_text"])
def test_an_editor_that_does_not_answer_in_time_fails_the_check_instead_of_hanging(hang_field):
    windows = notepad_windows(NOTE)
    windows[11][hang_field] = True

    with pytest.raises(DocumentReadError, match="500 ms"):
        notepad_reader.read_visible_editor_text(FakeWindowApi(windows))


def test_a_far_too_large_document_is_refused_without_reading_it():
    api = FakeWindowApi(notepad_windows("x" * (notepad_reader.MAX_DOCUMENT_CHARS + 1)))

    with pytest.raises(DocumentReadError, match="far more than the note"):
        notepad_reader.read_visible_editor_text(api)
    assert [call[0] for call in api.calls] == ["length"]


@pytest.mark.parametrize("where", ["foreground_window", "child_windows"])
def test_an_api_error_is_a_failed_check_and_its_message_never_reaches_the_report(where):
    with pytest.raises(DocumentReadError) as caught:
        notepad_reader.read_visible_editor_text(FakeWindowApi(notepad_windows(NOTE), fail_in=where))

    assert "RuntimeError" in str(caught.value) and "SECRET" not in str(caught.value)


def test_the_reader_module_can_only_make_the_read_only_calls_it_is_pinned_to():
    source = (APP_DIR / "demos" / "notepad_reader.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "ctypes", "typing", "win32gui", "app"}, imports
    gui_calls = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "_gui"}
    assert gui_calls == {"GetForegroundWindow", "EnumChildWindows", "GetClassName", "IsWindowVisible"}  # nothing that acts on a window
    messages = {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("WM_")}
    assert messages == {"WM_GETTEXT", "WM_GETTEXTLENGTH"}  # the only two messages that exist in the file, both reads
    assert source.count("SendMessageTimeoutW") == 1 and source.count("SMTO_ABORTIFHUNG, timeout_ms") == 1  # every message is bounded and abortable
    for forbidden in ("SendInput", "keybd_event", "mouse_event", "PostMessage", "PostThreadMessage", "SetForegroundWindow", "BringWindowToTop", "ShowWindow",
                      "SetCursorPos", "SetWindowText", "pyautogui", "subprocess", "AttachThreadInput", "open("):
        assert forbidden not in source, forbidden
    assert "logger" not in source and "print(" not in source  # it logs nothing: document text can only go to its caller


# --- the verification step itself --------------------------------------------------------------------------------------------------------------------


def document_tool(reader, **kwargs):
    return notepad_note.VerifyDocumentTextTool(NOTE, reader, sleep=kwargs.pop("sleep", lambda seconds: None), **kwargs)


def test_an_exact_match_passes_and_reports_only_the_length():
    result = document_tool(lambda: NOTE).execute()

    assert result.status is ToolResultStatus.SUCCESS and "36 characters" in result.message and NOTE not in result.message


@pytest.mark.parametrize(
    "actual",
    [
        "Jarvis emo note: written by Jarvis.",  # the second live run's missing "d"
        "Jarvis mo note: n  ja",  # the first live run's
        "",  # nothing was typed
        "Jarvis demo note: written by Jarvis.\n",  # an extra character
        "jarvis demo note: written by Jarvis.",  # wrong case
        NOTE + NOTE,  # stale text from before (a lost Ctrl+N) plus the note
        "an old document that was already in Notepad",  # stale
    ],
    ids=["missing d", "first run", "empty", "trailing newline", "case", "doubled", "stale"],
)
def test_any_difference_fails_and_the_message_has_lengths_but_never_the_document(actual):
    result = document_tool(lambda: actual).execute()

    assert result.status is ToolResultStatus.ERROR
    assert f"expected {len(NOTE)} characters, found {len(actual)}" in result.message
    assert "nothing was saved" in result.message and "Save As was not opened" in result.message
    assert actual not in result.message or actual == ""  # (the empty string is "in" everything)


def test_a_failure_to_read_is_a_failure_with_the_reason_and_without_any_exception_text():
    def hung():
        raise DocumentReadError("the editor did not answer within 500 ms (it may be busy or hung)")

    def broken():
        raise OSError("SECRET-WINDOW-CONTENT")

    assert "did not answer within 500 ms" in document_tool(hung).execute().message
    broken_result = document_tool(broken).execute()
    assert broken_result.status is ToolResultStatus.ERROR and "OSError" in broken_result.message and "SECRET" not in broken_result.message
    assert document_tool(lambda: None).execute().status is ToolResultStatus.ERROR  # not text at all


def test_the_step_waits_once_then_reads_exactly_once_and_never_again_even_on_failure():
    events = []

    def reader():
        events.append("read")
        return "wrong"

    tool = document_tool(reader, sleep=lambda seconds: events.append(("sleep", seconds)))
    result = tool.execute()

    assert result.status is ToolResultStatus.ERROR and events == [("sleep", notepad_note.SETTLE_SECONDS), "read"]


def test_the_document_tool_is_read_only_argument_free_not_automatable_and_its_wait_is_bounded():
    tool = document_tool(lambda: NOTE)

    assert tool.parameters == [] and tool.permission.value == "read" and tool.automatable is False
    with pytest.raises(TypeError):
        tool.execute(text="anything")
    for bad in (0, -1, notepad_note.MAX_SETTLE_SECONDS + 1, True, None, float("nan")):
        with pytest.raises(ValueError):
            document_tool(lambda: NOTE, settle_seconds=bad)
    with pytest.raises(DemoPathError):
        notepad_note.VerifyDocumentTextTool("two\nlines", lambda: NOTE)  # the expected note is validated like everywhere else


def test_the_document_tool_exists_only_in_the_demos_private_registry(tmp_path):
    from app.computer.defaults import create_computer_tools_registry

    assert "verify_document_text" not in JOB_COMPUTER_TOOL_NAMES
    assert create_computer_tools_registry(backend=FakeDesktop()).get("verify_document_text") is None
    users = [p.relative_to(APP_DIR).as_posix() for p in APP_DIR.rglob("*.py") if "verify_document_text" in p.read_text(encoding="utf-8")]
    assert users == ["demos/notepad_note.py"]
    assert build_registry(FakeDesktop(), setup_for(tmp_path), read_document=lambda: NOTE).get("verify_document_text") is not None


# --- where it sits in the plan, and what a failure leaves undone ---------------------------------------------------------------------------------------


def test_the_document_check_comes_right_after_the_note_and_before_any_save_action(tmp_path):
    plan = build_raw_plan(setup_for(tmp_path))
    ids = [step["id"] for step in plan]
    at = ids.index("verify_document_text")

    assert ids[ids.index("type_note") + 1] == "verify_document_text" and ids[at + 1] == "open_save_as"
    assert plan[at]["tool"] == "verify_document_text" and plan[at]["arguments"] == {}
    assert plan[at]["max_attempts"] == 1 and plan[at]["retry_delay_seconds"] == 0.0  # never re-read, never repaired
    before = plan[:at]
    assert not [s for s in before if s["tool"] == "press_key" and s["arguments"]["key"] in ("ctrl+s", "enter")]  # no save action precedes it
    assert [s["id"] for s in before if s["tool"] == "type_text"] == ["type_note"]  # the file name is typed only after it


def test_the_plan_still_fits_the_twenty_step_limit_with_no_check_removed_except_the_one_it_replaces(tmp_path):
    ids = [step["id"] for step in build_raw_plan(setup_for(tmp_path))]

    assert len(ids) <= notepad_note.MAX_STEPS == 20
    assert "verify_focus_for_save" not in ids  # replaced: the document check performs that foreground check itself, on the window it then reads
    for kept in ("verify_notepad", "verify_focus_for_note", "verify_save_dialog", "verify_focus_for_name", "verify_target_free", "verify_dialog_before_save",
                 "verify_focus_before_save", "verify_file", "verify_saved_title"):
        assert kept in ids, kept


@pytest.mark.parametrize(
    "desktop",
    [
        pytest.param(FakeDesktop(drop_index=7), id="a dropped d"),
        pytest.param(FakeDesktop(mistype=True), id="a wrong first letter"),
        pytest.param(FakeDesktop(swallow_typing=True), id="nothing typed"),
        pytest.param(FakeDesktop(initial_document="old text ", ignore_new_tab=True), id="stale document, Ctrl+N lost"),
    ],
)
def test_a_wrong_document_stops_the_run_before_ctrl_s_with_nothing_saved_repaired_or_retyped(tmp_path, desktop):
    setup, desktop, report, the_gate = run(tmp_path, desktop)

    steps = steps_of(report)
    assert not report.succeeded and report.job.status is JobStatus.FAILED
    assert steps["verify_document_text"].status is StepStatus.FAILED and steps["verify_document_text"].attempts == 1
    assert [k for k in desktop.log if k[0] == "type"] == [("type", NOTE)]  # typed once, never retyped
    assert ("key", "ctrl+s") not in desktop.log and ("key", "enter") not in desktop.log and desktop.log[-1] == ("type", NOTE)
    assert not desktop.dialog_open and list(setup.directory.iterdir()) == []
    assert all(steps[i].status is StepStatus.SKIPPED for i in ("open_save_as", "type_file_name", "confirm_save", "verify_file"))
    assert the_gate.remaining() == 10 - 3  # open, Ctrl+N and the note: the check used no slot and nothing after it ran
    assert "NOT confirmed in the Notepad document" in report.text and "Save As was never opened" in report.text
    assert "the Save As dialog may still be open" not in report.text and "Enter was pressed" not in report.text


def test_a_reader_that_fails_or_hangs_stops_the_run_before_ctrl_s(tmp_path):
    def hung():
        raise DocumentReadError("the editor did not answer within 500 ms (it may be busy or hung)")

    setup, desktop, report, _ = run(tmp_path, FakeDesktop(), read_document=hung)

    assert steps_of(report)["verify_document_text"].status is StepStatus.FAILED and "500 ms" in steps_of(report)["verify_document_text"].error
    assert ("key", "ctrl+s") not in desktop.log and not setup.path.exists()


def test_the_document_check_is_also_the_last_foreground_check_before_ctrl_s(tmp_path):
    setup, desktop, report, _ = run(tmp_path, FakeDesktop(override_from=(3, WindowInfo("Some other app", "other.exe", False))))

    steps = steps_of(report)
    assert steps["verify_document_text"].status is StepStatus.FAILED and "notepad.exe" in steps["verify_document_text"].error
    assert ("key", "ctrl+s") not in desktop.log and desktop.misdirected == []


# ====================== wiring and the desktop tripwires: what the offline tests above cannot see about the REAL reader and the REAL Windows API =================


def test_the_default_wiring_of_the_document_step_calls_the_name_notepad_note_read_visible_editor_text(tmp_path, monkeypatch, no_real_document_read):
    """Built WITHOUT a reader, the registry's document tool must reach the reader through `notepad_note.read_visible_editor_text` - looked up when the step runs,
    not captured when the registry is built. The autouse tripwire is that name here, so reaching it is the whole point of this test and cannot touch the desktop."""
    waits = []
    tool = build_registry(FakeDesktop(), setup_for(tmp_path), sleep=waits.append).get("verify_document_text")

    result = tool.execute()

    assert no_real_document_read == ["notepad_note.read_visible_editor_text"]  # the default path went through exactly that name, once
    assert result.status is ToolResultStatus.ERROR and "tripwire" in result.message and "nothing was saved" in result.message  # a reader failure is a failed check
    assert waits == [notepad_note.SETTLE_SECONDS]  # and the injected sleep was used (this test did not really wait)
    no_real_document_read.clear()  # deliberate: acknowledge the one intended hit so the teardown guard stays strict for every other test

    monkeypatch.setattr(notepad_note, "read_visible_editor_text", lambda: NOTE)  # the name is looked up at call time, so replacing it replaces the reader
    assert tool.execute().status is ToolResultStatus.SUCCESS
    assert no_real_document_read == []


def test_an_injected_reader_is_used_instead_of_the_default_and_the_default_is_never_touched(tmp_path, no_real_document_read):
    seen = []

    def injected():
        seen.append(1)
        return NOTE

    tool = build_registry(FakeDesktop(), setup_for(tmp_path), sleep=lambda seconds: None, read_document=injected).get("verify_document_text")

    assert tool.execute().status is ToolResultStatus.SUCCESS and seen == [1] and no_real_document_read == []


def test_the_reader_module_called_without_a_fake_api_hits_the_tripwire_and_never_builds_the_real_windows_api(no_real_document_read):
    """A future test that calls `notepad_reader.read_visible_editor_text()` with no fake API would build the real Win32WindowApi and ask Windows about the real
    foreground window. The autouse fixture replaces that class, so the attempt is recorded (and would fail the test at teardown) instead."""
    with pytest.raises(DocumentReadError, match=r"could not be inspected \(RuntimeError\)"):
        notepad_reader.read_visible_editor_text()

    assert no_real_document_read == ["notepad_reader.Win32WindowApi"]
    no_real_document_read.clear()  # deliberate, as above


def test_a_fake_api_is_never_stopped_by_the_tripwire(no_real_document_read):
    assert notepad_reader.read_visible_editor_text(FakeWindowApi(notepad_windows(NOTE))) == NOTE
    assert no_real_document_read == []


@pytest.mark.skipif(os.name != "nt", reason="the real window API is Windows only")
def test_the_real_windows_api_class_is_built_with_exactly_the_signature_of_the_probe_that_worked_without_calling_any_windows_api_method():
    """CONSTRUCTOR ONLY. Building it imports win32gui and opens a private handle to user32.dll to set the ctypes types; it sends no message, enumerates no
    window and reads nothing. No method of the instance is called, so no window is ever asked anything."""
    import ctypes
    from ctypes import wintypes

    api = REAL_WIN32_WINDOW_API()

    send = api._send
    assert send.__name__ == "SendMessageTimeoutW"  # the Unicode entry point, as in the probe
    assert send.argtypes == [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM, wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
    assert send.restype is ctypes.c_size_t  # non-zero = the window answered; the message's own result comes back through the pointer argument
    pointer = ctypes.sizeof(ctypes.c_void_p)
    assert ctypes.sizeof(wintypes.WPARAM) == ctypes.sizeof(wintypes.LPARAM) == ctypes.sizeof(ctypes.c_size_t) == ctypes.sizeof(wintypes.HWND) == pointer
    assert ctypes.sizeof(wintypes.UINT) == 4  # flags and timeout are 32-bit, so 0x0002 and 500 pass exactly
    assert send is not ctypes.windll.user32.SendMessageTimeoutW  # a PRIVATE handle: setting these types changed no other code's view of user32
    assert ctypes.sizeof(ctypes.create_unicode_buffer(1)) == ctypes.sizeof(ctypes.c_wchar) * 1 == 2  # UTF-16 units, as WM_GETTEXT on a Unicode window writes
    # win32gui is the module the reader will call; check only that the four read-only functions exist (not called)
    for name in ("GetForegroundWindow", "EnumChildWindows", "GetClassName", "IsWindowVisible"):
        assert callable(getattr(api._gui, name)), name


def test_the_windows_message_constants_and_flags_are_the_documented_values():
    assert notepad_reader.WM_GETTEXT == 0x000D and notepad_reader.WM_GETTEXTLENGTH == 0x000E and notepad_reader.SMTO_ABORTIFHUNG == 0x0002
    assert notepad_reader.READ_TIMEOUT_MS == 500 and notepad_reader.EDITOR_CLASS == "RichEditD2DPT" and notepad_reader.NOTEPAD_PROCESS == "notepad.exe"
