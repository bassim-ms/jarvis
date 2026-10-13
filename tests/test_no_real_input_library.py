"""
The suite-wide OS-input isolation (tests/conftest.py + tests/fake_input_library.py): the fake is installed, the backend uses it, the tripwires work, and the REAL input
library is not loaded.

SAFETY OF THESE TESTS THEMSELVES: every test first requires the isolation report to be all green (the autouse fixture below), and before calling anything that would be
dangerous if it were NOT a tripwire, a test asserts that it IS one. A tripwire is only ever called after it has been identified as a tripwire, so a broken installation
fails a test instead of reaching the OS.
"""

import ast
import ctypes
import importlib
import os
import pathlib
import subprocess
import sys
import webbrowser

import pytest

import fake_input_library as fake
from fake_input_library import BLOCKED_USER32, BLOCKED_WIN32API, BLOCKED_WIN32GUI, RECORDED_FUNCTIONS, TRIPWIRE_MARK, RealInputBlocked

TESTS_DIR = pathlib.Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def isolation_must_hold():
    report = fake.isolation_report()
    assert report["all_ok"], f"the OS-input isolation is NOT in place, so this file refuses to run anything: {report}"


def must_be_a_tripwire(obj, what):
    assert getattr(obj, TRIPWIRE_MARK, False) is True, f"{what} is NOT a tripwire: refusing to call it"
    return obj


def hits_after(call):
    before = len(fake.TRIPWIRE_HITS)
    with pytest.raises(RealInputBlocked):
        call()
    return fake.TRIPWIRE_HITS[before:]


# ================================================ the fake is installed, and the real library is not loaded =====================================


def test_the_fake_input_library_is_what_every_import_of_pyautogui_gets():
    import pyautogui

    assert pyautogui is fake.FAKE is sys.modules["pyautogui"] is importlib.import_module("pyautogui")
    assert type(pyautogui).IS_JARVIS_FAKE_INPUT_LIBRARY is True
    assert "__file__" not in vars(pyautogui) and not hasattr(pyautogui, "__path__")  # nothing on disk behind it, and not a package


def test_the_real_library_is_not_loaded_and_cannot_be_reached_by_any_other_route():
    assert not [name for name in sys.modules if name.startswith("pyautogui.")]
    assert "pyautogui._pyautogui_win" not in sys.modules
    for name in ("pyautogui._pyautogui_win", "pyautogui.tweens", "pydirectinput", "pynput", "pynput.keyboard"):
        with pytest.raises(ImportError):
            importlib.import_module(name)
    assert fake.isolation_report()["real_library_not_loaded"] is True


def test_the_fake_has_every_name_the_existing_tests_and_the_backend_use_and_nothing_made_up():
    for name in (*RECORDED_FUNCTIONS, "size", "position", "screenshot", "onScreen", "FAILSAFE", "PAUSE", "FAILSAFE_POINTS", "FailSafeException"):
        assert hasattr(fake.FAKE, name), name
    with pytest.raises(AttributeError, match="no attribute"):
        fake.FAKE.a_function_that_does_not_exist  # noqa: B018 - a typo in a test fails loudly instead of silently recording
    with pytest.raises(RuntimeError, match="no screen"):
        fake.FAKE.screenshot()  # it never takes a real screenshot either


def test_the_fake_records_calls_and_sends_nothing():
    import pyautogui

    pyautogui.write("hello", interval=0.03)
    pyautogui.click(x=1, y=2, button="left")
    pyautogui.scroll(5)

    assert fake.FAKE.calls == [("write", ("hello",), {"interval": 0.03}), ("click", (), {"x": 1, "y": 2, "button": "left"}), ("scroll", (5,), {})]


def test_each_test_starts_with_an_empty_record():
    assert fake.FAKE.calls == []  # tests/conftest.py clears it before every test


# ================================================ the backend uses the fake ====================================================================


def test_the_real_backend_module_holds_the_fake_not_the_real_library():
    import app.computer.windows_backend as backend_module

    assert vars(backend_module)["pyautogui"] is fake.FAKE
    assert fake.isolation_report()["backend_pyautogui_is_fake"] is True


def test_an_admitted_action_through_the_real_backend_lands_in_the_fake_and_nowhere_else():
    import app.computer.windows_backend as backend_module
    from app.computer.authority import InputStep
    from app.computer.gate import InputGate
    from app.computer.issuer import GrantIssuer

    assert vars(backend_module)["pyautogui"] is fake.FAKE  # checked BEFORE anything is called
    gate = InputGate(lambda: True, lambda: 10)
    grant = GrantIssuer(gate).issue([InputStep("scroll", (5,))], ttl_seconds=60, purpose="isolation_test", actor="test")

    backend_module.WindowsComputerBackend(gate=gate, authority=grant).scroll(5)

    assert fake.FAKE.calls == [("scroll", (5,), {})]


# ================================================ the tripwires =================================================================================


@pytest.mark.skipif(not hasattr(ctypes, "windll"), reason="Windows only")
@pytest.mark.parametrize("name", sorted(BLOCKED_USER32))
def test_the_raw_user32_input_functions_are_tripwires_on_the_shared_handle(name):
    function = must_be_a_tripwire(vars(ctypes.windll.user32)[name], f"ctypes.windll.user32.{name}")

    assert hits_after(function) == [f"user32.{name}"]


@pytest.mark.skipif(not hasattr(ctypes, "windll"), reason="Windows only")
@pytest.mark.parametrize("name", sorted(BLOCKED_USER32))
def test_a_freshly_opened_user32_handle_cannot_get_the_native_input_functions_either(name):
    handle = ctypes.WinDLL("user32")
    by_item = must_be_a_tripwire(handle[name], f"WinDLL('user32')[{name!r}]")
    by_attribute = must_be_a_tripwire(getattr(handle, name), f"WinDLL('user32').{name}")

    assert hits_after(by_item) == [f"user32.{name}"] and hits_after(by_attribute) == [f"user32.{name}"]


def test_other_user32_functions_are_left_alone_only_the_input_ones_are_blocked():
    handle = ctypes.WinDLL("user32")
    harmless = handle["GetSystemMetrics"]  # fetched, NEVER called here

    assert getattr(harmless, TRIPWIRE_MARK, False) is False


@pytest.mark.parametrize("name", sorted(BLOCKED_WIN32API))
def test_the_pywin32_twins_of_the_input_functions_are_tripwires(name):
    win32api = pytest.importorskip("win32api")
    function = must_be_a_tripwire(getattr(win32api, name), f"win32api.{name}")

    assert hits_after(function) == [f"win32api.{name}"]


@pytest.mark.parametrize("name", sorted(BLOCKED_WIN32GUI))
def test_the_pywin32_window_focus_calls_are_tripwires(name):
    win32gui = pytest.importorskip("win32gui")
    if not hasattr(win32gui, name):
        pytest.skip(f"this pywin32 has no win32gui.{name}")
    function = must_be_a_tripwire(getattr(win32gui, name), f"win32gui.{name}")

    assert hits_after(function) == [f"win32gui.{name}"]


def test_the_backends_private_user32_handle_cannot_reach_the_focus_functions_either():
    import app.computer.windows_backend as backend_module

    private = vars(backend_module)["_user32"]  # the backend's own WinDLL instance, opened after the guard was installed
    for name in ("SetForegroundWindow", "BringWindowToTop", "AttachThreadInput"):
        assert name in BLOCKED_USER32
        function = must_be_a_tripwire(getattr(private, name), f"_user32.{name}")
        assert hits_after(function) == [f"user32.{name}"]


@pytest.mark.parametrize("name", ["open", "open_new", "open_new_tab"])
def test_opening_a_browser_is_a_tripwire(name):
    function = must_be_a_tripwire(getattr(webbrowser, name), f"webbrowser.{name}")

    assert hits_after(lambda: function("https://example.invalid/")) == [f"webbrowser.{name}"]


@pytest.mark.skipif(not hasattr(os, "startfile"), reason="Windows only")
def test_os_startfile_is_a_tripwire():
    function = must_be_a_tripwire(os.startfile, "os.startfile")

    assert hits_after(lambda: function("notes.txt")) == ["os.startfile"]


@pytest.mark.parametrize(
    "command",
    [["notepad.exe"], ["calc.exe"], ["explorer.exe"], ["mspaint.exe"], ["chrome.exe"], ["msedge.exe", "--version"], ["firefox.exe"],
     r"C:\Windows\System32\notepad.exe", '"C:\\Program Files\\Mozilla Firefox\\firefox.exe" https://example.invalid', ["NOTEPAD.EXE"]],
    ids=lambda c: str(c)[:40],
)
def test_launching_any_program_open_application_can_launch_is_blocked_before_anything_starts(command):
    must_be_a_tripwire(subprocess.Popen, "subprocess.Popen")

    hits = hits_after(lambda: subprocess.Popen(command))

    assert len(hits) == 1 and hits[0].startswith("subprocess.Popen(")


def test_a_launch_through_subprocess_run_is_blocked_too():
    must_be_a_tripwire(subprocess.Popen, "subprocess.Popen")

    assert len(hits_after(lambda: subprocess.run(["notepad.exe"]))) == 1


def test_ordinary_subprocess_use_still_works_so_the_guard_is_narrow():
    completed = subprocess.run([sys.executable, "-c", "print(7)"], capture_output=True, text=True, timeout=60)

    assert completed.stdout.strip() == "7" and completed.returncode == 0


def test_the_tripwires_raise_a_base_exception_that_an_except_exception_cannot_swallow():
    assert issubclass(RealInputBlocked, BaseException) and not issubclass(RealInputBlocked, Exception)
    function = must_be_a_tripwire(vars(ctypes.windll.user32)["SendInput"], "SendInput") if hasattr(ctypes, "windll") else must_be_a_tripwire(webbrowser.open, "webbrowser.open")
    swallowed = False
    try:
        try:
            function()
        except Exception:  # an application-style handler
            swallowed = True
    except RealInputBlocked:
        pass

    assert swallowed is False


# ================================================ the installation itself =======================================================================


def test_install_is_idempotent_and_restores_the_fake_if_something_replaced_it():
    decoy = sys.modules["pyautogui"]
    try:
        sys.modules["pyautogui"] = object()  # a decoy
        assert fake.isolation_report()["fake_installed"] is False
        fake.install()
        assert sys.modules["pyautogui"] is fake.FAKE and fake.isolation_report()["all_ok"] is True
        before = (len(sys.meta_path), subprocess.Popen, ctypes.CDLL.__getitem__)
        fake.install()
        assert (len(sys.meta_path), subprocess.Popen, ctypes.CDLL.__getitem__) == before  # nothing is wrapped twice
    finally:
        sys.modules["pyautogui"] = decoy


def test_the_isolation_report_names_every_protection_and_is_all_green():
    report = fake.isolation_report()

    assert set(report) >= {
        "fake_installed", "real_library_not_loaded", "import_blocker_installed", "user32_shared_handle_tripwired", "user32_new_handles_guarded",
        "win32api_tripwired", "win32gui_focus_tripwired", "webbrowser_tripwired", "startfile_tripwired", "program_launch_guarded", "backend_pyautogui_is_fake", "all_ok",
    }
    assert report["all_ok"] is True


def test_the_isolation_report_turns_red_if_any_one_protection_is_missing(monkeypatch):
    damages = {
        "fake_installed": lambda scoped: scoped.setitem(sys.modules, "pyautogui", object()),  # a decoy where the fake should be
        "webbrowser_tripwired": lambda scoped: scoped.setattr(webbrowser, "open", lambda *a, **k: None),  # a harmless stand-in that is not a tripwire
        "win32gui_focus_tripwired": lambda scoped: scoped.setattr(pytest.importorskip("win32gui"), "ShowWindow", lambda *a, **k: None),
        "program_launch_guarded": lambda scoped: scoped.setattr(subprocess, "Popen", subprocess.Popen.__mro__[1]),  # the unguarded parent class
    }
    for key, damage in damages.items():
        with monkeypatch.context() as scoped:  # each damage is undone before the next one, and before the test ends
            damage(scoped)
            report = fake.isolation_report()
        assert report[key] is False and report["all_ok"] is False, key
    assert fake.isolation_report()["all_ok"] is True  # and everything is back


def test_the_support_files_never_import_the_real_library():
    for name in ("fake_input_library.py", "conftest.py"):  # (the test files say `import pyautogui` on purpose: that import returns the fake)
        tree = ast.parse((TESTS_DIR / name).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & {"pyautogui", "pydirectinput", "pynput"}), name


def test_the_fake_is_installed_before_the_application_can_import_anything():
    source = (TESTS_DIR / "conftest.py").read_text(encoding="utf-8")

    assert source.index("fake_input_library.install()") < source.index("def pytest_configure")  # installed at import time, not lazily
    assert "app." not in source.split("def pytest_configure")[0]  # and conftest itself imports nothing from the application first
