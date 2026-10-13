"""
Isolated mutation runner: break the enforcement ON PURPOSE, in a child process that cannot reach real OS input, and check that the tests notice.

    python tests/mutation_runner.py --canary                     prove the isolation, change nothing
    python tests/mutation_runner.py --mutant skip_scroll_admission
    python tests/mutation_runner.py --selftest-refuses           prove the runner REFUSES to apply a mutant when isolation is missing
    python tests/mutation_runner.py --list

This file is not named test_*.py, so pytest never collects it: it runs only when somebody runs it (tests/test_mutation_runner_isolation.py does, in child processes).

WHY IT EXISTS. An earlier, ad-hoc harness rebuilt a mutated backend method inside a COPY of the module namespace taken before the test fixtures ran. That copy held the REAL
`pyautogui`, so a mutant that skipped its admission called the real library. This runner cannot do that, by construction:

  1. The very first thing it does - before the application is imported - is `tests/fake_input_library.install()`: `sys.modules["pyautogui"]` becomes the fake, the raw
     user32 / win32api input functions, `webbrowser.open*`, `os.startfile` and program launches become tripwires, and the real library is import-blocked.
  2. Before ANY mutant is applied it calls `require_isolation()`, which re-checks all of that (plus that the backend module holds the fake) and exits with code 3 - having
     applied nothing - if even one check fails.
  3. A mutant is built with `types.FunctionType(code, vars(module), ...)`: its globals ARE the live module dict, so the `pyautogui` it calls is whatever the module holds at
     call time (the fake), never a stale copy.
  4. After the mutant ran, the isolation is checked again and the number of tripwire hits is reported.

Every mode prints one line `RESULT_JSON: {...}` for the parent test to read. Exit codes: 0 = as expected, 1 = unexpected outcome, 2 = bad usage, 3 = isolation not proved.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import os
import sys
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fake_input_library as fake  # noqa: E402  (imports no application code and no input library)

STATE = {"mutant_applied": False}


def emit(payload: dict) -> None:
    print("RESULT_JSON: " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def require_isolation(stage: str) -> dict:
    """Re-check every protection. If anything is missing: say so, apply nothing, and exit 3. Called before any mutant is applied and again afterwards."""
    report = fake.isolation_report()
    if not report["all_ok"]:
        emit({"refused": True, "stage": stage, "mutant_applied": STATE["mutant_applied"], "isolation": report})
        sys.exit(3)
    return report


def _stub_browser(wb) -> None:
    """The browser and the 'bring it forward' step become recorders (written into the fake's own record), so a mutant of `open_url` can reach nothing real."""
    wb.webbrowser = types.SimpleNamespace(open=lambda url: (fake.FAKE.calls.append(("browser", (url,), {})), True)[1])
    wb.WindowsComputerBackend._activate_url_handler_window = staticmethod(lambda url: fake.FAKE.calls.append(("activate", (url,), {})))


def _stub_windows(windows):
    """The window list, the foreground lookup and the focus call become recorders: nothing real is listed or brought forward."""

    def install(wb) -> None:
        state = {"foreground": None}

        def enumerate_windows():
            fake.FAKE.calls.append(("enumerate", (), {}))
            return list(windows)

        def bring(hwnd):
            fake.FAKE.calls.append(("bring", (hwnd,), {}))
            state["foreground"] = hwnd

        wb._list_switchable_windows = enumerate_windows
        wb._foreground_window = lambda: state["foreground"]
        wb._bring_to_foreground = bring
        wb.WindowsComputerBackend._is_maximized = staticmethod(lambda hwnd: False)

    return install


# The mutants. `old` must appear in the method's source; `new` replaces it. `probe` is the call made with NO grant, which the unmutated method refuses.
MUTANTS = {
    "skip_scroll_admission": {
        "method": "scroll",
        "old": 'admit_input(self.gate, self._authority, "scroll", amount)',
        "new": "pass",
        "probe": lambda backend, MouseButton: backend.scroll(5),
        "fake_call": ["scroll", [5], {}],
        "expected_failure": "tests/test_computer_authorization.py::test_every_input_method_is_denied_when_the_switch_is_off_even_with_a_grant[scroll]",
    },
    "skip_click_coordinate_admission": {
        "method": "click_mouse",
        "old": 'admit_input(self.gate, self._authority, "click_mouse", x, y, button)',
        "new": "pass",
        "probe": lambda backend, MouseButton: backend.click_mouse(10, 20, MouseButton.LEFT),
        "fake_call": ["click", [], {"button": "left", "x": 10, "y": 20}],
        "expected_failure": "tests/test_computer_authorization.py::test_every_input_method_is_denied_when_the_switch_is_off_even_with_a_grant[click_mouse]",
    },
    # Brings the removed budget-only mode BACK, at the gate: `InputGate.admit` stops requiring a grant (no "no_authority" refusal, no defensive tail) and, when it has none,
    # spends a stand-in entry instead. A real backend on a real gate is then asked to scroll with NO grant: unmutated it must refuse; mutated it reaches ONLY the fake.
    "admit_without_grant": {
        "module": "gate",
        "method": "admit",
        "edits": [
            ('code = "no_authority"', "pass"),  # the first one in `admit`: a missing/unknown grant is no longer refused
            ("if code is None and state is None:", "if False:"),  # and neither is the defensive tail that backs it up
            ("state.index += 1", 'state = state or __import__("types").SimpleNamespace(index=0, macs=(0,), closed=None, grant=__import__("types").SimpleNamespace(short_id="mutant")); state.index += 1'),
        ],
        "probe": lambda backend, MouseButton: backend.scroll(5),
        "fake_call": ["scroll", [5], {}],
        "expected_failure": "tests/test_computer_gate_no_bypass.py::test_admit_refuses_every_action_without_a_real_registered_grant_with_the_switch_on_and_a_huge_limit[nothing-scroll]",
    },
    # open_url and switch_window: the same two kinds of break, aimed at the browser and the focus calls, which are RECORDERS in the child process (see the stubs above).
    "skip_open_url_admission": {
        "method": "open_url",
        "old": 'admit_input(self.gate, self._authority, "open_url", url)',
        "new": "pass",
        "stubs": _stub_browser,
        "probe": lambda backend, MouseButton: backend.open_url("https://example.com/"),
        "fake_calls": [["browser", ["https://example.com/"], {}], ["activate", ["https://example.com/"], {}]],
        "expected_failure": "tests/test_computer_focus_authorization.py::test_a_real_backend_with_no_grant_refuses_even_with_the_switch_on_and_a_huge_limit_and_touches_nothing[open_url]",
    },
    "skip_switch_window_admission": {
        "method": "switch_window",
        "old": 'admit_input(self.gate, self._authority, "switch_window", exact_title, exact_process)',
        "new": "pass",
        "stubs": _stub_windows([(11, "Untitled - Notepad", "Notepad.exe")]),
        "probe": lambda backend, MouseButton: backend.switch_window("Untitled - Notepad", "notepad.exe"),
        "fake_calls": [["enumerate", [], {}], ["bring", [11], {}]],
        "expected_failure": "tests/test_computer_focus_authorization.py::test_a_real_backend_with_no_grant_refuses_even_with_the_switch_on_and_a_huge_limit_and_touches_nothing[switch_window]",
    },
    # The call IS authorized; the break is in the matcher: a title that merely CONTAINS the requested one is now accepted, so a window the grant never named gets the focus.
    "substring_window_match": {
        "function": "exact_window_matches",
        "module": "base",
        "rebind": ["backend"],
        "edits": [("window[1] == title", "title in window[1]")],
        "grant": ("switch_window", ("Untitled - Notepad", "notepad.exe")),
        "stubs": _stub_windows([(21, "Untitled - Notepad - extra", "Notepad.exe")]),
        "probe": lambda backend, MouseButton: backend.switch_window("Untitled - Notepad", "notepad.exe"),
        "control_calls": [["enumerate", [], {}]],  # unmutated: one look, no exact match, no focus
        "fake_calls": [["enumerate", [], {}], ["bring", [21], {}]],
        "expected_failure": "tests/test_computer_focus_authorization.py::test_zero_or_several_exact_matches_fail_without_any_focus_operation_and_the_entry_is_spent_before_the_enumeration[only a longer title]",
    },
}


def target_of(spec: dict):
    """(module, class) the mutant is applied to: the Windows backend by default, or the gate."""
    if spec.get("module") == "gate":
        import app.computer.gate as wb

        return wb, wb.InputGate
    import app.computer.windows_backend as wb

    return wb, wb.WindowsComputerBackend


def apply_function_mutant(spec: dict) -> None:
    """Mutate a module-level function of app.computer.base (e.g. the exact matcher), rebuilt on that module's LIVE dict, and re-point the backend's own reference to it."""
    import app.computer.base as base_module
    import app.computer.windows_backend as backend_module

    modules = {"base": base_module, "backend": backend_module}
    home, name = modules[spec["module"]], spec["function"]
    source = textwrap.dedent(inspect.getsource(getattr(home, name)))
    for old, _new in spec["edits"]:
        if old not in source:
            emit({"error": "the text to mutate was not found; the function changed", "function": name})
            sys.exit(2)
    for old, new in spec["edits"]:
        source = source.replace(old, new, 1)
    scratch = dict(vars(home))
    exec(compile(source, f"<mutant {name}>", "exec"), scratch)
    built = scratch[name]
    live = types.FunctionType(built.__code__, vars(home), built.__name__, built.__defaults__, built.__closure__)
    if live.__globals__ is not vars(home):  # pragma: no cover - cannot happen; a hard stop if it ever did
        emit({"error": "the mutant is not bound to the live module namespace"})
        sys.exit(2)
    setattr(home, name, live)
    for other in spec.get("rebind", ()):  # a module that imported the function by name still holds the ORIGINAL; point it at the mutant too
        setattr(modules[other], name, live)
    STATE["mutant_applied"] = True


def apply_mutant(spec: dict) -> None:
    if spec.get("function"):
        return apply_function_mutant(spec)
    wb, cls = target_of(spec)
    source = textwrap.dedent(inspect.getsource(getattr(cls, spec["method"])))
    edits = spec.get("edits") or [(spec["old"], spec["new"])]
    for old, _new in edits:
        if old not in source:
            emit({"error": "the text to mutate was not found; the method changed", "method": spec["method"]})
            sys.exit(2)
    for old, new in edits:
        source = source.replace(old, new, 1)
    scratch = dict(vars(wb))  # used ONLY to evaluate the `def` statement (annotations, defaults); the function that runs is rebuilt on the LIVE dict below
    exec(compile(source, f"<mutant {spec['method']}>", "exec"), scratch)
    built = scratch[spec["method"]]
    live = types.FunctionType(built.__code__, vars(wb), built.__name__, built.__defaults__, built.__closure__)
    live.__kwdefaults__ = built.__kwdefaults__
    if live.__globals__ is not vars(wb):  # pragma: no cover - cannot happen; a hard stop if it ever did
        emit({"error": "the mutant is not bound to the live module namespace"})
        sys.exit(2)
    setattr(cls, spec["method"], live)
    STATE["mutant_applied"] = True


def probe(spec: dict, mutated: bool) -> dict:
    """Make the call with NO grant and the switch on. Unmutated: refused. Mutated: it skips admission and reaches its OS call - which must be the fake."""
    import app.computer.windows_backend as wb
    from app.computer.base import ComputerControlError, MouseButton
    from app.computer.gate import InputGate

    fake.FAKE.calls.clear()
    gate = InputGate(lambda: True, lambda: 10)
    authority = None
    if spec.get("grant"):  # a mutant of the MATCHING, not of the admission: the call is authorized, and only the match is under test
        from app.computer.authority import InputStep
        from app.computer.issuer import GrantIssuer

        authority = GrantIssuer(gate).issue([InputStep(*spec["grant"])], ttl_seconds=60, purpose="mutation", actor="runner")
    if spec.get("stubs"):
        spec["stubs"](wb)
    backend = wb.WindowsComputerBackend(gate=gate, authority=authority)  # without a grant: authority required, none carried
    refused = False
    try:
        spec["probe"](backend, MouseButton)
    except ComputerControlError:
        refused = True
    calls = [[name, list(args), dict(kwargs)] for name, args, kwargs in fake.FAKE.calls]
    fake.FAKE.calls.clear()
    return {"mutated": mutated, "refused": refused, "fake_calls": calls}


class _Failures:
    def __init__(self):
        self.failed: list[str] = []

    def pytest_runtest_logreport(self, report):
        if report.failed and report.nodeid not in self.failed:
            self.failed.append(report.nodeid)


def run_expected_test(nodeid: str) -> dict:
    import pytest

    collector = _Failures()
    buffer = io.StringIO()
    os.chdir(ROOT)
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = pytest.main([nodeid, "-x", "-p", "no:cacheprovider", "-q", "--no-header", "-W", "ignore"], plugins=[collector])
    return {"pytest_exit_code": int(code), "failed": collector.failed}


def mode_canary() -> int:
    import app.computer.windows_backend as wb  # imported only AFTER fake.install()

    report = require_isolation("canary")
    import pyautogui

    fake.FAKE.calls.clear()
    pyautogui.click(1, 1)  # the probe: through the module-level name, exactly as any code would
    seen = [[name, list(args), dict(kwargs)] for name, args, kwargs in fake.FAKE.calls]
    fake.FAKE.calls.clear()
    ok = seen == [["click", [1, 1], {}]] and vars(wb)["pyautogui"] is fake.FAKE and len(fake.TRIPWIRE_HITS) == 0
    emit({"mode": "canary", "ok": ok, "isolation": report, "probe_seen_by_the_fake": seen, "tripwire_hits": len(fake.TRIPWIRE_HITS), "mutant_applied": False})
    return 0 if ok else 1


def mode_selftest_refuses() -> int:
    """Simulate lost isolation (a decoy where the fake should be) and show the runner refuses - having applied nothing."""
    import app.computer.windows_backend  # noqa: F401  (imported while the isolation was still good)

    sys.modules["pyautogui"] = types.ModuleType("pyautogui")  # NOT the fake: isolation is now unproven
    require_isolation("selftest")  # must exit(3) here
    emit({"mode": "selftest-refuses", "error": "the runner did NOT refuse"})  # pragma: no cover
    return 1  # pragma: no cover


def mode_mutant(name: str) -> int:
    spec = MUTANTS[name]
    import app.computer.windows_backend  # noqa: F401

    before = require_isolation("before-mutant")
    control = probe(spec, mutated=False)  # the unmutated method must refuse, and reach nothing
    apply_mutant(spec)
    hits_before = len(fake.TRIPWIRE_HITS)
    mutated = probe(spec, mutated=True)  # the mutant must reach ONLY the fake
    outcome = run_expected_test(spec["expected_failure"])
    after = require_isolation("after-mutant")
    ok = (
        control["refused"] and control["fake_calls"] == spec.get("control_calls", [])
        and not mutated["refused"] and mutated["fake_calls"] == (spec["fake_calls"] if "fake_calls" in spec else [spec["fake_call"]])
        and outcome["pytest_exit_code"] != 0 and outcome["failed"][:1] == [spec["expected_failure"]]
        and len(fake.TRIPWIRE_HITS) == hits_before == 0
    )
    emit({
        "mode": "mutant", "mutant": name, "ok": ok, "mutant_applied": STATE["mutant_applied"], "isolation_before": before, "isolation_after": after,
        "control_unmutated": control, "mutated_probe": mutated, "expected_test": spec["expected_failure"], "pytest": outcome,
        "tripwire_hits": len(fake.TRIPWIRE_HITS), "fake_calls_left_over": len(fake.FAKE.calls),
    })
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tests/mutation_runner.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--canary", action="store_true", help="prove the isolation; apply nothing")
    group.add_argument("--mutant", choices=sorted(MUTANTS), help="apply one mutant (after proving isolation) and run its expected test")
    group.add_argument("--selftest-refuses", action="store_true", help="prove the runner refuses to apply a mutant when isolation is missing")
    group.add_argument("--list", action="store_true", help="list the mutants")
    args = parser.parse_args(argv)
    if args.list:
        for name in sorted(MUTANTS):
            print(name)
        return 0
    fake.install()  # BEFORE the application is imported by any mode below
    if args.canary:
        return mode_canary()
    if args.selftest_refuses:
        return mode_selftest_refuses()
    return mode_mutant(args.mutant)


if __name__ == "__main__":
    sys.exit(main())
