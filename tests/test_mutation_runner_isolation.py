"""
The isolated mutation runner (tests/mutation_runner.py): its canary, its refusal to run without isolation, and a mutant that SKIPS ADMISSION - the very kind of mutant that,
in an earlier ad-hoc harness, reached the real input library. Here it is proved to reach ONLY the fake, and to be refused by the expected test.

ORDER MATTERS and is part of the safety design. The tests run top to bottom in this file:
  1. the canary - a child process proves the isolation and applies nothing;
  2. the refusal - a child process whose isolation is deliberately broken must REFUSE (exit 3) and apply nothing;
  3. the dangerous mutants - skipped unless the canary passed, and the child re-proves the isolation itself before it touches anything.
Every mutant runs in its own child process; this (parent) process never applies a mutant and never imports the mutated code in a mutated state.
"""

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

import fake_input_library as fake

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tests" / "mutation_runner.py"
PROVED = {"canary": False, "first_mutant": False}


@pytest.fixture(autouse=True)
def the_parent_process_is_isolated_too():
    assert fake.isolation_report()["all_ok"], "the OS-input isolation is not in place in the parent process; refusing to run"


def run_child(*args, timeout=300):
    env = dict(os.environ)
    env.update({"JARVIS_COMPUTER_INPUT_ENABLED": "false", "JARVIS_FREELLMAPI_API_KEY": "", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    completed = subprocess.run([sys.executable, str(RUNNER), *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON: ")]
    result = json.loads(lines[-1][len("RESULT_JSON: "):]) if lines else None
    return completed.returncode, result, completed


# ================================================ 1. the canary: prove the isolation, change nothing =============================================


def test_01_the_canary_proves_the_isolation_in_a_separate_process_and_applies_nothing():
    code, result, completed = run_child("--canary")

    assert result is not None, completed.stdout + completed.stderr
    assert code == 0 and result["ok"] is True, result
    assert result["mutant_applied"] is False
    assert all(result["isolation"].values()), result["isolation"]  # every protection green, including "the backend module holds the fake"
    assert result["isolation"]["real_library_not_loaded"] is True and result["isolation"]["backend_pyautogui_is_fake"] is True
    assert result["probe_seen_by_the_fake"] == [["click", [1, 1], {}]]  # a call made through the module-level name landed in the FAKE recorder
    assert result["tripwire_hits"] == 0
    PROVED["canary"] = True


def test_02_the_runner_lists_its_mutants_and_that_touches_nothing():
    code, result, completed = run_child("--list")

    assert code == 0 and completed.stdout.split() == [
        "admit_without_grant", "skip_click_coordinate_admission", "skip_open_url_admission", "skip_scroll_admission", "skip_switch_window_admission", "substring_window_match",
    ]


# ================================================ 2. the runner REFUSES when isolation is missing ===============================================


def test_03_with_the_isolation_deliberately_broken_the_runner_refuses_and_applies_no_mutant():
    code, result, completed = run_child("--selftest-refuses")

    assert code == 3, (code, completed.stdout, completed.stderr)  # 3 = isolation not proved
    assert result["refused"] is True and result["mutant_applied"] is False and result["stage"] == "selftest"
    assert result["isolation"]["fake_installed"] is False and result["isolation"]["all_ok"] is False


def test_04_the_runner_is_built_so_nothing_can_run_before_the_isolation_is_checked():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert source.index("fake.install()  # BEFORE the application") < source.index("return mode_canary()")  # installed first in main(), before any mode runs
    function = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = [(node.lineno, node.func.id) for node in ast.walk(function["mode_mutant"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    first = {name: min(line for line, called in calls if called == name) for name in ("require_isolation", "probe", "apply_mutant")}
    assert first["require_isolation"] < first["apply_mutant"]  # the check comes before the mutant is applied
    assert "types.FunctionType(built.__code__, vars(wb)" in source  # the mutant is bound to the LIVE module namespace, never a copy
    blocked = {"pyautogui", "pydirectinput", "pynput"}

    def blocked_imports(nodes):
        found = []
        for node in nodes:
            if isinstance(node, ast.Import):
                found += [alias.name.split(".")[0] for alias in node.names if alias.name.split(".")[0] in blocked]
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in blocked:
                found.append(node.module.split(".")[0])
        return found

    assert blocked_imports(tree.body) == []  # no module-level import of an input library, so nothing can load one before the fake is installed
    inside = {name: blocked_imports(ast.walk(node)) for name, node in function.items() if blocked_imports(ast.walk(node))}
    assert inside == {"mode_canary": ["pyautogui"]}  # the ONLY one is the canary's deliberate `import pyautogui` AFTER install(), to prove that import yields the fake
    assert "import_module" not in source


# ================================================ 3. a mutant that SKIPS ADMISSION: fake only, refused by the expected test ====================


@pytest.mark.parametrize(
    "mutant", ["skip_scroll_admission", "skip_click_coordinate_admission", "admit_without_grant", "skip_open_url_admission", "skip_switch_window_admission", "substring_window_match"]
)
def test_05_a_mutant_that_skips_admission_reaches_only_the_fake_recorder_and_is_caught_by_the_expected_test(mutant):
    if not PROVED["canary"]:
        pytest.skip("the canary has not passed, so the isolation is unproved: no dangerous mutant is run")
    if mutant != "skip_scroll_admission" and not PROVED["first_mutant"]:
        pytest.skip("the first mutant was not proved safe, so no further mutant is run")

    code, result, completed = run_child("--mutant", mutant)

    assert result is not None, completed.stdout + completed.stderr
    assert code == 0 and result["ok"] is True, json.dumps(result, indent=1)[:3000]
    assert result["mutant_applied"] is True
    assert all(result["isolation_before"].values()) and all(result["isolation_after"].values())  # proved before the mutant AND after it ran
    control_calls = {"substring_window_match": [["enumerate", [], {}]]}.get(mutant, [])  # (the matcher mutant is AUTHORIZED: the unmutated matcher looks once and finds no exact match)
    assert result["control_unmutated"]["refused"] is True and result["control_unmutated"]["fake_calls"] == control_calls  # the unmutated code refuses; nothing was focused or opened
    assert result["mutated_probe"]["refused"] is False  # the mutant skipped admission ...
    expected = {
        "skip_scroll_admission": [["scroll", [5], {}]],
        "skip_click_coordinate_admission": [["click", [], {"button": "left", "x": 10, "y": 20}]],
        "admit_without_grant": [["scroll", [5], {}]],
        "skip_open_url_admission": [["browser", ["https://example.com/"], {}], ["activate", ["https://example.com/"], {}]],  # the browser and the 'bring it forward' RECORDERS
        "skip_switch_window_admission": [["enumerate", [], {}], ["bring", [11], {}]],  # the window list and the focus call RECORDERS
        "substring_window_match": [["enumerate", [], {}], ["bring", [21], {}]],  # the window "... - extra" that the grant never named
    }[mutant]
    assert result["mutated_probe"]["fake_calls"] == expected  # ... and its OS calls went to the recorders, exactly these, in this order
    assert result["tripwire_hits"] == 0 and result["fake_calls_left_over"] == 0  # no raw input function was touched
    assert result["pytest"]["pytest_exit_code"] != 0 and result["pytest"]["failed"][:1] == [result["expected_test"]]  # and the expected test caught it
    if mutant == "skip_scroll_admission":
        PROVED["first_mutant"] = True
