"""
Tests for Notifier/WindowsToastNotifier/ConsoleNotifier
(app/reminders/notifier.py). `subprocess.run` is monkeypatched, so no real
powershell.exe process or toast notification is ever raised - same
convention as monkeypatching httpx in test_gemini_provider.py.
"""

import subprocess

import pytest

from app.reminders import notifier as notifier_module
from app.reminders.notifier import ConsoleNotifier, NotifierError, WindowsToastNotifier


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def test_notify_invokes_powershell_with_file_and_named_arguments(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(notifier_module.subprocess, "run", fake_run)

    tool = WindowsToastNotifier(app_id="JarvisTest")
    tool.notify("Reminder", "walk the dog")

    args = captured["args"]
    assert args[0] == "powershell.exe"
    assert "-File" in args
    assert args[args.index("-File") + 1] == tool._script_path
    assert "-AppId" in args and args[args.index("-AppId") + 1] == "JarvisTest"
    assert "-Title" in args and args[args.index("-Title") + 1] == "Reminder"
    assert "-Message" in args and args[args.index("-Message") + 1] == "walk the dog"


def test_notify_passes_title_and_message_as_literal_arguments_not_shell_text(
    monkeypatch,
):
    """A reminder's text can contain quotes/semicolons/backticks - since
    args are a list (never a shell string), that's just literal data to
    powershell.exe, never executable script text."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(notifier_module.subprocess, "run", fake_run)

    tricky = '"; Remove-Item C:\\ -Recurse -Force; echo "pwned'
    WindowsToastNotifier().notify("t", tricky)

    args = captured["args"]
    assert args[args.index("-Message") + 1] == tricky


def test_notify_raises_notifier_error_on_nonzero_exit(monkeypatch):
    def fake_run(args, **kwargs):
        return _FakeCompletedProcess(returncode=1, stderr="toast failed")

    monkeypatch.setattr(notifier_module.subprocess, "run", fake_run)

    with pytest.raises(NotifierError, match="toast failed"):
        WindowsToastNotifier().notify("t", "m")


def test_notify_raises_notifier_error_on_subprocess_failure(monkeypatch):
    def fake_run(args, **kwargs):
        raise OSError("powershell.exe not found")

    monkeypatch.setattr(notifier_module.subprocess, "run", fake_run)

    with pytest.raises(NotifierError, match="powershell.exe not found"):
        WindowsToastNotifier().notify("t", "m")


def test_notify_raises_notifier_error_on_timeout(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="powershell.exe", timeout=15)

    monkeypatch.setattr(notifier_module.subprocess, "run", fake_run)

    with pytest.raises(NotifierError):
        WindowsToastNotifier().notify("t", "m")


def test_console_notifier_prints_title_and_message(capsys):
    ConsoleNotifier().notify("Reminder", "walk the dog")

    out = capsys.readouterr().out
    assert "Reminder" in out
    assert "walk the dog" in out
