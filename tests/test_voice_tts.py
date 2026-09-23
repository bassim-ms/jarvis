"""
Tests for WindowsSapiTTSProvider (app/voice/tts.py). `subprocess.Popen` is
monkeypatched - no real powershell.exe process or speech synthesis is
ever raised, the same convention test_reminders_notifier.py already uses
for WindowsToastNotifier.
"""

import subprocess

import pytest

from app.voice import tts as tts_module
from app.voice.base import TextToSpeechError
from app.voice.tts import WindowsSapiTTSProvider


class FakePopen:
    def __init__(self, args, returncode: int = 0, stderr: str = "", hang: bool = False, **kwargs):
        self.args = args
        self.returncode = returncode
        self._stderr = stderr
        self._hang = hang
        self.terminated = False
        self.killed = False

    def communicate(self, timeout=None):
        if self._hang and not self.terminated and not self.killed:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout)
        return ("", self._stderr)

    def poll(self):
        return None if not (self.terminated or self.killed) else self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_speak_invokes_powershell_with_file_and_named_arguments(monkeypatch):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return FakePopen(args)

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)

    provider = WindowsSapiTTSProvider(voice_name="Zira", rate=2)
    provider.speak("hello there")

    args = captured["args"]
    assert args[0] == "powershell.exe"
    assert "-File" in args
    assert args[args.index("-File") + 1] == provider._script_path
    assert args[args.index("-Text") + 1] == "hello there"
    assert args[args.index("-VoiceName") + 1] == "Zira"
    assert args[args.index("-Rate") + 1] == "2"


def test_speak_passes_text_as_a_literal_argument_not_shell_text(monkeypatch):
    """Reply text can contain quotes/semicolons/backticks - since args are
    a list (never a shell string), that's just literal data to
    powershell.exe, never executable script text."""
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return FakePopen(args)

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)

    tricky = '"; Remove-Item C:\\ -Recurse -Force; echo "pwned'
    WindowsSapiTTSProvider().speak(tricky)

    args = captured["args"]
    assert args[args.index("-Text") + 1] == tricky


def test_speak_raises_on_nonzero_exit_with_stderr(monkeypatch):
    def fake_popen(args, **kwargs):
        return FakePopen(args, returncode=1, stderr="synthesis failed")

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)

    with pytest.raises(TextToSpeechError, match="synthesis failed"):
        WindowsSapiTTSProvider().speak("hello")


def test_speak_raises_on_subprocess_launch_failure(monkeypatch):
    def fake_popen(args, **kwargs):
        raise OSError("powershell.exe not found")

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)

    with pytest.raises(TextToSpeechError, match="powershell.exe not found"):
        WindowsSapiTTSProvider().speak("hello")


def test_speak_raises_and_kills_process_on_timeout(monkeypatch):
    created = {}

    def fake_popen(args, **kwargs):
        proc = FakePopen(args, hang=True)
        created["proc"] = proc
        return proc

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)

    with pytest.raises(TextToSpeechError, match="Timed out"):
        WindowsSapiTTSProvider(timeout=0.01).speak("hello")

    assert created["proc"].killed is True


def test_stop_terminates_an_in_progress_process(monkeypatch):
    """Simulates the one interruption path this milestone supports: a
    Ctrl+C that interrupts speak()'s blocking communicate() call leaves
    `_process` set, and stop() (called from run_forever's cleanup) must
    still be able to terminate it."""
    proc = FakePopen(["powershell.exe"])

    def fake_popen(args, **kwargs):
        return proc

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)

    provider = WindowsSapiTTSProvider()
    provider._process = proc  # simulate speak() having been interrupted mid-call

    provider.stop()

    assert proc.terminated is True


def test_stop_is_a_no_op_when_nothing_is_speaking():
    provider = WindowsSapiTTSProvider()

    provider.stop()  # must not raise
