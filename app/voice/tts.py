"""
Text-to-speech backed by Windows SAPI (`System.Speech.Synthesis`), raised
through `powershell.exe -File <script>` - the exact same free, local,
no-new-heavy-dependency technique app/reminders/notifier.py's
WindowsToastNotifier already established for native Windows Runtime
toasts. No pip package, no model download, no account of any kind.

Uses `subprocess.Popen` (not `.run`) specifically so `stop()` can
terminate an in-progress utterance - the "user interruption where
reasonably possible" this milestone asks for, without building full
barge-in (listening while speaking).
"""

import subprocess
import tempfile

from app.voice.base import TextToSpeechError, TextToSpeechProvider

# `-File` (not `-Command` with trailing args) binds -Text/-VoiceName/-Rate
# to this script's own param() block by name. Values reach PowerShell as
# separate argv entries (see WindowsSapiTTSProvider.speak building an
# argument list, never a shell string) and are only ever used here as
# property values/Speak() arguments - never interpolated into executable
# script text - so arbitrary reply text can't inject PowerShell code,
# the same reasoning app/reminders/notifier.py already documents.
_SPEAK_SCRIPT = """
param([string]$Text, [string]$VoiceName, [int]$Rate)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($VoiceName) {
    try { $synth.SelectVoice($VoiceName) } catch { }
}
$synth.Rate = $Rate
$synth.Speak($Text)
"""


class WindowsSapiTTSProvider(TextToSpeechProvider):
    """TextToSpeechProvider implementation backed by Windows SAPI."""

    def __init__(self, voice_name: str = "", rate: int = 0, timeout: float = 60.0):
        self.voice_name = voice_name
        self.rate = rate
        self.timeout = timeout
        self._script_path = self._write_script()
        self._process: subprocess.Popen | None = None

    @staticmethod
    def _write_script() -> str:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ps1",
            prefix="jarvis_speak_",
            delete=False,
            encoding="utf-8",
        )
        try:
            handle.write(_SPEAK_SCRIPT)
        finally:
            handle.close()
        return handle.name

    def speak(self, text: str) -> None:
        try:
            self._process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-WindowStyle",
                    "Hidden",
                    "-File",
                    self._script_path,
                    "-Text",
                    text,
                    "-VoiceName",
                    self.voice_name,
                    "-Rate",
                    str(self.rate),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            self._process = None
            raise TextToSpeechError(f"Failed to run powershell.exe: {exc}") from exc

        process = self._process
        try:
            _stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            self._process = None
            raise TextToSpeechError("Timed out waiting for speech synthesis.") from exc
        self._process = None

        # A terminated process (see stop()) exits with a negative
        # returncode on POSIX or a nonzero one on Windows - either way,
        # an interruption is not an error worth surfacing to the caller.
        if process.returncode != 0 and stderr:
            raise TextToSpeechError(
                f"powershell.exe exited {process.returncode}: {stderr.strip()}"
            )

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
