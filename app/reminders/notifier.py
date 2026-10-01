"""
Notifier abstraction and its Windows implementation (M7).

A Notifier's only job is surfacing a single notification to the user right
now - it never decides *whether* something is due (ReminderScheduler does,
see app/reminders/scheduler.py) and never touches ReminderStore. Kept as
its own small interface, the same "interface + swappable backend" shape as
AIProvider/SearchProvider, so a future notifier (a different OS, a toast
inside a future dashboard, ...) can be swapped in without touching the
scheduler.

WindowsToastNotifier is the only real implementation, and it's free and
fully local: it shells out to `powershell.exe`, which ships with every
supported Windows version, to raise a native Windows Runtime toast via
`Windows.UI.Notifications` - no new pip dependency, no external service, no
API key, matching this codebase's existing "no extra SDK when a standard
tool already does the job" stance (e.g. httpx instead of a vendor AI/search
SDK - see README's Stack section).
"""

import subprocess
import tempfile
from abc import ABC, abstractmethod

# `-File` (not `-Command` with trailing args) is what makes PowerShell bind
# -AppId/-Title/-Message to this script's own param() block by name, rather
# than trying to parse them as extra -Command arguments. Values only ever
# reach PowerShell as separate argv entries (see WindowsToastNotifier.notify
# building an argument list, never a shell string) and are only ever used
# here as CreateTextNode() *data* - never interpolated into executable
# script text - so arbitrary reminder text can't inject PowerShell code.
_TOAST_SCRIPT = """
param([string]$AppId, [string]$Title, [string]$Message)
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$textNodes = $template.GetElementsByTagName("text")
$textNodes.Item(0).AppendChild($template.CreateTextNode($Title)) | Out-Null
$textNodes.Item(1).AppendChild($template.CreateTextNode($Message)) | Out-Null
$toast = New-Object Windows.UI.Notifications.ToastNotification $template
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($AppId).Show($toast)
"""


class NotifierError(RuntimeError):
    """Raised when a Notifier fails to deliver a notification - a missing
    OS feature, a subprocess failure, or the OS itself rejecting it. Never
    silently swallowed here: ReminderScheduler is what decides a failed
    notification is still recorded as a lifecycle event (see
    ReminderStore.record_trigger) rather than retried in a tight loop."""


class Notifier(ABC):
    """Surfaces a single notification to the user right now."""

    @abstractmethod
    def notify(self, title: str, message: str) -> None:
        """Show `title`/`message` to the user, or raise NotifierError."""


class WindowsToastNotifier(Notifier):
    """Notifier backed by a native Windows Runtime toast, raised through
    `powershell.exe -File <script>`. The script is written once (a small
    temp `.ps1` file, cleaned up like any other temp file at process exit)
    and reused for every `notify()` call on this instance."""

    def __init__(self, app_id: str = "Jarvis", timeout: float = 15.0):
        self.app_id = app_id
        self.timeout = timeout
        self._script_path = self._write_script()

    @staticmethod
    def _write_script() -> str:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ps1",
            prefix="jarvis_toast_",
            delete=False,
            encoding="utf-8",
        )
        try:
            handle.write(_TOAST_SCRIPT)
        finally:
            handle.close()
        return handle.name

    def notify(self, title: str, message: str) -> None:
        try:
            result = subprocess.run(
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
                    "-AppId",
                    self.app_id,
                    "-Title",
                    title,
                    "-Message",
                    message,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NotifierError(f"Failed to run powershell.exe: {exc}") from exc

        if result.returncode != 0:
            raise NotifierError(
                f"powershell.exe exited {result.returncode}: {result.stderr.strip()}"
            )


class ConsoleNotifier(Notifier):
    """Fallback Notifier for non-Windows environments (see
    app/reminders/factory.py): prints instead of raising a native toast, so
    Jarvis still runs - just without OS notifications - on a platform
    WindowsToastNotifier doesn't support."""

    def notify(self, title: str, message: str) -> None:
        print(f"[{title}] {message}")
