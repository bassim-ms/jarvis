"""
READ-ONLY access to the text of the visible Notepad editor, for the Notepad demo's pre-save check. Private to `app/demos`: nothing else imports it.

    the window in front  ->  must belong to notepad.exe  ->  its ONE visible RichEditD2DPT editor  ->  its text, read with a bounded window message

Why this exists: `type_text` reporting success only means the keystrokes were queued, not that Notepad kept them (the first two live runs lost characters
in Notepad's editor). Reading the document back BEFORE anything is saved lets the demo stop on a wrong note instead of saving it.

What it does and does not do:
  * It SENDS NO INPUT and changes nothing: no keystroke, no mouse, no focus change, no window message other than the two READ messages WM_GETTEXTLENGTH and
    WM_GETTEXT. A test pins the exact set of OS calls this file may make.
  * Every message goes through SendMessageTimeout with SMTO_ABORTIFHUNG and a 500 ms limit, so a hung or busy editor fails the check instead of hanging the
    demo. A control that is missing, unreadable, ambiguous or that does not answer is a FAILURE (`DocumentReadError`), never a guess.
  * It picks the editor by what it IS, not by where it appears: only a RichEditD2DPT that is VISIBLE, inside the window in front, whose process is
    notepad.exe. Notepad keeps one editor per tab and the hidden tabs still hold their old text (the probe found the previous run's note in one), so the
    first RichEditD2DPT is never taken; two visible candidates are refused rather than guessed between.
  * It returns the text to its caller and nowhere else: it logs nothing and its errors never contain document text.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Protocol

NOTEPAD_PROCESS = "notepad.exe"
EDITOR_CLASS = "RichEditD2DPT"
READ_TIMEOUT_MS = 500
MAX_DOCUMENT_CHARS = 4096  # far more than the demo's note; a bigger document is not what the demo typed

# The ONLY two window messages this module sends. Both only read.
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
SMTO_ABORTIFHUNG = 0x0002


class DocumentReadError(Exception):
    """The document could not be read safely. The message says why, in plain words, and never contains document text."""


class WindowApi(Protocol):
    """The few read-only window questions the reader asks. The real one is `Win32WindowApi`; tests supply a fake, so no test ever touches a desktop."""

    def foreground_window(self) -> int: ...

    def process_name(self, hwnd: int) -> str: ...

    def child_windows(self, hwnd: int) -> list[int]: ...

    def class_name(self, hwnd: int) -> str: ...

    def is_visible(self, hwnd: int) -> bool: ...

    def text_length(self, hwnd: int, timeout_ms: int) -> int | None:
        """The editor's text length in characters, or None if it did not answer in time / the call failed."""

    def text(self, hwnd: int, length: int, timeout_ms: int) -> str | None:
        """Exactly `length` characters of the editor's text, or None if it did not answer in time / the call failed / the text changed under the read."""


class Win32WindowApi:
    """The real, read-only window API (Windows only). It is built and used only when the demo really runs."""

    def __init__(self) -> None:
        import win32gui

        self._gui = win32gui
        user32 = ctypes.WinDLL("user32", use_last_error=True)  # a private handle: its argument types are not shared with any other code
        send = user32.SendMessageTimeoutW
        send.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM, wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
        send.restype = ctypes.c_size_t
        self._send = send

    def foreground_window(self) -> int:
        return int(self._gui.GetForegroundWindow() or 0)

    def process_name(self, hwnd: int) -> str:
        from app.computer.windows_backend import WindowsComputerBackend

        return str(WindowsComputerBackend._resolve_process_name(hwnd))

    def child_windows(self, hwnd: int) -> list[int]:
        found: list[int] = []
        self._gui.EnumChildWindows(hwnd, lambda child, _extra: found.append(child) or True, None)
        return found

    def class_name(self, hwnd: int) -> str:
        return self._gui.GetClassName(hwnd)

    def is_visible(self, hwnd: int) -> bool:
        return bool(self._gui.IsWindowVisible(hwnd))

    def _ask(self, hwnd: int, message: int, wparam: int, lparam: int, timeout_ms: int) -> int | None:
        result = ctypes.c_size_t(0)
        answered = self._send(hwnd, message, wparam, lparam, SMTO_ABORTIFHUNG, timeout_ms, ctypes.byref(result))
        return result.value if answered else None

    def text_length(self, hwnd: int, timeout_ms: int) -> int | None:
        return self._ask(hwnd, WM_GETTEXTLENGTH, 0, 0, timeout_ms)

    def text(self, hwnd: int, length: int, timeout_ms: int) -> str | None:
        buffer = ctypes.create_unicode_buffer(length + 1)
        copied = self._ask(hwnd, WM_GETTEXT, length + 1, ctypes.addressof(buffer), timeout_ms)
        if copied is None or copied != length:
            return None  # no answer in time, or the document changed between the two reads
        return buffer.value


def read_visible_editor_text(api: WindowApi | None = None) -> str:
    """The text of the visible Notepad editor in the window in front, or DocumentReadError. Sends no input; every read is bounded."""
    try:
        return _read(api if api is not None else Win32WindowApi())
    except DocumentReadError:
        raise
    except Exception as exc:  # noqa: BLE001 - any API error is a failed check; the exception's own text could carry window content, so only its type is kept
        raise DocumentReadError(f"the Notepad window could not be inspected ({exc.__class__.__name__})") from None


def _read(api: WindowApi) -> str:
    window = api.foreground_window()
    if not window:
        raise DocumentReadError("no window is in front")
    if api.process_name(window).lower() != NOTEPAD_PROCESS:
        raise DocumentReadError(f"the window in front does not belong to {NOTEPAD_PROCESS}")
    editors = [child for child in api.child_windows(window) if api.class_name(child) == EDITOR_CLASS and api.is_visible(child)]
    if not editors:
        raise DocumentReadError("no visible Notepad editor control was found in the window in front")
    if len(editors) > 1:
        raise DocumentReadError(f"{len(editors)} visible editor controls were found; refusing to guess which one is the document")
    editor = editors[0]
    length = api.text_length(editor, READ_TIMEOUT_MS)
    if length is None:
        raise DocumentReadError(f"the editor did not answer within {READ_TIMEOUT_MS} ms (it may be busy or hung)")
    if length > MAX_DOCUMENT_CHARS:
        raise DocumentReadError(f"the document is {length} characters, far more than the note")
    text = api.text(editor, length, READ_TIMEOUT_MS)
    if text is None:
        raise DocumentReadError(f"the editor's text could not be read within {READ_TIMEOUT_MS} ms (or it changed while being read)")
    return text
