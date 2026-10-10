"""
Builds the application's default ComputerBackend - the one place
construction happens, same role app/calendar/factory.py and
app/search/factory.py play for their own backends.

Deliberately NOT called from app/tools/defaults.py yet (see
app/computer/tools.py's own module docstring) - a caller that wants to
exercise the real backend (e.g. a future milestone's registration code,
or a manual smoke test) uses this function directly.

THE COMPUTER INPUT GATE (app/computer/gate.py) IS ENFORCED BY THE REAL BACKEND ITSELF, not by this factory: `WindowsComputerBackend` admits each of its
six input methods (type_text, press_key, move_mouse, click_mouse, double_click, open_application) through the process-wide gate - off by default, one
shared rate limit - immediately before the OS call. So the real backend is gated however it is constructed; nothing depends on this function being the
one that built it.
"""

from app.computer.base import ComputerBackend
from app.computer.windows_backend import WindowsComputerBackend


def create_default_computer_backend() -> ComputerBackend:
    return WindowsComputerBackend()
