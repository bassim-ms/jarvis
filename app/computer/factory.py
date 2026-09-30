"""
Builds the application's default ComputerBackend - the one place
construction happens, same role app/calendar/factory.py and
app/search/factory.py play for their own backends.

Deliberately NOT called from app/tools/defaults.py yet (see
app/computer/tools.py's own module docstring) - a caller that wants to
exercise the real backend (e.g. a future milestone's registration code,
or a manual smoke test) uses this function directly.
"""

from app.computer.base import ComputerBackend
from app.computer.windows_backend import WindowsComputerBackend


def create_default_computer_backend() -> ComputerBackend:
    return WindowsComputerBackend()
