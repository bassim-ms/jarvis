"""
Builds the application's default ComputerBackend - the one place
construction happens, same role app/calendar/factory.py and
app/search/factory.py play for their own backends.

Deliberately NOT called from app/tools/defaults.py yet (see
app/computer/tools.py's own module docstring) - a caller that wants to
exercise the real backend (e.g. a future milestone's registration code,
or a manual smoke test) uses this function directly.

THE COMPUTER INPUT GATE (app/computer/gate.py) IS ENFORCED BY THE REAL BACKEND ITSELF, not by this factory: `WindowsComputerBackend` admits each of its
nine acting methods (type_text, press_key, move_mouse, click_mouse, double_click, scroll, open_application, open_url, switch_window) through the process-wide gate - off by default, one
shared rate limit, and an authorization grant - immediately before the OS call. So the real backend is gated however it is constructed; nothing depends on this
function being the one that built it.

The DEFAULT backend carries no grant, so it admits no input at all, switch or no switch. Only an explicit `create_authorized_computer_backend(grant)` gives one
backend instance the authority of one grant; nothing else in the process gains it.
"""

from app.computer.authority import InputGrant
from app.computer.base import ComputerBackend
from app.computer.windows_backend import WindowsComputerBackend


def create_default_computer_backend() -> ComputerBackend:
    return WindowsComputerBackend()


def create_authorized_computer_backend(authority: InputGrant) -> ComputerBackend:
    """A NEW backend instance that carries `authority` (and enforces it at the process-wide gate). The grant stays with this instance; pass it nowhere else."""
    if not isinstance(authority, InputGrant):
        raise TypeError("an authorized backend needs an InputGrant")
    return WindowsComputerBackend(authority=authority)
