"""
Test for app/computer/factory.py: proves `create_default_computer_backend` builds a real WindowsComputerBackend - which enforces the Computer Input Gate
(app/computer/gate.py) ITSELF, using the process-wide gate. Construction alone performs no OS interaction (WindowsComputerBackend has no `__init__`
side effects beyond picking its gate), so this is safe to run with no mocking.
"""

from app.computer.factory import create_default_computer_backend
from app.computer.gate import GatedComputerBackend, default_input_gate
from app.computer.windows_backend import WindowsComputerBackend


def test_create_default_computer_backend_returns_a_windows_backend_that_gates_itself():
    backend = create_default_computer_backend()

    assert isinstance(backend, WindowsComputerBackend) and not isinstance(backend, GatedComputerBackend)  # not wrapped: wrapping would count twice
    assert backend.gate is default_input_gate()  # the one process-wide gate, so the limit is one shared budget
