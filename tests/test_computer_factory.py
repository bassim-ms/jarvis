"""
Test for app/computer/factory.py: proves `create_default_computer_backend`
builds a real WindowsComputerBackend. Construction alone performs no OS
interaction (WindowsComputerBackend has no `__init__` side effects), so
this is safe to run with no mocking.
"""

from app.computer.factory import create_default_computer_backend
from app.computer.windows_backend import WindowsComputerBackend


def test_create_default_computer_backend_returns_a_windows_backend():
    backend = create_default_computer_backend()

    assert isinstance(backend, WindowsComputerBackend)
