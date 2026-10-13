"""
Suite-wide protection against REAL OS input (see tests/fake_input_library.py for the why and the how).

The fake input library and the tripwires are installed when this file is imported - before pytest imports any test module and therefore before anything can import the
real `pyautogui`. pytest_configure checks that everything is in place and refuses to run a single test otherwise.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fake_input_library  # noqa: E402

fake_input_library.install()

import pytest  # noqa: E402


def pytest_configure(config):
    fake_input_library.install()
    report = fake_input_library.isolation_report()
    if not report["all_ok"]:
        raise pytest.UsageError(f"the OS-input isolation is NOT in place, so no test will run: {report}")


@pytest.fixture(autouse=True)
def _fresh_fake_input_recorder():
    """Every test starts with an empty record of what reached the fake input library."""
    fake_input_library.FAKE.calls.clear()
    yield
