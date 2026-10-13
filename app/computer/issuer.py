"""
The ONLY place computer-input authorization grants are minted.

A grant (app/computer/authority.py) is registered with the gate (app/computer/gate.py) here and nowhere else. Today exactly one caller may import this module: the supervised
Notepad demo's explicit command line (its `--run` option), where the person typing that command is the authorization. Chat, Core, tools, Jobs,
schedulers, automations, action proposals and `app/main.py` must not import it: a static test lists every importer and fails on any other.

Minting needs the global switch ON (a grant cannot be created while input is switched off), a short fixed-vocabulary `purpose` and `actor`, a lifetime of at most
MAX_GRANT_SECONDS, and an ordered list of exact `InputStep`s. The caller must then PASS the returned grant to the one backend that should use it, and revoke it when done.

NOT A SECURITY BOUNDARY against code inside this process (which could call the gate's internals directly); the import test catches accidental drift only.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.computer.authority import MAX_GRANT_ENTRIES, MAX_GRANT_SECONDS, AuthorityError, InputGrant, InputStep, check_label
from app.computer.gate import InputGate, default_input_gate


class GrantIssuer:
    def __init__(self, gate: InputGate):
        self._gate = gate

    @property
    def gate(self) -> InputGate:
        return self._gate

    def issue(self, steps: Sequence[InputStep], *, ttl_seconds: float, purpose: str, actor: str) -> InputGrant:
        """A new grant for exactly these ordered steps, valid for `ttl_seconds` (at most MAX_GRANT_SECONDS) on the gate's monotonic clock. Raises AuthorityError for an
        unusable request and ComputerControlError while input is switched off; in both cases nothing was registered."""
        check_label(purpose, "the purpose")
        check_label(actor, "the actor")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or not 0 < ttl_seconds <= MAX_GRANT_SECONDS:
            raise AuthorityError(f"a grant must last more than 0 and at most {MAX_GRANT_SECONDS:g} seconds.")
        steps = tuple(steps)
        if not steps or len(steps) > MAX_GRANT_ENTRIES or not all(isinstance(step, InputStep) for step in steps):
            raise AuthorityError(f"a grant must list between 1 and {MAX_GRANT_ENTRIES} input steps.")
        grant = InputGrant()
        self._gate._register_grant(grant, steps, ttl_seconds=ttl_seconds, purpose=purpose, actor=actor)
        return grant


def default_issuer() -> GrantIssuer:
    """The issuer for the process-wide gate - the same gate the default backend enforces."""
    return GrantIssuer(default_input_gate())
