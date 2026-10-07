"""
Action Proposal (Productivity Action Loop, Slice 2): "what small, concrete plan should we follow?"

Builds a few honest steps from what Focus decided - the user's own ("Your part: ...") and at most an assistive step of Jarvis's ("My part:
...") - shows them, and waits for a yes. TEMPORARY, IN-MEMORY and bound to one conversation; until a yes it creates no Job, runs no tool, calls no
model and writes nothing. A valid approval hands the exact, revalidated proposal to `ExecutionHandoff` (Slice 3), which starts ONE Job born
RUNNING with exactly the approved steps; the existing JobEngine runs it, under the existing permission rules, with no re-planning.
Slice 4 (`ResultReports`) then tells the user how that Job really turned out - verified from its stored state, once, in the next reply in
the same conversation - and stops.
"""

from app.action_proposal.base import ActionProposal, ApprovedProposal, ProposalStep, ToolAction
from app.action_proposal.handoff import ExecutionHandoff
from app.action_proposal.result import ResultReports
from app.action_proposal.service import ActionProposalService

__all__ = ["ActionProposal", "ActionProposalService", "ApprovedProposal", "ExecutionHandoff", "ProposalStep", "ResultReports", "ToolAction"]
