"""
Action Proposal (Productivity Action Loop, Slice 2): "what small, concrete plan should we follow?"

Builds a few honest steps from what Focus decided - the user's own ("Your part: ...") and at most an assistive step of Jarvis's ("My part:
...") - shows them, and waits for a yes. TEMPORARY, IN-MEMORY and bound to one conversation; it creates no Job, runs no tool, calls no model
and writes nothing. Approval only hands the exact, revalidated proposal to Slice 3, which is what may create an execution Job.
"""

from app.action_proposal.base import ActionProposal, ApprovedProposal, ProposalStep, ToolAction
from app.action_proposal.service import ActionProposalService

__all__ = ["ActionProposal", "ActionProposalService", "ApprovedProposal", "ProposalStep", "ToolAction"]
