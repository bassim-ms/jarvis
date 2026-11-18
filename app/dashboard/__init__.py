"""
The local web dashboard (Dashboard Slice 1): Jarvis's main interface in a browser, served by the existing FastAPI app on this
computer only (127.0.0.1 - the app-wide loopback Host guard applies to every route here too).

    GET  /dashboard                      the page (static HTML/CSS/JS from app/dashboard/static, no external resources)
    GET  /api/overview                   health, provider chain + key status, counts
    GET  /api/tasks?status=todo|done|cancelled|all
    POST /api/tasks                      create      - through the EXISTING create_task tool (ToolRegistry permission check)
    POST /api/tasks/{id}/complete|cancel             - through the EXISTING complete_task / cancel_task tools
    GET  /api/reminders?status=active|all
    GET  /api/notes, /api/notes/{id}
    GET  /api/jobs                       recent goal-driven Jobs and their outcome
    GET  /api/interactions               questions / confirmations waiting for the user (answered through ordinary chat:
                                         "answer <code>: ..." - the existing InteractionResolver)
    GET  /api/conversations/{id}/messages   a conversation's history, to restore the chat panel

Chat itself uses the existing POST /chat. Reading the dashboard never calls an AI provider; only sending a chat message does.
"""

from app.dashboard.api import DashboardSources, build_dashboard_router

__all__ = ["DashboardSources", "build_dashboard_router"]
