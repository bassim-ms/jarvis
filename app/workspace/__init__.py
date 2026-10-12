"""
Workspace Files (Automation Slice A1): Jarvis can list and read small text files in ONE folder and create new ones there.

    base.py      Workspace: the whole policy - safe names, safe content, one folder, create-only, bounds, read-back verification
    tools.py     list_workspace_files / read_workspace_file (READ) and write_workspace_file (SAFE_ACTION, create-only, not automatable)
    factory.py   the default Workspace, from config

It adds no store, no planner, no executor and no permission rule: the tools are ordinary tools on the ordinary ToolRegistry, so the existing
Job planner, plan validator, JobEngine and permission levels remain the only authority over whether and when they run. It touches no GUI,
keyboard, mouse or process, and never deletes, renames or overwrites anything.
"""
