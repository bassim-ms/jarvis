"""
Reliability and observability (spec section 20): database integrity checks (457), service health checks (458), resource
monitoring (452), verified backups and backup verification (456), and one reliability report (460, partial).

Everything here READS, except `create_backup`, which only writes new files into a fresh backup folder (never touching the
originals, never deleting anything). Databases are opened read-only (`mode=ro`) for checking and copied with SQLite's own backup
API, so a database Jarvis is writing to at that moment is still copied consistently.
"""

from app.diagnostics.backup import BackupError, create_backup, list_backups, verify_backup
from app.diagnostics.checks import DatabaseCheck, check_database
from app.diagnostics.service import Diagnostics

__all__ = ["BackupError", "DatabaseCheck", "Diagnostics", "check_database", "create_backup", "list_backups", "verify_backup"]
