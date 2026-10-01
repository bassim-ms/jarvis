"""
Pure scheduling foundation (no clock, no I/O, no scheduler): time zones, the Schedule value type and its occurrence calculation,
and run planning / duplicate identity. Later slices (reminders, automations) build on it; nothing here is wired into them yet.
See timezones.py, schedule.py and runs.py for the rules.
"""
