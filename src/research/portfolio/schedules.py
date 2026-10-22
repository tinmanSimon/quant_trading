"""Calendar schedules compare known bar boundaries, never future prices."""

from datetime import datetime
from zoneinfo import ZoneInfo

from research.factors.models import ComponentSpec, positive_integer


def validate_schedule(schedule):
    if not isinstance(schedule, ComponentSpec):
        raise ValueError("schedule must be a ComponentSpec.")
    if schedule.version != "1" or schedule.name not in {"every_n_bars", "daily", "weekly", "monthly"}:
        raise ValueError(f"Unknown rebalance schedule: {(schedule.name, schedule.version)}.")
    if schedule.name == "every_n_bars":
        if set(schedule.config) - {"bars"}:
            raise ValueError("every_n_bars only accepts bars configuration.")
        positive_integer(schedule.config.get("bars", 1), "bars")
    elif schedule.config:
        raise ValueError(f"{schedule.name} schedule does not accept configuration.")
    return schedule


def should_rebalance(schedule, *, index, closed_at, next_closed_at, session_timezone="America/New_York"):
    validate_schedule(schedule)
    if type(index) is not int or index < 0:
        raise ValueError("Schedule index must be a nonnegative integer.")
    for name, value in (("closed_at", closed_at), ("next_closed_at", next_closed_at)):
        if value is not None and (not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{name} must be a timezone-aware datetime.")
    if closed_at is None:
        raise ValueError("closed_at is required.")
    if next_closed_at is not None and next_closed_at <= closed_at:
        raise ValueError("next_closed_at must be after closed_at.")
    if schedule.name == "every_n_bars":
        return index % schedule.config.get("bars", 1) == 0
    if next_closed_at is None:
        # A truncated dataset is not evidence of a daily/weekly/monthly close.
        return False
    zone = ZoneInfo(session_timezone)
    current = closed_at.astimezone(zone)
    following = next_closed_at.astimezone(zone)
    if schedule.name == "daily":
        return current.date() != following.date()
    if schedule.name == "weekly":
        return current.isocalendar()[:2] != following.isocalendar()[:2]
    return (current.year, current.month) != (following.year, following.month)
