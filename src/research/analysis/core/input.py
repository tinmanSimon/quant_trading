"""Conservative completion bounds for chart bars without a vendor close field.

Session labels use the supplied session timezone. In the absence of an explicit exchange
schedule, a calendar bar is eligible only once its whole labelled period ends.
"""

from datetime import UTC, datetime, time, timedelta
import re
from zoneinfo import ZoneInfo

import polars as pl


def _completion_bound(timeframe, session_timezone):
    match = re.fullmatch(r"([1-9][0-9]*)(m|h|d|wk|mo)", timeframe)
    if not match:
        raise ValueError("Unsupported analysis bar timeframe.")
    count, unit = int(match[1]), match[2]
    zone = ZoneInfo(session_timezone)
    def close_time(stamp):
        if unit in ("m", "h"):
            if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("Intraday analysis timestamps must be timezone-aware datetimes.")
            close = stamp + timedelta(minutes=count * (60 if unit == "h" else 1))
        else:
            label = stamp.date() if isinstance(stamp, datetime) else stamp
            if unit == "mo":
                month = label.year * 12 + label.month - 1 + count
                close = datetime(month // 12, month % 12 + 1, 1, tzinfo=zone)
            else:
                if unit == "wk":
                    label -= timedelta(days=label.weekday())
                end = label + timedelta(days=count * (7 if unit == "wk" else 1))
                close = datetime.combine(end, time.min, zone)
        return close.astimezone(UTC)
    return close_time


def bar_available_at(timestamp, *, timeframe: str,
                     session_timezone: str = "America/New_York") -> datetime:
    """Return the conservative completion instant used by analysis inputs."""
    return _completion_bound(timeframe, session_timezone)(timestamp)


def completed_prefix(frame: pl.DataFrame, *, timeframe: str, cutoff: datetime,
                     session_timezone: str = "America/New_York") -> pl.DataFrame:
    """Exclude a possibly incomplete suffix, preserving source-bar ordinals.

    Intraday bars wait their full nominal duration, including shortened final
    session bars. Daily/weekly/monthly labels wait until the next labelled
    calendar period. No missing bars are synthesized.
    """
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("The analysis cutoff must be timezone-aware.")
    cutoff = cutoff.astimezone(UTC)
    close_time = _completion_bound(timeframe, session_timezone)

    # The analysis context validates increasing timestamps. Completion bounds are also
    # monotone, so large histories need only logarithmically many conversions.
    stamps = frame["timestamp"]
    lower, upper = 0, frame.height
    while lower < upper:
        middle = (lower + upper) // 2
        if close_time(stamps[middle]) <= cutoff:
            lower = middle + 1
        else:
            upper = middle
    return frame.head(lower)
