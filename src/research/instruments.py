"""Explicit instrument calendars and completed bar intervals."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import exchange_calendars as xcals
import pandas as pd

from .errors import ResearchError
from .timeframes import bar_duration, normalize_timeframe


@dataclass(frozen=True)
class Instrument:
    symbol: str
    calendar: str = "XNYS"
    currency: str = "USD"

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ResearchError("An instrument requires a symbol.")
        if not isinstance(self.calendar, str) or not self.calendar:
            raise ResearchError("An explicit exchange calendar is required.")
        if self.currency != "USD":
            raise ResearchError("The initial research engine uses USD accounts; currency conversion is not implemented.")


@dataclass(frozen=True)
class ExpectedBar:
    timestamp: datetime
    open: datetime
    close: datetime


def expected_bars(instrument: Instrument, *, start: datetime, end: datetime,
                  timeframe: str, lookback: int = 0) -> tuple[ExpectedBar, ...]:
    """Return preceding warm-up bars followed by every requested bar.

    Daily keys are session dates at UTC midnight. Intraday keys represent real
    instants, anchored to the session open; early-close final bars can be short.
    """
    timeframe = normalize_timeframe(timeframe)
    is_daily = timeframe == "1d"
    duration = bar_duration(timeframe)
    if type(lookback) is not int or lookback < 0:
        raise ResearchError("lookback must be a nonnegative integer.")
    requested, previous = [], []
    remaining = lookback
    # The calendar API needs date bounds. Read older history in fixed chunks,
    # keeping collected bars instead of expanding and rescanning the same dates.
    history_chunk = timedelta(days=14)
    schedule_start = start.date()
    while not requested or remaining > 0:
        try:
            schedule_end = schedule_start - timedelta(days=1) if requested else end.date() + timedelta(days=2)
            schedule_start -= history_chunk
            calendar = xcals.get_calendar(instrument.calendar, start=str(schedule_start), end=str(schedule_end))
            sessions = calendar.schedule.iloc[::-1].iterrows()
        except xcals.errors.NoSessionsError:
            sessions = ()  # A closure can span a whole chunk; keep looking back.
        except (OverflowError, ValueError) as error:
            raise ResearchError(f"Cannot construct the required trading/warm-up calendar: {error}") from error
        # Walk backward so generating warm-up stops at exactly the needed bar.
        for label, session in sessions:
            opened = session["open"].to_pydatetime().astimezone(UTC)
            closed = session["close"].to_pydatetime().astimezone(UTC)
            if is_daily:
                key = datetime.combine(label.date(), datetime.min.time(), UTC)
            else:
                key = opened
            if key >= end:
                continue
            if not is_daily and pd.notna(session.get("break_start")):
                raise ResearchError("Intraday backtests require a continuous session calendar without a lunch break.")
            count = 1 if is_daily else (closed - opened + duration - timedelta(microseconds=1)) // duration
            for index in range(count - 1, -1, -1):
                stamp = key if is_daily else opened + index * duration
                if stamp >= end:
                    continue

                if is_daily:
                    bar = ExpectedBar(stamp, opened, closed)
                else:
                    bar = ExpectedBar(stamp, stamp, min(stamp + duration, closed))

                if stamp < start:
                    if not requested:
                        raise ResearchError("The requested interval contains no scheduled trading bars.")
                    if remaining > 0:
                        previous.append(bar)
                        remaining -= 1
                    if remaining == 0:
                        return tuple(reversed(previous)) + tuple(reversed(requested))
                else:
                    requested.append(bar)

        if not requested:
            raise ResearchError("The requested interval contains no scheduled trading bars.")
    return tuple(reversed(previous)) + tuple(reversed(requested))
