"""Explicit instrument calendars and completed bar intervals."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import exchange_calendars as xcals
import pandas as pd

from .errors import ResearchError


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
    if timeframe not in {"1h", "1d"}:
        raise ResearchError("Backtesting currently supports 1h and 1d bars.")
    if type(lookback) is not int or lookback < 0:
        raise ResearchError("lookback must be a nonnegative integer.")
    padding = max(30, lookback * 3 + 14)
    schedule_start = start.date() - timedelta(days=padding)
    schedule_end = end.date() + timedelta(days=2)
    calendar = xcals.get_calendar(instrument.calendar, start=str(schedule_start), end=str(schedule_end))
    bars = []
    for label, session in calendar.schedule.iterrows():
        opened = session["open"].to_pydatetime().astimezone(UTC)
        closed = session["close"].to_pydatetime().astimezone(UTC)
        if timeframe == "1d":
            key = datetime.combine(label.date(), datetime.min.time(), UTC)
            bars.append(ExpectedBar(key, opened, closed))
        else:
            if pd.notna(session.get("break_start")):
                raise ResearchError("Hourly backtests require a continuous session calendar without a lunch break.")
            stamp = opened
            while stamp < closed:
                bars.append(ExpectedBar(stamp, stamp, min(stamp + timedelta(hours=1), closed)))
                stamp += timedelta(hours=1)
    requested = [bar for bar in bars if start <= bar.timestamp < end]
    if not requested:
        raise ResearchError("The requested interval contains no scheduled trading bars.")
    previous = [bar for bar in bars if bar.timestamp < requested[0].timestamp]
    if len(previous) < lookback:
        raise ResearchError("The calendar cannot supply the required warm-up history.")
    return tuple((previous[-lookback:] if lookback else []) + requested)
