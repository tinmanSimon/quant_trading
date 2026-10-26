"""Causal bounded-wick periods and a latest-first nonoverlapping summary."""

from datetime import date, datetime
import math
from typing import Callable, Iterable, Iterator

import polars as pl

from .models import FlatAnalysisResult, FlatConfig, FlatPeriod
from .validation import prepare
from .windows import EnvelopeWindows, envelope


Progress = Callable[[int, int], None]


def _config(config: FlatConfig | None) -> FlatConfig:
    if config is None:
        return FlatConfig()
    if not isinstance(config, FlatConfig):
        raise ValueError("config must be a FlatConfig")
    return config


def _period(times: list, end: int, windows: EnvelopeWindows) -> FlatPeriod | None:
    if end < 0 or windows.starts[end] < 0:
        return None
    start = int(windows.starts[end])
    lower, upper = float(windows.lowers[end]), float(windows.uppers[end])
    center, width_pct = envelope(lower, upper)
    max_bar = float(windows.max_bar_lengths[end])
    median_bar = float(windows.median_bar_lengths[end])
    return FlatPeriod(
        period_id=f"F{start}-{end}", start_index=start, end_index=end,
        start_time=times[start], end_time=times[end], duration=end - start + 1,
        lower_price=lower, upper_price=upper, center_price=center, range_pct=width_pct,
        max_range_pct=float(windows.limits[end]),
        max_bar_range_pct=max_bar if math.isfinite(max_bar) else None,
        median_bar_range_pct=median_bar if math.isfinite(median_bar) else None,
    )


def _result(times: list, config: FlatConfig, mode: str,
            periods: list[FlatPeriod]) -> FlatAnalysisResult:
    status = ("insufficient_data" if len(times) < config.min_duration
              else "ok" if periods else "no_flat_period")
    return FlatAnalysisResult(
        periods=tuple(periods), config=config, source_count=len(times),
        analysis_start=times[0] if times else None, analysis_end=times[-1] if times else None,
        analysis_mode=mode, status=status,
    )


def analyze_current(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", high_col: str = "high",
                    low_col: str = "low", as_of: datetime | date | None = None,
                    config: FlatConfig | None = None) -> FlatAnalysisResult:
    """Return the longest valid suffix ending at the last eligible candle.

    Every candle high and low must lie in one sufficiently narrow horizontal
    envelope. The caller supplies completed bars; no prices or gaps are filled.
    """
    config = _config(config)
    times, highs, lows = prepare(frame, timestamp_col, high_col, low_col, as_of)
    windows = EnvelopeWindows(highs, lows, config)
    period = _period(times, len(times) - 1, windows)
    return _result(times, config, "current", [period] if period else [])


def analyze_history(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", high_col: str = "high",
                    low_col: str = "low", as_of: datetime | date | None = None,
                    config: FlatConfig | None = None,
                    progress: Progress | None = None) -> FlatAnalysisResult:
    """Keep at most one immutable result per endpoint in ascending order."""
    config = _config(config)
    times, highs, lows = prepare(frame, timestamp_col, high_col, low_col, as_of)
    windows = EnvelopeWindows(highs, lows, config, progress)
    periods = []
    for end in range(config.min_duration - 1, len(times)):
        period = _period(times, end, windows)
        if period is not None:
            periods.append(period)
    return _result(times, config, "history", periods)


def select_periods(periods: Iterable[FlatPeriod]) -> tuple[FlatPeriod, ...]:
    """Latest endpoint first, earliest start on a tie, then strictly before it."""
    selected = []
    cursor = None
    for period in sorted(periods, key=lambda item: (-item.end_index, item.start_index, item.period_id)):
        if cursor is None or period.end_index <= cursor:
            selected.append(period)
            cursor = period.start_index - 1
    return tuple(selected)


def analyze_summary(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", high_col: str = "high",
                    low_col: str = "low", as_of: datetime | date | None = None,
                    config: FlatConfig | None = None,
                    progress: Progress | None = None) -> FlatAnalysisResult:
    """Scan candles once, then select latest-first strictly disjoint periods.

    Period objects are only built for selected rows. The fixed, unfiltered scan
    takes O(N) work; adaptive/filter scans also depend on the duration bounds.
    Every period uses candles through its own endpoint;
    appending newer candles can change which historical periods the summary
    selects, without changing previously evaluated endpoint records.
    """
    config = _config(config)
    times, highs, lows = prepare(frame, timestamp_col, high_col, low_col, as_of)
    windows = EnvelopeWindows(highs, lows, config, progress)
    periods = []
    end = len(times) - 1
    while end >= config.min_duration - 1:
        period = _period(times, end, windows)
        if period is None:
            end -= 1
        else:
            periods.append(period)
            end = period.start_index - 1
    return _result(times, config, "summary", periods)


def analyze(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", high_col: str = "high",
            low_col: str = "low", as_of: datetime | date | None = None,
            config: FlatConfig | None = None,
            progress: Progress | None = None) -> FlatAnalysisResult:
    """Default analysis is the entire supplied history's disjoint summary."""
    return analyze_summary(frame, timestamp_col=timestamp_col, high_col=high_col, low_col=low_col,
                           as_of=as_of, config=config, progress=progress)


def replay(frame: pl.DataFrame, *, as_of_values: Iterable[datetime | date] | None = None,
           step: int = 1, timestamp_col: str = "timestamp", high_col: str = "high",
           low_col: str = "low", config: FlatConfig | None = None) -> Iterator[FlatAnalysisResult]:
    """Yield independent current-endpoint snapshots at the requested cutoffs."""
    config = _config(config)
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    if as_of_values is None:
        if not isinstance(frame, pl.DataFrame) or timestamp_col not in frame.columns:
            raise ValueError(f"Dataframe requires {timestamp_col!r}")
        as_of_values = frame[timestamp_col].to_list()[config.min_duration - 1::step]
    for cutoff in as_of_values:
        yield analyze_current(frame, timestamp_col=timestamp_col, high_col=high_col, low_col=low_col,
                              as_of=cutoff, config=config)
