"""Bounded chart views over a verified, unchanged snapshot of stored OHLCV bars.

Coordinates are source-bar ordinals, including when candles are grouped for
display. Browser navigation therefore does not depend on which slice is loaded.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
import math
import re
from zoneinfo import ZoneInfo

import polars as pl


MAX_WINDOW_CANDLES = 2000


@dataclass(frozen=True)
class ChartWindow:
    frame: pl.DataFrame
    positions: list[float]
    widths: list[float]
    first_indices: list[int]
    last_indices: list[int]
    source_counts: list[int]
    omitted_counts: list[int]
    starts: list[datetime]
    ends: list[datetime]
    view_range: tuple[float, float]
    loaded_range: tuple[float, float]
    resolution: str
    total_count: int


@dataclass(frozen=True)
class _Spans:
    first: tuple[int, ...]
    last: tuple[int, ...]


def _spans(keys) -> _Spans:
    first, last = [], []
    previous = object()
    for index, key in enumerate(keys):
        if key != previous:
            if first:
                last.append(index - 1)
            first.append(index)
            previous = key
    if first:
        last.append(index)
    return _Spans(tuple(first), tuple(last))


def _groups(spans: _Spans, step: int | None, first: int, last: int, *, reverse=False):
    """Yield complete, stable groups intersecting a source-index interval."""
    if first > last:
        return
    lower, upper = bisect_left(spans.last, first), bisect_right(spans.first, last)
    indices = range(upper - 1, lower - 1, -1) if reverse else range(lower, upper)
    for index in indices:
        begin, finish = spans.first[index], spans.last[index]
        if step is None:
            yield begin, finish
            continue
        start_group = max(0, (first - begin) // step)
        end_group = min((finish - begin) // step, (last - begin) // step)
        chunks = (range(end_group, start_group - 1, -1) if reverse
                  else range(start_group, end_group + 1))
        for chunk in chunks:
            group_start = begin + chunk * step
            yield group_start, min(finish, group_start + step - 1)


class ChartSeries:
    """Server-side snapshot; ``window`` returns at most 2,000 display candles.

    Intraday grouping uses local trading dates, retaining extended-hours bars.
    It never crosses a missing interval. Explicit calendar-day/week summaries
    can contain gaps; their source and known-omission counts expose that fact.
    """

    def __init__(self, frame: pl.DataFrame, *, timeframe: str,
                 session_timezone: str = "America/New_York", omitted_timestamps=()):
        if frame.is_empty():
            raise ValueError("A chart requires at least one bar.")
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError("Chart data must contain timestamps and OHLCV columns.")
        if not isinstance(frame.schema["timestamp"], pl.Datetime) or not frame.schema["timestamp"].time_zone:
            raise ValueError("Chart timestamps must be timezone-aware datetimes.")
        self._frame = frame.clone().with_columns(pl.col("timestamp").dt.convert_time_zone("UTC")).sort("timestamp")
        self.timestamps = tuple(self._frame["timestamp"].to_list())
        if any(stamp is None for stamp in self.timestamps):
            raise ValueError("Chart timestamps cannot be null.")
        if any(left >= right for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("Chart bars must have unique timestamps.")
        if "symbol" in self._frame.columns and self._frame["symbol"].n_unique() != 1:
            raise ValueError("A chart series must contain a single symbol.")
        self.timeframe = timeframe
        self.session_timezone = session_timezone
        zone = ZoneInfo(session_timezone)
        date_labels = timeframe.endswith(("d", "wk", "mo"))
        match = re.fullmatch(r"([1-9][0-9]*)(m|h)", timeframe)
        duration = (timedelta(minutes=int(match[1]) * (60 if match[2] == "h" else 1))
                    if match else None)
        if not date_labels and duration is None:
            raise ValueError("Unsupported chart timeframe.")
        self._duration = duration
        self._date_labels = date_labels
        dates = [stamp.date() if date_labels else stamp.astimezone(zone).date() for stamp in self.timestamps]
        self._days = _spans(dates)
        self._day_ordinals = tuple(dates[index].toordinal() for index in self._days.first)
        run_keys, run = [], 0
        for index, stamp in enumerate(self.timestamps):
            if index and (dates[index] != dates[index - 1]
                          or (duration is not None and stamp - self.timestamps[index - 1] > duration)):
                run += 1
            run_keys.append(run)
        self._runs = _spans(run_keys)
        present = set(self.timestamps)
        omissions = set()
        for stamp in omitted_timestamps:
            if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("Omitted chart timestamps must be timezone-aware datetimes.")
            stamp = stamp.astimezone(UTC)
            if stamp not in present:
                omissions.add(stamp)
        self.omitted_timestamps = tuple(sorted(omissions))
        self.total_count = self._frame.height
        # Include the retained Python indices/timestamps in the server LRU budget.
        self.estimated_size = (self._frame.estimated_size() + 64 * self.total_count
                               + 80 * (len(self._days.first) + len(self._runs.first))
                               + 64 * len(self.omitted_timestamps))

    @property
    def frame(self) -> pl.DataFrame:
        """Return a clone so callers cannot mutate the cached snapshot in place."""
        return self._frame.clone()

    def _weekly_spans(self, weeks: int) -> _Spans:
        # Python ordinals start on a Monday, so this anchors every group to a
        # calendar Monday without moving canonical daily labels into another day.
        days = _spans((ordinal - 1) // (7 * weeks) for ordinal in self._day_ordinals)
        return _Spans(tuple(self._days.first[index] for index in days.first),
                      tuple(self._days.last[index] for index in days.last))

    def _grouping(self, first: int, last: int, budget: int):
        if last - first + 1 <= budget:
            return _Spans((0,), (self.total_count - 1,)), 1, f"Original {self.timeframe} bars"
        if self._duration is not None:
            run_count = bisect_right(self._runs.first, last) - bisect_left(self._runs.last, first)
            if run_count <= budget:
                step = 2 ** max(1, math.ceil(math.log2((last - first + 1) / budget)))
                while step * self._duration < timedelta(days=1):
                    count = sum(1 for _ in islice(_groups(self._runs, step, first, last), budget + 1))
                    if count <= budget:
                        return self._runs, step, f"Up to {step} × {self.timeframe} bars (display only; {self.session_timezone} trading dates)"
                    step *= 2
        daily_count = bisect_right(self._days.first, last) - bisect_left(self._days.last, first)
        date_context = "session dates" if self._date_labels else f"{self.session_timezone} calendar days"
        if daily_count <= budget:
            return self._days, None, f"Daily summaries (display only; {date_context})"
        weeks = 1
        while True:
            spans = self._weekly_spans(weeks)
            count = bisect_right(spans.first, last) - bisect_left(spans.last, first)
            if count <= budget:
                label = "Calendar-week" if weeks == 1 else f"{weeks}-calendar-week"
                return spans, None, f"{label} summaries (display only; {date_context}; Monday boundaries)"
            weeks *= 2

    def window(self, start=None, end=None, *, width=1000) -> ChartWindow:
        """Return the visible range with small, bounded buffers on either side."""
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            raise ValueError("Chart width must be a finite positive number.")
        budget = max(200, min(1000, int(width)))
        if start is None and end is None:
            start, end = max(0, self.total_count - 50) - 0.5, self.total_count - 0.5
        else:
            start = -0.5 if start is None else start
            end = self.total_count - 0.5 if end is None else end
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end) or end <= start):
            raise ValueError("Chart range must contain two increasing finite numbers.")
        span = min(end - start, self.total_count)
        start = min(max(float(start), -0.5), self.total_count - 0.5 - span)
        end = start + span
        first = max(0, min(self.total_count - 1, math.floor(start + 0.5)))
        last = max(first, min(self.total_count - 1, math.ceil(end + 0.5) - 1))
        spans, step, resolution = self._grouping(first, last, budget)
        visible = list(_groups(spans, step, first, last))
        buffer = max(100, math.ceil((last - first + 1) / 4))
        extra_budget = (MAX_WINDOW_CANDLES - len(visible)) // 2
        left = list(islice(_groups(spans, step, max(0, first - buffer), visible[0][0] - 1,
                                   reverse=True), extra_budget))
        right = list(islice(_groups(spans, step, visible[-1][1] + 1,
                                    min(self.total_count - 1, last + buffer)), extra_budget))
        groups = list(reversed(left)) + visible + right
        first_indices = [begin for begin, _ in groups]
        last_indices = [finish for _, finish in groups]
        starts = [self.timestamps[index] for index in first_indices]
        ends = [self.timestamps[index] for index in last_indices]
        if step == 1:
            display = self._frame.slice(first_indices[0], last_indices[-1] - first_indices[0] + 1)
        else:
            rows = []
            for begin, finish in groups:
                chunk = self._frame.slice(begin, finish - begin + 1)
                try:
                    volume = math.fsum(chunk["volume"].to_list())
                except OverflowError as exc:
                    raise ValueError("Display aggregate volume must be finite.") from exc
                if not math.isfinite(volume):
                    raise ValueError("Display aggregate volume must be finite.")
                row = chunk.row(0, named=True)
                row.update(high=chunk["high"].max(), low=chunk["low"].min(),
                           close=chunk["close"][-1], volume=volume)
                rows.append(row)
            display = pl.DataFrame(rows, schema=self._frame.schema)
        return ChartWindow(
            frame=display,
            positions=[(begin + finish) / 2 for begin, finish in groups],
            widths=[float(finish - begin + 1) for begin, finish in groups],
            first_indices=first_indices, last_indices=last_indices,
            source_counts=[finish - begin + 1 for begin, finish in groups],
            omitted_counts=[bisect_right(self.omitted_timestamps, finish) - bisect_left(self.omitted_timestamps, begin)
                            for begin, finish in zip(starts, ends)],
            starts=starts, ends=ends, view_range=(start, end),
            loaded_range=(first_indices[0] - 0.5, last_indices[-1] + 0.5),
            resolution=resolution, total_count=self.total_count,
        )
