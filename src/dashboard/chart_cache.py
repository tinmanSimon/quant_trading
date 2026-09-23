"""Bounded per-session reuse of chart indexes over freshly verified local data."""

from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
from hashlib import sha256

import polars as pl

from data_pipeline import DataPipeline, DataQuery

from .chart_data import ChartSeries


class BoundedChartCache:
    """Cache chart preparation, never the storage integrity check.

    Each lookup reads through the public pipeline API before comparing source
    contents. Replacement, deletion, or corruption therefore cannot be hidden
    by an earlier successful read. Keep one instance in Streamlit session state
    so both entry and byte limits apply separately to each user's charts.
    """

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024, max_entries: int = 4):
        for name, value in (("max_bytes", max_bytes), ("max_entries", max_entries)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, ChartSeries] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def byte_size(self) -> int:
        """Estimated size of retained series, including their display indexes."""
        return sum(series.estimated_size for series in self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def load(
        self, pipeline: DataPipeline, query: DataQuery, *, dataset_id: str | None = None,
        timeframe: str, session_timezone: str = "America/New_York",
        omitted_timestamps: Iterable[datetime] = (),
    ) -> ChartSeries | None:
        """Return a chart series only after verifying its current source bytes."""
        try:
            frame = (pipeline.read(query) if dataset_id is None
                     else pipeline.read_dataset(dataset_id))
            # Explicit dataset reads contain the entire revision. Use microsecond
            # comparisons to preserve exact bounds against millisecond storage.
            timestamp = pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            if query.start is not None:
                frame = frame.filter(timestamp >= pl.lit(query.start, dtype=pl.Datetime("us", "UTC")))
            if query.end is not None:
                frame = frame.filter(timestamp < pl.lit(query.end, dtype=pl.Datetime("us", "UTC")))
            if frame.is_empty():
                self.clear()
                return None

            omissions = tuple(sorted(set(omitted_timestamps)))
            digest = sha256(frame.write_ipc(None).getvalue()).digest()
            key = (str(pipeline.store.data_dir), query, dataset_id, timeframe,
                   session_timezone, omissions, digest)
            if key in self._entries:
                series = self._entries.pop(key)
            else:
                series = ChartSeries(frame, timeframe=timeframe, session_timezone=session_timezone,
                                     omitted_timestamps=omissions)
            if series.estimated_size <= self.max_bytes and self.max_entries:
                self._entries[key] = series
            # Recount sizes so lazily built display indexes, if any, also count
            # against the budget on the next interaction.
            while self._entries and (len(self._entries) > self.max_entries
                                     or self.byte_size > self.max_bytes):
                self._entries.popitem(last=False)
            return series
        except Exception:
            self.clear()
            raise
