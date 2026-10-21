"""Causal input slicing without fetching, filling, sorting, or mutation."""

from datetime import date, datetime

import numpy as np
import polars as pl

from .models import TrendConfig


def window_start(length: int, history: int) -> int:
    """Start of the active 2H window, advancing H only after its last bar.

    Length is the exclusive endpoint in the original selected history. The
    first window starts at zero; subsequent windows retain H bars of overlap.
    """
    return max(0, (length - 1) // history - 1) * history


def prepare(frame: pl.DataFrame, timestamp_col: str, price_col: str,
            as_of: datetime | date | None, config: TrendConfig, *,
            truncate: bool = True) -> tuple[list, np.ndarray, int]:
    if not isinstance(frame, pl.DataFrame):
        raise ValueError("Expected a Polars DataFrame")
    if timestamp_col not in frame.columns or price_col not in frame.columns:
        raise ValueError(f"Dataframe requires {timestamp_col!r} and {price_col!r} columns")
    if frame.schema[timestamp_col].base_type() not in (pl.Date, pl.Datetime):
        raise ValueError("Timestamps must have a Date or Datetime dtype")
    times = frame[timestamp_col]
    if times.null_count() or not times.is_sorted() or times.n_unique() != len(times):
        raise ValueError("Timestamps must be non-null, unique, and strictly increasing")
    if as_of is not None:
        if not isinstance(as_of, (date, datetime)):
            raise ValueError("as_of must be a date or datetime")
        try:
            frame = frame.filter(pl.col(timestamp_col) <= pl.lit(as_of))
        except (pl.exceptions.PolarsError, TypeError) as exc:
            raise ValueError("as_of must use a timezone compatible with the timestamps") from exc
    # Only eligible observations determine instrument identity and valid prices.
    for name in ("symbol", "ticker", "instrument", "timeframe"):
        if name in frame.columns and frame[name].n_unique() > 1:
            raise ValueError(f"Mixed {name} values are not a single time series")
    try:
        values = frame[price_col].cast(pl.Float64, strict=True).to_numpy()
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise ValueError("Prices must be numeric") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and strictly positive")
    offset = window_start(frame.height, config.max_history) if truncate else 0
    return frame[timestamp_col].slice(offset).to_list(), np.log(values[offset:]), offset
