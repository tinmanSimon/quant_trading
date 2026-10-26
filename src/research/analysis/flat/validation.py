"""Validate completed single-series bars and slice before examining prices."""

from datetime import date, datetime

import numpy as np
import polars as pl


def prepare(frame: pl.DataFrame, timestamp_col: str, high_col: str, low_col: str,
            as_of: datetime | date | None) -> tuple[list, np.ndarray, np.ndarray]:
    if not isinstance(frame, pl.DataFrame):
        raise ValueError("Expected a Polars DataFrame")
    if any(name not in frame.columns for name in (timestamp_col, high_col, low_col)):
        raise ValueError(f"Dataframe requires {timestamp_col!r}, {high_col!r}, and {low_col!r} columns")
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
    for name in ("symbol", "ticker", "instrument", "timeframe"):
        if name in frame.columns and frame[name].n_unique() > 1:
            raise ValueError(f"Mixed {name} values are not a single time series")
    def prices(name: str) -> np.ndarray:
        try:
            values = frame[name].cast(pl.Float64, strict=True).to_numpy()
        except (pl.exceptions.PolarsError, TypeError) as exc:
            raise ValueError(f"{name} prices must be numeric") from exc
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{name} prices must be finite and strictly positive")
        return values

    highs, lows = prices(high_col), prices(low_col)
    if (highs < lows).any():
        raise ValueError("Candle highs must be greater than or equal to lows")
    for name in ("open", "close"):
        if name in frame.columns:
            values = prices(name)
            if ((values < lows) | (values > highs)).any():
                raise ValueError(f"Candle {name} must lie between its low and high")
    return frame[timestamp_col].to_list(), highs, lows
