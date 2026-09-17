"""The canonical OHLCV schema and its validation rules."""

from __future__ import annotations

import polars as pl

from ..exceptions import (
    DuplicateBarError,
    EmptyDataError,
    InvalidOHLCVError,
    SchemaValidationError,
)


CANONICAL_OHLCV_COLUMNS = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
"""Canonical column order for OHLCV data."""

OHLCV_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
"""Schema every canonical OHLCV frame must satisfy after validation."""

_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
_KEY_COLUMNS = ("symbol", "timestamp")


def validate_ohlcv(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a canonical, valid OHLCV frame or raise a specific domain error.

    The input must contain exactly the canonical columns. Timestamps must be
    timezone-aware UTC datetimes; numeric columns may use any Polars numeric
    type and are normalized to ``Float64``. A frame represents one timeframe,
    so its bar key is ``(symbol, timestamp)``. Output is ordered by
    ``(symbol, timestamp)`` only when the input already follows that order;
    unsorted input is rejected rather than silently rearranged.
    """
    if not isinstance(frame, pl.DataFrame):
        raise SchemaValidationError("OHLCV data must be supplied as a Polars DataFrame.")

    _validate_columns(frame)
    _validate_input_types(frame)

    if frame.is_empty():
        raise EmptyDataError("OHLCV data must contain at least one bar.")

    # Casting datetime precision is otherwise truncating, even with strict=True.
    unit = frame.schema["timestamp"].time_unit
    divisor = {"ms": 1, "us": 1_000, "ns": 1_000_000}[unit]
    if frame.select(((pl.col("timestamp").cast(pl.Int64) % divisor) != 0).any()).item():
        raise SchemaValidationError("timestamp precision would be lost at milliseconds.")
    for column in _NUMERIC_COLUMNS:
        dtype = frame.schema[column]
        if dtype.is_integer() or dtype.is_decimal():
            original = frame[column]
            try:
                restored = original.cast(pl.Float64).cast(dtype, strict=True)
            except pl.exceptions.PolarsError as error:
                raise SchemaValidationError(f"{column} cannot be represented losslessly as Float64.") from error
            if not original.equals(restored):
                raise SchemaValidationError(f"{column} cannot be represented losslessly as Float64.")

    canonical = frame.select(
        [
            pl.col("timestamp")
            .cast(OHLCV_SCHEMA["timestamp"], strict=True)
            .alias("timestamp"),
            pl.col("symbol").cast(pl.String, strict=True).alias("symbol"),
            *[
                pl.col(column)
                .cast(OHLCV_SCHEMA[column], strict=True)
                .alias(column)
                for column in _NUMERIC_COLUMNS
            ],
        ]
    )

    _validate_key_values(canonical)
    _validate_numeric_values(canonical)
    _validate_ohlc_relationships(canonical)
    _validate_no_duplicate_bars(canonical)
    _validate_sort_order(canonical)
    return canonical


def _validate_columns(frame: pl.DataFrame) -> None:
    columns = set(frame.columns)
    expected = set(CANONICAL_OHLCV_COLUMNS)
    missing = sorted(expected - columns)
    extra = sorted(columns - expected)

    if missing:
        raise SchemaValidationError(
            f"OHLCV frame is missing required columns: {', '.join(missing)}."
        )
    if extra:
        raise SchemaValidationError(
            f"OHLCV frame has unsupported columns: {', '.join(extra)}."
        )


def _validate_input_types(frame: pl.DataFrame) -> None:
    timestamp_dtype = frame.schema["timestamp"]
    if not isinstance(timestamp_dtype, pl.Datetime):
        raise SchemaValidationError("timestamp must use a timezone-aware datetime type.")
    if timestamp_dtype.time_zone != "UTC":
        raise SchemaValidationError("timestamp must use the UTC time zone.")

    if frame.schema["symbol"] != pl.String:
        raise SchemaValidationError("symbol must use Polars String type.")

    for column in _NUMERIC_COLUMNS:
        if not frame.schema[column].is_numeric():
            raise SchemaValidationError(f"{column} must use a numeric Polars type.")


def _validate_key_values(frame: pl.DataFrame) -> None:
    has_invalid_key = frame.select(
        (
            pl.col("timestamp").is_null()
            | pl.col("symbol").is_null()
            | (pl.col("symbol").str.strip_chars().str.len_chars() == 0)
            | (pl.col("symbol") != pl.col("symbol").str.strip_chars())
        )
        .any()
        .alias("has_invalid_key")
    ).item()
    if has_invalid_key:
        raise InvalidOHLCVError("timestamp and symbol must both be present and non-empty.")


def _validate_numeric_values(frame: pl.DataFrame) -> None:
    invalid_numeric_value = pl.any_horizontal(
        *[
            pl.col(column).is_null() | ~pl.col(column).is_finite()
            for column in _NUMERIC_COLUMNS
        ]
    )
    if frame.select(invalid_numeric_value.any()).item():
        raise InvalidOHLCVError("OHLCV numeric values must be finite and non-null.")

    if frame.select((pl.col("volume") < 0).any()).item():
        raise InvalidOHLCVError("volume must be greater than or equal to zero.")


def _validate_ohlc_relationships(frame: pl.DataFrame) -> None:
    invalid_bar = (
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    )
    if frame.select(invalid_bar.any()).item():
        raise InvalidOHLCVError(
            "Each bar must have positive prices and satisfy "
            "low <= open/close <= high."
        )


def _validate_no_duplicate_bars(frame: pl.DataFrame) -> None:
    has_duplicate = frame.select(
        pl.struct(list(_KEY_COLUMNS)).is_duplicated().any().alias("has_duplicate")
    ).item()
    if has_duplicate:
        raise DuplicateBarError("OHLCV data contains duplicate (symbol, timestamp) bars.")


def _validate_sort_order(frame: pl.DataFrame) -> None:
    if not frame.equals(frame.sort(list(_KEY_COLUMNS))):
        raise InvalidOHLCVError(
            "OHLCV rows must be ordered by symbol and then timestamp."
        )
