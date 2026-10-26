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
    canonical = _canonicalize_ohlcv(frame)
    _validate_bar_values(canonical)
    _validate_no_duplicate_bars(canonical)
    _validate_sort_order(canonical)
    return canonical


def validate_ohlcv_structure(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate keys, types and lossless conversion before checking bar values.

    Providers that explicitly exclude invalid bars must still run these checks
    on every row before removing any values. This function alone does not
    certify OHLCV values; returned data must also pass :func:`validate_ohlcv`.
    """
    canonical = _canonicalize_ohlcv(frame)
    _validate_no_duplicate_bars(canonical)
    _validate_sort_order(canonical)
    return canonical


def _canonicalize_ohlcv(frame: pl.DataFrame) -> pl.DataFrame:
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


def bar_value_flags(frame: pl.DataFrame) -> pl.DataFrame:
    # Evaluate all value rules together so a missing close does not hide other
    # bad bars. Comparisons only apply to finite operands; missing/inf/NaN have
    # their own reasons. Keep the valid-data path vectorized.
    rules = {}
    for column in _NUMERIC_COLUMNS:
        value = pl.col(column)
        rules[f"{column} is missing (null)"] = value.is_null()
        rules[f"{column} is not finite"] = value.is_not_null() & ~value.is_finite()
        if column != "volume":
            rules[f"{column} must be greater than zero"] = value.is_finite() & (value <= 0)
    rules["volume must be greater than or equal to zero"] = (
        pl.col("volume").is_finite() & (pl.col("volume") < 0)
    )
    for left, right in (("low", "high"), ("low", "open"), ("open", "high"),
                        ("low", "close"), ("close", "high")):
        rules[f"{left} exceeds {right}"] = (
            pl.col(left).is_finite() & pl.col(right).is_finite() & (pl.col(left) > pl.col(right))
        )
    return frame.select(expr.fill_null(False).alias(reason) for reason, expr in rules.items())


def _validate_bar_values(frame: pl.DataFrame) -> None:
    flags = bar_value_flags(frame)
    invalid = flags.select(pl.any_horizontal(pl.all())).to_series()
    if not invalid.any():
        return
    bad_rows, bad_flags = frame.filter(invalid), flags.filter(invalid)
    counts = ", ".join(f"{reason}: {count}" for reason, count in bad_flags.sum().row(0, named=True).items()
                       if count)
    details = []
    for row, failures in zip(bad_rows.head(10).iter_rows(named=True),
                             bad_flags.head(10).iter_rows(named=True), strict=True):
        values = ", ".join(f"{column}={row[column]!r}" for column in _NUMERIC_COLUMNS)
        reasons = ", ".join(reason for reason, failed in failures.items() if failed)
        details.append(f"{row['symbol']} at {row['timestamp'].isoformat()}: {values}; reasons: {reasons}")
    remaining = bad_rows.height - len(details)
    if remaining:
        details.append(f"... {remaining} more invalid rows (showing first 10)")
    raise InvalidOHLCVError(
        "OHLCV numeric values must be finite and non-null. Each bar must have positive prices "
        "and satisfy low <= open/close <= high; volume must be greater than or equal to zero. "
        f"Found {bad_rows.height} invalid rows (timestamps in UTC). "
        f"Violation counts (a row can violate multiple rules): {counts}. Details: "
        + "; ".join(details)
    )


def _validate_no_duplicate_bars(frame: pl.DataFrame) -> None:
    duplicates = frame.filter(pl.struct(list(_KEY_COLUMNS)).is_duplicated())
    if duplicates.height:
        keys = duplicates.group_by(list(_KEY_COLUMNS), maintain_order=True).len()
        details = [f"{row['symbol']} at {row['timestamp'].isoformat()}: {row['len']} occurrences"
                   for row in keys.head(10).iter_rows(named=True)]
        if keys.height > 10:
            details.append(f"... {keys.height - 10} more duplicate keys")
        raise DuplicateBarError(
            f"OHLCV data contains duplicate (symbol, timestamp) bars: {keys.height} duplicate keys "
            f"across {duplicates.height} rows (timestamps in UTC). " + "; ".join(details)
        )


def _validate_sort_order(frame: pl.DataFrame) -> None:
    if not frame.equals(frame.sort(list(_KEY_COLUMNS))):
        raise InvalidOHLCVError(
            "OHLCV rows must be ordered by symbol and then timestamp."
        )
