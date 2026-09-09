"""Tests for canonical OHLCV validation."""

from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import (
    CANONICAL_OHLCV_COLUMNS,
    DuplicateBarError,
    EmptyDataError,
    InvalidOHLCVError,
    OHLCV_SCHEMA,
    SchemaValidationError,
    validate_ohlcv,
)


def test_valid_ohlcv_frame_passes_and_uses_canonical_schema(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    validated = validate_ohlcv(sample_ohlcv_frame)

    assert validated.columns == list(CANONICAL_OHLCV_COLUMNS)
    assert validated.schema == OHLCV_SCHEMA
    assert_frame_equal(validated, sample_ohlcv_frame)


def test_valid_numeric_inputs_are_normalized_to_float64(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    integer_frame = sample_ohlcv_frame.with_columns(
        *[
            pl.col(column).round(0).cast(pl.Int64)
            for column in ("open", "high", "low", "close", "volume")
        ]
    )

    validated = validate_ohlcv(integer_frame)

    assert all(
        validated.schema[column] == pl.Float64
        for column in ("open", "high", "low", "close", "volume")
    )


def test_validator_returns_canonical_column_order(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    shuffled = sample_ohlcv_frame.select(
        "volume", "close", "symbol", "timestamp", "high", "open", "low"
    )

    assert validate_ohlcv(shuffled).columns == list(CANONICAL_OHLCV_COLUMNS)


def test_validator_rejects_missing_required_column(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaValidationError, match="missing required columns: close"):
        validate_ohlcv(sample_ohlcv_frame.drop("close"))


def test_validator_rejects_extra_column(sample_ohlcv_frame: pl.DataFrame) -> None:
    with pytest.raises(SchemaValidationError, match="unsupported columns: vendor"):
        validate_ohlcv(sample_ohlcv_frame.with_columns(pl.lit("yahoo").alias("vendor")))


def test_validator_rejects_non_utc_timestamp(sample_ohlcv_frame: pl.DataFrame) -> None:
    non_utc = sample_ohlcv_frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York")
    )

    with pytest.raises(SchemaValidationError, match="UTC"):
        validate_ohlcv(non_utc)


def test_validator_rejects_non_numeric_price_column(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaValidationError, match="open must use a numeric"):
        validate_ohlcv(sample_ohlcv_frame.with_columns(pl.lit("invalid").alias("open")))


def test_validator_rejects_duplicate_bars(sample_ohlcv_frame: pl.DataFrame) -> None:
    duplicate = pl.concat(
        [sample_ohlcv_frame.head(1), sample_ohlcv_frame.head(1), sample_ohlcv_frame.tail(2)]
    )

    with pytest.raises(DuplicateBarError, match="duplicate"):
        validate_ohlcv(duplicate)


def test_validator_rejects_unsorted_bars(sample_ohlcv_frame: pl.DataFrame) -> None:
    unsorted = pl.concat(
        [sample_ohlcv_frame.slice(1, 1), sample_ohlcv_frame.head(1), sample_ohlcv_frame.tail(1)]
    )

    with pytest.raises(InvalidOHLCVError, match="ordered by symbol"):
        validate_ohlcv(unsorted)


def test_validator_rejects_invalid_ohlc_relationship(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    invalid = sample_ohlcv_frame.with_columns(pl.lit(190.0).alias("low"))

    with pytest.raises(InvalidOHLCVError, match="low <= open/close <= high"):
        validate_ohlcv(invalid)


def test_validator_rejects_negative_volume(sample_ohlcv_frame: pl.DataFrame) -> None:
    with pytest.raises(InvalidOHLCVError, match="volume"):
        validate_ohlcv(sample_ohlcv_frame.with_columns(pl.lit(-1.0).alias("volume")))


def test_validator_rejects_non_finite_numeric_value(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    with pytest.raises(InvalidOHLCVError, match="finite"):
        validate_ohlcv(sample_ohlcv_frame.with_columns(pl.lit(math.nan).alias("close")))


def test_validator_rejects_missing_timestamp_value(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    timestamps = [None, *sample_ohlcv_frame["timestamp"].to_list()[1:]]
    invalid = sample_ohlcv_frame.with_columns(
        pl.Series("timestamp", timestamps, dtype=OHLCV_SCHEMA["timestamp"])
    )

    with pytest.raises(InvalidOHLCVError, match="timestamp and symbol"):
        validate_ohlcv(invalid)


def test_validator_rejects_blank_symbol(sample_ohlcv_frame: pl.DataFrame) -> None:
    with pytest.raises(InvalidOHLCVError, match="timestamp and symbol"):
        validate_ohlcv(sample_ohlcv_frame.with_columns(pl.lit(" ").alias("symbol")))


def test_validator_rejects_empty_data(sample_ohlcv_frame: pl.DataFrame) -> None:
    with pytest.raises(EmptyDataError, match="at least one bar"):
        validate_ohlcv(sample_ohlcv_frame.head(0))


def test_validator_rejects_non_dataframe_input() -> None:
    with pytest.raises(SchemaValidationError, match="Polars DataFrame"):
        validate_ohlcv("not a dataframe")  # type: ignore[arg-type]
