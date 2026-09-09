"""Shared, deterministic fixtures for data-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import polars as pl


@pytest.fixture
def isolated_data_dir(tmp_path: Path) -> Path:
    """Return a per-test data directory that is never the project's real data/ root."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_ohlcv_records() -> list[dict[str, object]]:
    """Load a small, stable OHLCV fixture without contacting a vendor."""
    fixture_path = Path(__file__).parent / "fixtures" / "ohlcv_1h.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_ohlcv_frame(sample_ohlcv_records: list[dict[str, object]]) -> pl.DataFrame:
    """Return the shared OHLCV fixture in canonical Polars data types."""
    return pl.DataFrame(sample_ohlcv_records).with_columns(
        pl.col("timestamp").str.to_datetime(time_unit="ms", time_zone="UTC"),
        *[
            pl.col(column).cast(pl.Float64)
            for column in ("open", "high", "low", "close", "volume")
        ],
    )
