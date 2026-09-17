"""Shared, deterministic fixtures for data-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import polars as pl


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", default=False,
                     help="Allow explicitly marked live vendor tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-network"):
        skip = pytest.mark.skip(reason="Live vendor test; pass --run-network to opt in")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def prevent_accidental_yahoo_requests(request, monkeypatch):
    """Offline tests must deliberately mock downloads instead of reaching Yahoo."""
    if "network" not in request.keywords:
        import yfinance as yf

        def blocked(*args, **kwargs):
            raise AssertionError("Unexpected live Yahoo request in an offline test")

        monkeypatch.setattr(yf, "download", blocked)


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
