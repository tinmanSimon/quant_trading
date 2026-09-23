"""Shared, deterministic fixtures for data-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import polars as pl


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", default=False,
                     help="Allow explicitly marked live vendor tests")
    parser.addoption("--run-browser", action="store_true", default=False,
                     help="Run offline chart interaction tests in installed Chrome/Chromium")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-browser"):
        skip_browser = pytest.mark.skip(reason="Browser test; pass --run-browser to opt in")
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(skip_browser)
    if not config.getoption("--run-network"):
        skip = pytest.mark.skip(reason="Live vendor test; pass --run-network to opt in")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate_private_strategy_discovery(tmp_path, monkeypatch):
    """Tests must never import the user's actual personal strategy package."""
    from research.strategies import loader
    monkeypatch.setattr(loader, "project_root", lambda: tmp_path / "test-project")


@pytest.fixture(autouse=True)
def prevent_accidental_yahoo_requests(request, monkeypatch):
    """Offline tests must deliberately mock downloads instead of reaching Yahoo."""
    if "network" not in request.keywords:
        import yfinance as yf

        def blocked(*args, **kwargs):
            raise AssertionError("Unexpected live Yahoo request in an offline test")

        monkeypatch.setattr(yf, "download", blocked)
        from yfinance.data import YfData
        monkeypatch.setattr(YfData, "get", blocked)
        monkeypatch.setattr(YfData, "cache_get", blocked)


@pytest.fixture(autouse=True)
def prevent_accidental_http_requests(request, monkeypatch):
    """Massive and other HTTP clients need explicit mocks in offline tests."""
    if "network" not in request.keywords:
        import requests

        def blocked(*args, **kwargs):
            raise AssertionError("Unexpected live HTTP request in an offline test")

        monkeypatch.setattr(requests.sessions.Session, "request", blocked)


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
