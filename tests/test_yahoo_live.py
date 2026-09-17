"""Optional live Yahoo check; never needed by the normal offline test suite.

Run explicitly with ``pytest --run-network -m network tests/test_yahoo_live.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
import pytest
import yfinance as yf

from data_pipeline.models import DataRequest
from data_pipeline import DataPipeline, DataQuery
from data_pipeline.providers import ProviderRegistry, YFinanceProvider
from data_pipeline.schemas.ohlcv import OHLCV_SCHEMA, validate_ohlcv


@pytest.mark.network
@pytest.mark.parametrize("timeframe", ["1d", "1h"])
def test_yahoo_live_fetch_store_and_read(tmp_path: Path, timeframe: str) -> None:
    # Keep yfinance's own timezone/cookie cache out of the user's normal cache.
    yf.set_tz_cache_location(str(tmp_path / "yfinance-cache"))
    # Intraday history is retention-limited. Use completed recent sessions,
    # while keeping the daily check at stable historical dates.
    end = (datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
           if timeframe == "1h" else datetime(2024, 1, 10, tzinfo=UTC))
    start = end - timedelta(days=8)
    request = DataRequest(
        symbol="aapl",
        start=start,
        end=end,
        timeframe=timeframe,
        provider="yahoo",
        price_adjustment="unadjusted",
    )
    providers = ProviderRegistry({"yahoo": YFinanceProvider(timeout=10)})
    pipeline = DataPipeline(tmp_path / "data", providers=providers)
    stored = pipeline.ingest(request)
    result = pipeline.read_dataset(stored.raw.dataset_id)
    assert stored.processed is None
    restarted = DataPipeline(tmp_path / "data")
    assert_frame_equal(restarted.read_dataset(stored.raw.dataset_id), result)
    assert_frame_equal(restarted.read(DataQuery(provider="yahoo", symbol="aapl", timeframe=timeframe)), result)
    assert restarted.store.audit() == [stored.raw.dataset_id]

    assert result.schema == OHLCV_SCHEMA
    assert_frame_equal(result, validate_ohlcv(result))
    assert result["symbol"].unique().to_list() == ["AAPL"]
    assert result.select(
        ((pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end)).all()
    ).item()
    if timeframe == "1d":
        assert all(stamp.hour == stamp.minute == stamp.second == 0 for stamp in result["timestamp"])
