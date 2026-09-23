"""Opt-in Massive checks: MASSIVE_API_KEY=... pytest --run-network -m network."""

from datetime import UTC, datetime, timedelta
import os

from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.providers import MassiveProvider, ProviderRegistry
from data_pipeline.schemas.ohlcv import OHLCV_SCHEMA, validate_ohlcv


@pytest.mark.network
@pytest.mark.parametrize("timeframe", ["1d", "15m"])
def test_massive_live_fetch_store_and_read(tmp_path, timeframe):
    if not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("Set MASSIVE_API_KEY to run the live Massive check")
    # A completed recent week avoids both delayed endpoints and older plan limits.
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    request = DataRequest("AAPL", end - timedelta(days=7), end,
                          provider="massive", timeframe=timeframe)
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "massive": MassiveProvider(timeout=15),
    }))
    stored = pipeline.ingest(request)
    frame = pipeline.read_dataset(stored.raw.dataset_id)
    assert frame.schema == OHLCV_SCHEMA
    assert_frame_equal(frame, validate_ohlcv(frame))
    assert frame["symbol"].unique().to_list() == ["AAPL"]
    assert all(request.start <= stamp < request.end for stamp in frame["timestamp"])
    assert stored.raw.quality.status == "unknown"
    reopened = DataPipeline(tmp_path / "data")
    assert_frame_equal(reopened.read(DataQuery(provider="massive", symbol="AAPL", timeframe=timeframe)), frame)
    assert reopened.store.audit() == [stored.raw.dataset_id]
    if timeframe == "1d":
        assert all(stamp.hour == stamp.minute == stamp.second == stamp.microsecond == 0
                   for stamp in frame["timestamp"])
