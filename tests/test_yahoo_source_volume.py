"""Exercise real yfinance processing with offline Yahoo HTTP responses."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from yfinance.base import TickerBase
from yfinance.data import YfData

from data_pipeline import DataPipeline, DataRequest, OmittedBar
from data_pipeline.exceptions import EmptyDataError, InvalidOHLCVError, SchemaValidationError
from data_pipeline.providers import ProviderRegistry, YFinanceProvider


@pytest.fixture
def source(monkeypatch):
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    timestamps = [int((start + timedelta(hours=i)).timestamp()) for i in range(3)]
    payload = {"chart": {"error": None, "result": [{
        "meta": {
            "instrumentType": "EQUITY", "exchangeTimezoneName": "America/New_York",
            "currency": "USD", "symbol": "AAPL",
            "tradingPeriods": [[{
                "start": timestamps[0], "end": int(start.replace(hour=21, minute=0).timestamp()),
                "timezone": "EST", "gmtoffset": -18000,
            }]],
        },
        "timestamp": timestamps,
        "indicators": {"adjclose": [{"adjclose": [50.5, 51., 51.5]}], "quote": [{
            "open": [100., 101., 102.], "high": [102., 103., 104.],
            "low": [99., 100., 101.], "close": [101., 102., 103.],
            "volume": [1000, 2000, 3000],
        }]},
    }]}}
    response = Mock(text="{}")
    response.json.side_effect = lambda: deepcopy(payload)
    cached = Mock(return_value=response)
    uncached = Mock(return_value=response)
    monkeypatch.setattr(YfData, "cache_get", cached)
    monkeypatch.setattr(YfData, "get", uncached)
    monkeypatch.setattr(TickerBase, "_get_ticker_tz", lambda self, timeout: "America/New_York")
    request = DataRequest("AAPL", start, start + timedelta(hours=3))
    return payload, request, cached, uncached


@pytest.mark.parametrize("skip_missing_ohlc", [False, True])
@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_source_volume_is_rejected_before_storage(source, tmp_path, missing, skip_missing_ohlc):
    payload, request, cached, _ = source
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = missing
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "yahoo": YFinanceProvider(skip_missing_ohlc=skip_missing_ohlc),
    }))

    with pytest.raises(InvalidOHLCVError, match="missing source volume.*2024-01-02T15:30:00"):
        pipeline.ingest(request)

    cached.assert_called_once()
    assert pipeline.list_datasets() == []
    assert not list((tmp_path / "data").rglob("*.parquet"))


def test_genuine_zero_volume_and_prices_survive_ingest(source, tmp_path):
    payload, request, cached, _ = source
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = 0
    before = deepcopy(payload)
    pipeline = DataPipeline(tmp_path / "data")
    result = pipeline.ingest(request)
    frame = pipeline.read_dataset(result.raw.dataset_id)

    assert frame["volume"].to_list() == [1000., 0., 3000.]
    assert frame["close"].to_list() == [101., 102., 103.]
    assert payload == before
    assert pipeline.store.audit() == [result.raw.dataset_id]
    cached.assert_called_once()


def test_wholly_missing_bar_is_skipped_before_yfinance_volume_fill(source, tmp_path):
    payload, request, _, _ = source
    quotes = payload["chart"]["result"][0]["indicators"]["quote"][0]
    for values in quotes.values():
        values[1] = None
    before = deepcopy(payload)
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "yahoo": YFinanceProvider(skip_missing_ohlc=True),
    }))

    with pytest.warns(UserWarning, match="skipped 1 bars.*2024-01-02T15:30:00"):
        result = pipeline.ingest(request)

    frame = pipeline.read_dataset(result.raw.dataset_id)
    assert frame["volume"].to_list() == [1000., 3000.]
    assert frame["close"].to_list() == [101., 103.]
    assert result.raw.quality.status == "reported"
    assert result.raw.quality.omitted_bars == (
        OmittedBar(request.start + timedelta(hours=1), "all_ohlc_missing"),
    )
    assert pipeline.get_metadata(result.raw.dataset_id).quality == result.raw.quality
    assert payload == before
    # Skip policy is local to the fetch; it must not mutate a shared response
    # or change another provider's strict behavior.
    with pytest.raises(InvalidOHLCVError, match="missing source volume"):
        YFinanceProvider().fetch(request)


def test_all_missing_source_bars_do_not_create_a_dataset(source, tmp_path):
    payload, request, _, _ = source
    quotes = payload["chart"]["result"][0]["indicators"]["quote"][0]
    for field in quotes:
        quotes[field] = [None] * 3
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "yahoo": YFinanceProvider(skip_missing_ohlc=True),
    }))
    with pytest.warns(UserWarning, match="skipped 3 bars"):
        with pytest.raises(EmptyDataError):
            pipeline.ingest(request)
    assert pipeline.list_datasets() == []


def test_uncached_history_also_checks_source_volume(source):
    payload, request, cached, uncached = source
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = None
    # A future end routes yfinance through get(), rather than cache_get().
    request = DataRequest("AAPL", request.start, datetime.now(UTC) + timedelta(days=1))
    with pytest.raises(InvalidOHLCVError, match="missing source volume"):
        YFinanceProvider().fetch(request)
    uncached.assert_called_once()
    cached.assert_not_called()


def test_malformed_volume_array_fails_closed(source):
    payload, request, _, _ = source
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"].pop()
    with pytest.raises(SchemaValidationError, match="Volume count"):
        YFinanceProvider().fetch(request)


def test_latest_quote_outside_requested_window_does_not_block_history(source):
    payload, request, _, _ = source
    raw = payload["chart"]["result"][0]
    raw["timestamp"][-1] = int(request.end.timestamp())
    raw["indicators"]["quote"][0]["volume"][-1] = None
    result = YFinanceProvider().fetch(request)
    assert result["volume"].to_list() == [1000., 2000.]


@pytest.mark.parametrize("volume", [None, 0])
def test_daily_source_volume_uses_same_validation(source, volume):
    payload, request, _, _ = source
    raw = payload["chart"]["result"][0]
    raw["timestamp"] = [int((request.start + timedelta(days=i)).timestamp()) for i in range(3)]
    raw["indicators"]["quote"][0]["volume"][1] = volume
    daily = DataRequest("AAPL", request.start.replace(hour=0, minute=0),
                        request.start.replace(hour=0, minute=0) + timedelta(days=3), timeframe="1d")
    if volume is None:
        with pytest.raises(InvalidOHLCVError, match="2024-01-03T14:30:00"):
            YFinanceProvider().fetch(daily)
    else:
        result = YFinanceProvider().fetch(daily)
        assert result["volume"].to_list() == [1000., 0., 3000.]
        assert all(timestamp.hour == 0 for timestamp in result["timestamp"])


def test_daily_skipped_source_bar_records_session_date_not_open_instant(source):
    payload, request, _, _ = source
    raw = payload["chart"]["result"][0]
    raw["timestamp"] = [int((request.start + timedelta(days=i)).timestamp()) for i in range(3)]
    quotes = raw["indicators"]["quote"][0]
    for field in quotes:
        quotes[field][1] = None
    start = request.start.replace(hour=0, minute=0)
    daily = DataRequest("AAPL", start, start + timedelta(days=3), timeframe="1d")
    with pytest.warns(UserWarning, match="skipped 1 bars"):
        result = YFinanceProvider(skip_missing_ohlc=True).fetch_result(daily)
    assert result.quality.omitted_bars == (OmittedBar(start + timedelta(days=1), "all_ohlc_missing"),)


@pytest.mark.parametrize("volume", [None, 0, 2000])
def test_source_omissions_recorded_before_yfinance_can_merge_rows(source, volume):
    payload, request, _, _ = source
    raw = payload["chart"]["result"][0]
    quotes = raw["indicators"]["quote"][0]
    for name in ("open", "high", "low", "close"):
        quotes[name][1] = None
    quotes["volume"][1] = volume
    with pytest.warns(UserWarning, match="skipped 1 bars"):
        fetched = YFinanceProvider(skip_missing_ohlc=True).fetch_result(request)
    assert fetched.quality.omitted_bars == (OmittedBar(request.start + timedelta(hours=1), "all_ohlc_missing"),)
    assert fetched.frame["volume"].to_list() == [1000., 3000.]


@pytest.mark.parametrize("value", [2**53 + 1, 12.5, True])
def test_raw_volume_cannot_be_rounded_or_truncated_before_validation(source, value):
    payload, request, _, _ = source
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"][1] = value
    with pytest.raises(SchemaValidationError):
        YFinanceProvider().fetch(request)


def test_source_duplicate_bars_cannot_be_silently_removed_by_yfinance(source):
    from data_pipeline import DuplicateBarError

    payload, request, _, _ = source
    raw = payload["chart"]["result"][0]
    raw["timestamp"][1] = raw["timestamp"][0]
    with pytest.raises(DuplicateBarError, match="duplicate source timestamp"):
        YFinanceProvider().fetch(request)
