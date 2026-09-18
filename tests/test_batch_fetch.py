"""Batch acquisition through the data-pipeline API and standalone function."""

from datetime import timedelta
import json
import subprocess
import sys

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import (
    BatchFetchReport, DataPipeline, FetchOutcome, FetchQuality, FetchResult,
    InvalidDataRequestError, OmittedBar, fetch_many,
)
from data_pipeline.providers import BaseDataProvider, ProviderRegistry, YFinanceProvider


class RecordingProvider(BaseDataProvider):
    def __init__(self, frame, *, failures=None, quality=None):
        self.frame = frame
        self.failures = failures or {}
        self.quality = quality or FetchQuality()
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        if request.symbol in self.failures:
            raise self.failures[request.symbol]
        return self.frame.with_columns(pl.lit(request.symbol).alias("symbol"))

    def fetch_result(self, request):
        return FetchResult(self.fetch(request), self.quality)


@pytest.fixture
def setup(tmp_path, sample_ohlcv_frame):
    provider = RecordingProvider(sample_ohlcv_frame)
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({"yahoo": provider}))
    bounds = dict(start=sample_ohlcv_frame["timestamp"][0],
                  end=sample_ohlcv_frame["timestamp"][-1] + timedelta(hours=1), timeframe="1h")
    return pipeline, provider, bounds


def test_pipeline_batch_continues_after_failure_and_preserves_values_and_quality(setup):
    pipeline, provider, bounds = setup
    provider.failures["BAD"] = RuntimeError("vendor unavailable")
    provider.frame = provider.frame.with_columns(pl.lit(1000.1234567890123).alias("volume"))
    provider.quality = FetchQuality(status="reported", skip_missing_ohlc=True,
                                    omitted_bars=(OmittedBar(bounds["start"] - timedelta(hours=1), "missing_ohlc"),))
    report = pipeline.fetch_many(["aapl", "BAD", "MSFT"], **bounds)
    assert type(report) is BatchFetchReport
    assert [request.symbol for request in provider.requests] == ["AAPL", "BAD", "MSFT"]
    assert [item.status for item in report.outcomes] == ["saved", "failed", "saved"]
    assert report.outcomes[1] == FetchOutcome("BAD", "failed", error_type="RuntimeError",
                                             error_message="vendor unavailable")
    assert not report.ok
    reopened = DataPipeline(pipeline.store.data_dir)
    assert len(reopened.list_datasets()) == 2
    for outcome in (report.outcomes[0], report.outcomes[2]):
        metadata = reopened.get_metadata(outcome.dataset_ids[0])
        assert metadata.layer == "raw"
        assert outcome.row_count == provider.frame.height
        assert metadata.quality == provider.quality
        assert outcome.quality == json.loads(metadata.to_json())["quality"]
        expected = provider.frame.with_columns(pl.lit(outcome.ticker).alias("symbol"))
        assert_frame_equal(reopened.read_dataset(metadata.dataset_id), expected)
    table = report.to_frame()
    assert table.columns == ["ticker", "status", "row_count", "dataset_ids", "error_type", "error_message", "quality"]
    assert table["ticker"].to_list() == ["AAPL", "BAD", "MSFT"]
    assert json.loads(table["quality"][0]) == report.outcomes[0].quality
    assert table["quality"][1] is None


def test_invalid_tickers_and_duplicate_writes_do_not_stop_later_tickers(setup):
    pipeline, provider, bounds = setup
    report = pipeline.fetch_many(["", 123, "aapl", "AAPL", "msft"], **bounds)
    assert [item.status for item in report.outcomes] == ["failed", "failed", "saved", "failed", "saved"]
    assert [item.error_type for item in report.outcomes[:2]] == ["InvalidDataRequestError"] * 2
    assert report.outcomes[3].error_type == "OverlappingDataError"
    assert [request.symbol for request in provider.requests] == ["AAPL", "AAPL", "MSFT"]
    assert len(pipeline.list_datasets()) == 2
    assert_frame_equal(pipeline.read_dataset(report.outcomes[2].dataset_ids[0]), provider.frame)


@pytest.mark.parametrize("skip", [False, True])
def test_explicit_yahoo_policy_is_batch_local_and_quality_is_persisted(setup, monkeypatch, skip):
    pipeline, original, bounds = setup
    registry = pipeline.providers
    calls = []

    def fetch_result(provider, request):
        calls.append((provider.skip_missing_ohlc, request))
        return FetchResult(original.frame, FetchQuality(status="reported", skip_missing_ohlc=provider.skip_missing_ohlc))

    monkeypatch.setattr(YFinanceProvider, "fetch_result", fetch_result)
    report = pipeline.fetch_many(["AAPL"], **bounds, skip_missing_ohlc=skip)
    assert report.ok
    assert len(calls) == 1 and calls[0][0] is skip
    assert report.outcomes[0].quality["skip_missing_ohlc"] is skip
    reopened = DataPipeline(pipeline.store.data_dir)
    assert reopened.get_metadata(report.outcomes[0].dataset_ids[0]).quality.skip_missing_ohlc is skip
    assert pipeline.providers is registry and registry.get("yahoo") is original
    assert original.requests == []
    assert pipeline.fetch_many(["MSFT"], **bounds).ok
    assert [request.symbol for request in original.requests] == ["MSFT"]
    assert len(calls) == 1


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_interrupt_stops_batch_without_rolling_back_previous_success(setup, exception):
    pipeline, provider, bounds = setup
    provider.failures["STOP"] = exception()
    with pytest.raises(exception):
        pipeline.fetch_many(["AAPL", "STOP", "MSFT"], **bounds)
    assert [request.symbol for request in provider.requests] == ["AAPL", "STOP"]
    assert [item.request.symbol for item in pipeline.list_datasets()] == ["AAPL"]


@pytest.mark.parametrize("entrypoint", ["pipeline", "standalone"])
@pytest.mark.parametrize("tickers,extra,message", [
    ("AAPL", {}, "list of ticker"),
    (b"AAPL", {}, "list of ticker"),
    ([], {}, "at least one"),
    (["AAPL"], {"provider": "other", "skip_missing_ohlc": True}, "Yahoo-specific"),
])
def test_batch_validation_uses_pipeline_errors_without_fetching(
    setup, entrypoint, tickers, extra, message,
):
    pipeline, provider, bounds = setup
    if entrypoint == "standalone":
        invoke = lambda tickers, **kwargs: fetch_many(pipeline, tickers=tickers, **kwargs)
    else:
        invoke = pipeline.fetch_many
    with pytest.raises(InvalidDataRequestError, match=message):
        invoke(tickers, **bounds, **extra)
    assert provider.requests == []
    assert not list(pipeline.store.data_dir.rglob("*.parquet"))


def test_standalone_pipeline_function_accepts_a_ticker_generator(setup):
    pipeline, provider, bounds = setup
    report = fetch_many(pipeline, tickers=(symbol for symbol in ["AAPL", "MSFT"]), **bounds)
    assert report.ok
    assert [request.symbol for request in provider.requests] == ["AAPL", "MSFT"]


def test_empty_report_preserves_table_schema():
    report = BatchFetchReport(())
    assert report.ok
    assert report.to_frame().shape == (0, 7)
    assert report.to_frame().schema["row_count"] == pl.Int64


def test_pipeline_import_and_batch_api_do_not_require_research(tmp_path):
    code = '''
import importlib.abc
import sys
class BlockResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'research', 'exchange_calendars', 'streamlit'}:
            raise AssertionError(f'Unexpected research dependency: {fullname}')
sys.meta_path.insert(0, BlockResearch())
from data_pipeline import DataPipeline, BatchFetchReport, FetchOutcome, InvalidDataRequestError
pipeline = DataPipeline(sys.argv[1])
try:
    pipeline.fetch_many([], start=None, end=None)
except InvalidDataRequestError:
    pass
else:
    raise AssertionError('Expected pipeline validation error')
assert BatchFetchReport((FetchOutcome('AAPL', 'saved'),)).ok
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "data")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
