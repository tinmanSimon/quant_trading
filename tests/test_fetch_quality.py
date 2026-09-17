"""Persistent omission provenance and exact filtering through public APIs."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from data_pipeline import (
    DataPipeline, DataQuery, DataRequest, FetchQuality, FetchResult, OmittedBar,
    SchemaValidationError, StoredDataset,
)
from data_pipeline.processing import ScalePrices
from data_pipeline.providers import BaseDataProvider, ProviderRegistry, YFinanceProvider
from data_pipeline.providers import yahoo
from data_pipeline.quality import merge_quality


@pytest.fixture
def request_data():
    return DataRequest("AAPL", datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                       datetime(2024, 1, 2, 17, 30, tzinfo=UTC))


class ReportedProvider(BaseDataProvider):
    def __init__(self, frame, quality):
        self.frame = frame
        self.quality = quality

    def fetch(self, request):
        return self.frame.clone()

    def fetch_result(self, request):
        return FetchResult(self.frame.clone(), self.quality)


def test_quality_survives_restart_processing_and_refetch(tmp_path, sample_ohlcv_frame, request_data):
    quality = FetchQuality("reported", True, (
        OmittedBar(request_data.start + timedelta(hours=1), "all_ohlc_missing"),
    ), "fixture/1")
    provider = ReportedProvider(sample_ohlcv_frame[::2], quality)
    pipeline = DataPipeline(tmp_path, providers=ProviderRegistry({"yahoo": provider}))
    result = pipeline.ingest(request_data, processors=[ScalePrices(2)])
    reopened = DataPipeline(tmp_path)
    assert reopened.get_metadata(result.raw.dataset_id).quality == quality
    assert reopened.get_metadata(result.processed.dataset_id).quality == quality
    assert reopened.get_metadata(result.processed.dataset_id).parent_ids == (result.raw.dataset_id,)
    # A successful correction gets a new quality report, not inherited gaps.
    provider.frame = sample_ohlcv_frame
    provider.quality = FetchQuality("reported", False, provider_version="fixture/2")
    repaired = pipeline.refetch(result.raw.dataset_id, confirm=result.raw.dataset_id)
    assert repaired.quality == provider.quality
    assert not pipeline.get_metadata(result.raw.dataset_id).active
    assert pipeline.get_metadata(result.raw.dataset_id).quality == quality
    assert not pipeline.get_metadata(result.processed.dataset_id).active


def test_manual_data_and_legacy_metadata_are_explicitly_unknown(tmp_path, sample_ohlcv_frame, request_data):
    pipeline = DataPipeline(tmp_path)
    saved = pipeline.ingest_frame(request_data, sample_ohlcv_frame).raw
    assert saved.quality == FetchQuality()
    fields = json.loads(saved.to_json())
    fields.pop("quality")
    old = StoredDataset.from_json(json.dumps(fields), active=True)
    assert old.quality.status == "unknown"
    assert old.quality.skip_missing_ohlc is None


def test_legacy_provider_default_result_does_not_claim_reported_quality(sample_ohlcv_frame, request_data):
    class LegacyProvider(BaseDataProvider):
        def fetch(self, request):
            return sample_ohlcv_frame
    result = LegacyProvider().fetch_result(request_data)
    assert result.quality == FetchQuality()
    assert_frame_equal(result.frame, sample_ohlcv_frame)


def test_compaction_retains_source_provenance(tmp_path, sample_ohlcv_frame, request_data):
    pipeline = DataPipeline(tmp_path)
    omitted = OmittedBar(request_data.start + timedelta(minutes=30), "all_ohlc_missing")
    first = pipeline.store.write_raw(request_data, sample_ohlcv_frame.head(1),
                                    quality=FetchQuality("reported", True, (omitted,)))
    second = pipeline.store.write_raw(request_data, sample_ohlcv_frame.tail(1),
                                     quality=FetchQuality("reported", False))
    ids = [first.dataset_id, second.dataset_id]
    merged = pipeline.store.compact_raw(ids, confirm=ids)
    assert merged.quality.status == "reported"
    assert merged.quality.omitted_bars == (omitted,)
    assert merged.quality.skip_missing_ohlc is True
    assert set(merged.supersedes) == set(ids)
    assert pipeline.get_metadata(merged.dataset_id).quality == merged.quality
    unknown = merge_quality([merged.quality, FetchQuality()])
    assert unknown.status == "unknown"
    assert unknown.omitted_bars == (omitted,)


def test_direct_replacement_cannot_claim_original_fetch_quality(tmp_path, sample_ohlcv_frame, request_data):
    pipeline = DataPipeline(tmp_path)
    saved = pipeline.store.write_raw(request_data, sample_ohlcv_frame, quality=FetchQuality("reported", False))
    new = pipeline.store.replace_raw(saved.dataset_id, sample_ohlcv_frame, confirm=saved.dataset_id)
    assert new.quality.status == "unknown"


def test_reports_are_immutable_normalized_and_deduplicated(request_data):
    bar = OmittedBar(request_data.start, "all_ohlc_missing")
    report = FetchQuality("reported", True, [bar, bar])
    assert report.omitted_bars == (bar,)
    with pytest.raises(FrozenInstanceError):
        report.status = "unknown"
    with pytest.raises(ValueError, match="timezone-aware"):
        OmittedBar(datetime(2024, 1, 1), "missing")
    with pytest.raises(ValueError, match="boolean"):
        FetchQuality(skip_missing_ohlc=1)


@pytest.mark.parametrize("boundary,expected", [("start", [1, 2]), ("end", [0, 1])])
def test_storage_keeps_microsecond_query_semantics(tmp_path, sample_ohlcv_frame, request_data, boundary, expected):
    pipeline = DataPipeline(tmp_path)
    pipeline.ingest_frame(request_data, sample_ohlcv_frame)
    timestamp = sample_ohlcv_frame["timestamp"][0 if boundary == "start" else 1]
    query = DataQuery(provider="yahoo", symbol="AAPL", timeframe="1h",
                      **{boundary: timestamp + timedelta(microseconds=1)})
    assert_frame_equal(pipeline.read(query), sample_ohlcv_frame[expected])


@pytest.fixture
def vendor_frame(sample_ohlcv_frame):
    frame = sample_ohlcv_frame.to_pandas().set_index("timestamp").drop(columns="symbol")
    frame.columns = [name.title() for name in frame.columns]
    return frame


def test_pandas_skip_report_is_per_fetch_and_persistent(monkeypatch, tmp_path, vendor_frame, request_data):
    download = Mock(return_value=vendor_frame)
    monkeypatch.setattr(yahoo, "_download_history", download)
    provider = YFinanceProvider(skip_missing_ohlc=True)
    pipeline = DataPipeline(tmp_path, providers=ProviderRegistry({"yahoo": provider}))
    vendor_frame.loc[vendor_frame.index[1], ["Open", "High", "Low", "Close"]] = None
    with pytest.warns(UserWarning, match="skipped 1 bars"):
        saved = pipeline.ingest(request_data).raw
    expected = OmittedBar(request_data.start + timedelta(hours=1), "all_ohlc_missing")
    assert pipeline.get_metadata(saved.dataset_id).quality.omitted_bars == (expected,)
    # The same provider object must never leak a previous call's omissions.
    download.return_value = vendor_frame.iloc[[0, 2]]
    assert provider.fetch_result(request_data).quality.omitted_bars == ()


@pytest.mark.parametrize("boundary,expected", [("start", [1, 2]), ("end", [0, 1])])
def test_yahoo_keeps_microsecond_bounds(monkeypatch, vendor_frame, request_data, boundary, expected):
    download = Mock(return_value=vendor_frame)
    monkeypatch.setattr(yahoo, "_download_history", download)
    timestamp = vendor_frame.index[0 if boundary == "start" else 1].to_pydatetime()
    request_data = replace(request_data, **{boundary: timestamp + timedelta(microseconds=1)})
    result = YFinanceProvider().fetch_result(request_data)
    assert result.frame["timestamp"].to_list() == list(vendor_frame.index[expected].to_pydatetime())
    if boundary == "end":
        assert download.call_args.kwargs["end"] > request_data.end


@pytest.mark.parametrize("large", [2**53 + 1, str(2**53 + 1), "9007199254740993.0"])
def test_mixed_values_cannot_lose_precision_before_schema_validation(monkeypatch, vendor_frame, request_data, large):
    vendor_frame["Volume"] = pd.Series([large, "1.5", "2"], index=vendor_frame.index, dtype=object)
    monkeypatch.setattr(yahoo, "_download_history", Mock(return_value=vendor_frame))
    with pytest.raises(SchemaValidationError, match="losslessly"):
        YFinanceProvider().fetch(request_data)


def test_normal_decimal_strings_remain_supported(monkeypatch, vendor_frame, request_data):
    vendor_frame["Open"] = vendor_frame["Open"].astype(str)
    monkeypatch.setattr(yahoo, "_download_history", Mock(return_value=vendor_frame))
    result = YFinanceProvider().fetch(request_data)
    assert result["open"].to_list() == [185.64, 185.95, 186.25]
