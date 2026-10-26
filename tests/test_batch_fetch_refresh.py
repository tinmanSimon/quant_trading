"""Union refresh through the public fetch API, without vendor network calls."""

from datetime import UTC, datetime

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import (
    DataPipeline, DataQuery, DataRequest, EmptyDataError, FetchQuality, FetchResult,
    InvalidDataRequestError, OHLCV_SCHEMA, fetch_many,
)
from data_pipeline.providers import BaseDataProvider, ProviderRegistry


def day(value):
    return datetime(2026, 10, value, tzinfo=UTC)


def bars(start=1, end=21, symbol="TEST", revision=0):
    prices = [100.0 + value + revision for value in range(start, end)]
    return pl.DataFrame({
        "timestamp": [day(value) for value in range(start, end)], "symbol": [symbol] * len(prices),
        "open": prices, "high": [p + 1 for p in prices], "low": [p - 1 for p in prices],
        "close": prices, "volume": [123.5] * len(prices),
    }, schema=OHLCV_SCHEMA)


class Vendor(BaseDataProvider):
    def __init__(self):
        self.requests = []
        self.omitted = set()
        self.errors = {}

    def fetch(self, request):
        return self.fetch_result(request).frame

    def fetch_result(self, request):
        self.requests.append(request)
        if request.symbol in self.errors:
            raise self.errors[request.symbol]
        frame = bars(symbol=request.symbol, revision=10).filter(
            (pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end)
            & ~pl.col("timestamp").is_in(pl.Series(list(self.omitted), dtype=OHLCV_SCHEMA["timestamp"]).implode()))
        if frame.is_empty():
            raise EmptyDataError("Vendor returned no bars")
        return FetchResult(frame, FetchQuality(status="reported", provider_version="test-refresh"))


@pytest.fixture
def setup(tmp_path):
    vendor = Vendor()
    return DataPipeline(tmp_path / "data", providers=ProviderRegistry({"test": vendor})), vendor


def seed(pipeline, start, end, symbol="TEST"):
    return pipeline.store.write_raw(DataRequest(symbol, day(start), day(end), "1d", "test"),
                                    bars(start, end, symbol))


def fetch(pipeline, start, end, tickers=("TEST",), **kwargs):
    return pipeline.fetch_many(tickers, start=day(start), end=day(end), timeframe="1d",
                               provider="test", overlap_policy="refresh_overlap", **kwargs)


def test_fully_covered_skips_vendor_and_does_not_create_revisions(setup):
    pipeline, vendor = setup
    old = seed(pipeline, 2, 16)
    report = fetch(pipeline, 6, 11)
    assert report.ok
    outcome, = report.outcomes
    assert outcome.status == "skipped"
    assert outcome.dataset_ids == (old.dataset_id,)
    assert outcome.row_count == 0 and outcome.fetch_start is None and outcome.fetch_end is None
    assert outcome.requested_start == day(6) and outcome.requested_end == day(11)
    assert vendor.requests == []
    assert pipeline.list_datasets() == [old]


@pytest.mark.parametrize("policy", ["refresh_overlap", "fetch_missing"])
def test_reusing_screening_data_preserves_exclusion_provenance_and_warns(setup, policy):
    from data_pipeline import OmittedBar

    pipeline, vendor = setup
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(day(7), "invalid_bar: open=0"),))
    frame = bars(2, 16).filter(pl.col("timestamp") != day(7))
    old = pipeline.store.write_raw(DataRequest("TEST", day(2), day(16), "1d", "test"),
                                   frame, quality=quality)
    report = pipeline.fetch_many(["TEST"], provider="test", timeframe="1d",
                                 start=day(6), end=day(11), overlap_policy=policy)
    outcome, = report.outcomes
    assert outcome.status == "skipped" and outcome.dataset_ids == (old.dataset_id,)
    assert vendor.requests == []
    assert outcome.quality["skip_invalid_bars"] is True
    assert outcome.quality["omitted_bars"][0]["timestamp"] == day(7).isoformat()
    assert "1 excluded bar" in outcome.warning_message
    assert "does not certify complete data" in outcome.warning_message


def test_exclusion_outside_requested_range_does_not_warn_about_selected_data(setup):
    from data_pipeline import OmittedBar

    pipeline, _ = setup
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(day(3), "invalid_bar: open=0"),))
    pipeline.store.write_raw(DataRequest("TEST", day(2), day(16), "1d", "test"),
                              bars(2, 16).filter(pl.col("timestamp") != day(3)), quality=quality)
    outcome, = fetch(pipeline, 6, 11).outcomes
    assert outcome.quality["skip_invalid_bars"] is True
    assert outcome.warning_message is None


@pytest.mark.parametrize("old_ranges, requested, expected, status", [
    ([(6, 16)], (1, 11), (1, 16), "refreshed"),
    ([(1, 8)], (6, 16), (1, 16), "refreshed"),
    ([(2, 5), (8, 11)], (3, 9), (2, 11), "refreshed"),
    ([(3, 5), (7, 9)], (1, 11), (1, 11), "refreshed"),
    ([(6, 16)], (1, 6), (1, 6), "saved"),
    ([], (1, 11), (1, 11), "saved"),
])
def test_partial_overlap_refreshes_union_and_preserves_original_history(
    setup, old_ranges, requested, expected, status,
):
    pipeline, vendor = setup
    old = [seed(pipeline, *bounds) for bounds in old_ranges]
    original = {item.dataset_id: pipeline.read_dataset(item.dataset_id) for item in old}
    report = fetch(pipeline, *requested)
    assert report.ok
    outcome, = report.outcomes
    assert outcome.status == status
    assert (vendor.requests[0].start, vendor.requests[0].end) == tuple(map(day, expected))
    assert len(vendor.requests) == 1
    assert (outcome.fetch_start, outcome.fetch_end) == tuple(map(day, expected))
    replacement = pipeline.get_metadata(outcome.dataset_ids[0])
    assert (outcome.first_timestamp, outcome.last_timestamp) == (day(expected[0]), day(expected[1] - 1))
    assert replacement.quality.provider_version == "test-refresh"
    assert_frame_equal(pipeline.read_dataset(replacement.dataset_id), bars(*expected, revision=10))
    expected_ids = tuple(item.dataset_id for item in old) if status == "refreshed" else ()
    assert outcome.replaced_dataset_ids == replacement.supersedes == expected_ids
    for item in old:
        assert pipeline.get_metadata(item.dataset_id).active is (status != "refreshed")
        assert_frame_equal(pipeline.read_dataset(item.dataset_id), original[item.dataset_id])
    retry = fetch(pipeline, *requested)
    assert retry.outcomes[0].status == "skipped"
    assert len(vendor.requests) == 1


def test_adjacent_batches_can_cover_request_but_disconnected_batches_cannot(setup):
    pipeline, vendor = setup
    first = seed(pipeline, 1, 4)
    second = seed(pipeline, 4, 7)
    assert fetch(pipeline, 1, 7).outcomes[0].dataset_ids == (first.dataset_id, second.dataset_id)
    assert vendor.requests == []
    third = seed(pipeline, 9, 12)
    outcome, = fetch(pipeline, 2, 11).outcomes
    assert outcome.status == "refreshed"
    assert outcome.replaced_dataset_ids == (first.dataset_id, second.dataset_id, third.dataset_id)
    assert (outcome.fetch_start, outcome.fetch_end) == (day(1), day(12))


@pytest.mark.parametrize("missing", [6, 10, 15])
def test_incomplete_response_keeps_old_dataset_and_attempts_next_ticker(setup, missing):
    pipeline, vendor = setup
    old = seed(pipeline, 6, 16)
    original = pipeline.read_dataset(old.dataset_id)
    vendor.omitted = {day(missing)}
    report = fetch(pipeline, 1, 11, tickers=("TEST", "OTHER"))
    assert [outcome.status for outcome in report.outcomes] == ["failed", "saved"]
    assert not report.ok
    assert report.outcomes[0].error_type == "RequestDataMismatchError"
    assert report.outcomes[0].replaced_dataset_ids == ()
    assert report.outcomes[0].fetch_end == day(16)
    assert pipeline.get_metadata(old.dataset_id).active
    assert_frame_equal(pipeline.read_dataset(old.dataset_id), original)
    assert len(pipeline.list_datasets(DataQuery(include_history=True))) == 2


@pytest.mark.parametrize("error, status", [(EmptyDataError("No data"), "empty"),
                                            (RuntimeError("Vendor unavailable"), "failed")])
def test_fetch_failure_leaves_previous_revision_active(setup, error, status):
    pipeline, vendor = setup
    old = seed(pipeline, 6, 16)
    vendor.errors["TEST"] = error
    report = fetch(pipeline, 1, 11)
    assert not report.ok and report.outcomes[0].status == status
    assert pipeline.list_datasets() == [old]
    assert len(pipeline.list_datasets(DataQuery(include_history=True))) == 1


def test_default_policy_still_rejects_overlapping_writes(setup):
    pipeline, vendor = setup
    old = seed(pipeline, 6, 16)
    report = pipeline.fetch_many(["TEST"], start=day(1), end=day(11), timeframe="1d", provider="test")
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_type == "OverlappingDataError"
    assert vendor.requests[0].end == day(11)
    assert pipeline.list_datasets() == [old]


@pytest.mark.parametrize("policy", [None, True, "ignore", "overwrite", ""])
def test_invalid_policy_fails_before_fetch(setup, policy):
    pipeline, vendor = setup
    with pytest.raises(InvalidDataRequestError, match="overlap_policy"):
        pipeline.fetch_many(["TEST"], start=day(1), end=day(11), provider="test", overlap_policy=policy)
    assert vendor.requests == []


def test_standalone_entrypoint_accepts_refresh_policy(setup):
    pipeline, vendor = setup
    seed(pipeline, 6, 16)
    report = fetch_many(pipeline, tickers=["TEST"], start=day(1), end=day(11), timeframe="1d",
                        provider="test", overlap_policy="refresh_overlap")
    assert report.outcomes[0].status == "refreshed"


def test_failed_refresh_reports_downloaded_rows_separately_from_saved_rows(setup):
    pipeline, vendor = setup
    old = seed(pipeline, 6, 16)
    vendor.omitted = {day(10)}
    outcome, = fetch(pipeline, 1, 11).outcomes
    assert outcome.status == "failed" and outcome.row_count == 0
    assert outcome.fetched_row_count == 14
    assert outcome.fetch_ranges[0]["row_count"] == 14
    assert outcome.fetch_ranges[0]["status"] == "fetched"
    assert pipeline.get_metadata(old.dataset_id).active
