"""Only missing windows reach the provider; old history survives local merging."""

from datetime import UTC, datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest, EmptyDataError, FetchQuality, FetchResult, OHLCV_SCHEMA
from data_pipeline.providers import BaseDataProvider, ProviderRegistry


def day(value):
    return datetime(2026, 10, value, tzinfo=UTC)


def bars(start, end, symbol="TEST", revision=0):
    stamps = []
    while start < end:
        stamps.append(start)
        start += timedelta(days=1)
    prices = [100.0 + index + revision for index in range(len(stamps))]
    return pl.DataFrame({"timestamp": stamps, "symbol": [symbol] * len(stamps),
                         "open": prices, "high": [p + 1 for p in prices], "low": [p - 1 for p in prices],
                         "close": prices, "volume": [123.5] * len(stamps)}, schema=OHLCV_SCHEMA)


class Vendor(BaseDataProvider):
    def __init__(self):
        self.requests = []
        self.errors = {}
        self.on_fetch = None
        self.earliest = None

    def fetch(self, request):
        return self.fetch_result(request).frame

    def fetch_result(self, request):
        if self.on_fetch:
            self.on_fetch(request)
        self.requests.append(request)
        if self.earliest is not None and request.start < self.earliest:
            raise AssertionError("Old data is outside vendor access; it must be reused locally")
        if (request.symbol, request.start) in self.errors:
            raise self.errors[request.symbol, request.start]
        return FetchResult(bars(request.start, request.end, request.symbol, revision=10),
                           FetchQuality(status="reported", provider_version="new-vendor"))


@pytest.fixture
def setup(tmp_path):
    vendor = Vendor()
    return DataPipeline(tmp_path / "data", providers=ProviderRegistry({"test": vendor})), vendor


def seed(pipeline, start, end, symbol="TEST"):
    request = DataRequest(symbol, start, end, "1d", "test")
    return pipeline.store.write_raw(request, bars(start, end, symbol),
                                    quality=FetchQuality(status="reported", provider_version="old-vendor"))


def fetch(pipeline, start, end, tickers=("TEST",), **kwargs):
    return pipeline.fetch_many(tickers, start=start, end=end, timeframe="1d", provider="test",
                               overlap_policy="fetch_missing", **kwargs)


def test_user_daily_extension_fetches_only_recent_tail_and_retains_inaccessible_history(setup):
    pipeline, vendor = setup
    first = datetime(2024, 10, 1, tzinfo=UTC)
    cutoff = datetime(2026, 9, 23, tzinfo=UTC)
    vendor.earliest = datetime(2024, 10, 20, tzinfo=UTC)
    originals = [seed(pipeline, first, cutoff, symbol) for symbol in ("AAPL", "MSFT")]
    request_start = datetime(2026, 9, 20, tzinfo=UTC)
    report = fetch(pipeline, request_start, day(20), tickers=("AAPL", "MSFT"))
    assert report.ok
    assert [(r.symbol, r.start, r.end) for r in vendor.requests] == [
        ("AAPL", cutoff, day(20)), ("MSFT", cutoff, day(20))]
    for old, outcome in zip(originals, report.outcomes):
        assert outcome.status == "merged"
        assert outcome.replaced_dataset_ids == (old.dataset_id,)
        merged = pipeline.read_dataset(outcome.dataset_ids[0])
        assert_frame_equal(merged.filter(pl.col("timestamp") < cutoff), pipeline.read_dataset(old.dataset_id))
        assert_frame_equal(merged.filter(pl.col("timestamp") >= cutoff), bars(cutoff, day(20), old.request.symbol, 10))
        assert outcome.fetched_row_count == (day(20) - cutoff).days
        assert outcome.row_count == merged.height
        assert len(outcome.fetch_ranges) == 1 and outcome.fetch_ranges[0]["start"] == cutoff.isoformat()
        assert not pipeline.get_metadata(old.dataset_id).active
    repeated = fetch(pipeline, request_start, day(20), tickers=("AAPL", "MSFT"))
    assert [item.status for item in repeated.outcomes] == ["skipped", "skipped"]
    assert len(vendor.requests) == 2


def test_multiple_missing_ranges_are_reported_before_fetch_and_merged_once(setup):
    pipeline, vendor = setup
    old = [seed(pipeline, day(3), day(5)), seed(pipeline, day(7), day(9))]
    events = []
    expected = [(day(1), day(3)), (day(5), day(7)), (day(9), day(11))]

    def on_fetch(request):
        assert events[-1]["status"] == "fetching"
        assert [(item["start"], item["end"]) for item in events[-1]["fetch_ranges"]] == [
            (start.isoformat(), end.isoformat()) for start, end in expected]
        assert pipeline.list_datasets(DataQuery()) == old  # no partial publication

    vendor.on_fetch = on_fetch
    outcome, = fetch(pipeline, day(1), day(11), progress=events.append).outcomes
    assert outcome.status == "merged"
    assert [(item.start, item.end) for item in vendor.requests] == expected
    assert [item["status"] for item in outcome.fetch_ranges] == ["fetched"] * 3
    assert outcome.row_count == 10 and outcome.fetched_row_count == 6
    merged = pipeline.read_dataset(outcome.dataset_ids[0])
    for item in old:
        original = pipeline.read_dataset(item.dataset_id)
        assert_frame_equal(merged.filter(pl.col("timestamp").is_in(original["timestamp"].implode())), original)
    assert events[-1]["status"] == "merged"
    assert events[0]["fetch_ranges"][0]["status"] == "pending"  # events are snapshots
    assert len(pipeline.list_datasets(DataQuery(include_history=True))) == 3


def test_covered_request_skips_without_provider_lookup_or_merge(setup, monkeypatch):
    pipeline, vendor = setup
    old = seed(pipeline, day(1), day(11))
    monkeypatch.setattr(pipeline.providers, "get", lambda *args: pytest.fail("Provider lookup on covered range"))
    monkeypatch.setattr(pipeline.store, "merge_raw", lambda *args, **kwargs: pytest.fail("Merge on covered range"))
    events = []
    outcome, = fetch(pipeline, day(3), day(8), progress=events.append).outcomes
    assert outcome.status == "skipped" and outcome.dataset_ids == (old.dataset_id,)
    assert outcome.fetch_ranges == () and outcome.fetched_row_count == 0
    assert [event["status"] for event in events] == ["skipped"]
    assert vendor.requests == []


def test_failed_later_gap_keeps_all_originals_and_continues_other_tickers(setup):
    pipeline, vendor = setup
    old = seed(pipeline, day(3), day(5))
    vendor.errors["TEST", day(5)] = RuntimeError("temporary failure")
    report = fetch(pipeline, day(1), day(8), tickers=("TEST", "OTHER"))
    assert [item.status for item in report.outcomes] == ["failed", "saved"]
    assert pipeline.get_metadata(old.dataset_id).active
    assert len(pipeline.list_datasets(DataQuery(include_history=True))) == 2
    assert report.outcomes[0].dataset_ids == ()
    assert report.outcomes[0].row_count == 0 and report.outcomes[0].fetched_row_count == 2
    assert [item["status"] for item in report.outcomes[0].fetch_ranges] == ["fetched", "failed"]


def test_empty_gap_is_reported_without_discarding_other_returned_ranges(setup):
    pipeline, vendor = setup
    old = seed(pipeline, day(3), day(5))
    vendor.errors["TEST", day(1)] = EmptyDataError("No bars")
    outcome, = fetch(pipeline, day(1), day(8)).outcomes
    assert outcome.status == "merged" and outcome.warning_message
    assert [item["status"] for item in outcome.fetch_ranges] == ["empty", "fetched"]
    assert outcome.first_timestamp == day(3) and outcome.last_timestamp == day(7)
    assert outcome.fetched_row_count == 3 and outcome.row_count == 5
    assert not pipeline.get_metadata(old.dataset_id).active


def test_all_empty_gaps_publish_nothing(setup):
    pipeline, vendor = setup
    old = seed(pipeline, day(3), day(5))
    vendor.errors = {("TEST", day(1)): EmptyDataError("No bars"), ("TEST", day(5)): EmptyDataError("No bars")}
    report = fetch(pipeline, day(1), day(8))
    assert report.outcomes[0].status == "empty" and not report.ok
    assert pipeline.list_datasets(DataQuery(include_history=True)) == [old]


def test_adjacent_only_history_is_merged_but_disconnected_history_is_left_alone(setup):
    pipeline, vendor = setup
    disconnected = seed(pipeline, day(1), day(3))
    adjacent = seed(pipeline, day(6), day(9))
    outcome, = fetch(pipeline, day(9), day(12)).outcomes
    assert outcome.replaced_dataset_ids == (adjacent.dataset_id,)
    assert [(r.start, r.end) for r in vendor.requests] == [(day(9), day(12))]
    assert pipeline.get_metadata(disconnected.dataset_id).active


def test_observer_failure_cannot_turn_committed_merge_into_failed_outcome(setup):
    pipeline, vendor = setup
    old = seed(pipeline, day(3), day(5))

    def broken_observer(event):
        raise RuntimeError("UI observer failed")

    outcome, = fetch(pipeline, day(1), day(8), progress=broken_observer).outcomes
    assert outcome.status == "merged" and not pipeline.get_metadata(old.dataset_id).active
    assert pipeline.get_metadata(outcome.dataset_ids[0]).active


def test_empty_internal_gap_is_not_filled_but_becomes_inside_merged_span(setup):
    pipeline, vendor = setup
    seed(pipeline, day(3), day(5))
    seed(pipeline, day(7), day(9))
    vendor.errors["TEST", day(5)] = EmptyDataError("No bars in middle gap")
    outcome, = fetch(pipeline, day(1), day(11)).outcomes
    assert outcome.status == "merged" and outcome.warning_message
    assert [item["status"] for item in outcome.fetch_ranges] == ["fetched", "empty", "fetched"]
    merged = pipeline.read_dataset(outcome.dataset_ids[0])
    assert merged.filter((pl.col("timestamp") >= day(5)) & (pl.col("timestamp") < day(7))).is_empty()
    # Storage intentionally tracks first-to-last spans, not bar completeness.
    assert fetch(pipeline, day(1), day(11)).outcomes[0].status == "skipped"
    assert len(vendor.requests) == 3
