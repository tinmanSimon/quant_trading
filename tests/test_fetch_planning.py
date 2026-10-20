"""Actual-coverage and strict-overlap planning without provider or store writes."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from data_pipeline.fetch_planning import FetchPlan, plan_fetch
from data_pipeline.models import DataRequest
from data_pipeline.storage.models import StoredDataset


def stamp(day, month=10, year=2024, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def request(start, end, timeframe="1d", **kwargs):
    return DataRequest("TEST", start, end, timeframe=timeframe, **kwargs)


def stored(identifier, first, last, *, timeframe="1d", bounds=None, **changes):
    item = StoredDataset(
        dataset_id=identifier, layer="raw",
        request=request(*(bounds or (first, last + timedelta(days=10))), timeframe=timeframe),
        first_timestamp=first, last_timestamp=last, row_count=2,
        relative_path=f"raw/{identifier}.parquet", checksum_sha256="test",
        schema_version=1, created_at=stamp(20),
    )
    return replace(item, **changes)


def test_contained_request_uses_actual_span_and_returns_immutable_skip():
    item = stored("one", stamp(1), stamp(10))
    original = request(stamp(3), stamp(11))
    plan = plan_fetch(original, [item])
    assert plan == FetchPlan(original, None, ("one",), ())
    assert plan.request is original
    with pytest.raises(FrozenInstanceError):
        plan.fetch_request = original
    # Coverage is intentionally range-based, even when only endpoint rows exist.
    assert item.row_count == 2


def test_adjacent_ranges_jointly_cover_request_in_deterministic_order():
    first = stored("earlier", stamp(1), stamp(5))
    second = stored("later", stamp(6), stamp(10))
    original = request(stamp(3), stamp(11))
    assert plan_fetch(original, iter([second, first])) == FetchPlan(
        original, None, ("earlier", "later"), (),
    )


def test_overlapping_coverage_orders_ties_by_terminal_then_identifier():
    datasets = [stored("z", stamp(1), stamp(10)), stored("b", stamp(1), stamp(6)),
                stored("a", stamp(1), stamp(6)), stored("outside", stamp(12), stamp(15))]
    original = request(stamp(2), stamp(9))
    assert plan_fetch(original, datasets).existing_dataset_ids == ("a", "b", "z")


@pytest.mark.parametrize("start,end", [(stamp(8), stamp(13)), (stamp(1), stamp(8))])
def test_partial_overlap_fetches_envelope_and_keeps_original_request(start, end):
    item = stored("one", stamp(3), stamp(10))
    original = request(start, end)
    plan = plan_fetch(original, [item])
    assert plan.request is original
    assert plan.existing_dataset_ids == ()
    assert plan.supersedes == ("one",)
    assert plan.fetch_request == replace(original, start=min(start, stamp(3)), end=max(end, stamp(11)))


def test_encompassing_several_ranges_and_gap_refreshes_all_overlaps():
    first = stored("first", stamp(1), stamp(4))
    second = stored("second", stamp(8), stamp(10))
    for original in (request(stamp(2), stamp(10)), request(stamp(1), stamp(12))):
        plan = plan_fetch(original, [second, first])
        assert plan.fetch_request == replace(original, start=stamp(1), end=max(stamp(11), original.end))
        assert plan.supersedes == ("first", "second")
        assert plan.existing_dataset_ids == ()


def test_range_envelope_does_not_mistake_disconnected_spans_for_coverage():
    original = request(stamp(5), stamp(7))
    intervals = [stored("before", stamp(1), stamp(4)), stored("after", stamp(7), stamp(10))]
    plan = plan_fetch(original, intervals)
    assert plan.fetch_request is original
    assert plan.supersedes == ()
    assert plan.existing_dataset_ids == ()


def test_refresh_follows_entire_overlap_chain_but_not_adjacent_component():
    intervals = [stored("first", stamp(1), stamp(8)), stored("second", stamp(7), stamp(12)),
                 stored("third", stamp(11), stamp(16)), stored("adjacent", stamp(18), stamp(21))]
    original = request(stamp(15), stamp(18))
    plan = plan_fetch(original, reversed(intervals))
    assert plan.fetch_request == replace(original, start=stamp(1))
    assert plan.supersedes == ("first", "second", "third")


@pytest.mark.parametrize("start,end", [(stamp(11), stamp(13)), (stamp(1), stamp(3))])
def test_touching_endpoint_alone_does_not_refresh_existing_data(start, end):
    original = request(start, end)
    plan = plan_fetch(original, [stored("touching", stamp(3), stamp(10))])
    assert plan.fetch_request is original
    assert plan.supersedes == ()


def test_actual_coverage_overrides_wider_vendor_request_bounds():
    item = stored("one", stamp(4), stamp(6), bounds=(stamp(1), stamp(20)))
    missing = request(stamp(1), stamp(3))
    assert plan_fetch(missing, [item]) == FetchPlan(missing, missing)
    overlap = request(stamp(5), stamp(10))
    plan = plan_fetch(overlap, [item])
    assert plan.fetch_request == replace(overlap, start=stamp(4))
    assert plan.fetch_request.end == stamp(10)
    assert plan.supersedes == ("one",)


@pytest.mark.parametrize("field,value", [
    ("provider", "other"), ("symbol", "OTHER"), ("timeframe", "1h"),
    ("dataset", "other"), ("price_adjustment", "adjusted"),
])
def test_identity_mismatches_do_not_cover_or_supersede(field, value):
    original = request(stamp(2), stamp(4))
    item = stored("other", stamp(1), stamp(10))
    metadata_request = copy(item.request)
    # Persisted metadata may come from an older/future provider policy; the
    # planner must compare every identity field, including unsupported values.
    object.__setattr__(metadata_request, field, value)
    item = replace(item, request=metadata_request)
    assert plan_fetch(original, [item]) == FetchPlan(original, original)


@pytest.mark.parametrize("changes", [{"active": False}, {"layer": "processed", "pipeline_id": "p"}])
def test_inactive_and_processed_datasets_are_excluded(changes):
    original = request(stamp(2), stamp(4))
    item = stored("excluded", stamp(1), stamp(10), **changes)
    assert plan_fetch(original, [item]) == FetchPlan(original, original)


@pytest.mark.parametrize("timeframe,last,end", [
    ("90m", stamp(1), stamp(1, hour=1, minute=30)),
    ("2h", stamp(1), stamp(1, hour=2)),
    ("3d", stamp(1), stamp(4)),
    ("2wk", stamp(1), stamp(15)),
    ("1mo", stamp(1, month=2), stamp(1, month=3)),
    ("2mo", stamp(1, month=11), stamp(1, month=1, year=2025)),
])
def test_terminal_bar_contributes_its_nominal_timeframe(timeframe, last, end):
    item = stored("terminal", last, last, timeframe=timeframe)
    original = request(last, end, timeframe=timeframe)
    assert plan_fetch(original, [item]) == FetchPlan(original, None, ("terminal",))
    after = request(end, end + timedelta(days=1), timeframe=timeframe)
    assert plan_fetch(after, [item]) == FetchPlan(after, after)


def test_calendar_month_end_uses_calendar_advancement_in_leap_year():
    first, end = stamp(31, month=1), stamp(29, month=2)
    item = stored("month", first, first, timeframe="1mo")
    original = request(first, end, timeframe="1mo")
    assert plan_fetch(original, [item]).fetch_request is None
    expanded = request(stamp(1, month=2), stamp(1, month=3), timeframe="1mo")
    assert plan_fetch(expanded, [item]).fetch_request == replace(expanded, start=first)


def test_intraday_end_uses_utc_elapsed_time_across_dst_change():
    local = datetime(2024, 3, 10, 1, 30, tzinfo=ZoneInfo("America/New_York"))
    item = stored("dst", local, local, timeframe="1h")
    end = datetime(2024, 3, 10, 7, 30, tzinfo=UTC)
    original = request(local, end, timeframe="1h")
    assert plan_fetch(original, [item]) == FetchPlan(original, None, ("dst",))
    after = request(end, end + timedelta(hours=1), timeframe="1h")
    assert plan_fetch(after, [item]) == FetchPlan(after, after)


def test_daily_date_labels_are_not_shifted_by_dst_or_weekend_guesses():
    first = datetime(2024, 3, 8, tzinfo=UTC)  # Friday; no exchange-calendar assumptions.
    end = first + timedelta(days=1)
    item = stored("day", first, first)
    original = request(first, end)
    assert plan_fetch(original, [item]).fetch_request is None
    weekend = request(end, end + timedelta(days=2))
    assert plan_fetch(weekend, [item]) == FetchPlan(weekend, weekend)


def test_no_datasets_preserves_exact_request():
    original = request(stamp(1), stamp(5), timeframe="1h", provider="other")
    assert plan_fetch(original, iter(())) == FetchPlan(original, original)
