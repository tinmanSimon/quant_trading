"""Missing-range acquisition stays inside the request while local merges expand."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from data_pipeline.fetch_planning import MissingFetchPlan, plan_fetch, plan_missing_fetch
from data_pipeline.models import DataRequest
from data_pipeline.storage.models import StoredDataset


def day(number):
    return datetime(2024, 10, number, tzinfo=UTC)


def request(start, end, timeframe="1d"):
    return DataRequest("TEST", start, end, timeframe=timeframe)


def stored(identifier, first, last, *, timeframe="1d", **changes):
    item = StoredDataset(
        dataset_id=identifier, layer="raw", request=request(first, last + timedelta(days=10), timeframe),
        first_timestamp=first, last_timestamp=last, row_count=2,
        relative_path=f"raw/{identifier}.parquet", checksum_sha256="test",
        schema_version=1, created_at=day(20),
    )
    return replace(item, **changes)


def ranges(plan):
    return [(item.start, item.end) for item in plan.fetch_requests]


def assert_clipped(plan):
    assert all(plan.request.start <= item.start < item.end <= plan.request.end
               for item in plan.fetch_requests)
    assert all(left.end < right.start for left, right in zip(plan.fetch_requests, plan.fetch_requests[1:]))
    assert all(replace(item, start=plan.request.start, end=plan.request.end) == plan.request
               for item in plan.fetch_requests)


@pytest.mark.parametrize("start,end,expected", [
    (day(1), day(8), [(day(1), day(4))]),
    (day(5), day(12), [(day(9), day(12))]),
    (day(1), day(12), [(day(1), day(4)), (day(9), day(12))]),
])
def test_left_right_and_both_missing_ranges_never_refetch_stored_data(start, end, expected):
    original = request(start, end)
    plan = plan_missing_fetch(original, [stored("middle", day(4), day(8))])
    assert plan.request is original
    assert ranges(plan) == expected
    assert plan.supersedes == ("middle",)
    assert plan.existing_dataset_ids == ()
    assert plan.merge_request == replace(original, start=min(start, day(4)), end=max(end, day(9)))
    assert_clipped(plan)


def test_multiple_missing_ranges_are_ordered_disjoint_and_bounded():
    original = request(day(1), day(15))
    intervals = [stored("second", day(9), day(11)), stored("first", day(3), day(5))]
    plan = plan_missing_fetch(original, iter(intervals))
    assert ranges(plan) == [(day(1), day(3)), (day(6), day(9)), (day(12), day(15))]
    assert plan.merge_request == original
    assert plan.supersedes == ("first", "second")
    assert_clipped(plan)


def test_uncovered_tail_merges_adjacent_older_data_without_fetching_it():
    original = request(day(11), day(15))
    previous = stored("older", day(1), day(10))
    plan = plan_missing_fetch(original, [previous])
    assert plan.fetch_requests == (original,)
    assert plan.merge_request == replace(original, start=day(1))
    assert plan.supersedes == ("older",)
    # Refresh mode deliberately continues to exclude endpoint-only adjacency.
    assert plan_fetch(original, [previous]).supersedes == ()


def test_adjacent_and_overlapping_chains_join_on_both_sides():
    original = request(day(10), day(12))
    intervals = [
        stored("left-far", day(1), day(4)), stored("left-near", day(5), day(9)),
        stored("right-near", day(12), day(15)), stored("right-far", day(14), day(20)),
        stored("disconnected", day(23), day(25)),
    ]
    plan = plan_missing_fetch(original, reversed(intervals))
    assert plan.fetch_requests == (original,)
    assert plan.merge_request == replace(original, start=day(1), end=day(21))
    assert plan.supersedes == ("left-far", "left-near", "right-near", "right-far")
    assert_clipped(plan)


def test_covered_union_does_not_compact_or_include_adjacent_outside_targets():
    original = request(day(4), day(11))
    intervals = [stored("outside-left", day(1), day(3)), stored("first", day(4), day(6)),
                 stored("second", day(7), day(10)), stored("outside-right", day(11), day(14))]
    plan = plan_missing_fetch(original, reversed(intervals))
    assert plan == MissingFetchPlan(original, (), original, ("first", "second"))
    assert plan.supersedes == ()
    with pytest.raises(FrozenInstanceError):
        plan.merge_request = replace(original, end=day(12))


def test_overlapping_union_has_no_false_gap_or_duplicate_fetches():
    original = request(day(1), day(15))
    intervals = [stored("wide", day(3), day(10)), stored("inner", day(5), day(7)),
                 stored("extends", day(9), day(12))]
    plan = plan_missing_fetch(original, reversed(intervals))
    assert ranges(plan) == [(day(1), day(3)), (day(13), day(15))]
    assert plan.supersedes == ("wide", "inner", "extends")
    assert_clipped(plan)


def test_fully_covered_request_uses_all_overlapping_ids_deterministically():
    original = request(day(5), day(7))
    intervals = [stored("z", day(1), day(10)), stored("b", day(1), day(7)),
                 stored("a", day(1), day(7))]
    plan = plan_missing_fetch(original, intervals)
    assert plan.fetch_requests == ()
    assert plan.existing_dataset_ids == ("a", "b", "z")
    assert plan.supersedes == ()
    assert plan.merge_request is original


def test_actual_span_not_old_request_bounds_determines_gaps_and_merge_extent():
    original = request(day(2), day(12))
    item = stored("actual", day(5), day(7), request=request(day(1), day(25)))
    plan = plan_missing_fetch(original, [item])
    assert ranges(plan) == [(day(2), day(5)), (day(8), day(12))]
    assert plan.merge_request == original
    assert plan.supersedes == ("actual",)


@pytest.mark.parametrize("field,value", [
    ("provider", "other"), ("symbol", "OTHER"), ("timeframe", "1h"),
    ("dataset", "other"), ("price_adjustment", "adjusted"),
])
def test_other_identity_cannot_cover_or_join_local_merge(field, value):
    original = request(day(4), day(7))
    item = stored("other", day(1), day(10))
    metadata_request = copy(item.request)
    object.__setattr__(metadata_request, field, value)
    item = replace(item, request=metadata_request)
    assert plan_missing_fetch(original, [item]) == MissingFetchPlan(original, (original,), original)


@pytest.mark.parametrize("changes", [{"active": False}, {"layer": "processed", "pipeline_id": "p"}])
def test_inactive_and_processed_data_are_excluded(changes):
    original = request(day(4), day(7))
    item = stored("other", day(1), day(10), **changes)
    assert plan_missing_fetch(original, [item]) == MissingFetchPlan(original, (original,), original)


@pytest.mark.parametrize("timeframe,start,boundary,end", [
    ("30m", day(1), day(1) + timedelta(minutes=30), day(1) + timedelta(hours=1)),
    ("2h", day(1), day(1) + timedelta(hours=2), day(1) + timedelta(hours=3)),
    ("1d", day(1), day(2), day(4)),
    ("2wk", day(1), day(15), day(18)),
    ("1mo", datetime(2024, 1, 31, tzinfo=UTC), datetime(2024, 2, 29, tzinfo=UTC),
     datetime(2024, 3, 1, tzinfo=UTC)),
    ("2mo", datetime(2024, 11, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
     datetime(2025, 2, 1, tzinfo=UTC)),
])
def test_missing_tail_starts_at_nominal_terminal_bar_end(timeframe, start, boundary, end):
    original = request(start, end, timeframe)
    plan = plan_missing_fetch(original, [stored("first", start, start, timeframe=timeframe)])
    assert ranges(plan) == [(boundary, end)]
    assert plan.merge_request == original
    assert plan.supersedes == ("first",)
    assert_clipped(plan)


def test_intraday_dst_boundary_uses_elapsed_utc_hour():
    local = datetime(2024, 3, 10, 1, 30, tzinfo=ZoneInfo("America/New_York"))
    boundary = datetime(2024, 3, 10, 7, 30, tzinfo=UTC)
    original = request(local, boundary + timedelta(hours=1), "1h")
    plan = plan_missing_fetch(original, [stored("dst", local, local, timeframe="1h")])
    assert ranges(plan) == [(boundary, boundary + timedelta(hours=1))]
    assert_clipped(plan)


def test_no_matching_or_nearby_data_fetches_original_without_merge_targets():
    original = request(day(10), day(12))
    for intervals in ([], [stored("before", day(1), day(5)), stored("after", day(15), day(20))]):
        plan = plan_missing_fetch(original, iter(intervals))
        assert plan == MissingFetchPlan(original, (original,), original)
        assert plan.merge_request is original
