"""Cached chart indexes must never bypass current storage verification."""

from dataclasses import replace
from datetime import timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from dashboard.chart_cache import BoundedChartCache
from data_pipeline import DataIntegrityError, DataPipeline, DataQuery, DataRequest, DatasetNotFoundError


@pytest.fixture
def stored_chart(tmp_path, sample_ohlcv_frame):
    pipeline = DataPipeline(tmp_path / "data")
    request = DataRequest("AAPL", sample_ohlcv_frame["timestamp"][0],
                          sample_ohlcv_frame["timestamp"][-1] + timedelta(hours=1))
    dataset = pipeline.ingest_frame(request, sample_ohlcv_frame).raw
    query = DataQuery(provider="yahoo", symbol="AAPL", timeframe="1h",
                      start=request.start, end=request.end)
    return pipeline, dataset, query, sample_ohlcv_frame


@pytest.mark.parametrize("explicit", [False, True])
def test_cache_reuses_preparation_but_reads_verified_storage_on_every_load(stored_chart, monkeypatch, explicit):
    pipeline, dataset, query, frame = stored_chart
    method = "read_dataset" if explicit else "read"
    original = getattr(pipeline, method)
    calls = []

    def read(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, method, read)
    cache = BoundedChartCache()
    arguments = {"dataset_id": dataset.dataset_id} if explicit else {}
    first = cache.load(pipeline, query, timeframe="1h", **arguments)
    assert cache.load(pipeline, query, timeframe="1h", **arguments) is first
    assert len(calls) == 2
    assert len(cache) == 1
    assert cache.byte_size == first.estimated_size
    assert_frame_equal(first.frame, frame)


def test_replacement_refreshes_series_but_explicit_historical_revision_still_works(stored_chart):
    pipeline, dataset, query, frame = stored_chart
    cache = BoundedChartCache()
    old = cache.load(pipeline, query, timeframe="1h")
    changed = frame.with_columns(pl.col("volume") + 0.125)
    pipeline.store.replace_raw(dataset.dataset_id, changed, confirm=dataset.dataset_id)

    current = cache.load(pipeline, query, timeframe="1h")
    assert current is not old
    assert_frame_equal(current.frame, changed)
    history = cache.load(pipeline, query, timeframe="1h", dataset_id=dataset.dataset_id)
    assert_frame_equal(history.frame, frame)


def test_partial_deletion_updates_chart_without_stale_rows(stored_chart):
    pipeline, _, query, frame = stored_chart
    cache = BoundedChartCache()
    original = cache.load(pipeline, query, timeframe="1h")
    plan = pipeline.plan_delete(replace(query, start=frame["timestamp"][1], end=frame["timestamp"][2]))
    pipeline.delete(plan, confirm=plan.operation_id)

    current = cache.load(pipeline, query, timeframe="1h")
    assert current is not original
    assert current.timestamps == (frame["timestamp"][0], frame["timestamp"][2])


@pytest.mark.parametrize("explicit", [False, True])
def test_complete_deletion_never_returns_cached_chart(stored_chart, explicit):
    pipeline, dataset, query, _ = stored_chart
    cache = BoundedChartCache()
    arguments = {"dataset_id": dataset.dataset_id} if explicit else {}
    assert cache.load(pipeline, query, timeframe="1h", **arguments) is not None
    plan = pipeline.plan_delete(query)
    pipeline.delete(plan, confirm=plan.operation_id)

    if explicit:
        with pytest.raises(DatasetNotFoundError):
            cache.load(pipeline, query, timeframe="1h", **arguments)
    else:
        assert cache.load(pipeline, query, timeframe="1h") is None
    assert len(cache) == cache.byte_size == 0


@pytest.mark.parametrize("damage", ["corruption", "missing"])
def test_cached_chart_does_not_mask_changed_or_missing_parquet(stored_chart, damage):
    pipeline, dataset, query, _ = stored_chart
    cache = BoundedChartCache()
    cache.load(pipeline, query, timeframe="1h")
    path = pipeline.store.data_dir / dataset.relative_path
    if damage == "corruption":
        content = bytearray(path.read_bytes())
        content[len(content) // 2] ^= 1
        path.write_bytes(content)
    else:
        path.unlink()

    with pytest.raises(DataIntegrityError):
        cache.load(pipeline, query, timeframe="1h")
    assert len(cache) == cache.byte_size == 0


def test_explicit_revision_filters_exact_microsecond_bounds(stored_chart):
    pipeline, dataset, query, frame = stored_chart
    query = replace(query, start=frame["timestamp"][0] + timedelta(microseconds=1),
                    end=frame["timestamp"][2] + timedelta(microseconds=1))
    series = BoundedChartCache().load(pipeline, query, dataset_id=dataset.dataset_id, timeframe="1h")
    assert_frame_equal(series.frame, frame.tail(2))


def test_lru_entry_limit_evicts_least_recently_viewed_series(stored_chart):
    pipeline, _, query, frame = stored_chart
    cache = BoundedChartCache(max_entries=2)
    queries = [replace(query, start=stamp, end=stamp + timedelta(hours=1)) for stamp in frame["timestamp"]]
    first = cache.load(pipeline, queries[0], timeframe="1h")
    second = cache.load(pipeline, queries[1], timeframe="1h")
    assert cache.load(pipeline, queries[0], timeframe="1h") is first
    cache.load(pipeline, queries[2], timeframe="1h")
    assert len(cache) == 2
    assert cache.load(pipeline, queries[0], timeframe="1h") is first
    assert cache.load(pipeline, queries[1], timeframe="1h") is not second
    assert len(cache) == 2


def test_byte_budget_evicts_old_series_and_skips_oversized_series(stored_chart):
    pipeline, _, query, frame = stored_chart
    one = replace(query, end=frame["timestamp"][1])
    other = replace(query, start=frame["timestamp"][1], end=frame["timestamp"][2])
    size = BoundedChartCache().load(pipeline, one, timeframe="1h").estimated_size
    cache = BoundedChartCache(max_bytes=size)
    first = cache.load(pipeline, one, timeframe="1h")
    assert len(cache) == 1 and cache.byte_size <= size
    cache.load(pipeline, other, timeframe="1h")
    assert len(cache) == 1 and cache.byte_size <= size
    assert cache.load(pipeline, one, timeframe="1h") is not first

    large = cache.load(pipeline, query, timeframe="1h")
    assert large.estimated_size > size
    assert cache.load(pipeline, query, timeframe="1h") is not large
    assert cache.byte_size <= size


def test_source_root_query_options_and_omission_changes_do_not_alias(stored_chart, tmp_path):
    pipeline, dataset, query, frame = stored_chart
    cache = BoundedChartCache(max_entries=10)
    original = cache.load(pipeline, query, timeframe="1h")
    other_root = DataPipeline(tmp_path / "other")
    other_root.ingest_frame(dataset.request, frame)
    assert cache.load(other_root, query, timeframe="1h") is not original
    assert cache.load(pipeline, query, timeframe="1h", session_timezone="UTC") is not original
    assert cache.load(pipeline, query, timeframe="2h") is not original
    assert cache.load(pipeline, replace(query, end=query.end + timedelta(hours=1)), timeframe="1h") is not original
    omitted = frame["timestamp"][0] + timedelta(minutes=1)
    with_omission = cache.load(pipeline, query, timeframe="1h", omitted_timestamps=[omitted])
    assert with_omission is not original
    assert with_omission.omitted_timestamps == (omitted,)
    assert cache.load(pipeline, query, timeframe="1h", omitted_timestamps=[omitted, omitted]) is with_omission


@pytest.mark.parametrize("limit", [{"max_bytes": 0}, {"max_entries": 0}])
def test_zero_budget_disables_retention_without_disabling_charts(stored_chart, limit):
    pipeline, _, query, _ = stored_chart
    cache = BoundedChartCache(**limit)
    first = cache.load(pipeline, query, timeframe="1h")
    assert first is not None
    assert cache.load(pipeline, query, timeframe="1h") is not first
    assert len(cache) == cache.byte_size == 0


@pytest.mark.parametrize("kwargs", [{"max_bytes": -1}, {"max_bytes": True}, {"max_entries": 1.5}])
def test_invalid_cache_limits_are_rejected(kwargs):
    with pytest.raises(ValueError, match="nonnegative integer"):
        BoundedChartCache(**kwargs)
