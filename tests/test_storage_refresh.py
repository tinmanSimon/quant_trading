"""Atomic refresh of expanded raw ranges, including complete bar preservation."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.exceptions import (
    DataIntegrityError,
    DatasetNotFoundError,
    OverlappingDataError,
    RequestDataMismatchError,
    StorageWriteError,
)
from data_pipeline.models import DataQuery, DataRequest
from data_pipeline.storage import store as store_module
from data_pipeline.storage.store import LocalDataStore


@pytest.fixture
def request_range():
    return DataRequest(
        symbol="AAPL", provider="yahoo", timeframe="1h",
        start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 17, 30, tzinfo=UTC),
    )


def _pipeline():
    processors_json = json.dumps(
        [{"name": "keep-bars", "version": "1", "config": {}}],
        sort_keys=True, separators=(",", ":"),
    )
    return {"pipeline_id": sha256(processors_json.encode()).hexdigest(),
            "processors_json": processors_json}


def _snapshot(store):
    items = sorted(
        (item for layer in ("raw", "processed")
         for item in store.list_datasets(DataQuery(layer=layer, include_history=True))),
        key=lambda item: item.dataset_id,
    )
    files = {
        str(path.relative_to(store.data_dir)): path.read_bytes()
        for folder in ("raw", "processed", "staging", "quarantine")
        for path in (store.data_dir / folder).rglob("*") if path.is_file()
    }
    return items, files


def _extended(request, frame):
    request = replace(request, start=request.start - timedelta(hours=1),
                      end=request.end + timedelta(hours=1))
    frame = pl.concat([
        frame.head(1).with_columns(pl.col("timestamp") - timedelta(hours=1)),
        frame,
        frame.tail(1).with_columns(pl.col("timestamp") + timedelta(hours=1)),
    ]).with_columns(pl.col("volume") + 10)
    return request, frame


def test_refresh_expands_multiple_raw_ranges_preserves_history_and_retires_derivatives(
    isolated_data_dir, request_range, sample_ohlcv_frame,
):
    store = LocalDataStore(isolated_data_dir)
    first = store.write_raw(request_range, sample_ohlcv_frame.head(1))
    last = store.write_raw(request_range, sample_ohlcv_frame.tail(1))
    existing = pl.concat([sample_ohlcv_frame.head(1), sample_ohlcv_frame.tail(1)])
    derived = store.write_processed([first.dataset_id, last.dataset_id], existing, **_pipeline())
    other = store.write_raw(replace(request_range, provider="other"), sample_ohlcv_frame)
    unaffected = store.write_processed(other.dataset_id, sample_ohlcv_frame, **_pipeline())
    original_files = _snapshot(store)[1]
    expanded_request, response = _extended(request_range, sample_ohlcv_frame)

    fresh = store.refresh_raw(expanded_request, response,
                              supersedes=[last.dataset_id, first.dataset_id])

    restarted = LocalDataStore(isolated_data_dir)
    assert fresh.request == expanded_request
    assert fresh.supersedes == (last.dataset_id, first.dataset_id)
    assert fresh.row_count == 5
    assert fresh.first_timestamp == expanded_request.start
    assert fresh.last_timestamp == expanded_request.end - timedelta(hours=1)
    assert fresh.active
    assert {item.dataset_id for item in restarted.list_raw()} == {fresh.dataset_id, other.dataset_id}
    assert restarted.list_datasets(DataQuery(layer="processed")) == [unaffected]
    assert_frame_equal(restarted.read(DataQuery(provider="yahoo")), response)
    for item, original in ((first, sample_ohlcv_frame.head(1)),
                           (last, sample_ohlcv_frame.tail(1)), (derived, existing)):
        assert not restarted.get_metadata(item.dataset_id).active
        assert_frame_equal(restarted.read_dataset(item.dataset_id), original)
    for path, contents in original_files.items():
        assert (isolated_data_dir / path).read_bytes() == contents
    assert set(restarted.audit()) == {
        first.dataset_id, last.dataset_id, derived.dataset_id, other.dataset_id,
        unaffected.dataset_id, fresh.dataset_id,
    }
    assert restarted.recover() == []


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_refresh_rejects_missing_existing_boundary_or_interior_bar(
    isolated_data_dir, request_range, sample_ohlcv_frame, missing_index,
):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request_range, sample_ohlcv_frame)
    derived = store.write_processed(old.dataset_id, sample_ohlcv_frame, **_pipeline())
    before = _snapshot(store)
    expanded_request, response = _extended(request_range, sample_ohlcv_frame)
    missing_timestamp = sample_ohlcv_frame["timestamp"][missing_index]
    response = response.filter(pl.col("timestamp") != missing_timestamp)
    # The response still extends beyond both old boundaries. Range comparison
    # alone would incorrectly accept it even when an interior bar is missing.
    assert response["timestamp"].min() < old.first_timestamp
    assert response["timestamp"].max() > old.last_timestamp

    with pytest.raises(RequestDataMismatchError, match="omits 1 existing bar"):
        store.refresh_raw(expanded_request, response, supersedes=[old.dataset_id])

    assert _snapshot(store) == before
    assert store.get_metadata(derived.dataset_id).active


@pytest.mark.parametrize("targets", [None, [], (), "one-id", b"one-id", 42, [""], [None], ["id", "id"]])
def test_refresh_requires_nonempty_distinct_target_id_sequence(
    isolated_data_dir, request_range, sample_ohlcv_frame, targets,
):
    store = LocalDataStore(isolated_data_dir)
    before = _snapshot(store)
    with pytest.raises(RequestDataMismatchError):
        store.refresh_raw(request_range, sample_ohlcv_frame, supersedes=targets)
    assert _snapshot(store) == before


@pytest.mark.parametrize("case", ["missing", "processed", "inactive", "provider", "symbol", "timeframe"])
def test_refresh_rejects_invalid_or_mixed_target_identity(
    isolated_data_dir, request_range, sample_ohlcv_frame, case,
):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request_range, sample_ohlcv_frame)
    ids = [old.dataset_id]
    response = sample_ohlcv_frame
    request = request_range
    if case == "missing":
        ids.append("does-not-exist")
    elif case == "processed":
        ids.append(store.write_processed(old.dataset_id, response, **_pipeline()).dataset_id)
    elif case == "inactive":
        # A concurrent fetch has already replaced the target chosen by the planner.
        store.replace_raw(old.dataset_id, response, confirm=old.dataset_id)
    else:
        changes = {"provider": "other", "symbol": "MSFT", "timeframe": "2h"}
        foreign_request = replace(request_range, **{case: changes[case]})
        foreign_frame = response.with_columns(pl.lit(foreign_request.symbol).alias("symbol"))
        foreign = store.write_raw(foreign_request, foreign_frame)
        ids.append(foreign.dataset_id)
    before = _snapshot(store)

    expected = DatasetNotFoundError if case == "missing" else RequestDataMismatchError
    with pytest.raises(expected):
        store.refresh_raw(request, response, supersedes=ids)

    assert _snapshot(store) == before


def test_refresh_validates_its_expanded_request_bounds(
    isolated_data_dir, request_range, sample_ohlcv_frame,
):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request_range, sample_ohlcv_frame)
    before = _snapshot(store)
    _, response = _extended(request_range, sample_ohlcv_frame)
    with pytest.raises(RequestDataMismatchError, match="outside"):
        store.refresh_raw(request_range, response, supersedes=[old.dataset_id])
    assert _snapshot(store) == before


def test_refresh_checks_request_identity_even_for_single_target(
    isolated_data_dir, request_range, sample_ohlcv_frame,
):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request_range, sample_ohlcv_frame)
    before = _snapshot(store)
    with pytest.raises(RequestDataMismatchError, match="match the request"):
        store.refresh_raw(replace(request_range, provider="other"), sample_ohlcv_frame,
                          supersedes=[old.dataset_id])
    assert _snapshot(store) == before


def test_refresh_rejects_overlapping_dataset_added_after_planning(
    isolated_data_dir, request_range, sample_ohlcv_frame,
):
    store = LocalDataStore(isolated_data_dir)
    planned = store.write_raw(request_range, sample_ohlcv_frame.head(1))
    # Another writer publishes into the requested extension after target IDs
    # have been selected, so refresh must fail without replacing either writer.
    concurrent = store.write_raw(request_range, sample_ohlcv_frame.tail(1))
    before = _snapshot(store)
    with pytest.raises(OverlappingDataError, match=concurrent.dataset_id):
        store.refresh_raw(request_range, sample_ohlcv_frame, supersedes=[planned.dataset_id])
    assert _snapshot(store) == before


def test_refresh_verifies_existing_files_before_replacing_them(
    isolated_data_dir, request_range, sample_ohlcv_frame,
):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request_range, sample_ohlcv_frame)
    (isolated_data_dir / old.relative_path).write_bytes(b"corrupt parquet")
    before = _snapshot(store)
    with pytest.raises(DataIntegrityError, match="Checksum mismatch"):
        store.refresh_raw(request_range, sample_ohlcv_frame, supersedes=[old.dataset_id])
    assert _snapshot(store) == before


@pytest.mark.parametrize("failure_point", ["writer", "readback", "publish", "catalog", "sync"])
def test_refresh_failure_preserves_all_originals_and_derived_outputs(
    isolated_data_dir, request_range, sample_ohlcv_frame, monkeypatch, failure_point,
):
    store = LocalDataStore(isolated_data_dir)
    first = store.write_raw(request_range, sample_ohlcv_frame.head(1))
    last = store.write_raw(request_range, sample_ohlcv_frame.tail(1))
    old_frame = pl.concat([sample_ohlcv_frame.head(1), sample_ohlcv_frame.tail(1)])
    store.write_processed([first.dataset_id, last.dataset_id], old_frame, **_pipeline())
    before = _snapshot(store)
    expanded_request, response = _extended(request_range, sample_ohlcv_frame)
    original_publish = store_module._publish
    original_insert = store_module.catalog.insert
    original_write = pl.DataFrame.write_parquet

    def fail_writer(self, handle, **kwargs):
        handle.write(b"partial parquet")
        raise OSError("injected write failure")

    def bad_writer(self, handle, **kwargs):
        return original_write(self.with_columns(pl.col("volume") + 1), handle, **kwargs)

    def fail_publish(staged, final):
        original_publish(staged, final)
        raise OSError("injected publication failure")

    def fail_insert(connection, item):
        original_insert(connection, item)
        raise RuntimeError("injected catalog failure")

    def fail_sync(path):
        raise OSError("injected sync failure")

    with monkeypatch.context() as patch:
        if failure_point in {"writer", "readback"}:
            patch.setattr(pl.DataFrame, "write_parquet", fail_writer if failure_point == "writer" else bad_writer)
        elif failure_point == "publish":
            patch.setattr(store_module, "_publish", fail_publish)
        elif failure_point == "catalog":
            patch.setattr(store_module.catalog, "insert", fail_insert)
        else:
            patch.setattr(store_module, "_sync_directory", fail_sync)
        expected = DataIntegrityError if failure_point == "readback" else StorageWriteError
        with pytest.raises(expected):
            store.refresh_raw(expanded_request, response, supersedes=[first.dataset_id, last.dataset_id])

    restarted = LocalDataStore(isolated_data_dir)
    assert _snapshot(restarted) == before
    assert_frame_equal(restarted.read(DataQuery()), old_frame)
    assert restarted.recover() == []
    retry = restarted.refresh_raw(expanded_request, response, supersedes=[first.dataset_id, last.dataset_id])
    assert_frame_equal(restarted.read_dataset(retry.dataset_id), response)
