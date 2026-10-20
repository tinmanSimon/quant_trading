"""Atomic missing-range merges preserve old history without refetching it."""

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.exceptions import (
    DataIntegrityError, DatasetNotFoundError, DuplicateBarError, EmptyDataError,
    OverlappingDataError, RequestDataMismatchError, StorageWriteError,
)
from data_pipeline.models import DataQuery, DataRequest
from data_pipeline.quality import FetchQuality, FetchResult, OmittedBar
from data_pipeline.schemas import OHLCV_SCHEMA
from data_pipeline.storage import store as store_module
from data_pipeline.storage.store import LocalDataStore


def day(number):
    return datetime(2024, 10, number, tzinfo=UTC)


def request(start=1, end=20, **identity):
    return DataRequest("AAPL", day(start), day(end), timeframe="1d", provider="massive", **identity)


def frame(*days, symbol="AAPL", price=100):
    return pl.DataFrame([
        {"timestamp": day(number), "symbol": symbol, "open": price, "high": price + 1,
         "low": price - 1, "close": price + 0.5, "volume": number * 100}
        for number in days
    ], schema=OHLCV_SCHEMA)


def fetched(start, end, *days, quality=None):
    return request(start, end), FetchResult(frame(*days), quality or FetchQuality())


def pipeline():
    processors = json.dumps([{"name": "keep-bars", "version": "1", "config": {}}],
                            sort_keys=True, separators=(",", ":"))
    return {"pipeline_id": sha256(processors.encode()).hexdigest(), "processors_json": processors}


def snapshot(store):
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


def test_missing_merge_retains_old_prices_and_publishes_multiple_gaps_once(isolated_data_dir, monkeypatch):
    store = LocalDataStore(isolated_data_dir)
    first_frame, last_frame = frame(4, 5, price=70), frame(8, 9, price=80)
    first = store.write_raw(request(4, 6), first_frame)
    last = store.write_raw(request(8, 10), last_frame)
    old_combined = pl.concat([first_frame, last_frame])
    derived = store.write_processed([first.dataset_id, last.dataset_id], old_combined, **pipeline())
    foreign = store.write_raw(replace(request(4, 6), provider="other"), first_frame)
    unaffected = store.write_processed(foreign.dataset_id, first_frame, **pipeline())
    old_files = snapshot(store)[1]
    publish = store_module._publish
    publications = []

    def record_publish(staged, final):
        publications.append(final)
        return publish(staged, final)

    monkeypatch.setattr(store_module, "_publish", record_publish)
    chunks = [fetched(10, 13, 10, 11, 12), fetched(1, 4, 1, 2, 3), fetched(6, 8, 6, 7)]
    merged = store.merge_raw(request(1, 13), chunks, supersedes=[last.dataset_id, first.dataset_id])

    assert len(publications) == 1
    assert merged.request == request(1, 13)
    assert merged.supersedes == (last.dataset_id, first.dataset_id)
    assert merged.first_timestamp == day(1)
    assert merged.last_timestamp == day(12)
    assert merged.row_count == 12
    expected = pl.concat([frame(1, 2, 3), first_frame, frame(6, 7), last_frame, frame(10, 11, 12)])
    assert_frame_equal(store.read_dataset(merged.dataset_id), expected)
    assert_frame_equal(store.read_dataset(first.dataset_id), first_frame)
    assert_frame_equal(store.read_dataset(last.dataset_id), last_frame)
    assert_frame_equal(store.read_dataset(derived.dataset_id), old_combined)
    assert {item.dataset_id for item in store.list_raw()} == {merged.dataset_id, foreign.dataset_id}
    assert store.list_datasets(DataQuery(layer="processed")) == [unaffected]
    for item in (first, last, derived):
        assert not store.get_metadata(item.dataset_id).active
    for path, content in old_files.items():
        assert (isolated_data_dir / path).read_bytes() == content
    assert store.recover() == []
    assert set(store.audit()) == {first.dataset_id, last.dataset_id, merged.dataset_id,
                                  derived.dataset_id, foreign.dataset_id, unaffected.dataset_id}


def test_missing_merge_without_existing_targets_creates_one_raw_dataset(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    merged = store.merge_raw(request(1, 7), [fetched(4, 7, 4, 6), fetched(1, 4, 1, 3)])
    assert merged.supersedes == ()
    assert store.list_raw() == [merged]
    assert_frame_equal(store.read_dataset(merged.dataset_id), frame(1, 3, 4, 6))


@pytest.mark.parametrize("chunks", [None, [], (), "invalid", [None], [(None, None)], [(request(),)],
                                     [(request(), frame(1))]])
def test_missing_merge_requires_nonempty_request_result_pairs(isolated_data_dir, chunks):
    store = LocalDataStore(isolated_data_dir)
    before = snapshot(store)
    with pytest.raises(RequestDataMismatchError):
        store.merge_raw(request(), chunks)
    assert snapshot(store) == before


@pytest.mark.parametrize("targets", [None, "id", b"id", [None], [""], ["same", "same"]])
def test_missing_merge_requires_distinct_target_id_sequence(isolated_data_dir, targets):
    store = LocalDataStore(isolated_data_dir)
    before = snapshot(store)
    with pytest.raises(RequestDataMismatchError):
        store.merge_raw(request(), [fetched(1, 2, 1)], supersedes=targets)
    assert snapshot(store) == before


def test_missing_merge_rejects_empty_chunk_without_publishing_other_chunks(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    before = snapshot(store)
    chunks = [fetched(1, 3, 1, 2), (request(5, 7), FetchResult(frame()))]
    with pytest.raises(EmptyDataError):
        store.merge_raw(request(1, 7), chunks, supersedes=[old.dataset_id])
    assert snapshot(store) == before


@pytest.mark.parametrize("case", ["provider", "symbol", "timeframe", "outside-request", "out-of-gap"])
def test_missing_merge_validates_each_gap_identity_and_its_own_bounds(isolated_data_dir, case):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    before = snapshot(store)
    gap, response = fetched(5, 7, 5, 6)
    if case in {"provider", "symbol", "timeframe"}:
        values = {"provider": "other", "symbol": "MSFT", "timeframe": "1h"}
        gap = replace(gap, **{case: values[case]})
        if case == "symbol":
            response = FetchResult(frame(5, 6, symbol="MSFT"))
    elif case == "outside-request":
        gap, response = fetched(6, 8, 6, 7)
    else:
        response = FetchResult(frame(5, 7))
    with pytest.raises(RequestDataMismatchError):
        store.merge_raw(request(3, 7), [(gap, response)], supersedes=[old.dataset_id])
    assert snapshot(store) == before


@pytest.mark.parametrize("changed", [False, True])
def test_missing_merge_rejects_incoming_duplicates_of_old_bars(isolated_data_dir, changed):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    before = snapshot(store)
    response = frame(4, 5, price=120 if changed else 100)
    with pytest.raises(RequestDataMismatchError, match="overlap stored coverage"):
        store.merge_raw(request(3, 6), [(request(4, 6), FetchResult(response))], supersedes=[old.dataset_id])
    assert snapshot(store) == before


def test_missing_merge_does_not_repair_holes_inside_old_dataset_span(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 8), frame(3, 7))
    before = snapshot(store)
    with pytest.raises(RequestDataMismatchError, match="overlap stored coverage"):
        store.merge_raw(request(3, 8), [fetched(4, 7, 4, 5, 6)], supersedes=[old.dataset_id])
    assert snapshot(store) == before


def test_missing_merge_rejects_overlapping_gap_requests_even_without_duplicate_rows(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    before = snapshot(store)
    with pytest.raises(RequestDataMismatchError, match="disjoint"):
        store.merge_raw(request(1, 7), [fetched(1, 5, 1, 2), fetched(4, 7, 5, 6)])
    assert snapshot(store) == before


@pytest.mark.parametrize("changed", [False, True])
def test_missing_merge_rejects_duplicate_bars_between_chunks(isolated_data_dir, changed):
    store = LocalDataStore(isolated_data_dir)
    before = snapshot(store)
    second = FetchResult(frame(3, 4, price=120 if changed else 100))
    with pytest.raises(RequestDataMismatchError, match="disjoint"):
        store.merge_raw(request(1, 5), [fetched(1, 4, 1, 2, 3), (request(3, 5), second)])
    assert snapshot(store) == before


def test_missing_merge_rejects_duplicate_bars_inside_one_chunk(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    before = snapshot(store)
    with pytest.raises(DuplicateBarError):
        store.merge_raw(request(1, 4), [(request(1, 4), FetchResult(frame(1, 2, 2, 3)))])
    assert snapshot(store) == before


@pytest.mark.parametrize("case", ["inactive", "processed", "foreign", "missing", "old-outside-envelope"])
def test_missing_merge_rechecks_targets_before_publication(isolated_data_dir, case):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    targets = [old.dataset_id]
    full = request(1, 7)
    if case == "inactive":
        store.replace_raw(old.dataset_id, frame(3, 4), confirm=old.dataset_id)
    elif case == "processed":
        targets.append(store.write_processed(old.dataset_id, frame(3, 4), **pipeline()).dataset_id)
    elif case == "foreign":
        targets.append(store.write_raw(replace(request(3, 5), provider="other"), frame(3, 4)).dataset_id)
    elif case == "missing":
        targets.append("not-in-catalog")
    else:
        full = request(4, 7)
    before = snapshot(store)
    expected = DatasetNotFoundError if case == "missing" else RequestDataMismatchError
    with pytest.raises(expected):
        store.merge_raw(full, [fetched(5, 7, 5, 6)], supersedes=targets)
    assert snapshot(store) == before


def test_missing_merge_checks_concurrent_foreign_overlaps_without_retiring_old_data(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    store.write_processed(old.dataset_id, frame(3, 4), **pipeline())
    concurrent = store.write_raw(request(6, 7), frame(6))
    before = snapshot(store)
    with pytest.raises(OverlappingDataError, match=concurrent.dataset_id):
        store.merge_raw(request(3, 8), [fetched(5, 8, 5, 6, 7)], supersedes=[old.dataset_id])
    assert snapshot(store) == before


def test_missing_merge_verifies_original_checksums(isolated_data_dir):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    (isolated_data_dir / old.relative_path).write_bytes(b"corrupt parquet")
    before = snapshot(store)
    with pytest.raises(DataIntegrityError, match="Checksum mismatch"):
        store.merge_raw(request(3, 7), [fetched(5, 7, 5, 6)], supersedes=[old.dataset_id])
    assert snapshot(store) == before


@pytest.mark.parametrize("known_new", [True, False])
def test_missing_merge_combines_old_and_new_quality_provenance(isolated_data_dir, known_new):
    store = LocalDataStore(isolated_data_dir)
    old_quality = FetchQuality(status="reported", skip_missing_ohlc=True,
                               omitted_bars=(OmittedBar(day(4), "old omission"),), provider_version="test-v1")
    new_quality = FetchQuality(status="reported" if known_new else "unknown", skip_missing_ohlc=False,
                               omitted_bars=(OmittedBar(day(6), "new omission"),),
                               provider_version="test-v1" if known_new else "test-v2")
    old = store.write_raw(request(3, 5), frame(3), quality=old_quality)
    merged = store.merge_raw(request(3, 7), [fetched(5, 7, 5, quality=new_quality)], supersedes=[old.dataset_id])
    assert merged.quality.status == ("reported" if known_new else "unknown")
    assert merged.quality.skip_missing_ohlc is True
    assert merged.quality.provider_version == ("test-v1" if known_new else None)
    assert merged.quality.omitted_bars == (OmittedBar(day(4), "old omission"), OmittedBar(day(6), "new omission"))
    assert store.get_metadata(merged.dataset_id).quality == merged.quality


@pytest.mark.parametrize("failure", ["writer", "readback", "publish", "catalog", "sync"])
def test_missing_merge_rolls_back_all_chunks_and_derivatives_on_failure(isolated_data_dir, monkeypatch, failure):
    store = LocalDataStore(isolated_data_dir)
    old = store.write_raw(request(3, 5), frame(3, 4))
    store.write_processed(old.dataset_id, frame(3, 4), **pipeline())
    before = snapshot(store)
    chunks = [fetched(1, 3, 1, 2), fetched(5, 7, 5, 6)]
    write, publish, insert = pl.DataFrame.write_parquet, store_module._publish, store_module.catalog.insert

    def fail_write(self, handle, **kwargs):
        handle.write(b"partial")
        raise OSError("injected writer failure")

    def bad_write(self, handle, **kwargs):
        return write(self.with_columns(pl.col("volume") + 1), handle, **kwargs)

    def fail_publish(staged, final):
        publish(staged, final)
        raise OSError("injected publication failure")

    def fail_insert(connection, item):
        insert(connection, item)
        raise RuntimeError("injected catalog failure")

    def fail_sync(path):
        raise OSError("injected sync failure")

    with monkeypatch.context() as patch:
        if failure in {"writer", "readback"}:
            patch.setattr(pl.DataFrame, "write_parquet", fail_write if failure == "writer" else bad_write)
        elif failure == "publish":
            patch.setattr(store_module, "_publish", fail_publish)
        elif failure == "catalog":
            patch.setattr(store_module.catalog, "insert", fail_insert)
        else:
            patch.setattr(store_module, "_sync_directory", fail_sync)
        with pytest.raises(DataIntegrityError if failure == "readback" else StorageWriteError):
            store.merge_raw(request(1, 7), chunks, supersedes=[old.dataset_id])
    restarted = LocalDataStore(isolated_data_dir)
    assert snapshot(restarted) == before
    assert restarted.recover() == []
    merged = restarted.merge_raw(request(1, 7), chunks, supersedes=[old.dataset_id])
    assert_frame_equal(restarted.read_dataset(merged.dataset_id), frame(1, 2, 3, 4, 5, 6))
