"""Storage contracts, revision history, and real filesystem failure recovery.

All stores and damaged artifacts live in pytest's temporary directories. Private
hooks are used only to inject failures at the persistence boundaries.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Event
from urllib.parse import unquote

import duckdb
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.exceptions import (
    ConfirmationRequiredError,
    DataAlreadyExistsError,
    DataIntegrityError,
    DatasetNotFoundError,
    DuplicateBarError,
    OverlappingDataError,
    RequestDataMismatchError,
    SchemaValidationError,
    StorageBusyError,
    StorageError,
    StorageWriteError,
)
from data_pipeline.models import DataQuery, DataRequest
from data_pipeline.schemas import OHLCV_SCHEMA
from data_pipeline.storage import store as store_module
from data_pipeline.storage.store import LocalDataStore


@pytest.fixture
def raw_request() -> DataRequest:
    return DataRequest(
        symbol="AAPL",
        start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 17, 30, tzinfo=UTC),
        timeframe="1h",
        provider="yahoo",
    )


def _shift(frame: pl.DataFrame, delta: timedelta) -> pl.DataFrame:
    return frame.with_columns(pl.col("timestamp") + delta)


def _pipeline(version: str = "1") -> dict[str, str]:
    processors_json = json.dumps(
        [{"name": "keep-bars", "version": version, "config": {}}],
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "pipeline_id": sha256(processors_json.encode()).hexdigest(),
        "processors_json": processors_json,
    }


def _artifacts(root: Path) -> dict[str, str]:
    """Include staging and quarantine so leaked files cannot hide from assertions."""
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for folder in ("raw", "processed", "staging", "quarantine")
        for path in (root / folder).rglob("*")
        if path.is_file()
    }


def _state(store: LocalDataStore) -> tuple:
    history = [
        item
        for layer in ("raw", "processed")
        for item in store.list_datasets(DataQuery(layer=layer, include_history=True))
    ]
    return sorted(history, key=lambda item: item.dataset_id), _artifacts(store.data_dir)


def _ids(items) -> set[str]:
    return {item.dataset_id for item in items}


def _edit_metadata(store: LocalDataStore, dataset_id: str, **updates) -> None:
    """Deliberately damage one catalog record without touching its data file."""
    with duckdb.connect(str(store.catalog_path)) as connection:
        value = connection.execute(
            "SELECT metadata_json FROM datasets WHERE dataset_id=?", [dataset_id]
        ).fetchone()[0]
        metadata = json.loads(value)
        metadata.update(updates)
        connection.execute(
            "UPDATE datasets SET metadata_json=? WHERE dataset_id=?",
            [json.dumps(metadata), dataset_id],
        )


def test_roundtrip_restart_metadata_checksum_zstd_and_canonical_schema(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    before = datetime.now(UTC)
    item = store.write_raw(raw_request, sample_ohlcv_frame)
    after = datetime.now(UTC)
    path = isolated_data_dir / item.relative_path

    assert item.layer == "raw"
    assert item.request == raw_request
    assert item.row_count == sample_ohlcv_frame.height
    assert item.first_timestamp == raw_request.start
    assert item.last_timestamp == raw_request.end - timedelta(hours=1)
    assert item.schema_version == 1
    assert item.active
    assert item.parent_ids == item.supersedes == ()
    assert item.pipeline_id == ""
    assert item.processors_json == "[]"
    assert item.timestamp_convention == "bar_start"
    assert before <= item.created_at <= after
    assert item.checksum_sha256 == sha256(path.read_bytes()).hexdigest()
    assert not Path(item.relative_path).is_absolute()
    assert path.resolve().is_relative_to(isolated_data_dir.resolve())
    assert pl.read_parquet_schema(path) == OHLCV_SCHEMA
    with duckdb.connect() as connection:
        codecs = connection.execute(
            "SELECT DISTINCT compression FROM parquet_metadata(?)", [str(path)]
        ).fetchall()
    assert codecs == [("ZSTD",)]

    restarted = LocalDataStore(isolated_data_dir)
    assert restarted.get_metadata(item.dataset_id) == item
    assert restarted.list_datasets() == [item]
    for frame in (
        restarted.read_dataset(item.dataset_id),
        restarted.read(DataQuery(provider="yahoo")),
        restarted.scan(DataQuery(provider="yahoo")).collect(),
    ):
        # The partition directories must not add provider/year/etc. columns.
        assert frame.schema == OHLCV_SCHEMA
        assert_frame_equal(frame, sample_ohlcv_frame)
    assert restarted.audit() == [item.dataset_id]
    assert _artifacts(isolated_data_dir) == {item.relative_path: item.checksum_sha256}


@pytest.mark.parametrize("case", ["before-start", "at-end", "wrong-symbol", "extra-column"])
def test_write_rejects_request_or_schema_mismatch_without_artifacts(
    isolated_data_dir, raw_request, sample_ohlcv_frame, case
):
    store = LocalDataStore(isolated_data_dir)
    if case == "before-start":
        frame = _shift(sample_ohlcv_frame, -timedelta(milliseconds=1))
    elif case == "at-end":
        frame = _shift(sample_ohlcv_frame, timedelta(hours=1))
    elif case == "wrong-symbol":
        frame = sample_ohlcv_frame.with_columns(pl.lit("MSFT").alias("symbol"))
    else:
        frame = sample_ohlcv_frame.with_columns(pl.lit(2024).alias("year"))
    error = SchemaValidationError if case == "extra-column" else RequestDataMismatchError
    with pytest.raises(error):
        store.write_raw(raw_request, frame)
    assert store.list_datasets() == []
    assert _artifacts(isolated_data_dir) == {}


def test_duplicate_keys_within_one_batch_are_rejected(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    duplicate = pl.concat([sample_ohlcv_frame.head(1), sample_ohlcv_frame])
    with pytest.raises(DuplicateBarError):
        store.write_raw(raw_request, duplicate)
    assert store.list_datasets() == []
    assert _artifacts(isolated_data_dir) == {}


@pytest.mark.parametrize(
    "case", ["exact", "contained", "enclosing", "shared-first", "shared-last", "interleaved"]
)
def test_overlapping_writes_preserve_original_including_shared_boundary_bar(
    isolated_data_dir, raw_request, sample_ohlcv_frame, case
):
    store = LocalDataStore(isolated_data_dir)
    original = store.write_raw(raw_request, sample_ohlcv_frame)
    before = _state(store)
    request = replace(
        raw_request,
        start=raw_request.start - timedelta(hours=3),
        end=raw_request.end + timedelta(hours=3),
    )
    candidates = {
        "exact": sample_ohlcv_frame,
        "contained": sample_ohlcv_frame.slice(1, 1),
        "enclosing": pl.concat([
            _shift(sample_ohlcv_frame, -timedelta(hours=3)),
            sample_ohlcv_frame,
            _shift(sample_ohlcv_frame, timedelta(hours=3)),
        ]),
        "shared-first": _shift(sample_ohlcv_frame, -timedelta(hours=2)),
        "shared-last": _shift(sample_ohlcv_frame, timedelta(hours=2)),
        # Coverage overlaps even though none of the bar keys are equal.
        "interleaved": _shift(sample_ohlcv_frame, timedelta(minutes=30)),
    }
    with pytest.raises(OverlappingDataError):
        store.write_raw(request, candidates[case])
    assert _state(LocalDataStore(isolated_data_dir)) == before
    assert_frame_equal(store.read_dataset(original.dataset_id), sample_ohlcv_frame)


def test_disjoint_actual_coverage_can_share_request_and_fill_a_gap(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    late = store.write_raw(raw_request, sample_ohlcv_frame.tail(1))
    early = store.write_raw(raw_request, sample_ohlcv_frame.head(1))
    assert_frame_equal(
        store.read(DataQuery()),
        pl.concat([sample_ohlcv_frame.head(1), sample_ohlcv_frame.tail(1)]),
    )
    middle = store.write_raw(raw_request, sample_ohlcv_frame.slice(1, 1))
    assert len({late.relative_path, early.relative_path, middle.relative_path}) == 3
    assert all(item.request == raw_request for item in store.list_datasets())
    assert_frame_equal(LocalDataStore(isolated_data_dir).read(DataQuery()), sample_ohlcv_frame)


def test_provider_namespaces_and_escaped_identifiers_are_distinct_and_contained(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    requests = [
        raw_request,
        replace(raw_request, provider="other-vendor"),
        replace(raw_request, provider="../VENDOR/x\\=' %", symbol="../../A/B=é\\%"),
        replace(raw_request, symbol="A/B"),
        replace(raw_request, symbol="A%2FB"),
    ]
    items = []
    for request in requests:
        frame = sample_ohlcv_frame.with_columns(pl.lit(request.symbol).alias("symbol"))
        item = store.write_raw(request, frame)
        items.append(item)
        path = Path(item.relative_path)
        assert len(path.parts) == 7
        assert path.parts[0] == "raw"
        partitions = dict(part.split("=", 1) for part in path.parts[1:-1])
        for field in ("dataset", "provider", "symbol", "timeframe"):
            assert unquote(partitions[field]) == getattr(request, field)
        assert (isolated_data_dir / path).resolve().is_relative_to(isolated_data_dir.resolve())
        query = DataQuery(provider=request.provider, symbol=request.symbol, timeframe="1h")
        assert store.list_datasets(query) == [item]
        assert_frame_equal(store.read(query), frame)
    assert len({Path(item.relative_path).parent for item in items}) == len(items)
    assert len(_artifacts(isolated_data_dir)) == len(items)


def test_replacement_requires_exact_id_confirmation_and_entire_original_bounds(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    item = store.write_raw(raw_request, sample_ohlcv_frame)
    before = _state(store)
    for confirmation in ("", "yes", item.dataset_id[:-1], item.dataset_id + "extra"):
        with pytest.raises(ConfirmationRequiredError):
            store.replace_raw(item.dataset_id, sample_ohlcv_frame, confirm=confirmation)
        assert _state(store) == before
    for partial in (
        sample_ohlcv_frame.head(2),
        sample_ohlcv_frame.tail(2),
        _shift(sample_ohlcv_frame, timedelta(minutes=15)),
    ):
        with pytest.raises(RequestDataMismatchError):
            store.replace_raw(item.dataset_id, partial, confirm=item.dataset_id)
        assert _state(store) == before


def test_processed_lineage_duplicate_raw_pipeline_pair_and_distinct_versions(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    parent = store.write_raw(raw_request, sample_ohlcv_frame)
    frame = sample_ohlcv_frame.tail(2)
    first = store.write_processed(parent.dataset_id, frame, **_pipeline("1"))
    before = _state(store)
    # A disjoint output still duplicates the same raw revision/pipeline pair.
    with pytest.raises(DataAlreadyExistsError):
        store.write_processed(parent.dataset_id, sample_ohlcv_frame.head(1), **_pipeline("1"))
    assert _state(store) == before
    second = store.write_processed(parent.dataset_id, frame, **_pipeline("2"))
    delta = timedelta(days=1)
    later = store.write_raw(
        replace(raw_request, start=raw_request.start + delta, end=raw_request.end + delta),
        _shift(sample_ohlcv_frame, delta),
    )
    third = store.write_processed(later.dataset_id, _shift(frame, delta), **_pipeline("1"))

    restarted = LocalDataStore(isolated_data_dir)
    for item, version in ((first, "1"), (second, "2")):
        metadata = restarted.get_metadata(item.dataset_id)
        assert metadata == item
        assert metadata.layer == "processed"
        assert metadata.parent_ids == (parent.dataset_id,)
        assert metadata.request == replace(raw_request, start=frame["timestamp"].min(),
                                           end=frame["timestamp"].max() + timedelta(milliseconds=1))
        assert metadata.pipeline_id == _pipeline(version)["pipeline_id"]
        assert metadata.processors_json == _pipeline(version)["processors_json"]
        assert_frame_equal(restarted.read_dataset(item.dataset_id), frame)
    assert len({first.dataset_id, second.dataset_id, third.dataset_id}) == 3
    assert _ids(restarted.list_datasets(DataQuery(
        layer="processed", pipeline_id=first.pipeline_id
    ))) == {first.dataset_id, third.dataset_id}
    assert_frame_equal(restarted.read(DataQuery(
        layer="processed", pipeline_id=second.pipeline_id
    )), frame)
    with pytest.raises(RequestDataMismatchError):
        restarted.scan(DataQuery(layer="processed"))


@pytest.mark.parametrize("case", ["empty-pipeline", "empty-processors", "wrong-symbol", "processed-parent"])
def test_processed_writes_require_pipeline_and_compatible_raw_parents(
    isolated_data_dir, raw_request, sample_ohlcv_frame, case
):
    store = LocalDataStore(isolated_data_dir)
    parent = store.write_raw(raw_request, sample_ohlcv_frame)
    frame = sample_ohlcv_frame
    pipeline = _pipeline()
    if case == "empty-pipeline":
        pipeline["pipeline_id"] = ""
    elif case == "empty-processors":
        pipeline["processors_json"] = "[]"
    elif case == "wrong-symbol":
        frame = frame.with_columns(pl.lit("MSFT").alias("symbol"))
    else:
        parent = store.write_processed(parent.dataset_id, frame, **pipeline)
    before = _state(store)
    with pytest.raises(RequestDataMismatchError):
        store.write_processed(parent.dataset_id, frame, **pipeline)
    assert _state(store) == before


def test_replacement_keeps_history_and_invalidates_only_its_processed_descendants(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    original = store.write_raw(raw_request, sample_ohlcv_frame)
    outputs = [
        store.write_processed(original.dataset_id, sample_ohlcv_frame, **_pipeline(version))
        for version in ("1", "2")
    ]
    other = store.write_raw(replace(raw_request, provider="other"), sample_ohlcv_frame)
    unaffected = store.write_processed(other.dataset_id, sample_ohlcv_frame, **_pipeline())
    old_files = _artifacts(isolated_data_dir)
    historical_scan = store.scan(DataQuery(provider="yahoo"))
    revised = sample_ohlcv_frame.with_columns(pl.col("volume") + 1)
    replacement = store.replace_raw(original.dataset_id, revised, confirm=original.dataset_id)
    store = LocalDataStore(isolated_data_dir)

    assert replacement.dataset_id != original.dataset_id
    assert replacement.relative_path != original.relative_path
    assert replacement.request == original.request
    assert replacement.first_timestamp == original.first_timestamp
    assert replacement.last_timestamp == original.last_timestamp
    assert replacement.supersedes == (original.dataset_id,)
    assert replacement.active
    assert _ids(store.list_datasets(DataQuery())) == {replacement.dataset_id, other.dataset_id}
    assert _ids(store.list_datasets(DataQuery(layer="processed"))) == {unaffected.dataset_id}
    assert_frame_equal(store.read(DataQuery(provider="yahoo")), revised)
    assert_frame_equal(historical_scan.collect(), sample_ohlcv_frame)
    for historical in (original, *outputs):
        assert not store.get_metadata(historical.dataset_id).active
        assert_frame_equal(store.read_dataset(historical.dataset_id), sample_ohlcv_frame)
    assert all(_artifacts(isolated_data_dir)[path] == checksum for path, checksum in old_files.items())
    assert _ids(store.list_datasets(DataQuery(include_history=True))) == {
        original.dataset_id, replacement.dataset_id, other.dataset_id,
    }
    assert _ids(store.list_datasets(DataQuery(layer="processed", include_history=True))) == {
        *(item.dataset_id for item in outputs), unaffected.dataset_id,
    }
    new_output = store.write_processed(replacement.dataset_id, revised, **_pipeline())
    assert new_output.parent_ids == (replacement.dataset_id,)
    assert new_output.dataset_id != outputs[0].dataset_id
    for inactive_or_processed in (original, new_output):
        with pytest.raises(RequestDataMismatchError):
            store.replace_raw(inactive_or_processed.dataset_id, revised, confirm=inactive_or_processed.dataset_id)
    with pytest.raises(RequestDataMismatchError):
        store.write_processed(original.dataset_id, sample_ohlcv_frame, **_pipeline("3"))
    assert set(store.audit()) == {
        original.dataset_id, replacement.dataset_id, other.dataset_id,
        *(item.dataset_id for item in outputs), unaffected.dataset_id, new_output.dataset_id,
    }


def test_compaction_requires_multiple_exact_ids_and_preserves_revision_history(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    split = raw_request.start + timedelta(hours=1)
    first = store.write_raw(replace(raw_request, end=split), sample_ohlcv_frame.head(1))
    second = store.write_raw(replace(raw_request, start=split), sample_ohlcv_frame.tail(2))
    outputs = [
        store.write_processed(item.dataset_id, frame, **_pipeline())
        for item, frame in ((first, sample_ohlcv_frame.head(1)), (second, sample_ohlcv_frame.tail(2)))
    ]
    foreign = store.write_raw(replace(raw_request, provider="other"), sample_ohlcv_frame)
    before = _state(store)
    ids = [first.dataset_id, second.dataset_id]
    for invalid_ids in ([], ids[:1], [ids[0], ids[0]], [ids[0], foreign.dataset_id], [ids[0], outputs[0].dataset_id]):
        with pytest.raises(RequestDataMismatchError):
            store.compact_raw(invalid_ids, confirm=invalid_ids)
        assert _state(store) == before
    for confirmation in ([], ids[:1], [ids[0], ids[0]], [*ids, foreign.dataset_id]):
        with pytest.raises(ConfirmationRequiredError):
            store.compact_raw(ids, confirm=confirmation)
        assert _state(store) == before

    compacted = store.compact_raw(ids[::-1], confirm=ids)
    store = LocalDataStore(isolated_data_dir)
    assert compacted.request == raw_request
    assert set(compacted.supersedes) == set(ids)
    assert compacted.row_count == sample_ohlcv_frame.height
    assert compacted.first_timestamp == first.first_timestamp
    assert compacted.last_timestamp == second.last_timestamp
    assert_frame_equal(store.read_dataset(compacted.dataset_id), sample_ohlcv_frame)
    assert_frame_equal(store.read(DataQuery(provider="yahoo")), sample_ohlcv_frame)
    assert _ids(store.list_datasets(DataQuery())) == {compacted.dataset_id, foreign.dataset_id}
    assert store.list_datasets(DataQuery(layer="processed")) == []
    for item, expected in ((first, sample_ohlcv_frame.head(1)), (second, sample_ohlcv_frame.tail(2))):
        assert not store.get_metadata(item.dataset_id).active
        assert_frame_equal(store.read_dataset(item.dataset_id), expected)
    assert all(not store.get_metadata(item.dataset_id).active for item in outputs)
    assert all(_artifacts(isolated_data_dir)[path] == checksum for path, checksum in before[1].items())
    with pytest.raises(RequestDataMismatchError):
        store.compact_raw(ids, confirm=ids)
    assert set(store.audit()) == {first.dataset_id, second.dataset_id, foreign.dataset_id,
                                  compacted.dataset_id, *(item.dataset_id for item in outputs)}
    assert store.recover() == []


@pytest.mark.parametrize(
    "failure_point", ["staged-writer", "publish-before", "publish-after", "catalog-insert", "directory-sync"]
)
def test_failed_replacement_rolls_back_catalog_and_removes_all_new_artifacts(
    isolated_data_dir, raw_request, sample_ohlcv_frame, monkeypatch, failure_point
):
    store = LocalDataStore(isolated_data_dir)
    original = store.write_raw(raw_request, sample_ohlcv_frame)
    derived = store.write_processed(original.dataset_id, sample_ohlcv_frame, **_pipeline())
    before = _state(store)
    publish = store_module._publish
    insert = store_module.catalog.insert

    def fail_writer(self, file, **kwargs):
        file.write(b"partial parquet")
        raise OSError("injected staged writer failure")

    def fail_publish(staged, final):
        if failure_point == "publish-after":
            publish(staged, final)
        raise OSError("injected file publication failure")

    def fail_insert(connection, item):
        insert(connection, item)
        raise RuntimeError("injected catalog insert failure")

    def fail_sync(path):
        raise OSError("injected directory sync failure")

    with monkeypatch.context() as patch:
        if failure_point == "staged-writer":
            patch.setattr(pl.DataFrame, "write_parquet", fail_writer)
        elif failure_point.startswith("publish-"):
            patch.setattr(store_module, "_publish", fail_publish)
        elif failure_point == "catalog-insert":
            patch.setattr(store_module.catalog, "insert", fail_insert)
        else:
            patch.setattr(store_module, "_sync_directory", fail_sync)
        with pytest.raises(StorageWriteError, match="injected"):
            store.replace_raw(
                original.dataset_id,
                sample_ohlcv_frame.with_columns(pl.col("volume") + 1),
                confirm=original.dataset_id,
            )

    restarted = LocalDataStore(isolated_data_dir)
    # No recovery call: an ordinary exception must clean up synchronously.
    assert _state(restarted) == before
    assert restarted.get_metadata(original.dataset_id).active
    assert restarted.get_metadata(derived.dataset_id).active
    assert_frame_equal(restarted.read(DataQuery()), sample_ohlcv_frame)
    assert set(restarted.audit()) == {original.dataset_id, derived.dataset_id}
    retry = restarted.replace_raw(original.dataset_id, sample_ohlcv_frame, confirm=original.dataset_id)
    assert retry.supersedes == (original.dataset_id,)


def test_staged_parquet_must_match_input_before_publication(
    isolated_data_dir, raw_request, sample_ohlcv_frame, monkeypatch
):
    store = LocalDataStore(isolated_data_dir)
    original = store.write_raw(raw_request, sample_ohlcv_frame)
    before = _state(store)
    write_parquet = pl.DataFrame.write_parquet

    def silently_change_values(self, file, **kwargs):
        return write_parquet(self.with_columns(pl.col("volume") + 7), file, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pl.DataFrame, "write_parquet", silently_change_values)
        with pytest.raises(DataIntegrityError):
            store.replace_raw(original.dataset_id, sample_ohlcv_frame, confirm=original.dataset_id)
    assert _state(store) == before


@pytest.mark.parametrize("overlap", [True, False], ids=["competing-coverage", "disjoint-coverage"])
def test_concurrent_store_instances_serialize_writes_without_lost_updates(
    isolated_data_dir, raw_request, sample_ohlcv_frame, overlap
):
    barrier = Barrier(2)
    delta = timedelta(0) if overlap else timedelta(days=1)
    later_request = replace(raw_request, start=raw_request.start + delta, end=raw_request.end + delta)

    def write(request, frame):
        store = LocalDataStore(isolated_data_dir)
        barrier.wait(timeout=10)
        try:
            return store.write_raw(request, frame)
        except OverlappingDataError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(write, raw_request, sample_ohlcv_frame),
            pool.submit(write, later_request, _shift(sample_ohlcv_frame, delta)),
        ]
        results = [future.result(timeout=20) for future in futures]
    errors = [result for result in results if isinstance(result, OverlappingDataError)]
    successful = [result for result in results if not isinstance(result, OverlappingDataError)]
    assert len(errors) == int(overlap)
    assert len(successful) == (1 if overlap else 2)
    store = LocalDataStore(isolated_data_dir)
    assert _ids(store.list_datasets()) == _ids(successful)
    assert len(_artifacts(isolated_data_dir)) == len(successful)
    expected = sample_ohlcv_frame if overlap else pl.concat([
        sample_ohlcv_frame, _shift(sample_ohlcv_frame, delta)
    ])
    assert_frame_equal(store.read(DataQuery()), expected)
    assert set(store.audit()) == _ids(successful)


def test_lock_timeout_while_another_store_publishes_then_lock_is_released(
    isolated_data_dir, raw_request, sample_ohlcv_frame, monkeypatch
):
    entered, release = Event(), Event()
    publish = store_module._publish
    pending_ids = []

    def hold_publication(staged, final):
        pending_ids.append(final.stem)
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the writer")
        publish(staged, final)

    monkeypatch.setattr(store_module, "_publish", hold_publication)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(LocalDataStore(isolated_data_dir).write_raw, raw_request, sample_ohlcv_frame)
        try:
            assert entered.wait(timeout=10), "writer never reached publication"
            contender = LocalDataStore(isolated_data_dir, lock_timeout=0.05)
            for operation in (
                lambda: contender.write_raw(raw_request, sample_ohlcv_frame),
                lambda: contender.read_dataset(pending_ids[0]),
                contender.recover,
            ):
                with pytest.raises(StorageBusyError):
                    operation()
        finally:
            release.set()
        item = future.result(timeout=15)
    store = LocalDataStore(isolated_data_dir, lock_timeout=0.05)
    assert store.get_metadata(item.dataset_id) == item
    assert_frame_equal(store.read_dataset(item.dataset_id), sample_ohlcv_frame)
    assert len(_artifacts(isolated_data_dir)) == 1


def test_crash_after_publish_recovers_orphans_to_quarantine_without_exposing_them(
    isolated_data_dir, raw_request, sample_ohlcv_frame, tmp_path
):
    store = LocalDataStore(isolated_data_dir)
    historical = store.write_raw(raw_request, sample_ohlcv_frame)
    parent = store.replace_raw(historical.dataset_id, sample_ohlcv_frame, confirm=historical.dataset_id)
    store.write_processed(parent.dataset_id, sample_ohlcv_frame, **_pipeline())
    before = _state(store)
    script = """
import os
import sys
import polars as pl
from data_pipeline.storage import store as store_module

store = store_module.LocalDataStore(sys.argv[1])
parent_id = sys.argv[2]
frame = store.read_dataset(parent_id).with_columns(pl.col("volume") + 1)
publish = store_module._publish
def crash(staged, final):
    publish(staged, final)
    os._exit(73)
store_module._publish = crash
store.replace_raw(parent_id, frame, confirm=parent_id)
"""
    env = os.environ.copy()
    source_root = str(Path(store_module.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    env.update(POLARS_MAX_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(isolated_data_dir), parent.dataset_id],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 73, result.stdout + result.stderr

    restarted = LocalDataStore(isolated_data_dir)
    assert _state(restarted)[0] == before[0]
    assert_frame_equal(restarted.read(DataQuery()), sample_ohlcv_frame)
    orphans = {path: checksum for path, checksum in _artifacts(isolated_data_dir).items() if path not in before[1]}
    assert any(Path(path).parts[0] == "raw" for path in orphans)
    assert all(Path(path).parts[0] in {"raw", "staging"} for path in orphans)
    for path in orphans:
        assert_frame_equal(
            pl.read_parquet(isolated_data_dir / path, hive_partitioning=False),
            sample_ohlcv_frame.with_columns(pl.col("volume") + 1),
        )

    moved = restarted.recover()
    assert len(moved) == len(orphans)
    assert all(Path(path).parts[0] == "quarantine" for path in moved)
    assert sorted(sha256((isolated_data_dir / path).read_bytes()).hexdigest() for path in moved) == sorted(orphans.values())
    assert all(not (isolated_data_dir / path).exists() for path in orphans)
    remaining = _artifacts(isolated_data_dir)
    assert {path: value for path, value in remaining.items() if Path(path).parts[0] != "quarantine"} == before[1]
    assert set(remaining) - set(before[1]) == set(moved)
    assert _state(restarted)[0] == before[0]
    assert set(restarted.audit()) == _ids(before[0])
    assert restarted.recover() == []


@pytest.mark.parametrize("damage", ["missing", "checksum", "schema", "schema-version", "row-count", "bounds"])
def test_missing_corrupt_or_incompatible_data_is_not_silently_read(
    isolated_data_dir, raw_request, sample_ohlcv_frame, damage
):
    store = LocalDataStore(isolated_data_dir)
    item = store.write_raw(raw_request, sample_ohlcv_frame)
    path = isolated_data_dir / item.relative_path
    if damage == "missing":
        path.unlink()
    elif damage == "checksum":
        path.write_bytes(b"corrupted parquet")
    elif damage == "schema":
        sample_ohlcv_frame.drop("volume").write_parquet(path)
        _edit_metadata(store, item.dataset_id, checksum_sha256=sha256(path.read_bytes()).hexdigest())
    elif damage == "schema-version":
        _edit_metadata(store, item.dataset_id, schema_version=999)
    elif damage == "row-count":
        _edit_metadata(store, item.dataset_id, row_count=item.row_count + 1)
    else:
        _edit_metadata(store, item.dataset_id, last_timestamp=raw_request.end.isoformat())
    restarted = LocalDataStore(isolated_data_dir)
    with pytest.raises(DataIntegrityError):
        restarted.read_dataset(item.dataset_id)
    with pytest.raises(DataIntegrityError):
        restarted.audit()
    if damage not in {"row-count", "bounds"}:
        with pytest.raises(DataIntegrityError):
            restarted.scan(DataQuery()).collect()


def test_catalog_path_cannot_redirect_reads_outside_store(
    isolated_data_dir, raw_request, sample_ohlcv_frame, tmp_path
):
    store = LocalDataStore(isolated_data_dir)
    item = store.write_raw(raw_request, sample_ohlcv_frame)
    outside = tmp_path / "outside.parquet"
    sample_ohlcv_frame.write_parquet(outside)
    checksum = sha256(outside.read_bytes()).hexdigest()
    for relative_path in ("../outside.parquet", str(outside)):
        _edit_metadata(store, item.dataset_id, relative_path=relative_path, checksum_sha256=checksum)
        with pytest.raises(DataIntegrityError):
            store.read_dataset(item.dataset_id)
        with pytest.raises(DataIntegrityError):
            store.scan(DataQuery())
    assert sha256(outside.read_bytes()).hexdigest() == checksum


def test_missing_catalog_is_empty_and_missing_id_is_explicit(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    assert store.list_datasets() == []
    assert store.read(DataQuery()).schema == OHLCV_SCHEMA
    assert store.read(DataQuery()).is_empty()
    with pytest.raises(DatasetNotFoundError):
        store.read_dataset("missing")
    assert not store.catalog_path.exists()
    # Files without a catalog may belong to a damaged store, not a new one.
    orphan = isolated_data_dir / "raw" / "orphan.parquet"
    orphan.parent.mkdir()
    sample_ohlcv_frame.write_parquet(orphan)
    for operation in (store.list_datasets, lambda: store.read(DataQuery()), store.recover):
        with pytest.raises(DataIntegrityError, match="catalog is missing"):
            operation()
    assert not store.catalog_path.exists()
    assert_frame_equal(pl.read_parquet(orphan), sample_ohlcv_frame)


def test_lost_catalog_blocks_all_operations_without_moving_or_overwriting_data(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    item = store.write_raw(raw_request, sample_ohlcv_frame)
    backup = store.catalog_path.with_suffix(".backup")
    store.catalog_path.rename(backup)
    before = _artifacts(isolated_data_dir)
    for operation in (
        store.list_datasets, lambda: store.read(DataQuery()),
        lambda: store.read_dataset(item.dataset_id), store.audit, store.recover,
        lambda: store.write_raw(raw_request, sample_ohlcv_frame),
    ):
        with pytest.raises(DataIntegrityError, match="catalog is missing"):
            operation()
        assert not store.catalog_path.exists()
        assert _artifacts(isolated_data_dir) == before
    # Restoring the catalog restores access and duplicate protection.
    backup.rename(store.catalog_path)
    assert_frame_equal(store.read_dataset(item.dataset_id), sample_ohlcv_frame)
    with pytest.raises(OverlappingDataError):
        store.write_raw(raw_request, sample_ohlcv_frame)


@pytest.mark.parametrize("damage", ["datasets", "catalog_version", "empty-version"])
def test_incomplete_catalog_is_not_silently_reinitialized(
    isolated_data_dir, raw_request, sample_ohlcv_frame, damage
):
    store = LocalDataStore(isolated_data_dir)
    store.write_raw(raw_request, sample_ohlcv_frame)
    before = _artifacts(isolated_data_dir)
    with duckdb.connect(str(store.catalog_path)) as connection:
        if damage == "empty-version":
            connection.execute("DELETE FROM catalog_version")
        else:
            connection.execute(f"DROP TABLE {damage}")
        tables = connection.execute("SHOW TABLES").fetchall()
    for operation in (
        store.list_datasets, store.recover,
        lambda: store.write_raw(raw_request, sample_ohlcv_frame),
    ):
        with pytest.raises(DataIntegrityError, match="catalog"):
            operation()
        assert _artifacts(isolated_data_dir) == before
        with duckdb.connect(str(store.catalog_path)) as connection:
            assert connection.execute("SHOW TABLES").fetchall() == tables
            if damage == "empty-version":
                assert connection.execute("SELECT * FROM catalog_version").fetchall() == []


@pytest.mark.parametrize("timeframe", ["1d", "5d", "1wk", "1mo", "3mo"])
def test_session_date_batches_reject_intraday_timestamps(
    isolated_data_dir, raw_request, sample_ohlcv_frame, timeframe
):
    store = LocalDataStore(isolated_data_dir)
    with pytest.raises(RequestDataMismatchError, match="midnight UTC"):
        store.write_raw(replace(raw_request, timeframe=timeframe), sample_ohlcv_frame)
    assert store.list_datasets() == []
    assert not _artifacts(isolated_data_dir)


@pytest.mark.parametrize("timeframe", ["1d", "5d", "1wk", "1mo", "3mo"])
def test_session_date_batches_roundtrip_with_midnight_utc_labels(
    isolated_data_dir, raw_request, sample_ohlcv_frame, timeframe
):
    request = replace(raw_request, timeframe=timeframe, start=datetime(2024, 1, 2, tzinfo=UTC))
    frame = sample_ohlcv_frame.head(1).with_columns(pl.col("timestamp").dt.truncate("1d"))
    store = LocalDataStore(isolated_data_dir)
    item = store.write_raw(request, frame)
    assert item.timestamp_convention == "session_date"
    assert_frame_equal(store.read_dataset(item.dataset_id), frame)


def test_corrupt_catalog_is_reported_and_never_overwritten(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    store.catalog_path.parent.mkdir()
    content = b"This is not a DuckDB catalog."
    store.catalog_path.write_bytes(content)
    for operation in (store.list_datasets, store.recover, lambda: store.write_raw(raw_request, sample_ohlcv_frame)):
        with pytest.raises(StorageError):
            operation()
        assert store.catalog_path.read_bytes() == content
        assert _artifacts(isolated_data_dir) == {}


@pytest.mark.parametrize("versions", [[999], [1, 1]], ids=["unsupported", "ambiguous"])
def test_catalog_version_is_checked_before_reads_writes_or_recovery(
    isolated_data_dir, raw_request, sample_ohlcv_frame, versions
):
    store = LocalDataStore(isolated_data_dir)
    item = store.write_raw(raw_request, sample_ohlcv_frame)
    before_files = _artifacts(isolated_data_dir)
    with duckdb.connect(str(store.catalog_path)) as connection:
        connection.execute("DELETE FROM catalog_version")
        connection.executemany("INSERT INTO catalog_version VALUES (?)", [(version,) for version in versions])
        before_rows = connection.execute("SELECT * FROM datasets").fetchall()
    for operation in (
        store.list_datasets, store.recover,
        lambda: store.read_dataset(item.dataset_id),
        lambda: store.write_raw(raw_request, sample_ohlcv_frame),
    ):
        with pytest.raises(DataIntegrityError, match="catalog version"):
            operation()
    assert _artifacts(isolated_data_dir) == before_files
    with duckdb.connect(str(store.catalog_path)) as connection:
        assert connection.execute("SELECT version FROM catalog_version").fetchall() == [(version,) for version in versions]
        assert connection.execute("SELECT * FROM datasets").fetchall() == before_rows


def test_scan_filters_projects_and_sorts_across_files_and_rejects_ambiguous_queries(
    isolated_data_dir, raw_request, sample_ohlcv_frame
):
    store = LocalDataStore(isolated_data_dir)
    msft = sample_ohlcv_frame.with_columns(pl.lit("MSFT").alias("symbol"), pl.col("volume") + 10)
    store.write_raw(replace(raw_request, symbol="MSFT"), msft)
    late = store.write_raw(raw_request, sample_ohlcv_frame.tail(2))
    store.write_raw(raw_request, sample_ohlcv_frame.head(1))
    store.write_raw(replace(raw_request, provider="other"), sample_ohlcv_frame)
    # Sparse minute data is valid but belongs to a different timeframe.
    store.write_raw(replace(raw_request, timeframe="1m"), sample_ohlcv_frame)

    query = DataQuery(provider="yahoo", timeframe="1h")
    expected = pl.concat([sample_ohlcv_frame, msft])
    lazy = store.scan(query)
    assert isinstance(lazy, pl.LazyFrame)
    assert_frame_equal(lazy.collect(), expected)
    bounded = replace(query, start=raw_request.start + timedelta(hours=1), end=raw_request.end - timedelta(hours=1))
    columns = ["close", "timestamp", "symbol"]
    assert_frame_equal(store.scan(bounded, columns=columns).collect(), pl.concat([
        sample_ohlcv_frame.slice(1, 1), msft.slice(1, 1)
    ]).select(columns))
    assert_frame_equal(store.read(query, columns=["volume", "close"]), expected.select("volume", "close"))
    assert store.list_datasets(replace(bounded, symbol="AAPL")) == [late]
    assert_frame_equal(store.read(replace(query, symbol="AAPL")), sample_ohlcv_frame)
    empty = store.scan(replace(query, start=raw_request.end), columns=["timestamp", "close"]).collect()
    assert empty.is_empty()
    assert empty.schema == {"timestamp": OHLCV_SCHEMA["timestamp"], "close": pl.Float64}
    for ambiguous in (
        DataQuery(symbol="AAPL", timeframe="1h"),  # mixed providers
        DataQuery(symbol="AAPL", provider="yahoo"),  # mixed timeframes
        replace(query, include_history=True),
    ):
        with pytest.raises(RequestDataMismatchError):
            store.scan(ambiguous)
    for columns in ([], ["close", "close"], ["provider"]):
        with pytest.raises(RequestDataMismatchError):
            store.scan(query, columns=columns)
