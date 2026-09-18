"""Deletion migration, crash consistency, cleanup failures, and precision guards."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import duckdb
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.exceptions import DataIntegrityError, DatasetNotFoundError, StorageWriteError
from data_pipeline.models import DataQuery, DataRequest
from data_pipeline.storage import catalog
from data_pipeline.storage import deletion
from data_pipeline.storage.store import LocalDataStore


@pytest.fixture
def seeded(isolated_data_dir, sample_ohlcv_frame):
    store = LocalDataStore(isolated_data_dir)
    request = DataRequest("AAPL", datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                          datetime(2024, 1, 2, 17, 30, tzinfo=UTC))
    original = store.write_raw(request, sample_ohlcv_frame)
    query = DataQuery(provider="yahoo", symbol="AAPL", timeframe="1h",
                      start=request.start + timedelta(hours=1), end=request.start + timedelta(hours=2))
    return store, original, query, sample_ohlcv_frame


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for name in ("raw", "processed", "staging", "quarantine")
            for path in (root / name).rglob("*.parquet")}


def _assert_original(seed, previous_files):
    store, original, _, frame = seed
    assert store.list_datasets() == [original]
    assert _files(store.data_dir) == previous_files
    assert_frame_equal(store.read_dataset(original.dataset_id), frame)
    assert deletion.list_deletions(store) == []
    with store._session() as connection:
        assert connection.execute("SELECT count(*) FROM deleted_datasets").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM deletion_preparations").fetchone()[0] == 0


def _legacy_catalog(store):
    with duckdb.connect(str(store.catalog_path)) as connection:
        for table in ("deletion_operations", "deleted_datasets", "deletion_preparations"):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("UPDATE catalog_version SET version=1")


def test_valid_v1_migrates_atomically_preserving_data_and_identity(seeded):
    store, original, query, frame = seeded
    _legacy_catalog(store)
    previous = _files(store.data_dir)
    assert store.list_datasets() == [original]
    assert_frame_equal(store.read_dataset(original.dataset_id), frame)
    assert _files(store.data_dir) == previous
    with duckdb.connect(str(store.catalog_path)) as connection:
        assert connection.execute("SELECT version FROM catalog_version").fetchall() == [(2,)]
        assert {row[0] for row in connection.execute("SHOW TABLES").fetchall()} == {
            "datasets", "catalog_version", "deletion_operations", "deleted_datasets", "deletion_preparations",
        }
    plan = deletion.plan_delete(store, query)
    assert deletion.execute_delete(store, plan, confirm=plan.operation_id).status == "completed"


def test_malformed_v1_is_not_partially_migrated(seeded):
    store, _, _, _ = seeded
    _legacy_catalog(store)
    with duckdb.connect(str(store.catalog_path)) as connection:
        connection.execute("ALTER TABLE datasets ADD COLUMN unexpected INTEGER")
    before = _files(store.data_dir)
    with pytest.raises(DataIntegrityError, match="Malformed catalog schema"):
        store.list_datasets()
    with duckdb.connect(str(store.catalog_path)) as connection:
        assert connection.execute("SELECT version FROM catalog_version").fetchall() == [(1,)]
        assert {row[0] for row in connection.execute("SHOW TABLES").fetchall()} == {"datasets", "catalog_version"}
    assert _files(store.data_dir) == before


def test_v1_without_unique_dataset_paths_is_rejected_before_migration(seeded):
    store, _, _, _ = seeded
    _legacy_catalog(store)
    before = _files(store.data_dir)
    with duckdb.connect(str(store.catalog_path)) as connection:
        ddl = connection.execute("SELECT sql FROM duckdb_tables() WHERE table_name='datasets'").fetchone()[0]
        connection.execute("ALTER TABLE datasets RENAME TO old_datasets")
        connection.execute(ddl.replace(" UNIQUE", ""))
        connection.execute("INSERT INTO datasets SELECT * FROM old_datasets")
        connection.execute("DROP TABLE old_datasets")
    with pytest.raises(DataIntegrityError, match="paths must remain unique"):
        store.list_datasets()
    with duckdb.connect(str(store.catalog_path)) as connection:
        assert connection.execute("SELECT version FROM catalog_version").fetchall() == [(1,)]
        assert {row[0] for row in connection.execute("SHOW TABLES").fetchall()} == {"datasets", "catalog_version"}
    assert _files(store.data_dir) == before


def test_interrupted_migration_rolls_back_schema_and_can_be_retried(seeded, monkeypatch):
    store, original, _, frame = seeded
    _legacy_catalog(store)
    before = _files(store.data_dir)
    create_tables = catalog._create_deletion_tables

    def interrupted(connection):
        create_tables(connection)
        raise RuntimeError("Simulated migration interruption")

    monkeypatch.setattr(catalog, "_create_deletion_tables", interrupted)
    with pytest.raises(RuntimeError, match="migration interruption"):
        store.list_datasets()
    with duckdb.connect(str(store.catalog_path)) as connection:
        assert connection.execute("SELECT version FROM catalog_version").fetchall() == [(1,)]
        assert {row[0] for row in connection.execute("SHOW TABLES").fetchall()} == {"datasets", "catalog_version"}
    monkeypatch.setattr(catalog, "_create_deletion_tables", create_tables)
    assert store.list_datasets() == [original]
    assert_frame_equal(store.read_dataset(original.dataset_id), frame)
    assert _files(store.data_dir) == before


@pytest.mark.parametrize("table", ["deletion_operations", "deleted_datasets", "deletion_preparations"])
def test_missing_v2_tables_are_not_recreated(seeded, table):
    store, _, _, _ = seeded
    with duckdb.connect(str(store.catalog_path)) as connection:
        connection.execute(f"DROP TABLE {table}")
    with pytest.raises(DataIntegrityError, match="Incomplete catalog"):
        store.list_datasets()
    with duckdb.connect(str(store.catalog_path)) as connection:
        assert table not in {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def test_disk_full_during_second_survivor_rolls_back_every_file(seeded, monkeypatch):
    store, _, query, _ = seeded
    before = _files(store.data_dir)
    plan = deletion.plan_delete(store, query)
    real = pl.DataFrame.write_parquet
    calls = 0

    def disk_full(frame, handle, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            handle.write(b"partial failed parquet")
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(frame, handle, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", disk_full)
    with pytest.raises(StorageWriteError, match="No space left"):
        deletion.execute_delete(store, plan, confirm=plan.operation_id)
    _assert_original(seeded, before)


def test_staged_roundtrip_mismatch_aborts_without_deleting_original(seeded, monkeypatch):
    store, _, query, _ = seeded
    before = _files(store.data_dir)
    plan = deletion.plan_delete(store, query)
    real = pl.read_parquet

    def damaged_read(path, **kwargs):
        result = real(path, **kwargs)
        if "staging" in Path(path).parts:
            return result.with_columns(pl.col("volume") + 1)
        return result

    monkeypatch.setattr(pl, "read_parquet", damaged_read)
    with pytest.raises(DataIntegrityError, match="differs from the original rows"):
        deletion.execute_delete(store, plan, confirm=plan.operation_id)
    _assert_original(seeded, before)


def test_publication_link_then_raise_removes_both_owned_links(seeded, monkeypatch):
    store, _, query, _ = seeded
    before = _files(store.data_dir)
    plan = deletion.plan_delete(store, query)

    def fail_after_link(staged, final):
        os.link(staged, final)
        raise OSError("Publication failed after linking")

    monkeypatch.setattr(deletion, "_publish_survivor", fail_after_link)
    with pytest.raises(StorageWriteError, match="after linking"):
        deletion.execute_delete(store, plan, confirm=plan.operation_id)
    _assert_original(seeded, before)


def test_catalog_insert_failure_rolls_back_tombstones_and_all_replacements(seeded, monkeypatch):
    store, _, query, _ = seeded
    before = _files(store.data_dir)
    plan = deletion.plan_delete(store, query)
    real = catalog.insert
    calls = 0

    def fail_second(connection, item):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Simulated catalog insertion failure")
        return real(connection, item)

    monkeypatch.setattr(catalog, "insert", fail_second)
    with pytest.raises(StorageWriteError, match="catalog insertion failure"):
        deletion.execute_delete(store, plan, confirm=plan.operation_id)
    _assert_original(seeded, before)


def test_cleanup_failure_is_pending_and_retry_finishes_exact_same_operation(seeded, monkeypatch):
    store, original, query, _ = seeded
    plan = deletion.plan_delete(store, query)
    real = deletion._remove_file

    def deny(_):
        raise PermissionError("Mocked access denied")

    monkeypatch.setattr(deletion, "_remove_file", deny)
    with pytest.raises(deletion.DeletionCleanupError) as caught:
        deletion.execute_delete(store, plan, confirm=plan.operation_id)
    assert caught.value.operation_id == plan.operation_id
    assert (store.data_dir / original.relative_path).exists()
    pending = deletion.get_deletion_report(store, plan.operation_id)
    assert pending.status == "pending" and pending.removed_bytes == 0
    assert deletion.list_deletions(store, pending_only=True) == [pending]
    with pytest.raises(DatasetNotFoundError):
        store.read_dataset(original.dataset_id)
    monkeypatch.setattr(deletion, "_remove_file", real)
    result = deletion.execute_delete(store, plan, confirm=plan.operation_id)
    assert result.status == "completed" and result.removed_bytes == plan.source_bytes
    assert not (store.data_dir / original.relative_path).exists()
    assert deletion.execute_delete(store, plan, confirm=plan.operation_id) == result
    assert deletion.list_deletions(store, pending_only=True) == []
    assert set(store.audit()) == set(result.replacement_ids)


def test_selected_staging_and_quarantine_copies_removed_unrelated_copy_preserved(seeded):
    store, original, query, _ = seeded
    source = store.data_dir / original.relative_path
    staged = store.data_dir / "staging" / f"{original.dataset_id}.parquet"
    quarantined = store.data_dir / "quarantine" / "previous" / original.relative_path
    unrelated = store.data_dir / "quarantine" / "unrelated.parquet"
    staged.parent.mkdir(exist_ok=True)
    quarantined.parent.mkdir(parents=True)
    os.link(source, staged)
    shutil.copyfile(source, quarantined)
    unrelated.write_bytes(b"Unrelated artifact must remain untouched")
    plan = deletion.plan_delete(store, query)
    assert len(plan.cleanup_files) == 3
    deletion.execute_delete(store, plan, confirm=plan.operation_id)
    assert not source.exists() and not staged.exists() and not quarantined.exists()
    assert unrelated.read_bytes() == b"Unrelated artifact must remain untouched"


def test_alias_checksum_mismatch_blocks_plan_instead_of_deleting_unknown_copy(seeded):
    store, original, query, _ = seeded
    alias = store.data_dir / "staging" / f"{original.dataset_id}.parquet"
    alias.parent.mkdir(exist_ok=True)
    alias.write_bytes(b"different data")
    before = _files(store.data_dir)
    with pytest.raises(DataIntegrityError, match="Unverified copy"):
        deletion.plan_delete(store, query)
    assert _files(store.data_dir) == before


def test_symlink_redirect_within_root_cannot_delete_another_layer(seeded):
    store, original, query, _ = seeded
    raw = store.data_dir / original.relative_path
    other = store.data_dir / "processed" / "other.parquet"
    other.parent.mkdir()
    raw.rename(other)
    raw.symlink_to(other)
    with pytest.raises(DataIntegrityError, match="redirected data paths"):
        deletion.plan_delete(store, query)
    assert raw.is_symlink() and other.exists()


@pytest.mark.parametrize("damage", ["omit_source", "unrelated_path"])
def test_malformed_cleanup_journal_never_reports_completion(seeded, monkeypatch, damage):
    store, original, query, _ = seeded
    plan = deletion.plan_delete(store, query)
    real = deletion._remove_file
    monkeypatch.setattr(deletion, "_remove_file", lambda _: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(deletion.DeletionCleanupError):
        deletion.execute_delete(store, plan, confirm=plan.operation_id)
    monkeypatch.setattr(deletion, "_remove_file", real)
    with duckdb.connect(str(store.catalog_path)) as connection:
        value = connection.execute("SELECT payload_json FROM deletion_operations").fetchone()[0]
        payload = json.loads(value)
        if damage == "omit_source":
            payload["cleanup_files"] = []
        else:
            payload["cleanup_files"].append({"relative_path": "metadata/catalog.duckdb",
                                              "checksum_sha256": original.checksum_sha256, "size": 0})
        connection.execute("UPDATE deletion_operations SET payload_json=?", [json.dumps(payload)])
    with pytest.raises(deletion.DeletionCleanupError):
        store.recover()
    assert deletion.get_deletion_report(store, plan.operation_id).status == "pending"
    # Restore the small journal; retry is idempotent even if a previous cleanup
    # attempt already unlinked one authorized path before finding corruption.
    with duckdb.connect(str(store.catalog_path)) as connection:
        connection.execute("UPDATE deletion_operations SET payload_json=?", [value])
    store.recover()
    assert deletion.get_deletion_report(store, plan.operation_id).status == "completed"


@pytest.mark.parametrize("crash_at", ["before_commit", "after_commit"])
def test_actual_process_death_recovers_files_without_quarantining_deleted_copies(seeded, tmp_path, crash_at):
    store, original, query, frame = seeded
    marker = tmp_path / "operation.txt"
    code = '''
import os, sys
from datetime import datetime
from pathlib import Path
from data_pipeline.models import DataQuery
from data_pipeline.storage.store import LocalDataStore
from data_pipeline.storage import deletion
store = LocalDataStore(sys.argv[1])
query = DataQuery(provider="yahoo",symbol="AAPL",timeframe="1h",
                  start=datetime.fromisoformat(sys.argv[2]),end=datetime.fromisoformat(sys.argv[3]))
plan = deletion.plan_delete(store, query)
Path(sys.argv[4]).write_text(plan.operation_id)
if sys.argv[5] == "before_commit":
    def crash(staged, final):
        os.link(staged, final)
        os._exit(91)
    deletion._publish_survivor = crash
else:
    deletion._remove_file = lambda path: os._exit(92)
deletion.execute_delete(store, plan, confirm=plan.operation_id)
'''
    result = subprocess.run([sys.executable, "-c", code, str(store.data_dir), query.start.isoformat(),
                             query.end.isoformat(), str(marker), crash_at], capture_output=True, text=True)
    assert result.returncode == (91 if crash_at == "before_commit" else 92), result.stderr
    operation_id = marker.read_text()
    restarted = LocalDataStore(store.data_dir)
    pending = deletion.list_deletions(restarted, pending_only=True)
    assert len(pending) == 1 and pending[0].operation_id == operation_id
    assert pending[0].phase == ("preparation" if crash_at == "before_commit" else "cleanup")
    restarted.recover()
    assert deletion.list_deletions(restarted, pending_only=True) == []
    assert not list((store.data_dir / "quarantine").rglob("*.parquet"))
    assert not list((store.data_dir / "staging").rglob("*.parquet"))
    if crash_at == "before_commit":
        assert restarted.list_datasets() == [original]
        assert_frame_equal(restarted.read_dataset(original.dataset_id), frame)
        assert set(_files(store.data_dir)) == {original.relative_path}
    else:
        report = deletion.get_deletion_report(restarted, operation_id)
        assert report.status == "completed"
        assert not (store.data_dir / original.relative_path).exists()
        assert set(restarted.audit()) == set(report.replacement_ids)
        expected = frame.filter(pl.col("timestamp") != query.start)
        assert_frame_equal(restarted.read(replace(query, start=original.request.start,
                                                   end=original.request.end)), expected)
    # A later whole-range deletion cannot leave precommit survivor copies behind.
    whole = deletion.plan_delete(restarted, replace(query, start=original.request.start, end=original.request.end))
    deletion.execute_delete(restarted, whole, confirm=whole.operation_id)
    assert not _files(store.data_dir)
