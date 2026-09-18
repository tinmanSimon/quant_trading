"""Physical, non-cascading range deletion with recoverable filesystem cleanup.

The catalog transaction is the visibility boundary. Surviving rows are verified
before publication; old files are removed only after tombstones and the cleanup
journal commit together. A completed operation has no retained original copies
in the store's publication, staging, or quarantine directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import duckdb
import polars as pl

from ..exceptions import (
    ConfirmationRequiredError, DataIntegrityError, DatasetNotFoundError,
    RequestDataMismatchError, StorageError, StorageWriteError,
)
from ..models import DataQuery, DataRequest
from . import catalog
from .layout import contained_path, relative_path
from .models import StoredDataset


class DeletionPlanStaleError(StorageError):
    """The preview no longer describes the selected data or belongs elsewhere."""


class DeletionCleanupError(StorageError):
    """Catalog deletion committed, but physical cleanup needs a retry."""

    def __init__(self, operation_id: str, message: str):
        self.operation_id = operation_id
        super().__init__(
            f"Deletion {operation_id} is pending physical cleanup: {message}. "
            "Run recover() to retry; the data is not yet confirmed physically removed."
        )


@dataclass(frozen=True)
class DeletionItem:
    dataset_id: str
    active: bool
    removed_rows: int
    retained_rows: int
    before_rows: int
    after_rows: int
    source_bytes: int
    relative_path: str
    checksum_sha256: str


@dataclass(frozen=True)
class _CleanupFile:
    relative_path: str
    checksum_sha256: str
    size: int


@dataclass(frozen=True)
class DeletionPlan:
    operation_id: str
    query: DataQuery
    data_root: str
    created_at: datetime
    fingerprint: str
    items: tuple[DeletionItem, ...]
    cleanup_files: tuple[_CleanupFile, ...]

    @property
    def removed_rows(self) -> int:
        """Stored rows across every affected revision, including history."""
        return sum(item.removed_rows for item in self.items)

    @property
    def retained_rows(self) -> int:
        return sum(item.retained_rows for item in self.items)

    @property
    def source_bytes(self) -> int:
        return sum(item.source_bytes for item in self.items)

    @property
    def estimated_temporary_bytes(self) -> int:
        # A split can produce two small Parquet files, each with its own footer.
        # This is an estimate, not a promise about compression or available disk.
        return sum(item.source_bytes * ((item.before_rows > 0) + (item.after_rows > 0))
                   for item in self.items)

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([asdict(item) for item in self.items], schema={
            "dataset_id": pl.String, "active": pl.Boolean,
            "removed_rows": pl.Int64, "retained_rows": pl.Int64,
            "before_rows": pl.Int64, "after_rows": pl.Int64,
            "source_bytes": pl.Int64, "relative_path": pl.String,
            "checksum_sha256": pl.String,
        })


@dataclass(frozen=True)
class DeletionReport:
    operation_id: str
    status: str
    removed_rows: int
    deleted_ids: tuple[str, ...]
    replacement_ids: tuple[str, ...]
    removed_bytes: int
    written_bytes: int
    phase: str = "cleanup"

    def to_dict(self) -> dict:
        return asdict(self)


def _query(query: DataQuery) -> DataQuery:
    if not isinstance(query, DataQuery):
        raise RequestDataMismatchError("Deletion requires a validated DataQuery.")
    if any(getattr(query, name) is None for name in ("provider", "symbol", "timeframe", "start", "end")):
        raise RequestDataMismatchError("Deletion requires provider, symbol, timeframe, start and end.")
    if query.layer == "processed" and query.pipeline_id is None:
        raise RequestDataMismatchError("Processed deletion requires an explicit pipeline_id.")
    if query.layer == "raw" and query.pipeline_id is not None:
        raise RequestDataMismatchError("Raw deletion must not specify pipeline_id.")
    return replace(query, include_history=True)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=lambda item: item.isoformat(), allow_nan=False)


def _range_frames(frame, query):
    # Cast the stored millisecond labels up, never truncate user boundaries.
    timestamp = pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    start = pl.lit(query.start, dtype=pl.Datetime("us", "UTC"))
    end = pl.lit(query.end, dtype=pl.Datetime("us", "UTC"))
    return frame.filter(timestamp < start), frame.filter(timestamp >= end)


def _safe_file(root: Path, relative: str) -> Path:
    path = contained_path(root, relative)
    # Do not resolve a selected raw link into some other dataset's file.
    if root / relative != path:
        raise DataIntegrityError(f"Deletion refuses redirected data paths: {relative}")
    return path


def _snapshot(store, connection, query, *, operation_id, created_at):
    from .store import _checksum

    candidates = [] if connection is None else catalog.select(connection, query)
    entries, cleanup, affected, fingerprint_data = [], [], {}, []
    for item in candidates:
        expected = str(relative_path(item.request, item.layer, item.dataset_id, item.first_timestamp.year))
        if item.relative_path != expected or (
            item.layer, item.request.provider, item.request.symbol, item.request.timeframe, item.pipeline_id
        ) != (query.layer, query.provider, query.symbol, query.timeframe, query.pipeline_id or ""):
            raise DataIntegrityError(f"Deletion source identity or path mismatch: {item.dataset_id}")
        path = _safe_file(store.data_dir, item.relative_path)
        frame = store._read_verified(item)
        before, after = _range_frames(frame, query)
        removed = frame.height - before.height - after.height
        fingerprint_data.append(item.to_json())
        if not removed:
            continue
        entry = DeletionItem(item.dataset_id, item.active, removed,
                             before.height + after.height, before.height, after.height,
                             path.stat().st_size, item.relative_path, item.checksum_sha256)
        entries.append(entry)
        affected[item.dataset_id] = item
        cleanup.append(_CleanupFile(item.relative_path, item.checksum_sha256, entry.source_bytes))
    # Publication uses hard links. Include known abandoned links/copies of these
    # exact revisions so deleting the cataloged path actually releases them.
    if affected:
        for folder in ("staging", "quarantine"):
            for alias in sorted(_safe_file(store.data_dir, folder).rglob("*.parquet")):
                old = affected.get(alias.stem)
                if old is None:
                    continue
                relative = str(alias.relative_to(store.data_dir))
                alias = _safe_file(store.data_dir, relative)
                if not alias.is_file() or _checksum(alias) != old.checksum_sha256:
                    raise DataIntegrityError(f"Unverified copy of selected dataset: {relative}")
                cleanup.append(_CleanupFile(relative, old.checksum_sha256, alias.stat().st_size))
    cleanup.sort(key=lambda item: item.relative_path)
    fingerprint = sha256(_json({
        "root": str(store.data_dir), "query": asdict(query),
        "metadata": fingerprint_data, "cleanup": [asdict(item) for item in cleanup],
    }).encode()).hexdigest()
    return DeletionPlan(operation_id, query, str(store.data_dir), created_at,
                        fingerprint, tuple(entries), tuple(cleanup))


def plan_delete(store, query: DataQuery) -> DeletionPlan:
    """Preview deletion across all revisions of one explicitly selected series."""
    query = _query(query)
    with store._session(create=False) as connection:
        if connection is not None:
            _require_clean_journal(connection)
        return _snapshot(store, connection, query, operation_id=uuid4().hex, created_at=datetime.now(UTC))


def _require_clean_journal(connection):
    if (connection.execute("SELECT 1 FROM deletion_operations WHERE status='pending' LIMIT 1").fetchone()
            or connection.execute("SELECT 1 FROM deletion_preparations LIMIT 1").fetchone()):
        raise StorageError("Unfinished deletion files remain; run recover() before another deletion.")


def _publish_survivor(staged: Path, final: Path):
    # Exclusive publication; never overwrite a pre-existing path.
    os.link(staged, final)


def _stage_survivor(store, connection, operation_id, old, frame, query, *, before):
    from .store import _checksum, _sync_directory, _validate_batch

    request = replace(old.request, end=min(old.request.end, query.start)) if before else replace(
        old.request, start=max(old.request.start, query.end))
    frame = _validate_batch(request, frame)
    key = uuid4().hex
    first, last = frame["timestamp"].min(), frame["timestamp"].max()
    relative = relative_path(request, old.layer, key, first.year)
    staged_relative = str(Path("staging") / "deletions" / operation_id / f"{key}.parquet")
    staged = _safe_file(store.data_dir, staged_relative)
    final = _safe_file(store.data_dir, str(relative))
    if staged.exists() or final.exists():
        raise StorageWriteError("Refusing to replace an existing deletion preparation file.")
    # Persist ownership before the first file is created. A process death before
    # catalog publication must not leave untracked copies of surviving rows.
    preparation = json.loads(connection.execute(
        "SELECT payload_json FROM deletion_preparations WHERE operation_id=?", [operation_id],
    ).fetchone()[0])
    preparation["files"].append({"dataset_id": key, "relative_path": str(relative),
                                 "staged": staged_relative, "request": asdict(request),
                                 "layer": old.layer, "year": first.year,
                                 "checksum_sha256": None, "published": False})
    connection.execute("UPDATE deletion_preparations SET payload_json=? WHERE operation_id=?",
                       [_json(preparation), operation_id])
    staged.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("xb") as handle:
        frame.write_parquet(handle, compression="zstd", statistics=True, row_group_size=128_000)
        handle.flush()
        os.fsync(handle.fileno())
    persisted = pl.read_parquet(staged, hive_partitioning=False)
    if persisted.schema != frame.schema or not persisted.equals(frame):
        raise DataIntegrityError("Deletion survivor Parquet differs from the original rows.")
    item = replace(old, dataset_id=key, request=request, first_timestamp=first, last_timestamp=last,
                   row_count=frame.height, relative_path=str(relative), checksum_sha256=_checksum(staged),
                   created_at=datetime.now(UTC), supersedes=(old.dataset_id,))
    preparation["files"][-1]["checksum_sha256"] = item.checksum_sha256
    connection.execute("UPDATE deletion_preparations SET payload_json=? WHERE operation_id=?",
                       [_json(preparation), operation_id])
    final.parent.mkdir(parents=True, exist_ok=True)
    _publish_survivor(staged, final)
    preparation["files"][-1]["published"] = True
    connection.execute("UPDATE deletion_preparations SET payload_json=? WHERE operation_id=?",
                       [_json(preparation), operation_id])
    # Remove our staging link before committing: replacements then have exactly
    # one store-owned link, and recovery will not quarantine an unnecessary copy.
    staged.unlink()
    _sync_directory(staged.parent)
    directory = final.parent
    while directory != store.data_dir.parent:
        _sync_directory(directory)
        directory = directory.parent
    return item, final.stat().st_size


def _discard_preparation(store, connection, operation_id):
    """Remove only this operation's unpublished replacements, including partial writes."""
    from .store import _checksum, _sync_directory

    row = connection.execute("SELECT payload_json FROM deletion_preparations WHERE operation_id=?",
                             [operation_id]).fetchone()
    if row is None:
        return
    try:
        payload = json.loads(row[0])
        if payload["data_root"] != str(store.data_dir) or not re.fullmatch(r"[0-9a-f]{32}", operation_id):
            raise DataIntegrityError("Deletion preparation belongs to another store or operation.")
        for entry in payload["files"]:
            key = entry["dataset_id"]
            fields = dict(entry["request"])
            for field in ("start", "end"):
                fields[field] = datetime.fromisoformat(fields[field])
            request = DataRequest(**fields)
            if (not re.fullmatch(r"[0-9a-f]{32}", key) or entry["layer"] not in ("raw", "processed")
                    or entry["relative_path"] != str(relative_path(request, entry["layer"], key, entry["year"]))
                    or entry["staged"] != str(Path("staging") / "deletions" / operation_id / f"{key}.parquet")):
                raise DataIntegrityError("Invalid deletion preparation file identity.")
            for relative in (entry["relative_path"], entry["staged"]):
                if connection.execute("SELECT 1 FROM datasets WHERE relative_path=? OR dataset_id=?",
                                      [relative, key]).fetchone():
                    raise DataIntegrityError("Deletion preparation unexpectedly refers to published data.")
                path = _safe_file(store.data_dir, relative)
                if relative == entry["relative_path"] and path.exists():
                    staged = _safe_file(store.data_dir, entry["staged"])
                    if not entry["published"]:
                        # An exclusive publication can lose to another file.
                        # Only a still-shared staging inode proves ownership in
                        # the crash window before the published flag persisted.
                        if not staged.exists() or not os.path.samefile(staged, path):
                            continue
                    if _checksum(path) != entry["checksum_sha256"]:
                        raise DataIntegrityError("Prepared replacement changed before rollback.")
                path.unlink(missing_ok=True)
                if path.parent.exists():
                    _sync_directory(path.parent)
        connection.execute("DELETE FROM deletion_preparations WHERE operation_id=?", [operation_id])
    except Exception as error:
        raise StorageWriteError(
            f"Deletion {operation_id} was not committed; its prepared files need recover(): {error}"
        ) from error


def _report(payload, status):
    return DeletionReport(
        operation_id=payload["operation_id"], status=status,
        removed_rows=payload["removed_rows"], deleted_ids=tuple(payload["deleted_ids"]),
        replacement_ids=tuple(payload["replacement_ids"]),
        removed_bytes=payload["removed_bytes"] if status == "completed" else 0,
        written_bytes=payload["written_bytes"],
    )


def _load_operation(connection, operation_id):
    row = connection.execute(
        "SELECT status, payload_json FROM deletion_operations WHERE operation_id=?", [operation_id],
    ).fetchone()
    if row is None:
        raise DatasetNotFoundError(f"Deletion operation {operation_id!r} was not found.")
    try:
        payload = json.loads(row[1])
        if row[0] not in ("pending", "completed") or payload["operation_id"] != operation_id:
            raise ValueError("Invalid deletion state or identity")
        _report(payload, row[0])
    except (KeyError, TypeError, ValueError) as error:
        raise DataIntegrityError(f"Malformed deletion journal: {operation_id}") from error
    return row[0], payload


def _remove_file(path):
    """Small failure-injection boundary for filesystem-cleanup tests."""
    path.unlink()


def _finish_cleanup(store, connection, payload):
    from .store import _checksum, _sync_directory

    operation_id = payload["operation_id"]
    try:
        if payload["data_root"] != str(store.data_dir):
            raise DataIntegrityError("Deletion journal belongs to a different data root.")
        # Only tombstoned revisions owned by this operation can authorize unlink.
        originals = {}
        for dataset_id, value in connection.execute(
            "SELECT dataset_id, metadata_json FROM deleted_datasets WHERE operation_id=?", [operation_id],
        ).fetchall():
            fields = json.loads(value)
            item = StoredDataset.from_json(value, active=fields["active"])
            if item.dataset_id != dataset_id or item.relative_path != str(
                relative_path(item.request, item.layer, item.dataset_id, item.first_timestamp.year)
            ):
                raise DataIntegrityError("Malformed deletion tombstone identity.")
            originals[dataset_id] = item
        if set(originals) != set(payload["deleted_ids"]):
            raise DataIntegrityError("Deletion tombstones do not match the cleanup journal.")
        cleanup_paths = [entry["relative_path"] for entry in payload["cleanup_files"]]
        if (len(cleanup_paths) != len(set(cleanup_paths))
                or not {old.relative_path for old in originals.values()} <= set(cleanup_paths)):
            raise DataIntegrityError("Deletion cleanup journal omits or repeats source paths.")
        for entry in payload["cleanup_files"]:
            relative = Path(entry["relative_path"])
            old = originals.get(relative.stem)
            if (old is None or entry["checksum_sha256"] != old.checksum_sha256
                    or relative.suffix != ".parquet"
                    or not (str(relative) == old.relative_path
                            or relative.parts[0] in ("staging", "quarantine"))):
                raise DataIntegrityError("Cleanup path is outside the selected revisions.")
            path = _safe_file(store.data_dir, entry["relative_path"])
            if path.exists():
                if not path.is_file() or _checksum(path) != entry["checksum_sha256"]:
                    raise DataIntegrityError(f"Cleanup file changed: {entry['relative_path']}")
                # Never trust the journal alone to remove another live file.
                if connection.execute("SELECT 1 FROM datasets WHERE relative_path=?",
                                      [entry["relative_path"]]).fetchone():
                    raise DataIntegrityError("Cleanup path is still referenced by a dataset.")
                _remove_file(path)
            if path.parent.exists():
                _sync_directory(path.parent)
        connection.execute("UPDATE deletion_operations SET status='completed' WHERE operation_id=?",
                           [operation_id])
    except Exception as error:
        raise DeletionCleanupError(operation_id, str(error)) from error
    return _report(payload, "completed")


def execute_delete(store, plan: DeletionPlan, *, confirm: str) -> DeletionReport:
    """Publish survivors and physically remove only the exact confirmed selection."""
    if not isinstance(plan, DeletionPlan):
        raise RequestDataMismatchError("Supply a DeletionPlan from plan_delete().")
    if confirm != plan.operation_id:
        raise ConfirmationRequiredError("confirm must equal the exact deletion operation_id.")
    if plan.data_root != str(store.data_dir):
        raise DeletionPlanStaleError("Deletion plan belongs to a different data root.")
    if not re.fullmatch(r"[0-9a-f]{32}", plan.operation_id):
        raise RequestDataMismatchError("Invalid deletion operation identity.")
    query = _query(plan.query)
    with store._session() as connection:
        existing = connection.execute("SELECT 1 FROM deletion_operations WHERE operation_id=?",
                                      [plan.operation_id]).fetchone()
        if existing:
            status, payload = _load_operation(connection, plan.operation_id)
            if payload["fingerprint"] != plan.fingerprint or payload["data_root"] != plan.data_root:
                raise DeletionPlanStaleError("Deletion operation does not match this plan.")
            return _report(payload, status) if status == "completed" else _finish_cleanup(store, connection, payload)
        _discard_preparation(store, connection, plan.operation_id)
        _require_clean_journal(connection)
        current = _snapshot(store, connection, query, operation_id=plan.operation_id, created_at=plan.created_at)
        if current != plan:
            raise DeletionPlanStaleError("Selected storage changed after preview; create a new deletion plan.")
        replacements, originals = [], []
        transaction = False
        committed = False
        uncertain = False
        written_bytes = 0
        try:
            connection.execute("INSERT INTO deletion_preparations VALUES (?,?)",
                               [plan.operation_id, _json({"data_root": str(store.data_dir), "files": []})])
            for entry in plan.items:
                old = catalog.get(connection, entry.dataset_id)
                before, after = _range_frames(store._read_verified(old), query)
                originals.append(old)
                for part, is_before in ((before, True), (after, False)):
                    if part.height:
                        survivor, size = _stage_survivor(store, connection, plan.operation_id,
                                                         old, part, query, before=is_before)
                        replacements.append(survivor)
                        written_bytes += size
            payload = {
                "operation_id": plan.operation_id, "data_root": plan.data_root,
                "fingerprint": plan.fingerprint, "query": asdict(query),
                "created_at": plan.created_at, "committed_at": datetime.now(UTC),
                "removed_rows": plan.removed_rows,
                "deleted_ids": [item.dataset_id for item in originals],
                "replacement_ids": [item.dataset_id for item in replacements],
                # Logical source file sizes; hard-link aliases do not count twice.
                "removed_bytes": plan.source_bytes, "written_bytes": written_bytes,
                "cleanup_files": [asdict(item) for item in plan.cleanup_files],
            }
            connection.execute("BEGIN TRANSACTION")
            transaction = True
            for old in originals:
                connection.execute("INSERT INTO deleted_datasets VALUES (?,?,?,?)",
                                   [old.dataset_id, plan.operation_id, datetime.now(UTC), old.to_json()])
                connection.execute("DELETE FROM datasets WHERE dataset_id=?", [old.dataset_id])
            for item in replacements:
                catalog.insert(connection, item)
            connection.execute("INSERT INTO deletion_operations VALUES (?,?,?)",
                               [plan.operation_id, "pending", _json(payload)])
            connection.execute("DELETE FROM deletion_preparations WHERE operation_id=?", [plan.operation_id])
            connection.execute("COMMIT")
            transaction = False
            committed = True
        except Exception as error:
            if transaction:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error as rollback_error:
                    uncertain = True
                    raise StorageWriteError(
                        "Deletion catalog outcome is uncertain; files retained for audit/recover."
                    ) from rollback_error
            if isinstance(error, StorageError):
                raise
            raise StorageWriteError(f"Deletion could not be published: {error}") from error
        finally:
            if not committed and not uncertain:
                _discard_preparation(store, connection, plan.operation_id)
        return _finish_cleanup(store, connection, payload)


def get_deletion_report(store, operation_id: str) -> DeletionReport:
    with store._session(create=False) as connection:
        if connection is None:
            raise DatasetNotFoundError(f"Deletion operation {operation_id!r} was not found.")
        if connection.execute("SELECT 1 FROM deletion_preparations WHERE operation_id=?",
                              [operation_id]).fetchone():
            # Recovery rolls these uncommitted operations back. No original rows
            # have been removed, and no replacement is visible in the catalog.
            return DeletionReport(operation_id, "pending", 0, (), (), 0, 0, "preparation")
        status, payload = _load_operation(connection, operation_id)
        return _report(payload, status)


def list_deletions(store, *, pending_only: bool = False) -> list[DeletionReport]:
    with store._session(create=False) as connection:
        if connection is None:
            return []
        condition = " WHERE status='pending'" if pending_only else ""
        ids = connection.execute("SELECT operation_id FROM deletion_operations" + condition
                                 + " ORDER BY operation_id").fetchall()
        reports = []
        for (operation_id,) in ids:
            status, payload = _load_operation(connection, operation_id)
            reports.append(_report(payload, status))
        for (operation_id,) in connection.execute(
            "SELECT operation_id FROM deletion_preparations ORDER BY operation_id",
        ).fetchall():
            reports.append(DeletionReport(operation_id, "pending", 0, (), (), 0, 0, "preparation"))
        return reports


def resume_deletions(store, connection) -> list[str]:
    """Retry cleanup before orphan quarantine, under the caller's store lock."""
    completed = []
    for (operation_id,) in connection.execute("SELECT operation_id FROM deletion_preparations").fetchall():
        _discard_preparation(store, connection, operation_id)
    for (operation_id,) in connection.execute(
        "SELECT operation_id FROM deletion_operations WHERE status='pending' ORDER BY operation_id",
    ).fetchall():
        _, payload = _load_operation(connection, operation_id)
        _finish_cleanup(store, connection, payload)
        completed.append(operation_id)
    return completed
