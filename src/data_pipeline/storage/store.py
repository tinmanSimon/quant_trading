"""Immutable raw and processed Parquet storage with transactional catalog publication."""

from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import duckdb
import polars as pl

from ..exceptions import (
    ConfirmationRequiredError, DataAlreadyExistsError, DataIntegrityError,
    DatasetNotFoundError, OverlappingDataError, RequestDataMismatchError,
    StorageError, StorageWriteError,
)
from ..models import DataQuery, DataRequest
from ..schemas import OHLCV_SCHEMA, validate_ohlcv
from ..processing.contracts import DataContract
from ..quality import FetchQuality, merge_quality
from . import catalog
from .layout import contained_path, relative_path
from .locking import store_lock
from .models import StoredDataset


class LocalDataStore:
    """A local POSIX store. All catalog operations hold one cross-process lock.

    Immutable files are flushed before publication; the DuckDB commit is the
    visibility boundary. Abandoned files after abrupt termination are retained
    by ``recover`` in quarantine. Reads by ID include historical revisions.
    """

    def __init__(self, data_dir: str | Path, *, lock_timeout: float = 10) -> None:
        self.data_dir = Path(data_dir).resolve()
        if self.data_dir.exists() and not self.data_dir.is_dir():
            raise StorageError(f"data_dir must be a directory: {self.data_dir}")
        self.lock_timeout = lock_timeout

    @property
    def catalog_path(self) -> Path:
        return contained_path(self.data_dir, "metadata/catalog.duckdb")

    @contextmanager
    def _session(self, *, create: bool = True):
        try:
            with store_lock(self.data_dir, self.lock_timeout):
                if not self.catalog_path.exists():
                    # An absent catalog is not proof of an empty store. Check
                    # under the same lock as initialization to avoid a race.
                    has_data = any(
                        path.is_file()
                        for folder in ("raw", "processed")
                        for path in contained_path(self.data_dir, folder).rglob("*")
                    )
                    if has_data or self.catalog_path.with_suffix(".duckdb.wal").exists():
                        raise DataIntegrityError(
                            "The catalog is missing but stored data or a journal remains; "
                            "restore metadata/catalog.duckdb from backup before proceeding."
                        )
                    if not create:
                        yield None
                        return
                with catalog.connect(self.catalog_path) as connection:
                    yield connection
        except duckdb.Error as error:
            raise StorageError(f"Catalog operation failed: {error}") from error

    def write_raw(
        self, request: DataRequest, frame: pl.DataFrame, *, quality: FetchQuality | None = None,
    ) -> StoredDataset:
        canonical = _validate_batch(request, frame)
        with self._session() as connection:
            return self._write(connection, request, canonical, layer="raw", quality=quality)

    def write_processed(
        self, parent_id: str | Sequence[str], frame: pl.DataFrame, *, pipeline_id: str,
        processors_json: str, output_contract: DataContract | None = None,
    ) -> StoredDataset:
        with self._session() as connection:
            parents = self._raw_parents(connection, parent_id)
            parent_ids = tuple(item.dataset_id for item in parents)
            parent = parents[0]
            if not pipeline_id or processors_json == "[]":
                raise RequestDataMismatchError("Processed storage requires a nonempty processor pipeline.")
            try:
                specs = json.loads(processors_json)
                if not isinstance(specs, list) or not specs or any(
                    not isinstance(spec, dict) or set(spec) != {"name", "version", "config"}
                    or not isinstance(spec["name"], str) or not spec["name"].strip()
                    or not isinstance(spec["version"], str) or not spec["version"].strip()
                    or not isinstance(spec["config"], dict) for spec in specs
                ):
                    raise ValueError("Invalid processor identities")
                canonical_json = json.dumps(specs, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if sha256(canonical_json.encode()).hexdigest() != pipeline_id:
                    raise ValueError("Pipeline fingerprint does not match processor metadata")
                processors_json = canonical_json
            except (TypeError, ValueError) as error:
                raise RequestDataMismatchError(f"Invalid processing lineage: {error}") from error
            contract = DataContract.from_request(parent.request) if output_contract is None else output_contract
            if not isinstance(contract, DataContract):
                raise RequestDataMismatchError("Processed output requires a DataContract.")
            canonical = contract.validate(frame)
            first, last = _bounds(canonical)
            # Source requests stay available through parent_ids. A processed
            # request describes the output, including labels outside intraday
            # request bounds (e.g. midnight for a session opening at 14:30).
            request = replace(parent.request, dataset=contract.dataset, timeframe=contract.timeframe,
                              start=first, end=last + timedelta(milliseconds=1))
            canonical = _validate_batch(request, canonical)
            for item in parents:
                self._read_verified(item)
            for existing in catalog.select(connection, DataQuery(layer="processed", pipeline_id=pipeline_id)):
                if set(existing.parent_ids) == set(parent_ids):
                    raise DataAlreadyExistsError(
                        f"Inputs {parent_ids} already processed by {pipeline_id}: {existing.dataset_id}"
                    )
            return self._write(connection, request, canonical, layer="processed",
                               pipeline_id=pipeline_id, processors_json=processors_json,
                               parent_ids=parent_ids, quality=merge_quality([item.quality for item in parents]))

    def _raw_parents(self, connection, raw_ids: str | Sequence[str]) -> list[StoredDataset]:
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, bytes):
            raise RequestDataMismatchError("Supply an ordered sequence of raw dataset IDs.")
        if not raw_ids or any(not isinstance(key, str) or not key for key in raw_ids):
            raise RequestDataMismatchError("At least one nonempty raw dataset ID is required.")
        if len(set(raw_ids)) != len(raw_ids):
            raise RequestDataMismatchError("Raw input IDs must be distinct.")
        items = [catalog.get(connection, key) for key in raw_ids]
        if any(not item.active or item.layer != "raw" for item in items):
            raise RequestDataMismatchError("Processing requires active raw input revisions.")
        if len({_identity(item.request) for item in items}) != 1:
            raise RequestDataMismatchError("Raw inputs must share dataset, provider, symbol and timeframe.")
        return sorted(items, key=lambda item: (item.first_timestamp, item.dataset_id))

    def read_raw_inputs(self, raw_ids: str | Sequence[str]) -> tuple[list[StoredDataset], pl.DataFrame]:
        """Load compatible active raw batches in timestamp order under one lock.

        Publication rechecks that every parent is still active after processing.
        """
        with self._session(create=False) as connection:
            if connection is None:
                raise DatasetNotFoundError("No raw datasets were found.")
            parents = self._raw_parents(connection, raw_ids)
            frame = pl.concat([self._read_verified(item) for item in parents]).sort("symbol", "timestamp")
            return parents, validate_ohlcv(frame)

    def replace_raw(
        self, dataset_id: str, frame: pl.DataFrame, *, confirm: str,
        quality: FetchQuality | None = None,
    ) -> StoredDataset:
        if confirm != dataset_id:
            raise ConfirmationRequiredError("confirm must equal the exact dataset ID being replaced.")
        with self._session() as connection:
            old = catalog.get(connection, dataset_id)
            if old.layer != "raw" or not old.active:
                raise RequestDataMismatchError("Replacement requires an active raw dataset.")
            canonical = _validate_batch(old.request, frame)
            if _bounds(canonical) != (old.first_timestamp, old.last_timestamp):
                raise RequestDataMismatchError("Replacement must cover the entire target's first/last bar range.")
            return self._write(connection, old.request, canonical, layer="raw", supersedes=(dataset_id,),
                               quality=quality)

    def compact_raw(self, dataset_ids: list[str], *, confirm: list[str]) -> StoredDataset:
        """Merge whole batches; keep old revisions and retire their derivatives."""
        if len(set(dataset_ids)) < 2 or len(dataset_ids) != len(set(dataset_ids)):
            raise RequestDataMismatchError("Compaction needs at least two distinct dataset IDs.")
        if sorted(confirm) != sorted(dataset_ids):
            raise ConfirmationRequiredError("Confirm every exact dataset ID to compact.")
        with self._session() as connection:
            items = [catalog.get(connection, key) for key in dataset_ids]
            if any(not item.active or item.layer != "raw" for item in items):
                raise RequestDataMismatchError("Compaction requires active raw datasets.")
            if len({_identity(item.request) for item in items}) != 1:
                raise RequestDataMismatchError("Compaction inputs must share provider, symbol and timeframe.")
            request = replace(items[0].request,
                              start=min(item.request.start for item in items),
                              end=max(item.request.end for item in items))
            frame = pl.concat([self._read_verified(item) for item in items]).sort("symbol", "timestamp")
            return self._write(connection, request, _validate_batch(request, frame),
                               layer="raw", supersedes=tuple(dataset_ids),
                               quality=merge_quality([item.quality for item in items]))

    def _write(self, connection, request, frame, *, layer, pipeline_id="",
               processors_json="[]", parent_ids=(), supersedes=(), quality=None) -> StoredDataset:
        if quality is None:
            quality = FetchQuality()
        if not isinstance(quality, FetchQuality):
            raise RequestDataMismatchError("quality must be a FetchQuality report.")
        first, last = _bounds(frame)
        existing = catalog.select(connection, DataQuery(
            layer=layer, provider=request.provider, symbol=request.symbol,
            timeframe=request.timeframe, pipeline_id=pipeline_id or None,
        ))
        conflicts = [item.dataset_id for item in existing
                     if item.dataset_id not in supersedes
                     and item.first_timestamp <= last and item.last_timestamp >= first]
        if conflicts:
            raise OverlappingDataError(f"Incoming data overlaps active datasets: {', '.join(conflicts)}")

        key = uuid4().hex
        relative = relative_path(request, layer, key, first.year)
        final = contained_path(self.data_dir, relative)
        staged = contained_path(self.data_dir, Path("staging") / f"{key}.parquet")
        published = False
        committed = False
        transaction = False
        staged_owned = False
        try:
            staged.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation: neither staging nor final may replace a file.
            with staged.open("xb") as handle:
                staged_owned = True
                frame.write_parquet(handle, compression="zstd", statistics=True, row_group_size=128_000)
                handle.flush()
                os.fsync(handle.fileno())
            persisted = pl.read_parquet(staged, hive_partitioning=False)
            if persisted.schema != frame.schema or not persisted.equals(frame):
                raise DataIntegrityError("Staged Parquet differs from the validated input.")
            item = StoredDataset(
                dataset_id=key, layer=layer, request=request, first_timestamp=first,
                last_timestamp=last, row_count=frame.height, relative_path=str(relative),
                checksum_sha256=_checksum(staged), schema_version=1, created_at=datetime.now(UTC),
                pipeline_id=pipeline_id, processors_json=processors_json,
                parent_ids=tuple(parent_ids), supersedes=tuple(supersedes),
                quality=quality,
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            connection.execute("BEGIN TRANSACTION")
            transaction = True
            if supersedes:
                retired = set(supersedes)
                # Raw replacement/compaction invalidates all active derived outputs.
                for derived in catalog.select(connection, DataQuery(layer="processed")):
                    if retired.intersection(derived.parent_ids):
                        retired.add(derived.dataset_id)
                for retired_id in retired:
                    connection.execute("UPDATE datasets SET active=false WHERE dataset_id=?", [retired_id])
            catalog.insert(connection, item)
            _publish(staged, final)
            published = True
            # Flush the directory chain so newly created partitions are durable too.
            directory = final.parent
            while directory != self.data_dir.parent:
                _sync_directory(directory)
                directory = directory.parent
            connection.execute("COMMIT")
            transaction = False
            committed = True
            return item
        except Exception as error:
            if transaction:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error as rollback_error:
                    raise StorageWriteError(
                        "Catalog outcome is uncertain; data retained for audit/recover."
                    ) from rollback_error
            # A failing publication hook may have linked successfully before raising.
            own_final = (staged_owned and final.exists() and staged.exists()
                         and os.path.samefile(staged, final))
            if (published or own_final) and not committed:
                final.unlink(missing_ok=True)
            if isinstance(error, StorageError):
                raise
            raise StorageWriteError(f"Failed to store {layer} data: {error}") from error
        finally:
            # A stale staging link is harmless and recoverable after a successful commit.
            try:
                if staged_owned:
                    staged.unlink(missing_ok=True)
            except OSError:
                if not committed:
                    raise

    def get_metadata(self, dataset_id: str) -> StoredDataset:
        with self._session(create=False) as connection:
            if connection is None:
                raise DatasetNotFoundError(f"Dataset {dataset_id!r} was not found.")
            return catalog.get(connection, dataset_id)

    def list_datasets(self, query: DataQuery | None = None) -> list[StoredDataset]:
        """Without a query, list active datasets in both layers."""
        with self._session(create=False) as connection:
            if connection is None:
                return []
            if query is not None:
                return catalog.select(connection, query)
            return catalog.select(connection, DataQuery()) + catalog.select(connection, DataQuery(layer="processed"))

    def read_dataset(self, dataset_id: str) -> pl.DataFrame:
        return self._read_verified(self.get_metadata(dataset_id))

    def _verified_path(self, item: StoredDataset) -> Path:
        if item.schema_version != 1:
            raise DataIntegrityError(f"Unsupported data schema version: {item.schema_version}")
        path = contained_path(self.data_dir, item.relative_path)
        if not path.is_file():
            raise DataIntegrityError(f"Cataloged data file is missing: {path}")
        if _checksum(path) != item.checksum_sha256:
            raise DataIntegrityError(f"Checksum mismatch: {path}")
        if pl.read_parquet_schema(path) != OHLCV_SCHEMA:
            raise DataIntegrityError(f"Stored schema mismatch: {path}")
        return path

    def _read_verified(self, item: StoredDataset) -> pl.DataFrame:
        try:
            frame = pl.read_parquet(self._verified_path(item), hive_partitioning=False)
            _validate_batch(item.request, frame)
            if frame.height != item.row_count or _bounds(frame) != (item.first_timestamp, item.last_timestamp):
                raise DataIntegrityError(f"Catalog metadata mismatch: {item.dataset_id}")
            return frame
        except DataIntegrityError:
            raise
        except Exception as error:
            raise DataIntegrityError(f"Cannot read dataset {item.dataset_id}: {error}") from error

    def scan(self, query: DataQuery, *, columns: list[str] | None = None) -> pl.LazyFrame:
        """Snapshot immutable files, verify checksums, then push filters into Parquet scans."""
        if query.include_history:
            raise RequestDataMismatchError("Query history via list_datasets and read_dataset by ID; revisions may overlap.")
        items = self.list_datasets(query)
        identities = {(_identity(item.request), item.pipeline_id) for item in items}
        # Multiple symbols are fine; provider/timeframe/pipeline mixtures are ambiguous.
        if len({(identity[0][0], identity[0][1], identity[0][3], identity[1]) for identity in identities}) > 1:
            raise RequestDataMismatchError("Narrow query to a single provider, timeframe and processor pipeline.")
        paths = [str(self._verified_path(item)) for item in items]
        lazy = (pl.scan_parquet(paths, hive_partitioning=False) if paths
                else pl.DataFrame(schema=OHLCV_SCHEMA).lazy())
        if query.start is not None:
            lazy = lazy.filter(pl.col("timestamp").cast(pl.Datetime("us", "UTC")) >= pl.lit(query.start, dtype=pl.Datetime("us", "UTC")))
        if query.end is not None:
            lazy = lazy.filter(pl.col("timestamp").cast(pl.Datetime("us", "UTC")) < pl.lit(query.end, dtype=pl.Datetime("us", "UTC")))
        lazy = lazy.sort("symbol", "timestamp")
        if columns is not None:
            if not columns or len(columns) != len(set(columns)) or set(columns) - set(OHLCV_SCHEMA):
                raise RequestDataMismatchError("Select distinct canonical OHLCV columns.")
            lazy = lazy.select(columns)
        return lazy

    def read(self, query: DataQuery, *, columns: list[str] | None = None) -> pl.DataFrame:
        return self.scan(query, columns=columns).collect()

    def recover(self) -> list[str]:
        """Move uncommitted Parquet/staging artifacts to quarantine; never delete history."""
        moved = []
        with self._session() as connection:
            known = {row[0] for row in connection.execute("SELECT relative_path FROM datasets").fetchall()}
            for folder in ("raw", "processed", "staging"):
                base = contained_path(self.data_dir, folder)
                for source in sorted(base.rglob("*.parquet")):
                    relative = source.relative_to(self.data_dir)
                    contained_path(self.data_dir, relative)
                    if str(relative) in known:
                        continue
                    destination = contained_path(self.data_dir, Path("quarantine") / uuid4().hex / relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(destination)
                    moved.append(str(destination.relative_to(self.data_dir)))
        return moved

    def audit(self) -> list[str]:
        """Verify all cataloged revisions, returning IDs that passed integrity checks."""
        items = self.list_datasets(DataQuery(include_history=True))
        items += self.list_datasets(DataQuery(layer="processed", include_history=True))
        for item in items:
            self._read_verified(item)
        return [item.dataset_id for item in items]

    # Initial raw-store API remains available.
    get_raw = get_metadata
    read_raw = read_dataset

    def list_raw(self) -> list[StoredDataset]:
        return self.list_datasets(DataQuery())


RawDataStore = LocalDataStore


def _identity(request: DataRequest) -> tuple[str, str, str, str]:
    return request.dataset, request.provider, request.symbol, request.timeframe


def _bounds(frame: pl.DataFrame) -> tuple[datetime, datetime]:
    return frame["timestamp"].min(), frame["timestamp"].max()


def _validate_batch(request: DataRequest, frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(request, DataRequest):
        raise RequestDataMismatchError("A validated DataRequest is required.")
    frame = validate_ohlcv(frame)
    if frame["symbol"].unique().to_list() != [request.symbol]:
        raise RequestDataMismatchError("Data must contain exactly the request's symbol.")
    first, last = _bounds(frame)
    if first < request.start or last >= request.end:
        raise RequestDataMismatchError("Data falls outside the request's [start, end) bounds.")
    if not request.timeframe.endswith(("m", "h")):
        if frame.select((pl.col("timestamp") != pl.col("timestamp").dt.truncate("1d")).any()).item():
            raise RequestDataMismatchError(
                "Daily and longer bars require session-date labels at midnight UTC."
            )
    return frame


def _checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _publish(staged: Path, final: Path) -> None:
    try:
        os.link(staged, final)
    except FileExistsError as error:
        raise DataAlreadyExistsError(f"Refusing to replace existing file: {final}") from error


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
