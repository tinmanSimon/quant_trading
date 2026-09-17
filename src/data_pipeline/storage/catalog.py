"""Catalog is the source of truth for published immutable files.

Callers hold the store lock for the lifetime of each connection.
"""

from contextlib import contextmanager
from pathlib import Path

import duckdb

from ..exceptions import DataIntegrityError, DatasetNotFoundError
from ..models import DataQuery
from .models import StoredDataset


@contextmanager
def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_catalog = not path.exists()
    connection = duckdb.connect(str(path))
    try:
        connection.execute("SET TimeZone='UTC'")
        if new_catalog:
            # Publish the schema atomically. Never repair an existing catalog by
            # creating empty replacement tables: its data may still be on disk.
            connection.execute("BEGIN TRANSACTION")
            connection.execute("CREATE TABLE catalog_version(version INTEGER)")
            connection.execute("INSERT INTO catalog_version VALUES (1)")
            connection.execute('''CREATE TABLE datasets (
                dataset_id VARCHAR PRIMARY KEY, layer VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL, provider VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, timeframe VARCHAR NOT NULL,
                first_timestamp TIMESTAMPTZ NOT NULL, last_timestamp TIMESTAMPTZ NOT NULL,
                pipeline_id VARCHAR NOT NULL, active BOOLEAN NOT NULL,
                relative_path VARCHAR UNIQUE NOT NULL, metadata_json VARCHAR NOT NULL
            )''')
            connection.execute("COMMIT")
        else:
            tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
            if not {"catalog_version", "datasets"} <= tables:
                raise DataIntegrityError("Incomplete catalog schema; restore the catalog from backup.")
            versions = connection.execute("SELECT version FROM catalog_version").fetchall()
            if versions != [(1,)]:
                raise DataIntegrityError("Unsupported or missing catalog version; refusing to modify it.")
        yield connection
    finally:
        connection.close()


def get(connection, dataset_id: str) -> StoredDataset:
    row = connection.execute(
        "SELECT metadata_json, active FROM datasets WHERE dataset_id=?", [dataset_id]
    ).fetchone()
    if row is None:
        raise DatasetNotFoundError(f"Dataset {dataset_id!r} was not found.")
    return StoredDataset.from_json(row[0], active=row[1])


def select(connection, query: DataQuery) -> list[StoredDataset]:
    conditions = ["layer=?", "dataset=?"]
    values = [query.layer, query.dataset]
    if not query.include_history:
        conditions.append("active=true")
    for name in ("provider", "symbol", "timeframe", "pipeline_id"):
        value = getattr(query, name)
        if value is not None:
            if name == "symbol" and query.provider in (None, "yahoo"):
                # Also handles provider-unspecified discovery without changing
                # the case-sensitive identities of other vendors.
                conditions.append("((provider='yahoo' AND upper(symbol)=?) OR (provider<>'yahoo' AND symbol=?))")
                values.extend([value.upper(), value])
            else:
                conditions.append(f"{name}=?")
                values.append(value)
    if query.start is not None:
        conditions.append("last_timestamp >= ?")
        values.append(query.start)
    if query.end is not None:
        conditions.append("first_timestamp < ?")
        values.append(query.end)
    rows = connection.execute(
        "SELECT metadata_json, active FROM datasets WHERE " + " AND ".join(conditions)
        + " ORDER BY symbol, first_timestamp, dataset_id", values,
    ).fetchall()
    return [StoredDataset.from_json(row[0], active=row[1]) for row in rows]


def insert(connection, item: StoredDataset) -> None:
    request = item.request
    connection.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        item.dataset_id, item.layer, request.dataset, request.provider,
        request.symbol, request.timeframe, item.first_timestamp, item.last_timestamp,
        item.pipeline_id, item.active, item.relative_path, item.to_json(),
    ])
