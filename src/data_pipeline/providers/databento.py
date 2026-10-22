"""Databento historical OHLCV with explicit feed identity and lazy SDK access.

Only native 1-minute and 1-hour bars are downloaded. Prices are unadjusted,
timestamps identify UTC interval starts, and absent bars are never fabricated.
The SDK handles fixed-point prices and date-dependent instrument mappings.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from copy import copy
from datetime import UTC, datetime, timedelta
import importlib
import math
import os
from pathlib import Path
import re
from typing import Any
import warnings

import polars as pl

from ..exceptions import (
    EmptyDataError, InvalidDataRequestError, ProviderError,
    SchemaValidationError, UnsupportedRequestError,
)
from ..models import DataRequest
from ..quality import FetchQuality, FetchResult
from ..schemas.ohlcv import CANONICAL_OHLCV_COLUMNS, validate_ohlcv
from .base import BaseDataProvider

_SCHEMAS = {"1m": "ohlcv-1m", "1h": "ohlcv-1h"}
_DATASET = re.compile(r"[A-Z0-9]+\.[A-Z0-9]+\Z")
_JOB_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_JOB_FIELDS = frozenset({
    "id", "state", "dataset", "symbols", "schema", "start", "end",
    "cost_usd", "record_count", "billed_size", "actual_size", "progress",
    "ts_received", "ts_queued", "ts_process_start", "ts_process_done",
    "ts_expiration", "encoding", "compression", "split_symbols",
    "split_duration", "stype_in", "stype_out",
})


def provider_name(dataset: str) -> str:
    """Return a stable storage namespace distinct from every other feed."""
    return "databento-" + _dataset_code(dataset).lower()


def _dataset_code(dataset: str) -> str:
    if not isinstance(dataset, str) or not _DATASET.fullmatch(dataset.strip().upper()):
        raise InvalidDataRequestError("Choose an explicit Databento dataset code, such as XNAS.ITCH.")
    return dataset.strip().upper()


class DatabentoProvider(BaseDataProvider):
    """Official SDK adapter; constructing it performs no I/O and needs no key.

    ``dataset=None`` supports dataset discovery only. Fetches must use an
    explicit dataset and its namespaced provider name. ``client`` and
    ``store_loader`` permit offline testing without importing the SDK.
    """

    name = "databento"
    supported_timeframes = frozenset(_SCHEMAS)
    supported_datasets = frozenset({"ohlcv"})
    supported_price_adjustments = frozenset({"unadjusted"})

    def __init__(self, dataset: str | None = None, *, api_key: str | None = None,
                 client: Any = None, store_loader: Any = None,
                 chunk_size: int = 100_000, download_workers: int = 4) -> None:
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be a nonempty string or None.")
        if type(chunk_size) is not int or not 1 <= chunk_size <= 1_000_000:
            raise ValueError("chunk_size must be an integer in [1, 1000000].")
        if type(download_workers) is not int or not 1 <= download_workers <= 8:
            raise ValueError("download_workers must be an integer in [1, 8].")
        self.download_workers = download_workers
        self.dataset = _dataset_code(dataset) if dataset is not None else None
        self.name = provider_name(self.dataset) if self.dataset else "databento"
        self._api_key = api_key.strip() if api_key is not None else None
        self._client = client
        self._store_loader = store_loader
        self.chunk_size = chunk_size

    def with_dataset(self, dataset: str) -> DatabentoProvider:
        """Bind the same credentials/client to another explicit dataset."""
        provider = copy(self)
        provider.dataset = _dataset_code(dataset)
        provider.name = provider_name(provider.dataset)
        return provider

    def _require_dataset(self) -> str:
        if self.dataset is None:
            raise UnsupportedRequestError("Choose a Databento dataset before requesting data.")
        return self.dataset

    def _sdk_client(self):
        if self._client is None:
            key = self._api_key or os.environ.get("DATABENTO_API_KEY", "").strip()
            if not key or not key.isascii() or any(c.isspace() for c in key):
                raise ProviderError("Set DATABENTO_API_KEY or pass a valid api_key to DatabentoProvider.")
            sdk = _sdk()
            try:
                self._client = sdk.Historical(key=key)
            except Exception:
                raise ProviderError("Could not initialize the Databento historical client.") from None
        return self._client

    def _call(self, family: str, method: str, **kwargs):
        client = self._sdk_client()
        try:
            return getattr(getattr(client, family), method)(**kwargs)
        except Exception as exc:
            # SDK errors can contain request headers, signed URLs or credentials.
            status = getattr(exc, "http_status", None) or getattr(exc, "status_code", None)
            detail = {401: "authentication failed", 403: "dataset access denied",
                      429: "rate limit reached; retry later"}.get(status, "request failed")
            raise ProviderError(f"Databento {family}.{method}: {detail}.") from None

    def list_datasets(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._call("metadata", "list_datasets"))

    def list_schemas(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._call(
            "metadata", "list_schemas", dataset=self._require_dataset()))

    def get_dataset_range(self) -> dict:
        return dict(self._call("metadata", "get_dataset_range", dataset=self._require_dataset()))

    def get_dataset_conditions(self, start: datetime, end: datetime) -> list[dict]:
        start, end = _bounds(start, end)
        return list(self._call(
            "metadata", "get_dataset_condition", dataset=self._require_dataset(),
            start_date=start.date(), end_date=(end - timedelta(microseconds=1)).date()))

    def preflight_symbols(self, symbols: Iterable[str], start: datetime, end: datetime) -> dict:
        """Resolve symbols without buying data; partially available symbols remain usable."""
        start, end = _bounds(start, end)
        requested = _symbols(symbols)
        last_day = (end - timedelta(microseconds=1)).date() + timedelta(days=1)
        resolved = self._call(
            "symbology", "resolve", dataset=self._require_dataset(), symbols=list(requested),
            stype_in="raw_symbol", stype_out="instrument_id", start_date=start.date(), end_date=last_day)
        if not isinstance(resolved, Mapping) or not isinstance(resolved.get("result"), Mapping):
            raise ProviderError("Databento returned invalid symbol resolution metadata.")
        missing = set(resolved.get("not_found", ()))
        partial = set(resolved.get("partial", ()))
        valid, errors, notices = [], {}, []
        for symbol in requested:
            if symbol in missing or not resolved["result"].get(symbol):
                errors[symbol] = "Ticker did not resolve in this dataset and date range."
            else:
                valid.append(symbol)
                if symbol in partial:
                    notices.append(f"{symbol}: symbol resolves for only part of the requested date range.")
        return {"valid": tuple(valid), "errors": errors, "warnings": notices}

    def _request_args(self, symbols, start, end, timeframe) -> dict:
        if timeframe not in _SCHEMAS:
            raise UnsupportedRequestError("Databento supports native 1m and 1h bars; derive 30m locally from 1m.")
        start, end = _bounds(start, end)
        return {"dataset": self._require_dataset(), "symbols": list(_symbols(symbols)),
                "schema": _SCHEMAS[timeframe], "start": start, "end": end,
                "stype_in": "raw_symbol"}

    def estimate_cost(self, symbols, start, end, timeframe) -> float:
        value = self._call("metadata", "get_cost", **self._request_args(symbols, start, end, timeframe))
        try:
            cost = float(value)
        except (ValueError, TypeError, OverflowError):
            raise ProviderError("Databento returned an invalid cost estimate.") from None
        if isinstance(value, bool) or not math.isfinite(cost) or cost < 0:
            raise ProviderError("Databento returned an invalid cost estimate.")
        return cost

    def submit_job(self, symbols, start, end, timeframe) -> dict:
        """Submit once. Callers must persist intent first and never retry blindly."""
        value = self._call(
            "batch", "submit_job", **self._request_args(symbols, start, end, timeframe),
            stype_out="instrument_id", encoding="dbn", compression="zstd",
            split_symbols=False, split_duration="day", delivery="download")
        return _job_details(value)

    def job_status(self, job_id: str) -> dict:
        return _job_details(self._call("batch", "get_job_details", job_id=_job_id(job_id)))

    def download_job(self, job_id: str, directory: str | Path, *, progress=None) -> tuple[Path, ...]:
        """Download a saved job with one manifest lookup and bounded concurrency.

        Progress contains byte/file counts and runs on the caller's thread.
        Existing complete files are verified and partial files resume by offset.
        """
        from .databento_download import download_files

        job_id = _job_id(job_id)
        destination = Path(directory).resolve()
        target = destination / job_id
        if not target.resolve().is_relative_to(destination):
            raise ProviderError("Invalid Databento job download directory.")
        manifest = self._call("batch", "list_files", job_id=job_id)
        key = self._sdk_client().key
        return download_files(manifest, target, key=key, workers=self.download_workers,
                              progress=progress)

    def iter_frames(self, paths: Iterable[str | Path], timeframe: str) -> Iterator[pl.DataFrame]:
        """Decode DBN files in bounded chunks, retaining mapped symbols per date.

        Per-ticker OHLCV validation is left to the importer so a bad ticker
        cannot invalidate another ticker. Missing mapping metadata is fatal.
        """
        if timeframe not in _SCHEMAS:
            raise UnsupportedRequestError("Databento supports only native 1m and 1h bars.")
        loader = self._store_loader or _sdk().DBNStore.from_file
        for value in paths:
            path = Path(value)
            if not (path.name.endswith(".dbn") or path.name.endswith(".dbn.zst")):
                continue  # Metadata/support files are downloaded too.
            try:
                store = loader(path)
            except Exception:
                raise ProviderError("Could not read a downloaded Databento DBN file.") from None
            yield from self._store_frames(store, timeframe)

    def _store_frames(self, store, timeframe) -> Iterator[pl.DataFrame]:
        if str(store.dataset) != self._require_dataset() or str(store.schema) != _SCHEMAS[timeframe]:
            raise SchemaValidationError("Databento file dataset/schema does not match the request.")
        try:
            # Truncated DBN files otherwise produce warnings and partial rows.
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*truncated.*", category=Warning)
                frames = store.to_df(price_type="float", pretty_ts=True, map_symbols=True,
                                     tz="UTC", count=self.chunk_size)
                for frame in frames:
                    normalized = _canonical_frame(frame)
                    if not normalized.is_empty():
                        yield normalized
        except (SchemaValidationError, ProviderError):
            raise
        except Exception:
            raise ProviderError("Could not decode complete Databento OHLCV data.") from None

    def fetch(self, request: DataRequest) -> pl.DataFrame:
        return self.fetch_result(request).frame

    def fetch_result(self, request: DataRequest) -> FetchResult:
        if not isinstance(request, DataRequest):
            raise InvalidDataRequestError("fetch expects a DataRequest.")
        self._require_dataset()
        if (request.provider != self.name or request.dataset != "ohlcv"
                or request.price_adjustment != "unadjusted"):
            raise UnsupportedRequestError("Databento request must match its dataset-specific provider and unadjusted OHLCV.")
        store = self._call("timeseries", "get_range", **self._request_args(
            [request.symbol], request.start, request.end, request.timeframe), stype_out="instrument_id")
        chunks = list(self._store_frames(store, request.timeframe))
        if not chunks:
            raise EmptyDataError(f"Databento returned no bars for {request.symbol!r}.")
        frame = pl.concat(chunks).filter(
            (pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end))
        if frame.is_empty():
            raise EmptyDataError(f"Databento returned no bars for {request.symbol!r} in [start, end).")
        if frame["symbol"].unique().to_list() != [request.symbol]:
            raise SchemaValidationError("Databento returned a symbol other than the requested ticker.")
        frame = validate_ohlcv(frame.sort("symbol", "timestamp"))
        return FetchResult(frame, FetchQuality(status="unknown", skip_missing_ohlc=False,
                                               provider_version="databento-historical-v0"))


def _sdk():
    try:
        return importlib.import_module("databento")
    except ImportError:
        raise ProviderError("Databento SDK is not installed. Install the project's requirements.txt.") from None


def _bounds(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if any(not isinstance(value, datetime) or value.utcoffset() is None for value in (start, end)):
        raise InvalidDataRequestError("Databento range bounds must be timezone-aware datetimes.")
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if start >= end:
        raise InvalidDataRequestError("start must be earlier than end.")
    return start, end


def _symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    if isinstance(symbols, str):
        symbols = (symbols,)
    result = []
    for symbol in symbols:
        if (not isinstance(symbol, str) or not symbol or symbol != symbol.strip()
                or "," in symbol or any(c.isspace() for c in symbol)
                or symbol == "ALL_SYMBOLS"):
            raise InvalidDataRequestError("Databento requires explicit individual ticker symbols.")
        if symbol not in result:
            result.append(symbol)
    if not 1 <= len(result) <= 2000:
        raise InvalidDataRequestError("Databento requests require between 1 and 2000 distinct tickers.")
    return tuple(result)


def _job_id(value) -> str:
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise ProviderError("Invalid Databento job ID.")
    return value


def _job_details(value) -> dict:
    if not isinstance(value, Mapping):
        raise ProviderError("Databento returned invalid batch job metadata.")
    _job_id(value.get("id"))
    # Never save account identifiers, API key labels, signed URLs or headers.
    return {key: item for key, item in value.items() if key in _JOB_FIELDS}


def _canonical_frame(frame) -> pl.DataFrame:
    try:
        frame = pl.from_pandas(frame.reset_index())
        required = {"ts_event", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise SchemaValidationError("Databento OHLCV data lacks timestamps, mapped symbols or price/volume fields.")
        timestamp = frame.schema["ts_event"]
        if not isinstance(timestamp, pl.Datetime) or timestamp.time_zone != "UTC":
            raise SchemaValidationError("Databento bar timestamps must be timezone-aware UTC.")
        if frame["symbol"].null_count() or frame["symbol"].cast(pl.String).str.len_chars().min() == 0:
            raise SchemaValidationError("Databento did not resolve every instrument ID to a ticker.")
        # Keep source precision/types until each symbol is validated. In
        # particular, uint64 volume must not silently round when cast to float.
        return frame.rename({"ts_event": "timestamp"}).select(
            pl.col("timestamp"), pl.col("symbol").cast(pl.String),
            *(pl.col(name) for name in CANONICAL_OHLCV_COLUMNS[2:]),
        ).sort("symbol", "timestamp")
    except SchemaValidationError:
        raise
    except Exception:
        raise SchemaValidationError("Could not normalize Databento OHLCV fields.") from None
