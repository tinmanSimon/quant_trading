"""Sequential independent fetch attempts with structured outcomes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import TYPE_CHECKING, Callable

import polars as pl

from .exceptions import EmptyDataError, InvalidDataRequestError
from .fetch_planning import plan_fetch, plan_missing_fetch
from .models import DataQuery, DataRequest
from .providers import MassiveProvider, ProviderRegistry, YFinanceProvider
from .providers.massive import _validate_request_interval
from .quality import merge_quality

if TYPE_CHECKING:
    from .api import DataPipeline


@dataclass(frozen=True)
class FetchOutcome:
    ticker: str
    status: str
    dataset_ids: tuple[str, ...] = ()
    row_count: int = 0
    quality: dict | None = None
    error_type: str | None = None
    error_message: str | None = None
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    fetch_start: datetime | None = None
    fetch_end: datetime | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    replaced_dataset_ids: tuple[str, ...] = ()
    fetch_ranges: tuple[dict, ...] = ()
    fetched_row_count: int = 0
    warning_message: str | None = None


@dataclass(frozen=True)
class BatchFetchReport:
    outcomes: tuple[FetchOutcome, ...]

    @property
    def ok(self):
        return all(item.status not in ("failed", "empty") for item in self.outcomes)

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([
            {"ticker": item.ticker, "status": item.status, "row_count": item.row_count,
             "dataset_ids": ", ".join(item.dataset_ids), "error_type": item.error_type,
             "error_message": item.error_message,
             "quality": json.dumps(item.quality) if item.quality is not None else None,
             "requested_start": item.requested_start, "requested_end": item.requested_end,
             "fetch_start": item.fetch_start, "fetch_end": item.fetch_end,
             "first_timestamp": item.first_timestamp, "last_timestamp": item.last_timestamp,
             "replaced_dataset_ids": ", ".join(item.replaced_dataset_ids),
             "fetch_ranges": json.dumps(item.fetch_ranges), "fetched_row_count": item.fetched_row_count,
             "warning_message": item.warning_message}
            for item in self.outcomes
        ], schema={"ticker": pl.String, "status": pl.String, "row_count": pl.Int64,
                   "dataset_ids": pl.String, "error_type": pl.String, "error_message": pl.String,
                   "quality": pl.String, "requested_start": pl.Datetime("us", "UTC"),
                   "requested_end": pl.Datetime("us", "UTC"), "fetch_start": pl.Datetime("us", "UTC"),
                   "fetch_end": pl.Datetime("us", "UTC"), "first_timestamp": pl.Datetime("us", "UTC"),
                   "last_timestamp": pl.Datetime("us", "UTC"), "replaced_dataset_ids": pl.String,
                   "fetch_ranges": pl.String, "fetched_row_count": pl.Int64, "warning_message": pl.String})


def _quality_warning(quality, request):
    recorded = {bar.timestamp for bar in quality.omitted_bars
                if request.start <= bar.timestamp < request.end}
    if recorded:
        return (f"Dataset provenance records {len(recorded)} excluded bar timestamp(s) in the requested range. "
                "See quality for original reasons. Fetch success or stored-range coverage does not certify "
                "complete data; strict research must pass calendar and source-coverage validation.")
    return None


def fetch_many(pipeline: DataPipeline, *, tickers, start, end, timeframe="1d",
               provider="yahoo", skip_missing_ohlc: bool | None = None,
               skip_invalid_bars: bool | None = None,
               massive_request_interval_seconds: float | None = None,
               overlap_policy: str = "error",
               progress: Callable[[dict], None] | None = None) -> BatchFetchReport:
    """Save each ticker independently, recording ordinary failures and quality.

    ``None`` preserves the pipeline's registered provider configuration. An
    explicit Yahoo skip setting uses a separate provider for this batch only,
    preserving the other policy and timeout from the registered provider.
    Massive uses one batch-local provider so pacing also spans ticker changes;
    an explicit interval overrides its configured value without changing the registry.
    ``error`` keeps strict overlap rejection. ``refresh_overlap`` skips covered
    requests and otherwise fetches the union with overlapping active raw batches,
    atomically superseding them only after preserving all their stored timestamps.
    ``fetch_missing`` requests only uncovered subranges and merges those bars
    with verified local originals in one publication per ticker. Empty subranges
    are reported; a nonempty failure publishes nothing for that ticker.
    ``progress`` is a best-effort observer receiving copied event dictionaries;
    ordinary observer errors cannot cancel or misreport a storage transaction.
    Interrupts and SystemExit propagate instead of becoming ticker failures.
    """
    if overlap_policy not in ("error", "refresh_overlap", "fetch_missing"):
        raise InvalidDataRequestError("overlap_policy must be 'error', 'refresh_overlap', or 'fetch_missing'.")
    if progress is not None and not callable(progress):
        raise InvalidDataRequestError("progress must be callable or None.")
    if isinstance(tickers, (str, bytes)):
        raise InvalidDataRequestError("Supply a list of ticker symbols.")
    tickers = tuple(tickers)
    if not tickers:
        raise InvalidDataRequestError("Supply at least one ticker.")
    if massive_request_interval_seconds is not None:
        if not isinstance(provider, str) or provider.strip().lower() != "massive":
            raise InvalidDataRequestError("massive_request_interval_seconds is a Massive-specific setting.")
        try:
            massive_request_interval_seconds = _validate_request_interval(massive_request_interval_seconds)
        except ValueError as exc:
            raise InvalidDataRequestError(str(exc)) from None
    if skip_missing_ohlc is not None or skip_invalid_bars is not None:
        if provider.strip().lower() != "yahoo":
            raise InvalidDataRequestError("skip_missing_ohlc and skip_invalid_bars are Yahoo-specific settings.")
        # Import at call time: the public DataPipeline API delegates here.
        from .api import DataPipeline

        current = pipeline.providers.get("yahoo")
        options = {name: getattr(current, name, default) for name, default in
                   (("timeout", 10.0), ("skip_missing_ohlc", False), ("skip_invalid_bars", False))}
        if skip_missing_ohlc is not None:
            options["skip_missing_ohlc"] = skip_missing_ohlc
        if skip_invalid_bars is not None:
            options["skip_invalid_bars"] = skip_invalid_bars
        pipeline = DataPipeline(pipeline.store.data_dir, providers=ProviderRegistry({
            "yahoo": YFinanceProvider(**options),
        }))
    outcomes = []
    batch_pipeline = None
    for index, ticker in enumerate(tickers, 1):
        symbol = str(ticker)
        details, ranges = {}, []

        def notify(status):
            if progress is not None:
                event = {"ticker": symbol, "index": index, "total": len(tickers), "status": status,
                         "requested_start": details.get("requested_start"),
                         "requested_end": details.get("requested_end"), "fetch_ranges": deepcopy(ranges)}
                try:
                    progress(event)
                except Exception:
                    # Presentation failures must not change publication outcomes.
                    pass

        try:
            request = DataRequest(symbol=ticker, start=start, end=end, timeframe=timeframe, provider=provider)
            symbol = request.symbol
            details.update(requested_start=request.start, requested_end=request.end)
            supersedes, covering = (), ()
            requests, merge_request = (request,), request
            if overlap_policy != "error":
                existing = pipeline.list_datasets(DataQuery(
                    layer="raw", dataset=request.dataset, provider=request.provider,
                    symbol=request.symbol, timeframe=request.timeframe,
                ))
                if overlap_policy == "fetch_missing":
                    plan = plan_missing_fetch(request, existing)
                    requests, merge_request = plan.fetch_requests, plan.merge_request
                else:
                    plan = plan_fetch(request, existing)
                    requests = (plan.fetch_request,) if plan.fetch_request is not None else ()
                supersedes, covering = plan.supersedes, plan.existing_dataset_ids
            if not requests:
                quality = merge_quality([item.quality for item in existing if item.dataset_id in covering])
                quality_data = json.loads(json.dumps(asdict(quality), default=lambda value: value.isoformat()))
                outcomes.append(FetchOutcome(symbol, "skipped", covering, quality=quality_data,
                                             warning_message=_quality_warning(quality, request), **details))
                notify("skipped")
                continue
            ranges = [{"start": item.start.isoformat(), "end": item.end.isoformat(),
                       "status": "pending", "row_count": 0} for item in requests]
            details.update(fetch_start=requests[0].start, fetch_end=requests[-1].end)
            notify("fetching")
            if batch_pipeline is None:
                if request.provider == "massive":
                    adapter = pipeline.providers.get("massive")
                    if isinstance(adapter, MassiveProvider):
                        adapter = adapter.with_request_interval(massive_request_interval_seconds)
                    elif massive_request_interval_seconds is not None:
                        raise InvalidDataRequestError("Massive request pacing requires a MassiveProvider.")
                    from .api import DataPipeline

                    batch_pipeline = DataPipeline(pipeline.store.data_dir, providers=ProviderRegistry({
                        "massive": adapter,
                    }))
                else:
                    batch_pipeline = pipeline
            if overlap_policy == "fetch_missing":
                chunks = []
                adapter = batch_pipeline.providers.get(request.provider)
                for gap, record in zip(requests, ranges):
                    record["status"] = "fetching"
                    notify("fetching")
                    try:
                        fetched = adapter.fetch_result(gap)
                        if fetched.frame.is_empty():
                            raise EmptyDataError("Provider returned no bars for this missing range.")
                    except EmptyDataError:
                        record["status"] = "empty"
                        notify("fetching")
                        continue
                    except Exception:
                        record["status"] = "failed"
                        raise
                    record.update(status="fetched", row_count=fetched.frame.height)
                    details["fetched_row_count"] = details.get("fetched_row_count", 0) + fetched.frame.height
                    chunks.append((gap, fetched))
                    notify("fetching")
                if not chunks:
                    raise EmptyDataError("No bars returned for any missing range; existing data was preserved.")
                result = batch_pipeline.store.merge_raw(merge_request, chunks, supersedes=supersedes)
                empty_count = sum(item["status"] == "empty" for item in ranges)
                if empty_count:
                    details["warning_message"] = (f"{empty_count} missing range(s) returned no bars. "
                                                  "Only returned bars and existing history were saved; "
                                                  "see fetch_ranges for the empty ranges.")
                status = "merged" if supersedes else "saved"
            else:
                target = requests[0]
                ranges[0]["status"] = "fetching"
                notify("fetching")
                try:
                    fetched = batch_pipeline.providers.get(target.provider).fetch_result(target)
                except Exception:
                    ranges[0]["status"] = "failed"
                    raise
                ranges[0].update(status="fetched", row_count=fetched.frame.height)
                details["fetched_row_count"] = fetched.frame.height
                if supersedes:
                    result = batch_pipeline.store.refresh_raw(
                        target, fetched.frame, supersedes=supersedes, quality=fetched.quality,
                    )
                else:
                    result = batch_pipeline.store.write_raw(target, fetched.frame, quality=fetched.quality)
                status = "refreshed" if supersedes else "saved"
            metadata = json.loads(result.to_json())
            quality_warning = _quality_warning(result.quality, request)
            if quality_warning:
                details["warning_message"] = " ".join(filter(None, [details.get("warning_message"), quality_warning]))
            outcomes.append(FetchOutcome(
                symbol, status, (result.dataset_id,), result.row_count,
                quality=metadata.get("quality"), first_timestamp=result.first_timestamp,
                last_timestamp=result.last_timestamp, replaced_dataset_ids=supersedes,
                fetch_ranges=tuple(deepcopy(ranges)), **details,
            ))
            notify(status)
        except Exception as exc:
            # Deliberate interrupt/SystemExit still stop the batch.
            status = "empty" if overlap_policy != "error" and isinstance(exc, EmptyDataError) else "failed"
            outcomes.append(FetchOutcome(symbol, status, error_type=type(exc).__name__,
                                         error_message=str(exc), fetch_ranges=tuple(deepcopy(ranges)), **details))
            notify(status)
    return BatchFetchReport(tuple(outcomes))
