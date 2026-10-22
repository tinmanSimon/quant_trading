"""Cost-planned, restartable Databento jobs with independent ticker publication.

Only ``advance(..., submit=True)`` authorizes billing. Authorization and an
intent journal are flushed *before* submitting each remote job; an ambiguous
submission is never retried automatically. Downloaded DBN and staged Arrow data
remain local so importing a result again never submits a paid replacement job.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

import polars as pl
import pyarrow as pa

from .batch_fetch import BatchFetchReport, FetchOutcome
from .exceptions import EmptyDataError, InvalidDataRequestError, ProviderError
from .fetch_planning import plan_fetch, plan_missing_fetch
from .models import DataQuery, DataRequest
from .quality import FetchResult
from .schemas import validate_ohlcv
from .storage.layout import contained_path
from .storage.locking import store_lock

_MAX_SYMBOLS = 2000
_TERMINAL = {"completed", "completed_with_errors", "stale"}
_DATES = {"requested_start", "requested_end", "fetch_start", "fetch_end", "first_timestamp", "last_timestamp"}


def _root(pipeline):
    return contained_path(pipeline.store.data_dir, "metadata/databento")


def _path(pipeline, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise InvalidDataRequestError("Invalid Databento plan ID.")
    return contained_path(_root(pipeline), identifier + ".json")


def _save(pipeline, plan):
    path = _path(pipeline, plan["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    plan["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix("." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_plan(pipeline, plan_id):
    """Read a saved plan without constructing a provider or contacting Databento."""
    with _path(pipeline, plan_id).open(encoding="utf-8") as handle:
        return json.load(handle)


def list_plans(pipeline):
    root = _root(pipeline)
    if not root.exists():
        return []
    return sorted((load_plan(pipeline, path.stem) for path in root.glob("*.json")),
                  key=lambda plan: plan["created_at"], reverse=True)


def _provider(pipeline, dataset):
    from .providers.databento import DatabentoProvider, provider_name

    name = provider_name(dataset)
    if name in pipeline.providers.names():
        return pipeline.providers.get(name)
    if "databento" in pipeline.providers.names():
        return pipeline.providers.get("databento").with_dataset(dataset)
    return DatabentoProvider(dataset)


def _encode_request(request):
    result = asdict(request)
    result.update(start=request.start.isoformat(), end=request.end.isoformat())
    return result


def _request(value):
    return DataRequest(**{**value, "start": datetime.fromisoformat(value["start"]),
                          "end": datetime.fromisoformat(value["end"])})


def _existing(pipeline, request):
    return pipeline.list_datasets(DataQuery(layer="raw", dataset=request.dataset,
        provider=request.provider, symbol=request.symbol, timeframe=request.timeframe))


def _snapshot(pipeline, ticker_plans):
    result = {symbol: [] for symbol in ticker_plans}
    if ticker_plans:
        request = _request(next(iter(ticker_plans.values()))["request"])
        for item in pipeline.list_datasets(DataQuery(layer="raw", dataset=request.dataset,
                provider=request.provider, timeframe=request.timeframe)):
            if item.request.symbol in result:
                result[item.request.symbol].append(item.dataset_id)
    return {symbol: sorted(identifiers) for symbol, identifiers in result.items()}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _outcome(symbol, status, request=None, **values):
    if request is not None:
        values.update(requested_start=request["start"], requested_end=request["end"])
    return {"ticker": symbol, "status": status, **values}


def report_from_plan(plan):
    """Reuse the regular batch report, including timestamps and per-ticker errors."""
    records = []
    for value in plan["outcomes"]:
        fields = {key: datetime.fromisoformat(item) if key in _DATES and item is not None else item
                  for key, item in value.items()}
        for key in ("dataset_ids", "replaced_dataset_ids", "fetch_ranges"):
            if key in fields:
                fields[key] = tuple(fields[key])
        records.append(FetchOutcome(**fields))
    return BatchFetchReport(tuple(records))


def estimate(pipeline, tickers, *, start, end, dataset, timeframe="1m",
             overlap_policy="fetch_missing", max_estimated_cost=None):
    """Plan actual uncovered requests and estimate them without submitting jobs."""
    from .providers.databento import provider_name

    if overlap_policy not in {"error", "fetch_missing", "refresh_overlap"}:
        raise InvalidDataRequestError("Unsupported overlap policy.")
    if isinstance(tickers, (str, bytes)):
        raise InvalidDataRequestError("Supply a list of ticker symbols.")
    tickers = list(tickers)
    if not tickers:
        raise InvalidDataRequestError("Supply at least one ticker.")
    if max_estimated_cost is not None:
        if isinstance(max_estimated_cost, bool):
            raise InvalidDataRequestError("Maximum estimated cost must be a nonnegative finite number.")
        try:
            max_estimated_cost = float(max_estimated_cost)
        except (TypeError, ValueError, OverflowError):
            raise InvalidDataRequestError("Maximum estimated cost must be a nonnegative finite number.") from None
        if not math.isfinite(max_estimated_cost) or max_estimated_cost < 0:
            raise InvalidDataRequestError("Maximum estimated cost must be a nonnegative finite number.")
    provider = _provider(pipeline, dataset)
    name = provider_name(dataset)
    # Validate global bounds and interval even when every ticker is invalid.
    prototype = DataRequest("VALIDATION", start, end, timeframe=timeframe, provider=name)
    if prototype.timeframe not in provider.supported_timeframes:
        raise InvalidDataRequestError("Databento downloads support native 1m and 1h bars; resample 1m locally.")
    plan = dict(version=1, id=uuid4().hex, created_at=datetime.now(UTC).isoformat(),
                status="estimated", authorized=False, dataset=provider.dataset, provider=name,
                timeframe=prototype.timeframe, start=prototype.start.isoformat(), end=prototype.end.isoformat(),
                overlap_policy=overlap_policy, max_estimated_cost=max_estimated_cost,
                tickers=[], ticker_plans={}, groups=[], outcomes=[], warnings=[], estimated_cost=0.0)
    grouped = {}
    local = {}
    for item in pipeline.list_datasets(DataQuery(layer="raw", dataset=prototype.dataset,
            provider=name, timeframe=prototype.timeframe)):
        local.setdefault(item.request.symbol, []).append(item)
    for ticker in tickers:
        try:
            request = DataRequest(ticker, prototype.start, prototype.end,
                                  timeframe=prototype.timeframe, provider=name)
            if (request.symbol in {"ALL_SYMBOLS", "*"} or "," in request.symbol
                    or any(char.isspace() for char in request.symbol)):
                raise InvalidDataRequestError("An explicit ticker list is required; ALL_SYMBOLS is not allowed.")
            if request.symbol in plan["tickers"]:
                continue
            plan["tickers"].append(request.symbol)
            existing = local.get(request.symbol, ())
            supersedes, covering, merge_request = (), (), request
            if overlap_policy == "fetch_missing":
                selected = plan_missing_fetch(request, existing)
                requests, merge_request = selected.fetch_requests, selected.merge_request
                supersedes, covering = selected.supersedes, selected.existing_dataset_ids
            elif overlap_policy == "refresh_overlap":
                selected = plan_fetch(request, existing)
                requests = (selected.fetch_request,) if selected.fetch_request else ()
                supersedes, covering = selected.supersedes, selected.existing_dataset_ids
            else:
                if any(item.first_timestamp < request.end and item.last_timestamp >= request.start for item in existing):
                    raise InvalidDataRequestError("Requested range overlaps local raw data; choose fetch missing or refresh overlap.")
                requests = (request,)
            item_plan = dict(request=_encode_request(request), merge_request=_encode_request(merge_request),
                             supersedes=list(supersedes), requests=[_encode_request(item) for item in requests],
                             groups=[], status="pending")
            plan["ticker_plans"][request.symbol] = item_plan
            if not requests:
                item_plan["status"] = "skipped"
                plan["outcomes"].append(_outcome(request.symbol, "skipped", item_plan["request"], dataset_ids=list(covering)))
            for item in requests:
                grouped.setdefault((item.start.isoformat(), item.end.isoformat()), []).append(request.symbol)
        except (ValueError, InvalidDataRequestError) as exc:
            plan["outcomes"].append(_outcome(str(ticker), "failed", error_type=type(exc).__name__, error_message=str(exc)))
    plan["snapshot"] = _snapshot(pipeline, plan["ticker_plans"])
    plan["request_fingerprint"] = _digest({key: plan[key] for key in
        ("dataset", "timeframe", "start", "end", "tickers", "overlap_policy")})
    plan["fingerprint"] = _digest({key: plan[key] for key in
        ("dataset", "timeframe", "start", "end", "tickers", "overlap_policy", "max_estimated_cost", "snapshot")})
    # Repeated estimate clicks or browser reruns must not create duplicate jobs.
    for previous in list_plans(pipeline):
        if _same_active_plan(previous, plan):
            return previous
    if grouped:
        schemas = provider.list_schemas()
        schema = {"1m": "ohlcv-1m", "1h": "ohlcv-1h"}[prototype.timeframe]
        if schema not in schemas:
            raise InvalidDataRequestError(f"Dataset {dataset} does not offer {schema}.")
        available = provider.get_dataset_range()
        bounds = available.get("schema", {}).get(schema, available)
        lower, upper = bounds.get("start"), bounds.get("end")
        if not lower or not upper:
            raise ProviderError("Databento did not provide available history bounds for this schema.")
        for begin, finish in grouped:
            if ((lower and datetime.fromisoformat(begin) < _parse_stamp(lower))
                    or (upper and datetime.fromisoformat(finish) > _parse_stamp(upper))):
                raise InvalidDataRequestError(f"Requested range is outside {dataset} {schema} history: {lower} to {upper}.")
        for (begin, finish), symbols in grouped.items():
            for offset in range(0, len(symbols), _MAX_SYMBOLS):
                group_symbols = symbols[offset:offset + _MAX_SYMBOLS]
                checked = provider.preflight_symbols(group_symbols, _parse_stamp(begin), _parse_stamp(finish))
                plan["warnings"].extend(checked.get("warnings", ()))
                valid = set(checked["valid"])
                for symbol in group_symbols:
                    if symbol not in valid:
                        item_plan = plan["ticker_plans"][symbol]
                        if item_plan["status"] == "pending":
                            item_plan["status"] = "failed"
                            plan["outcomes"].append(_outcome(symbol, "failed", item_plan["request"],
                                error_type="SymbolResolutionError", error_message=checked.get("errors", {}).get(symbol, "Symbol was not resolved.")))
                accepted = [symbol for symbol in group_symbols if symbol in valid]
                if accepted:
                    group = dict(id=uuid4().hex, start=begin, end=finish, symbols=accepted,
                                 status="planned", job_id=None, estimated_cost=0.0)
                    plan["groups"].append(group)
        # A symbol unresolved in one gap cannot publish a partial request. Remove
        # it from *all* groups before costing, rather than billing an unusable half.
        for group in plan["groups"]:
            group["symbols"] = [s for s in group["symbols"] if plan["ticker_plans"][s]["status"] == "pending"]
        plan["groups"] = [group for group in plan["groups"] if group["symbols"]]
        for group in plan["groups"]:
            cost = float(provider.estimate_cost(group["symbols"], _parse_stamp(group["start"]),
                                               _parse_stamp(group["end"]), prototype.timeframe))
            if not math.isfinite(cost) or cost < 0:
                raise ProviderError("Databento returned an invalid cost estimate.")
            group["estimated_cost"] = cost
            for symbol in group["symbols"]:
                plan["ticker_plans"][symbol]["groups"].append(group["id"])
        plan["estimated_cost"] = math.fsum(group["estimated_cost"] for group in plan["groups"])
        try:
            conditions = provider.get_dataset_conditions(min(_parse_stamp(g["start"]) for g in plan["groups"]),
                                                        max(_parse_stamp(g["end"]) for g in plan["groups"])) if plan["groups"] else []
            for item in conditions:
                if str(item.get("condition", "")).lower() != "available":
                    plan["warnings"].append(f"Dataset condition: {json.dumps(item, default=str, sort_keys=True)}")
        except Exception as exc:
            plan["warnings"].append(f"Dataset conditions could not be checked: {exc}")
    if max_estimated_cost is not None and plan["estimated_cost"] > max_estimated_cost:
        plan["warnings"].append("Estimated cost exceeds the configured maximum; this plan cannot be submitted.")
    plan["warnings"] = list(dict.fromkeys(plan["warnings"]))
    if _snapshot(pipeline, plan["ticker_plans"]) != plan["snapshot"]:
        raise InvalidDataRequestError("Local data changed during estimation. Estimate again before downloading.")
    if not plan["groups"]:
        _update_status(plan)
    # Two browser sessions may finish the same estimate concurrently. Deduping
    # and publication share one lock, not just each plan's later submit lock.
    with store_lock(_root(pipeline) / "planning", timeout=2):
        for previous in list_plans(pipeline):
            if _same_active_plan(previous, plan):
                return previous
        _save(pipeline, plan)
        return plan


def _same_active_plan(previous, proposed):
    return previous["status"] not in _TERMINAL and (
        previous.get("fingerprint") == proposed["fingerprint"]
        or (previous.get("authorized") and previous.get("request_fingerprint") == proposed["request_fingerprint"]))


def _parse_stamp(stamp):
    if isinstance(stamp, datetime):
        return stamp.astimezone(UTC)
    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(UTC)


def _group_dir(pipeline, plan, group):
    return contained_path(_root(pipeline), f"{plan['id']}/{group['id']}")


def _update_status(plan):
    if all(item["status"] != "pending" and item["status"] != "publishing" for item in plan["ticker_plans"].values()):
        plan["status"] = "completed_with_errors" if any(item["status"] in {"failed", "empty"} for item in plan["outcomes"]) else "completed"
    elif any(group["status"] in {"submitting", "needs_reconciliation"} for group in plan["groups"]):
        plan["status"] = "needs_reconciliation"
    else:
        plan["status"] = "running" if plan["authorized"] else "estimated"


def advance(pipeline, plan_id, *, submit=False, progress=None):
    """Perform one resumable stage; no polling loops or sleeps.

    A saved authorization covers every group in the reviewed estimate. Later
    calls can submit the remaining groups without requesting another approval.
    """
    # Independent from the catalog lock: the latter is acquired only for short
    # per-ticker transactions. The job lock prevents concurrent double submits.
    _path(pipeline, plan_id)
    with store_lock(_root(pipeline) / "locks" / plan_id, timeout=0.1):
        plan = load_plan(pipeline, plan_id)
        if plan["status"] in _TERMINAL:
            return plan
        if not plan["authorized"]:
            if not submit:
                return plan
            cap = plan["max_estimated_cost"]
            if cap is not None and plan["estimated_cost"] > cap:
                raise InvalidDataRequestError("Estimated cost exceeds the maximum. Create a new estimate with a suitable budget.")
            if _snapshot(pipeline, plan["ticker_plans"]) != plan["snapshot"]:
                plan["status"] = "stale"
                _save(pipeline, plan)
                raise InvalidDataRequestError("Local coverage changed after estimation. Estimate again before submitting.")
            plan["authorized"] = True
            plan["authorized_at"] = datetime.now(UTC).isoformat()
            _save(pipeline, plan)
        provider = _provider(pipeline, plan["dataset"])
        for group in plan["groups"]:
            status = group["status"]
            if status == "submitting":
                group["status"] = "needs_reconciliation"
                group["error"] = "Submission was interrupted. Reconcile its job ID before continuing; it will not be resubmitted."
                _update_status(plan)
                _save(pipeline, plan)
                return plan
            if status == "planned":
                if _snapshot(pipeline, plan["ticker_plans"]) != plan["snapshot"]:
                    group.update(status="failed", error="Local coverage changed after estimation; this job was not submitted. Estimate again for remaining data.")
                    _save(pipeline, plan)
                    return plan
                group["status"] = "submitting"
                group["submission_started_at"] = datetime.now(UTC).isoformat()
                _save(pipeline, plan)
                try:
                    job = provider.submit_job(group["symbols"], _parse_stamp(group["start"]),
                                              _parse_stamp(group["end"]), plan["timeframe"])
                    identifier = job.get("id") or job.get("job_id")
                    if not identifier:
                        raise ProviderError("Submit response did not contain a job ID.")
                    group.update(job_id=str(identifier), status="submitted")
                except Exception as exc:
                    group.update(status="needs_reconciliation", error=f"Submission outcome is uncertain: {exc}")
                _update_status(plan)
                _save(pipeline, plan)
                return plan
            if status == "submitted":
                try:
                    job = provider.job_status(group["job_id"])
                    remote = str(job.get("state", job.get("status", "unknown"))).lower()
                    group["remote_status"] = remote
                    if job.get("cost_usd") is not None:
                        cost = float(job["cost_usd"])
                        if math.isfinite(cost) and cost >= 0:
                            group["reported_cost_usd"] = cost
                    if isinstance(job.get("progress"), (float, int)) and math.isfinite(job["progress"]):
                        group["progress"] = job["progress"]
                    if remote in {"done", "completed", "ready"}:
                        group["status"] = "ready"
                    elif remote in {"expired", "failed", "canceled", "cancelled"}:
                        group.update(status="failed", error=f"Databento job {group['job_id']} is {remote}; no replacement job was submitted.")
                    group.pop("error", None) if group["status"] != "failed" else None
                except Exception as exc:
                    group["error"] = f"Status check failed; retrying will reuse this job: {exc}"
                _update_status(plan)
                _save(pipeline, plan)
                return plan
            if status == "ready":
                retry_at = group.get("next_download_attempt_at")
                if retry_at and datetime.now(UTC) < _parse_stamp(retry_at):
                    return plan
                group.pop("next_download_attempt_at", None)
                try:
                    directory = _group_dir(pipeline, plan, group) / "download"
                    directory.mkdir(parents=True, exist_ok=True)
                    if progress is None:
                        paths = provider.download_job(group["job_id"], directory)
                    else:
                        last_saved = None

                        def report_progress(value):
                            nonlocal last_saved
                            group["download_progress"] = {
                                key: int(value[key]) for key in
                                ("downloaded_bytes", "total_bytes", "completed_files", "total_files")
                            }
                            now = time.monotonic()
                            # Persist useful restart information without turning
                            # every network chunk into a journal/fsync operation.
                            if last_saved is None or now - last_saved >= 1:
                                _save(pipeline, plan)
                                last_saved = now
                            progress(dict(group["download_progress"]))

                        paths = provider.download_job(group["job_id"], directory, progress=report_progress)
                    group["files"] = [str(Path(path).resolve().relative_to(_root(pipeline).resolve())) for path in paths]
                    if not paths:
                        raise ProviderError("Completed job contained no downloadable files.")
                    group["status"] = "downloaded"
                    group.pop("error", None)
                except Exception as exc:
                    group["error"] = f"Download failed; retrying will reuse this job: {exc}"
                    retry_after = getattr(exc, "retry_after", None)
                    if (isinstance(retry_after, (float, int)) and not isinstance(retry_after, bool)
                            and math.isfinite(retry_after) and retry_after > 0):
                        try:
                            retry_at = datetime.now(UTC) + timedelta(seconds=retry_after)
                        except OverflowError:
                            retry_at = datetime.max.replace(tzinfo=UTC)
                        group["next_download_attempt_at"] = retry_at.isoformat()
                _save(pipeline, plan)
                return plan
            if status == "downloaded":
                try:
                    _decode(pipeline, provider, plan, group)
                    group["status"] = "decoded"
                    group.pop("error", None)
                except Exception as exc:
                    group.update(status="failed", error=f"Could not import the downloaded job: {exc}")
                _save(pipeline, plan)
                return plan
        # Publish bounded groups of tickers, flushing each outcome. Avoid adding
        # one dashboard polling interval of overhead per ticker in a large batch.
        groups = {group["id"]: group for group in plan["groups"]}
        deadline, published = time.monotonic() + 1.0, 0
        for symbol, item_plan in plan["ticker_plans"].items():
            if item_plan["status"] not in {"pending", "publishing"}:
                continue
            dependencies = [groups[identifier] for identifier in item_plan["groups"]]
            if any(group["status"] not in {"decoded", "failed"} for group in dependencies):
                continue
            try:
                failed = [group for group in dependencies if group["status"] == "failed"]
                if failed:
                    raise ProviderError("; ".join(group["error"] for group in failed))
                result = _publish(pipeline, plan, symbol, item_plan, dependencies)
                item_plan["status"] = result["status"]
            except Exception as exc:
                item_plan["status"] = "empty" if isinstance(exc, EmptyDataError) else "failed"
                result = _outcome(symbol, item_plan["status"], item_plan["request"], error_type=type(exc).__name__, error_message=str(exc))
            plan["outcomes"].append(result)
            _update_status(plan)
            _save(pipeline, plan)
            published += 1
            if published >= 20 or time.monotonic() >= deadline:
                return plan
        _update_status(plan)
        _save(pipeline, plan)
        return plan


def _decode(pipeline, provider, plan, group):
    directory = _group_dir(pipeline, plan, group)
    temporary = directory / "decode.partial"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    manifest = {}
    files = [contained_path(_root(pipeline), path) for path in group["files"]]
    try:
        for frame in provider.iter_frames(files, plan["timeframe"]):
            for key, partition in frame.partition_by("symbol", as_dict=True).items():
                symbol = key[0]
                if symbol not in group["symbols"]:
                    raise ProviderError(f"Downloaded data contains unrequested symbol {symbol!r}.")
                # Each append is a self-contained Arrow stream, so no writer or
                # ticker history needs to stay in memory. This avoids hundreds
                # of thousands of tiny files for 500 tickers over several years.
                name = sha256(symbol.encode()).hexdigest() + ".arrow"
                table = partition.to_arrow()
                with (temporary / name).open("ab") as handle:
                    with pa.ipc.new_stream(handle, table.schema,
                            options=pa.ipc.IpcWriteOptions(compression="zstd")) as writer:
                        writer.write_table(table)
                manifest[symbol] = [name]
        target = directory / "decoded"
        shutil.rmtree(target, ignore_errors=True)
        temporary.rename(target)
        group["parts"] = {symbol: [str((target / name).relative_to(_root(pipeline))) for name in names]
                          for symbol, names in manifest.items()}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _read_parts(pipeline, paths):
    frames = []
    for value in paths:
        path = contained_path(_root(pipeline), value)
        if path.suffix == ".parquet":  # Existing interrupted import journals.
            frames.append(pl.read_parquet(path))
            continue
        size = path.stat().st_size
        with path.open("rb") as handle:
            while handle.tell() < size:
                with pa.ipc.open_stream(handle) as reader:
                    frames.append(pl.from_arrow(reader.read_all()))
    return pl.concat(frames).sort("symbol", "timestamp")


def _publish(pipeline, plan, symbol, item_plan, groups):
    chunks, ranges = [], []
    for group in groups:
        paths = group.get("parts", {}).get(symbol, [])
        gap = _request({**item_plan["request"], "start": group["start"], "end": group["end"]})
        record = dict(start=group["start"], end=group["end"], status="empty", row_count=0)
        if paths:
            frame = validate_ohlcv(_read_parts(pipeline, paths))
            if not frame.is_empty():
                if frame.filter((pl.col("timestamp") < gap.start) | (pl.col("timestamp") >= gap.end)).height:
                    raise ProviderError("Downloaded bars lie outside the planned request range.")
                chunks.append((gap, FetchResult(frame)))
                record.update(status="fetched", row_count=frame.height)
        ranges.append(record)
    if not chunks:
        raise EmptyDataError("No bars were returned for this ticker; existing history was preserved.")
    supersedes = item_plan["supersedes"]
    target = _request(item_plan["merge_request"]) if plan["overlap_policy"] == "fetch_missing" else chunks[0][0]
    expected = pl.concat(([pipeline.read_dataset(identifier) for identifier in supersedes]
                          if plan["overlap_policy"] == "fetch_missing" else []) + [result.frame for _, result in chunks]).sort("symbol", "timestamp")
    recovered = None
    if item_plan["status"] == "publishing":
        # A process can stop after the catalog commit but before its journal save.
        # Recover that publication by exact content, without another write/job.
        for candidate in _existing(pipeline, target):
            if candidate.dataset_id not in plan["snapshot"][symbol] and candidate.request == target:
                if pipeline.read_dataset(candidate.dataset_id).equals(expected):
                    recovered = candidate
                    break
    item_plan["status"] = "publishing"
    _save(pipeline, plan)
    if recovered is not None:
        stored = recovered
    elif plan["overlap_policy"] == "fetch_missing":
        stored = pipeline.store.merge_raw(target, chunks, supersedes=supersedes)
    elif supersedes:
        stored = pipeline.store.refresh_raw(target, chunks[0][1].frame, supersedes=supersedes)
    else:
        stored = pipeline.store.write_raw(target, chunks[0][1].frame)
    status = "saved" if not supersedes else "merged" if plan["overlap_policy"] == "fetch_missing" else "refreshed"
    warnings = ["Some requested ranges returned no bars; only returned and existing bars were saved."] if any(row["status"] == "empty" for row in ranges) else []
    return _outcome(symbol, status, item_plan["request"], dataset_ids=[stored.dataset_id], row_count=stored.row_count,
        quality=json.loads(stored.to_json())["quality"], first_timestamp=stored.first_timestamp.isoformat(),
        last_timestamp=stored.last_timestamp.isoformat(), replaced_dataset_ids=supersedes,
        fetch_start=min(row["start"] for row in ranges), fetch_end=max(row["end"] for row in ranges),
        fetch_ranges=ranges, fetched_row_count=sum(row["row_count"] for row in ranges),
        warning_message=" ".join(warnings) or None)


def reconcile_job(pipeline, plan_id, group_id, job_id):
    """Attach an explicitly identified remote job after an uncertain submission.

    The provider response must match the reviewed dataset, schema, symbol set,
    and time range. This operation never creates a remote job.
    """
    _path(pipeline, plan_id)
    with store_lock(_root(pipeline) / "locks" / plan_id, timeout=0.1):
        plan = load_plan(pipeline, plan_id)
        group = next((item for item in plan["groups"] if item["id"] == group_id), None)
        if group is None or group["status"] not in {"submitting", "needs_reconciliation"}:
            raise InvalidDataRequestError("Only an uncertain submission can be reconciled.")
        job = _provider(pipeline, plan["dataset"]).job_status(job_id)
        expected_schema = {"1m": "ohlcv-1m", "1h": "ohlcv-1h"}[plan["timeframe"]]
        if (job.get("dataset") != plan["dataset"] or job.get("schema") != expected_schema
                or set(job.get("symbols", [])) != set(group["symbols"])
                or _parse_stamp(job.get("start")) != _parse_stamp(group["start"])
                or _parse_stamp(job.get("end")) != _parse_stamp(group["end"])):
            raise InvalidDataRequestError("Remote job does not match the reviewed request; it was not attached.")
        group.update(job_id=str(job_id), status="submitted")
        group.pop("error", None)
        _update_status(plan)
        _save(pipeline, plan)
        return plan
