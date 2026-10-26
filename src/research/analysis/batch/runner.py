"""Bounded multi-ticker orchestration using the ordinary analysis engine."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import copy, deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import inspect
import json
from pathlib import Path

import polars as pl

from data_pipeline import DataQuery

from ..core.base import LogicContext, LogicSpec, json_copy
from ..core.runner import LogicRunner
from ..core.omissions import omission_metadata, omission_report
from .models import BatchRequest, status_counts


def _now():
    return datetime.now(UTC).isoformat()


def _cancelled(cancel):
    return bool(cancel and (cancel.is_set() if hasattr(cancel, "is_set") else cancel()))


def _engine_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _text_result(result, integrity=None):
    """Keep the existing text/table contract; omit chart and navigation archives."""
    data = {}
    if "metadata" in result.data:
        data["metadata"] = result.data["metadata"]
    if "implementation_fingerprint" in result.data:
        data["implementation_fingerprint"] = result.data["implementation_fingerprint"]
    if "analysis" in result.data:
        data["analysis"] = {"text": result.data["analysis"].get("text", [])}
    messages = list(result.messages)
    if integrity is not None:
        data["data_integrity"] = integrity
        messages.extend(message for message in integrity.get("messages", []) if message not in messages)
    return json_copy({"overlays": [], "tables": result.tables,
                      "messages": messages, "data": data})


class BatchRunner:
    """Freeze a run with ``create``, then execute it with ``run``.

    ``create`` returns immediately after catalog discovery/configuration; it
    never loads every ticker into memory. ``run`` may be called in a background
    thread. Only its worker pool runs analysis functions, and no dashboard or
    Streamlit objects are involved. Cancellation stops further submissions;
    already running ticker calculations finish and their results are saved.
    """

    def __init__(self, pipeline, registry, store):
        self.pipeline = pipeline
        self.store = store
        # Retain the loaded function/class identities when a caller subsequently
        # reloads private scripts or changes a registry's registrations.
        self.registry = copy(registry)
        self.registry._classes = dict(registry._classes)
        self.registry._definitions = deepcopy(registry._definitions)
        self.registry._issues = list(registry._issues)
        data_root = Path(pipeline.store.data_dir).resolve()
        result_root = store.directory
        if data_root == result_root or data_root in result_root.parents or result_root in data_root.parents:
            raise ValueError("Batch results and market data require non-overlapping directories.")

    def _implementations(self, specs):
        result = {}
        for spec in specs:
            definition = self.registry.definition(spec.name, spec.version)
            logic = self.registry.create(spec.name, spec.version)
            fingerprint = definition.implementation_fingerprint
            if not fingerprint:
                try:
                    fingerprint = sha256(inspect.getsource(type(logic)).encode()).hexdigest()
                except (OSError, TypeError):
                    fingerprint = ""
            result[spec.instance_id] = {"name": spec.name, "version": spec.version,
                "label": definition.label, "result_version": definition.result_version,
                "implementation_fingerprint": fingerprint}
        return result

    def create(self, request, specs):
        if not isinstance(request, BatchRequest):
            raise TypeError("request must be a BatchRequest.")
        request = replace(request, as_of=request.as_of or datetime.now(UTC))
        specs = LogicRunner(self.registry, cache_limit=0).validate(specs)
        if not specs:
            raise ValueError("Select at least one analysis.")
        # A single catalog lookup fixes every source revision before the first
        # analysis executes. Earlier revisions supply initialization history.
        query = DataQuery(layer=request.layer, provider=request.provider,
                          timeframe=request.timeframe, pipeline_id=request.pipeline_id, end=request.end)
        sources = {ticker: [] for ticker in request.tickers}
        for item in self.pipeline.list_datasets(query):
            if item.request.symbol in sources:
                sources[item.request.symbol].append(json.loads(item.to_json()))
        for values in sources.values():
            values.sort(key=lambda item: item["dataset_id"])
        tickers = {ticker: {"status": "pending", "error": None,
                            "analysis_status": {}, "summaries": {}} for ticker in request.tickers}
        manifest = {"status": "pending", "request": request.to_dict(),
                    "specs": [spec.to_dict() for spec in specs],
                    "implementations": self._implementations(specs), "engine_sha256": _engine_fingerprint(),
                    "data_directory": str(self.pipeline.store.data_dir), "sources": sources,
                    "tickers": tickers, "counts": status_counts(tickers)}
        return self.store.create(manifest)

    def _ticker(self, ticker, request, specs, sources):
        payload = {"ticker": ticker, "status": "failed", "error": None, "started_at": _now(),
                   "source_key": None, "keys": {}, "errors": {}, "results": {}}
        try:
            if not sources:
                payload.update(status="no_data", error="No matching local data revisions.")
                return payload
            frames = []
            stamp = pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            for item in sources:
                frame = self.pipeline.read_dataset(item["dataset_id"])
                if request.end is not None:
                    frame = frame.filter(stamp < pl.lit(request.end, dtype=pl.Datetime("us", "UTC")))
                frames.append(frame)
            history = pl.concat(frames).sort("timestamp")
            if history["timestamp"].n_unique() != history.height:
                raise ValueError("Selected data revisions contain overlapping bar timestamps.")
            if history.height and history["symbol"].unique().to_list() != [ticker]:
                raise ValueError("Stored data contains a different ticker.")
            support = None
            selected = history
            if request.start is not None:
                boundary = pl.lit(request.start, dtype=pl.Datetime("us", "UTC"))
                support = history.filter(stamp < boundary)
                selected = history.filter(stamp >= boundary)
            if selected.is_empty():
                payload.update(status="no_data", error="No saved bars fall inside the selected range.")
                return payload
            cutoff = request.as_of
            if request.end is not None and not request.timeframe.endswith(("d", "wk", "mo")):
                cutoff = min(cutoff, request.end)
            identity = (str(self.pipeline.store.data_dir), ticker, request.provider,
                        request.timeframe, request.layer, request.pipeline_id,
                        request.start, request.end, tuple(item["dataset_id"] for item in sources))
            context = LogicContext(selected, request.timeframe, identity, cutoff,
                                   session_timezone=request.session_timezone, support_frame=support,
                                   metadata=omission_metadata(sources, pipeline=self.pipeline))
            integrity = omission_report(context, start=request.start, end=request.end)
            if support is not None and not support.is_empty():
                initialization = omission_report(replace(context, frame=support, support_frame=None,
                    metadata={"omitted_bar_timestamps": context.metadata["omitted_bar_timestamps"]}),
                    end=request.start)
                integrity["initialization_omissions"] = initialization
                integrity["messages"].extend(message.replace("selected data", "initialization history")
                                              for message in initialization["messages"])
            integrity["validation"] = {"status": "not_checked"}
            payload["data_integrity"] = integrity
            if request.strict_validation:
                from .validation import validate_batch_data
                integrity["validation"] = {"status": "failed"}
                integrity["validation"] = validate_batch_data(self.pipeline, request, sources, context)
                # The strict gate checked every supplied input and its pinned
                # source support. Historical omissions outside that scope must
                # not block flat analysis or imply these inputs remain broken.
                integrity["messages"] = []
                context = replace(context, metadata={**context.metadata,
                    "processed_omission_sources": [], "strict_validation": integrity["validation"]})
            applied = LogicRunner(self.registry, cache_limit=0).apply(specs, context)
            payload.update(source_key=applied.source_key, keys=applied.keys, errors=applied.errors,
                           results={identifier: _text_result(result, integrity)
                                    for identifier, result in applied.results.items()})
            payload["status"] = ("partial" if applied.results else "failed") if applied.errors else "completed"
            if applied.errors:
                payload["error"] = "; ".join(f"{name}: {message}" for name, message in applied.errors.items())
        except Exception as error:
            payload["error"] = str(error)
        finally:
            payload["finished_at"] = _now()
        return payload

    def run(self, run_id, *, progress=None, cancel=None):
        from .aggregation import extract_metrics, infer_metrics

        manifest = self.store.load_manifest(run_id)
        if manifest["status"] != "pending":
            raise ValueError("Only a newly prepared batch can be executed.")
        request = BatchRequest.from_dict(manifest["request"])
        specs = [LogicSpec.from_dict(item) for item in manifest["specs"]]
        if self._implementations(specs) != manifest["implementations"]:
            raise ValueError("Analysis implementations changed since this batch was prepared.")
        if _engine_fingerprint() != manifest["engine_sha256"]:
            raise ValueError("Analysis engine files changed since this batch was prepared; prepare a new run.")
        if str(self.pipeline.store.data_dir) != manifest["data_directory"]:
            raise ValueError("This batch was prepared against a different data directory.")
        self.store.claim_run(run_id)
        manifest.update(status="running", started_at=_now())

        def publish():
            manifest["counts"] = status_counts(manifest["tickers"])
            manifest.update(self.store.save_manifest(manifest))
            if progress is not None:
                progress(deepcopy(manifest))

        remaining = iter(request.tickers)
        futures = {}
        stopped = False
        try:
            publish()
            with ThreadPoolExecutor(max_workers=request.workers, thread_name_prefix="analysis-batch") as pool:
                def submit():
                    if _cancelled(cancel):
                        return False
                    ticker = next(remaining, None)
                    if ticker is None:
                        return False
                    manifest["tickers"][ticker]["status"] = "running"
                    futures[pool.submit(self._ticker, ticker, request, specs, manifest["sources"][ticker])] = ticker
                    return True

                for _ in range(request.workers):
                    if not submit():
                        break
                publish()
                while futures:
                    done, _ = wait(futures, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        ticker = futures.pop(future)
                        payload = future.result()
                        summaries, summary_errors = {}, {}
                        for identifier, result in payload["results"].items():
                            try:
                                metrics = infer_metrics(result["tables"])
                                summaries[identifier] = {"metrics": [item.to_dict() for item in metrics],
                                                         "values": extract_metrics(result["tables"], metrics)}
                            except (TypeError, ValueError) as error:
                                # Keep unusual/custom tables accessible even when
                                # automatic metric selection cannot interpret them.
                                summary_errors[identifier] = str(error)
                        entry = manifest["tickers"][ticker]
                        entry.update(status=payload["status"], error=payload["error"], summaries=summaries,
                                     summary_errors=summary_errors,
                                     data_warnings=payload.get("data_integrity", {}).get("messages", []),
                                     data_validation=payload.get("data_integrity", {}).get("validation", {}),
                                     analysis_status={spec.instance_id: ("failed" if spec.instance_id in payload["errors"]
                                         else "completed" if spec.instance_id in payload["results"] else payload["status"])
                                         for spec in specs},
                                     **self.store.save_ticker(run_id, ticker, payload))
                        publish()
                        submit()
                    if done:
                        publish()
                    if _cancelled(cancel):
                        stopped = True
                stopped = stopped or _cancelled(cancel)
            for entry in manifest["tickers"].values():
                if entry["status"] == "pending":
                    entry["status"] = "cancelled"
            manifest.update(status="cancelled" if stopped else "completed", finished_at=_now())
            publish()
        except Exception as error:
            manifest.update(status="failed", error=str(error), finished_at=_now())
            for entry in manifest["tickers"].values():
                if entry["status"] in ("pending", "running"):
                    entry.update(status="cancelled", error="Batch execution stopped before this result was saved.")
            self.store.save_manifest({**manifest, "counts": status_counts(manifest["tickers"])})
            raise
        return self.store.load_manifest(run_id)

    def execute(self, request, specs, *, progress=None, cancel=None):
        """Convenience entry point for synchronous research scripts."""
        manifest = self.create(request, specs)
        return self.run(manifest["run_id"], progress=progress, cancel=cancel)
