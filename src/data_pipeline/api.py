"""Public orchestration of providers, raw storage, processors and local queries."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable, Sequence

import polars as pl

from .batch_fetch import BatchFetchReport, fetch_many
from .exceptions import ProcessingError
from .models import DataQuery, DataRequest
from .processing import Pipeline, Processor
from .providers import ProviderRegistry, registry as default_providers
from .storage import DeletionPlan, DeletionReport, LocalDataStore, StoredDataset


@dataclass(frozen=True)
class IngestionResult:
    raw: StoredDataset
    processed: StoredDataset | None = None


class DataPipeline:
    """Fetch, save, process and query canonical OHLCV using one local data root."""

    def __init__(self, data_dir: str | Path = "data", *, providers: ProviderRegistry | None = None):
        self.store = LocalDataStore(data_dir)
        self.providers = default_providers if providers is None else providers

    def fetch_many(self, tickers, *, start: datetime, end: datetime, timeframe="1d",
                   provider="yahoo", skip_missing_ohlc: bool | None = None,
                   massive_request_interval_seconds: float | None = None,
                   overlap_policy: str = "error", progress=None) -> BatchFetchReport:
        """Fetch tickers independently with strict, union-refresh, or missing-range merge policy."""
        return fetch_many(self, tickers=tickers, start=start, end=end, timeframe=timeframe,
                          provider=provider, skip_missing_ohlc=skip_missing_ohlc,
                          massive_request_interval_seconds=massive_request_interval_seconds,
                          overlap_policy=overlap_policy, progress=progress)

    def estimate_databento(self, tickers, *, start: datetime, end: datetime,
                           dataset: str, timeframe="1m", overlap_policy="fetch_missing",
                           max_estimated_cost=None) -> dict:
        """Estimate and persist an explicit multi-symbol download; never submit it."""
        from .databento_batch import estimate

        return estimate(self, tickers, start=start, end=end, dataset=dataset,
                        timeframe=timeframe, overlap_policy=overlap_policy,
                        max_estimated_cost=max_estimated_cost)

    def advance_databento(self, plan_id: str, *, submit=False, progress=None) -> dict:
        """Advance one saved job stage, optionally reporting download bytes/files.

        ``submit=True`` authorizes the reviewed plan. The progress callback runs
        on the calling thread and never authorizes or creates a replacement job.
        Its dict contains downloaded_bytes, total_bytes, completed_files, and
        total_files for the current job, including previously cached files.
        """
        from .databento_batch import advance

        return advance(self, plan_id, submit=submit, progress=progress)

    def databento_plan(self, plan_id: str) -> dict:
        from .databento_batch import load_plan

        return load_plan(self, plan_id)

    def recover_databento(self, plan_id: str, *, drop_invalid_outside_regular=False, progress=None) -> dict:
        """Recover failed imports from decoded cache without any network requests."""
        from .databento_batch import recover_cached

        return recover_cached(self, plan_id, drop_invalid_outside_regular=drop_invalid_outside_regular,
                              progress=progress)

    def databento_plans(self) -> list[dict]:
        from .databento_batch import list_plans

        return list_plans(self)

    def databento_report(self, plan_id: str) -> BatchFetchReport:
        from .databento_batch import report_from_plan

        return report_from_plan(self.databento_plan(plan_id))

    def reconcile_databento_job(self, plan_id: str, group_id: str, job_id: str) -> dict:
        """Attach a verified existing job after an uncertain submit; never resubmit."""
        from .databento_batch import reconcile_job

        return reconcile_job(self, plan_id, group_id, job_id)

    def ingest(
        self, request: DataRequest, *, processors: Pipeline | Iterable[Processor] = (),
    ) -> IngestionResult:
        pipeline = _pipeline(processors)
        provider = self.providers.get(request.provider)
        fetched = provider.fetch_result(request)
        raw = self.store.write_raw(request, fetched.frame, quality=fetched.quality)
        return self._finish_ingestion(raw, pipeline)

    def ingest_frame(
        self, request: DataRequest, frame: pl.DataFrame, *,
        processors: Pipeline | Iterable[Processor] = (),
    ) -> IngestionResult:
        """Persist supplied raw data through exactly the same validation path."""
        pipeline = _pipeline(processors)
        raw = self.store.write_raw(request, frame)
        return self._finish_ingestion(raw, pipeline)

    def _finish_ingestion(self, raw: StoredDataset, pipeline: Pipeline) -> IngestionResult:
        if not pipeline.identities:
            return IngestionResult(raw)
        try:
            processed = self.process(raw.dataset_id, processors=pipeline)
        except Exception as error:
            raise ProcessingError(
                f"Raw data was saved as {raw.dataset_id}; processing failed. "
                f"Retry process() with this ID. Cause: {error}"
            ) from error
        return IngestionResult(raw, processed)

    def process(
        self, raw_id: str | Sequence[str], *, processors: Pipeline | Iterable[Processor] = (),
    ) -> StoredDataset:
        """Process compatible raw revisions together; record all contributing IDs.

        With no processors, only a single input is accepted and returned unchanged.
        """
        from .exceptions import RequestDataMismatchError

        pipeline = _pipeline(processors)
        parents, frame = self.store.read_raw_inputs(raw_id)
        action_specs = [spec for spec in pipeline.identities if spec["name"] == "adjust_corporate_actions"]
        if action_specs:
            if len(action_specs) > 1:
                raise ProcessingError("Only one corporate-action adjustment step is allowed per pipeline.")
            if parents[0].request.provider == "yahoo":
                raise ProcessingError("Yahoo raw prices may already contain split adjustments. "
                                      "Use a source with verified unadjusted OHLCV.")
            from .corporate_actions.models import ActionSnapshot
            from .corporate_actions.storage import SnapshotStore
            SnapshotStore(self.store.data_dir / "corporate_actions").save(
                ActionSnapshot.from_dict(action_specs[0]["config"]["snapshot"]))
        if not pipeline.identities:
            if len(parents) != 1:
                raise RequestDataMismatchError("An empty processor pipeline requires a single raw input.")
            return parents[0]
        output = pipeline.run(frame, parents[0].contract)
        return self.store.write_processed([item.dataset_id for item in parents], output.frame,
                                          pipeline_id=pipeline.fingerprint,
                                          processors_json=pipeline.canonical_json,
                                          output_contract=output.contract)

    def process_many(self, inputs, *, processors, preview=False):
        """Process ticker-to-raw-ID groups with an all-ticker corporate-action report."""
        from .processing_batch import process_many
        return process_many(self, inputs, processors=processors, preview=preview)

    def refetch(self, raw_id: str, *, confirm: str) -> StoredDataset:
        """Fetch the original request again and explicitly replace its entire raw batch."""
        from .exceptions import ConfirmationRequiredError, RequestDataMismatchError

        if confirm != raw_id:
            raise ConfirmationRequiredError("confirm must equal the exact target dataset ID.")
        old = self.store.get_metadata(raw_id)
        if old.layer != "raw" or not old.active:
            raise RequestDataMismatchError("Replacement requires an active raw dataset ID.")
        fetched = self.providers.get(old.request.provider).fetch_result(old.request)
        return self.store.replace_raw(raw_id, fetched.frame, confirm=confirm, quality=fetched.quality)

    def list_datasets(self, query: DataQuery | None = None) -> list[StoredDataset]:
        return self.store.list_datasets(query)

    def list_coverage(self, query: DataQuery | None = None) -> list[StoredDataset]:
        """Return actual per-batch first/last bars; bounds do not imply gap-free sessions."""
        return self.store.list_datasets(query)

    def get_metadata(self, dataset_id: str) -> StoredDataset:
        return self.store.get_metadata(dataset_id)

    def read_dataset(self, dataset_id: str) -> pl.DataFrame:
        return self.store.read_dataset(dataset_id)

    def read(self, query: DataQuery, *, columns: list[str] | None = None) -> pl.DataFrame:
        return self.store.read(query, columns=columns)

    def scan(self, query: DataQuery, *, columns: list[str] | None = None) -> pl.LazyFrame:
        return self.store.scan(query, columns=columns)

    def plan_delete(self, query: DataQuery) -> DeletionPlan:
        """Preview physical deletion in one layer, including historical revisions."""
        return self.store.plan_delete(query)

    def delete(self, plan: DeletionPlan, *, confirm: str) -> DeletionReport:
        """Physically remove only the confirmed selection; layers are independent."""
        return self.store.delete(plan, confirm=confirm)

    def deletion_status(self, operation_id: str) -> DeletionReport:
        return self.store.deletion_status(operation_id)

    def list_deletions(self, *, pending_only: bool = False) -> list[DeletionReport]:
        return self.store.list_deletions(pending_only=pending_only)

    def recover(self) -> list[str]:
        """Resume pending physical cleanup and quarantine unrelated abandoned writes."""
        return self.store.recover()


def _pipeline(processors: Pipeline | Iterable[Processor]) -> Pipeline:
    return processors if isinstance(processors, Pipeline) else Pipeline(processors)
