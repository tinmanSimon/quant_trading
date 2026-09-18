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
                   provider="yahoo", skip_missing_ohlc: bool | None = None) -> BatchFetchReport:
        """Fetch and save every ticker independently, returning per-ticker outcomes."""
        return fetch_many(self, tickers=tickers, start=start, end=end, timeframe=timeframe,
                          provider=provider, skip_missing_ohlc=skip_missing_ohlc)

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
        if not pipeline.identities:
            if len(parents) != 1:
                raise RequestDataMismatchError("An empty processor pipeline requires a single raw input.")
            return parents[0]
        output = pipeline.run(frame, parents[0].contract)
        return self.store.write_processed([item.dataset_id for item in parents], output.frame,
                                          pipeline_id=pipeline.fingerprint,
                                          processors_json=pipeline.canonical_json,
                                          output_contract=output.contract)

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
