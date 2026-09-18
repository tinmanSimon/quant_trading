"""Sequential independent fetch attempts with structured outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import polars as pl

from .exceptions import InvalidDataRequestError
from .models import DataRequest
from .providers import ProviderRegistry, YFinanceProvider

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


@dataclass(frozen=True)
class BatchFetchReport:
    outcomes: tuple[FetchOutcome, ...]

    @property
    def ok(self):
        return all(item.status != "failed" for item in self.outcomes)

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([
            {"ticker": item.ticker, "status": item.status, "row_count": item.row_count,
             "dataset_ids": ", ".join(item.dataset_ids), "error_type": item.error_type,
             "error_message": item.error_message,
             "quality": json.dumps(item.quality) if item.quality is not None else None}
            for item in self.outcomes
        ], schema={"ticker": pl.String, "status": pl.String, "row_count": pl.Int64,
                   "dataset_ids": pl.String, "error_type": pl.String, "error_message": pl.String,
                   "quality": pl.String})


def fetch_many(pipeline: DataPipeline, *, tickers, start, end, timeframe="1d",
               provider="yahoo", skip_missing_ohlc: bool | None = None) -> BatchFetchReport:
    """Save each ticker independently, recording ordinary failures and quality.

    ``None`` preserves the pipeline's registered provider configuration. An
    explicit Yahoo skip setting uses a separate provider for this batch only.
    Interrupts and SystemExit propagate instead of becoming ticker failures.
    """
    if isinstance(tickers, (str, bytes)):
        raise InvalidDataRequestError("Supply a list of ticker symbols.")
    tickers = tuple(tickers)
    if not tickers:
        raise InvalidDataRequestError("Supply at least one ticker.")
    if skip_missing_ohlc is not None:
        if provider.strip().lower() != "yahoo":
            raise InvalidDataRequestError("skip_missing_ohlc is a Yahoo-specific setting.")
        # Import at call time: the public DataPipeline API delegates here.
        from .api import DataPipeline

        pipeline = DataPipeline(pipeline.store.data_dir, providers=ProviderRegistry({
            "yahoo": YFinanceProvider(skip_missing_ohlc=skip_missing_ohlc),
        }))
    outcomes = []
    for ticker in tickers:
        symbol = str(ticker)
        try:
            request = DataRequest(symbol=ticker, start=start, end=end, timeframe=timeframe, provider=provider)
            symbol = request.symbol
            result = pipeline.ingest(request).raw
            metadata = json.loads(result.to_json())
            outcomes.append(FetchOutcome(symbol, "saved", (result.dataset_id,), result.row_count,
                                         quality=metadata.get("quality")))
        except Exception as exc:
            # Deliberate interrupt/SystemExit still stop the batch.
            outcomes.append(FetchOutcome(symbol, "failed", error_type=type(exc).__name__, error_message=str(exc)))
    return BatchFetchReport(tuple(outcomes))
