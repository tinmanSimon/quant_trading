"""Provider contract and the compatibility adapter for legacy callers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

import polars as pl

from ..exceptions import InvalidDataRequestError
from ..models import DataRequest
from ..quality import FetchResult


class BaseDataProvider(ABC):
    """Fetch one request as validated, sorted canonical OHLCV data.

    Implementations must return only bars in ``[request.start, request.end)``
    and raise a domain error for empty, invalid, or unsupported results.
    ``name`` identifies the provider used by the legacy request adapter.
    """

    name: str = "base"

    @abstractmethod
    def fetch(self, request: DataRequest) -> pl.DataFrame:
        """Fetch a provider-neutral request without changing its semantics."""
        raise NotImplementedError

    def fetch_result(self, request: DataRequest) -> FetchResult:
        """Fetch with provenance; legacy providers explicitly report unknown quality."""
        return FetchResult(self.fetch(request))

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1h",
    ) -> pl.DataFrame:
        """Legacy OHLCV entry point, implemented through :meth:`fetch`.

        ISO date strings and naive ISO datetimes are interpreted as UTC;
        explicit offsets are preserved and normalized by ``DataRequest``.
        As with ``fetch``, the start is inclusive and the end is exclusive.
        New code should pass aware datetimes to ``DataRequest`` directly.
        """
        return self.fetch(
            DataRequest(
                symbol=symbol,
                start=_legacy_datetime(start_date, "start_date"),
                end=_legacy_datetime(end_date, "end_date"),
                timeframe=timeframe,
                provider=self.name,
                dataset="ohlcv",
                price_adjustment="unadjusted",
            )
        )


def _legacy_datetime(value: str, field: str) -> datetime:
    """
    Convert legacy fetch_ohlcv() date strings into timezone-aware datetimes
    for DataRequest, assuming UTC when no timezone is provided.
    """
    if not isinstance(value, str):
        raise InvalidDataRequestError(f"{field} must be an ISO date/datetime string.")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise InvalidDataRequestError(
            f"{field} must be an ISO date/datetime string."
        ) from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
