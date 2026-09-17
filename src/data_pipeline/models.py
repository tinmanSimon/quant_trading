"""Provider-neutral models used at the public data-pipeline boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re

from .exceptions import InvalidDataRequestError


_TIMEFRAME_PATTERN = re.compile(r"^[1-9][0-9]*(m|h|d|wk|mo)$")
_SUPPORTED_DATASETS = frozenset({"ohlcv"})


@dataclass(frozen=True, slots=True)
class DataRequest:
    """An immutable, provider-neutral request for one market-data dataset.

    ``start`` is inclusive and ``end`` is exclusive. Both bounds must be
    timezone-aware and are normalized to UTC. A timestamp in the canonical
    OHLCV result identifies the bar start for intraday data, or a session-date
    label at midnight UTC for daily and longer bars.
    """

    symbol: str
    start: datetime
    end: datetime
    timeframe: str = "1h"
    provider: str = "yahoo"
    dataset: str = "ohlcv"
    price_adjustment: str = "unadjusted"

    def __post_init__(self) -> None:
        if self.price_adjustment != "unadjusted":
            raise InvalidDataRequestError("Only price_adjustment='unadjusted' is supported.")
        symbol = _require_non_empty_string(self.symbol, "symbol")
        provider = _require_non_empty_string(self.provider, "provider").lower()
        # Yahoo ticker case is not a distinct instrument/storage namespace.
        # Do not impose that vendor convention on case-sensitive providers.
        if provider == "yahoo":
            symbol = symbol.upper()
        dataset = _require_non_empty_string(self.dataset, "dataset").lower()
        timeframe = _require_non_empty_string(self.timeframe, "timeframe").lower()

        if dataset not in _SUPPORTED_DATASETS:
            supported = ", ".join(sorted(_SUPPORTED_DATASETS))
            raise InvalidDataRequestError(
                f"Unsupported dataset {dataset!r}; supported datasets: {supported}."
            )

        if not _TIMEFRAME_PATTERN.fullmatch(timeframe):
            raise InvalidDataRequestError(
                "timeframe must be a positive integer followed by one of "
                "'m', 'h', 'd', 'wk', or 'mo'."
            )

        # These aliases describe the same intraday bars and must share storage identity.
        if timeframe.endswith("m") and int(timeframe[:-1]) % 60 == 0:
            timeframe = f"{int(timeframe[:-1]) // 60}h"

        start = _normalize_utc_datetime(self.start, "start")
        end = _normalize_utc_datetime(self.end, "end")
        if start >= end:
            raise InvalidDataRequestError("start must be earlier than end.")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidDataRequestError(f"{field_name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise InvalidDataRequestError(f"{field_name} must not be empty.")
    return normalized


def _normalize_utc_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise InvalidDataRequestError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidDataRequestError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DataQuery:
    """Filters for local data; range bounds use the same [start, end) convention."""

    layer: str = "raw"
    dataset: str = "ohlcv"
    provider: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    pipeline_id: str | None = None
    include_history: bool = False

    def __post_init__(self) -> None:
        if self.layer not in {"raw", "processed"}:
            raise InvalidDataRequestError("layer must be 'raw' or 'processed'.")
        if self.dataset != "ohlcv":
            raise InvalidDataRequestError("Only dataset='ohlcv' is supported.")
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _normalize_utc_datetime(value, name))
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise InvalidDataRequestError("start must be earlier than end.")
        for name in ("provider", "symbol", "timeframe", "pipeline_id"):
            value = getattr(self, name)
            if value is not None:
                value = _require_non_empty_string(value, name)
                object.__setattr__(self, name, value.lower() if name in {"provider", "timeframe"} else value)
        if self.provider == "yahoo" and self.symbol is not None:
            object.__setattr__(self, "symbol", self.symbol.upper())
        if self.timeframe is not None:
            if not _TIMEFRAME_PATTERN.fullmatch(self.timeframe):
                raise InvalidDataRequestError("Invalid query timeframe.")
            if self.timeframe.endswith("m") and int(self.timeframe[:-1]) % 60 == 0:
                object.__setattr__(self, "timeframe", f"{int(self.timeframe[:-1]) // 60}h")
