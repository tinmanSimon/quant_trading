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
    OHLCV result identifies the start of its bar.
    """

    symbol: str
    start: datetime
    end: datetime
    timeframe: str = "1h"
    provider: str = "yahoo"
    dataset: str = "ohlcv"

    def __post_init__(self) -> None:
        symbol = _require_non_empty_string(self.symbol, "symbol")
        provider = _require_non_empty_string(self.provider, "provider").lower()
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
