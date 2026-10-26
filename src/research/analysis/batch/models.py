"""Validated, serializable settings for a multi-ticker analysis run."""

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from data_pipeline import DataQuery
from data_pipeline.exceptions import InvalidDataRequestError


def default_strict_validation(provider: str) -> bool:
    """Databento research defaults to strict coverage; Yahoo screening does not."""
    return provider == "databento" or provider.startswith("databento-")


@dataclass(frozen=True)
class BatchRequest:
    tickers: tuple[str, ...]
    provider: str = "yahoo"
    timeframe: str = "1d"
    start: datetime | None = None
    end: datetime | None = None
    layer: str = "raw"
    pipeline_id: str | None = None
    workers: int = 1
    session_timezone: str = "America/New_York"
    as_of: datetime | None = None
    strict_validation: bool | None = None
    calendar: str = "XNYS"

    def __post_init__(self):
        if isinstance(self.tickers, (str, bytes)):
            raise ValueError("Supply a sequence of ticker symbols.")
        try:
            query = DataQuery(layer=self.layer, provider=self.provider, timeframe=self.timeframe,
                              start=self.start, end=self.end, pipeline_id=self.pipeline_id)
            symbols = tuple(DataQuery(provider=query.provider, symbol=value).symbol for value in self.tickers)
        except InvalidDataRequestError as error:
            raise ValueError(str(error)) from error
        if query.provider is None or query.timeframe is None:
            raise ValueError("Select a provider and timeframe.")
        if self.layer == "processed" and query.pipeline_id is None:
            raise ValueError("Select one processed pipeline for the batch.")
        if self.layer == "raw" and query.pipeline_id is not None:
            raise ValueError("Raw analysis must not specify a processed pipeline.")
        if not symbols or None in symbols or len(set(symbols)) != len(symbols):
            raise ValueError("Supply at least one distinct ticker symbol.")
        if type(self.workers) is not int or not 1 <= self.workers <= 32:
            raise ValueError("workers must be an integer between 1 and 32.")
        if self.strict_validation is not None and type(self.strict_validation) is not bool:
            raise ValueError("strict_validation must be a boolean or None.")
        object.__setattr__(self, "strict_validation", default_strict_validation(query.provider)
                           if self.strict_validation is None else self.strict_validation)
        if not isinstance(self.calendar, str) or not self.calendar.strip():
            raise ValueError("calendar must be a nonempty string.")
        object.__setattr__(self, "calendar", self.calendar.strip())
        ZoneInfo(self.session_timezone)
        if self.as_of is not None:
            if (not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None
                    or self.as_of.utcoffset() is None):
                raise ValueError("as_of must be a timezone-aware datetime.")
            object.__setattr__(self, "as_of", self.as_of.astimezone(UTC))
            if self.as_of > datetime.now(UTC):
                raise ValueError("as_of cannot be in the future.")
        for name in ("provider", "timeframe", "start", "end", "pipeline_id"):
            object.__setattr__(self, name, getattr(query, name))
        object.__setattr__(self, "tickers", symbols)

    def to_dict(self):
        return {"tickers": list(self.tickers), "provider": self.provider,
                "timeframe": self.timeframe, "start": self.start.isoformat() if self.start else None,
                "end": self.end.isoformat() if self.end else None, "layer": self.layer,
                "pipeline_id": self.pipeline_id, "workers": self.workers,
                "session_timezone": self.session_timezone,
                "as_of": self.as_of.isoformat() if self.as_of else None,
                "strict_validation": self.strict_validation, "calendar": self.calendar}

    @classmethod
    def from_dict(cls, value):
        settings = dict(value)
        # Old saved runs predate coverage validation; do not relabel them as
        # having passed a strict check merely because they use Databento.
        settings.setdefault("strict_validation", False)
        for name in ("start", "end", "as_of"):
            if settings.get(name) is not None:
                settings[name] = datetime.fromisoformat(settings[name])
        return cls(**settings)


TICKER_STATUSES = ("pending", "running", "completed", "partial", "failed", "no_data", "cancelled")


def status_counts(tickers):
    counts = {name: 0 for name in TICKER_STATUSES}
    for entry in tickers.values():
        counts[entry["status"]] += 1
    return {"requested": len(tickers), **counts}
