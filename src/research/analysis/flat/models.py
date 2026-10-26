"""Immutable settings and results for causal flat-period analysis."""

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
import math
from typing import Any, Mapping

import polars as pl


ALGORITHM_VERSION = "2.0.0"
Timestamp = datetime | date


@dataclass(frozen=True)
class FlatConfig:
    """Observed-bar duration bounds and total wick-envelope width in percent."""

    min_duration: int = 20
    max_duration: int = 200
    max_range_pct: float = 2.0

    def __post_init__(self) -> None:
        if (isinstance(self.min_duration, bool)
                or not isinstance(self.min_duration, int) or self.min_duration < 2):
            raise ValueError("min_duration must be an integer of at least 2")
        if (isinstance(self.max_duration, bool)
                or not isinstance(self.max_duration, int)
                or self.max_duration <= self.min_duration):
            raise ValueError("max_duration must be an integer greater than min_duration")
        value = self.max_range_pct
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 < value < 200 or not math.isfinite(value)):
            raise ValueError("max_range_pct must be a finite percentage greater than 0 and less than 200")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FlatConfig":
        if not isinstance(values, Mapping):
            raise ValueError("Flat configuration must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown flat settings: {', '.join(sorted(map(str, unknown)))}")
        return cls(**dict(values))


@dataclass(frozen=True)
class FlatPeriod:
    period_id: str
    start_index: int
    end_index: int
    start_time: Timestamp
    end_time: Timestamp
    duration: int
    lower_price: float
    upper_price: float
    center_price: float
    range_pct: float


@dataclass(frozen=True)
class FlatAnalysisResult:
    periods: tuple[FlatPeriod, ...]
    config: FlatConfig
    source_count: int
    analysis_start: Timestamp | None
    analysis_end: Timestamp | None
    analysis_mode: str
    status: str
    algorithm_version: str = ALGORITHM_VERSION

    def to_frame(self) -> pl.DataFrame:
        """Stable numeric types; range_pct is a percentage of the band midpoint."""
        timezone = (str(self.analysis_start.tzinfo)
                    if isinstance(self.analysis_start, datetime) and self.analysis_start.tzinfo else None)
        schema = {
            "period_id": pl.String, "start_index": pl.Int64, "end_index": pl.Int64,
            "start": pl.Datetime("us", timezone), "end": pl.Datetime("us", timezone),
            "bars": pl.Int64, "lower_price": pl.Float64, "upper_price": pl.Float64,
            "center_price": pl.Float64, "range_pct": pl.Float64,
        }
        def timestamp(value: Timestamp) -> datetime:
            return value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time())

        rows = [{
            "period_id": period.period_id, "start_index": period.start_index,
            "end_index": period.end_index, "start": timestamp(period.start_time), "end": timestamp(period.end_time),
            "bars": period.duration, "lower_price": period.lower_price,
            "upper_price": period.upper_price, "center_price": period.center_price,
            "range_pct": period.range_pct,
        } for period in self.periods]
        return pl.DataFrame(rows, schema=schema)
