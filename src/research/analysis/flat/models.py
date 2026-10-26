"""Immutable settings and results for causal flat-period analysis."""

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
import math
from typing import Any, Mapping

import polars as pl


ALGORITHM_VERSION = "2.2.0"
Timestamp = datetime | date


@dataclass(frozen=True)
class FlatConfig:
    """Observed-bar duration bounds and total wick-envelope width in percent."""

    min_duration: int = 20
    max_duration: int = 200
    max_range_pct: float = 2.0
    band_width_mode: str = "fixed"
    auto_percentile: float = 90.0
    auto_multiplier: float = 2.0
    long_bar_filter: bool = False
    long_bar_multiplier: float = 2.0

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
        if self.band_width_mode not in ("fixed", "automatic"):
            raise ValueError("band_width_mode must be 'fixed' or 'automatic'")
        for name in ("auto_percentile", "auto_multiplier", "long_bar_multiplier"):
            value = getattr(self, name)
            try:
                finite = math.isfinite(value)
            except (TypeError, OverflowError):
                finite = False
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite:
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.auto_percentile <= 100:
            raise ValueError("auto_percentile must be between 0 and 100 inclusive")
        if self.auto_multiplier <= 0:
            raise ValueError("auto_multiplier must be greater than 0")
        if type(self.long_bar_filter) is not bool:
            raise ValueError("long_bar_filter must be a boolean")
        if self.long_bar_multiplier < 1:
            raise ValueError("long_bar_multiplier must be at least 1")

    def to_dict(self) -> dict[str, int | float | str | bool]:
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
    max_range_pct: float
    max_bar_range_pct: float | None
    median_bar_range_pct: float | None


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
            "max_range_pct": pl.Float64, "max_bar_range_pct": pl.Float64,
            "median_bar_range_pct": pl.Float64,
        }
        def timestamp(value: Timestamp) -> datetime:
            return value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time())

        rows = [{
            "period_id": period.period_id, "start_index": period.start_index,
            "end_index": period.end_index, "start": timestamp(period.start_time), "end": timestamp(period.end_time),
            "bars": period.duration, "lower_price": period.lower_price,
            "upper_price": period.upper_price, "center_price": period.center_price,
            "range_pct": period.range_pct,
            "max_range_pct": period.max_range_pct,
            "max_bar_range_pct": period.max_bar_range_pct,
            "median_bar_range_pct": period.median_bar_range_pct,
        } for period in self.periods]
        return pl.DataFrame(rows, schema=schema)
