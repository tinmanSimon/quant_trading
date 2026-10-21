"""Public immutable configuration and data-only results for trend analysis."""

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
import math
from typing import Any, Mapping

import polars as pl


ALGORITHM_VERSION = "5.0.0"
Timestamp = datetime | date


@dataclass(frozen=True)
class TrendConfig:
    # Window advance and overlap; each analysis window holds up to twice this.
    max_history: int = 2000
    min_segment_length: int = 8
    penalty_multiplier: float = 1.0
    penalty_grid_step: float = 1.0
    penalty_grid_levels: int = 20
    weight_ols: float = 0.25
    weight_robust: float = 0.20
    weight_ordering: float = 0.35
    weight_efficiency: float = 0.20
    clarity_threshold: float = 0.65
    magnitude_threshold: float = 1.0
    boundary_fraction: float = 0.05
    boundary_score_threshold: float = 0.55
    boundary_acceptance: float = 0.80
    min_tail_length: int = 8
    max_tail_fraction: float = 0.5
    tail_fit_threshold: float = 0.35
    tail_slope_threshold: float = 1.0
    tail_level_threshold: float = 3.0
    slowing_ratio: float = 0.25
    retracement_threshold: float = 0.5
    countertrend_fraction: float = 1 / 3
    shock_threshold: float = 3.0
    redundancy_percentage: float = 5.0
    redundancy_boundary_rule: str = "and"
    redundancy_min_bars: int = 5
    # Retained for compatibility with saved configurations. Redundancy is now
    # selected for display by endpoint distance, without dropping engine results.
    dedup_ratio: float = 1.25

    def __post_init__(self) -> None:
        if (not isinstance(self.redundancy_boundary_rule, str)
                or self.redundancy_boundary_rule not in ("and", "or")):
            raise ValueError("redundancy_boundary_rule must be 'and' or 'or'")
        for field in fields(self):
            if field.name == "redundancy_boundary_rule":
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
        for name in ("max_history", "min_segment_length", "min_tail_length", "redundancy_min_bars", "penalty_grid_levels"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if not 8 <= self.min_segment_length <= self.max_history <= 100_000:
            raise ValueError("Require 8 <= min_segment_length <= max_history <= 100000")
        if not 8 <= self.min_tail_length <= self.max_history:
            raise ValueError("Require 8 <= min_tail_length <= max_history")
        if self.penalty_multiplier <= 0:
            raise ValueError("penalty_multiplier must be positive")
        if not 0.25 <= self.penalty_grid_step <= 4:
            raise ValueError("penalty_grid_step must be between 0.25 and 4")
        if not 1 <= self.penalty_grid_levels <= 256:
            raise ValueError("penalty_grid_levels must be between 1 and 256")
        if any(value < 0 for value in self.weights) or not math.isclose(sum(self.weights), 1, abs_tol=1e-8):
            raise ValueError("Method weights must be nonnegative and sum to one")
        for name in ("clarity_threshold", "boundary_fraction", "boundary_score_threshold",
                     "boundary_acceptance", "tail_fit_threshold", "slowing_ratio",
                     "retracement_threshold", "countertrend_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not 0 < self.max_tail_fraction <= 0.5:
            raise ValueError("max_tail_fraction must be greater than zero and at most 0.5")
        for name in ("magnitude_threshold", "tail_slope_threshold", "tail_level_threshold", "shock_threshold"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dedup_ratio < 1:
            raise ValueError("dedup_ratio must be at least one")
        if not 0 <= self.redundancy_percentage <= 100:
            raise ValueError("redundancy_percentage must be between zero and 100")
        if self.redundancy_min_bars < 0:
            raise ValueError("redundancy_min_bars must be nonnegative")

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return self.weight_ols, self.weight_robust, self.weight_ordering, self.weight_efficiency

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrendConfig":
        if not isinstance(values, Mapping):
            raise ValueError("Trend configuration must be an object")
        values = dict(values)
        if "redundancy_threshold_bars" in values:
            # A fixed bar count cannot be converted to a pair-dependent
            # percentage. Old configurations receive the new display defaults;
            # explicit new settings, if present, take precedence.
            legacy = values.pop("redundancy_threshold_bars")
            if isinstance(legacy, bool) or not isinstance(legacy, int) or legacy < 0:
                raise ValueError("Legacy redundancy_threshold_bars must be a nonnegative integer")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown trend settings: {', '.join(sorted(map(str, unknown)))}")
        return cls(**values)


@dataclass(frozen=True)
class Trend:
    trend_id: str
    start_index: int
    end_index: int
    start_time: Timestamp
    end_time: Timestamp
    duration: int
    direction: int
    clarity: float
    slope: float
    intercept: float
    rate_per_bar: float
    boundary_robustness: float
    magnitude: float
    scores: tuple[float, float, float, float]
    tail_status: str
    tail_start_time: Timestamp | None
    retracement: float
    onset_truncated: bool
    tail_fit_improvement: float = 0.0
    tail_slope_change: float = 0.0
    tail_level_change: float = 0.0
    parent_id: str | None = None
    nearby_start_indices: tuple[int, ...] = ()
    lambda_values: tuple[float, ...] = ()
    lambda_levels: tuple[int, ...] = ()
    origin: str = "segmentation"


@dataclass(frozen=True)
class CandidateDiagnostic:
    start_index: int
    end_index: int
    duration: int
    direction: int
    clarity: float
    magnitude: float
    boundary_robustness: float
    scores: tuple[float, float, float, float]
    accepted: bool
    rejection_reason: str
    lambda_values: tuple[float, ...] = ()
    lambda_levels: tuple[int, ...] = ()
    origin: str = "segmentation"


@dataclass(frozen=True)
class AnalysisResult:
    trends: tuple[Trend, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]
    status: str
    analysis_start: Timestamp | None
    analysis_end: Timestamp | None
    source_offset: int
    config: TrendConfig
    algorithm_version: str = ALGORITHM_VERSION
    analysis_mode: str = "current"

    def to_frame(self) -> pl.DataFrame:
        """A stable, dashboard-independent tabular representation."""
        if not self.trends:
            return pl.DataFrame(schema={
                "trend_id": pl.String, "start_index": pl.Int64, "end_index": pl.Int64,
                "direction": pl.String, "start": pl.Datetime,
                "end": pl.Datetime, "bars": pl.Int64, "clarity": pl.Float64,
                "rate_per_bar": pl.Float64, "slope": pl.Float64, "boundary_robustness": pl.Float64,
                "magnitude": pl.Float64, "tail_status": pl.String,
                "retracement": pl.Float64, "onset_truncated": pl.Boolean,
                "lambda_values": pl.List(pl.Float64), "lambda_levels": pl.List(pl.Int64),
                "origin": pl.String,
            })
        return pl.DataFrame([{
            "trend_id": trend.trend_id, "start_index": trend.start_index, "end_index": trend.end_index,
            "direction": "Up" if trend.direction > 0 else "Down",
            "start": trend.start_time, "end": trend.end_time, "bars": trend.duration,
            "clarity": trend.clarity, "rate_per_bar": trend.rate_per_bar, "slope": trend.slope,
            "boundary_robustness": trend.boundary_robustness, "magnitude": trend.magnitude,
            "tail_status": trend.tail_status, "retracement": trend.retracement,
            "onset_truncated": trend.onset_truncated,
            "lambda_values": list(trend.lambda_values), "lambda_levels": list(trend.lambda_levels),
            "origin": trend.origin,
        } for trend in self.trends], schema_overrides={
            "lambda_values": pl.List(pl.Float64), "lambda_levels": pl.List(pl.Int64),
        })

    def diagnostics_frame(self) -> pl.DataFrame:
        schema = {"start_index": pl.Int64, "end_index": pl.Int64, "duration": pl.Int64,
                  "direction": pl.Int64, "clarity": pl.Float64, "magnitude": pl.Float64,
                  "boundary_robustness": pl.Float64, "score_ols": pl.Float64,
                  "score_robust": pl.Float64, "score_ordering": pl.Float64,
                  "score_efficiency": pl.Float64, "accepted": pl.Boolean,
                  "rejection_reason": pl.String, "lambda_values": pl.List(pl.Float64),
                  "lambda_levels": pl.List(pl.Int64), "origin": pl.String}
        rows = []
        for item in self.diagnostics:
            row = asdict(item)
            scores = row.pop("scores")
            row.update(zip(("score_ols", "score_robust", "score_ordering", "score_efficiency"), scores))
            rows.append(row)
        return pl.DataFrame(rows, schema=schema)
