"""Counts and return volatility restricted to disjoint detected flat periods."""

from bisect import bisect_left
from calendar import monthrange
from copy import deepcopy
import math

import numpy as np

from research.analysis.core.base import BaseChartLogic, LogicResult
from research.analysis.flat import ALGORITHM_VERSION, FlatConfig
from research.analysis.core.omissions import analyze_flat_context
from .flat import FlatLogic


_FLAT_PROPERTIES = deepcopy(FlatLogic.config_schema["properties"])
_COUNT_COLUMNS = [
    "flat_bar_count", "red_bar_count", "red_high_above_count",
    "red_high_not_above_count", "all_high_above_count", "return_count",
]
_STAT_COLUMNS = [*_COUNT_COLUMNS, "return_volatility_pct"]
_PERIOD_COLUMNS = [
    "period_id", "start_index", "end_index", "start", "end", "lower_price",
    "upper_price", "range_pct", "max_range_pct", *_STAT_COLUMNS,
]


def _return_volatility(returns):
    """Population standard deviation of simple returns, in percentage points."""
    if not returns.size:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.std(returns, ddof=0) * 100)
    if not math.isfinite(value):
        raise ValueError("Close-to-close returns are too large to calculate finite volatility.")
    return value


def _previous_month(value):
    """Same calendar day/time in the prior month, clamped at month end."""
    year, month = (value.year, value.month - 1) if value.month > 1 else (value.year - 1, 12)
    return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))


def _bar_statistics(opens, highs, closes, factor):
    red = closes < opens
    # An overflowing threshold is above all finite prices and cannot qualify.
    with np.errstate(over="ignore"):
        above = highs > opens * factor
    with np.errstate(over="ignore", invalid="ignore"):
        returns = closes[1:] / closes[:-1] - 1
    if not np.isfinite(returns).all():
        raise ValueError("Close-to-close returns must be finite.")
    red_count = int(np.count_nonzero(red))
    red_above = int(np.count_nonzero(red & above))
    return {
        "flat_bar_count": len(closes), "red_bar_count": red_count,
        "red_high_above_count": red_above, "red_high_not_above_count": red_count - red_above,
        "all_high_above_count": int(np.count_nonzero(above)),
        "return_count": len(returns), "return_volatility_pct": _return_volatility(returns),
    }, returns


def _summarize(rows, return_samples, source_count):
    pooled_returns = np.concatenate(return_samples) if return_samples else np.empty(0)
    counts = {name: sum(row[name] for row in rows) for name in _COUNT_COLUMNS}
    return {
        "source_bar_count": source_count, "flat_period_count": len(rows),
        **counts,
        "red_high_above_ratio": (counts["red_high_above_count"] / counts["red_bar_count"]
                                 if counts["red_bar_count"] else None),
        "return_volatility_pct": _return_volatility(pooled_returns),
    }


class FlatBarStatisticsLogic(BaseChartLogic):
    name = "flat_bar_statistics"
    version = f"1.3.0-engine.{ALGORITHM_VERSION}"
    result_version = "3"
    label = "Flat-period bar statistics"
    description = (
        "Detect flat periods, then count red bars and all bars whose high exceeds a percentage "
        "of their own open. Report counts and close-to-close return volatility for the full range "
        "and the last calendar month."
    )
    required_columns = ("timestamp", "open", "high", "low", "close")
    input_policy = "completed"
    config_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "high_percentage": {
                "type": "number", "exclusiveMinimum": 0, "default": 102.0,
                "title": "High threshold (% of open)",
                "description": (
                    "A direct percentage of each bar's opening price: 102 means high > open × 1.02. "
                    "Equality does not qualify. Red means close < open; doji bars are not red."
                ),
            },
            **_FLAT_PROPERTIES,
        },
        "x-ui-groups": [
            {"title": "Bar statistics", "fields": ["high_percentage"], "expanded": True},
            {"title": "Flat-period detection", "fields": list(_FLAT_PROPERTIES), "expanded": True},
        ],
    }

    def validate(self, config):
        values = dict(config)
        percentage = values.pop("high_percentage", 102.0)
        try:
            finite = math.isfinite(percentage)
        except (TypeError, OverflowError):
            finite = False
        if (isinstance(percentage, bool) or not isinstance(percentage, (int, float))
                or not finite or percentage <= 0):
            raise ValueError("high_percentage must be a finite number greater than zero; 102 means 102% of open.")
        return {**FlatConfig.from_dict(values).to_dict(), "high_percentage": float(percentage)}

    def calculate(self, context, config, progress=None):
        values = self.validate(config)
        flat_config = FlatConfig.from_dict({key: value for key, value in values.items()
                                           if key != "high_percentage"})
        frame = context.completed_frame
        missing = set(self.required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Flat-period bar statistics requires columns: {', '.join(sorted(missing))}")
        # The shared detector validates OHLC and returns a strictly disjoint
        # summary. Overlapping historical candidates must not multiply counts.
        analysis = analyze_flat_context(context, flat_config, progress)
        opens = frame["open"].cast(float).to_numpy()
        highs = frame["high"].cast(float).to_numpy()
        closes = frame["close"].cast(float).to_numpy()
        factor = values["high_percentage"] / 100
        timestamps = frame["timestamp"].to_list()
        tail_end = analysis.analysis_end
        tail_cutoff = _previous_month(tail_end) if tail_end else None
        tail_index = bisect_left(timestamps, tail_cutoff) if tail_cutoff else 0
        rows, return_samples, tail_rows, tail_returns = [], [], [], []
        for period in analysis.periods:
            selected = slice(period.start_index, period.end_index + 1)
            stats, returns = _bar_statistics(opens[selected], highs[selected], closes[selected], factor)
            return_samples.append(returns)
            rows.append({
                "period_id": period.period_id,
                "start_index": period.start_index, "end_index": period.end_index,
                "start": period.start_time.isoformat(), "end": period.end_time.isoformat(),
                "lower_price": period.lower_price, "upper_price": period.upper_price,
                "range_pct": period.range_pct, "max_range_pct": period.max_range_pct,
                **stats,
            })
            if period.end_index >= tail_index:
                # Clip existing periods; do not rerun detection or bridge the cutoff.
                recent = slice(max(period.start_index, tail_index), period.end_index + 1)
                stats, returns = _bar_statistics(opens[recent], highs[recent], closes[recent], factor)
                tail_rows.append(stats)
                tail_returns.append(returns)
        summary = _summarize(rows, return_samples, analysis.source_count)
        tail_summary = {
            "last_month_cutoff": tail_cutoff.isoformat() if tail_cutoff else None,
            "last_month_start": timestamps[tail_index].isoformat() if timestamps else None,
            "last_month_end": tail_end.isoformat() if tail_end else None,
            **{f"last_month_{key}": value for key, value in
               _summarize(tail_rows, tail_returns, analysis.source_count - tail_index).items()},
        }
        if analysis.status == "insufficient_data":
            message = f"Insufficient completed bars: need at least {flat_config.min_duration:,}."
        elif not rows:
            message = "No flat periods match the configured detection settings. Counts are zero; volatility is unavailable."
        else:
            message = (f"{summary['flat_period_count']:,} disjoint flat periods; "
                       f"{summary['flat_bar_count']:,} bars counted once. "
                       f"{summary['red_high_above_count']:,} red bars exceed the high threshold; "
                       f"{summary['red_high_not_above_count']:,} red bars do not.")
        return LogicResult(
            tables=[
                {"name": "Flat-bar summary", "columns": list(summary), "rows": [summary]},
                {"name": "Last-month flat-bar summary", "columns": list(tail_summary), "rows": [tail_summary]},
                {"name": "Flat-period details", "columns": _PERIOD_COLUMNS, "rows": rows, "default_page": 1},
            ],
            messages=[
                message,
                "Red high-above ratio = red_high_above_count / red_bar_count. "
                "It is unavailable when there are no red bars; 0.6 means 60%.",
                f"Threshold: high > open × {factor:g} ({values['high_percentage']:g}% of open). "
                "Equality is counted as not above. Red bars have close < open. All counts use only "
                "completed bars in the flat package's latest-first disjoint periods; overlapping candidates "
                "are not counted repeatedly.",
                "Flat periods and return samples do not cross recorded missing bars in the selected data.",
                "Return volatility (%) is 100 × the population standard deviation of close / previous close − 1. "
                "Both closes must belong to the same flat period, so the first bar of every period contributes "
                "no return. The ticker summary pools those within-period returns before calculating volatility; "
                "it does not average period volatilities or include jumps between periods. Values are not annualized.",
                "The last-month summary covers one calendar month ending at the latest completed source bar "
                "(cutoff inclusive). It uses the same detected periods, clipped to that window; both closes "
                "must be inside the window for a return. Period count includes any period with bars in the window. "
                "Shorter histories use all available selected bars. Cutoff, first included source bar, and end "
                "are shown in the summary.",
            ],
            data={"metadata": {
                "config": values, "flat_algorithm_version": analysis.algorithm_version,
                "flat_status": analysis.status, "flat_selection": "latest_first_disjoint",
                "analysis_start": analysis.analysis_start.isoformat() if analysis.analysis_start else None,
                "analysis_end": analysis.analysis_end.isoformat() if analysis.analysis_end else None,
                "volatility": "population_std_within_period_simple_returns_pct",
                "last_month_window": "calendar_month_to_latest_completed_source_bar_inclusive",
            }},
        )
