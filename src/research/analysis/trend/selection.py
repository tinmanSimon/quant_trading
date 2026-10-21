"""Display distinct causal trend records while retaining the complete archive.

This is presentation only. With unchanged settings and earlier records, appending
later endpoints cannot replace earlier selections. Every underlying interval
retains the result computed at its own endpoint.
"""

from dataclasses import dataclass
import math

import numpy as np

from .models import Trend


@dataclass(frozen=True)
class TrendSelection:
    chart_trends: tuple[Trend, ...]
    redundant_to: dict[str, str]


def filter_trends_by_lambda(trends, selected_lambdas=None, *, include_without_lambda=False):
    """Filter original intervals by any discovery λ; None means show all."""
    if selected_lambdas is None:
        return tuple(trends)
    selected = set(selected_lambdas)
    return tuple(trend for trend in trends if
                 selected.intersection(trend.lambda_values)
                 or (include_without_lambda and not trend.lambda_values))


def filter_trends_by_slope(trends, mode="all", *, minimum=None, maximum=None):
    """Filter original fits by signed log-price slope per bar, inclusively for ranges."""
    if mode not in ("all", "positive", "negative", "range"):
        raise ValueError("Unknown slope filter mode.")
    if mode == "range":
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (minimum, maximum)):
            raise ValueError("Slope bounds must be finite numbers.")
        if minimum > maximum:
            raise ValueError("Minimum slope must be less than or equal to maximum slope.")
    if mode == "all":
        return tuple(trends)
    return tuple(trend for trend in trends if
                 (mode == "positive" and trend.slope > 0)
                 or (mode == "negative" and trend.slope < 0)
                 or (mode == "range" and minimum <= trend.slope <= maximum))


def select_trends(
    trends, *, redundancy_percentage: float = 5.0,
    redundancy_boundary_rule: str = "and",
    redundancy_min_bars: int = 5,
) -> TrendSelection:
    """Keep the earliest-ending representative of nearby same-direction intervals.

    Process completed endpoints chronologically. Within one endpoint, prefer
    higher clarity, then longer duration, then interval ID. A later endpoint
    never replaces an earlier representative.

    The tolerance is a percentage of the shorter interval's inclusive bar count,
    without rounding, with a configurable minimum in bars. The boundary rule
    requires either both distances ("and")
    or at least one ("or") to be within that tolerance. Only retained
    representatives can suppress another interval, preventing a chain of small
    differences from collapsing a whole history. Lambda provenance does not
    affect redundancy. No result is deleted, merged, rescored, or refitted.
    """
    if (isinstance(redundancy_percentage, bool)
            or not isinstance(redundancy_percentage, (int, float))
            or not math.isfinite(redundancy_percentage)
            or not 0 <= redundancy_percentage <= 100):
        raise ValueError("redundancy_percentage must be a finite number between 0 and 100")
    if not isinstance(redundancy_boundary_rule, str) or redundancy_boundary_rule not in ("and", "or"):
        raise ValueError('redundancy_boundary_rule must be "and" or "or"')
    if (isinstance(redundancy_min_bars, bool) or not isinstance(redundancy_min_bars, int)
            or redundancy_min_bars < 0):
        raise ValueError("redundancy_min_bars must be a nonnegative integer")

    ranked = sorted(trends, key=lambda item: (
        item.end_index, -item.clarity, -item.duration, item.trend_id,
    ))
    # Pair-dependent tolerances cannot use the old fixed-width spatial buckets.
    # Vectorize comparisons with retained representatives for large archives.
    starts, ends, durations, directions = np.empty((4, len(ranked)), dtype=np.int64)
    retained = []
    redundant_to = {}
    for item in ranked:
        count = len(retained)
        # Compare in percent units to avoid division roundoff at exact limits,
        # e.g. (58 / 100) * 100 is just below 58 in binary floating point.
        # The bar minimum must use the same scale: 5 bars becomes 500 here.
        scaled_resolution = np.maximum(
            100 * redundancy_min_bars,
            redundancy_percentage * np.minimum(item.duration, durations[:count]),
        )
        starts_close = 100 * np.abs(item.start_index - starts[:count]) <= scaled_resolution
        ends_close = 100 * np.abs(item.end_index - ends[:count]) <= scaled_resolution
        close = starts_close & ends_close if redundancy_boundary_rule == "and" else starts_close | ends_close
        matches = close & (directions[:count] == item.direction)
        if matches.any():
            # Retained order is chronological, with deterministic same-endpoint ties.
            representative = retained[int(matches.argmax())]
            redundant_to[item.trend_id] = representative.trend_id
        else:
            retained.append(item)
            starts[count] = item.start_index
            ends[count] = item.end_index
            durations[count] = item.duration
            directions[count] = item.direction

    return TrendSelection(
        chart_trends=tuple(sorted(retained, key=lambda item: (
            item.start_index, item.end_index, item.trend_id,
        ))),
        redundant_to=redundant_to,
    )
