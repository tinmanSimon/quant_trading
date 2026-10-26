"""Causal longest wick envelopes with optional adaptive and candle-size limits."""

from collections import deque
from heapq import heappop, heappush
import math
from typing import Callable

import numpy as np

from .models import FlatConfig


_COMPARISON_TOLERANCE = 1 + 32 * np.finfo(float).eps


def envelope(lower: float, upper: float) -> tuple[float, float]:
    """Midpoint and total width percent without overflowing finite prices."""
    width = upper - lower
    center = lower + width / 2
    return center, 100 * (width / center)


class _CandidatePercentile:
    """Exact linear percentile for a growing candidate in O(log M) per bar.

    The lower heap ends at the quantile's lower rank; the upper heap supplies
    the next rank for interpolation. A fresh pair of heaps is used for each
    endpoint, so storage is bounded by the candidate's maximum duration.
    """

    def __init__(self, percentile: float):
        self.fraction = percentile / 100
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, value: float) -> None:
        if not self.lower or value <= -self.lower[0]:
            heappush(self.lower, -value)
        else:
            heappush(self.upper, value)
        size = len(self.lower) + len(self.upper)
        target = int((size - 1) * self.fraction) + 1
        if len(self.lower) > target:
            heappush(self.upper, -heappop(self.lower))
        elif len(self.lower) < target:
            heappush(self.lower, -heappop(self.upper))

    def value(self) -> float:
        rank = (len(self.lower) + len(self.upper) - 1) * self.fraction
        fraction = rank - int(rank)
        lower = -self.lower[0]
        if not fraction:
            return lower
        upper = self.upper[0]
        # Match NumPy's stable linear interpolation on either side of 0.5.
        if fraction >= 0.5:
            return upper - (upper - lower) * (1 - fraction)
        return lower + (upper - lower) * fraction


class EnvelopeWindows:
    """One earliest valid start per endpoint, without an interval matrix.

    Fixed bands without the candle filter retain the linear deque scan.
    Candidate percentiles and medians can change eligibility in either
    direction. Those modes inspect bounded suffixes independently at each
    endpoint; every statistic uses exactly the candidate's own candles.
    """

    def __init__(self, highs: np.ndarray, lows: np.ndarray, config: FlatConfig,
                 progress: Callable[[int, int], None] | None = None):
        count = len(highs)
        self.starts = np.full(count, -1, dtype=np.int64)
        self.lowers = np.full(count, np.nan)
        self.uppers = np.full(count, np.nan)
        self.limits = np.full(count, config.max_range_pct, dtype=float)
        self.max_bar_lengths = np.full(count, np.nan)
        self.median_bar_lengths = np.full(count, np.nan)
        total = max(0, count - config.min_duration + 1)
        if progress is not None:
            progress(0, total)
        if config.band_width_mode == "fixed" and not config.long_bar_filter:
            self._scan_fixed(highs, lows, config, progress, total)
            return

        widths = highs - lows
        lengths = 100 * (widths / (lows + widths / 2))
        if config.band_width_mode == "automatic":
            self.limits.fill(np.nan)

        for end in range(config.min_duration - 1, count):
            if not self._accept_longest_endpoint(end, highs, lows, lengths, config):
                self._scan_candidate_endpoint(end, highs, lows, lengths, config)
            if progress is not None:
                progress(end - config.min_duration + 2, total)

    def _scan_fixed(self, highs: np.ndarray, lows: np.ndarray, config: FlatConfig,
                    progress: Callable[[int, int], None] | None, total: int) -> None:
        high_indices, low_indices = deque(), deque()
        start = 0
        # Fixed per-configuration tolerance preserves monotonic eligibility.
        limit = config.max_range_pct * _COMPARISON_TOLERANCE
        for end in range(len(highs)):
            while high_indices and highs[high_indices[-1]] <= highs[end]:
                high_indices.pop()
            high_indices.append(end)
            while low_indices and lows[low_indices[-1]] >= lows[end]:
                low_indices.pop()
            low_indices.append(end)
            start = max(start, end - config.max_duration + 1)
            while high_indices and high_indices[0] < start:
                high_indices.popleft()
            while low_indices and low_indices[0] < start:
                low_indices.popleft()
            while start <= end:
                lower, upper = float(lows[low_indices[0]]), float(highs[high_indices[0]])
                if envelope(lower, upper)[1] <= limit:
                    break
                start += 1
                if high_indices[0] < start:
                    high_indices.popleft()
                if low_indices[0] < start:
                    low_indices.popleft()
            if end - start + 1 >= config.min_duration:
                self.starts[end] = start
                self.lowers[end] = lows[low_indices[0]]
                self.uppers[end] = highs[high_indices[0]]
            if progress is not None and end + 1 >= config.min_duration:
                progress(end - config.min_duration + 2, total)

    @staticmethod
    def _automatic_limit(percentile: float, config: FlatConfig) -> float:
        limit = percentile * config.auto_multiplier
        if not math.isfinite(limit):
            raise ValueError("Automatic maximum band width must be finite; reduce auto_multiplier")
        return limit

    @staticmethod
    def _within_width(width: float, limit: float) -> bool:
        # Envelope widths cannot exceed 200%, so large finite limits need no
        # tolerance multiplication that could overflow.
        tolerated = limit * _COMPARISON_TOLERANCE if limit < 200 else limit
        return width <= tolerated

    @staticmethod
    def _within_bar_limit(maximum: float, median: float, config: FlatConfig) -> bool:
        # A huge multiplier must not underflow the quotient and accidentally
        # admit a positive maximum when the median is zero.
        return (maximum == 0 if median == 0 else
                maximum / config.long_bar_multiplier <= median * _COMPARISON_TOLERANCE)

    def _accept_longest_endpoint(self, end: int, highs: np.ndarray, lows: np.ndarray,
                                 lengths: np.ndarray, config: FlatConfig) -> bool:
        """Accept the earliest permitted start immediately when it qualifies."""
        earliest = max(0, end - config.max_duration + 1)
        lower = float(np.min(lows[earliest:end + 1]))
        upper = float(np.max(highs[earliest:end + 1]))
        selected = lengths[earliest:end + 1]
        limit = config.max_range_pct
        median = math.nan
        if config.band_width_mode == "automatic":
            quantiles = ([config.auto_percentile, 50] if config.long_bar_filter
                         else config.auto_percentile)
            values = np.percentile(selected, quantiles, method="linear")
            if config.long_bar_filter:
                percentile, median = map(float, values)
            else:
                percentile = float(values)
            limit = self._automatic_limit(percentile, config)
        if not self._within_width(envelope(lower, upper)[1], limit):
            return False
        if config.long_bar_filter:
            maximum = float(np.max(selected))
            if config.band_width_mode == "fixed":
                median = float(np.median(selected))
            if not self._within_bar_limit(maximum, median, config):
                return False
            self.max_bar_lengths[end] = maximum
            self.median_bar_lengths[end] = median
        self.starts[end] = earliest
        self.lowers[end] = lower
        self.uppers[end] = upper
        self.limits[end] = limit
        return True

    def _scan_candidate_endpoint(self, end: int, highs: np.ndarray, lows: np.ndarray,
                                 lengths: np.ndarray, config: FlatConfig) -> None:
        earliest = max(0, end - config.max_duration + 1)
        lower, upper = math.inf, -math.inf
        automatic = config.band_width_mode == "automatic"
        percentile = _CandidatePercentile(config.auto_percentile) if automatic else None
        median_stat = _CandidatePercentile(50) if config.long_bar_filter else None
        maximum = 0.0
        # The longest candidate was already checked and rejected. Each
        # remaining suffix gets its own percentile and optional median.
        for start in range(end, earliest, -1):
            lower = min(lower, float(lows[start]))
            upper = max(upper, float(highs[start]))
            width = envelope(lower, upper)[1]
            if not automatic and not self._within_width(width, config.max_range_pct):
                break
            length = float(lengths[start])
            maximum = max(maximum, length)
            if percentile is not None:
                percentile.add(length)
            if median_stat is not None:
                median_stat.add(length)
            if end - start + 1 < config.min_duration:
                continue
            limit = (self._automatic_limit(percentile.value(), config)
                     if percentile is not None else config.max_range_pct)
            if not self._within_width(width, limit):
                continue
            if median_stat is not None:
                median = median_stat.value()
                if not self._within_bar_limit(maximum, median, config):
                    continue
                self.max_bar_lengths[end] = maximum
                self.median_bar_lengths[end] = median
            self.starts[end] = start
            self.lowers[end] = lower
            self.uppers[end] = upper
            self.limits[end] = limit
            # Neither an automatic width rejection nor a median rejection
            # can stop expansion: older bars can change either statistic.
