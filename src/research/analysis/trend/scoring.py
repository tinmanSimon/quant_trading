"""Exact robust slopes and four complementary descriptive scores."""

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

from .models import TrendConfig
from .segmentation import LinearCosts


VOLATILITY_FLOOR = 1e-8


def volatility(values: np.ndarray) -> float:
    return max(VOLATILITY_FLOOR, float(np.std(np.diff(values), ddof=1))) if len(values) > 2 else VOLATILITY_FLOOR


def prefix_robust_slopes(values: np.ndarray) -> np.ndarray:
    """Exact Theil-Sen slope for every prefix, without cubic repeated medians.

    All pair slopes are ranked once. As each new observation arrives, its
    pair ranks are added to a two-level histogram. Rank selection gives the
    exact median (averaging both middle values for an even number of pairs).
    This is O(n² log n) for sorting, with O(n²) memory, bounded by the window.
    """
    n = len(values)
    result = np.zeros(n + 1)
    if n < 2:
        return result
    later, earlier = np.tril_indices(n, -1)
    pair_slopes = (values[later] - values[earlier]) / (later - earlier)
    if n <= 96:
        # For tiny prefixes direct C-level partitioning costs less than the
        # histogram setup and Python loop overhead. Both paths are exact.
        for end in range(2, n + 1):
            result[end] = np.median(pair_slopes[:end * (end - 1) // 2])
        return result
    unique, ranks = np.unique(pair_slopes, return_inverse=True)
    width = max(1, int(math.sqrt(len(unique))))
    bucket_count = (len(unique) + width - 1) // width
    fine = np.zeros(len(unique), dtype=np.int64)
    coarse = np.zeros(bucket_count, dtype=np.int64)
    offset = 0
    for end in range(1, n):
        new = ranks[offset:offset + end]
        offset += end
        np.add.at(fine, new, 1)
        coarse += np.bincount(new // width, minlength=bucket_count)
        cumulative = coarse.cumsum()
        low, high = (offset - 1) // 2, offset // 2
        bucket = int(cumulative.searchsorted(low, side="right"))
        before = int(cumulative[bucket - 1]) if bucket else 0
        local = fine[bucket * width:(bucket + 1) * width].cumsum()
        rank_low = bucket * width + int(local.searchsorted(low - before, side="right"))
        if high < cumulative[bucket]:
            rank_high = bucket * width + int(local.searchsorted(high - before, side="right"))
        else:
            next_bucket = int(cumulative.searchsorted(high, side="right"))
            local = fine[next_bucket * width:(next_bucket + 1) * width].cumsum()
            rank_high = next_bucket * width + int(local.searchsorted(high - cumulative[next_bucket - 1], side="right"))
        result[end + 1] = (unique[rank_low] + unique[rank_high]) / 2
    return result


@dataclass(frozen=True)
class Score:
    evidence: float
    clarity: float
    direction: int
    slope: float
    magnitude: float
    sigma: float
    scores: tuple[float, float, float, float]
    values: np.ndarray

    @cached_property
    def intercept(self) -> float:
        return float(np.median(self.values - self.slope * np.arange(len(self.values))))


class SharedIntervals:
    """Reuse exact interval estimators across historical endpoints.

    Each start's prefix medians are grown geometrically, so short local
    intervals do not cause work on an entire future history. A prefix's median
    includes only its own pairs: future values used to build the rank index
    cannot affect it. Cached arrays are discarded once outside the lookback.
    """

    def __init__(self, values: np.ndarray, max_history: int):
        self.values = values
        self.max_history = max_history
        self._prefixes: dict[int, np.ndarray] = {}
        self._intercepts: dict[int, dict[int, float]] = {}
        self._ordering_end = 1
        self._ordering_totals = np.zeros(len(values), dtype=np.int64)

    def prefixes(self, start: int, end: int) -> np.ndarray:
        length = end - start
        cached = self._prefixes.get(start)
        if cached is not None and len(cached) > length:
            return cached
        capacity = min(max(16, 1 << (length - 1).bit_length()),
                       self.max_history, len(self.values) - start)
        values = self.values[start:start + capacity]
        slopes = prefix_robust_slopes(values)
        self._prefixes[start] = slopes
        return slopes

    def ordering_at(self, end: int, offset: int) -> np.ndarray:
        """Update all interval ordering counts in O(lookback) per new endpoint."""
        if end < self._ordering_end:
            raise ValueError("Shared interval endpoints must be visited in increasing order")
        for last in range(self._ordering_end, end):
            start = max(0, last + 1 - self.max_history)
            new_pairs = np.sign(self.values[last] - self.values[start:last]).astype(np.int64)
            self._ordering_totals[start:last] += new_pairs[::-1].cumsum()[::-1]
        self._ordering_end = end
        lengths = end - np.arange(offset, end)
        return np.divide(2 * self._ordering_totals[offset:end], lengths * (lengths - 1),
                         out=np.zeros(end - offset), where=lengths > 1)

    def discard_before(self, start: int) -> None:
        for key in tuple(self._prefixes):
            if key < start:
                del self._prefixes[key]
                self._intercepts.pop(key, None)

    def intercept(self, start: int, end: int, slope: float) -> float:
        cached = self._intercepts.setdefault(start, {})
        if end not in cached:
            cached[end] = float(np.median(self.values[start:end] - slope * np.arange(end - start)))
        return cached[end]


class Scorer:
    """Cache all suffix scores and boundary checks within one analysis."""

    def __init__(self, values: np.ndarray, costs: LinearCosts, config: TrendConfig,
                 *, shared: SharedIntervals | None = None, offset: int = 0):
        self.values = values
        self.costs = costs
        self.config = config
        self.n = len(values)
        self._scores: dict[int, Score] = {}
        self._boundaries: dict[int, float] = {}
        self.shared = shared
        self.offset = offset
        starts = np.arange(self.n)
        _, self._ols, _, self._r2 = costs.fit(starts, self.n)
        returns = np.diff(values)
        sums = np.r_[np.cumsum(returns[::-1])[::-1], 0.0]
        squares = np.r_[np.cumsum((returns * returns)[::-1])[::-1], 0.0]
        count = self.n - starts - 1
        centered = squares - np.divide(sums * sums, count, out=np.zeros_like(sums), where=count > 0)
        variance = np.divide(centered, count - 1, out=np.zeros_like(sums), where=count > 1)
        self._sigma = np.maximum(VOLATILITY_FLOOR, np.sqrt(np.maximum(0.0, variance)))
        distances = np.r_[np.cumsum(np.abs(returns)[::-1])[::-1], 0.0]
        self._efficiency = np.divide(values[-1] - values, distances,
                                     out=np.zeros_like(values), where=distances > 0)
        self._range = np.maximum.accumulate(values[::-1])[::-1] - np.minimum.accumulate(values[::-1])[::-1]
        if shared is not None:
            self._ordering = shared.ordering_at(self.n + offset, offset)
            return
        self._slopes = -prefix_robust_slopes(values[::-1])[self.n - np.arange(self.n)]
        # Accumulate ordering counts for suffixes. Ties contribute zero.
        ordering = np.zeros(self.n)
        total = 0.0
        for start in range(self.n - 2, -1, -1):
            total += float(np.sign(values[start + 1:] - values[start]).sum())
            count = self.n - start
            ordering[start] = total * 2 / (count * (count - 1))
        self._ordering = ordering

    def prefix_slopes(self, start: int) -> np.ndarray:
        if self.shared is not None:
            return self.shared.prefixes(start + self.offset, self.n + self.offset)[:self.n - start + 1]
        return prefix_robust_slopes(self.values[start:])

    def prefix_intercept(self, start: int, end: int, slope: float) -> float:
        if self.shared is not None:
            return self.shared.intercept(start + self.offset, end + self.offset, slope)
        return float(np.median(self.values[start:end] - slope * np.arange(end - start)))

    @cached_property
    def evidence_without_robust(self) -> np.ndarray:
        lengths = self.n - np.arange(self.n) - 1
        ols_scores = self._r2 * np.tanh(self._ols * np.sqrt(lengths) / self._sigma / 2)
        return (self.config.weight_ols * ols_scores + self.config.weight_ordering * self._ordering
                + self.config.weight_efficiency * self._efficiency)

    @cached_property
    def can_reach_clarity(self) -> np.ndarray:
        # Even the most favorable robust score cannot rescue these intervals.
        return (np.abs(self.evidence_without_robust) + self.config.weight_robust + 1e-12
                >= self.config.clarity_threshold)

    def score(self, start: int) -> Score:
        if start in self._scores:
            return self._scores[start]
        values = self.values[start:]
        count = len(values)
        ols, r2 = float(self._ols[start]), float(self._r2[start])
        sigma = float(self._sigma[start])
        if self.shared is None:
            slope = float(self._slopes[start])
            ordering = float(self._ordering[start])
        else:
            slopes = self.shared.prefixes(start + self.offset, self.n + self.offset)
            slope, ordering = float(slopes[count]), float(self._ordering[start])
        # Remove numerical directional noise on effectively constant prices.
        if self._range[start] < 1e-13:
            slope, ols, r2 = 0.0, 0.0, 0.0
        z_ols = float(ols) * math.sqrt(count - 1) / sigma
        z_robust = slope * math.sqrt(count - 1) / sigma
        efficiency = float(self._efficiency[start])
        scores = (float(r2) * math.tanh(z_ols / 2), math.tanh(z_robust / 2),
                  ordering, efficiency)
        evidence = float(np.dot(self.config.weights, scores))
        result = Score(evidence=evidence, clarity=abs(evidence), direction=int(np.sign(evidence)),
                       slope=slope, magnitude=abs(z_robust), sigma=sigma, scores=scores, values=values)
        self._scores[start] = result
        return result

    def boundary(self, start: int) -> float:
        if start in self._boundaries:
            return self._boundaries[start]
        score = self.score(start)
        h = (max(1, round(self.config.boundary_fraction * (self.n - start)))
             if self.config.boundary_fraction > 0 else 0)
        candidates = {start - h, start - h // 2, start, start + h // 2, start + h}
        candidates = {item for item in candidates if 0 <= item <= self.n - self.config.min_segment_length}
        supports = [
            score.direction * self.evidence_without_robust[item] + self.config.weight_robust + 1e-12
            >= self.config.boundary_score_threshold
            and score.direction * self.score(item).evidence >= self.config.boundary_score_threshold
            and score.direction * self.score(item).slope > 0
            for item in candidates
        ]
        value = float(sum(supports) / len(supports)) if supports else 0.0
        self._boundaries[start] = value
        return value

    def rejection(self, start: int) -> str:
        score = self.score(start)
        reasons = []
        if score.clarity < self.config.clarity_threshold:
            reasons.append("insufficient clarity")
        if score.magnitude < self.config.magnitude_threshold:
            reasons.append("insufficient normalized magnitude")
        if score.direction * score.slope <= 0:
            reasons.append("robust slope disagrees with direction")
        if self.boundary(start) < self.config.boundary_acceptance:
            reasons.append("boundary-sensitive evidence")
        return "; ".join(reasons)
