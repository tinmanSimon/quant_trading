"""Exact all-boundary least-squares segmentation on a multi-resolution grid."""

import math

import numpy as np

from .models import TrendConfig


class LinearCosts:
    """Prefix sums allow constant-time regression on half-open intervals."""

    def __init__(self, values: np.ndarray):
        self.values = values - values[0]
        self.n = len(values)
        t = np.arange(self.n, dtype=float)
        self.y = np.r_[0.0, np.cumsum(self.values)]
        self.yy = np.r_[0.0, np.cumsum(self.values ** 2)]
        self.ty = np.r_[0.0, np.cumsum(t * self.values)]

    def fit(self, start, end):
        count = np.asarray(end) - np.asarray(start)
        sy = self.y[end] - self.y[start]
        syy = self.yy[end] - self.yy[start]
        sty = self.ty[end] - self.ty[start]
        mean_t = (np.asarray(start) + np.asarray(end) - 1) / 2
        sxx = count * (count * count - 1) / 12
        sxy = sty - mean_t * sy
        sy_centered = np.maximum(0.0, syy - sy * sy / count)
        slope = np.divide(sxy, sxx, out=np.zeros_like(sxy, dtype=float), where=sxx > 0)
        raw_cost = sy_centered - slope * sxy
        # Cancellation in exactly linear paths must not create spurious changes.
        tolerance = 128 * np.finfo(float).eps * np.maximum(np.finfo(float).tiny, np.abs(syy))
        cost = np.where(raw_cost <= tolerance, 0.0, raw_cost)
        intercept = sy / count - slope * (count - 1) / 2
        r2 = np.divide(slope * sxy, sy_centered, out=np.zeros_like(sxy, dtype=float), where=sy_centered > tolerance)
        return cost, slope, intercept, np.clip(r2, 0, 1)


def _penalties(n: int, full_cost: float, config: TrendConfig) -> np.ndarray:
    """Construct one prefix's penalty grid, including its no-split level."""
    base = config.penalty_multiplier * math.log(n)
    first = base * 2.0 ** -4
    if not np.isfinite(first) or first <= 0:
        raise ValueError("penalty_multiplier is outside the numerically supported range")
    levels = [first]
    while levels[-1] <= full_cost:
        if len(levels) >= 256:
            raise ValueError("Penalty grid exceeds 256 levels; increase penalty_multiplier or penalty_grid_step")
        next_level = levels[-1] * 2.0 ** config.penalty_grid_step
        if not math.isfinite(next_level) or next_level <= levels[-1]:
            raise ValueError("penalty_multiplier is outside the numerically supported range")
        levels.append(next_level)
    return np.asarray(levels)


PenaltyProvenance = dict[int, tuple[tuple[int, float], ...]]


def _provenance(starts: np.ndarray, penalties: np.ndarray) -> PenaltyProvenance:
    """Collect all grid levels that chose a boundary, in increasing λ order."""
    collected: dict[int, list[tuple[int, float]]] = {}
    for level, (start, penalty) in enumerate(zip(starts, penalties)):
        collected.setdefault(int(start), []).append((level, float(penalty)))
    return {start: tuple(collected[start]) for start in sorted(collected)}


def candidate_provenance(costs: LinearCosts, sigma: float, config: TrendConfig) -> PenaltyProvenance:
    """Map each final-segment start to every exact ``(grid level, λ)`` selecting it.

    λ is the numerical penalty applied to volatility-normalized fitting error,
    not the multiplier or grid level. One boundary may be selected by several
    penalties. DP uses every admissible bar boundary for every penalty.
    """
    n, minimum = costs.n, config.min_segment_length
    if n < minimum:
        return {}
    full_cost = float(costs.fit(0, n)[0]) / sigma ** 2
    penalties = _penalties(n, full_cost, config)
    # Avoid allocating a second dense N x N interval-cost matrix.
    dp = np.full((n + 1, len(penalties)), np.inf)
    dp[0] = -penalties
    final_starts = None
    for end in range(minimum, n + 1):
        starts = np.arange(end - minimum + 1)
        segment_cost = costs.fit(starts, end)[0] / sigma ** 2
        options = dp[starts] + segment_cost[:, None]
        minimum_cost = np.min(options, axis=0)
        # Equivalent partitions occur on symmetric curves. Choose the earliest
        # boundary within floating-point regression error, so multiplying all
        # prices by a constant cannot randomly break a numerical tie.
        roundoff = (512 * np.finfo(float).eps * max(np.finfo(float).tiny, float(costs.yy[end])) / sigma ** 2
                    + 1e-12 * np.abs(minimum_cost))
        best = np.argmax(options <= minimum_cost + roundoff, axis=0)
        dp[end] = options[best, np.arange(len(penalties))] + penalties
        if end == n:
            final_starts = best
    return _provenance(final_starts, penalties)


def candidate_starts(costs: LinearCosts, sigma: float, config: TrendConfig) -> list[int]:
    """Compatibility helper returning the unique starts, without provenance."""
    return list(candidate_provenance(costs, sigma, config))


class PrefixSegmentations:
    """Reuse exact segmentation costs across independently analyzed prefixes.

    A single penalty-based DP cannot be reused directly: volatility and the
    penalty grid both change with the endpoint. Instead, retain the minimum
    *unscaled* fitting error for each prefix and segment count. A query then
    applies that prefix's own volatility and penalty grid to those costs.

    ``candidate_starts(prefix_length)`` uses the exclusive prefix length, so a
    query of 20 analyzes observations 0 through 19. Every possible boundary is
    considered. Later observations never determine the costs or penalties of
    an earlier query. Arithmetic is shared; endpoints are not subsampled.

    The table is extended lazily. At the 2,000-bar input limit it contains at
    most 502,251 states, rather than one independent DP for every endpoint.
    """

    def __init__(self, values: np.ndarray, config: TrendConfig):
        self.values = np.asarray(values, dtype=float)
        self.config = config
        self.n = len(self.values)
        if self.n > config.max_history:
            raise ValueError("PrefixSegmentations requires at most max_history observations")
        self.costs = LinearCosts(self.values) if self.n else None
        shape = (self.n + 1, self.n // config.min_segment_length + 1)
        self._errors = np.full(shape, np.inf)
        self._last_starts = np.full(shape, -1, dtype=np.int32)
        self._errors[0, 0] = 0.0
        self._computed_length = config.min_segment_length - 1
        self._candidates: dict[int, PenaltyProvenance] = {}

    def _extend(self, prefix_length: int) -> None:
        minimum = self.config.min_segment_length
        for end in range(self._computed_length + 1, prefix_length + 1):
            starts = np.arange(end - minimum + 1)
            counts = np.arange(end // minimum)
            segment_cost = self.costs.fit(starts, end)[0]
            options = self._errors[starts, :len(counts)] + segment_cost[:, None]
            minimum_cost = np.min(options, axis=0)
            roundoff = (512 * np.finfo(float).eps
                        * max(np.finfo(float).tiny, float(self.costs.yy[end]))
                        + 1e-12 * np.abs(minimum_cost))
            best = np.argmax(options <= minimum_cost + roundoff, axis=0)
            self._errors[end, counts + 1] = options[best, counts]
            self._last_starts[end, counts + 1] = best
        self._computed_length = max(self._computed_length, prefix_length)

    def candidate_starts(self, prefix_length: int) -> list[int]:
        """Return the candidate starts for exactly ``values[:prefix_length]``."""
        return list(self.candidate_provenance(prefix_length))

    def candidate_provenance(self, prefix_length: int) -> PenaltyProvenance:
        """Return all exact selecting penalties, using only this prefix's data."""
        if (isinstance(prefix_length, bool) or not isinstance(prefix_length, int)
                or not 0 <= prefix_length <= self.n):
            raise ValueError("prefix_length must be an integer within the available history")
        if prefix_length < self.config.min_segment_length:
            return {}
        if prefix_length in self._candidates:
            return dict(self._candidates[prefix_length])
        # Local import avoids an import cycle: scoring also uses LinearCosts.
        # Sharing this function keeps normalization identical to one-shot runs.
        from .scoring import volatility

        sigma = volatility(self.values[:prefix_length])
        full_cost = float(self.costs.fit(0, prefix_length)[0]) / sigma ** 2
        penalties = _penalties(prefix_length, full_cost, self.config)
        self._extend(prefix_length)
        counts = np.arange(1, prefix_length // self.config.min_segment_length + 1)
        # Omit the same final penalty as the one-shot DP's comparison. This
        # constant has no effect on the optimum and preserves tie tolerances.
        options = (self._errors[prefix_length, counts, None] / sigma ** 2
                   + (counts[:, None] - 2) * penalties)
        minimum_cost = np.min(options, axis=0)
        roundoff = (512 * np.finfo(float).eps
                    * max(np.finfo(float).tiny, float(self.costs.yy[prefix_length])) / sigma ** 2
                    + 1e-12 * np.abs(minimum_cost))
        starts = self._last_starts[prefix_length, counts]
        best = np.min(np.where(options <= minimum_cost + roundoff, starts[:, None], self.n), axis=0)
        result = _provenance(best, penalties)
        self._candidates[prefix_length] = result
        return dict(result)
