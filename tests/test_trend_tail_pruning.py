"""Optimized tail scans must match an exhaustive scan, including score ties."""

from dataclasses import replace
import math

import numpy as np
import pytest

from trend_analysis import TrendConfig
from trend_analysis.scoring import Scorer
from trend_analysis.segmentation import LinearCosts
from trend_analysis.tails import TailInfo, diagnose


def exhaustive_tail(start, scorer):
    """Independent all-split oracle without optimistic bounds or early exits."""
    config, n = scorer.config, scorer.n
    full = scorer.score(start)
    values = scorer.values[start:]
    count = len(values)
    extreme = int(np.argmax(full.direction * values))
    movement = max(0.0, full.direction * float(values[extreme] - values[0]))
    retreat = max(0.0, full.direction * float(values[extreme] - values[-1]))
    retracement = retreat / max(movement, 1e-12)
    base = TailInfo(retracement=retracement)
    young = count - 1 - extreme
    if 0 < young < config.min_tail_length and retreat / (full.sigma * math.sqrt(max(1, young))) >= config.shock_threshold:
        base = TailInfo("Abrupt counter-move; insufficient observations", start + extreme + 1, retracement)
    minimum = max(config.min_tail_length, config.min_segment_length)
    maximum = min(int(count * config.max_tail_fraction), count - config.min_segment_length)
    full_error = float(scorer.costs.fit(start, n)[0])
    if maximum < minimum or full_error <= 0:
        return base
    prefix_slopes = scorer.prefix_slopes(start)
    choices, opposite, qualified = [], [], []
    for length in range(minimum, maximum + 1):
        split = n - length
        error = float(scorer.costs.fit(start, split)[0] + scorer.costs.fit(split, n)[0])
        gain = float(np.clip((full_error - error) / full_error, 0, 1))
        if gain < config.tail_fit_threshold:
            continue
        prefix_slope = float(prefix_slopes[split - start])
        tail = scorer.score(split)
        slope_change = abs(tail.slope - prefix_slope) * math.sqrt(length) / full.sigma
        prefix_intercept = float(np.median(values[:split - start] - prefix_slope * np.arange(split - start)))
        level_change = abs(tail.intercept - prefix_intercept - prefix_slope * (split - start)) / full.sigma
        if slope_change < config.tail_slope_threshold and level_change < config.tail_level_threshold:
            continue
        accepted = not scorer.rejection(split)
        reverses = accepted and tail.direction != full.direction
        slows = abs(tail.slope) <= config.slowing_ratio * abs(prefix_slope)
        if reverses:
            status = ("Broader direction materially challenged" if retracement >= config.retracement_threshold
                      or length / count >= config.countertrend_fraction else "Contains a local countertrend")
        elif slows:
            status = "Recent movement slowed or paused"
        elif slope_change >= config.tail_slope_threshold:
            status = "Recent slope changed"
        else:
            status = "Recent level shift"
        if level_change >= config.tail_level_threshold and slows and not reverses:
            status = "Level shift; recent movement slowed or paused"
        info = TailInfo(status, split, retracement, gain, slope_change, level_change, split if accepted else None)
        choices.append(info)
        if accepted:
            qualified.append(info)
            if reverses:
                opposite.append(info)
    if not choices:
        return base
    selected = max(opposite or choices, key=lambda item: item.fit_improvement)
    if selected.promoted_start is None and qualified:
        selected = replace(selected, promoted_start=max(qualified, key=lambda item: item.fit_improvement).start)
    if base.status.startswith("Abrupt") and not opposite:
        return base
    return selected


@pytest.mark.parametrize("config", [
    TrendConfig(),
    TrendConfig(clarity_threshold=.3, magnitude_threshold=.1, boundary_score_threshold=.2,
                boundary_acceptance=.5, tail_fit_threshold=.1),
    TrendConfig(weight_ols=0, weight_robust=1, weight_ordering=0, weight_efficiency=0,
                clarity_threshold=.001, magnitude_threshold=.001, boundary_fraction=0,
                boundary_score_threshold=0, boundary_acceptance=0, tail_fit_threshold=0,
                tail_slope_threshold=.001, tail_level_threshold=.001),
])
def test_tail_pruning_matches_every_split_oracle(config):
    rng = np.random.default_rng(13)
    paths = [np.cumsum(rng.normal(drift, .02, 72)) for drift in (-.005, 0, .005) for _ in range(6)]
    paths.extend([
        np.r_[np.arange(16) * .03, .45 + np.array([0., 7., 6., 2., 3., 4., 1., 5.]) * .01],
        np.r_[np.arange(50) * .01, np.full(20, .49)],
        np.r_[np.zeros(40), np.full(30, .3)],
    ])
    for values in paths:
        scorer = Scorer(values, LinearCosts(values), config)
        expected = exhaustive_tail(0, scorer)
        actual = diagnose(0, scorer)
        assert (actual.status, actual.start, actual.promoted_start) == (expected.status, expected.start, expected.promoted_start)
        for name in ("retracement", "fit_improvement", "slope_change", "level_change"):
            assert getattr(actual, name) == pytest.approx(getattr(expected, name), abs=1e-12)


def test_even_pair_ordering_tie_can_have_negative_robust_median():
    values = np.array([0., 7., 6., 2., 3., 4., 1., 5.]) * .01
    scorer = Scorer(values, LinearCosts(values), TrendConfig())
    assert scorer._ordering[0] == 0
    assert scorer.score(0).slope < 0
