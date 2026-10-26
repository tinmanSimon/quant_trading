"""Exact incremental percentiles for candidate-local adaptive bands."""

import numpy as np
import pytest

from research.analysis.flat.windows import _CandidatePercentile


@pytest.mark.parametrize("percentile", [0, .1, 12.5, 50, 66.67, 90, 99.9, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_candidate_percentile_matches_numpy_with_duplicates_and_random_values(percentile, reverse):
    rng = np.random.default_rng(671)
    values = np.concatenate([rng.choice([0., .1, 1., 2.], 140),
                             rng.uniform(0, 200, 140)])
    if reverse:
        values = values[::-1]
    candidate = _CandidatePercentile(percentile)
    for index, value in enumerate(values):
        candidate.add(float(value))
        expected = np.percentile(values[:index + 1], percentile, method="linear")
        assert candidate.value() == pytest.approx(expected, rel=2e-15, abs=0)
        assert len(candidate.lower) + len(candidate.upper) == index + 1


@pytest.mark.parametrize("percentile", [0, 50, 90, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_candidate_percentile_handles_ordered_bars_and_interpolation(percentile, reverse):
    values = np.arange(200, dtype=float)
    if reverse:
        values = values[::-1]
    candidate = _CandidatePercentile(percentile)
    for index, value in enumerate(values):
        candidate.add(float(value))
        assert candidate.value() == pytest.approx(np.percentile(values[:index + 1], percentile))


def test_candidate_percentile_includes_each_new_bar_and_can_rise_or_fall():
    candidate = _CandidatePercentile(50)
    medians = []
    for value in [1., 1., 9., 9., 9., 1.]:
        candidate.add(value)
        medians.append(candidate.value())
    assert medians == [1., 1., 1., 5., 9., 5.]
