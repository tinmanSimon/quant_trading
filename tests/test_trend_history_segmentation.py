"""All-endpoint discovery must match independent historical-prefix runs."""

from time import perf_counter

import numpy as np
import pytest

from trend_analysis.models import TrendConfig
from trend_analysis.scoring import volatility
from trend_analysis.segmentation import LinearCosts, PrefixSegmentations, candidate_provenance, candidate_starts


@pytest.mark.parametrize("config", [
    TrendConfig(),
    TrendConfig(min_segment_length=12, penalty_multiplier=2.3, penalty_grid_step=.5),
    TrendConfig(penalty_multiplier=.3, penalty_grid_step=2),
])
@pytest.mark.parametrize("path", [
    np.cumsum(np.random.default_rng(54).normal(.001, .01, 88)),
    np.r_[np.arange(35) * .01, .34 - np.arange(1, 31) * .02, -.26 + np.arange(1, 24) * .03],
    np.square(np.arange(88) - 44) * .001,
    np.arange(88) * .003,
    np.r_[np.zeros(44), np.ones(44)],
    np.zeros(88),
])
def test_all_prefix_candidates_match_independent_segmentation(path, config):
    shared = PrefixSegmentations(path, config)
    for length in range(config.min_segment_length, len(path) + 1):
        prefix = path[:length]
        expected = candidate_starts(LinearCosts(prefix), volatility(prefix), config)
        assert shared.candidate_starts(length) == expected, length
        assert shared.candidate_provenance(length) == candidate_provenance(
            LinearCosts(prefix), volatility(prefix), config), length


def test_appending_or_changing_future_prices_cannot_change_prefix_candidates():
    config = TrendConfig()
    past = np.cumsum(np.random.default_rng(519).normal(0, .02, 72))
    baseline = PrefixSegmentations(past, config)
    extended = PrefixSegmentations(np.r_[past, [100, -100, 2, 9] * 7], config)
    # Query the future first: cached arithmetic must not revise earlier states.
    extended.candidate_starts(100)
    for length in range(8, 73):
        assert extended.candidate_starts(length) == baseline.candidate_starts(length)
        assert extended.candidate_provenance(length) == baseline.candidate_provenance(length)


def test_earlier_query_is_independent_of_later_queries_and_returned_list_edits():
    path = np.cumsum(np.random.default_rng(71).normal(0, .01, 80))
    shared = PrefixSegmentations(path, TrendConfig())
    expected = shared.candidate_starts(40)
    shared.candidate_starts(80)
    edited = shared.candidate_starts(40)
    edited.append(999)
    assert shared.candidate_starts(40) == expected
    expected_provenance = shared.candidate_provenance(40)
    edited_provenance = shared.candidate_provenance(40)
    edited_provenance[999] = ((0, 100.),)
    assert shared.candidate_provenance(40) == expected_provenance


def test_prefix_lengths_and_history_limit_are_validated():
    shared = PrefixSegmentations(np.arange(20.), TrendConfig())
    for length in (-1, 21, True, 8.0):
        with pytest.raises(ValueError, match="prefix_length"):
            shared.candidate_starts(length)
    assert shared.candidate_starts(0) == []
    assert shared.candidate_starts(7) == []
    assert PrefixSegmentations(np.array([]), TrendConfig()).candidate_starts(0) == []
    with pytest.raises(ValueError, match="max_history"):
        PrefixSegmentations(np.arange(21.), TrendConfig(max_history=20))


@pytest.mark.parametrize("config, message", [
    (TrendConfig(penalty_multiplier=1e-300, penalty_grid_step=.25), "256 levels"),
    (TrendConfig(penalty_multiplier=1e308), "numerically supported"),
])
def test_extreme_penalty_grid_fails_before_expanding_history(config, message):
    shared = PrefixSegmentations(np.arange(40) ** 2 * .001, config)
    with pytest.raises(ValueError, match=message):
        shared.candidate_starts(40)
    assert shared._computed_length == config.min_segment_length - 1


def test_all_two_thousand_endpoints_share_bounded_work():
    values = np.cumsum(np.random.default_rng(741).normal(.001, .01, 2000))
    began = perf_counter()
    shared = PrefixSegmentations(values, TrendConfig())
    results = [shared.candidate_starts(length) for length in range(8, 2001)]
    elapsed = perf_counter() - began
    assert all(results)
    assert len(results[-1]) > 1
    assert elapsed < 20, f"Exact segmentation of all 2,000 prefixes took {elapsed:.2f}s"
