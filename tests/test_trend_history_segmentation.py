"""All-endpoint discovery must match independent historical-prefix runs."""

from time import perf_counter

import numpy as np
import pytest

from research.analysis.trend.models import TrendConfig
from research.analysis.trend.scoring import volatility
from research.analysis.trend.segmentation import LinearCosts, PrefixSegmentations, candidate_provenance, candidate_starts


@pytest.mark.parametrize("config", [
    TrendConfig(),
    TrendConfig(min_segment_length=12, penalty_multiplier=2.3, penalty_grid_step=.5),
    TrendConfig(penalty_multiplier=.3, penalty_grid_step=2),
    TrendConfig(max_history=44),
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
        PrefixSegmentations(np.arange(41.), TrendConfig(max_history=20))


def test_internal_window_capacity_can_exceed_the_public_setting_limit():
    config = TrendConfig(max_history=100_000, min_segment_length=100_000)
    shared = PrefixSegmentations(np.zeros(200_000), config)
    assert shared.candidate_starts(100_001) == [0]


def test_overflowing_penalty_grid_fails_before_allocating_history():
    with pytest.raises(ValueError, match="numerically supported"):
        PrefixSegmentations(np.arange(40) ** 2 * .001, TrendConfig(penalty_multiplier=1e308))


def test_all_two_thousand_endpoints_share_bounded_work():
    values = np.cumsum(np.random.default_rng(741).normal(.001, .01, 2000))
    began = perf_counter()
    shared = PrefixSegmentations(values, TrendConfig())
    results = [shared.candidate_starts(length) for length in range(8, 2001)]
    elapsed = perf_counter() - began
    assert all(results)
    assert len(results[-1]) > 1
    assert elapsed < 20, f"Exact segmentation of all 2,000 prefixes took {elapsed:.2f}s"
