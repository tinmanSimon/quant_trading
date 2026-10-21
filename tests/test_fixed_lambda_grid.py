"""A configured λ has the same numerical meaning at every historical endpoint."""

import numpy as np
import pytest

from research.analysis.trend import TrendConfig, analyze, penalty_grid
from test_trend_history import prices


def test_fixed_grid_has_explicit_start_geometric_step_and_level_count():
    config = TrendConfig(penalty_multiplier=2.1, penalty_grid_step=3, penalty_grid_levels=4)
    assert penalty_grid(config) == (2.1, 16.8, 134.4, 1075.2)
    assert penalty_grid(TrendConfig(penalty_grid_levels=1, penalty_multiplier=.125)) == (.125,)
    assert len(penalty_grid(TrendConfig())) == 20


@pytest.mark.parametrize("lookback", [2000, 24])
def test_every_endpoint_tests_the_identical_complete_grid_including_rolling_windows(lookback):
    frame = prices(np.cumsum(np.random.default_rng(52).normal(.002, .005, 64)))
    config = TrendConfig(max_history=lookback, penalty_multiplier=2.1,
                         penalty_grid_step=3, penalty_grid_levels=4)
    result = analyze(frame, config=config)
    expected = list(enumerate((2.1, 16.8, 134.4, 1075.2)))
    for end in range(7, 64):
        discovered = [(level, value) for record in result.diagnostics if record.end_index == end
                      for level, value in zip(record.lambda_levels, record.lambda_values)]
        assert sorted(discovered) == expected


def test_flat_prices_do_not_shorten_the_configured_grid():
    result = analyze(prices(np.zeros(16)))
    for record in result.diagnostics:
        assert record.lambda_values == penalty_grid(result.config)
        assert record.lambda_levels == tuple(range(result.config.penalty_grid_levels))


def test_grid_is_built_once_for_whole_history(monkeypatch):
    import research.analysis.trend.api as api
    import research.analysis.trend.segmentation as segmentation

    calls = []

    def counted(config):
        calls.append(config)
        return penalty_grid(config)

    monkeypatch.setattr(api, "penalty_grid", counted)
    monkeypatch.setattr(segmentation, "penalty_grid", counted)
    config = TrendConfig(max_history=16)
    analyze(prices(np.arange(40) * .003), config=config)
    assert calls == [config]


@pytest.mark.parametrize("value", [0, 257, 1.5, True, "12"])
def test_grid_level_count_must_be_an_integer_in_range(value):
    with pytest.raises(ValueError, match="penalty_grid_levels"):
        TrendConfig(penalty_grid_levels=value)


def test_tiny_positive_start_is_valid_for_a_bounded_fixed_grid():
    grid = penalty_grid(TrendConfig(penalty_multiplier=1e-300, penalty_grid_step=.25))
    assert len(grid) == 20
    assert all(0 < first < second for first, second in zip(grid, grid[1:]))
