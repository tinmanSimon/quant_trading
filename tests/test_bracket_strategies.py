"""Bracket signals cannot accidentally enter the target-allocation executor."""

import polars as pl
import pytest

from research.strategies import BracketStrategy, Momentum, WeightedStrategy, strategy_spec


class ExampleBracket(BracketStrategy):
    name = "example_bracket"
    lookback = 2
    stop_percentage = 0.9
    win_percentage = 1.1

    @property
    def config(self):
        return {}

    def entry_signal(self, history):
        return history["close"][-1] > history["close"][-2]


def test_bracket_is_a_reproducible_strategy_with_entry_signals():
    strategy = ExampleBracket()
    assert strategy_spec(strategy) == {"name": "example_bracket", "version": "1", "config": {}}
    assert strategy.entry_signal(pl.DataFrame({"close": [1., 2.]}))
    with pytest.raises(ValueError, match="shared portfolio execution"):
        strategy.target_weight(pl.DataFrame({"close": [1., 2.]}))


@pytest.mark.parametrize("weights", [[1., 0.], [0., 1.], [0.5, 0.5]])
def test_weighted_combination_rejects_bracket_even_at_zero_weight(weights):
    with pytest.raises(ValueError, match="cannot be included in a weighted"):
        WeightedStrategy([ExampleBracket(), Momentum(lookback=1)], weights)


def test_bracket_subclass_requires_entry_signal():
    class MissingSignal(BracketStrategy):
        @property
        def config(self):
            return {}

    with pytest.raises(TypeError, match="entry_signal"):
        MissingSignal()
