"""Interchangeable strategies are deterministic, bounded, and reproducible."""

import math

import polars as pl
import pytest

from research.strategies import Momentum, MovingAverageCross, Strategy, StrategyRegistry, WeightedStrategy, load_strategy, strategy_spec


class Constant(Strategy):
    name = "constant"
    lookback = 1

    def __init__(self, target=1.0):
        self.target = target

    @property
    def config(self):
        return {"target": self.target}

    def target_weight(self, history):
        return self.target


def test_moving_average_and_momentum_direction():
    up = pl.DataFrame({"close": [10.0, 11.0, 12.0]})
    down = pl.DataFrame({"close": [12.0, 11.0, 10.0]})
    for strategy in [MovingAverageCross(fast=1, slow=3), Momentum(lookback=2)]:
        assert strategy.lookback == 3
        assert strategy.target_weight(up) == 1.0
        assert strategy.target_weight(down) == 0.0
        with pytest.raises(ValueError, match="Insufficient history"):
            strategy.target_weight(up.head(1))


@pytest.mark.parametrize("factory", [lambda: MovingAverageCross(3, 3), lambda: MovingAverageCross(0, 3), lambda: MovingAverageCross(True, 3), lambda: Momentum(0), lambda: Momentum(2.5)])
def test_invalid_builtin_configuration(factory):
    with pytest.raises(ValueError):
        factory()


def test_combination_is_weighted_signal_not_sum_of_accounts():
    strategy = WeightedStrategy([Constant(1.0), Constant(0.0)], [0.6, 0.4])
    assert strategy.target_weight(pl.DataFrame({"close": [1.0]})) == 0.6
    assert strategy.config["strategies"][0]["name"] == "constant"
    assert strategy.lookback == 1


@pytest.mark.parametrize("weights", [[], [0.5], [-0.1, 1.1], [0.1, 0.2], [math.nan, 0.0], [True, False]])
def test_bad_combination_weights(weights):
    with pytest.raises(ValueError):
        WeightedStrategy([Constant(), Constant()], weights)


def test_strategy_registry_and_nested_ensemble_round_trip():
    specs = [
        {"name": "moving_average_cross", "version": "1", "config": {"fast": 2, "slow": 5}},
        {"name": "momentum", "version": "1", "config": {"lookback": 3}},
    ]
    spec = {"name": "weighted", "version": "1", "config": {"strategies": specs, "weights": [0.3, 0.7]}}
    strategy = load_strategy(spec)
    assert strategy_spec(strategy) == spec
    assert strategy.lookback == 5


def test_custom_factory_receives_detached_configuration():
    registry = StrategyRegistry()

    def factory(config):
        target = config.pop("target")
        return Constant(target)

    registry.register("constant", "1", factory)
    spec = {"name": "constant", "version": "1", "config": {"target": 0.7}}
    assert registry.load(spec).target == 0.7
    assert spec["config"] == {"target": 0.7}
    with pytest.raises(ValueError, match="already registered"):
        registry.register("constant", "1", factory)


@pytest.mark.parametrize("spec", [{}, {"name": "momentum", "version": "2", "config": {}}, {"name": "momentum", "version": "1", "config": []}, {"name": [], "version": "1", "config": {}}])
def test_invalid_specs_fail(spec):
    with pytest.raises(ValueError):
        load_strategy(spec)


def test_factory_wrong_identity_rejected():
    registry = StrategyRegistry()
    registry.register("wrong", "1", lambda config: Constant())
    with pytest.raises(ValueError, match="different name"):
        registry.load({"name": "wrong", "version": "1", "config": {}})


def test_combination_clones_each_child_history():
    class Mutate(Constant):
        def target_weight(self, history):
            history.replace_column(0, pl.Series("close", [999.0]))
            return 0.0

    class Inspect(Constant):
        def target_weight(self, history):
            assert history["close"][0] == 1.0
            return 1.0

    frame = pl.DataFrame({"close": [1.0]})
    assert WeightedStrategy([Mutate(), Inspect()], [0.5, 0.5]).target_weight(frame) == 0.5
    assert frame["close"][0] == 1.0
