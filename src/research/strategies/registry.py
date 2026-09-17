"""Versioned strategy construction from explicit JSON configuration."""

from copy import deepcopy
from typing import Callable

from .base import Strategy, strategy_spec
from .definitions import definition
from .combine import WeightedStrategy
from .momentum import Momentum
from .moving_average import MovingAverageCross


class StrategyRegistry:
    def __init__(self):
        self._factories: dict[tuple[str, str], Callable[[dict], Strategy]] = {}
        self._definitions = {}

    def register(self, name: str, version: str, factory: Callable[[dict], Strategy], *,
                 label=None, description="", default_config=None, parameters=None):
        if not all(isinstance(value, str) and value.strip() for value in (name, version)):
            raise ValueError("Strategy name and version must be nonempty strings.")
        if not callable(factory):
            raise TypeError("Strategy factory must be callable.")
        key = (name, version)
        if key in self._factories:
            raise ValueError(f"Strategy already registered: {key}.")
        entry = definition(name, version, label=label, description=description,
                           default_config=default_config, parameters=parameters)
        self._factories[key] = factory
        self._definitions[key] = entry

    def definitions(self):
        """Return detached descriptions; callers cannot mutate registry metadata."""
        return tuple(deepcopy(item) for item in self._definitions.values())

    def load(self, spec: dict) -> Strategy:
        if not isinstance(spec, dict) or set(spec) != {"name", "version", "config"}:
            raise ValueError("Strategy spec requires exactly name, version, and config.")
        if not isinstance(spec["config"], dict):
            raise ValueError("Strategy config must be an object.")
        if not all(isinstance(spec[key], str) and spec[key].strip() for key in ("name", "version")):
            raise ValueError("Strategy name and version must be nonempty strings.")
        key = (spec["name"], spec["version"])
        if key not in self._factories:
            raise ValueError(f"Unknown strategy: {key}.")
        config = self._definitions[key].validate_config(spec["config"])
        result = self._factories[key](config)
        actual = strategy_spec(result)
        if (actual["name"], actual["version"]) != key:
            raise ValueError("Strategy factory returned a different name or version.")
        return result


def builtin_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register("moving_average_cross", "1", lambda config: MovingAverageCross(**config),
                      label="Moving-average crossover", default_config={"fast": 10, "slow": 30},
                      parameters={"fast": {"type": "integer", "minimum": 1, "label": "Fast moving average (bars)"},
                                  "slow": {"type": "integer", "minimum": 2, "label": "Slow moving average (bars)"}})
    registry.register("momentum", "1", lambda config: Momentum(**config), label="Momentum",
                      default_config={"lookback": 20},
                      parameters={"lookback": {"type": "integer", "minimum": 1, "label": "Momentum lookback (bars)"}})

    def weighted(config):
        if set(config) != {"strategies", "weights"}:
            raise ValueError("Weighted configuration requires strategies and weights.")
        return WeightedStrategy([registry.load(spec) for spec in config["strategies"]], config["weights"])

    registry.register("weighted", "1", weighted, label="Weighted combination",
                      description="Nested strategy specifications and weights can also be edited as JSON.",
                      default_config={"strategies": [
                          {"name": "moving_average_cross", "version": "1", "config": {"fast": 10, "slow": 30}},
                          {"name": "momentum", "version": "1", "config": {"lookback": 20}},
                      ], "weights": [0.5, 0.5]})
    return registry


def load_strategy(spec: dict, *, registry: StrategyRegistry | None = None) -> Strategy:
    return (registry if registry is not None else builtin_registry()).load(spec)
