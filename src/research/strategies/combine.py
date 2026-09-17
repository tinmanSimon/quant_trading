"""Combine allocations before passing one target to one execution ledger."""

from copy import deepcopy
import math

import polars as pl

from .base import Strategy, strategy_spec, validate_weight


class WeightedStrategy(Strategy):
    name = "weighted"
    version = "1"

    def __init__(self, strategies, weights):
        children = list(strategies)
        values = list(weights)
        if not children or len(children) != len(values):
            raise ValueError("Provide equally sized, nonempty strategies and weights.")
        self.weights = tuple(validate_weight(value) for value in values)
        total = math.fsum(self.weights)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Strategy weights must sum to 1.")
        # Normalize only accepted floating point summation noise, ensuring <= 1.
        self.weights = tuple(value / total for value in self.weights)
        for child in children:
            strategy_spec(child)
        self.strategies = tuple(deepcopy(child) for child in children)
        self.lookback = max(child.lookback for child in self.strategies)

    @property
    def config(self):
        return {
            "strategies": [strategy_spec(child) for child in self.strategies],
            "weights": list(self.weights),
        }

    def target_weight(self, history: pl.DataFrame) -> float:
        if history.height < self.lookback:
            raise ValueError("Insufficient history for combined strategy.")
        values = [
            weight * validate_weight(child.target_weight(history.clone()))
            for child, weight in zip(self.strategies, self.weights)
        ]
        # fsum can exceed one by an ulp; each constituent was strictly validated.
        return min(1.0, math.fsum(values))
