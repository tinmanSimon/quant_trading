"""Simple causal moving-average crossover."""

import polars as pl

from .base import Strategy, positive_integer


class MovingAverageCross(Strategy):
    name = "moving_average_cross"
    version = "1"

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = positive_integer(fast, "fast")
        self.slow = positive_integer(slow, "slow")
        if fast >= slow:
            raise ValueError("fast must be smaller than slow.")
        self.lookback = slow

    @property
    def config(self):
        return {"fast": self.fast, "slow": self.slow}

    def target_weight(self, history: pl.DataFrame) -> float:
        if history.height < self.lookback:
            raise ValueError("Insufficient history for moving-average strategy.")
        close = history["close"]
        return float(close.tail(self.fast).mean() > close.tail(self.slow).mean())
