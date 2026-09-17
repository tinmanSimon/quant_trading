"""Long when price has increased over the configured number of bars."""

import polars as pl

from .base import Strategy, positive_integer


class Momentum(Strategy):
    name = "momentum"
    version = "1"

    def __init__(self, lookback: int = 20):
        self.period = positive_integer(lookback, "lookback")
        # N-bar change needs both endpoints: N + 1 observations.
        self.lookback = self.period + 1

    @property
    def config(self):
        return {"lookback": self.period}

    def target_weight(self, history: pl.DataFrame) -> float:
        if history.height < self.lookback:
            raise ValueError("Insufficient history for momentum strategy.")
        close = history["close"]
        return float(close[-1] > close[-self.lookback])
