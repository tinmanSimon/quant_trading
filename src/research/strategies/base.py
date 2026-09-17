"""A strategy sees completed bars and requests a long-only target allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import json
import math
from numbers import Real

import polars as pl


class Strategy(ABC):
    """User strategies receive an isolated history ending at the decision bar.

    A target is a fraction of account equity, not an order or a fill. Strategies
    must be deterministic from their configuration and the supplied history.
    External data accessed by user code cannot be policed by the engine.
    """

    name: str
    version: str = "1"
    lookback: int

    @property
    @abstractmethod
    def config(self) -> dict:
        """Complete JSON configuration needed to reproduce this strategy."""

    @abstractmethod
    def target_weight(self, history: pl.DataFrame) -> float:
        """Return a finite target in [0, 1] using only completed history."""


def validate_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Strategy target weight must be a finite number in [0, 1].")
    weight = float(value)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Strategy target weight must be a finite number in [0, 1].")
    return weight


def strategy_spec(strategy: Strategy) -> dict:
    if not isinstance(strategy, Strategy):
        raise TypeError("strategy must be a Strategy instance.")
    for field in ("name", "version"):
        value = getattr(strategy, field, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Strategy {field} must be a nonempty string.")
    lookback = getattr(strategy, "lookback", None)
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Strategy lookback must be a positive integer.")
    config = deepcopy(strategy.config)
    if not isinstance(config, dict):
        raise TypeError("Strategy config must be a JSON object.")
    # Round-tripping detaches nested containers and rejects NaN/Infinity.
    config = json.loads(json.dumps(config, allow_nan=False, sort_keys=True))
    return {"name": strategy.name, "version": strategy.version, "config": config}


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value
