"""Serializable, versioned configuration for factor portfolio experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType
from collections.abc import Mapping


def _json_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Component config must use string keys at every level.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def json_config(value):
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("Component config must be an object with string keys.")
    try:
        return json.loads(json.dumps(_json_value(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Component config must contain finite JSON values.") from exc


def finite_number(value, name, *, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its supported range.")
    return float(value)


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _identity(name, version):
    if not all(isinstance(item, str) and item.strip() and item == item.strip()
               for item in (name, version)):
        raise ValueError("Component name and version must be nonempty strings without surrounding whitespace.")


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    name: str
    version: str = "1"
    config: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _identity(self.name, self.version)
        object.__setattr__(self, "config", _freeze(json_config(self.config)))

    def to_dict(self):
        return {"name": self.name, "version": self.version, "config": json_config(self.config)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or "name" not in value or set(value) - {"name", "version", "config"}:
            raise ValueError("Component specification requires name, optional version, and config.")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FactorSpec:
    name: str
    lookback: int = 20
    weight: float = 1.0
    direction: int = 1
    version: str = "1"
    config: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _identity(self.name, self.version)
        if self.name in {"timestamp", "symbol", "score"} or self.name.startswith("__"):
            raise ValueError("Factor name is reserved for panel bookkeeping.")
        positive_integer(self.lookback, "lookback")
        weight = finite_number(self.weight, "weight")
        if weight == 0:
            raise ValueError("Factor weight must be positive; remove unused factors instead.")
        if type(self.direction) is not int or self.direction not in {-1, 1}:
            raise ValueError("Factor direction must be 1 or -1.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "config", _freeze(json_config(self.config)))

    def to_dict(self):
        return {"name": self.name, "lookback": self.lookback, "weight": self.weight,
                "direction": self.direction, "version": self.version, "config": json_config(self.config)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or "name" not in value or set(value) - {
            "name", "lookback", "weight", "direction", "version", "config"
        }:
            raise ValueError("Invalid factor specification fields.")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FactorBacktestSpec:
    factors: tuple[FactorSpec, ...] = field(default_factory=lambda: (
        FactorSpec("momentum"), FactorSpec("volatility", direction=-1)))
    portfolio: ComponentSpec = field(default_factory=lambda: ComponentSpec("equal_weight", config={"top_n": 10}))
    schedule: ComponentSpec = field(default_factory=lambda: ComponentSpec("every_n_bars", config={"bars": 1}))
    cash_reserve: float = 0.05
    max_position_weight: float = 0.10
    min_weight_change: float = 0.0
    max_turnover: float = 1.0
    sale_proceeds: str = "same_open"
    funding: str = "pro_rata"

    def __post_init__(self):
        if not isinstance(self.factors, (tuple, list)) or not self.factors or not all(
            isinstance(factor, FactorSpec) for factor in self.factors
        ):
            raise ValueError("factors must be a nonempty sequence of FactorSpec objects.")
        if len({factor.name for factor in self.factors}) != len(self.factors):
            raise ValueError("Factor names must be unique in an experiment.")
        if not math.isfinite(sum(factor.weight for factor in self.factors)):
            raise ValueError("Total factor weight must be finite.")
        object.__setattr__(self, "factors", tuple(self.factors))
        if not isinstance(self.portfolio, ComponentSpec) or not isinstance(self.schedule, ComponentSpec):
            raise ValueError("portfolio and schedule must be ComponentSpec objects.")
        for name in ("cash_reserve", "max_position_weight", "min_weight_change"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name, maximum=1))
        if self.max_position_weight == 0:
            raise ValueError("max_position_weight must be positive.")
        object.__setattr__(self, "max_turnover", finite_number(self.max_turnover, "max_turnover", maximum=2))
        if not isinstance(self.sale_proceeds, str) or self.sale_proceeds not in {"same_open", "next_open"}:
            raise ValueError("sale_proceeds must be same_open or next_open.")
        if not isinstance(self.funding, str) or self.funding not in {"pro_rata", "score_priority"}:
            raise ValueError("funding must be pro_rata or score_priority.")

    @property
    def lookback(self):
        """Required observations, including the first return's starting price."""
        required = max(factor.lookback + 1 for factor in self.factors)
        if self.portfolio.name == "inverse_volatility":
            column = self.portfolio.config.get("volatility_factor", "volatility")
            if column not in {factor.name for factor in self.factors}:
                required = max(required, positive_integer(
                    self.portfolio.config.get("volatility_lookback", 20), "volatility_lookback") + 1)
        return required

    def to_dict(self):
        return {"factors": [factor.to_dict() for factor in self.factors],
                "portfolio": self.portfolio.to_dict(), "schedule": self.schedule.to_dict(),
                **{name: getattr(self, name) for name in (
                    "cash_reserve", "max_position_weight", "min_weight_change", "max_turnover", "sale_proceeds", "funding")}}

    @classmethod
    def from_dict(cls, value):
        fields = {"factors", "portfolio", "schedule", "cash_reserve", "max_position_weight",
                  "min_weight_change", "max_turnover", "sale_proceeds", "funding"}
        if not isinstance(value, dict) or set(value) - fields:
            raise ValueError("Invalid factor backtest specification fields.")
        kwargs = dict(value)
        if "factors" in kwargs:
            if not isinstance(kwargs["factors"], (tuple, list)):
                raise ValueError("factors must be a list.")
            kwargs["factors"] = tuple(FactorSpec.from_dict(factor) for factor in kwargs["factors"])
        for name in ("portfolio", "schedule"):
            if name in kwargs:
                kwargs[name] = ComponentSpec.from_dict(kwargs[name])
        return cls(**kwargs)
