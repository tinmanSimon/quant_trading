"""Factor factories are independent of portfolio selection and execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import polars as pl

from .models import FactorSpec, json_config, finite_number


@dataclass(frozen=True)
class ComponentDefinition:
    name: str
    version: str
    label: str
    description: str
    default_config: dict
    parameters: dict

    def validate_config(self, config):
        config = json_config(config)
        if set(config) - set(self.parameters):
            raise ValueError(f"Unknown configuration for {self.name}: {sorted(set(config) - set(self.parameters))}.")
        merged = {**deepcopy(self.default_config), **config}
        for name, parameter in self.parameters.items():
            if name not in merged:
                raise ValueError(f"Missing configuration {name} for {self.name}.")
            value = merged[name]
            kind = parameter.get("type")
            if kind == "integer":
                if type(value) is not int:
                    raise ValueError(f"{name} must be an integer.")
            elif kind == "number":
                finite_number(value, name, minimum=parameter.get("minimum", float("-inf")))
            elif kind == "string":
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string.")
            elif kind == "boolean":
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean.")
            elif kind != "json":
                raise ValueError(f"Unsupported parameter type for {name}.")
            for bound, compare in (("minimum", lambda x, y: x < y), ("maximum", lambda x, y: x > y)):
                if bound in parameter and compare(value, parameter[bound]):
                    raise ValueError(f"{name} violates its {bound}.")
            if "choices" in parameter and not any(type(value) is type(choice) and value == choice for choice in parameter["choices"]):
                raise ValueError(f"{name} must be one of its registered choices.")
        return merged


class _Registry:
    kind = "component"

    def __init__(self):
        self._factories = {}
        self._definitions = {}

    def register(self, name, version, factory, *, label=None, description="", default_config=None, parameters=None):
        from .models import ComponentSpec
        ComponentSpec(name, version)
        if not callable(factory):
            raise TypeError(f"{self.kind} factory must be callable.")
        key = (name, version)
        if key in self._factories:
            raise ValueError(f"{self.kind} already registered: {key}.")
        definition = ComponentDefinition(name, version, label or name, description,
                                         json_config(default_config or {}), json_config(parameters or {}))
        definition.validate_config(definition.default_config)
        self._factories[key] = factory
        self._definitions[key] = definition

    def definitions(self):
        return tuple(deepcopy(item) for item in self._definitions.values())

    def _resolve(self, spec):
        key = (spec.name, spec.version)
        if key not in self._factories:
            raise ValueError(f"Unknown {self.kind}: {key}.")
        return self._factories[key], self._definitions[key].validate_config(spec.config)


class _PriceFactor:
    def __init__(self, spec):
        self.spec = spec

    def calculate(self, frame):
        lookback = self.spec.lookback
        close = pl.col("close").cast(pl.Float64)
        if self.spec.name == "momentum":
            value = close / close.shift(lookback) - 1
        elif self.spec.name == "volatility":
            returns = close / close.shift(1) - 1
            value = returns.rolling_std(lookback, min_samples=lookback, ddof=0)
        else:
            red = pl.col("close") < pl.col("open")
            count = red.cast(pl.Float64).rolling_sum(lookback, min_samples=lookback)
            total = pl.when(red).then((pl.col("high") - pl.col("open")) / pl.col("high")).otherwise(0).rolling_sum(
                lookback, min_samples=lookback)
            value = pl.when(count > 0).then(total / count).otherwise(None)
        return frame.select(value.alias(self.spec.name)).to_series()


class FactorRegistry(_Registry):
    """Register ``factory(FactorSpec) -> object.calculate(frame) -> Series``.

    Factories must create a fresh instance. Calculations must be causal: output
    at a timestamp may depend only on that timestamp and earlier completed bars.
    External/custom code is trusted; the engine cannot enforce its causality.
    """

    kind = "factor"

    def __init__(self, *, include_builtins=True):
        super().__init__()
        if include_builtins:
            for name, label, description in (
                ("momentum", "Momentum", "Close-to-close return over lookback bars."),
                ("volatility", "Volatility", "Population standard deviation of lookback one-bar returns."),
                ("red_bar_high_percentage", "Red-bar high percentage", "Mean (high-open)/high over red bars in the lookback window."),
            ):
                self.register(name, "1", _PriceFactor, label=label, description=description)

    def load(self, spec):
        if isinstance(spec, dict):
            spec = FactorSpec.from_dict(spec)
        if not isinstance(spec, FactorSpec):
            raise ValueError("Factor registry requires a FactorSpec.")
        factory, config = self._resolve(spec)
        resolved = FactorSpec(spec.name, spec.lookback, spec.weight, spec.direction, spec.version, config)
        result = factory(resolved)
        if not callable(getattr(result, "calculate", None)):
            raise TypeError("Factor factory must return an object with calculate(frame).")
        return result
