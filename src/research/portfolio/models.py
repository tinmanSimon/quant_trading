"""Portfolio decisions express complete desired holdings, never fills."""

from dataclasses import dataclass, field
import math
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType


def validated_weights(value):
    if not isinstance(value, Mapping):
        raise ValueError("Portfolio weights must be a symbol-to-weight mapping.")
    weights = {}
    for symbol, weight in value.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Portfolio symbols must be nonempty strings.")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Portfolio weights must be finite numbers in [0, 1].")
        weights[symbol] = float(weight)
    weights = {symbol: weights[symbol] for symbol in sorted(weights)}
    if sum(weights.values()) > 1 + 1e-10:
        raise ValueError("Long-only portfolio weights cannot total more than one.")
    return weights


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Complete target: omitted holdings are sold; ``no_change`` does nothing.

    Frozen positions preserve quantities in the execution engine. Their weights
    here are the decision-time marks, not orders to resize them at the next open.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    proposed_weights: Mapping[str, float] = field(default_factory=dict)
    frozen_symbols: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)
    no_change: bool = False

    def __post_init__(self):
        if type(self.no_change) is not bool:
            raise ValueError("no_change must be a boolean.")
        if not isinstance(self.frozen_symbols, (list, tuple)) or any(
            not isinstance(symbol, str) or not symbol for symbol in self.frozen_symbols
        ):
            raise ValueError("frozen_symbols must be a sequence of symbols.")
        if not isinstance(self.reasons, Mapping) or any(
            not isinstance(key, str) or not isinstance(reason, str) for key, reason in self.reasons.items()
        ):
            raise ValueError("Portfolio decision reasons must map strings to strings.")
        object.__setattr__(self, "weights", MappingProxyType(validated_weights(self.weights)))
        object.__setattr__(self, "proposed_weights", MappingProxyType(validated_weights(self.proposed_weights)))
        object.__setattr__(self, "frozen_symbols", tuple(sorted(set(self.frozen_symbols))))
        object.__setattr__(self, "reasons", MappingProxyType({symbol: self.reasons[symbol] for symbol in sorted(self.reasons)}))

    def to_dict(self):
        return {"weights": dict(self.weights), "proposed_weights": dict(self.proposed_weights),
                "frozen_symbols": list(self.frozen_symbols), "reasons": dict(self.reasons), "no_change": self.no_change}
