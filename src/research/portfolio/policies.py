"""Interchangeable selection/weighting policies and shared allocation limits."""

from __future__ import annotations

from types import MappingProxyType

import polars as pl

from research.factors.models import ComponentSpec, FactorBacktestSpec
from research.factors.registry import _Registry
from .models import TargetPortfolio, validated_weights


class _RankedPolicy:
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def propose(self, scores, snapshot):
        ordered = scores.filter(pl.col("score").is_not_null() & pl.col("score").is_finite()).sort(
            ["score", "symbol"], descending=[True, False])
        selected = ordered.head(self.config["top_n"])
        reasons = {symbol: "outside selection" for symbol in sorted(set(scores["symbol"]) | set(snapshot))}
        if not selected.height:
            return TargetPortfolio(reasons=reasons)
        if self.name == "equal_weight":
            values = [1.0] * selected.height
        elif self.name == "score_weight":
            values = selected["score"].to_list()
            if any(value <= 0 for value in values):
                raise ValueError("Score weighting requires strictly positive selected scores.")
        else:
            column = self.config["volatility_factor"]
            if column not in selected.columns:
                raise ValueError(f"Inverse volatility requires factor column {column}.")
            selected = selected.filter(pl.col(column).is_not_null() & pl.col(column).is_finite() & (pl.col(column) > 0))
            # Relative inverse weights avoid overflow for tiny finite values.
            minimum = selected[column].min()
            values = [minimum / value for value in selected[column]]
        total = sum(values)
        weights = {symbol: value / total for symbol, value in zip(selected["symbol"], values)} if total else {}
        reasons.update({symbol: "selected by score rank" for symbol in weights})
        return TargetPortfolio(weights=weights, reasons=reasons)


class PortfolioRegistry(_Registry):
    """Register ``factory(config) -> policy.propose(scores, snapshot)``.

    Construct one fresh policy with ``load`` at run start, then pass it to every
    ``build_targets(..., policy=policy)`` call. This supports stateful holding
    rules without sharing state between independent backtests.
    """

    kind = "portfolio policy"

    def __init__(self, *, include_builtins=True):
        super().__init__()
        if include_builtins:
            for name, label in (("equal_weight", "Top-N equal weight"), ("score_weight", "Top-N score weight"),
                                ("inverse_volatility", "Top-N inverse volatility")):
                defaults = {"top_n": 10}
                parameters = {"top_n": {"type": "integer", "minimum": 1}}
                if name == "inverse_volatility":
                    defaults["volatility_factor"] = "volatility"
                    defaults["volatility_lookback"] = 20
                    parameters["volatility_factor"] = {"type": "string"}
                    parameters["volatility_lookback"] = {"type": "integer", "minimum": 1}
                self.register(name, "1", lambda config, name=name: _RankedPolicy(name, config), label=label,
                              default_config=defaults, parameters=parameters)

    def load(self, spec):
        if isinstance(spec, dict):
            spec = ComponentSpec.from_dict(spec)
        if not isinstance(spec, ComponentSpec):
            raise ValueError("Portfolio registry requires a ComponentSpec.")
        factory, config = self._resolve(spec)
        if spec.name == "inverse_volatility":
            column = config["volatility_factor"]
            if not column.strip() or column in {"timestamp", "symbol", "score"} or column.startswith("__"):
                raise ValueError("volatility_factor must be a nonreserved factor column name.")
        policy = factory(config)
        if not callable(getattr(policy, "propose", None)):
            raise TypeError("Portfolio factory must return a policy with propose(scores, snapshot).")
        return policy


def _capped_allocation(relative, budget, cap):
    """Redistribute after caps without ever exceeding either cap or budget."""
    remaining = {symbol: relative[symbol] for symbol in sorted(relative) if relative[symbol] > 0}
    result = {symbol: 0.0 for symbol in sorted(relative)}
    while remaining and budget > 1e-14:
        total = sum(remaining.values())
        scale = budget / total
        capped = [symbol for symbol, weight in remaining.items() if weight * scale >= cap]
        if not capped:
            result.update({symbol: weight * scale for symbol, weight in remaining.items()})
            break
        for symbol in capped:
            result[symbol] = cap
            budget -= cap
            del remaining[symbol]
    return result


def build_targets(scores, snapshot, spec: FactorBacktestSpec, registry=None, *, policy=None):
    """Apply one policy and shared constraints using immutable current weights.

    ``max_turnover`` is gross traded weight (sum of absolute position changes).
    Hard caps/cash constraints take priority when an existing holding drifts
    beyond them; mandatory reductions are explicitly identified in reasons.
    Missing-score holdings are frozen, even when that prevents a cash reserve.
    """
    if not isinstance(spec, FactorBacktestSpec):
        raise ValueError("spec must be a FactorBacktestSpec.")
    current = validated_weights(snapshot)
    if not isinstance(scores, pl.DataFrame) or not {"symbol", "score"}.issubset(scores.columns):
        raise ValueError("Portfolio scores require symbol and score columns.")
    if scores["symbol"].null_count() or scores["symbol"].n_unique() != scores.height:
        raise ValueError("Portfolio scores must have one row per symbol.")
    finite_scores = scores.filter(pl.col("score").is_not_null() & pl.col("score").is_finite()).sort("symbol")
    scored_symbols = set(finite_scores["symbol"])
    if spec.portfolio.name == "inverse_volatility":
        column = spec.portfolio.config.get("volatility_factor", "volatility")
        if column not in finite_scores.columns:
            raise ValueError(f"Inverse volatility requires factor column {column}.")
        finite_scores = finite_scores.filter(pl.col(column).is_not_null() & pl.col(column).is_finite() & (pl.col(column) > 0))
    valid = set(finite_scores["symbol"])
    known_set = set(scores["symbol"]) | set(current)
    known = tuple(sorted(known_set))
    if policy is None:
        policy = (registry if registry is not None else PortfolioRegistry()).load(spec.portfolio)
    proposal = policy.propose(finite_scores.clone(), MappingProxyType(dict(current)))
    if not isinstance(proposal, TargetPortfolio):
        raise ValueError("Portfolio policy must return TargetPortfolio.")
    if set(proposal.weights) - known_set or set(proposal.frozen_symbols) - set(current):
        raise ValueError("Portfolio policy returned unknown or unheld symbols.")
    if proposal.no_change:
        return TargetPortfolio(weights=current, proposed_weights=current, reasons=proposal.reasons, no_change=True)
    frozen_set = {symbol for symbol, weight in current.items() if weight > 0 and symbol not in valid} | set(proposal.frozen_symbols)
    frozen = tuple(sorted(frozen_set))
    unfrozen = tuple(symbol for symbol in known if symbol not in frozen_set)
    reasons = {symbol: proposal.reasons.get(symbol, "not selected") for symbol in known}
    for symbol in frozen:
        if symbol not in scored_symbols:
            reasons[symbol] = "frozen: missing factor score"
        elif symbol not in valid:
            reasons[symbol] = "frozen: missing or nonpositive sizing factor"
        else:
            reasons[symbol] = "frozen by portfolio policy"
    frozen_weight = sum(current[symbol] for symbol in frozen)
    budget = max(0.0, 1 - spec.cash_reserve - frozen_weight)
    relative = {symbol: proposal.weights.get(symbol, 0.0) if symbol in valid else 0.0 for symbol in unfrozen}
    # Preserve intentional cash in a policy's complete allocation: allocating
    # only 50% never silently normalizes that policy into a 100% investment.
    proposed_total = sum(relative.values())
    usable_budget = min(budget, proposed_total)
    desired = _capped_allocation(relative, usable_budget, spec.max_position_weight)
    desired.update({symbol: current[symbol] for symbol in frozen})
    reserve_scaled = {symbol: 0.0 for symbol in known}
    if proposed_total:
        reserve_scaled.update({symbol: weight / proposed_total * usable_budget for symbol, weight in relative.items()})
    reserve_scaled.update({symbol: current[symbol] for symbol in frozen})
    raw_proposed = {symbol: proposal.weights.get(symbol, 0.0) for symbol in known}
    for symbol in relative:
        if reserve_scaled[symbol] + 1e-12 < raw_proposed[symbol]:
            reasons[symbol] += "; reduced for cash reserve/frozen holdings"
        if desired[symbol] + 1e-12 < reserve_scaled[symbol]:
            reasons[symbol] += "; reduced by position cap"
    if frozen_weight > 1 - spec.cash_reserve + 1e-12:
        for symbol in frozen:
            reasons[symbol] += "; frozen holdings prevent cash reserve"

    # Construct a feasible baseline before considering discretionary turnover.
    baseline = {symbol: current.get(symbol, 0.0) if symbol in frozen else min(
        current.get(symbol, 0.0), spec.max_position_weight) for symbol in known}
    baseline_nonfrozen = sum(baseline[symbol] for symbol in unfrozen)
    if baseline_nonfrozen > budget and baseline_nonfrozen:
        for symbol in unfrozen:
            baseline[symbol] *= budget / baseline_nonfrozen
    mandatory = sum(abs(baseline[symbol] - current.get(symbol, 0.0)) for symbol in known)
    for symbol in known:
        if abs(baseline[symbol] - current.get(symbol, 0.0)) > 1e-12:
            reasons[symbol] += "; mandatory reduction for position cap/cash reserve"
            if mandatory > spec.max_turnover + 1e-12:
                reasons[symbol] += "; hard constraints exceed turnover budget"
        if (symbol not in frozen and desired[symbol] > 0 and
                abs(desired[symbol] - baseline[symbol]) < spec.min_weight_change):
            desired[symbol] = baseline[symbol]
            reasons[symbol] += "; retained within weight-change tolerance"
    # Tolerances may suppress a sale that would have funded another increase.
    reductions = sum(max(0, baseline[symbol] - desired[symbol]) for symbol in known)
    increases = sum(max(0, desired[symbol] - baseline[symbol]) for symbol in known)
    available = max(0.0, 1 - spec.cash_reserve - sum(baseline.values())) + reductions
    if increases > available + 1e-12:
        for symbol in known:
            if desired[symbol] > baseline[symbol]:
                desired[symbol] = baseline[symbol] + (desired[symbol] - baseline[symbol]) * available / increases
                reasons[symbol] += "; reduced to preserve cash reserve"
    discretionary = sum(abs(desired[symbol] - baseline[symbol]) for symbol in known)
    remaining = max(0.0, spec.max_turnover - mandatory)
    if discretionary > remaining + 1e-12:
        fraction = remaining / discretionary
        for symbol in known:
            if abs(desired[symbol] - baseline[symbol]) > 1e-12:
                desired[symbol] = baseline[symbol] + fraction * (desired[symbol] - baseline[symbol])
                reasons[symbol] += "; limited by turnover budget"
    return TargetPortfolio(weights={symbol: desired[symbol] for symbol in known},
                           proposed_weights=raw_proposed, frozen_symbols=frozen, reasons=reasons)
