from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from research.factors import ComponentSpec, FactorSpec, FactorBacktestSpec, FactorRegistry, build_factor_panel, score_factor_panel
from research.portfolio import PortfolioRegistry, TargetPortfolio, build_targets, should_rebalance, validate_schedule


def bars(closes):
    return pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(len(closes))],
                         "open": [10.0] * len(closes), "high": [12.0] * len(closes), "close": closes})


def specification(**kwargs):
    defaults = dict(factors=(FactorSpec("momentum", lookback=2),), cash_reserve=0,
                    max_position_weight=1, max_turnover=2,
                    portfolio=ComponentSpec("equal_weight", config={"top_n": 2}))
    defaults.update(kwargs)
    return FactorBacktestSpec(**defaults)


def scores(**values):
    return pl.DataFrame({"symbol": list(values), "score": list(values.values())}, schema={"symbol": pl.String, "score": pl.Float64})


def test_spec_roundtrip_and_detached_nested_configuration():
    config = {"nested": [1, {"other": 2}]}
    component = ComponentSpec("custom", config=config)
    config["nested"][1]["other"] = 10
    assert component.to_dict()["config"]["nested"][1]["other"] == 2
    with pytest.raises(TypeError):
        component.config["nested"][1]["other"] = 3
    spec = FactorBacktestSpec()
    assert FactorBacktestSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert spec.lookback == 21


@pytest.mark.parametrize("kwargs", [{"cash_reserve": True}, {"max_turnover": math.nan},
                                     {"max_position_weight": 0}, {"funding": "random"},
                                     {"sale_proceeds": "immediately"}, {"min_weight_change": -1}])
def test_invalid_experiment_configuration(kwargs):
    with pytest.raises(ValueError):
        specification(**kwargs)


def test_non_json_nested_keys_and_overflowing_weight_sum_are_rejected():
    with pytest.raises(ValueError):
        ComponentSpec("custom", config={"nested": {1: "bad"}})
    with pytest.raises(ValueError):
        specification(factors=(FactorSpec("momentum", weight=1e308), FactorSpec("volatility", weight=1e308)))


@pytest.mark.parametrize("kwargs", [{"lookback": True}, {"direction": True}, {"direction": 0},
                                     {"weight": 0}, {"weight": math.inf}, {"name": "score"}])
def test_invalid_factor_configuration(kwargs):
    with pytest.raises(ValueError):
        FactorSpec(**{"name": "momentum", **kwargs})


def test_unknown_versions_and_configs_are_rejected():
    with pytest.raises(ValueError, match="Unknown factor"):
        FactorRegistry().load(FactorSpec("momentum", version="2"))
    with pytest.raises(ValueError, match="Unknown configuration"):
        FactorRegistry().load(FactorSpec("momentum", config={"typo": 1}))
    with pytest.raises(ValueError, match="Unknown portfolio"):
        PortfolioRegistry().load(ComponentSpec("equal_weight", version="2"))
    with pytest.raises(ValueError):
        PortfolioRegistry().load(ComponentSpec("equal_weight", config={"top_n": True}))
    with pytest.raises(ValueError):
        validate_schedule(ComponentSpec("weekly", config={"bars": 2}))
    with pytest.raises(ValueError):
        FactorBacktestSpec.from_dict({"typo": True})


def test_panel_is_causal_ranked_only_within_timestamp_and_handles_missing_values():
    spec = specification()
    frames = {"B": bars([10., 11., 11., 11.]), "A": bars([10., 11., 12., 20.])}
    result = build_factor_panel(frames, spec)
    assert result.filter(pl.col("timestamp") == frames["A"]["timestamp"][1])["score"].null_count() == 2
    day = result.filter(pl.col("timestamp") == frames["A"]["timestamp"][2])
    assert day["symbol"].to_list() == ["A", "B"]
    assert day["score"].to_list() == [1., .5]
    changed = {**frames, "A": bars([10., 11., 12., 10000.])}
    assert result.head(6).equals(build_factor_panel(changed, spec).head(6))
    assert result.equals(build_factor_panel(dict(reversed(list(frames.items()))), spec))


def test_factor_directions_and_tied_scores():
    frames = {"A": bars([10., 11., 12.]), "B": bars([10., 11., 12.]), "C": bars([10., 9., 8.])}
    result = build_factor_panel(frames, specification(factors=(FactorSpec("momentum", 2, direction=-1),))).tail(3)
    assert result["score"].to_list() == [.5, .5, 1.]


def test_score_helper_reranks_eligible_rows_without_recalculating_features():
    spec = specification(factors=(FactorSpec("momentum", 1),), portfolio=ComponentSpec(
        "inverse_volatility", config={"volatility_lookback": 3}))
    frames = {"A": bars([10., 11., 12., 15.]), "B": bars([10., 9., 8., 10.]), "C": bars([10., 11., 12., 100.])}
    original = build_factor_panel(frames, spec).tail(3)
    eligible = original.filter(pl.col("symbol") != "C")
    result = score_factor_panel(eligible, spec)
    assert result["score"].to_list() == [.75, .75]
    assert original.filter(pl.col("symbol") == "A")["score"].item() == .5
    assert result.drop("score").equals(eligible.drop("score"))


def test_score_helper_handles_empty_and_nonfinite_factor_rows():
    spec = specification()
    panel = build_factor_panel({"A": bars([10., 11., 12.]), "B": bars([10., 11., 13.])}, spec).tail(2)
    invalid = panel.with_columns(pl.when(pl.col("symbol") == "B").then(float("nan")).otherwise(pl.col("momentum")).alias("momentum"))
    assert score_factor_panel(invalid, spec)["score"].to_list() == [1., None]
    empty = score_factor_panel(panel.head(0), spec)
    assert empty.height == 0 and empty.schema["score"] == pl.Float64
    with pytest.raises(ValueError, match="unique"):
        score_factor_panel(pl.concat([panel, panel]), spec)


def test_volatility_and_red_bar_factor_definitions():
    frame = bars([10., 11., 9., 10.])
    vol = FactorRegistry().load(FactorSpec("volatility", 2)).calculate(frame)
    assert vol[:2].null_count() == 2
    assert vol[2] == pytest.approx(abs(.1 - (9 / 11 - 1)) / 2)
    red = FactorRegistry().load(FactorSpec("red_bar_high_percentage", 2)).calculate(frame)
    assert red[1] is None
    assert red[2] == pytest.approx(2 / 12)
    assert red[3] == pytest.approx(2 / 12)


def test_incomplete_factor_rows_are_excluded_from_all_factor_ranks():
    spec = specification(factors=(FactorSpec("momentum", 2), FactorSpec("red_bar_high_percentage", 2)))
    result = build_factor_panel({"A": bars([10., 11., 12.]), "B": bars([10., 11., 9.])}, spec).tail(2)
    assert result["score"].to_list() == [None, 1.]


def test_registry_supports_custom_factor_config_and_rejects_bad_output():
    class Custom:
        def __init__(self, spec):
            self.spec = spec

        def calculate(self, frame):
            return frame["close"] * self.spec.config["scale"]

    registry = FactorRegistry()
    registry.register("scaled_close", "2", Custom, default_config={"scale": 2.0},
                      parameters={"scale": {"type": "number", "minimum": 0}})
    spec = specification(factors=(FactorSpec("scaled_close", 1, version="2"),))
    result = build_factor_panel({"A": bars([10., 11.])}, spec, registry)
    assert result["scaled_close"].to_list() == [20., 22.]
    assert result["score"].to_list() == [1., 1.]


def test_policy_swap_changes_weights_without_changing_selection():
    equal = build_targets(scores(A=1., B=.5, C=.1), {}, specification())
    weighted = build_targets(scores(A=1., B=.5, C=.1), {}, specification(
        portfolio=ComponentSpec("score_weight", config={"top_n": 2})))
    assert dict(equal.weights) == {"A": .5, "B": .5, "C": 0.}
    assert weighted.weights["A"] == pytest.approx(2 / 3)
    assert weighted.weights["B"] == pytest.approx(1 / 3)
    assert weighted.weights["C"] == 0


def test_position_caps_redistribute_but_never_exceed_cap_or_reserve():
    spec = specification(portfolio=ComponentSpec("score_weight", config={"top_n": 3}),
                         max_position_weight=.4, cash_reserve=.1)
    target = build_targets(scores(A=9., B=1., C=1.), {}, spec)
    assert dict(target.weights) == pytest.approx({"A": .4, "B": .25, "C": .25})
    sparse = build_targets(scores(A=1.), {}, spec)
    assert sparse.weights["A"] == .4


def test_missing_score_holding_is_frozen_and_does_not_dilute_its_weight():
    target = build_targets(scores(A=None, B=1., C=.5), {"A": .4}, specification(cash_reserve=.1))
    assert target.frozen_symbols == ("A",)
    assert dict(target.weights) == pytest.approx({"A": .4, "B": .25, "C": .25})
    assert sum(target.weights.values()) == pytest.approx(.9)
    scarce = build_targets(scores(A=None, B=1.), {"A": .95}, specification(cash_reserve=.1))
    assert scarce.weights["A"] == .95
    assert scarce.weights["B"] == 0
    assert "prevent cash reserve" in scarce.reasons["A"]


def test_turnover_budget_and_zero_turnover_are_explicit():
    target = build_targets(scores(A=.1, B=1.), {"A": 1.}, specification(
        portfolio=ComponentSpec("equal_weight", config={"top_n": 1}), max_turnover=.5))
    assert dict(target.weights) == {"A": .75, "B": .25}
    unchanged = build_targets(scores(A=.1, B=1.), {"A": 1.}, specification(max_turnover=0))
    assert dict(unchanged.weights) == {"A": 1., "B": 0.}
    assert "turnover budget" in unchanged.reasons["A"]


def test_hard_constraint_reductions_report_turnover_override():
    target = build_targets(scores(A=1., B=.5), {"A": .8}, specification(max_position_weight=.4, max_turnover=0))
    assert dict(target.weights) == {"A": .4, "B": 0.}
    assert "hard constraints exceed turnover budget" in target.reasons["A"]


def test_weight_change_tolerance_never_prevents_full_liquidation():
    target = build_targets(scores(A=.1, B=1.), {"A": .01, "B": .99}, specification(
        portfolio=ComponentSpec("equal_weight", config={"top_n": 1}), min_weight_change=.05))
    assert target.weights["A"] == 0
    assert target.weights["B"] == .99


def test_custom_policy_no_change_liquidation_and_intentional_cash_are_distinct():
    class Custom:
        def __init__(self, decision):
            self.decision = decision

        def propose(self, scores, snapshot):
            with pytest.raises(TypeError):
                snapshot["A"] = 0
            return self.decision

    current = {"A": .5}
    unchanged = build_targets(scores(A=1.), current, specification(), policy=Custom(TargetPortfolio(no_change=True)))
    liquidated = build_targets(scores(A=1.), current, specification(), policy=Custom(TargetPortfolio()))
    cash = build_targets(scores(A=1.), {}, specification(), policy=Custom(TargetPortfolio(weights={"A": .25})))
    reserved_cash = build_targets(scores(A=1.), {}, specification(cash_reserve=.05),
                                  policy=Custom(TargetPortfolio(weights={"A": .5})))
    assert unchanged.no_change and unchanged.weights["A"] == .5
    assert not liquidated.no_change and liquidated.weights["A"] == 0
    assert cash.weights["A"] == .25
    assert reserved_cash.weights["A"] == .5


def test_proposal_audit_retains_original_targets_before_constraints():
    target = build_targets(scores(A=1.), {}, specification(cash_reserve=.05, max_position_weight=.8))
    assert target.proposed_weights["A"] == 1.
    assert target.weights["A"] == .8
    assert "cash reserve" in target.reasons["A"]
    assert "position cap" in target.reasons["A"]


def test_stateful_policy_instances_are_per_run():
    class Stateful:
        def __init__(self, config):
            self.calls = 0

        def propose(self, scores, snapshot):
            self.calls += 1
            return TargetPortfolio(weights={"A": .2 * self.calls})

    registry = PortfolioRegistry()
    registry.register("stateful", "1", Stateful)
    spec = specification(portfolio=ComponentSpec("stateful"))
    first = registry.load(spec.portfolio)
    assert build_targets(scores(A=1.), {}, spec, policy=first).weights["A"] == .2
    assert build_targets(scores(A=1.), {}, spec, policy=first).weights["A"] == .4
    second = registry.load(spec.portfolio)
    assert build_targets(scores(A=1.), {}, spec, policy=second).weights["A"] == .2


def test_inverse_volatility_policy_weights():
    data = scores(A=1., B=.5).with_columns(pl.Series("volatility", [.1, .2]))
    target = build_targets(data, {}, specification(portfolio=ComponentSpec("inverse_volatility", config={"top_n": 2})))
    assert target.weights["A"] == pytest.approx(2 / 3)
    assert target.weights["B"] == pytest.approx(1 / 3)


def test_inverse_volatility_auxiliary_feature_preserves_scoring_and_extends_warmup():
    frames = {"A": bars([10., 11., 12., 10.]), "B": bars([10., 9., 9., 11.])}
    equal = specification(factors=(FactorSpec("momentum", 1),))
    inverse = specification(factors=equal.factors, portfolio=ComponentSpec("inverse_volatility", config={
        "top_n": 2, "volatility_lookback": 3}))
    assert equal.lookback == 2
    assert inverse.lookback == 4
    first = build_factor_panel(frames, equal)
    second = build_factor_panel(frames, inverse)
    assert first.equals(second.drop("volatility"))
    assert second["volatility"].null_count() == 6
    assert second.tail(2)["volatility"].is_finite().all()


def test_unavailable_sizing_factor_freezes_holding_and_selects_eligible_candidates():
    data = scores(A=1., B=.8, C=.5).with_columns(pl.Series("volatility", [None, .1, .2]))
    target = build_targets(data, {"A": .4}, specification(portfolio=ComponentSpec("inverse_volatility", config={"top_n": 1})))
    assert target.frozen_symbols == ("A",)
    assert target.weights == {"A": .4, "B": .6, "C": 0.}
    assert "sizing factor" in target.reasons["A"]


@pytest.mark.parametrize("name,current,following,expected", [
    ("daily", "2024-11-29T18:00:00+00:00", "2024-12-02T15:00:00+00:00", True),
    ("daily", "2024-11-29T17:00:00+00:00", "2024-11-29T18:00:00+00:00", False),
    ("weekly", "2024-07-03T17:00:00+00:00", "2024-07-05T20:00:00+00:00", False),
    ("weekly", "2024-07-05T20:00:00+00:00", "2024-07-08T20:00:00+00:00", True),
    ("monthly", "2024-05-31T20:00:00+00:00", "2024-06-03T20:00:00+00:00", True),
    ("monthly", "2024-05-30T20:00:00+00:00", "2024-05-31T20:00:00+00:00", False),
])
def test_calendar_rebalance_schedules(name, current, following, expected):
    assert should_rebalance(ComponentSpec(name), index=0, closed_at=datetime.fromisoformat(current),
                            next_closed_at=datetime.fromisoformat(following)) is expected


def test_every_n_bars_schedule_and_invalid_boundaries():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rule = ComponentSpec("every_n_bars", config={"bars": 3})
    assert [should_rebalance(rule, index=i, closed_at=now, next_closed_at=None) for i in range(5)] == [True, False, False, True, False]
    with pytest.raises(ValueError):
        should_rebalance(rule, index=0, closed_at=now.replace(tzinfo=None), next_closed_at=None)
    with pytest.raises(ValueError):
        should_rebalance(rule, index=0, closed_at=now, next_closed_at=now)


@pytest.mark.parametrize("name", ["daily", "weekly", "monthly"])
def test_calendar_schedule_does_not_treat_truncated_data_as_period_end(name):
    now = datetime(2024, 1, 2, 15, tzinfo=timezone.utc)
    assert not should_rebalance(ComponentSpec(name), index=0, closed_at=now, next_closed_at=None)


def test_equal_scores_tie_break_by_symbol_not_input_order():
    spec = specification(portfolio=ComponentSpec("equal_weight", config={"top_n": 1}))
    one = build_targets(scores(B=1., A=1.), {}, spec)
    two = build_targets(scores(A=1., B=1.), {}, spec)
    assert dict(one.weights) == dict(two.weights) == {"A": 1., "B": 0.}


def test_portfolio_arithmetic_is_identical_across_python_hash_seeds():
    script = """
import json
import polars as pl
from research.factors import ComponentSpec, FactorBacktestSpec
from research.portfolio import build_targets
from research.portfolio.policies import _capped_allocation

symbols = set(f'SYM{i:03}' for i in range(40))
rows = [{'symbol': symbol, 'score': None if symbol in {'SYM001', 'SYM003'} else
         (int(symbol[3:]) + 1) / 40} for symbol in symbols]
snapshot = {symbol: (int(symbol[3:]) + 1) / 200 for symbol in symbols if int(symbol[3:]) < 10}
spec = FactorBacktestSpec(portfolio=ComponentSpec('score_weight', config={'top_n': 20}),
                         cash_reserve=.07, max_position_weight=.12, max_turnover=.23,
                         min_weight_change=.001)
target = build_targets(pl.DataFrame(rows), snapshot, spec)
relative = {symbol: (int(symbol[3:]) + 1) / 40 for symbol in symbols}
allocation = _capped_allocation(relative, .93, .03)
print(json.dumps({'target': target.to_dict(), 'allocation': allocation}, sort_keys=True))
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    outputs = [subprocess.run([sys.executable, "-c", script], env={**environment, "PYTHONHASHSEED": seed},
                              capture_output=True, text=True, check=True, timeout=30).stdout
               for seed in ("1", "17", "101")]
    assert outputs[0] == outputs[1] == outputs[2]
