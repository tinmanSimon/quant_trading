from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from research.backtesting.factor_engine import run_factor_backtest
from research.backtesting.factor_execution import execute_targets
from research.backtesting.models import ExecutionSettings
from research.factors import ComponentSpec, FactorBacktestSpec, FactorSpec
from research.portfolio import PortfolioRegistry, TargetPortfolio, build_targets


def spec(**overrides):
    return FactorBacktestSpec(**{**dict(factors=(FactorSpec("momentum", 1),), cash_reserve=0,
                                     max_position_weight=1, max_turnover=2,
                                     portfolio=ComponentSpec("equal_weight", config={"top_n": 1})), **overrides})


def day(index):
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index)


def frame(symbol, closes, *, opens=None, volumes=None, offset=0):
    opens = closes if opens is None else opens
    return pl.DataFrame({"timestamp": [day(index + offset) for index in range(len(closes))],
                         "symbol": [symbol] * len(closes), "open": list(map(float, opens)),
                         "close": list(map(float, closes)),
                         "high": [float(max(o, c)) + 1 for o, c in zip(opens, closes)],
                         "low": [float(min(o, c)) / 2 for o, c in zip(opens, closes)],
                         "volume": volumes if volumes is not None else [100.] * len(closes)})


def simulate(frames, configuration=None, **kwargs):
    times = {symbol: {label: (label + timedelta(hours=14, minutes=30), label + timedelta(hours=21))
                      for label in data["timestamp"]} for symbol, data in frames.items()}
    return run_factor_backtest(frames, configuration or spec(), start=kwargs.pop("start", day(2)),
                               end=kwargs.pop("end", max(data["timestamp"][-1] for data in frames.values()) + timedelta(days=1)),
                               settings=kwargs.pop("settings", ExecutionSettings(initial_cash=100, fractional_shares=True)),
                               bar_times=kwargs.pop("bar_times", times), **kwargs)


def execute(cash, holdings, prices, weights, *, configuration=None, settings=None, **kwargs):
    target = weights if isinstance(weights, TargetPortfolio) else TargetPortfolio(weights=weights)
    return execute_targets(cash, holdings, prices, target, spec=configuration or spec(),
                           settings=settings or ExecutionSettings(initial_cash=100, fractional_shares=True), **kwargs)


def custom_policy(targets):
    class Policy:
        def __init__(self):
            self.index = 0

        def propose(self, scores, snapshot):
            result = targets[min(self.index, len(targets) - 1)]
            self.index += 1
            return result
    registry = PortfolioRegistry()
    registry.register("scripted", "1", lambda config: Policy())
    return registry


def test_decision_uses_completed_prices_and_fills_next_open():
    data = {"A": frame("A", [10, 12, 15], opens=[10, 12, 20]),
            "B": frame("B", [10, 11, 1000])}
    result = simulate(data)
    first = result.trades.row(0, named=True)
    assert first["symbol"] == "A"
    assert first["decision_timestamp"] == day(1) + timedelta(hours=21)
    assert first["timestamp"] == day(2) + timedelta(hours=14, minutes=30)
    assert first["price"] == 20
    assert first["quantity"] == 5
    assert result.metrics["final_equity"] == 75
    assert result.rebalances["status"][-1] == "expired"
    assert result.orders.height == 1


def test_future_perturbation_cannot_change_earlier_scores_targets_or_fills():
    original = {"A": frame("A", [10, 12, 13, 14, 15]), "B": frame("B", [10, 11, 12, 13, 14])}
    changed = {**original, "B": frame("B", [10, 11, 12, 1300, 1400])}
    first, second = simulate(original), simulate(changed)
    cutoff = day(3)
    for table, column in (("factor_values", "timestamp"), ("orders", "timestamp"), ("equity", "valuation_time")):
        assert getattr(first, table).filter(pl.col(column) < cutoff).equals(
            getattr(second, table).filter(pl.col(column) < cutoff))
    assert first.targets.filter(pl.col("rebalance_id") == "rebalance-000001").equals(
        second.targets.filter(pl.col("rebalance_id") == "rebalance-000001"))


def test_ticker_permutation_is_identical_with_fees_and_fractional_positions():
    data = {"B": frame("B", [10, 11, 12, 15]), "A": frame("A", [10, 12, 13, 14])}
    config = spec(portfolio=ComponentSpec("equal_weight", config={"top_n": 2}))
    settings = ExecutionSettings(initial_cash=100, fractional_shares=True, commission_fixed=1, slippage_bps=5)
    first = simulate(data, config, settings=settings)
    second = simulate(dict(reversed(list(data.items()))), config, settings=settings)
    for table in ("equity", "orders", "trades", "targets", "factor_values", "rebalances", "positions"):
        assert getattr(first, table).equals(getattr(second, table)), table
    assert first.metrics == second.metrics


def test_common_equity_snapshot_and_sell_before_buy():
    result = execute(0, {"A": 10}, {"A": 20, "B": 10}, {"A": .5, "B": .5})
    assert result.equity_before == 200
    assert result.holdings == {"A": 5, "B": 10}
    assert [order["side"] for order in result.orders] == ["sell", "buy"]
    assert result.cash == 0
    assert result.turnover == 1


@pytest.mark.parametrize("funding", ["pro_rata", "score_priority"])
def test_failed_sale_does_not_fund_purchase(funding):
    result = execute(0, {"A": 10}, {"A": 10, "B": 10}, {"B": 1},
                     unavailable=["A"], configuration=spec(funding=funding))
    assert result.holdings == {"A": 10}
    assert result.cash == 0
    assert all(order["quantity"] == 0 for order in result.orders)
    assert result.orders[0]["reason"] == "no_reported_volume"


def test_delayed_sale_proceeds_become_usable_on_later_event():
    config = spec(sale_proceeds="next_open")
    first = execute(0, {"A": 10}, {"A": 10, "B": 10}, {"B": 1}, configuration=config)
    assert first.holdings == {}
    assert first.cash == 100
    second = execute(first.cash, first.holdings, {"A": 10, "B": 10}, {"B": 1}, configuration=config)
    assert second.holdings == {"B": 10}
    assert second.cash == 0


def test_purchase_minimum_is_budget_including_fees_not_a_cash_reserve():
    settings = ExecutionSettings(initial_cash=5, fractional_shares=True, commission_fixed=1)
    exactly = execute(5, {}, {"A": 1000}, {"A": 1}, settings=settings)
    assert exactly.holdings == {"A": Decimal(".004")}
    assert exactly.cash == 0
    below = execute(Decimal("4.99"), {}, {"A": 1000}, {"A": 1}, settings=settings)
    assert below.holdings == {}
    assert below.orders[0]["reason"] == "below_minimum_trade_budget"


def test_small_position_liquidation_is_not_blocked_by_purchase_minimum():
    result = execute(0, {"A": Decimal(".01")}, {"A": 100}, {})
    assert result.cash == 1
    assert result.holdings == {}


def test_profitable_sales_can_pay_commission_for_tiny_closures():
    settings = ExecutionSettings(initial_cash=100, fractional_shares=True, commission_fixed=2)
    result = execute(0, {"A": Decimal(".01"), "Z": 1}, {"A": 100, "Z": 100}, {}, settings=settings)
    assert result.holdings == {}
    assert result.cash == 97
    assert [order["symbol"] for order in result.orders] == ["Z", "A"]


def test_joint_pro_rata_budget_preserves_reserve_after_costs():
    settings = ExecutionSettings(initial_cash=100, fractional_shares=True, commission_fixed=1, commission_bps=10)
    result = execute(100, {}, {"A": 10, "B": 10}, {"A": .4, "B": .4},
                     configuration=spec(cash_reserve=.2), settings=settings)
    assert result.holdings["A"] == result.holdings["B"]
    assert result.cash >= 20
    assert result.cash < Decimal("20.000001")
    assert all(order["status"] == "reduced" for order in result.orders)


def test_priority_funding_is_explicit_and_uses_scores():
    result = execute(10, {}, {"A": 10, "B": 10}, {"A": .5, "B": .5}, scores={"A": .5, "B": 1},
                     settings=ExecutionSettings(initial_cash=10, fractional_shares=True, commission_fixed=1),
                     configuration=spec(funding="score_priority"))
    assert result.holdings == {"B": Decimal(".5")}
    assert result.cash == 4
    assert result.orders[0]["symbol"] == "B"


def test_frozen_and_no_change_preserve_quantities_after_price_gaps():
    frozen = execute(50, {"A": 5}, {"A": 30, "B": 10},
                     TargetPortfolio(weights={"A": .5, "B": .5}, frozen_symbols=("A",)))
    assert frozen.holdings["A"] == 5
    assert frozen.holdings["B"] == 5
    assert frozen.cash == 0
    unchanged = execute(50, {"A": 5}, {"A": 30}, TargetPortfolio(weights={"A": .5}, no_change=True))
    assert unchanged.holdings == {"A": 5}
    assert unchanged.cash == 50
    assert unchanged.orders == []


def test_mandatory_cap_reduction_is_not_suppressed_by_tolerance():
    configuration = spec(max_position_weight=.2, min_weight_change=.005)
    scores = pl.DataFrame({"symbol": ["A"], "score": [1.]})
    target = build_targets(scores, {"A": .201}, configuration)
    result = execute(799, {"A": Decimal("2.01")}, {"A": 100}, target, configuration=configuration)
    assert result.holdings["A"] == 2
    assert result.orders[0]["side"] == "sell"


def test_policy_state_is_fresh_per_run_and_empty_target_liquidates():
    targets = [TargetPortfolio(weights={"A": 1}), TargetPortfolio(no_change=True), TargetPortfolio()]
    registry = custom_policy(targets)
    config = spec(portfolio=ComponentSpec("scripted"))
    data = {"A": frame("A", [10, 10, 10, 20, 20])}
    first = simulate(data, config, portfolio_registry=registry)
    second = simulate(data, config, portfolio_registry=registry)
    assert first.orders.equals(second.orders)
    assert first.trades["side"].to_list() == ["buy", "sell"]
    assert first.rebalances["status"][1] == "no_change"
    assert first.metrics["final_cash"] == 200
    assert first.open_positions.is_empty()


def test_rejected_order_expires_instead_of_filling_on_unscheduled_bar():
    data = {"A": frame("A", [10, 11, 12, 13, 14], volumes=[100., 100., 0., 100., 100.])}
    result = simulate(data, spec(schedule=ComponentSpec("every_n_bars", config={"bars": 3})))
    assert result.trades.is_empty()
    assert result.orders.height == 1
    assert result.orders["reason"][0] == "no_reported_volume"


def test_missing_new_execution_bar_is_rejected_and_missing_held_bar_is_error():
    data = {"A": frame("A", [10, 20, 20, 20]), "B": frame("B", [10, 11, 11, 11])}
    data["A"] = data["A"].filter(pl.col("timestamp") != day(2))
    result = simulate(data)
    assert result.orders["reason"][0] == "missing_execution_bar"
    data = {"A": frame("A", [10, 20, 20, 20]), "B": frame("B", [10, 11, 11, 11])}
    data["A"] = data["A"].filter(pl.col("timestamp") != day(3))
    with pytest.raises(ValueError, match="held ticker"):
        simulate(data)


def test_later_listing_waits_for_completed_warmup():
    data = {"A": frame("A", [10, 11, 12, 13, 14]), "B": frame("B", [10, 100, 1000], offset=2)}
    result = simulate(data, effective_starts={"B": day(4)})
    buys = result.trades.filter((pl.col("symbol") == "B") & (pl.col("side") == "buy"))
    assert buys.height == 1
    assert buys["bar_timestamp"][0] == day(4)


def test_opening_risk_check_handles_gap_and_records_actual_achieved_weights():
    targets = [TargetPortfolio(weights={"A": .5, "B": .5})]
    data = {"A": frame("A", [10, 10, 10, 100]), "B": frame("B", [10, 10, 10, 10])}
    config = spec(portfolio=ComponentSpec("scripted"), max_turnover=.1)
    result = simulate(data, config, portfolio_registry=custom_policy(targets))
    # First event invests 10%; after the gap the second event still trades no
    # more than 10% of its common opening equity (rounding only reduces it).
    executed = result.rebalances.filter(pl.col("status") == "executed")
    assert all(turnover <= .10000001 for turnover in executed["turnover"])
    assert result.targets["achieved_weight"].drop_nulls().len() > 0


def test_calendar_schedule_does_not_treat_truncated_series_as_period_end():
    data = {"A": frame("A", [10, 11, 12])}
    result = simulate(data, spec(schedule=ComponentSpec("monthly")))
    assert result.rebalances.is_empty()
    assert result.trades.is_empty()


def test_invalid_calendar_overlap_or_unfinished_bar_rejected():
    data = {"A": frame("A", [10, 11, 12])}
    with pytest.raises(ValueError, match="incomplete"):
        simulate(data, end=day(2) + timedelta(hours=15))
    with pytest.raises(ValueError, match="calendar bar_times"):
        simulate(data, bar_times=None)


def test_unchanged_price_and_target_produce_no_unnecessary_trades():
    result = simulate({"A": frame("A", [10, 10, 10, 10, 10])})
    assert result.trades.height == 1
    assert result.metrics["final_equity"] == 100
    assert result.metrics["gross_turnover"] == 1


def test_warming_up_ticker_does_not_influence_cross_sectional_ranks():
    data = {"A": frame("A", [10, 11, 12, 13, 14, 15]),
            "B": frame("B", [10, 10, 11, 13, 16, 18]),
            "C": frame("C", [10, 100, 110, 120], offset=2)}
    config = spec(portfolio=ComponentSpec("inverse_volatility", config={"top_n": 2, "volatility_lookback": 3}))
    result = simulate(data, config, start=day(4), effective_starts={"C": day(6)})
    initial = result.factor_values.filter(pl.col("rebalance_id") == "rebalance-000001")
    assert initial["symbol"].to_list() == ["A", "B"]
    assert initial["score"].to_list() == [.5, 1.]


def test_adjacent_intraday_bars_share_boundary_but_never_share_decision_bar():
    labels = [day(0) + timedelta(hours=14, minutes=30 + index * 30) for index in range(4)]
    data = {"A": frame("A", [10, 12, 15, 16]).with_columns(pl.Series("timestamp", labels))}
    times = {"A": {label: (label, label + timedelta(minutes=30)) for label in labels}}
    result = simulate(data, start=labels[2], end=labels[-1] + timedelta(minutes=30), bar_times=times)
    trade = result.trades.row(0, named=True)
    assert trade["timestamp"] == trade["decision_timestamp"] == labels[2]
    assert trade["decision_bar_timestamp"] == labels[1]
    assert trade["bar_timestamp"] == labels[2]


def test_missing_factor_score_freezes_held_quantity_instead_of_liquidating():
    data = {"A": frame("A", [10, 9, 10, 20], opens=[11, 10, 9, 20])}
    config = spec(factors=(FactorSpec("red_bar_high_percentage", 1),))
    result = simulate(data, config)
    assert result.trades.height == 1
    frozen = result.targets.filter(pl.col("rebalance_id") == "rebalance-000002")
    assert frozen["frozen"].to_list() == [True]
    assert "missing factor score" in frozen["reason"][0]
    close_positions = result.positions.filter(pl.col("phase") == "close")
    assert close_positions["quantity_exact"].n_unique() == 1
