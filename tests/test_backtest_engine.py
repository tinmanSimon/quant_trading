"""Hand-calculated execution and causal-accounting regression tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import math

import polars as pl
import pytest

from research.backtesting import ExecutionSettings, compare_results, run_backtest
from research.strategies import Momentum, Strategy, WeightedStrategy

BASE = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)


class Targets(Strategy):
    name = "test_targets"
    lookback = 1

    def __init__(self, targets=(1.0,)):
        self.targets = tuple(targets)
        self.calls = 0

    @property
    def config(self):
        return {"targets": list(self.targets)}

    def target_weight(self, history):
        value = self.targets[min(self.calls, len(self.targets) - 1)]
        self.calls += 1
        return value


def bars(opens, closes=None):
    opens = [float(value) for value in opens]
    if closes is None:
        closes = opens
    closes = [float(value) for value in closes]
    return pl.DataFrame({
        "timestamp": [BASE + timedelta(hours=i) for i in range(len(opens))],
        "symbol": ["AAPL"] * len(opens),
        "open": opens,
        "high": [max(a, b) for a, b in zip(opens, closes)],
        "low": [min(a, b) for a, b in zip(opens, closes)],
        "close": closes,
        "volume": [100.0] * len(opens),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")), pl.col("open", "high", "low", "close").cast(pl.Float64))


def run(frame, strategy=None, **kwargs):
    return run_backtest(frame, strategy if strategy is not None else Targets(), start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=frame.height), **kwargs)


def test_gap_up_uses_actual_open_and_includes_fee_in_affordability():
    frame = bars([90, 101])
    result = run(frame, settings=ExecutionSettings(initial_cash=1000, commission_fixed=1))
    fill = result.trades.row(0, named=True)
    assert fill["quantity"] == 9
    assert fill["cash"] == 90
    assert fill["fees"] == 1
    assert fill["price"] == 101
    assert result.orders["requested_quantity"][0] == 9
    assert result.metrics["final_equity"] == 999
    assert result.metrics["total_return"] == pytest.approx(-0.001)
    assert result.metrics["max_drawdown"] == pytest.approx(0.001)


def test_buy_reduced_for_fees_and_slippage_then_sell_correctly():
    result = run(bars([10, 10, 12]), Targets([1.0, 0.0]), settings=ExecutionSettings(initial_cash=100, commission_fixed=1, commission_bps=100, slippage_bps=100))
    buy, sell = result.trades.to_dicts()
    assert buy["price_exact"] == "10.100"
    assert buy["quantity"] == 9
    assert Decimal(buy["fees_exact"]) == Decimal("1.909")
    assert Decimal(buy["cash_exact"]) == Decimal("7.191")
    assert result.orders["status"][0] == "reduced"
    assert sell["side"] == "sell" and sell["quantity"] == 9
    assert Decimal(sell["price_exact"]) == Decimal("11.88")
    assert Decimal(sell["fees_exact"]) == Decimal("2.0692")
    assert Decimal(sell["cash_exact"]) == Decimal("112.0418")
    assert result.metrics["open_shares"] == 0


def test_no_warmup_trades_and_next_open_only():
    result = run(bars([10, 20, 30]))
    assert result.trades["timestamp"][0] == BASE + timedelta(hours=1)
    assert result.trades["decision_timestamp"][0] == BASE
    assert result.trades["price"][0] == 20
    assert result.equity["phase"].to_list() == ["initial", "close", "close"]
    assert result.metrics["open_shares"] > 0
    assert result.metadata["end_position_policy"] == "mark_to_market_at_last_close"


def test_exact_cash_does_not_accumulate_binary_float_error():
    result = run(bars([0.1, 0.1, 0.1]), Targets([1.0, 0.0]), settings=ExecutionSettings(initial_cash=0.3))
    assert result.trades["quantity"].to_list() == [3, 3]
    assert Decimal(result.trades["cash_exact"][0]) == 0
    assert Decimal(result.trades["cash_exact"][1]) == Decimal("0.3")


def test_tiny_commission_rounds_up_without_negative_cash():
    result = run(bars([1, 1]), settings=ExecutionSettings(initial_cash=10, commission_bps=Decimal("0.000000001")))
    assert result.trades["quantity"][0] == 9
    assert Decimal(result.trades["fees_exact"][0]) == Decimal("0.00000001")
    assert Decimal(result.trades["cash_exact"][0]) >= 0


def test_buy_rejection_is_recorded_and_empty_trades_have_schema():
    result = run(bars([100, 100]), settings=ExecutionSettings(initial_cash=100, commission_fixed=100))
    assert result.trades.is_empty()
    assert "fees_exact" in result.trades.columns
    assert result.orders["reason"][0] == "insufficient_cash"
    assert result.metrics["final_equity"] == 100


def test_sale_that_cannot_pay_fee_is_rejected_not_negative():
    result = run(bars([10, 10, 0.01]), Targets([1.0, 0.0]), settings=ExecutionSettings(initial_cash=21, commission_fixed=10))
    assert result.trades.height == 1
    assert result.orders["reason"][-1] == "sale_proceeds_cannot_cover_fees"
    assert result.equity["cash"][-1] == 1
    assert result.equity["holdings"][-1] == 1


def test_all_cash_no_trade_is_valid_and_comparable():
    result = run(bars([10, 20]), Targets([0.0]))
    assert result.metrics["total_return"] == 0
    assert result.metrics["max_drawdown"] == 0
    assert result.orders.is_empty()
    assert compare_results([result])["symbol"].to_list() == ["AAPL"]


@pytest.mark.parametrize("value", [-1.0, 1.01, float("inf"), float("nan"), True, "0.5"])
def test_invalid_signal_never_executes(value):
    class Bad(Targets):
        @property
        def config(self):
            return {}

        def target_weight(self, history):
            return value

    with pytest.raises(ValueError, match="weight"):
        run(bars([10, 20]), Bad())


@pytest.mark.parametrize("kwargs", [{"initial_cash": 0}, {"initial_cash": -1}, {"initial_cash": True}, {"commission_fixed": -1}, {"commission_bps": math.nan}, {"slippage_bps": math.inf}, {"slippage_bps": 10000}])
def test_invalid_execution_settings(kwargs):
    with pytest.raises(ValueError):
        ExecutionSettings(**kwargs)


def test_future_prices_do_not_change_earlier_orders_or_trades():
    original = bars([10, 11, 12, 13, 14, 15, 16])
    changed = bars([10, 11, 12, 13, 14, 999, 3])
    kwargs = {"start": BASE + timedelta(hours=2), "end": BASE + timedelta(hours=7), "settings": ExecutionSettings(initial_cash=1000)}
    first = run_backtest(original, Momentum(1), **kwargs)
    second = run_backtest(changed, Momentum(1), **kwargs)
    boundary = BASE + timedelta(hours=5)
    assert first.trades.filter(pl.col("timestamp") < boundary).equals(second.trades.filter(pl.col("timestamp") < boundary))
    assert first.orders.filter(pl.col("timestamp") < boundary).equals(second.orders.filter(pl.col("timestamp") < boundary))


def test_strategy_never_receives_execution_bar_or_future_rows():
    class Inspect(Targets):
        def target_weight(self, history):
            assert history.height == self.calls + 1
            assert history["timestamp"][-1] == BASE + timedelta(hours=self.calls)
            return super().target_weight(history)

    run(bars([10, 20, 30, 40]), Inspect())


def test_strategy_mutation_does_not_change_prices_or_other_run():
    class Mutate(Targets):
        def target_weight(self, history):
            history.replace_column(history.get_column_index("close"), pl.Series("close", [999.0] * history.height))
            return super().target_weight(history)

    source = bars([10, 20, 30])
    saved = source.clone()
    strategy = Mutate()
    first = run(source, strategy)
    second = run(source, strategy)
    assert source.equals(saved)
    assert first.equity.equals(second.equity)
    assert strategy.calls == 0
    assert first.trades["price"][0] == 20


def test_mutating_declared_config_is_rejected():
    class MutateConfig(Targets):
        def target_weight(self, history):
            self.targets = (0.0,)
            return 1.0

    with pytest.raises(ValueError, match="configuration"):
        run(bars([10, 20]), MutateConfig())


def test_insufficient_warmup_and_multiple_symbols_fail():
    with pytest.raises(ValueError, match="warm-up"):
        run(bars([10, 20, 30]), Momentum(2))
    mixed = bars([10, 20]).with_columns(pl.Series("symbol", ["AAPL", "MSFT"]))
    with pytest.raises(ValueError, match="one symbol"):
        run(mixed)


def test_many_rebalances_keep_ledger_nonnegative_and_accounting_exact():
    opens = [10.0 + ((i * 17) % 31) for i in range(100)]
    closes = [price * (0.95 if i % 2 else 1.05) for i, price in enumerate(opens)]
    result = run(bars(opens, closes), Targets([0.0, 0.5, 1.0, 0.2] * 25), settings=ExecutionSettings(initial_cash=1000, commission_fixed=0.37, commission_bps=7, slippage_bps=13))
    for row in result.equity.to_dicts():
        cash = Decimal(row["cash_exact"])
        assert cash >= 0 and row["holdings"] >= 0
        if row["price"] is not None:
            assert Decimal(row["equity_exact"]) == cash + row["holdings"] * Decimal(str(row["price"]))
    assert (result.trades["quantity"] > 0).all()


def test_combined_strategies_use_single_capital_ledger():
    combined = WeightedStrategy([Targets([1.0]), Targets([0.0])], [0.5, 0.5])
    result = run(bars([10, 10]), combined, settings=ExecutionSettings(initial_cash=100))
    assert result.trades["quantity"][0] == 5
    assert result.trades["cash"][0] == 50


def test_actual_calendar_times_for_daily_labels():
    labels = [datetime(2025, 1, day, tzinfo=UTC) for day in (2, 3)]
    frame = bars([10, 20]).with_columns(pl.Series("timestamp", labels, dtype=pl.Datetime("ms", "UTC")))
    times = {label: (label + timedelta(hours=14, minutes=30), label + timedelta(hours=21)) for label in labels}
    result = run_backtest(frame, Targets(), start=labels[1], end=labels[1] + timedelta(days=1), bar_times=times)
    fill = result.trades.row(0, named=True)
    assert fill["timestamp"] == times[labels[1]][0]
    assert fill["decision_timestamp"] == times[labels[0]][1]
    assert fill["bar_timestamp"] == labels[1]
    assert result.equity["valuation_time"][-1] == times[labels[1]][1]


def test_calendar_rejects_incomplete_or_overlapping_bar():
    frame = bars([10, 20])
    times = {BASE: (BASE, BASE + timedelta(hours=1)), BASE + timedelta(hours=1): (BASE + timedelta(hours=1), BASE + timedelta(hours=3))}
    with pytest.raises(ValueError, match="incomplete"):
        run(frame, bar_times=times)
    times[BASE] = (BASE, BASE + timedelta(hours=2))
    with pytest.raises(ValueError, match="overlap"):
        run(frame, bar_times=times)


def test_fine_query_boundaries_do_not_truncate():
    frame = bars([10, 20, 30])
    result = run_backtest(frame, Targets(), start=BASE + timedelta(hours=1, microseconds=1), end=BASE + timedelta(hours=3))
    assert result.trades["timestamp"][0] == BASE + timedelta(hours=2)


def test_extreme_share_count_and_nonfinite_output_rejected():
    with pytest.raises(ValueError, match="share count"):
        run(bars([1, 1e-300]))
    with pytest.raises(ValueError, match="display range"):
        run(bars([1, 1, 1e308]), Targets([1.0, 1.0]), settings=ExecutionSettings(initial_cash=100))


def test_execution_settings_bound_decimal_precision():
    with pytest.raises(ValueError, match="decimal places"):
        ExecutionSettings(initial_cash=Decimal("1e-1000"))


def test_zero_volume_prevents_buy_execution():
    frame = bars([10, 10]).with_columns(pl.Series("volume", [100.0, 0.0]))
    result = run(frame, settings=ExecutionSettings(initial_cash=100, commission_fixed=1))
    assert result.trades.is_empty()
    assert result.orders["reason"][0] == "no_reported_volume"
    assert result.equity["cash"][-1] == 100
    assert result.equity["holdings"][-1] == 0


def test_zero_volume_prevents_sell_but_marks_position_to_market():
    frame = bars([10, 10, 20]).with_columns(pl.Series("volume", [100.0, 100.0, 0.0]))
    result = run(frame, Targets([1.0, 0.0]), settings=ExecutionSettings(initial_cash=100))
    assert result.trades.height == 1
    assert result.orders["reason"][-1] == "no_reported_volume"
    assert result.equity["holdings"][-1] == 10
    assert result.metrics["final_equity"] == 200
