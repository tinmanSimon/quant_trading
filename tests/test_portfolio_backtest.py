"""Hand-calculated shared-capital bracket execution and timing regressions."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from research.backtesting import ExecutionSettings
from research.backtesting.portfolio import run_portfolio_backtest
from research.strategies.bracket import BracketStrategy


BASE = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)


class Signals(BracketStrategy):
    name = "test_bracket_signals"
    lookback = 1
    stop_percentage = 0.9
    win_percentage = 1.1

    def __init__(self, signals=None):
        self.signals = signals if signals is not None else {"A": (0,)}
        self.calls = 0

    @property
    def config(self):
        return {
            "signals": {symbol: list(indices) for symbol, indices in self.signals.items()},
            "stop_percentage": self.stop_percentage,
            "win_percentage": self.win_percentage,
        }

    def entry_signal(self, history):
        self.calls += 1
        return history.height - 1 in self.signals.get(history["symbol"][-1], ())


def bars(symbol, opens, *, highs=None, lows=None, closes=None, volumes=None):
    opens = [float(value) for value in opens]
    closes = opens if closes is None else [float(value) for value in closes]
    return pl.DataFrame({
        "timestamp": [BASE + timedelta(hours=i) for i in range(len(opens))],
        "symbol": [symbol] * len(opens),
        "open": opens,
        "high": highs if highs is not None else [max(a, b) for a, b in zip(opens, closes)],
        "low": lows if lows is not None else [min(a, b) for a, b in zip(opens, closes)],
        "close": closes,
        "volume": volumes if volumes is not None else [100.0] * len(opens),
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms", "UTC")),
        pl.col("open", "high", "low", "close", "volume").cast(pl.Float64),
    )


def run(frames, strategy=None, *, settings=None, **kwargs):
    return run_portfolio_backtest(
        frames,
        Signals() if strategy is None else strategy,
        start=BASE + timedelta(hours=1),
        end=BASE + timedelta(hours=max(frame.height for frame in frames.values())),
        settings=ExecutionSettings(initial_cash=1000) if settings is None else settings,
        **kwargs,
    )


def buys(result):
    return result.trades.filter(pl.col("side") == "buy")


def test_five_thousand_cash_divided_across_ten_nonheld_tickers():
    symbols = ["HOLD", "NEW", *[f"IDLE{i}" for i in range(9)]]
    result = run(
        {symbol: bars(symbol, [10, 10, 10]) for symbol in symbols},
        Signals({"HOLD": (0,), "NEW": (1,)}),
        settings=ExecutionSettings(initial_cash=5500),
    )
    first, second = buys(result).to_dicts()
    assert first["symbol"] == "HOLD" and first["quantity"] == 50
    assert first["cash"] == 5000
    assert second["symbol"] == "NEW" and second["quantity"] == 50
    assert second["cash"] == 4500
    assert result.open_positions.height == 2


def test_simultaneous_entries_get_equal_budgets_independent_of_ticker_order():
    frames = {symbol: bars(symbol, [10, 10]) for symbol in ("A", "B", "IDLE")}
    strategy = Signals({"A": (0,), "B": (0,)})
    first = run(frames, strategy)
    second = run(dict(reversed(list(frames.items()))), strategy)
    allocation_columns = ["symbol", "quantity", "price", "fees"]
    assert buys(first).select(allocation_columns).sort("symbol").equals(
        buys(second).select(allocation_columns).sort("symbol")
    )
    assert buys(first)["quantity"].to_list() == [33, 33]
    assert first.equity["cash"][-1] == second.equity["cash"][-1] == 340


def test_signal_enters_next_open_with_source_bar_positions():
    result = run({"A": bars("A", [9, 10, 20], highs=[9, 10, 23])}, Signals({"A": (1,)}))
    entry = buys(result).row(0, named=True)
    trade = result.completed_trades.row(0, named=True)
    assert entry["timestamp"] == BASE + timedelta(hours=2)
    assert entry["decision_timestamp"] == BASE + timedelta(hours=1)
    assert entry["price"] == 20
    assert (trade["signal_index"], trade["entry_index"], trade["exit_index"]) == (1, 2, 2)
    assert trade["stop_price"] == 18
    assert trade["win_price"] == 22


@pytest.mark.parametrize(
    ("high", "low", "outcome", "exit_price"),
    [(12, 10, "win", 11), (10, 8, "loss", 9), (12, 8, "inconclusive", 9)],
)
def test_entry_bar_can_exit_and_both_thresholds_mean_inconclusive_stop(high, low, outcome, exit_price):
    result = run({"A": bars("A", [10, 10], highs=[10, high], lows=[10, low])})
    trade = result.completed_trades.row(0, named=True)
    assert trade["outcome"] == outcome
    assert trade["entry_index"] == trade["exit_index"] == 1
    assert trade["entry_price"] == 10
    assert trade["exit_price"] == exit_price
    assert trade["quantity"] == 100
    assert trade["net_pnl"] == (exit_price - 10) * 100
    assert result.open_positions.is_empty()
    assert result.metrics["completed_trade_count"] == 1
    assert result.metrics["wins"] == int(outcome == "win")
    assert result.metrics["losses"] == int(outcome == "loss")
    assert result.metrics["inconclusive"] == int(outcome == "inconclusive")


@pytest.mark.parametrize(("high", "low"), [(11, 10), (10, 9), (11, 9)])
def test_touching_thresholds_does_not_exit(high, low):
    result = run({"A": bars("A", [10, 10], highs=[10, high], lows=[10, low])})
    assert result.completed_trades.is_empty()
    assert result.open_positions.height == 1
    assert result.trades["side"].to_list() == ["buy"]


def test_holding_blocks_overlapping_entries_and_exit_close_can_signal_reentry():
    frame = bars("A", [10] * 5, highs=[10, 10, 12, 10, 12])
    result = run({"A": frame}, Signals({"A": (0, 1, 2, 3)}))
    assert result.completed_trades["entry_index"].to_list() == [1, 3]
    assert result.completed_trades["exit_index"].to_list() == [2, 4]
    assert result.completed_trades["quantity"].to_list() == [100, 110]
    assert result.trades["side"].to_list() == ["buy", "sell", "buy", "sell"]
    assert result.metrics["final_equity"] == 1210


def test_same_bar_sale_proceeds_only_fund_later_open_and_closed_ticker_rejoins_pool():
    result = run(
        {
            "A": bars("A", [10] * 4, highs=[10, 10, 12, 10]),
            "B": bars("B", [10] * 4),
        },
        Signals({"A": (0, 2), "B": (1,)}),
    )
    entries = buys(result).to_dicts()
    assert [(row["symbol"], row["quantity"]) for row in entries] == [("A", 50), ("B", 50), ("A", 55)]
    assert entries[1]["cash"] == 0
    assert entries[2]["cash"] == 0
    assert result.metrics["realized_pnl"] == 50
    assert result.metrics["final_equity"] == 1050


def test_same_open_entry_budgets_do_not_include_entry_bar_exit_proceeds():
    result = run(
        {"A": bars("A", [10, 10], highs=[10, 12]), "B": bars("B", [10, 10])},
        Signals({"A": (0,), "B": (0,)}),
    )
    assert buys(result)["quantity"].to_list() == [50, 50]
    assert result.equity["cash"][-1] == 550
    assert result.completed_trades["symbol"].to_list() == ["A"]
    assert result.open_positions["symbol"].to_list() == ["B"]


def test_calendar_close_proceeds_can_fund_next_open_at_same_timestamp():
    frames = {
        "A": bars("A", [10] * 3, highs=[10, 12, 10]),
        "B": bars("B", [10] * 3),
    }
    times = {
        symbol: {label: (label, label + timedelta(hours=1)) for label in frame["timestamp"]}
        for symbol, frame in frames.items()
    }
    result = run(frames, Signals({"A": (0,), "B": (1,)}), bar_times=times)
    entries = buys(result).to_dicts()
    assert [(row["symbol"], row["quantity"]) for row in entries] == [("A", 50), ("B", 52)]
    assert entries[1]["entry_budget"] == 525
    assert entries[1]["decision_timestamp"] == BASE + timedelta(hours=2)
    assert result.completed_trades["exit_timestamp"][0] == entries[1]["timestamp"]


def test_overlapping_cross_ticker_sessions_do_not_release_cash_before_close():
    frames = {
        "A": bars("A", [10] * 3, highs=[10, 12, 10]),
        "B": bars("B", [10] * 4),
    }
    times = {
        "A": {
            BASE: (BASE, BASE + timedelta(hours=1)),
            BASE + timedelta(hours=1): (BASE + timedelta(hours=1), BASE + timedelta(hours=3)),
            BASE + timedelta(hours=2): (BASE + timedelta(hours=3), BASE + timedelta(hours=4)),
        },
        "B": {label: (label, label + timedelta(hours=1)) for label in frames["B"]["timestamp"]},
    }
    result = run_portfolio_backtest(
        frames,
        Signals({"A": (0, 1), "B": (1,)}),
        start=BASE + timedelta(hours=1),
        end=BASE + timedelta(hours=5),
        settings=ExecutionSettings(initial_cash=1000),
        bar_times=times,
    )
    entries = buys(result).to_dicts()
    assert [(row["symbol"], row["quantity"]) for row in entries] == [("A", 50), ("B", 50), ("A", 55)]
    assert entries[1]["timestamp"] == BASE + timedelta(hours=2)
    assert result.completed_trades["exit_timestamp"][0] == BASE + timedelta(hours=3)
    assert entries[2]["timestamp"] == BASE + timedelta(hours=3)


def test_fees_and_slippage_fit_entry_budget_and_thresholds_use_raw_open():
    settings = ExecutionSettings(initial_cash=100, commission_fixed=1, commission_bps=100, slippage_bps=100)
    result = run(
        {"A": bars("A", [10, 10], highs=[10, 12]), "IDLE": bars("IDLE", [10, 10])},
        settings=settings,
    )
    entry = buys(result).row(0, named=True)
    trade = result.completed_trades.row(0, named=True)
    entry_cost = entry["quantity"] * Decimal(entry["price_exact"]) + Decimal(entry["fees_exact"])
    assert entry["quantity"] == 4
    assert entry_cost == Decimal("41.804") <= Decimal(50)
    assert trade["stop_price"] == 9 and trade["win_price"] == 11
    assert Decimal(trade["entry_price_exact"]) == Decimal("10.1")
    assert Decimal(trade["exit_price_exact"]) == Decimal("10.89")
    assert Decimal(trade["fees_exact"]) == Decimal("2.8396")
    assert Decimal(trade["gross_pnl_exact"]) == Decimal("3.16")
    assert Decimal(trade["net_pnl_exact"]) == Decimal("0.3204")
    assert result.metrics["total_pnl"] == pytest.approx(0.3204)


def test_realized_and_unrealized_pnl_reconcile_to_equity_including_entry_fees():
    settings = ExecutionSettings(initial_cash=1000, commission_fixed=1, commission_bps=100, slippage_bps=100)
    result = run(
        {
            "A": bars("A", [10, 10, 10], highs=[10, 10, 12]),
            "B": bars("B", [10, 10, 10], closes=[10, 10, 10.5]),
            "IDLE": bars("IDLE", [10, 10, 10]),
        },
        Signals({"A": (0,), "B": (0,)}),
        settings=settings,
    )
    assert result.metrics["realized_pnl"] == pytest.approx(16.5632)
    assert result.metrics["unrealized_pnl"] == pytest.approx(8.568)
    assert result.metrics["total_pnl"] == pytest.approx(25.1312)
    assert result.metrics["final_equity"] - 1000 == pytest.approx(result.metrics["total_pnl"])
    assert result.metrics["total_pnl"] == pytest.approx(result.metrics["realized_pnl"] + result.metrics["unrealized_pnl"])
    assert all(Decimal(value) >= 0 for value in result.equity["cash_exact"])


def test_final_signal_has_no_next_open_and_unresolved_position_is_not_a_loss():
    no_entry = run({"A": bars("A", [10, 10, 10])}, Signals({"A": (2,)}))
    assert no_entry.trades.is_empty()
    assert no_entry.completed_trades.is_empty()
    assert no_entry.open_positions.is_empty()
    opened = run({"A": bars("A", [10, 10, 10], closes=[10, 10, 10.5])})
    assert opened.trades["side"].to_list() == ["buy"]
    assert opened.completed_trades.is_empty()
    assert opened.open_positions.height == 1
    assert opened.metrics["completed_trade_count"] == 0
    assert opened.metrics["wins"] == opened.metrics["losses"] == opened.metrics["inconclusive"] == 0
    assert opened.metrics["realized_pnl"] == 0
    assert opened.metrics["unrealized_pnl"] == 50
    assert opened.metrics["final_equity"] == 1050


def test_zero_volume_rejects_entry_with_no_cash_change():
    result = run({"A": bars("A", [10, 10], volumes=[100, 0])})
    assert result.trades.is_empty()
    assert result.completed_trades.is_empty()
    assert result.orders["reason"][0] == "no_reported_volume"
    assert result.equity["cash"][-1] == 1000


def test_zero_volume_rejects_exit_until_later_executable_threshold_crossing():
    result = run({"A": bars("A", [10, 10, 10, 10], highs=[10, 10, 12, 12], volumes=[100, 100, 0, 100])})
    assert result.completed_trades["exit_index"].to_list() == [3]
    rejected = result.orders.filter(pl.col("status") == "rejected")
    assert rejected["side"].to_list() == ["sell"]
    assert rejected["reason"].to_list() == ["no_reported_volume"]


def test_unaffordable_entry_records_rejection_and_keeps_empty_schemas():
    result = run({"A": bars("A", [100, 100])}, settings=ExecutionSettings(initial_cash=100, commission_fixed=100))
    assert result.trades.is_empty()
    assert result.orders["reason"][0] == "insufficient_cash"
    assert {"outcome", "entry_index", "exit_index", "net_pnl_exact"} <= set(result.completed_trades.columns)
    assert "symbol" in result.open_positions.columns
    assert result.metrics["total_pnl"] == 0


def test_future_prices_do_not_change_earlier_orders_fills_or_valuations():
    first_frames = {symbol: bars(symbol, [10] * 6) for symbol in ("A", "B")}
    changed_frames = dict(first_frames)
    changed_frames["A"] = bars("A", [10, 10, 10, 10, 500, 3])
    strategy = Signals({"A": (0,), "B": (2,)})
    first = run(first_frames, strategy)
    changed = run(changed_frames, strategy)
    before_change = pl.col("timestamp") < BASE + timedelta(hours=4)
    for table in ("trades", "orders", "equity"):
        assert getattr(first, table).filter(before_change).equals(getattr(changed, table).filter(before_change))


def test_strategy_is_cloned_per_symbol_and_original_is_not_mutated():
    class FirstSignal(Signals):
        def entry_signal(self, history):
            self.calls += 1
            return self.calls == 1

    strategy = FirstSignal()
    frames = {symbol: bars(symbol, [10, 10]) for symbol in ("A", "B")}
    original = {symbol: frame.clone() for symbol, frame in frames.items()}
    result = run(frames, strategy)
    assert buys(result)["symbol"].sort().to_list() == ["A", "B"]
    assert strategy.calls == 0
    assert all(frames[symbol].equals(original[symbol]) for symbol in frames)


def test_rejected_entry_does_not_share_completed_trade_identity():
    result = run(
        {"A": bars("A", [10, 10], volumes=[100, 0]),
         "B": bars("B", [10, 10], highs=[10, 12])},
        Signals({"A": (0,), "B": (0,)}),
    )
    rejected = result.orders.filter(pl.col("status") == "rejected")
    assert rejected["record_id"].to_list() == [None]
    assert result.completed_trades["record_id"].to_list() == [1]


def test_extreme_aggregate_holdings_rejected_with_clear_range_error():
    with pytest.raises(ValueError, match="Aggregate portfolio share count"):
        run({symbol: bars(symbol, [.1, .1]) for symbol in ("A", "B")},
            Signals({"A": (0,), "B": (0,)}), settings=ExecutionSettings(initial_cash=10**18))


def test_late_ticker_joins_only_after_effective_start_and_warmup():
    frames = {"A": bars("A", [10] * 5),
              "LATE": bars("LATE", [10] * 5).slice(2)}
    result = run_portfolio_backtest(
        frames, Signals({"A": (0,), "LATE": (0,)}),
        start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=5),
        effective_starts={"LATE": BASE + timedelta(hours=3)},
        settings=ExecutionSettings(initial_cash=1000),
    )
    # Before LATE is eligible, A gets the entire available allocation.
    assert buys(result)["symbol"].to_list() == ["A"]
    assert buys(result)["quantity"].to_list() == [100]
    attempted = result.orders.filter(pl.col("symbol") == "LATE")
    assert attempted["timestamp"].to_list() == [BASE + timedelta(hours=3)]
    assert attempted["reason"].to_list() == ["insufficient_cash"]


def test_invalid_prepared_signal_array_is_rejected():
    class Bad(Signals):
        def prepare_entry_signals(self, history):
            return pl.Series([True])

    with pytest.raises(ValueError, match="one non-null boolean"):
        run({"A": bars("A", [10, 10])}, Bad())
