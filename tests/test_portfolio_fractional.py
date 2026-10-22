"""Shared fractional allocations retain exact share and money ledgers."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import polars as pl
import pytest

from research.backtesting import ExecutionSettings
from research.backtesting.portfolio import run_portfolio_backtest
from research.runs import load_run, save_run
from research.strategies.bracket import BracketStrategy


BASE = datetime(2025, 1, 2, tzinfo=UTC)


class Signals(BracketStrategy):
    name = "fractional_bracket_test"
    lookback = 1
    stop_percentage = 0.9
    win_percentage = 1.1

    def __init__(self, signals):
        self.signals = signals

    @property
    def config(self):
        return {"signals": self.signals, "stop_percentage": self.stop_percentage,
                "win_percentage": self.win_percentage}

    def entry_signal(self, history):
        return history.height - 1 in self.signals.get(history["symbol"][-1], [])


def bars(symbol, prices, *, highs=None, lows=None, closes=None, volumes=None):
    prices = [float(value) for value in prices]
    return pl.DataFrame({
        "timestamp": [BASE + timedelta(days=i) for i in range(len(prices))],
        "symbol": [symbol] * len(prices), "open": prices,
        "high": prices if highs is None else highs,
        "low": prices if lows is None else lows,
        "close": prices if closes is None else closes,
        "volume": [100.0] * len(prices) if volumes is None else volumes,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")),
                    pl.col("open", "high", "low", "close", "volume").cast(pl.Float64))


def run(frames, signals, *, cash=10, **settings):
    return run_portfolio_backtest(
        frames, Signals(signals), start=BASE + timedelta(days=1),
        end=BASE + timedelta(days=max(frame.height for frame in frames.values())),
        settings=ExecutionSettings(initial_cash=cash, fractional_shares=True, **settings),
    )


def exact(record, name):
    return Decimal(record[name + "_exact"])


def assert_cash_ledger(result):
    with localcontext() as context:
        context.prec = 800
        cash = result.settings.initial_cash
        for fill in result.trades.to_dicts():
            notional = exact(fill, "quantity") * exact(fill, "price")
            cash += (-notional if fill["side"] == "buy" else notional) - exact(fill, "fees")
            assert cash == exact(fill, "cash")
            assert cash >= 0
        assert cash == Decimal(result.equity["cash_exact"][-1])
        realized = sum((Decimal(value) for value in result.completed_trades["net_pnl_exact"]), Decimal(0))
        unrealized = sum((Decimal(value) for value in result.open_positions["unrealized_pnl_exact"]), Decimal(0))
        assert Decimal(result.metrics["total_pnl_exact"]) == realized + unrealized
        assert Decimal(result.metrics["final_equity_exact"]) == result.settings.initial_cash + realized + unrealized
        assert all(Decimal(value) >= 0 for value in result.equity["cash_exact"])


def test_each_five_dollar_allocation_can_buy_expensive_stocks():
    frames = {symbol: bars(symbol, [10000, 10000]) for symbol in ("A", "B")}
    result = run(frames, {"A": [0], "B": [0]})
    assert result.trades["quantity"].to_list() == [0.0005, 0.0005]
    assert result.trades["entry_budget"].to_list() == [5.0, 5.0]
    assert result.equity["cash"][-1] == 0
    assert result.open_positions["quantity_exact"].to_list() == ["0.00050000", "0.00050000"]
    assert Decimal(result.equity["holdings_exact"][-1]) == Decimal("0.001")
    assert_cash_ledger(result)


def test_idle_tickers_count_toward_minimum_purchase_budget():
    frames = {symbol: bars(symbol, [1000, 1000]) for symbol in ("A", "IDLE")}
    result = run(frames, {"A": [0]}, cash=Decimal("9.99"))
    assert result.trades.is_empty()
    assert result.orders["reason"].to_list() == ["below_minimum_trade_budget"]
    assert Decimal(result.orders["entry_budget_exact"][0]) == Decimal("4.995")
    assert result.equity["cash"][-1] == 9.99
    assert_cash_ledger(result)


@pytest.mark.parametrize(("high", "low", "outcome", "exit_price"),
                         [(1200, 1000, "win", 1100), (1000, 800, "loss", 900),
                          (1200, 800, "inconclusive", 900)])
def test_fractional_entry_bar_exits_use_exact_quantity(high, low, outcome, exit_price):
    result = run({"A": bars("A", [1000, 1000], highs=[1000, high], lows=[1000, low])},
                 {"A": [0]}, cash=5)
    record = result.completed_trades.row(0, named=True)
    assert record["outcome"] == outcome
    assert record["entry_index"] == record["exit_index"] == 1
    assert exact(record, "quantity") == Decimal("0.005")
    assert exact(record, "exit_price") == exit_price
    assert exact(record, "net_pnl") == Decimal("0.005") * (exit_price - 1000)
    assert result.open_positions.is_empty()
    assert result.metrics["open_shares"] == 0
    assert_cash_ledger(result)


def test_rounding_fees_slippage_and_open_position_reconcile_exactly():
    frames = {
        "A": bars("A", [123.456789] * 3, highs=[123.456789, 123.456789, 150]),
        "B": bars("B", [234.567891] * 3, highs=[234.567891, 234.567891, 240],
                  closes=[234.567891, 234.567891, 240]),
        "IDLE": bars("IDLE", [1000] * 3),
    }
    result = run(frames, {"A": [0], "B": [0]}, cash=100,
                 commission_fixed=Decimal("0.037"), commission_bps=7, slippage_bps=13)
    for entry in result.trades.filter(pl.col("side") == "buy").to_dicts():
        quantity = exact(entry, "quantity")
        cost = quantity * exact(entry, "price") + exact(entry, "fees")
        assert quantity.as_tuple().exponent == -8
        assert cost <= exact(entry, "entry_budget")
    assert result.completed_trades["symbol"].to_list() == ["A"]
    assert result.open_positions["symbol"].to_list() == ["B"]
    record = result.completed_trades.row(0, named=True)
    assert exact(record, "net_pnl") == (
        exact(record, "quantity") * (exact(record, "exit_price") - exact(record, "entry_price"))
        - exact(record, "entry_fees") - exact(record, "exit_fees")
    )
    assert_cash_ledger(result)


def test_simultaneous_fractional_entries_are_independent_of_ticker_order():
    frames = {symbol: bars(symbol, [123.456789] * 2) for symbol in ("A", "B", "C")}
    signals = {symbol: [0] for symbol in frames}
    first = run(frames, signals, cash=100, commission_fixed=Decimal("0.013"))
    second = run(dict(reversed(list(frames.items()))), signals, cash=100, commission_fixed=Decimal("0.013"))
    assert first.trades.equals(second.trades)
    assert first.equity.equals(second.equity)
    assert_cash_ledger(first)


def test_zero_volume_fractional_entry_preserves_cash():
    result = run({"A": bars("A", [1000, 1000], volumes=[100, 0])}, {"A": [0]}, cash=5)
    assert result.trades.is_empty()
    assert result.orders["reason"].to_list() == ["no_reported_volume"]
    assert result.orders["quantity_exact"].to_list() == ["0"]
    assert_cash_ledger(result)


def test_zero_volume_fractional_exit_keeps_exact_holding_until_next_bar():
    result = run({"A": bars("A", [1000] * 4, highs=[1000, 1000, 1200, 1200], volumes=[100, 100, 0, 100])},
                 {"A": [0]}, cash=5)
    rejected = result.orders.filter(pl.col("status") == "rejected").row(0, named=True)
    assert rejected["side"] == "sell"
    assert rejected["reason"] == "no_reported_volume"
    assert exact(rejected, "holdings") == Decimal("0.005")
    assert result.completed_trades["exit_index"].to_list() == [3]
    assert_cash_ledger(result)


def test_fractional_exit_proceeds_fund_next_purchase_budget():
    result = run({"A": bars("A", [1000] * 3, highs=[1000, 1200, 1000]),
                  "B": bars("B", [1000] * 3)}, {"A": [0], "B": [1]}, cash=10)
    entries = result.trades.filter(pl.col("side") == "buy").to_dicts()
    assert exact(entries[0], "entry_budget") == 5
    assert exact(entries[1], "entry_budget") == Decimal("5.25")
    assert exact(entries[1], "quantity") == Decimal("0.00525")
    assert_cash_ledger(result)


def test_fractional_portfolio_details_and_settings_survive_roundtrip(tmp_path):
    original = run({"A": bars("A", [123.456789] * 2, highs=[123.456789, 150]),
                    "B": bars("B", [234.567891] * 2)}, {"A": [0], "B": [0]}, cash=100,
                   commission_fixed=Decimal("0.017"), slippage_bps=11)
    saved = save_run(tmp_path, [original], {})
    restored = load_run(tmp_path, saved.run_id).results[0]
    assert restored.settings == original.settings
    for name in ("trades", "orders", "equity", "completed_trades", "open_positions"):
        assert getattr(restored, name).equals(getattr(original, name))
    assert restored.metrics == original.metrics
    assert_cash_ledger(restored)
