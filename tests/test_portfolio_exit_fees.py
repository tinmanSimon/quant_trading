"""Simultaneous exit fees cannot give alphabetically earlier tickers priority."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from research.backtesting import ExecutionSettings, run_portfolio_backtest
from research.strategies import BracketStrategy


START = datetime(2024, 1, 2, tzinfo=UTC)


class EnterOnce(BracketStrategy):
    name = "simultaneous_exit_test"
    lookback = 1
    stop_percentage = 0.1
    win_percentage = 1.1

    @property
    def config(self):
        return {}

    def entry_signal(self, history):
        return history.height == 1


def bars(symbol, outcome, *, exit_volume=1.0):
    # Enter on the first execution bar, then trigger exits together on the next.
    prices = {
        "win": (1.2, 1.0, 1.1),
        "loss": (1.0, 0.05, 0.1),
        "hold": (1.0, 1.0, 1.0),
    }
    high, low, close = prices[outcome]
    return pl.DataFrame({
        "timestamp": [START - timedelta(days=1), START, START + timedelta(days=1)],
        "symbol": [symbol] * 3,
        "open": [1.0] * 3,
        "high": [1.0, 1.0, high], "low": [1.0, 1.0, low], "close": [1.0, 1.0, close],
        "volume": [1.0, 1.0, exit_volume],
    })


def run(frames, **settings):
    return run_portfolio_backtest(
        frames, EnterOnce(), start=START, end=START + timedelta(days=2),
        settings=ExecutionSettings(commission_fixed=1, **settings),
    )


@pytest.mark.parametrize("winner", ["A", "B"])
def test_simultaneous_win_can_fund_stop_fee_regardless_of_ticker_names(winner):
    result = run({symbol: bars(symbol, "win" if symbol == winner else "loss")
                  for symbol in ("A", "B")}, initial_cash=10)
    assert result.metrics["wins"] == result.metrics["losses"] == 1
    assert result.open_positions.is_empty()
    assert result.metrics["final_equity"] == pytest.approx(2.8)
    assert result.metrics["total_fees"] == 4
    assert result.metrics["total_pnl"] == pytest.approx(-7.2)
    assert "all_or_none" in result.metadata["simultaneous_exit_fees"]


def test_insufficient_fee_pool_rejects_every_cash_consuming_exit():
    # Each entry buys 4 shares and leaves $0.40; shared cash $0.80 can
    # fund one $0.60 exit deficit, but cannot fund both at the same time.
    result = run({symbol: bars(symbol, "loss") for symbol in ("A", "B")}, initial_cash=10.8)
    assert result.completed_trades.is_empty()
    assert result.open_positions["symbol"].to_list() == ["A", "B"]
    exits = result.orders.filter(pl.col("side") == "sell")
    assert exits["status"].to_list() == ["rejected", "rejected"]
    assert exits["reason"].to_list() == ["sale_proceeds_cannot_cover_fees"] * 2
    assert exits["cash"].to_list() == pytest.approx([0.8, 0.8])
    assert result.metrics["total_fees"] == 2


def test_zero_volume_winner_cannot_fund_another_exit():
    result = run({"A": bars("A", "win", exit_volume=0), "B": bars("B", "loss")}, initial_cash=10)
    assert result.completed_trades.is_empty()
    assert result.open_positions.height == 2
    exits = result.orders.filter(pl.col("side") == "sell")
    assert dict(zip(exits["symbol"], exits["reason"])) == {
        "A": "no_reported_volume", "B": "sale_proceeds_cannot_cover_fees",
    }


def test_zero_volume_cash_consuming_exit_does_not_use_the_fee_pool():
    # Both stops need $0.60; $0.80 can fund the sole executable stop.
    result = run({"A": bars("A", "loss", exit_volume=0), "B": bars("B", "loss")}, initial_cash=10.8)
    assert result.completed_trades["symbol"].to_list() == ["B"]
    assert result.open_positions["symbol"].to_list() == ["A"]
    assert result.orders.filter(pl.col("side") == "sell")["fees"].sum() == 1
