"""Cash/remaining-slot allocation, capacity, exits and normal-backtest integration."""
from datetime import datetime, UTC, timedelta
from decimal import Decimal

import polars as pl
import pytest

from research import ResearchError
from research.backtesting import ExecutionSettings
from research.strategies import MovingAverageCross
from test_portfolio_backtest import BASE, Signals, bars, buys, run
from test_initial_split import EveryBar, research


def slots(frames, signals, **settings):
    return run(frames, Signals(signals), allocation_mode="holding_slots",
               settings=ExecutionSettings(initial_cash=2500, **settings))


@pytest.mark.parametrize("outcome,high,low,released", [
    ("win", 12., 10., 110.), ("loss", 10., 8., 90.), ("inconclusive", 12., 8., 90.),
])
def test_capacity_recycles_only_after_executed_exit(outcome, high, low, released):
    symbols = [f"T{i:02}" for i in range(27)]
    frames = {s: bars(s, [10.] * 4,
                      highs=[10., high if s == "T00" else 10., 10., 10.],
                      lows=[10., low if s == "T00" else 10., 10., 10.]) for s in symbols}
    signals = {s: (0,) for s in symbols}
    signals["T25"] = (0, 1)
    result = slots(frames, signals)
    entries = buys(result)
    assert entries["symbol"].to_list() == symbols[:26]
    assert entries["entry_budget"].to_list() == [100.] * 25 + [released]
    assert entries["timestamp"][-1] == BASE + timedelta(hours=2)
    assert result.completed_trades["outcome"].to_list() == [outcome]
    assert result.orders.filter(pl.col("status") == "rejected")["reason"].to_list() == ["no_available_holdings"] * 2
    assert result.equity["available_holdings"].to_list() == [25, 1, 0, 0]
    assert result.open_positions.height == 25
    # T26 has no later signal: it is not automatically queued after rejection.
    assert "T26" not in entries["symbol"].to_list()


def test_budget_recomputed_from_actual_cash_and_slots_with_rounding():
    result = slots({"A": bars("A", [30., 30.]), "B": bars("B", [30., 30.])}, {"A": (0,), "B": (0,)})
    first, second = buys(result).to_dicts()
    assert first["entry_budget"] == 100.
    assert first["cash"] == 2410.
    assert second["entry_budget"] == pytest.approx(2410 / 24)
    assert second["quantity"] == 3
    assert result.metadata["available_holdings"] == 23


def test_failed_entries_do_not_consume_slots_and_costs_fit_budget():
    result = slots({"A": bars("A", [10., 10.], volumes=[100., 0.]),
                    "B": bars("B", [10., 10.]), "C": bars("C", [10., 10.])},
                   {s: (0,) for s in "ABC"}, fractional_shares=True, commission_fixed=1, slippage_bps=100)
    assert result.orders["reason"][0] == "no_reported_volume"
    entries = buys(result)
    assert entries["symbol"].to_list() == ["B", "C"]
    assert entries["entry_budget"][0] == 100.
    assert entries["entry_budget"][1] == pytest.approx(entries["cash"][0] / 24)
    for entry in entries.to_dicts():
        spent = Decimal(entry["quantity_exact"]) * Decimal(entry["price_exact"]) + Decimal(entry["fees_exact"])
        assert spent <= Decimal(entry["entry_budget_exact"])
    assert result.equity["available_holdings"].to_list() == [25, 23]


def test_failed_exit_keeps_slot_occupied():
    result = slots({"A": bars("A", [10.] * 4, highs=[10., 10., 12., 12.], volumes=[100., 100., 0., 100.])},
                   {"A": (0,)})
    assert result.equity["available_holdings"].to_list() == [25, 24, 24, 25]
    assert result.orders.filter(pl.col("status") == "rejected")["reason"].to_list() == ["no_reported_volume"]


@pytest.mark.parametrize("capacity", [1, 7, 25])
def test_api_persists_slot_mode_and_rejects_non_bracket_strategies(research, capacity):
    kwargs = dict(start=datetime(2024, 1, 3, tzinfo=UTC), end=datetime(2024, 1, 6, tzinfo=UTC))
    saved = research.backtest(["A", "B"], strategies=[EveryBar()], allocation_mode="holding_slots",
                              initial_available_holdings=capacity, **kwargs)
    restored = research.load_run(saved.run_id)
    assert restored.manifest["initial_available_holdings"] == capacity
    assert restored.results[0].metadata["initial_available_holdings"] == capacity
    assert restored.results[0].metadata["allocation_mode"] == "holding_slots"
    assert restored.results[0].equity.equals(saved.results[0].equity)
    assert restored.comparison.equals(saved.comparison)
    other = research.backtest(["A", "B"], strategies=[EveryBar()], allocation_mode="holding_slots",
                              initial_available_holdings=capacity + 1, **kwargs)
    with pytest.raises(ResearchError, match="different"):
        research.compare_runs([saved.run_id, other.run_id])
    with pytest.raises(ValueError, match="requires stop/target"):
        research.backtest(["A"], strategies=[MovingAverageCross(1, 2)], allocation_mode="holding_slots", **kwargs)


def test_dashboard_runs_and_remembers_slot_allocation(research, monkeypatch):
    from test_dashboard_processed_backtest import _open
    app = _open(monkeypatch, research)
    app.radio(key="backtest-allocation-mode").set_value("holding_slots").run()
    assert app.number_input(key="backtest-holding-slots").value == 25
    app.number_input(key="backtest-holding-slots").set_value(7).run()
    assert app.number_input(key="backtest-max-workers").disabled
    next(f for f in app.multiselect if f.label == "Strategies").set_value([(EveryBar.name, "1")])
    app.multiselect(key="backtest-tickers").set_value(["A", "B"])
    app.date_input(key="backtest-start").set_value(datetime(2024, 1, 3).date())
    app.date_input(key="backtest-end").set_value(datetime(2024, 1, 6).date())
    next(b for b in app.button if b.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["backtest-run"].manifest["allocation_mode"] == "holding_slots"
    assert app.session_state["backtest-run"].manifest["initial_available_holdings"] == 7
    restarted = _open(monkeypatch, research)
    assert restarted.radio(key="backtest-allocation-mode").value == "holding_slots"
    assert restarted.number_input(key="backtest-holding-slots").value == 7


def test_short_exit_releases_slot_and_uses_available_cash_not_locked_proceeds():
    from test_portfolio_shorts import DirectionSignals
    result = run({"A": bars("A", [10.] * 3, lows=[10., 7., 10.]),
                  "B": bars("B", [10.] * 3)},
                 DirectionSignals(shorts={"A": (0,)}, longs={"B": (0,)}),
                 allocation_mode="holding_slots", settings=ExecutionSettings(initial_cash=2500))
    entries = result.trades.filter(pl.col("position_action") == "entry")
    assert entries["entry_budget"].to_list() == [100., 100.]
    assert result.completed_trades["direction"].to_list() == ["short"]
    assert result.equity["available_holdings"].to_list() == [25, 24, 24]
    assert result.metadata["available_holdings"] == 24


@pytest.mark.parametrize("capacity", [1, 3, 30])
def test_custom_capacity_controls_budget_and_entry_limit(capacity):
    symbols = [f"T{i:02}" for i in range(capacity + 1)]
    result = run({s: bars(s, [10., 10.]) for s in symbols},
                 Signals({s: (0,) for s in symbols}), allocation_mode="holding_slots",
                 initial_available_holdings=capacity,
                 settings=ExecutionSettings(initial_cash=capacity * 100))
    assert buys(result).height == capacity
    assert buys(result)["entry_budget"].to_list() == [100.] * capacity
    assert result.orders["reason"][-1] == "no_available_holdings"
    assert result.equity["available_holdings"].to_list() == [capacity, 0]


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "10", None])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        run({"A": bars("A", [10., 10.])}, initial_available_holdings=capacity,
            allocation_mode="holding_slots")
