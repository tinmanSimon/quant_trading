"""Fixed entry allocations stop when cash is exhausted and resume on new signals."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from research.backtesting import ExecutionSettings
from research.backtesting.portfolio import run_portfolio_backtest
from research.strategies import MovingAverageCross
from test_portfolio_backtest import BASE, Signals, bars, buys
from test_initial_split import EveryBar, research


def test_fixed_tenth_skips_unfunded_signals_and_reuses_only_completed_sales():
    symbols = [f"T{i:02}" for i in range(12)]
    frames = {symbol: bars(symbol, [10.] * 4, highs=[10., 10., 12. if symbol == "T00" else 10., 10.])
              for symbol in symbols}
    signals = {symbol: (0,) for symbol in symbols}
    signals["T10"] = (0, 1, 2)
    result = run_portfolio_backtest(
        frames, Signals(signals), start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=4),
        settings=ExecutionSettings(initial_cash=1000, fractional_shares=True), allocation_mode="fixed_tenth",
    )
    entries = buys(result)
    assert entries["symbol"].to_list() == symbols[:10] + ["T10"]
    assert entries["entry_budget"].to_list() == [100.] * 11
    assert entries["quantity"].to_list() == [10.] * 11
    assert entries["timestamp"][-1] == BASE + timedelta(hours=3)
    rejected = result.orders.filter(pl.col("status") == "rejected")
    assert rejected["symbol"].to_list() == ["T10", "T11", "T10"]
    assert set(rejected["reason"]) == {"insufficient_cash_for_fixed_allocation"}
    assert all(Decimal(value) >= 0 for value in result.equity["cash_exact"])
    assert result.equity["cash"][-1] == 10
    assert result.metrics["total_pnl"] == 10
    # Skipped T11 is never entered automatically when cash later becomes available.
    assert "T11" not in entries["symbol"].to_list()


def test_fixed_budget_is_independent_of_ticker_count_and_preserves_fractional_minimum():
    frames = {"A": bars("A", [200, 200])}
    for capital, quantity in [(50, .025), (1000, .5)]:
        result = run_portfolio_backtest(frames, Signals(), start=BASE + timedelta(hours=1),
            end=BASE + timedelta(hours=2), settings=ExecutionSettings(initial_cash=capital, fractional_shares=True),
            allocation_mode="fixed_tenth")
        assert result.trades["quantity"][0] == quantity
        assert result.trades["entry_budget"][0] == capital / 10
    result = run_portfolio_backtest(frames, Signals(), start=BASE + timedelta(hours=1),
        end=BASE + timedelta(hours=2), settings=ExecutionSettings(initial_cash=40, fractional_shares=True),
        allocation_mode="fixed_tenth")
    assert result.trades.is_empty()
    assert result.orders["reason"][0] == "below_minimum_trade_budget"


def test_fixed_budget_includes_fees_and_slippage():
    result = run_portfolio_backtest({"A": bars("A", [200, 200])}, Signals(),
        start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=2),
        settings=ExecutionSettings(initial_cash=1000, fractional_shares=True, commission_fixed=1, slippage_bps=100),
        allocation_mode="fixed_tenth")
    entry = result.trades.row(0, named=True)
    spent = Decimal(entry["quantity_exact"]) * Decimal(entry["price_exact"]) + Decimal(entry["fees_exact"])
    assert spent <= 100 and spent > Decimal("99.99")
    assert Decimal(entry["cash_exact"]) == 1000 - spent


def test_fixed_tenth_api_persists_choice_and_rejects_incompatible_strategies(research):
    run = research.backtest(["A", "B"], strategies=[EveryBar()], allocation_mode="fixed_tenth",
                           start=datetime(2024, 1, 3, tzinfo=UTC), end=datetime(2024, 1, 6, tzinfo=UTC),
                           settings=ExecutionSettings(initial_cash=1000, fractional_shares=True))
    restored = research.load_run(run.run_id)
    assert restored.manifest["allocation_mode"] == "fixed_tenth"
    assert restored.results[0].metadata["allocation_mode"] == "fixed_tenth"
    assert set(buys(restored.results[0])["entry_budget"]) == {100.}
    with pytest.raises(ValueError, match="requires stop/target"):
        research.backtest(["A"], strategies=[MovingAverageCross(1, 2)], allocation_mode="fixed_tenth",
                          start=BASE, end=BASE + timedelta(days=5))


def test_dashboard_exposes_fixed_tenth_and_runs_it(research, monkeypatch):
    import dashboard.app as dashboard
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    app.radio(key="backtest-allocation-mode").set_value("fixed_tenth").run()
    assert app.number_input(key="backtest-max-workers").disabled
    next(field for field in app.multiselect if field.label == "Strategies").set_value([(EveryBar.name, "1")])
    app.multiselect(key="backtest-tickers").set_value(["A", "B"])
    app.date_input(key="backtest-start").set_value(datetime(2024, 1, 3).date())
    app.date_input(key="backtest-end").set_value(datetime(2024, 1, 6).date())
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["backtest-run"].manifest["allocation_mode"] == "fixed_tenth"
    assert any("10% of initial capital" in caption.value for caption in app.caption)


def test_remaining_cash_below_full_allocation_is_not_used_for_a_smaller_entry():
    symbols = [f"T{i:02}" for i in range(11)]
    frames = {symbol: bars(symbol, [10.] * 3, lows=[10., 8. if symbol == "T00" else 10., 10.])
              for symbol in symbols}
    signals = {symbol: (0,) for symbol in symbols}
    signals["T10"] = (0, 1)
    result = run_portfolio_backtest(frames, Signals(signals),
        start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=3),
        settings=ExecutionSettings(initial_cash=1000, fractional_shares=True), allocation_mode="fixed_tenth")
    assert buys(result).height == 10
    assert result.equity["cash"][-1] == 90
    assert result.orders["reason"][-1] == "insufficient_cash_for_fixed_allocation"


@pytest.mark.parametrize('percent,entries', [(5,20), (7,14)])
def test_custom_percentage_exhausts_cash_without_using_smaller_budget(percent, entries):
    frames = {f'T{i:02}': bars(f'T{i:02}', [10.,10.]) for i in range(22)}
    result = run_portfolio_backtest(frames, Signals({symbol: (0,) for symbol in frames}), start=BASE+timedelta(hours=1),
        end=BASE+timedelta(hours=2), settings=ExecutionSettings(initial_cash=1000, fractional_shares=True),
        allocation_mode='fixed_tenth', entry_percentage=percent)
    assert buys(result).height == entries
    assert set(buys(result)['entry_budget']) == {percent*10.}
    assert result.equity['cash'][-1] == 1000-entries*percent*10


@pytest.mark.parametrize('percent', [0,-1,101,True,float('nan'),float('inf')])
def test_invalid_entry_percentage_is_rejected(percent):
    with pytest.raises(ValueError, match='entry_percentage'):
        run_portfolio_backtest({'A':bars('A',[10,10])}, Signals(), start=BASE,
                              end=BASE+timedelta(hours=2), entry_percentage=percent)


def test_custom_percentage_saved_and_compared(research):
    kwargs=dict(strategies=[EveryBar()], allocation_mode='fixed_tenth',
                start=datetime(2024,1,3,tzinfo=UTC),end=datetime(2024,1,6,tzinfo=UTC),
                settings=ExecutionSettings(initial_cash=1000,fractional_shares=True))
    run=research.backtest(['A','B'],entry_percentage=7,**kwargs)
    restored=research.load_run(run.run_id)
    assert restored.manifest['entry_percentage']=='7'
    assert set(buys(restored.results[0])['entry_budget'])=={70.}
    other=research.backtest(['A','B'],entry_percentage=5,**kwargs)
    from research import ResearchError
    with pytest.raises(ResearchError,match='different'):
        research.compare_runs([run.run_id,other.run_id])


def test_dashboard_percentage_persists(research, monkeypatch):
    from test_dashboard_processed_backtest import _open
    app=_open(monkeypatch,research)
    app.radio(key='backtest-allocation-mode').set_value('fixed_tenth').run()
    app.number_input(key='backtest-entry-percentage').set_value(7.).run()
    app.sidebar.radio[0].set_value('Saved runs').run()
    app.sidebar.radio[0].set_value('Backtest').run()
    for session in (app,_open(monkeypatch,research)):
        assert not session.exception
        assert session.number_input(key='backtest-entry-percentage').value==7.
