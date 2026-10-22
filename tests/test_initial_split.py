"""Fixed ticker budgets, real process parallelism, and allocation controls."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import os
from types import SimpleNamespace

import polars as pl
import pytest

from data_pipeline import DataRequest
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from research.backtesting.initial_split import run_initial_split_backtests, summarize_initial_split
from research.strategies import MovingAverageCross, builtin_registry, load_registry
from research.strategies.bracket import BracketStrategy


BASE = datetime(2024, 1, 2, tzinfo=UTC)


class EveryBar(BracketStrategy):
    name, lookback, config = "test_split_every_bar", 1, {}
    stop_percentage, win_percentage = 0.9, 1.1

    def entry_signal(self, history):
        return True


class Broken(EveryBar):
    name = "test_split_broken"

    def entry_signal(self, history):
        raise ValueError("strategy failed in worker")


def frame(symbol, high=10.0):
    return pl.DataFrame({
        "timestamp": [BASE + timedelta(hours=i) for i in range(4)],
        "symbol": [symbol] * 4, "open": [10.] * 4, "high": [high] * 4,
        "low": [10.] * 4, "close": [10.] * 4, "volume": [1000.] * 4,
    })


def run_split(frames=None, strategies=None, **kwargs):
    strategies = [EveryBar()] if strategies is None else strategies
    return run_initial_split_backtests(
        {"A": frame("A", 12.), "B": frame("B")} if frames is None else frames,
        strategies,
        start=BASE + timedelta(hours=max(item.lookback for item in strategies)), end=BASE + timedelta(hours=4),
        settings=ExecutionSettings(initial_cash=1000), **kwargs,
    )


def test_initial_split_reuses_only_each_tickers_own_cash():
    a, b = run_split()
    assert [a.symbol, b.symbol] == ["A", "B"]
    assert [a.settings.initial_cash, b.settings.initial_cash] == [500, 500]
    assert a.trades.filter(pl.col("side") == "buy")["quantity"].to_list() == [50, 55, 60]
    assert b.trades.filter(pl.col("side") == "buy")["quantity"].to_list() == [50]
    summary = summarize_initial_split([a, b])[0]
    assert summary["initial_cash"] == 1000
    assert summary["final_equity"] == 1165
    assert summary["total_pnl"] == summary["realized_pnl"] == 165
    assert summary["wins"] == 3 and summary["open_positions"] == 1


def test_parallel_results_equal_serial_results_and_restore_environment(monkeypatch):
    monkeypatch.setenv("POLARS_MAX_THREADS", "3")
    frames = {symbol: frame(symbol, 12. if symbol == "A" else 10.) for symbol in ("A", "B", "C")}
    serial = run_split(frames, strategies=[EveryBar(), MovingAverageCross(1, 2)])
    parallel = run_split(frames, strategies=[EveryBar(), MovingAverageCross(1, 2)], max_workers=2)
    assert os.environ["POLARS_MAX_THREADS"] == "3"
    for left, right in zip(serial, parallel):
        assert left.symbol == right.symbol
        assert left.metrics == right.metrics and left.metadata == right.metadata
        for name in ("equity", "trades", "orders", "completed_trades", "open_positions"):
            assert getattr(left, name).equals(getattr(right, name))


def test_initial_split_retains_small_rounding_remainder_in_total_equity():
    results = run_split({symbol: frame(symbol) for symbol in ("A", "B", "C")})
    assert {item.settings.initial_cash for item in results} == {Decimal("333.33333333")}
    summary = summarize_initial_split(results)[0]
    assert Decimal(summary["unallocated_cash_exact"]) == Decimal("0.00000001")
    assert summary["final_equity"] == summary["initial_cash"] == 1000
    assert Decimal(summary["total_pnl_exact"]) == 0


def test_late_ticker_keeps_reserved_budget_until_effective_start():
    results = run_split(effective_starts={"B": BASE + timedelta(hours=3)})
    a, b = results
    assert a.trades.filter(pl.col("side") == "buy")["entry_budget"][0] == 500
    assert b.trades["timestamp"].to_list() == [BASE + timedelta(hours=3)]
    assert b.trades["entry_budget"].to_list() == [500]


@pytest.mark.parametrize(("total", "quantity"), [(10, Decimal("0.5")), (9.98, Decimal(0))])
def test_fractional_split_requires_five_dollars_per_ticker_budget(total, quantity):
    results = run_initial_split_backtests(
        {symbol: frame(symbol) for symbol in ("A", "B")}, [EveryBar()],
        start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=4),
        settings=ExecutionSettings(initial_cash=total, fractional_shares=True),
    )
    for result in results:
        assert sum(result.trades["quantity"]) == quantity
    assert summarize_initial_split(results)[0]["total_pnl"] == 0


def test_each_compared_strategy_has_its_own_total_not_multiplied_by_ticker_count():
    results = run_split(strategies=[EveryBar(), EveryBar(), MovingAverageCross(1, 2)])
    summaries = summarize_initial_split(results)
    assert len(summaries) == 3
    assert [item["initial_cash"] for item in summaries] == [1000] * 3
    assert [item["tickers"] for item in summaries] == [2] * 3


def test_parallel_local_class_has_clear_serial_option():
    class Local(EveryBar):
        pass

    assert len(run_split(strategies=[Local()], max_workers=1)) == 2
    with pytest.raises(ValueError, match="max_workers=1"):
        run_split(strategies=[Local()], max_workers=2)


def test_parallel_loads_private_strategy_from_custom_project_root(tmp_path):
    package = tmp_path / "private_strategies"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "register.py").write_text(
        "from research.strategies.bracket import BracketStrategy\n"
        "class PrivateEntry(BracketStrategy):\n"
        "    name, lookback, config = 'private_parallel_test', 1, {}\n"
        "    stop_percentage, win_percentage = .9, 1.1\n"
        "    def entry_signal(self, history): return True\n"
        "def register_strategies(registry):\n"
        "    registry.register('private_parallel_test', '1', lambda config: PrivateEntry())\n",
        encoding="utf-8",
    )
    strategy = load_registry(project_dir=tmp_path).load(
        {"name": "private_parallel_test", "version": "1", "config": {}},
    )
    serial = run_split(strategies=[strategy])
    parallel = run_split(strategies=[strategy], max_workers=2)
    assert [item.metrics for item in serial] == [item.metrics for item in parallel]


@pytest.fixture
def research(tmp_path):
    registry = builtin_registry()
    registry.register(EveryBar.name, "1", lambda config: EveryBar(), label="Every bar bracket", default_config={})
    app = Research(tmp_path / "data", tmp_path / "runs", strategies=registry)
    for symbol in ("A", "B"):
        data = frame(symbol, 12. if symbol == "A" else 10.).with_columns(
            pl.Series("timestamp", [datetime(2024, 1, day, tzinfo=UTC) for day in (2, 3, 4, 5)]),
        )
        app.pipeline.ingest_frame(DataRequest(symbol, BASE, datetime(2024, 1, 6, tzinfo=UTC), timeframe="1d"), data)
    return app


def test_research_persists_modes_summaries_and_parallel_comparison(research):
    kwargs = dict(strategies=[EveryBar()], start=datetime(2024, 1, 3, tzinfo=UTC),
                  end=datetime(2024, 1, 6, tzinfo=UTC), settings=ExecutionSettings(initial_cash=1000))
    first = research.backtest(["A", "B"], allocation_mode="initial_split", **kwargs)
    second = research.backtest(["A", "B"], allocation_mode="initial_split", max_workers=2, **kwargs)
    loaded = research.load_run(second.run_id)
    assert loaded.manifest["allocation_summary"] == first.manifest["allocation_summary"]
    assert sum(item.settings.initial_cash for item in loaded.results) == 1000
    assert research.compare_runs([first.run_id, second.run_id]).height == 4
    shared = research.backtest(["A", "B"], **kwargs)
    with pytest.raises(ResearchError, match="different"):
        research.compare_runs([first.run_id, shared.run_id])


def test_parallel_failure_does_not_save_partial_run(research):
    with pytest.raises(ValueError, match="strategy failed in worker"):
        research.backtest(["A", "B"], strategies=[EveryBar(), Broken()], allocation_mode="initial_split",
                          max_workers=2, start=datetime(2024, 1, 3, tzinfo=UTC),
                          end=datetime(2024, 1, 6, tzinfo=UTC))
    assert research.list_runs() == []


def test_initial_split_includes_late_ticker_excludes_short_history_and_still_rejects_gap(tmp_path, monkeypatch):
    import research.api as api

    app = Research(tmp_path / "data", tmp_path / "runs", strategies=builtin_registry())
    request_end = datetime(2024, 1, 6, tzinfo=UTC)
    for symbol, days in (("A", (2, 3, 4, 5)), ("B", (3, 4, 5)), ("C", (2, 3, 4, 5)),
                         ("SHORT", (5,)), ("GAP", (2, 3, 5))):
        data = frame(symbol).with_columns(
            pl.Series("timestamp", [datetime(2024, 1, day, tzinfo=UTC) for day in (2, 3, 4, 5)]),
        ).filter(pl.col("timestamp").dt.day().is_in(days))
        app.pipeline.ingest_frame(DataRequest(symbol, BASE, request_end, timeframe="1d"), data)
    kwargs = dict(strategies=[EveryBar()], start=datetime(2024, 1, 3, tzinfo=UTC), end=request_end,
                  settings=ExecutionSettings(initial_cash=1000), allocation_mode="initial_split",
                  skip_initial_missing_bars=True, max_workers=2)
    run = app.backtest(["A", "B", "C", "SHORT", "MISSING"], **kwargs)
    assert [item.symbol for item in run.results] == ["A", "B", "C"]
    assert set(run.manifest["skipped_tickers"]) == {"SHORT", "MISSING"}
    assert run.manifest["effective_starts"]["B"] == datetime(2024, 1, 4, tzinfo=UTC).isoformat()
    assert {item.settings.initial_cash for item in run.results} == {Decimal("333.33333333")}
    summary = run.manifest["allocation_summary"][0]
    assert summary["initial_cash"] == summary["final_equity"] == 1000
    assert Decimal(summary["unallocated_cash_exact"]) == Decimal("0.00000001")
    assert app.load_run(run.run_id).manifest["skipped_tickers"] == run.manifest["skipped_tickers"]

    def unexpected_execution(*args, **kwargs):
        pytest.fail("Workers must not start before all included tickers pass preflight.")

    monkeypatch.setattr(api, "run_initial_split_backtests", unexpected_execution)
    with pytest.raises(PreflightError) as caught:
        app.backtest(["A", "B", "SHORT", "GAP"], **kwargs)
    assert any(issue.ticker == "GAP" and issue.code == "missing_bars" for issue in caught.value.report.issues)
    assert len(app.list_runs()) == 1


@pytest.mark.parametrize("kwargs", [{"allocation_mode": "wrong"}, {"max_workers": 0},
                                    {"max_workers": 33}, {"max_workers": True}])
def test_invalid_allocation_options_fail_before_reading_data(research, kwargs):
    with pytest.raises(ValueError):
        research.backtest(["MISSING"], strategies=[EveryBar()], start=BASE,
                          end=BASE + timedelta(days=10), **kwargs)


def test_dashboard_allocation_controls_route_to_engine(research, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as dashboard

    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    assert app.number_input(key="backtest-max-workers").disabled
    app.radio(key="backtest-allocation-mode").set_value("initial_split").run()
    assert not app.number_input(key="backtest-max-workers").disabled
    app.number_input(key="backtest-max-workers").set_value(1)
    app.checkbox(key="backtest-skip-initial-missing").set_value(True)
    app.multiselect(key="backtest-tickers").set_value(["A", "B"])
    app.date_input(key="backtest-start").set_value(BASE.date())
    app.date_input(key="backtest-end").set_value(datetime(2024, 1, 6).date())
    next(field for field in app.number_input if field.label == "Fast moving average (bars)").set_value(1)
    next(field for field in app.number_input if field.label == "Slow moving average (bars)").set_value(2)
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    run = app.session_state["backtest-run"]
    assert run.manifest["allocation_mode"] == "initial_split"
    assert run.manifest["skip_initial_missing_bars"]
    assert run.manifest["execution_settings"]["fractional_shares"] is True
    assert sum(result.settings.initial_cash for result in run.results) == 10000
    assert any(title.value == "All-ticker totals" for title in app.subheader)


def test_cli_routes_allocation_fractional_and_history_options(tmp_path, monkeypatch, capsys):
    import research.cli as cli

    seen = {}
    def backtest(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(manifest={}, comparison="comparison", run_id="test", path="runs/test")

    monkeypatch.setattr(cli, "Research", lambda *args: SimpleNamespace(backtest=backtest))
    path = tmp_path / "strategies.json"
    path.write_text(json.dumps([{"name": "momentum", "version": "1", "config": {"lookback": 2}}]))
    status = cli.main(["backtest", "--tickers", "A", "B", "--start", "2024-01-03", "--end", "2024-01-06",
                       "--strategies", str(path), "--allocation-mode", "initial_split", "--max-workers", "2",
                       "--skip-initial-missing-bars", "--fractional-shares"])
    assert status == 0, capsys.readouterr().out
    assert seen["allocation_mode"] == "initial_split" and seen["max_workers"] == 2
    assert seen["skip_initial_missing_bars"] and seen["settings"].fractional_shares
    assert seen["settings"].minimum_trade_budget == 5
