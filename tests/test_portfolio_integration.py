"""Shared accounts use the existing data, artifact, and dashboard workflows."""

from datetime import UTC, datetime
from hashlib import sha256
import json

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from data_pipeline.processing import ScalePrices
from research import Research, ResearchError
from research.backtesting import BacktestResult, ExecutionSettings
from research.runs import comparison_identity, load_run, save_run
from research.strategies import MovingAverageCross, builtin_registry
from research.strategies.bracket import BracketStrategy


class EntryEveryBar(BracketStrategy):
    name, lookback, config = "entry_every_bar", 1, {}
    stop_percentage, win_percentage = 0.95, 1.05

    def entry_signal(self, history):
        return True


def _date(day):
    return datetime(2024, 1, day, tzinfo=UTC)


@pytest.fixture
def research(tmp_path):
    registry = builtin_registry()
    registry.register(EntryEveryBar.name, "1", lambda config: EntryEveryBar(),
                      label="Every bar bracket", default_config={})
    app = Research(tmp_path / "data", tmp_path / "runs", strategies=registry)
    for symbol, high in (("AAPL", 106.), ("MSFT", 102.)):
        days = (2, 3, 4, 5, 8, 9)
        frame = pl.DataFrame({
            "timestamp": [_date(day) for day in days], "symbol": [symbol] * len(days),
            "open": [100.] * len(days), "high": [high] * len(days),
            "low": [99.] * len(days), "close": [100.] * len(days),
            "volume": [1000.] * len(days),
        })
        app.pipeline.ingest_frame(DataRequest(symbol, _date(2), _date(10), timeframe="1d"), frame)
    return app


def _run(research, strategies=None):
    return research.backtest(["AAPL", "MSFT"], strategies=strategies or [EntryEveryBar()],
                             start=_date(5), end=_date(10), settings=ExecutionSettings(initial_cash=1000))


@pytest.mark.parametrize("mixed", [False, True])
def test_shared_strategy_runs_once_and_retains_roundtrip_details(research, mixed):
    strategies = [EntryEveryBar()]
    if mixed:
        strategies.insert(0, MovingAverageCross(1, 2))
    run = _run(research, strategies)
    assert len(run.results) == (3 if mixed else 1)
    portfolio = next(result for result in run.results if result.symbol == "PORTFOLIO")
    assert portfolio.metadata["tickers"] == ["AAPL", "MSFT"]
    assert portfolio.metrics["initial_cash"] == 1000
    assert portfolio.metadata["layer"] == "raw"
    assert portfolio.completed_trades["symbol"].unique().to_list() == ["AAPL"]
    assert portfolio.open_positions["symbol"].to_list() == ["MSFT"]
    assert run.manifest["account_mode"] == ("mixed_independent_and_shared_usd_accounts" if mixed
                                             else "shared_long_only_usd_portfolios")
    restored = research.load_run(run.run_id)
    assert restored.comparison.equals(run.comparison)
    for before, after in zip(run.results, restored.results):
        assert after.completed_trades.equals(before.completed_trades)
        assert after.open_positions.equals(before.open_positions)
        assert after.metrics == before.metrics
    assert research.compare_runs([run.run_id, run.run_id]).height == 2 * len(run.results)


def test_account_mode_remains_part_of_saved_comparison_identity(research):
    run = _run(research)
    changed = dict(run.manifest, account_mode="independent_long_only_usd_accounts")
    assert comparison_identity(run.manifest) != comparison_identity(changed)


def test_shared_portfolio_retains_processed_pipeline_and_price_basis(research):
    pipeline_id = None
    for dataset in research.pipeline.list_datasets():
        processed = research.pipeline.process(dataset.dataset_id, processors=[ScalePrices(2)])
        pipeline_id = processed.pipeline_id
    run = research.backtest(["AAPL", "MSFT"], strategies=[EntryEveryBar()],
                             start=_date(5), end=_date(10), layer="processed", pipeline_id=pipeline_id)
    result = research.load_run(run.run_id).results[0]
    assert result.metadata["pipeline_id"] == pipeline_id
    assert result.metadata["price_basis"] == "processed_transformed_prices"
    assert result.completed_trades["entry_open"].unique().to_list() == [200.]


@pytest.mark.parametrize("name", ["completed_trades", "open_positions"])
def test_extra_trade_artifacts_are_checksummed(research, name):
    run = _run(research)
    artifact = run.path / run.manifest["results"][0]["files"][name]["path"]
    with artifact.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ResearchError, match="checksum"):
        research.load_run(run.run_id)


def test_empty_trade_schemas_and_legacy_missing_artifacts_roundtrip(tmp_path):
    original = BacktestResult(
        "AAPL", {"name": "test", "version": "1", "config": {}}, ExecutionSettings(),
        pl.DataFrame({"equity": [10000.]}), pl.DataFrame(), pl.DataFrame(), {},
        completed_trades=pl.DataFrame(schema={"outcome": pl.String, "net_pnl": pl.Float64}),
        open_positions=pl.DataFrame(schema={"symbol": pl.String, "quantity": pl.Int64}),
    )
    run = save_run(tmp_path, [original], {})
    restored = load_run(tmp_path, run.run_id).results[0]
    assert restored.completed_trades.equals(original.completed_trades)
    assert restored.open_positions.equals(original.open_positions)
    # Version 1 runs saved before shared accounts have only the three ledgers.
    manifest = run.manifest
    for name in ("completed_trades", "open_positions"):
        item = manifest["results"][0]["files"].pop(name)
        (run.path / item["path"]).unlink()
    path = run.path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (run.path / "manifest.sha256").write_text(sha256(path.read_bytes()).hexdigest(), encoding="ascii")
    legacy = load_run(tmp_path, run.run_id).results[0]
    assert legacy.completed_trades.shape == legacy.open_positions.shape == (0, 0)


def test_dashboard_shows_shared_cash_outcomes_and_position_details(research, monkeypatch):
    import dashboard.pages.backtest as page

    run = _run(research)
    monkeypatch.setattr(page, "_test_run", run, raising=False)
    app = AppTest.from_string(
        "from dashboard.pages.backtest import _show_result, _test_run\n_show_result(_test_run)"
    ).run()
    assert not app.exception and not app.error
    assert any("one shared cash account" in caption.value for caption in app.caption)
    assert {metric.label for metric in app.metric} == {
        "Wins", "Losses", "Inconclusive stops", "Realized P&L", "Unrealized P&L", "Total P&L",
    }
    assert any("Completed trades" in item.value for item in app.markdown)
    assert any("Open positions" in item.value for item in app.markdown)


def test_dashboard_reports_unsupported_weighted_bracket_without_crashing(research, monkeypatch):
    import dashboard.app as dashboard

    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    next(field for field in app.multiselect if field.label == "Strategies").set_value(
        [("moving_average_cross", "1"), (EntryEveryBar.name, "1")]
    ).run()
    app.date_input(key="backtest-start").set_value(_date(5).date())
    app.date_input(key="backtest-end").set_value(_date(10).date())
    next(field for field in app.checkbox if field.label == "Also test a weighted combination").check().run()
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception
    assert any("Backtest aborted:" in item.value and "bracket" in item.value.lower() for item in app.error)
    assert research.list_runs() == []


def test_large_mixed_comparison_keeps_portfolio_outcome_metrics():
    from research.backtesting import compare_results

    independent = BacktestResult(
        "A", {"name": "allocation", "version": "1", "config": {}}, ExecutionSettings(),
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {"final_equity": 10000.},
    )
    portfolio = BacktestResult(
        "PORTFOLIO", {"name": "bracket", "version": "1", "config": {}}, ExecutionSettings(),
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {"final_equity": 11000., "wins": 12},
    )
    comparison = compare_results([independent] * 101 + [portfolio])
    assert comparison["wins"][-1] == 12
