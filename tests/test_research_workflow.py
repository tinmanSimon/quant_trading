"""Offline end-to-end research acceptance tests with isolated local storage."""

from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from data_pipeline import DataRequest, FetchQuality, OmittedBar
from data_pipeline.providers import BaseDataProvider, ProviderRegistry
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from research.strategies import MovingAverageCross, Strategy


def date(day):
    return datetime(2024, 1, day, tzinfo=UTC)


def frame(symbol, days=(2, 3, 4, 5, 8, 9)):
    return pl.DataFrame({"timestamp": [date(day) for day in days], "symbol": [symbol] * len(days),
                         "open": [100. + i for i in range(len(days))],
                         "high": [102. + i for i in range(len(days))],
                         "low": [99. + i for i in range(len(days))],
                         "close": [101. + i for i in range(len(days))],
                         "volume": [1000.] * len(days)}).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


def save(app, symbol, days=(2, 3, 4, 5, 8, 9), quality=None):
    request = DataRequest(symbol=symbol, provider="yahoo", timeframe="1d", start=date(2), end=date(10))
    return app.pipeline.store.write_raw(request, frame(symbol, days), quality=quality)


@pytest.fixture
def app(tmp_path):
    return Research(tmp_path / "data", tmp_path / "runs")


def run(app, tickers=("AAPL",), **kwargs):
    return app.backtest(tickers, strategies=[MovingAverageCross(1, 2)], start=date(5), end=date(10), **kwargs)


def test_batch_failure_continues_and_reports_each_ticker(tmp_path):
    class Provider(BaseDataProvider):
        calls = []

        def fetch(self, request):
            self.calls.append(request.symbol)
            if request.symbol == "BAD":
                raise RuntimeError("synthetic provider error")
            return frame(request.symbol)

    provider = Provider()
    app = Research(tmp_path / "data", tmp_path / "runs", providers=ProviderRegistry({"yahoo": provider}))
    report = app.pipeline.fetch_many(["AAPL", "BAD", "MSFT"], start=date(2), end=date(10))
    assert provider.calls == ["AAPL", "BAD", "MSFT"]
    assert [item.status for item in report.outcomes] == ["saved", "failed", "saved"]
    assert report.outcomes[1].error_message == "synthetic provider error"
    assert not report.ok
    assert report.to_frame().height == 3
    assert len(app.pipeline.list_datasets()) == 2


def test_invalid_ticker_does_not_stop_following_fetches(tmp_path):
    class Provider(BaseDataProvider):
        def fetch(self, request):
            return frame(request.symbol)

    app = Research(tmp_path / "data", tmp_path / "runs", providers=ProviderRegistry({"yahoo": Provider()}))
    report = app.pipeline.fetch_many(["", "AAPL"], start=date(2), end=date(10))
    assert [item.status for item in report.outcomes] == ["failed", "saved"]


def test_interrupt_does_not_get_swallowed(tmp_path):
    class Provider(BaseDataProvider):
        def fetch(self, request):
            raise KeyboardInterrupt

    app = Research(tmp_path / "data", tmp_path / "runs", providers=ProviderRegistry({"yahoo": Provider()}))
    with pytest.raises(KeyboardInterrupt):
        app.pipeline.fetch_many(["AAPL", "MSFT"], start=date(2), end=date(10))


def test_missing_second_ticker_aborts_before_any_strategy_call(app):
    save(app, "AAPL")

    class Spy(Strategy):
        name, lookback, calls = "spy", 2, []
        config = {}

        def target_weight(self, history):
            self.calls.append(history.height)
            return 1.

    with pytest.raises(PreflightError) as caught:
        app.backtest(["AAPL", "MSFT"], strategies=[Spy()], start=date(5), end=date(10))
    assert any(issue.ticker == "MSFT" for issue in caught.value.report.issues)
    assert Spy.calls == []
    assert app.list_runs() == []


def test_omitted_bar_in_required_window_blocks(app):
    quality = FetchQuality(status="reported", skip_missing_ohlc=True,
                           omitted_bars=(OmittedBar(date(8), "missing_ohlc"),))
    save(app, "AAPL", (2, 3, 4, 5, 9), quality)
    with pytest.raises(PreflightError) as caught:
        run(app)
    issue = next(item for item in caught.value.report.issues if item.code == "missing_bars")
    assert issue.timestamps == (date(8),)


def test_complete_unknown_quality_is_explicit_not_assumed_reported(app):
    save(app, "AAPL")
    report = app.preflight(["AAPL"], strategies=[MovingAverageCross(1, 2)], start=date(5), end=date(10))
    assert report.ok
    assert "unknown" in report.notes[0]


def test_multi_strategy_multi_ticker_roundtrip_and_raw_unchanged(app):
    datasets = [save(app, symbol) for symbol in ("AAPL", "MSFT")]
    originals = {item.dataset_id: app.pipeline.read_dataset(item.dataset_id) for item in datasets}
    outcome = app.backtest(["AAPL", "MSFT"], strategies=[MovingAverageCross(1, 2), MovingAverageCross(2, 3)],
                           start=date(5), end=date(10), settings=ExecutionSettings(commission_bps=10))
    assert len(outcome.results) == outcome.comparison.height == 4
    loaded = app.load_run(outcome.run_id)
    assert loaded.comparison.equals(outcome.comparison)
    for original, restored in zip(outcome.results, loaded.results):
        assert restored.equity.equals(original.equity)
        assert restored.trades.equals(original.trades)
        assert restored.orders.equals(original.orders)
        assert all(Decimal(value) >= 0 for value in restored.equity["cash_exact"])
        assert restored.trades["timestamp"][0].hour == 14  # Actual NY open, never UTC midnight.
        assert all(decision < filled for decision, filled in restored.trades.select("decision_timestamp", "timestamp").iter_rows())
    for dataset_id, original in originals.items():
        assert app.pipeline.read_dataset(dataset_id).equals(original)
    assert app.list_runs()[0]["run_id"] == outcome.run_id
    assert outcome.manifest["sources"]["AAPL"][0]["checksum_sha256"] == datasets[0].checksum_sha256


def test_failure_during_strategy_execution_publishes_no_partial_run(app):
    save(app, "AAPL")

    class Broken(Strategy):
        name, lookback, config = "broken", 1, {}

        def target_weight(self, history):
            raise RuntimeError("strategy failed")

    with pytest.raises(RuntimeError, match="strategy failed"):
        app.backtest(["AAPL"], strategies=[MovingAverageCross(1, 2), Broken()], start=date(5), end=date(10))
    assert app.list_runs() == []


def test_saved_comparison_accepts_same_inputs_rejects_cost_change(app):
    save(app, "AAPL")
    first, second = run(app), run(app)
    assert app.compare_runs([first.run_id, second.run_id]).height == 2
    third = run(app, settings=ExecutionSettings(commission_fixed=1))
    with pytest.raises(ResearchError, match="different"):
        app.compare_runs([first.run_id, third.run_id])


def test_fine_start_boundary_does_not_include_midnight_bar(app):
    save(app, "AAPL")
    result = app.backtest(["AAPL"], strategies=[MovingAverageCross(1, 2)],
                          start=date(5).replace(microsecond=1), end=date(10))
    assert result.results[0].equity.filter(pl.col("phase") == "close")["timestamp"].to_list() == [date(8), date(9)]


def test_duplicate_normalized_tickers_rejected(app):
    with pytest.raises(ResearchError, match="distinct"):
        run(app, ("aapl", "AAPL"))


def test_off_grid_data_aborts_even_when_all_expected_bars_exist(app):
    save(app, "AAPL", (2, 3, 4, 5, 6, 8, 9))
    with pytest.raises(PreflightError) as caught:
        run(app)
    assert any(issue.code == "unexpected_bars" for issue in caught.value.report.issues)


@pytest.mark.parametrize("data_name,runs_name", [("data", "data"), ("data", "data/runs"), ("runs/data", "runs")])
def test_research_results_cannot_pollute_market_data_root(tmp_path, data_name, runs_name):
    with pytest.raises(ResearchError, match="non-overlapping"):
        Research(tmp_path / data_name, tmp_path / runs_name)
    assert list(tmp_path.iterdir()) == []


def test_all_tickers_use_revisions_pinned_before_simulation(app, monkeypatch):
    import research.api as api

    save(app, "AAPL")
    original = save(app, "MSFT")
    execute = api.run_backtest
    changed = False

    def concurrent_replacement(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            replacement = frame("MSFT").with_columns(*(pl.col(name) * 2 for name in ("open", "high", "low", "close")))
            app.pipeline.store.replace_raw(original.dataset_id, replacement, confirm=original.dataset_id)
        return execute(*args, **kwargs)

    monkeypatch.setattr(api, "run_backtest", concurrent_replacement)
    result = run(app, ("AAPL", "MSFT"))
    msft = next(item for item in result.results if item.symbol == "MSFT")
    assert msft.equity["price"][-1] == frame("MSFT")["close"][-1]
    assert result.manifest["sources"]["MSFT"][0]["dataset_id"] == original.dataset_id


def test_real_cli_runs_and_persists_a_local_backtest(app, tmp_path, capsys):
    import json
    from research.cli import main

    save(app, "AAPL")
    specs = tmp_path / "strategies.json"
    specs.write_text(json.dumps([{"name": "moving_average_cross", "version": "1",
                                 "config": {"fast": 1, "slow": 2}}]), encoding="utf-8")
    status = main(["--data-dir", str(app.pipeline.store.data_dir), "--runs-dir", str(app.runs_dir),
                   "backtest", "--tickers", "AAPL", "--start", "2024-01-05", "--end", "2024-01-10",
                   "--strategies", str(specs)])
    assert status == 0
    assert "Saved run:" in capsys.readouterr().out
    assert len(app.list_runs()) == 1
