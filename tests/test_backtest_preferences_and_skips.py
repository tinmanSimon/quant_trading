"""Backtest preferences survive restarts; invalid tickers never receive capital."""

from datetime import date
import json

import polars as pl
import pytest

from data_pipeline.processing import ScalePrices
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from test_dashboard_processed_backtest import _open, _small_strategy, _source
from test_initial_split import EveryBar, Broken
from test_research_leading_history import day, ingest, snapshot


@pytest.mark.parametrize("mode,workers", [("per_entry", 1), ("fixed_tenth", 1), ("initial_split", 2)])
def test_bad_tickers_are_excluded_before_allocation_and_errors_are_saved(tmp_path, mode, workers):
    research = Research(tmp_path / "data", tmp_path / "runs")
    ingest(research.pipeline, "GOOD")
    ingest(research.pipeline, "GAP", days=(3, 4, 5, 9, 10, 11))
    kwargs = dict(strategies=[EveryBar()], start=day(5), end=day(12),
                  allocation_mode=mode, max_workers=workers,
                  settings=ExecutionSettings(initial_cash=1000, fractional_shares=True))
    with pytest.raises(PreflightError):
        research.backtest(["GOOD", "GAP", "ABSENT"], **kwargs)
    assert research.list_runs() == []
    clean = research.backtest(["GOOD"], **kwargs)
    run = research.backtest(["GOOD", "GAP", "ABSENT"], skip_problematic_tickers=True, **kwargs)
    restored = research.load_run(run.run_id)
    assert restored.manifest["tickers"] == ["GOOD"]
    assert restored.manifest["requested_tickers"] == ["GOOD", "GAP", "ABSENT"]
    assert set(restored.manifest["sources"]) == {"GOOD"}
    assert set(restored.manifest["skipped_tickers"]) == {"GAP", "ABSENT"}
    errors = restored.manifest["ticker_errors"]
    assert [(e["ticker"], e["code"]) for e in errors] == [("GAP", "missing_bars"), ("ABSENT", "no_data")]
    assert errors[0]["timestamps"] == [day(8).isoformat()]
    assert len(restored.results) == len(clean.results) == 1
    assert restored.results[0].metrics == clean.results[0].metrics
    assert restored.results[0].trades.equals(clean.results[0].trades)


def test_all_invalid_still_reports_errors_and_saves_no_run(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    ingest(research.pipeline, "BAD", days=(3, 4, 6, 8, 9, 10, 11))
    with pytest.raises(PreflightError) as caught:
        research.backtest(["BAD", "ABSENT"], strategies=[EveryBar()], start=day(5), end=day(12),
                          skip_problematic_tickers=True)
    assert {e.code for e in caught.value.report.issues} == {"missing_bars", "unexpected_bars", "no_data"}
    assert research.list_runs() == []


def test_late_join_and_error_exclusion_work_together(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    ingest(research.pipeline, "LATE", days=(8, 9, 10, 11))
    ingest(research.pipeline, "BAD", days=(8, 10, 11))
    selected = snapshot(research.pipeline, ["LATE", "BAD"],
                        skip_problematic_tickers=True, skip_initial_missing_bars=True)
    assert selected.effective_starts == {"LATE": day(10)}
    assert selected.manifest["ticker_errors"][0]["ticker"] == "BAD"
    assert list(selected.frames) == list(selected.bars) == ["LATE"]
    with pytest.raises(ResearchError, match="must be a boolean"):
        snapshot(research.pipeline, skip_problematic_tickers="yes")


def test_corrupt_processed_parent_is_excluded_with_its_lineage(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    good = ingest(research.pipeline, "GOOD")
    bad = ingest(research.pipeline, "BAD")
    processed = research.pipeline.process(good.dataset_id, processors=[ScalePrices(2)])
    research.pipeline.process(bad.dataset_id, processors=[ScalePrices(2)])
    (research.pipeline.store.data_dir / bad.relative_path).write_bytes(b"corrupt")
    selected = snapshot(research.pipeline, ["BAD", "GOOD"], layer="processed",
                        pipeline_id=processed.pipeline_id, skip_problematic_tickers=True)
    assert list(selected.manifest["parent_sources"]) == ["GOOD"]
    assert "Checksum mismatch" in selected.manifest["ticker_errors"][0]["message"]


def test_strategy_errors_still_abort_without_partial_results(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    ingest(research.pipeline, "GOOD")
    with pytest.raises(ValueError, match="strategy failed"):
        research.backtest(["GOOD"], strategies=[Broken()], start=day(5), end=day(12),
                          skip_problematic_tickers=True)
    assert research.list_runs() == []


def test_every_backtest_setting_survives_navigation_and_new_session(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    strategies = [("moving_average_cross", "1"), ("momentum", "1")]
    app.multiselect(key="backtest-strategies").set_value(strategies).run()
    _small_strategy(app)
    app.checkbox(key="backtest-combine").check().run()
    weights = {field.key: .3 if "momentum" in field.key else .7
               for field in app.number_input if field.key.startswith("strategy-weight-")}
    numbers = {"backtest-initial-capital": 4321., "backtest-commission-fixed": 1.2,
               "backtest-commission-bps": 2.3, "backtest-slippage-bps": 3.4,
               "backtest-max-workers": 2, **weights}
    app.radio(key="backtest-allocation-mode").set_value("initial_split").run()
    for key, value in numbers.items():
        app.number_input(key=key).set_value(value)
    app.checkbox(key="backtest-skip-problematic").check()
    app.checkbox(key="backtest-skip-initial-missing").check()
    app.checkbox(key="backtest-fractional-shares").uncheck().run()
    params = {field.key: field.value for field in app.number_input if field.key.startswith("strategy-param-")}
    app.sidebar.radio[0].set_value("Saved runs").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    for session in (app, _open(monkeypatch, research)):
        assert not session.exception and not session.error
        assert session.multiselect(key="backtest-strategies").value == strategies
        assert session.checkbox(key="backtest-combine").value
        assert session.checkbox(key="backtest-skip-problematic").value
        assert session.checkbox(key="backtest-skip-initial-missing").value
        assert not session.checkbox(key="backtest-fractional-shares").value
        assert session.radio(key="backtest-allocation-mode").value == "initial_split"
        assert session.date_input(key="backtest-start").value == date(2024, 1, 5)
        for key, value in (numbers | params).items():
            assert session.number_input(key=key).value == value


def test_json_drafts_and_hidden_strategy_settings_survive_restart(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    key = json.dumps(("moving_average_cross", "1"))
    app.checkbox(key=f"strategy-json-mode-{key}").check().run()
    app.text_area(key=f"strategy-json-{key}").set_value('{"fast": 2, "slow":').run()
    assert app.error  # Preserve incomplete drafts too, without running them.
    app.multiselect(key="backtest-strategies").set_value([("momentum", "1")]).run()
    app = _open(monkeypatch, research)
    app.multiselect(key="backtest-strategies").set_value([("moving_average_cross", "1")]).run()
    assert not app.exception
    assert app.checkbox(key=f"strategy-json-mode-{key}").value
    assert app.text_area(key=f"strategy-json-{key}").value == '{"fast": 2, "slow":'


def test_dashboard_continues_and_shows_skipped_error_report(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    _small_strategy(app)
    app.text_input(key="backtest-extra").set_value("ABSENT")
    app.checkbox(key="backtest-skip-problematic").check()
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    run = app.session_state["backtest-run"]
    assert run.manifest["tickers"] == ["AAPL"]
    assert any("Skipped 1 problematic" in item.value for item in app.warning)
    report = next(item.value for item in app.dataframe if "code" in item.value.columns)
    assert report["ticker"].tolist() == ["ABSENT"]
    assert report["code"].tolist() == ["no_data"]
