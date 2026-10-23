"""Dashboard source selections route backtests to one persisted processing pipeline."""

from datetime import UTC, date, datetime, time, timedelta

import polars as pl
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from data_pipeline.processing import ScalePrices
from research import Research, ResearchError


def _source(research, symbol="AAPL", provider="yahoo", offset=0):
    days = (2, 3, 4, 5, 8, 9)
    frame = pl.DataFrame({
        "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in days],
        "symbol": [symbol] * len(days),
        **{column: [offset + base + index for index in range(len(days))]
           for column, base in [("open", 100.), ("high", 102.), ("low", 99.), ("close", 101.)]},
        "volume": [1000.] * len(days),
    })
    request = DataRequest(symbol, datetime(2024, 1, 2, tzinfo=UTC),
                          datetime(2024, 1, 10, tzinfo=UTC), provider=provider, timeframe="1d")
    return research.pipeline.ingest_frame(request, frame).raw


def _adjusted_source(research):
    from data_pipeline.corporate_actions.models import ActionCoverage, ActionSnapshot, CorporateAction
    from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
    from data_pipeline.processing.processors.price_adjustment import AdjustCorporateActions
    from data_pipeline.schemas import OHLCV_SCHEMA

    daily = ResampleRegularHoursDaily.for_dates(date(2024, 1, 2), date(2024, 1, 9))
    rows = [{"timestamp": session.open + timedelta(minutes=minute), "symbol": "AAPL",
             "open": 100. + index, "high": 102. + index, "low": 99. + index,
             "close": 101. + index, "volume": 1000.}
            for index, session in enumerate(daily.sessions)
            for minute in range(int((session.close - session.open).total_seconds() // 60))]
    request = DataRequest("AAPL", datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 10, tzinfo=UTC),
                          provider="verified-import", timeframe="1m")
    raw = research.pipeline.ingest_frame(request, pl.DataFrame(rows, schema=OHLCV_SCHEMA)).raw
    snapshot = ActionSnapshot(
        "verified-import", date(2024, 1, 31), datetime(2024, 2, 1, tzinfo=UTC),
        (ActionCoverage("AAPL", date(2024, 1, 1), date(2024, 1, 31),
                        source_basis="unadjusted", security_id="security-AAPL"),),
        (CorporateAction("split", "AAPL", "forward_split", datetime(2024, 1, 16, 5, tzinfo=UTC),
                         security_id="security-AAPL", share_ratio=2.),
         CorporateAction("dividend", "AAPL", "cash_dividend", datetime(2024, 1, 17, 5, tzinfo=UTC),
                         security_id="security-AAPL", price_factor=.98)),
    )
    return research.pipeline.process(raw.dataset_id, processors=[AdjustCorporateActions(snapshot), daily])


def _open(monkeypatch, research):
    import dashboard.app as dashboard
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    assert not app.exception and not app.error
    return app


def _small_strategy(app):
    app.date_input(key="backtest-start").set_value(date(2024, 1, 5))
    app.date_input(key="backtest-end").set_value(date(2024, 1, 10))
    next(field for field in app.number_input if field.label == "Fast moving average (bars)").set_value(1)
    next(field for field in app.number_input if field.label == "Slow moving average (bars)").set_value(2)


def test_dashboard_uses_exact_processed_prices_and_clears_old_run(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    raw = _source(research)
    twice = research.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    three = research.pipeline.process(raw.dataset_id, processors=[ScalePrices(3)])
    app = _open(monkeypatch, research)
    app.radio(key="backtest-layer").set_value("processed").run()
    app.selectbox(key="backtest-pipeline").set_value(twice.pipeline_id).run()
    _small_strategy(app)
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    run = app.session_state["backtest-run"]
    assert run.manifest["layer"] == "processed"
    assert run.manifest["pipeline_id"] == twice.pipeline_id
    assert run.manifest["sources"]["AAPL"][0]["dataset_id"] == twice.dataset_id
    assert run.results[0].equity["price"][-1] == 212.
    assert any("transformed price units" in info.value for info in app.info)
    app.selectbox(key="backtest-pipeline").set_value(three.pipeline_id).run()
    assert "backtest-run" not in app.session_state
    app.radio(key="backtest-layer").set_value("raw").run()
    assert not any(field.key == "backtest-pipeline" for field in app.selectbox)
    assert not app.exception and not app.error


def test_pipeline_menu_scopes_symbols_by_provider_and_rejects_missing_processed_ticker(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    aapl = _source(research)
    msft = _source(research, "MSFT")
    other = _source(research, "OtherSymbol", provider="other")
    twice = research.pipeline.process(aapl.dataset_id, processors=[ScalePrices(2)])
    three = research.pipeline.process(msft.dataset_id, processors=[ScalePrices(3)])
    four = research.pipeline.process(other.dataset_id, processors=[ScalePrices(4)])
    app = _open(monkeypatch, research)
    app.radio(key="backtest-layer").set_value("processed").run()
    app.selectbox(key="backtest-provider").set_value("yahoo").run()
    app.selectbox(key="backtest-pipeline").set_value(twice.pipeline_id).run()
    assert app.multiselect(key="backtest-tickers").options == ["AAPL"]
    assert app.selectbox(key="backtest-timeframe").options == ["1d"]
    app.selectbox(key="backtest-pipeline").set_value(three.pipeline_id).run()
    assert app.multiselect(key="backtest-tickers").options == ["MSFT"]
    app.selectbox(key="backtest-provider").set_value("other").run()
    assert app.selectbox(key="backtest-pipeline").value == four.pipeline_id
    assert app.multiselect(key="backtest-tickers").options == ["OtherSymbol"]
    app.selectbox(key="backtest-provider").set_value("yahoo").run()
    app.selectbox(key="backtest-pipeline").set_value(twice.pipeline_id).run()
    app.multiselect(key="backtest-tickers").set_value(["AAPL"])
    app.text_input(key="backtest-extra").set_value("MSFT")
    _small_strategy(app)
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception
    assert any("required data did not pass preflight" in error.value for error in app.error)
    assert research.list_runs() == []


def test_processed_data_selection_persists_after_navigation_and_new_session(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    aapl = _source(research)
    msft = _source(research, "MSFT")
    twice = research.pipeline.process(aapl.dataset_id, processors=[ScalePrices(2)])
    research.pipeline.process(msft.dataset_id, processors=[ScalePrices(2)])
    app = _open(monkeypatch, research)
    app.radio(key="backtest-layer").set_value("processed").run()
    app.selectbox(key="backtest-pipeline").set_value(twice.pipeline_id).run()
    app.radio(key="backtest-research-mode").set_value("adjusted_price").run()
    app.multiselect(key="backtest-tickers").set_value(["MSFT"])
    app.text_input(key="backtest-extra").set_value("TSLA")
    app.date_input(key="backtest-start").set_value(date(2024, 1, 4))
    app.date_input(key="backtest-end").set_value(date(2024, 1, 9)).run()
    app.sidebar.radio[0].set_value("Saved runs").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    for session in (app, _open(monkeypatch, research)):
        assert not session.exception and not session.error
        assert session.radio(key="backtest-layer").value == "processed"
        assert session.selectbox(key="backtest-pipeline").value == twice.pipeline_id
        assert session.radio(key="backtest-research-mode").value == "adjusted_price"
        assert session.multiselect(key="backtest-tickers").value == ["MSFT"]
        assert session.text_input(key="backtest-extra").value == "TSLA"
        assert session.date_input(key="backtest-start").value == date(2024, 1, 4)
        assert session.date_input(key="backtest-end").value == date(2024, 1, 9)


def test_intraday_backtest_times_persist_and_empty_processed_store_has_no_run_form(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    app.selectbox(key="backtest-timeframe").set_value("1h").run()
    app.time_input(key="backtest-start-time").set_value(time(14, 30))
    app.time_input(key="backtest-end-time").set_value(time(21)).run()
    app.sidebar.radio[0].set_value("Saved runs").run()
    app.sidebar.radio[0].set_value("Backtest").run()
    assert app.time_input(key="backtest-start-time").value == time(14, 30)
    assert app.time_input(key="backtest-end-time").value == time(21)
    app.radio(key="backtest-layer").set_value("processed").run()
    assert not app.exception and not app.error
    assert any("No active processed datasets" in info.value for info in app.info)
    assert not any(button.label == "Validate all data and run" for button in app.button)


def test_adjusted_research_mode_runs_adjusted_pipeline_and_invalidates_strict_result(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    processed = _adjusted_source(research)
    app = _open(monkeypatch, research)
    app.radio(key="backtest-layer").set_value("processed").run()
    assert app.radio(key="backtest-research-mode").value == "strict"
    _small_strategy(app)
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception
    assert any("required data did not pass preflight" in error.value for error in app.error)
    assert research.list_runs() == []
    app.radio(key="backtest-research-mode").set_value("adjusted_price").run()
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    run = app.session_state["backtest-run"]
    assert run.manifest["research_mode"] == "adjusted_price"
    assert run.manifest["pipeline_id"] == processed.pipeline_id
    assert run.results[0].equity["price"][-1] == 106. * .5 * .98
    assert any("Research mode: Adjusted-price research" in item.value for item in app.caption)
    assert any("without additional cash credits" in item.value for item in app.info)
    assert not any("returns exclude dividends" in item.value for item in app.caption)
    app.radio(key="backtest-research-mode").set_value("strict").run()
    assert "backtest-run" not in app.session_state


def test_raw_source_forces_strict_after_adjusted_mode_selection(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    raw = _source(research)
    research.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        raise ResearchError("intentional test stop")

    monkeypatch.setattr(research, "backtest", record)
    app = _open(monkeypatch, research)
    app.radio(key="backtest-layer").set_value("processed").run()
    app.radio(key="backtest-research-mode").set_value("adjusted_price").run()
    app.radio(key="backtest-layer").set_value("raw").run()
    assert not any(field.key == "backtest-research-mode" for field in app.radio)
    _small_strategy(app)
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception
    assert calls[0]["research_mode"] == "strict"
