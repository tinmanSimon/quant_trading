"""Dashboard source selections route backtests to one persisted processing pipeline."""

from datetime import UTC, date, datetime, time

import polars as pl
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from data_pipeline.processing import ScalePrices
from research import Research


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
