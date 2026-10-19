"""Exercise preview, persistence, output routing and immutable processing through the UI."""
from datetime import UTC, date, datetime
import json

import polars as pl
from polars.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from data_pipeline import DataQuery, DataRequest
from research import Research


def daily_frame(symbol):
    days = (2, 3, 4, 5, 8, 9)
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in days], "symbol": [symbol] * len(days),
        **{column: [base + i for i in range(len(days))]
           for column, base in [("open", 100.), ("high", 102.), ("low", 99.), ("close", 101.)]},
        "volume": [1000.] * len(days),
    })


def setup(monkeypatch, tmp_path):
    import dashboard.app as dashboard
    research = Research(tmp_path / "data", tmp_path / "runs")
    raw = research.pipeline.ingest_frame(DataRequest(symbol="AAPL", provider="yahoo", timeframe="1d",
        start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 10, tzinfo=UTC)), daily_frame("AAPL"))
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=15).run()
    app.sidebar.radio[0].set_value("Process data").run()
    assert not app.exception and not app.error
    return research, raw.raw, app


def button(app, label):
    return next(item for item in app.button if item.label == label)


def add_scaling(app, factor):
    app.selectbox(key="process-add").set_value(("scale_prices", "1")).run()
    button(app, "Add processor").click().run()
    next(item for item in app.number_input if item.label == "Scaling factor").set_value(factor).run()
    assert not app.exception and not app.error


def test_preview_save_duplicate_and_market_handoff(monkeypatch, tmp_path):
    research, raw, app = setup(monkeypatch, tmp_path)
    add_scaling(app, 2.)
    app.text_input(key="process-name").set_value("Double prices").run()
    button(app, "Preview processing").click().run()
    assert not app.exception and not app.error
    assert research.pipeline.list_datasets(DataQuery(layer="processed")) == []
    tables = [item.value for item in app.dataframe if "close" in item.value.columns]
    assert tables[0]["close"].iloc[0] == 101.
    assert tables[1]["close"].iloc[0] == 202.
    button(app, "Process and save").click().run()
    assert not app.exception and not app.error
    outputs = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert len(outputs) == 1
    assert_frame_equal(research.pipeline.read_dataset(raw.dataset_id), daily_frame("AAPL").with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC"))))
    button(app, "Process and save").click().run()
    assert not app.exception and not app.error
    assert len(research.pipeline.list_datasets(DataQuery(layer="processed"))) == 1
    assert any("Already processed" in result.value for result in app.success)
    button(app, "Open in Market data").click().run()
    assert not app.exception and not app.error
    assert app.radio(key="browse-layer").value == "processed"
    assert app.selectbox(key="browse-revision").value == outputs[0].dataset_id
    assert "Double prices" in app.selectbox(key="browse-series").options[0]


def test_processing_config_restores_and_backtest_handoff(monkeypatch, tmp_path):
    research, _, app = setup(monkeypatch, tmp_path)
    add_scaling(app, 3.)
    app.text_input(key="process-name").set_value("Triple").run()
    button(app, "Save pipeline recipe").click().run()
    app.sidebar.radio[0].set_value("Saved runs").run()
    app.sidebar.radio[0].set_value("Process data").run()
    assert next(n.value for n in app.number_input if n.label == "Scaling factor") == 3.
    reopened = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=15).run()
    reopened.sidebar.radio[0].set_value("Process data").run()
    assert next(n.value for n in reopened.number_input if n.label == "Scaling factor") == 3.
    assert reopened.text_input(key="process-name").value == "Triple"
    button(reopened, "Process and save").click().run()
    output = research.pipeline.list_datasets(DataQuery(layer="processed"))[0]
    button(reopened, "Use in Backtest").click().run()
    assert not reopened.exception and not reopened.error
    assert reopened.radio(key="backtest-layer").value == "processed"
    assert reopened.selectbox(key="backtest-pipeline").value == output.pipeline_id
    assert reopened.multiselect(key="backtest-tickers").value == ["AAPL"]
    assert reopened.date_input(key="backtest-start").value == date(2024, 1, 2)


def test_ordered_steps_and_recipe_reload(monkeypatch, tmp_path):
    research, _, app = setup(monkeypatch, tmp_path)
    add_scaling(app, 2.)
    button(app, "Add processor").click().run()
    controls = [n for n in app.number_input if n.label == "Scaling factor"]
    controls[1].set_value(3.).run()
    next(b for b in app.button if b.label == "Move up" and not b.disabled).click().run()
    assert [n.value for n in app.number_input if n.label == "Scaling factor"] == [3., 2.]
    button(app, "Save pipeline recipe").click().run()
    button(app, "Remove").click().run()
    assert len(app.number_input) == 1
    button(app, "Load saved pipeline").click().run()
    assert [n.value for n in app.number_input if n.label == "Scaling factor"] == [3., 2.]
    button(app, "Process and save").click().run()
    assert not app.exception and not app.error
    output = research.pipeline.list_datasets(DataQuery(layer="processed"))[0]
    assert [p["config"]["factor"] for p in json.loads(output.processors_json)] == [3., 2.]


def test_invalid_configuration_disables_processing(monkeypatch, tmp_path):
    research, _, app = setup(monkeypatch, tmp_path)
    app.selectbox(key="process-add").set_value(("resample_ohlcv", "1")).run()
    button(app, "Add processor").click().run()
    app.text_area[0].set_value("invalid JSON").run()
    assert app.error
    assert button(app, "Process and save").disabled
    assert button(app, "Preview processing").disabled
    assert research.pipeline.list_datasets(DataQuery(layer="processed")) == []


def test_resampling_schedule_preview_and_partial_failure(monkeypatch, tmp_path):
    import dashboard.app as dashboard
    from datetime import timedelta
    research = Research(tmp_path / "data", tmp_path / "runs")
    for symbol, count in (("AAPL", 7), ("MSFT", 6)):
        frame = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(hours=i) for i in range(count)],
            "symbol": [symbol] * count,
            "open": [100. + i for i in range(count)], "high": [102. + i for i in range(count)],
            "low": [99. + i for i in range(count)], "close": [101. + i for i in range(count)],
            "volume": [1000.] * count,
        })
        research.pipeline.ingest_frame(DataRequest(symbol=symbol, provider="yahoo", timeframe="1h",
            start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 3, tzinfo=UTC)), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=15).run()
    app.sidebar.radio[0].set_value("Process data").run()
    app.multiselect(key="process-tickers").set_value(["AAPL", "MSFT"]).run()
    app.selectbox(key="process-add").set_value(("resample_ohlcv", "1")).run()
    button(app, "Add processor").click().run()
    button(app, "Generate session schedule").click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    schedule = json.loads(app.text_area[0].value)
    assert schedule == [{"label": "2024-01-02", "open": "2024-01-02T14:30:00+00:00", "close": "2024-01-02T21:00:00+00:00"}]
    button(app, "Preview processing").click().run()
    assert not app.exception
    assert any("Incomplete" in error.value for error in app.error)
    assert not research.pipeline.list_datasets(DataQuery(layer="processed"))
    button(app, "Process and save").click().run()
    assert not app.exception
    outputs = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert len(outputs) == 1 and outputs[0].request.symbol == "AAPL"
    assert outputs[0].request.timeframe == "1d" and outputs[0].row_count == 1
    saved = research.pipeline.read_dataset(outputs[0].dataset_id).row(0, named=True)
    assert (saved["open"], saved["high"], saved["low"], saved["close"], saved["volume"]) == (100., 108., 99., 107., 7000.)
    assert any("MSFT" in error.value for error in app.error)


def test_output_handoff_resets_previous_intraday_times(monkeypatch, tmp_path):
    from datetime import time
    import dashboard.app as dashboard
    research = Research(tmp_path / "data", tmp_path / "runs")
    frame = daily_frame("AAPL").head(1).with_columns(pl.lit(datetime(2024, 1, 2, 14, 30, tzinfo=UTC)).alias("timestamp"))
    research.pipeline.ingest_frame(DataRequest(symbol="AAPL", provider="yahoo", timeframe="1h",
        start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 3, tzinfo=UTC)), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=15).run()
    app.sidebar.radio[0].set_value("Backtest").run()
    app.selectbox(key="backtest-timeframe").set_value("1h").run()
    app.time_input(key="backtest-start-time").set_value(time(16))
    app.time_input(key="backtest-end-time").set_value(time(20)).run()
    app.sidebar.radio[0].set_value("Process data").run()
    add_scaling(app, 2.)
    button(app, "Process and save").click().run()
    button(app, "Use in Backtest").click().run()
    assert not app.exception and not app.error
    assert app.time_input(key="backtest-start-time").value == time.min
    assert app.time_input(key="backtest-end-time").value == time.min


def test_custom_session_hours_process_extended_hour_bars(monkeypatch, tmp_path):
    from datetime import timedelta
    import dashboard.app as dashboard
    research = Research(tmp_path / "data", tmp_path / "runs")
    frame = pl.DataFrame({
        "timestamp": [datetime(2024, 10, 11, 8, tzinfo=UTC) + timedelta(hours=i) for i in range(16)],
        "symbol": ["AAPL"] * 16,
        **{column: [base + i for i in range(16)] for column, base in
           (("open", 100.), ("high", 102.), ("low", 99.), ("close", 101.))},
        "volume": [1000.] * 16,
    })
    source = research.pipeline.ingest_frame(DataRequest(symbol="AAPL", provider="yahoo", timeframe="1h",
        start=datetime(2024, 10, 11, tzinfo=UTC), end=datetime(2024, 10, 12, tzinfo=UTC)), frame).raw
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=15).run()
    app.sidebar.radio[0].set_value("Process data").run()
    app.selectbox(key="process-add").set_value(("resample_ohlcv", "1")).run()
    button(app, "Add processor").click().run()
    button(app, "Generate session schedule").click().run()
    button(app, "Preview processing").click().run()
    assert any("outside the explicit session schedule" in error.value for error in app.error)
    next(r for r in app.radio if r.label == "Session hours").set_value("Custom session hours").run()
    button(app, "Generate session schedule").click().run()
    assert not app.exception and not app.error
    button(app, "Preview processing").click().run()
    assert not app.exception and not app.error
    assert not research.pipeline.list_datasets(DataQuery(layer="processed"))
    button(app, "Process and save").click().run()
    assert not app.exception and not app.error
    output, = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert research.pipeline.read_dataset(output.dataset_id).row(0, named=True) == {
        "timestamp": datetime(2024, 10, 11, tzinfo=UTC), "symbol": "AAPL", "open": 100., "high": 117.,
        "low": 99., "close": 116., "volume": 16000.,
    }
    assert research.pipeline.read_dataset(source.dataset_id).height == 16
    app.sidebar.radio[0].set_value("Saved runs").run()
    app.sidebar.radio[0].set_value("Process data").run()
    assert next(r for r in app.radio if r.label == "Session hours").value == "Custom session hours"
    assert json.loads(app.text_area[0].value)[0]["open"] == "2024-10-11T08:00:00+00:00"
