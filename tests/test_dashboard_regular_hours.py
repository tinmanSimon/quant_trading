"""The regular-session processor is available with a fixed exchange schedule."""
from datetime import UTC, date, datetime, timedelta
import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from data_pipeline import DataQuery, DataRequest
from research import Research


def _button(app, label):
    return next(item for item in app.button if item.label == label)


@pytest.mark.parametrize("day,minutes,expected_bars,missing", [
    (date(2024, 1, 2), 390, 13, False),
    (date(2024, 11, 29), 210, 7, False),
    (date(2024, 1, 2), 390, 13, True),
])
def test_regular_hours_processor_preview_save_and_restore(monkeypatch, tmp_path, day, minutes, expected_bars, missing):
    import dashboard.app as dashboard

    opening = datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=14, minutes=30)
    # The minute before opening and the one at closing must both be discarded.
    prices = [10000.] + [100. + i for i in range(minutes)] + [10000.]
    frame = pl.DataFrame({
        "timestamp": [opening + timedelta(minutes=i) for i in range(-1, minutes + 1)],
        "symbol": ["AAPL"] * len(prices), "open": prices,
        "high": [price + 2 for price in prices], "low": [price - 1 for price in prices],
        "close": [price + 1 for price in prices], "volume": [10.] * len(prices),
    })
    if missing:
        frame = frame.filter(pl.col("timestamp") != opening + timedelta(minutes=7))
    research = Research(tmp_path / "data", tmp_path / "runs")
    raw = research.pipeline.ingest_frame(DataRequest(
        symbol="AAPL", provider="databento-xnas.itch", timeframe="1m",
        start=datetime.combine(day, datetime.min.time(), UTC),
        end=datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC),
    ), frame).raw
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    script = "from dashboard.app import main\nmain()"
    app = AppTest.from_string(script, default_timeout=15).run()
    app.sidebar.radio[0].set_value("Process data").run()
    assert "Regular hours: 1m to 30m (v1)" in app.selectbox(key="process-add").options
    app.selectbox(key="process-add").set_value(("resample_regular_hours_30m", "1")).run()
    _button(app, "Add processor").click().run()
    assert not app.exception
    assert not any(field.label == "Session hours" for field in app.radio)
    assert not any(field.label == "Exchange calendar" for field in app.selectbox)
    assert not app.time_input
    _button(app, "Generate session schedule").click().run()
    assert not app.exception and not app.error, [item.value for item in app.error]
    schedule = json.loads(app.text_area[0].value)
    assert schedule == [{
        "label": day.isoformat(), "open": opening.isoformat(),
        "close": (opening + timedelta(minutes=minutes)).isoformat(),
    }]
    if missing:
        next(field for field in app.selectbox if field.label == "Incomplete intervals").set_value("aggregate_available").run()
    _button(app, "Preview processing").click().run()
    assert not app.exception and not app.error
    if missing:
        assert any("Aggregated available minutes in 1" in item.value for item in app.warning)
    assert not research.pipeline.list_datasets(DataQuery(layer="processed"))
    tables = [table.value for table in app.dataframe if "close" in table.value.columns]
    assert len(tables[1]) == expected_bars
    assert tables[1]["open"].iloc[0] == 100.
    assert tables[1]["high"].max() < 10000.
    _button(app, "Process and save").click().run()
    assert not app.exception and not app.error
    output, = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert output.request.timeframe == "30m" and output.row_count == expected_bars
    actual = research.pipeline.read_dataset(output.dataset_id)
    assert actual["timestamp"].to_list() == [opening + timedelta(minutes=i * 30) for i in range(expected_bars)]
    assert actual.row(0, named=True) == {
        "timestamp": opening, "symbol": "AAPL", "open": 100., "high": 131.,
        "low": 99., "close": 130., "volume": 290. if missing else 300.,
    }
    assert actual["close"][-1] == 100. + minutes
    assert_frame_equal(research.pipeline.read_dataset(raw.dataset_id),
                       frame.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC"))))
    restored = AppTest.from_string(script, default_timeout=15).run()
    restored.sidebar.radio[0].set_value("Process data").run()
    assert not restored.exception and not restored.error
    assert json.loads(restored.text_area[0].value) == schedule
    assert not any(field.label == "Session hours" for field in restored.radio)
    assert not any(field.label == "Exchange calendar" for field in restored.selectbox)
    if missing:
        assert next(field for field in restored.selectbox if field.label == "Incomplete intervals").value == "aggregate_available"
        assert any("Aggregated available minutes in 1" in item.value for item in restored.warning)


def test_regular_schedule_rejects_custom_hours_in_edited_json(tmp_path):
    """Editable pinned JSON must not bypass the regular-hours restriction."""
    from data_pipeline.processing import discover_processors

    definition = discover_processors().definition("resample_regular_hours_30m", "1")
    schema = json.dumps(definition.schema)
    script = f'''
import json
import streamlit as st
from pathlib import Path
from types import SimpleNamespace
from dashboard.components.schema_controls import processor_controls
from dashboard.workspace import MarketMemory
from data_pipeline.processing import discover_processors
from data_pipeline.exceptions import ProcessingError
if "memory" not in st.session_state:
    st.session_state["memory"] = MarketMemory("regular-controls", Path({str(tmp_path)!r}))
memory = st.session_state["memory"]
try:
    config = processor_controls(SimpleNamespace(schema=json.loads({schema!r})), memory, "regular")
    discover_processors().configure("resample_regular_hours_30m", "1", config)
except (ValueError, ProcessingError) as error:
    st.error(str(error))
memory.save()
'''
    app = AppTest.from_string(script, default_timeout=15).run()
    app.date_input(key="regular-sessions-first").set_value(date(2024, 1, 2))
    app.date_input(key="regular-sessions-last").set_value(date(2024, 1, 2))
    app.button(key="regular-sessions-generate").click().run()
    assert not app.exception and not app.error
    schedule = json.loads(app.text_area[0].value)
    schedule[0]["open"] = "2024-01-02T09:00:00+00:00"
    app.text_area[0].set_value(json.dumps(schedule)).run()
    assert not app.exception
    assert app.error
