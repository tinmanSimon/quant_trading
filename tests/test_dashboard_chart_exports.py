"""Original chart data is prepared on demand and reverified before download."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

import dashboard.app as dashboard
from data_pipeline import DataIntegrityError, DataQuery, DataRequest, DatasetNotFoundError
from research import Research


@pytest.fixture
def chart_source(tmp_path, sample_ohlcv_frame):
    research = Research(tmp_path / "data", tmp_path / "runs")
    request = DataRequest("AAPL", sample_ohlcv_frame["timestamp"][0],
                          sample_ohlcv_frame["timestamp"][-1] + timedelta(hours=1))
    dataset = research.pipeline.ingest_frame(request, sample_ohlcv_frame).raw
    query = DataQuery(provider="yahoo", symbol="AAPL", timeframe="1h",
                      start=request.start, end=request.end)
    return research, dataset, query, sample_ohlcv_frame


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_csv_rechecks_active_data_after_replacement_and_partial_deletion(chart_source):
    research, dataset, query, frame = chart_source
    pipeline = research.pipeline
    assert [float(row["volume"]) for row in _rows(dashboard._chart_csv(pipeline, query))] == frame["volume"].to_list()

    changed = frame.with_columns(pl.col("volume") + 0.125)
    pipeline.store.replace_raw(dataset.dataset_id, changed, confirm=dataset.dataset_id)
    assert [float(row["volume"]) for row in _rows(dashboard._chart_csv(pipeline, query))] == changed["volume"].to_list()
    plan = pipeline.plan_delete(replace(query, start=frame["timestamp"][1], end=frame["timestamp"][2]))
    pipeline.delete(plan, confirm=plan.operation_id)
    assert [float(row["volume"]) for row in _rows(dashboard._chart_csv(pipeline, query))] == [
        changed["volume"][0], changed["volume"][2],
    ]


@pytest.mark.parametrize("explicit", [False, True])
def test_csv_preserves_precise_bounds_and_original_ohlcv(chart_source, explicit):
    research, dataset, query, frame = chart_source
    selected = replace(query, start=frame["timestamp"][0] + timedelta(microseconds=1),
                       end=frame["timestamp"][2] + timedelta(microseconds=1))
    rows = _rows(dashboard._chart_csv(research.pipeline, selected, dataset.dataset_id if explicit else None))
    assert len(rows) == 2
    assert [row["symbol"] for row in rows] == ["AAPL", "AAPL"]
    for column in ("open", "high", "low", "close", "volume"):
        assert [float(row[column]) for row in rows] == frame[column].tail(2).to_list()


@pytest.mark.parametrize("explicit", [False, True])
def test_csv_checks_current_file_integrity_after_an_earlier_success(chart_source, explicit):
    research, dataset, query, _ = chart_source
    dataset_id = dataset.dataset_id if explicit else None
    assert dashboard._chart_csv(research.pipeline, query, dataset_id)
    path = research.pipeline.store.data_dir / dataset.relative_path
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 1
    path.write_bytes(content)
    with pytest.raises(DataIntegrityError):
        dashboard._chart_csv(research.pipeline, query, dataset_id)


def test_csv_of_deleted_exact_revision_fails_and_active_query_is_empty(chart_source):
    research, dataset, query, _ = chart_source
    assert dashboard._chart_csv(research.pipeline, query, dataset.dataset_id)
    plan = research.pipeline.plan_delete(query)
    research.pipeline.delete(plan, confirm=plan.operation_id)
    with pytest.raises(DatasetNotFoundError):
        dashboard._chart_csv(research.pipeline, query, dataset.dataset_id)
    assert _rows(dashboard._chart_csv(research.pipeline, query)) == []


def test_dashboard_defers_csv_and_table_preparation_until_requested(chart_source, monkeypatch):
    research, dataset, _, frame = chart_source
    calls, tables, buttons = [], [], []
    original_csv, original_table = dashboard._chart_csv, dashboard.bar_table

    def export(*args):
        calls.append(args)
        return original_csv(*args)

    def table(*args, **kwargs):
        tables.append(args[0].height)
        return original_table(*args, **kwargs)

    def download(label, data, **kwargs):
        buttons.append(data)
        return False

    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    monkeypatch.setattr(dashboard, "_chart_csv", export)
    monkeypatch.setattr(dashboard, "bar_table", table)
    monkeypatch.setattr(dashboard.st, "download_button", download)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    assert not app.exception and not app.error
    assert len(buttons) == 1 and callable(buttons[0])
    assert calls == tables == []
    # The deferred callback must observe storage at click time, not the frame
    # that happened to be displayed when Streamlit created the button.
    changed = frame.with_columns(pl.col("volume") + 0.125)
    research.pipeline.store.replace_raw(dataset.dataset_id, changed, confirm=dataset.dataset_id)
    assert [float(row["volume"]) for row in _rows(buttons[0]())] == changed["volume"].to_list()
    assert len(calls) == 1
    app.checkbox(key="browse-show-original-bars").check().run()
    assert not app.exception and not app.error
    assert tables == [3]
    assert len(calls) == 1


def test_original_table_pages_are_limited_to_500_rows_and_keep_source_values(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    start = datetime(2024, 1, 2, tzinfo=UTC)
    count = 1203
    frame = pl.DataFrame({
        "timestamp": [start + timedelta(minutes=index) for index in range(count)],
        "symbol": ["AAPL"] * count,
        "open": [10.] * count, "high": [11.] * count,
        "low": [9.] * count, "close": [10.5] * count,
        "volume": [index + 0.125 for index in range(count)],
    })
    research.pipeline.ingest_frame(DataRequest("AAPL", start, start + timedelta(days=1), timeframe="1m"), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    assert not app.exception and not app.error
    assert not [item for item in app.dataframe if "close" in item.value.columns]
    app.checkbox(key="browse-show-original-bars").check().run()
    assert not app.exception and not app.error
    page = next(field for field in app.number_input if field.label == "Original bar page")
    assert page.value == 3
    rows = next(item.value for item in app.dataframe if "close" in item.value.columns)
    assert len(rows) == 203 and rows["volume"].iloc[0] == 1000.125
    page.set_value(1).run()
    rows = next(item.value for item in app.dataframe if "close" in item.value.columns)
    assert len(rows) == 500 and rows["volume"].iloc[0] == 0.125 and rows["volume"].iloc[-1] == 499.125
    next(field for field in app.number_input if field.label == "Original bar page").set_value(2).run()
    rows = next(item.value for item in app.dataframe if "close" in item.value.columns)
    assert len(rows) == 500 and rows["volume"].iloc[0] == 500.125 and rows["volume"].iloc[-1] == 999.125
