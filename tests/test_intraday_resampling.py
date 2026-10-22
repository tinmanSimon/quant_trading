"""Minute-derived OHLCV, missing-data policies and execution-calendar agreement."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataQuery, DataRequest
from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import DataContract, Pipeline, ResampleIntraday, TradingSession, discover_processors, load_pipeline
from data_pipeline.processing.processors.intraday import ResamplingWarning
from data_pipeline.schemas import OHLCV_SCHEMA
from dashboard.components.session_schedules import generate_schedule
from research import Research
from research.instruments import Instrument, expected_bars
from research.strategies import MovingAverageCross


def session(day=date(2024, 1, 2), minutes=60):
    start = datetime.combine(day, datetime.min.time(), UTC).replace(hour=14, minute=30)
    return TradingSession(day, start, start + timedelta(minutes=minutes))


def minute_frame(sessions, symbol="AAPL"):
    stamps = []
    for item in sessions:
        stamps.extend(item.open + timedelta(minutes=i) for i in range(int((item.close - item.open).total_seconds() / 60)))
    return pl.DataFrame({
        "timestamp": stamps, "symbol": [symbol] * len(stamps),
        "open": [100 + i for i in range(len(stamps))],
        "high": [102 + i for i in range(len(stamps))],
        "low": [99 + i for i in range(len(stamps))],
        "close": [101 + i for i in range(len(stamps))],
        "volume": [10] * len(stamps),
    }, schema=OHLCV_SCHEMA)


def test_hand_calculated_30m_ohlcv_and_extended_hours_are_excluded():
    schedule = (session(),)
    frame = minute_frame(schedule)
    outside = frame.head(1).with_columns(pl.lit(schedule[0].open - timedelta(minutes=1)).alias("timestamp"))
    with pytest.warns(ResamplingWarning, match="Excluded 1"):
        result = ResampleIntraday(schedule).transform(pl.concat([outside.cast(OHLCV_SCHEMA), frame]), DataContract("1m"))
    assert result.contract == DataContract("30m")
    assert result.frame.to_dicts() == [
        {"timestamp": schedule[0].open, "symbol": "AAPL", "open": 100., "high": 131., "low": 99., "close": 130., "volume": 300.},
        {"timestamp": schedule[0].open + timedelta(minutes=30), "symbol": "AAPL", "open": 130., "high": 161., "low": 129., "close": 160., "volume": 300.},
    ]


@pytest.mark.parametrize("missing_index", [0, 11, 29, 30, 59])
def test_missing_first_middle_or_last_minute_raises_or_drops_whole_interval(missing_index):
    schedule = (session(),)
    frame = minute_frame(schedule)
    incomplete = frame.filter(pl.col("timestamp") != frame["timestamp"][missing_index])
    with pytest.raises(ProcessingError, match="1 incomplete intraday intervals"):
        Pipeline([ResampleIntraday(schedule)]).run(incomplete, DataContract("1m"))
    processor = ResampleIntraday(schedule, incomplete="drop")
    report = processor.quality_report(incomplete)
    assert report.height == 1
    assert report["missing_minutes"][0] == 1
    with pytest.warns(ResamplingWarning, match="Dropped 1 incomplete 30m intervals"):
        result = processor.transform(incomplete, DataContract("1m")).frame
    assert result.height == 1
    assert result["timestamp"][0] == schedule[0].open + timedelta(minutes=30 if missing_index < 30 else 0)


def test_whole_missing_session_and_multiple_symbols_reported_separately():
    schedule = (session(), session(date(2024, 1, 3)))
    frame = pl.concat([minute_frame(schedule), minute_frame(schedule[1:], "MSFT")])
    processor = ResampleIntraday(schedule, incomplete="drop")
    assert processor.quality_report(frame).select("symbol", "missing_minutes").to_dicts() == [
        {"symbol": "MSFT", "missing_minutes": 30}, {"symbol": "MSFT", "missing_minutes": 30},
    ]
    with pytest.warns(ResamplingWarning, match="Dropped 2"):
        result = processor.transform(frame, DataContract("1m")).frame
    assert result.filter(pl.col("symbol") == "AAPL").height == 4
    assert result.filter(pl.col("symbol") == "MSFT").height == 2


def test_drop_all_incomplete_intervals_is_an_error():
    schedule = (session(),)
    with pytest.warns(ResamplingWarning, match="Dropped 2"):
        with pytest.raises(ProcessingError, match="No complete intraday intervals"):
            ResampleIntraday(schedule, incomplete="drop").transform(minute_frame(schedule).head(1), DataContract("1m"))


@pytest.mark.parametrize("timeframe", ["30m", "1h"])
def test_dst_early_close_and_short_final_hour_agree_with_backtest(timeframe):
    sessions = []
    for day in (date(2025, 3, 7), date(2025, 3, 10), date(2025, 11, 28)):
        sessions.extend(TradingSession.from_config(item) for item in generate_schedule("XNYS", day, day))
    processor = ResampleIntraday(tuple(sessions), target_timeframe=timeframe)
    frame = processor.transform(minute_frame(sessions), DataContract("1m")).frame
    assert sessions[0].open.hour == 14 and sessions[1].open.hour == 13
    assert sessions[-1].close.hour == 18  # Thanksgiving Friday, 13:00 New York.
    expected = []
    for item in sessions:
        expected.extend(expected_bars(Instrument("AAPL"), start=item.open, end=item.close, timeframe=timeframe))
    assert frame["timestamp"].to_list() == [bar.timestamp for bar in expected]
    processor.validate_backtest_calendar(expected, input_contract=DataContract("1m"), output_contract=DataContract(timeframe))
    if timeframe == "1h":
        assert expected[-1].close - expected[-1].open == timedelta(minutes=30)
        assert frame["volume"][-1] == 300.
    with pytest.raises(ProcessingError, match="execution calendar"):
        processor.validate_backtest_calendar([replace(expected[0], close=expected[0].close + timedelta(minutes=1))],
                                             input_contract=DataContract("1m"), output_contract=DataContract(timeframe))


def test_minute_grid_contract_config_and_registry_roundtrip():
    schedule = (session(),)
    frame = minute_frame(schedule)
    processor = ResampleIntraday(schedule)
    with pytest.raises(ProcessingError, match="off the whole-minute grid"):
        processor.transform(frame.with_columns(pl.col("timestamp") + timedelta(seconds=1)), DataContract("1m"))
    with pytest.raises(ProcessingError, match="1m input contract"):
        processor.transform(frame, DataContract("1h"))
    with pytest.raises(ProcessingError, match="whole minutes"):
        ResampleIntraday((replace(schedule[0], close=schedule[0].close + timedelta(seconds=1)),))
    registry = discover_processors()
    assert not registry.issues
    assert registry.definition("resample_intraday", "1").schema["properties"]["sessions"]["format"] == "trading_sessions"
    pipeline = Pipeline([processor])
    restored = load_pipeline(pipeline.canonical_json)
    assert restored.fingerprint == pipeline.fingerprint
    assert_frame_equal(restored.run(frame, DataContract("1m")).frame, pipeline.run(frame, DataContract("1m")).frame)
    assert Pipeline([replace(processor, incomplete="drop")]).fingerprint != pipeline.fingerprint


@pytest.mark.parametrize("column", ["high", "volume"])
def test_semantic_validator_rejects_incorrect_aggregation(column):
    class Broken(ResampleIntraday):
        def _transform(self, frame, input_contract, output_contract):
            return super()._transform(frame, input_contract, output_contract).with_columns(
                (pl.col(column) + 1).alias(column)
            )
    schedule = (session(),)
    with pytest.raises(ProcessingError, match=f"{column} aggregation is incorrect"):
        Broken(schedule).transform(minute_frame(schedule), DataContract("1m"))


def test_saved_30m_source_is_accepted_by_research_with_raw_lineage(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    sessions = tuple(TradingSession.from_config(item) for item in generate_schedule(
        "XNYS", date(2024, 1, 2), date(2024, 1, 3)
    ))
    source = minute_frame(sessions)
    raw = research.pipeline.ingest_frame(DataRequest("AAPL", datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC), timeframe="1m"), source).raw
    output = research.pipeline.process(raw.dataset_id, processors=[ResampleIntraday(sessions)])
    report = research.preflight(["AAPL"], strategies=[MovingAverageCross(1, 2)], timeframe="30m",
        start=datetime(2024, 1, 3, tzinfo=UTC), end=datetime(2024, 1, 4, tzinfo=UTC),
        layer="processed", pipeline_id=output.pipeline_id)
    assert report.ok, report.issues
    assert output.parent_ids == (raw.dataset_id,)
    assert_frame_equal(research.pipeline.read_dataset(raw.dataset_id), source)


def test_dashboard_can_generate_regular_sessions_and_persist_drop_warnings(monkeypatch, tmp_path):
    import dashboard.app as dashboard
    research = Research(tmp_path / "data", tmp_path / "runs")
    schedule = (session(minutes=390),)
    source = minute_frame(schedule).filter(pl.col("timestamp") != schedule[0].open + timedelta(minutes=7))
    raw = research.pipeline.ingest_frame(DataRequest(symbol="AAPL", provider="yahoo", timeframe="1m",
        start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 3, tzinfo=UTC)), source).raw
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=20).run()
    app.sidebar.radio[0].set_value("Process data").run()
    app.selectbox(key="process-add").set_value(("resample_intraday", "1")).run()
    next(button for button in app.button if button.label == "Add processor").click().run()
    next(button for button in app.button if button.label == "Generate session schedule").click().run()
    assert json.loads(app.text_area[0].value) == [schedule[0].to_config()]
    next(select for select in app.selectbox if select.label == "Incomplete intervals").set_value("drop").run()
    next(button for button in app.button if button.label == "Preview processing").click().run()
    assert not app.exception and not app.error
    assert any("Dropped 1 incomplete" in item.value for item in app.warning)
    next(button for button in app.button if button.label == "Process and save").click().run()
    assert not app.exception and not app.error
    output, = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert output.request.timeframe == "30m" and output.row_count == 12
    assert research.pipeline.read_dataset(raw.dataset_id).height == 389
    reopened = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=20).run()
    reopened.sidebar.radio[0].set_value("Process data").run()
    assert any("AAPL: Dropped 1 incomplete" in item.value for item in reopened.warning)
