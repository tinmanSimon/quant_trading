"""Regular-hours daily aggregation, completeness, discovery and saved workflows."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import DataContract, Pipeline, TradingSession, discover_processors, load_pipeline
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.intraday import ResamplingWarning
from data_pipeline.schemas import OHLCV_SCHEMA
from research import Research
from research.instruments import Instrument, expected_bars
from research.strategies import MovingAverageCross


def midnight(day):
    return datetime.combine(day, datetime.min.time(), UTC)


def minute_frame(sessions, symbol="AAPL", offset=0):
    rows = []
    for index, session in enumerate(sessions):
        for minute in range(int((session.close - session.open) / timedelta(minutes=1))):
            price = float(offset + 100 * (index + 1) + minute)
            rows.append({"timestamp": session.open + timedelta(minutes=minute), "symbol": symbol,
                         "open": price, "high": price + 2, "low": price - 1,
                         "close": price + 1, "volume": 10.})
    return pl.DataFrame(rows, schema=OHLCV_SCHEMA)


def outside_frame(source, timestamps):
    return pl.concat([source.head(1).with_columns(
        pl.lit(stamp).cast(pl.Datetime("ms", "UTC")).alias("timestamp"),
        pl.lit(20000.).alias("open"), pl.lit(20001.).alias("high"),
        pl.lit(0.001).alias("low"), pl.lit(20000.).alias("close"),
        pl.lit(50000.).alias("volume"),
    ) for stamp in timestamps])


@pytest.fixture
def processor():
    return ResampleRegularHoursDaily.for_dates(date(2024, 1, 2), date(2024, 1, 4))


def test_daily_ohlcv_excludes_extended_hours():
    processor = ResampleRegularHoursDaily.for_dates(date(2024, 1, 2), date(2024, 1, 3))
    source = minute_frame(processor.sessions).with_columns(
        pl.when(pl.col("timestamp") == processor.sessions[0].open + timedelta(minutes=20))
        .then(9000.).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("timestamp") == processor.sessions[0].open + timedelta(minutes=30))
        .then(0.5).otherwise(pl.col("low")).alias("low"),
    )
    outside = outside_frame(source, [stamp for session in processor.sessions
                                    for stamp in (session.open - timedelta(minutes=1), session.close)])
    source = pl.concat([source, outside]).sort("symbol", "timestamp")
    original = source.clone()
    with pytest.warns(ResamplingWarning, match="Excluded 4"):
        result = processor.transform(source, DataContract("1m"))
    expected = pl.DataFrame([
        {"timestamp": midnight(date(2024, 1, 2)), "symbol": "AAPL", "open": 100., "high": 9000.,
         "low": 0.5, "close": 490., "volume": 3900.},
        {"timestamp": midnight(date(2024, 1, 3)), "symbol": "AAPL", "open": 200., "high": 591.,
         "low": 199., "close": 590., "volume": 3900.},
    ], schema=OHLCV_SCHEMA)
    assert result.contract == DataContract("1d")
    assert_frame_equal(result.frame, expected)
    assert_frame_equal(source, original)


@pytest.mark.parametrize("day,utc_open_hour,minutes", [
    (date(2025, 3, 7), 14, 390), (date(2025, 3, 10), 13, 390),
    (date(2025, 10, 31), 13, 390), (date(2025, 11, 3), 14, 390),
    (date(2025, 11, 28), 14, 210),
])
def test_dst_early_close_and_daily_availability(day, utc_open_hour, minutes):
    processor = ResampleRegularHoursDaily.for_dates(day, day)
    session, = processor.sessions
    assert (session.open.hour, session.open.minute) == (utc_open_hour, 30)
    assert session.close - session.open == timedelta(minutes=minutes)
    source = minute_frame(processor.sessions)
    # The minute at an early close must also be excluded.
    source = pl.concat([source, outside_frame(source, [session.close])])
    with pytest.warns(ResamplingWarning, match="Excluded 1"):
        result = processor.transform(source, DataContract("1m"))
    assert result.frame.row(0, named=True) == {
        "timestamp": midnight(day), "symbol": "AAPL", "open": 100., "high": 101. + minutes,
        "low": 99., "close": 100. + minutes, "volume": minutes * 10.,
    }
    bars = expected_bars(Instrument("AAPL"), start=midnight(day),
                         end=midnight(day + timedelta(days=1)), timeframe="1d")
    processor.validate_backtest_calendar(bars, input_contract=DataContract("1m"),
                                         output_contract=result.contract)
    for changes in ({"open": session.open - timedelta(minutes=1)},
                    {"close": session.close - timedelta(minutes=1)}, {"timestamp": session.open}):
        with pytest.raises(ProcessingError, match="execution calendar"):
            processor.validate_backtest_calendar([replace(bars[0], **changes)],
                input_contract=DataContract("1m"), output_contract=result.contract)
    with pytest.raises(ProcessingError, match="output contract"):
        processor.validate_backtest_calendar(bars, input_contract=DataContract("1m"),
                                             output_contract=DataContract("1h"))


def test_holidays_and_weekends_are_excluded():
    processor = ResampleRegularHoursDaily.for_dates(date(2024, 7, 3), date(2024, 7, 8))
    assert [session.label for session in processor.sessions] == [
        date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8),
    ]
    source = minute_frame(processor.sessions)
    source = pl.concat([source, outside_frame(source, [
        datetime(2024, 7, 4, 14, tzinfo=UTC), datetime(2024, 7, 6, 14, tzinfo=UTC),
    ])]).sort("symbol", "timestamp")
    with pytest.warns(ResamplingWarning, match="Excluded 2"):
        result = processor.transform(source, DataContract("1m"))
    assert result.frame["volume"].to_list() == [2100., 3900., 3900.]
    assert result.frame["timestamp"].dt.date().to_list() == [s.label for s in processor.sessions]


@pytest.mark.parametrize("day", [date(2024, 7, 4), date(2024, 7, 6)])
def test_nontrading_sessions_are_rejected(day):
    with pytest.raises(ProcessingError, match="No XNYS regular sessions"):
        ResampleRegularHoursDaily.for_dates(day, day)
    opening = midnight(day) + timedelta(hours=13, minutes=30)
    with pytest.raises(ProcessingError, match="must match XNYS regular hours"):
        ResampleRegularHoursDaily((TradingSession(day, opening, opening + timedelta(hours=6, minutes=30)),))


@pytest.mark.parametrize("boundary,minutes", [("open", -30), ("open", 30), ("close", -30), ("close", 180)])
def test_edited_json_cannot_override_regular_or_early_close_hours(boundary, minutes):
    processor = ResampleRegularHoursDaily.for_dates(date(2024, 11, 29), date(2024, 11, 29))
    config = processor.config
    session, = processor.sessions
    config["sessions"][0][boundary] = (getattr(session, boundary) + timedelta(minutes=minutes)).isoformat()
    with pytest.raises(ProcessingError, match="must match XNYS regular hours"):
        discover_processors().create(processor.name, processor.version, config)


def test_missing_minutes_and_wholly_missing_days_follow_session_policy(processor):
    first, missing, last = processor.sessions
    source = minute_frame(processor.sessions).filter(
        (pl.col("timestamp") != first.open)
        & (pl.col("timestamp") != first.close - timedelta(minutes=1))
        & ~pl.col("timestamp").is_between(missing.open, missing.close, closed="left")
    )
    report = processor.quality_report(source)
    assert report["timestamp"].to_list() == [midnight(first.label), midnight(missing.label)]
    assert report["observed_minutes"].to_list() == [388, 0]
    assert report["missing_minutes"].to_list() == [2, 390]
    with pytest.raises(ProcessingError, match="2 incomplete daily sessions"):
        processor.transform(source, DataContract("1m"))
    with pytest.warns(ResamplingWarning, match="Dropped 2 incomplete daily sessions"):
        dropped = replace(processor, incomplete="drop").transform(source, DataContract("1m")).frame
    assert dropped["timestamp"].to_list() == [midnight(last.label)]
    with pytest.warns(ResamplingWarning, match="1 incomplete daily sessions; omitted 1 empty sessions"):
        available = replace(processor, incomplete="aggregate_available").transform(source, DataContract("1m")).frame
    assert available["timestamp"].to_list() == [midnight(first.label), midnight(last.label)]
    assert available.row(0, named=True) == {
        "timestamp": midnight(first.label), "symbol": "AAPL", "open": 101., "high": 490.,
        "low": 100., "close": 489., "volume": 3880.,
    }
    assert_frame_equal(available.tail(1), dropped)


@pytest.mark.parametrize("policy", ["raise", "drop", "aggregate_available"])
def test_extended_hours_only_never_fabricates_a_daily_bar(processor, policy):
    source = outside_frame(minute_frame(processor.sessions), [processor.sessions[0].open - timedelta(minutes=1)])
    message = "incomplete daily sessions" if policy == "raise" else "No daily bars"
    with pytest.warns(ResamplingWarning) as captured:
        with pytest.raises(ProcessingError, match=message):
            replace(processor, incomplete=policy).transform(source, DataContract("1m"))
    assert any("Excluded 1" in str(item.message) for item in captured)


def test_dropping_all_partial_days_fails(processor):
    with pytest.warns(ResamplingWarning, match="Dropped 3"):
        with pytest.raises(ProcessingError, match="No daily bars"):
            replace(processor, incomplete="drop").transform(minute_frame(processor.sessions).head(1), DataContract("1m"))


def test_completeness_and_aggregation_do_not_mix_symbols(processor):
    source = pl.concat([
        minute_frame(processor.sessions).slice(1),
        minute_frame(processor.sessions, "MSFT", offset=1000),
    ]).sort("symbol", "timestamp")
    assert processor.quality_report(source).select("symbol", "missing_minutes").rows() == [("AAPL", 1)]
    with pytest.warns(ResamplingWarning, match="Dropped 1"):
        result = replace(processor, incomplete="drop").transform(source, DataContract("1m")).frame
    assert result["symbol"].to_list() == ["AAPL", "AAPL", "MSFT", "MSFT", "MSFT"]
    assert result["open"].to_list() == [200., 300., 1100., 1200., 1300.]
    assert result["volume"].to_list() == [3900.] * 5


def test_input_contract_minute_grid_and_duplicate_validation(processor):
    source = minute_frame(processor.sessions)
    with pytest.raises(ProcessingError, match="1m input contract"):
        processor.transform(source, DataContract("1h"))
    shifted = source.with_columns(pl.col("timestamp") + timedelta(seconds=1))
    with pytest.raises(ProcessingError, match="whole-minute grid"):
        processor.transform(shifted, DataContract("1m"))
    # A duplicate cannot substitute for a missing minute in the count check.
    duplicated = pl.concat([source.slice(1), source.slice(1, 1)])
    with pytest.raises(ProcessingError, match="[Dd]uplicate"):
        Pipeline([processor]).run(duplicated, DataContract("1m"))
    with pytest.raises(ProcessingError):
        replace(processor, incomplete="fill")
    with pytest.raises(ProcessingError, match="unique"):
        replace(processor, sessions=(processor.sessions[0], processor.sessions[0]))
    with pytest.raises(ProcessingError, match="nonempty"):
        replace(processor, sessions=())


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume", "timestamp"])
def test_semantic_validation_rejects_corrupted_aggregation(processor, column):
    class Broken(ResampleRegularHoursDaily):
        def _transform(self, frame, input_contract, output_contract):
            result = super()._transform(frame, input_contract, output_contract)
            delta = timedelta(days=7) if column == "timestamp" else (0.25 if column != "low" else -0.25)
            return result.with_columns((pl.col(column) + delta).alias(column))

    message = "expected sessions" if column == "timestamp" else f"{column} aggregation"
    with pytest.raises(ProcessingError, match=message):
        Broken(processor.sessions).transform(minute_frame(processor.sessions), DataContract("1m"))


def test_auto_discovery_fixed_timeframe_and_recipe_roundtrip(processor):
    registry = discover_processors()
    assert not registry.issues
    definition = registry.definition(processor.name, processor.version)
    assert definition.causal and definition.price_basis == "preserve"
    assert set(definition.schema["properties"]) == {"sessions", "incomplete"}
    assert definition.schema["properties"]["sessions"]["regular_only"] is True
    assert definition.schema["properties"]["sessions"]["calendar"] == "XNYS"
    pipeline = Pipeline([processor])
    restored = load_pipeline(pipeline.canonical_json)
    assert restored.fingerprint == pipeline.fingerprint
    source = minute_frame(processor.sessions)
    assert_frame_equal(restored.run(source, DataContract("1m")).frame,
                       pipeline.run(source, DataContract("1m")).frame)
    assert Pipeline([replace(processor, incomplete="drop")]).fingerprint != pipeline.fingerprint
    with pytest.raises(ProcessingError, match="Additional properties"):
        registry.create(processor.name, processor.version, processor.config | {"target_timeframe": "1h"})


def test_split_raw_batches_save_daily_contract_and_work_in_research(tmp_path, processor, capsys):
    from data_pipeline.cli import main

    research = Research(tmp_path / "data", tmp_path / "runs")
    source = minute_frame(processor.sessions)
    request = DataRequest("AAPL", midnight(date(2024, 1, 2)), midnight(date(2024, 1, 5)), timeframe="1m")
    chunks = (source.head(170), source.slice(170))  # Split the first trading session.
    parents = [research.pipeline.ingest_frame(request, chunk).raw for chunk in chunks]
    recipe = tmp_path / "daily.json"
    recipe.write_text(Pipeline([processor]).canonical_json)
    assert main(["--data-dir", str(tmp_path / "data"), "process", *[p.dataset_id for p in reversed(parents)],
                 "--processors", str(recipe)]) == 0
    metadata = json.loads(capsys.readouterr().out)
    output = research.pipeline.get_metadata(metadata["dataset_id"])
    assert output.parent_ids == tuple(p.dataset_id for p in parents)
    assert output.request.timeframe == "1d" and output.timestamp_convention == "session_date"
    assert output.row_count == 3 and "timeframe=1d" in output.relative_path
    restarted = DataPipeline(tmp_path / "data")
    assert_frame_equal(restarted.read(DataQuery(layer="processed", timeframe="1d", pipeline_id=output.pipeline_id)),
                       processor.transform(source, DataContract("1m")).frame)
    for parent, chunk in zip(parents, chunks, strict=True):
        assert_frame_equal(restarted.read_dataset(parent.dataset_id), chunk)
    report = research.preflight(["AAPL"], strategies=[MovingAverageCross(1, 2)], timeframe="1d",
        start=midnight(date(2024, 1, 4)), end=midnight(date(2024, 1, 5)),
        layer="processed", pipeline_id=output.pipeline_id)
    assert report.ok, report.issues


def test_dashboard_discovers_previews_saves_and_restores_daily_processor(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as dashboard

    processor = ResampleRegularHoursDaily.for_dates(date(2024, 11, 29), date(2024, 11, 29))
    session, = processor.sessions
    source = minute_frame(processor.sessions)
    source = pl.concat([source, outside_frame(source, [session.open - timedelta(minutes=1), session.close])]).sort(
        "symbol", "timestamp"
    )
    research = Research(tmp_path / "data", tmp_path / "runs")
    raw = research.pipeline.ingest_frame(DataRequest("AAPL", midnight(session.label),
        midnight(session.label + timedelta(days=1)), timeframe="1m"), source).raw
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    script = "from dashboard.app import main\nmain()"
    app = AppTest.from_string(script, default_timeout=20).run()
    app.sidebar.radio[0].set_value("Process data").run()
    assert "Regular hours: 1m to 1d (v1)" in app.selectbox(key="process-add").options
    app.selectbox(key="process-add").set_value((processor.name, processor.version)).run()
    next(button for button in app.button if button.label == "Add processor").click().run()
    assert not any(field.label == "Session hours" for field in app.radio)
    assert not any(field.label == "Exchange calendar" for field in app.selectbox)
    assert not app.time_input
    next(button for button in app.button if button.label == "Generate session schedule").click().run()
    assert json.loads(app.text_area[0].value) == processor.config["sessions"]
    next(button for button in app.button if button.label == "Preview processing").click().run()
    assert not app.exception and not app.error
    assert any("Excluded 2" in item.value for item in app.warning)
    assert not research.pipeline.list_datasets(DataQuery(layer="processed"))
    next(button for button in app.button if button.label == "Process and save").click().run()
    assert not app.exception and not app.error
    output, = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert output.request.timeframe == "1d" and output.row_count == 1
    assert research.pipeline.read_dataset(output.dataset_id).row(0, named=True) == {
        "timestamp": midnight(session.label), "symbol": "AAPL", "open": 100., "high": 311.,
        "low": 99., "close": 310., "volume": 2100.,
    }
    assert_frame_equal(research.pipeline.read_dataset(raw.dataset_id), source.sort("symbol", "timestamp"))
    restored = AppTest.from_string(script, default_timeout=20).run()
    restored.sidebar.radio[0].set_value("Process data").run()
    assert not restored.exception and not restored.error
    assert json.loads(restored.text_area[0].value) == processor.config["sessions"]
    assert any("Excluded 2" in item.value for item in restored.warning)
