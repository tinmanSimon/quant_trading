"""Regular-hours-only resampling: exchange boundaries, gaps and saved recipes."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataRequest
from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import (
    DataContract, Pipeline, ResampleRegularHours30m, TradingSession,
    discover_processors, load_pipeline,
)
from data_pipeline.processing.processors.intraday import ResamplingWarning
from data_pipeline.schemas import OHLCV_SCHEMA
from research import Research
from research.instruments import Instrument, expected_bars
from research.strategies import MovingAverageCross


def minute_frame(sessions):
    timestamps = [session.open + timedelta(minutes=i) for session in sessions
                  for i in range(int((session.close - session.open) / timedelta(minutes=1)))]
    return pl.DataFrame({
        "timestamp": timestamps, "symbol": ["AAPL"] * len(timestamps),
        "open": [100.] * len(timestamps), "high": [102.] * len(timestamps),
        "low": [99.] * len(timestamps), "close": [101.] * len(timestamps),
        "volume": [10.] * len(timestamps),
    }, schema=OHLCV_SCHEMA)


@pytest.mark.parametrize("day,utc_open_hour,count", [
    (date(2025, 3, 7), 14, 13),
    (date(2025, 3, 10), 13, 13),
    (date(2025, 10, 31), 13, 13),
    (date(2025, 11, 3), 14, 13),
    (date(2025, 11, 28), 14, 7),
])
def test_dst_and_early_close_match_the_execution_calendar(day, utc_open_hour, count):
    processor = ResampleRegularHours30m.for_dates(day, day)
    session, = processor.sessions
    assert (session.open.hour, session.open.minute) == (utc_open_hour, 30)
    result = processor.transform(minute_frame(processor.sessions), DataContract("1m"))
    bars = expected_bars(Instrument("AAPL"), start=session.open, end=session.close, timeframe="30m")
    assert result.contract == DataContract("30m")
    assert result.frame.height == count
    assert result.frame["timestamp"].to_list() == [bar.timestamp for bar in bars]
    assert result.frame["timestamp"][-1] + timedelta(minutes=30) == session.close
    processor.validate_backtest_calendar(bars, input_contract=DataContract("1m"),
                                         output_contract=result.contract)


def test_schedule_omits_holidays_and_weekends():
    processor = ResampleRegularHours30m.for_dates(date(2024, 7, 3), date(2024, 7, 8))
    assert [session.label for session in processor.sessions] == [
        date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8),
    ]
    assert processor.sessions[0].close.hour == 17  # July 3 closes at 13:00 EDT.
    holiday = minute_frame(processor.sessions).head(1).with_columns(
        pl.lit(datetime(2024, 7, 4, 14, tzinfo=UTC)).cast(pl.Datetime("ms", "UTC")).alias("timestamp")
    )
    with pytest.warns(ResamplingWarning, match="Excluded 1"):
        source = pl.concat([minute_frame(processor.sessions), holiday]).sort("symbol", "timestamp")
        result = processor.transform(source, DataContract("1m"))
    assert result.frame.height == 7 + 13 + 13
    assert date(2024, 7, 4) not in result.frame["timestamp"].dt.date().to_list()


@pytest.mark.parametrize("day", [date(2024, 7, 4), date(2024, 7, 6)])
def test_nontrading_dates_cannot_be_injected(day):
    with pytest.raises(ProcessingError, match="No XNYS regular sessions"):
        ResampleRegularHours30m.for_dates(day, day)
    opening = datetime.combine(day, datetime.min.time(), UTC).replace(hour=13, minute=30)
    with pytest.raises(ProcessingError, match="must match XNYS regular hours"):
        ResampleRegularHours30m((TradingSession(day, opening, opening + timedelta(hours=6, minutes=30)),))


@pytest.mark.parametrize("boundary,delta", [("open", -30), ("close", 60)])
def test_extended_hours_cannot_be_injected(boundary, delta):
    session, = ResampleRegularHours30m.for_dates(date(2024, 1, 2), date(2024, 1, 2)).sessions
    changed = replace(session, **{boundary: getattr(session, boundary) + timedelta(minutes=delta)})
    with pytest.raises(ProcessingError, match="must match XNYS regular hours"):
        ResampleRegularHours30m((changed,))


def test_early_close_cannot_be_overridden_to_a_normal_close():
    session, = ResampleRegularHours30m.for_dates(date(2024, 11, 29), date(2024, 11, 29)).sessions
    with pytest.raises(ProcessingError, match="including early closes"):
        ResampleRegularHours30m((replace(session, close=session.close + timedelta(hours=3)),))


def test_missing_regular_minute_raises_or_drops_the_whole_interval():
    processor = ResampleRegularHours30m.for_dates(date(2024, 1, 2), date(2024, 1, 2))
    source = minute_frame(processor.sessions)
    source = source.filter(pl.col("timestamp") != processor.sessions[0].open + timedelta(minutes=7))
    with pytest.raises(ProcessingError, match="1 incomplete intraday intervals"):
        processor.transform(source, DataContract("1m"))
    dropping = replace(processor, incomplete="drop")
    assert dropping.quality_report(source)["missing_minutes"].to_list() == [1]
    with pytest.warns(ResamplingWarning, match="Dropped 1 incomplete 30m intervals"):
        output = dropping.transform(source, DataContract("1m")).frame
    assert output.height == 12
    assert output["timestamp"][0] == processor.sessions[0].open + timedelta(minutes=30)
    assert output["volume"].to_list() == [300.] * 12


def test_fixed_contract_and_pinned_recipe_roundtrip():
    processor = ResampleRegularHours30m.for_dates(date(2024, 1, 2), date(2024, 1, 3))
    registry = discover_processors()
    assert not registry.issues
    definition = registry.definition(processor.name, processor.version)
    assert set(definition.schema["properties"]) == {"sessions", "incomplete"}
    assert definition.schema["properties"]["sessions"]["regular_only"] is True
    original = Pipeline([processor])
    restored = load_pipeline(original.canonical_json)
    assert original.fingerprint == restored.fingerprint
    source = minute_frame(processor.sessions)
    assert_frame_equal(original.run(source, DataContract("1m")).frame,
                       restored.run(source, DataContract("1m")).frame)
    with pytest.raises(ProcessingError, match="1m input contract"):
        processor.transform(source, DataContract("1h"))


def test_saved_output_works_in_research_and_preserves_raw_lineage(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    processor = ResampleRegularHours30m.for_dates(date(2024, 1, 2), date(2024, 1, 3))
    source = minute_frame(processor.sessions)
    raw = research.pipeline.ingest_frame(DataRequest("AAPL", datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC), timeframe="1m"), source).raw
    output = research.pipeline.process(raw.dataset_id, processors=[processor])
    report = research.preflight(["AAPL"], strategies=[MovingAverageCross(1, 2)], timeframe="30m",
        start=datetime(2024, 1, 3, tzinfo=UTC), end=datetime(2024, 1, 4, tzinfo=UTC),
        layer="processed", pipeline_id=output.pipeline_id)
    assert report.ok, report.issues
    assert output.parent_ids == (raw.dataset_id,)
    assert_frame_equal(research.pipeline.read_dataset(raw.dataset_id), source)
