"""Time-window filtering preserves exact input bars and handles local clock boundaries."""
from datetime import UTC, date, datetime, time, timedelta
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import DataContract, Pipeline, ResampleOHLCV, TradingSession, discover_processors, load_pipeline
from data_pipeline.processing.processors.time_window import TimeWindowFilter
from data_pipeline.schemas import OHLCV_SCHEMA


def bars(timestamps, symbol="AAPL"):
    count = len(timestamps)
    return pl.DataFrame({
        "timestamp": [datetime.fromisoformat(value) if isinstance(value, str) else value for value in timestamps],
        "symbol": [symbol] * count,
        **{column: [base + i for i in range(count)] for column, base in
           (("open", 100.), ("high", 102.), ("low", 99.), ("close", 101.))},
        "volume": [1000. + i for i in range(count)],
    }, schema=OHLCV_SCHEMA)


def test_start_inclusive_end_exclusive_and_retained_rows_unchanged():
    frame = bars([f"2024-10-11T{clock}+00:00" for clock in
                  ("12:59:59.999", "13:00:00", "13:30:00", "19:59:59.999", "20:00:00", "21:00:00")])
    original = frame.clone()
    output = TimeWindowFilter().transform(frame, DataContract("1m"))
    assert output.contract == DataContract("1m")
    assert_frame_equal(output.frame, frame.slice(1, 3))
    assert_frame_equal(frame, original)


def test_local_window_uses_the_correct_dst_offset_for_each_date():
    frame = bars(["2024-03-08T13:59:00+00:00", "2024-03-08T14:00:00+00:00",
                  "2024-03-11T12:59:00+00:00", "2024-03-11T13:00:00+00:00"])
    output = TimeWindowFilter(end_time=time(10)).transform(frame, DataContract("1m"))
    assert_frame_equal(output.frame, frame[[1, 3]])


def test_repeated_local_hour_keeps_both_distinct_utc_instants():
    frame = bars([f"2024-11-03T{clock}:00+00:00" for clock in ("05:00", "05:30", "06:00", "06:30", "07:00")])
    output = TimeWindowFilter(time(1), time(2)).transform(frame, DataContract("30m"))
    assert_frame_equal(output.frame, frame.head(4))


def test_overnight_window_and_multiple_symbols():
    stamps = ["2024-01-02T21:00:00+00:00", "2024-01-02T22:00:00+00:00", "2024-01-02T23:00:00+00:00",
              "2024-01-03T00:00:00+00:00", "2024-01-03T01:00:00+00:00", "2024-01-03T02:00:00+00:00"]
    first, second = bars(stamps), bars(stamps, "MSFT")
    frame = pl.concat([first, second])
    output = TimeWindowFilter(time(22), time(2), "UTC").transform(frame, DataContract("1h"))
    assert_frame_equal(output.frame, pl.concat([first.slice(1, 4), second.slice(1, 4)]))


def test_equal_times_keep_all_bars():
    frame = bars(["2024-01-02T00:00:00+00:00", "2024-01-02T09:00:00+00:00", "2024-01-02T23:00:00+00:00"])
    assert_frame_equal(TimeWindowFilter(time(9), time(9), "UTC").transform(frame, DataContract("1h")).frame, frame)


def test_positive_offset_timezone_and_boundary_crossing_bars_are_not_rebuilt():
    frame = bars(["2024-01-02T00:00:00+00:00", "2024-01-02T01:00:00+00:00",
                  "2024-01-02T07:30:00+00:00", "2024-01-02T08:00:00+00:00"])
    # The 15:30 local hourly bar remains whole; it is not shortened at 16:00.
    output = TimeWindowFilter(time(9), time(16), "Asia/Shanghai").transform(frame, DataContract("1h"))
    assert_frame_equal(output.frame, frame.slice(1, 2))


@pytest.mark.parametrize("changes", [
    {"start_time": "09:00"}, {"end_time": 16}, {"start_time": time(9, tzinfo=UTC)},
    {"end_time": time(16, tzinfo=UTC)}, {"timezone": "New York"}, {"timezone": ""}, {"timezone": "localtime"},
])
def test_invalid_settings_fail_during_construction(changes):
    with pytest.raises(ProcessingError):
        TimeWindowFilter(**changes)


@pytest.mark.parametrize("timeframe", ["1d", "1wk", "1mo"])
def test_session_date_bars_are_rejected(timeframe):
    with pytest.raises(ProcessingError, match="requires intraday bars"):
        TimeWindowFilter().transform(bars(["2024-01-02T00:00:00+00:00"]), DataContract(timeframe))


def test_no_matching_bars_reports_the_window_problem():
    with pytest.raises(ProcessingError, match="No bars remain inside"):
        TimeWindowFilter().transform(bars(["2024-01-02T01:00:00+00:00"]), DataContract("1h"))


def test_automatic_configuration_and_fingerprint_round_trip():
    registry = discover_processors()
    definition = registry.definition("filter_time_window", "1")
    assert definition.causal and definition.price_basis == "preserve"
    assert definition.schema["properties"]["start_time"]["format"] == "time"
    assert "America/New_York" in definition.schema["properties"]["timezone"]["enum"]
    pipeline = registry.load([{"name": "filter_time_window", "version": "1", "config": {
        "start_time": "09:00", "end_time": "16:00", "timezone": "America/New_York",
    }}])
    assert pipeline.fingerprint == Pipeline([TimeWindowFilter()]).fingerprint
    assert load_pipeline(pipeline.canonical_json).fingerprint == pipeline.fingerprint
    with pytest.raises(ProcessingError, match="start_time"):
        registry.create("filter_time_window", "1", {"start_time": "25:00"})


def test_filter_then_resample_preserves_raw_data_and_saved_lineage(tmp_path):
    api = DataPipeline(tmp_path / "data")
    frame = bars([datetime(2024, 10, 11, 8, tzinfo=UTC) + timedelta(hours=i) for i in range(16)])
    raw = api.ingest_frame(DataRequest(symbol="AAPL", timeframe="1h", provider="yahoo",
        start=datetime(2024, 10, 11, tzinfo=UTC), end=datetime(2024, 10, 12, tzinfo=UTC)), frame).raw
    session = TradingSession(date(2024, 10, 11), datetime(2024, 10, 11, 13, tzinfo=UTC), datetime(2024, 10, 11, 20, tzinfo=UTC))
    recipe = Pipeline([TimeWindowFilter(), ResampleOHLCV((session,))])
    output = api.process(raw.dataset_id, processors=load_pipeline(recipe.canonical_json))
    saved = api.read_dataset(output.dataset_id)
    assert saved.row(0, named=True) == {
        "timestamp": datetime(2024, 10, 11, tzinfo=UTC), "symbol": "AAPL", "open": 105.,
        "high": 113., "low": 104., "close": 112., "volume": 7056.,
    }
    assert output.parent_ids == (raw.dataset_id,)
    assert json.loads(output.processors_json)[0]["name"] == "filter_time_window"
    assert_frame_equal(api.read_dataset(raw.dataset_id), frame)
    with pytest.raises(ProcessingError, match="No bars remain inside"):
        api.process(raw.dataset_id, processors=[TimeWindowFilter(time(1), time(2), "UTC")])
    assert len(api.list_datasets(DataQuery(layer="processed"))) == 1


def test_output_validation_rejects_changed_prices():
    class CorruptFilter(TimeWindowFilter):
        def _transform(self, frame, input_contract, output_contract):
            return super()._transform(frame, input_contract, output_contract).with_columns(pl.col("volume") + 1)

    with pytest.raises(ProcessingError, match="unchanged values"):
        CorruptFilter().transform(bars(["2024-10-11T13:00:00+00:00"]), DataContract("1h"))
