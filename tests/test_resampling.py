"""Independent expected aggregates and session-boundary resampling tests."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import DataAlreadyExistsError, ProcessingError, RequestDataMismatchError, OverlappingDataError
from data_pipeline.processing import DataContract, Pipeline, ResampleOHLCV, ScalePrices, TradingSession, load_pipeline
from data_pipeline.schemas import OHLCV_SCHEMA


@pytest.fixture
def sessions():
    # Synthetic sessions: a short final hourly bar on day one, early close on day two.
    return (
        TradingSession(date(2024, 1, 2), datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                       datetime(2024, 1, 2, 17, tzinfo=UTC)),
        TradingSession(date(2024, 1, 3), datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
                       datetime(2024, 1, 3, 16, 30, tzinfo=UTC)),
    )


@pytest.fixture
def hourly(sessions):
    timestamps = [sessions[0].open + timedelta(hours=i) for i in range(3)]
    timestamps += [sessions[1].open + timedelta(hours=i) for i in range(2)]
    return pl.DataFrame({"timestamp": timestamps, "symbol": ["AAPL"] * 5,
                         "open": [10, 12, 11, 20, 22], "high": [13, 15, 14, 24, 25],
                         "low": [9, 10, 8, 19, 20], "close": [12, 11, 13, 22, 21],
                         "volume": [100, 200, 300, 400, 500]}, schema=OHLCV_SCHEMA)


@pytest.fixture
def daily():
    # Hand-calculated, not generated with the processor or its validation helper.
    return pl.DataFrame({"timestamp": [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)],
                         "symbol": ["AAPL", "AAPL"], "open": [10, 20], "high": [15, 25],
                         "low": [8, 19], "close": [13, 21], "volume": [600, 900]}, schema=OHLCV_SCHEMA)


def test_hourly_to_daily_exact_expected_values(hourly, daily, sessions):
    recipe = Pipeline([ResampleOHLCV(sessions)])
    result = recipe.run(hourly, DataContract("1h"))
    assert result.contract == DataContract("1d")
    assert_frame_equal(result.frame, daily)
    direct = ResampleOHLCV(sessions).transform(hourly, DataContract("1h"))
    assert direct.contract == result.contract
    assert_frame_equal(direct.frame, daily)
    assert_frame_equal(recipe.transform(hourly, contract=DataContract("1h")), daily)
    assert_frame_equal(Pipeline([ScalePrices(2), ResampleOHLCV(sessions), ScalePrices(0.5)])
                       .run(hourly, DataContract("1h")).frame, daily)


def test_resampling_multiple_symbols(hourly, daily, sessions):
    combined = pl.concat([hourly, hourly.with_columns(pl.lit("MSFT").alias("symbol"))])
    expected = pl.concat([daily, daily.with_columns(pl.lit("MSFT").alias("symbol"))])
    assert_frame_equal(Pipeline([ResampleOHLCV(sessions)]).run(combined, DataContract("1h")).frame, expected)


@pytest.mark.parametrize("missing", ["first", "middle", "last", "whole-session"])
def test_incomplete_groups_raise_or_drop(hourly, daily, sessions, missing):
    if missing == "whole-session":
        incomplete = hourly.tail(2)
    else:
        index = {"first": 0, "middle": 1, "last": 2}[missing]
        incomplete = hourly.filter(pl.col("timestamp") != hourly["timestamp"][index])
    with pytest.raises(ProcessingError, match="Incomplete"):
        Pipeline([ResampleOHLCV(sessions)]).run(incomplete, DataContract("1h"))
    result = Pipeline([ResampleOHLCV(sessions, incomplete="drop")]).run(incomplete, DataContract("1h"))
    assert_frame_equal(result.frame, daily.tail(1))


def test_drop_all_groups_is_still_an_error(hourly, sessions):
    with pytest.raises(ProcessingError):
        Pipeline([ResampleOHLCV(sessions, incomplete="drop")]).run(hourly.head(1), DataContract("1h"))


@pytest.mark.parametrize("offset,message", [(timedelta(minutes=1), "off.*grid"),
                                            (timedelta(hours=-1), "outside")])
def test_out_of_session_and_off_grid_not_silently_dropped(hourly, sessions, offset, message):
    frame = hourly.with_columns(pl.col("timestamp") + offset)
    with pytest.raises(ProcessingError, match=message):
        Pipeline([ResampleOHLCV(sessions, incomplete="drop")]).run(frame.head(1), DataContract("1h"))


def test_close_boundary_is_exclusive(hourly, sessions):
    boundary = hourly.head(1).with_columns(pl.lit(sessions[0].close).cast(OHLCV_SCHEMA["timestamp"]).alias("timestamp"))
    with pytest.raises(ProcessingError, match="outside"):
        Pipeline([ResampleOHLCV(sessions)]).run(boundary, DataContract("1h"))


def test_explicit_schedule_handles_offset_change_and_overnight_date_labels(hourly):
    # Two caller-supplied offset regimes. No timezone/calendar inference occurs.
    east = timezone(timedelta(hours=9))
    changed = timezone(timedelta(hours=8))
    schedule = [TradingSession(date(2024, 1, 2), datetime(2024, 1, 2, 1, tzinfo=east),
                               datetime(2024, 1, 2, 2, tzinfo=east)),
                TradingSession(date(2024, 1, 3), datetime(2024, 1, 3, 1, tzinfo=changed),
                               datetime(2024, 1, 3, 2, tzinfo=changed))]
    frame = hourly.head(2).with_columns(pl.Series("timestamp", [s.open for s in schedule], dtype=OHLCV_SCHEMA["timestamp"]))
    result = Pipeline([ResampleOHLCV(schedule)]).run(frame, DataContract("1h"))
    assert result.frame["timestamp"].to_list() == [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)]


@pytest.mark.parametrize("contract", [None, DataContract("1d"), DataContract("15m")])
def test_resampler_requires_explicit_hourly_contract(hourly, sessions, contract):
    with pytest.raises(ProcessingError, match="contract"):
        ResampleOHLCV(sessions).transform(hourly, contract)


@pytest.mark.parametrize("column", ["close", "volume"])
def test_output_validator_catches_valid_schema_but_wrong_aggregation(hourly, sessions, column):
    class Broken(ResampleOHLCV):
        def _transform(self, frame, input_contract, output_contract):
            return super()._transform(frame, input_contract, output_contract).with_columns((pl.col(column) + 0.25).alias(column))
    with pytest.raises(ProcessingError, match="aggregation is incorrect"):
        Pipeline([Broken(sessions)]).run(hourly, DataContract("1h"))


def test_json_loading_and_schedule_identity(hourly, daily, sessions):
    processor = ResampleOHLCV(sessions)
    direct = Pipeline([processor])
    loaded = load_pipeline(direct.canonical_json)
    assert loaded.fingerprint == direct.fingerprint
    assert_frame_equal(loaded.run(hourly, DataContract("1h")).frame, daily)
    config = processor.config
    config["sessions"][0]["close"] = "bad"
    assert processor.config != config
    assert Pipeline([ResampleOHLCV(sessions, incomplete="drop")]).fingerprint != direct.fingerprint
    assert Pipeline([ResampleOHLCV([replace(sessions[0], close=sessions[0].close + timedelta(minutes=5)), sessions[1]])]).fingerprint != direct.fingerprint


@pytest.mark.parametrize("changes", [{"incomplete": "guess"}, {"target_timeframe": "1wk"},
                                    {"sessions": []}, {"sessions": ["bad"]}])
def test_invalid_resampler_config(sessions, changes):
    with pytest.raises(ProcessingError):
        ResampleOHLCV(**({"sessions": sessions} | changes))


def test_invalid_sessions(sessions):
    for kwargs in (
        {"open": sessions[0].open.replace(tzinfo=None)},
        {"close": sessions[0].open}, {"label": sessions[0].open},
        {"open": sessions[0].open + timedelta(microseconds=1)},
    ):
        with pytest.raises(ProcessingError):
            replace(sessions[0], **kwargs)
    with pytest.raises(ProcessingError, match="unique"):
        ResampleOHLCV([sessions[0], sessions[0]])
    with pytest.raises(ProcessingError, match="overlap"):
        ResampleOHLCV([replace(sessions[0], close=sessions[1].open + timedelta(minutes=1)), sessions[1]])


def test_multi_batch_process_store_restart_query_and_invalidate(isolated_data_dir, hourly, daily, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    # One session deliberately crosses the two file boundaries.
    first = pipeline.ingest_frame(request, hourly.head(2)).raw
    second = pipeline.ingest_frame(request, hourly.tail(3)).raw
    recipe = Pipeline([ResampleOHLCV(sessions)])
    processed = pipeline.process([second.dataset_id, first.dataset_id], processors=recipe)
    assert processed.parent_ids == (first.dataset_id, second.dataset_id)
    assert processed.request.timeframe == "1d"
    assert processed.timestamp_convention == "session_date"
    assert "timeframe=1d" in processed.relative_path
    assert processed.request.start == daily["timestamp"].min()
    restarted = DataPipeline(isolated_data_dir)
    assert_frame_equal(restarted.read_dataset(processed.dataset_id), daily)
    query = DataQuery(layer="processed", provider="yahoo", timeframe="1d", pipeline_id=recipe.fingerprint)
    assert_frame_equal(restarted.read(query), daily)
    assert len(restarted.list_datasets(DataQuery(timeframe="1h"))) == 2
    with pytest.raises(DataAlreadyExistsError):
        pipeline.process([first.dataset_id, second.dataset_id], processors=recipe)
    with pytest.raises(RequestDataMismatchError):
        pipeline.process([first.dataset_id, first.dataset_id], processors=recipe)
    replacement = pipeline.store.replace_raw(second.dataset_id, hourly.tail(3), confirm=second.dataset_id)
    assert not pipeline.get_metadata(processed.dataset_id).active
    assert_frame_equal(pipeline.read_dataset(processed.dataset_id), daily)
    assert pipeline.read(query).is_empty()
    with pytest.raises(RequestDataMismatchError):
        pipeline.process([first.dataset_id, second.dataset_id], processors=recipe)
    new = pipeline.process([replacement.dataset_id, first.dataset_id], processors=recipe)
    assert_frame_equal(pipeline.read_dataset(new.dataset_id), daily)
    assert len(pipeline.store.audit()) == 5


def test_failed_resampling_keeps_raw_and_no_processed_file(isolated_data_dir, hourly, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    with pytest.raises(ProcessingError, match="Raw data was saved"):
        pipeline.ingest_frame(DataRequest("AAPL", sessions[0].open, sessions[-1].close), hourly.head(1),
                              processors=[ResampleOHLCV(sessions)])
    assert len(pipeline.list_datasets()) == 1
    assert not list((isolated_data_dir / "processed").rglob("*.parquet"))


def test_cli_loads_resampler_and_multiple_raw_ids(isolated_data_dir, tmp_path, capsys, hourly, daily, sessions):
    from data_pipeline.cli import main
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    ids = [pipeline.ingest_frame(request, frame).raw.dataset_id for frame in (hourly.head(2), hourly.tail(3))]
    path = tmp_path / "processors.json"
    path.write_text(Pipeline([ResampleOHLCV(sessions)]).canonical_json)
    assert main(["--data-dir", str(isolated_data_dir), "process", *ids, "--processors", str(path)]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["parent_ids"] == ids
    assert metadata["request"]["timeframe"] == "1d"
    assert_frame_equal(pipeline.read_dataset(metadata["dataset_id"]), daily)


def test_processed_overlap_checked_in_output_timeframe(isolated_data_dir, hourly, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    first = pipeline.ingest_frame(request, hourly.head(3)).raw
    second = pipeline.ingest_frame(request, hourly.tail(2)).raw
    recipe = Pipeline([ResampleOHLCV(sessions, incomplete="drop")])
    original = pipeline.process(first.dataset_id, processors=recipe)
    # Different parent set, same recipe and overlapping DAILY output coverage.
    with pytest.raises(OverlappingDataError):
        pipeline.process([first.dataset_id, second.dataset_id], processors=recipe)
    assert pipeline.list_datasets(DataQuery(layer="processed")) == [original]


@pytest.mark.parametrize("change", [{"symbol": "MSFT"}, {"provider": "other"}, {"timeframe": "1m"}])
def test_multi_batch_rejects_incompatible_inputs(isolated_data_dir, hourly, sessions, change):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    first = pipeline.ingest_frame(request, hourly.head(2)).raw
    other_request = replace(request, **change)
    other_frame = hourly.tail(3).with_columns(pl.lit(other_request.symbol).alias("symbol"))
    second = pipeline.ingest_frame(other_request, other_frame).raw
    with pytest.raises(RequestDataMismatchError, match="must share"):
        pipeline.process([first.dataset_id, second.dataset_id], processors=[ResampleOHLCV(sessions)])
    assert not pipeline.list_datasets(DataQuery(layer="processed"))


def test_parent_replaced_during_processing_blocks_publication(isolated_data_dir, hourly, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    raw = pipeline.ingest_frame(request, hourly).raw

    class ConcurrentReplacement(ScalePrices):
        def _transform(self, frame, input_contract, output_contract):
            pipeline.store.replace_raw(raw.dataset_id, hourly, confirm=raw.dataset_id)
            return super()._transform(frame, input_contract, output_contract)

    with pytest.raises(RequestDataMismatchError, match="active raw"):
        pipeline.process(raw.dataset_id, processors=[ConcurrentReplacement(2)])
    assert not pipeline.list_datasets(DataQuery(layer="processed"))
    assert not list((isolated_data_dir / "processed").rglob("*.parquet"))


def test_empty_multi_batch_pipeline_is_explicitly_rejected(isolated_data_dir, hourly, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    ids = [pipeline.ingest_frame(request, frame).raw.dataset_id for frame in (hourly.head(2), hourly.tail(3))]
    with pytest.raises(RequestDataMismatchError, match="empty processor"):
        pipeline.process(ids)
    assert len(pipeline.list_datasets()) == 2


def test_storage_checks_declared_daily_contract_independently(isolated_data_dir, hourly, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    raw = pipeline.ingest_frame(request, hourly).raw
    recipe = Pipeline([ResampleOHLCV(sessions)])
    with pytest.raises(ProcessingError, match="midnight UTC"):
        pipeline.store.write_processed(raw.dataset_id, hourly, pipeline_id=recipe.fingerprint,
                                       processors_json=recipe.canonical_json, output_contract=DataContract("1d"))
    assert not pipeline.list_datasets(DataQuery(layer="processed"))


def test_compaction_invalidates_multi_parent_output(isolated_data_dir, hourly, daily, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    ids = [pipeline.ingest_frame(request, frame).raw.dataset_id for frame in (hourly.head(2), hourly.tail(3))]
    recipe = Pipeline([ResampleOHLCV(sessions)])
    processed = pipeline.process(ids, processors=recipe)
    compacted = pipeline.store.compact_raw(ids, confirm=ids)
    assert not pipeline.get_metadata(processed.dataset_id).active
    assert_frame_equal(pipeline.read_dataset(processed.dataset_id), daily)
    regenerated = pipeline.process(compacted.dataset_id, processors=recipe)
    assert regenerated.parent_ids == (compacted.dataset_id,)
    assert_frame_equal(pipeline.read_dataset(regenerated.dataset_id), daily)


def test_daily_raw_and_resampled_daily_data_coexist(isolated_data_dir, hourly, daily, sessions):
    pipeline = DataPipeline(isolated_data_dir)
    intraday_request = DataRequest("AAPL", sessions[0].open, sessions[-1].close)
    raw_hourly = pipeline.ingest_frame(intraday_request, hourly).raw
    pipeline.ingest_frame(replace(intraday_request, timeframe="1d", start=daily["timestamp"][0]), daily)
    pipeline.process(raw_hourly.dataset_id, processors=[ResampleOHLCV(sessions)])
    assert_frame_equal(pipeline.read(DataQuery(layer="raw", timeframe="1d")), daily)
    assert_frame_equal(pipeline.read(DataQuery(layer="processed", timeframe="1d")), daily)
