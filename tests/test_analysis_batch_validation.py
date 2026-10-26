"""Optional analysis validation checks frozen data, including initialization."""

from datetime import UTC, datetime, timedelta
import json
import warnings

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest, FetchQuality, OmittedBar
from data_pipeline.processing import ScalePrices, discover_processors
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from research.analysis import LogicContext, LogicRegistry, LogicSpec
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore
from research.analysis.batch import validation
from research.analysis.builtins.flat import FlatLogic
from research.errors import ResearchError
from research.instruments import Instrument, expected_bars


START = datetime(2024, 1, 2, tzinfo=UTC)
SECOND = datetime(2024, 1, 3, tzinfo=UTC)
END = datetime(2024, 1, 4, tzinfo=UTC)
AS_OF = datetime(2024, 1, 5, tzinfo=UTC)


def frame(timeframe="1d", *, calendar="XNYS"):
    bars = expected_bars(Instrument("AAPL", calendar), start=START, end=END, timeframe=timeframe)
    size = len(bars)
    return pl.DataFrame({
        "timestamp": [bar.timestamp for bar in bars], "symbol": ["AAPL"] * size,
        "open": [100.] * size, "high": [102.] * size, "low": [99.] * size,
        "close": [101.] * size, "volume": [10.] * size,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


def save(pipeline, data, *, timeframe="1d", provider="yahoo", quality=None):
    return pipeline.store.write_raw(DataRequest("AAPL", START, END, timeframe=timeframe, provider=provider),
                                    data, quality=quality)


def context(data, request):
    support = None
    selected = data
    if request.start is not None:
        support = data.filter(pl.col("timestamp") < request.start)
        selected = data.filter(pl.col("timestamp") >= request.start)
    if request.end is not None:
        selected = selected.filter(pl.col("timestamp") < request.end)
    return LogicContext(selected, request.timeframe, "fixture", request.as_of,
                        session_timezone=request.session_timezone, support_frame=support)


def validate(pipeline, saved, data, **kwargs):
    values = {"start": START, "end": END, "as_of": AS_OF, "timeframe": saved.request.timeframe,
              "provider": saved.request.provider, "layer": saved.layer,
              "pipeline_id": saved.pipeline_id or None, **kwargs}
    request = BatchRequest(("AAPL",), **values)
    return validation.validate_batch_data(pipeline, request, [json.loads(saved.to_json())], context(data, request))


@pytest.mark.parametrize("provider", ["yahoo", "databento-xnas.itch"])
@pytest.mark.parametrize("timeframe", ["1d", "1m", "1h"])
def test_complete_regular_grid_passes_without_quality_metadata(tmp_path, provider, timeframe):
    pipeline = DataPipeline(tmp_path)
    data = frame(timeframe)
    saved = save(pipeline, data, timeframe=timeframe, provider=provider)
    report = validate(pipeline, saved, data)
    assert report["status"] == "passed"
    assert report["expected_bar_count"] == report["actual_bar_count"] == data.height
    assert report["source_dataset_ids"] == [saved.dataset_id]
    assert report["calendar"] == "XNYS"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("missing_index", [0, 3, -1])
@pytest.mark.parametrize("reported", [False, True])
def test_leading_interior_trailing_missing_minutes_fail(tmp_path, missing_index, reported):
    pipeline = DataPipeline(tmp_path)
    data = frame("1m")
    missing = data["timestamp"][missing_index]
    quality = (FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(missing, "invalid_bar: open must be greater than zero"),))
               if reported else None)
    data = data.filter(pl.col("timestamp") != missing)
    saved = save(pipeline, data, timeframe="1m", quality=quality)
    with pytest.raises(ResearchError, match="Missing 1 required") as caught:
        validate(pipeline, saved, data)
    assert missing.isoformat() in str(caught.value)
    if reported:
        assert "open must be greater than zero" in str(caught.value)


def test_initialization_history_is_included_in_strict_validation(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame("1m")
    missing = data["timestamp"][3]
    data = data.filter(pl.col("timestamp") != missing)
    saved = save(pipeline, data, timeframe="1m")
    with pytest.raises(ResearchError, match="Missing 1 required") as caught:
        validate(pipeline, saved, data, start=SECOND)
    assert missing.isoformat() in str(caught.value)


def test_complete_support_history_is_reported(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame()
    saved = save(pipeline, data)
    report = validate(pipeline, saved, data, start=SECOND)
    assert report["initialization_bar_count"] == 1
    assert report["selected_bar_count"] == 1
    assert report["checked_start"] == START.isoformat()
    assert report["requested_start"] == SECOND.isoformat()


def test_no_bounds_only_certifies_available_span(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame().tail(1)
    saved = save(pipeline, data)
    report = validate(pipeline, saved, data, start=None, end=None)
    assert report["checked_start"] == SECOND.isoformat()
    assert report["checked_end_exclusive"] == END.isoformat()
    assert report["start_bound"] == report["end_bound"] == "available_span"
    assert any("beyond those bounds is not certified" in note for note in report["notes"])


def test_daily_exchange_close_and_analysis_conservative_completion_are_distinct(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame()
    saved = save(pipeline, data)
    report = validate(pipeline, saved, data, as_of=SECOND + timedelta(hours=22))
    assert report["actual_bar_count"] == 2
    assert report["analysis_completed_selected_bar_count"] == 1
    with pytest.raises(ResearchError, match="unfinished"):
        validate(pipeline, saved, data, as_of=SECOND + timedelta(hours=20))


def test_explicit_end_cannot_cut_off_requested_bar_before_exchange_close(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame()
    saved = save(pipeline, data)
    with pytest.raises(ResearchError, match="unfinished"):
        validate(pipeline, saved, data, end=SECOND + timedelta(hours=20))


def test_extended_hours_and_off_grid_rows_fail_instead_of_disappearing(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame("1m")
    extra = data.head(1).with_columns(pl.lit(START + timedelta(hours=12)).alias("timestamp"))
    data = pl.concat([extra.cast(data.schema), data]).sort("timestamp")
    saved = save(pipeline, data, timeframe="1m")
    with pytest.raises(ResearchError, match="unexpected bars outside the regular-session grid"):
        validate(pipeline, saved, data)


def test_frozen_historical_revision_is_never_replaced_by_active_corrected_data(tmp_path, monkeypatch):
    pipeline = DataPipeline(tmp_path)
    complete = frame("1m")
    missing = complete["timestamp"][3]
    original = complete.filter(pl.col("timestamp") != missing)
    saved = save(pipeline, original, timeframe="1m")
    pipeline.store.replace_raw(saved.dataset_id, complete, confirm=saved.dataset_id)
    monkeypatch.setattr(pipeline, "list_datasets", lambda *args, **kwargs: pytest.fail("Never repin active revisions"))
    with pytest.raises(ResearchError, match="Missing 1 required"):
        validate(pipeline, saved, original)


def test_complete_frozen_revision_remains_usable_after_replacement(tmp_path, monkeypatch):
    pipeline = DataPipeline(tmp_path)
    original = frame()
    saved = save(pipeline, original)
    pipeline.store.replace_raw(saved.dataset_id, original.with_columns(pl.lit(100.5).alias("close")),
                               confirm=saved.dataset_id)
    monkeypatch.setattr(pipeline, "list_datasets", lambda *args, **kwargs: pytest.fail("Never repin active revisions"))
    assert validate(pipeline, saved, original)["status"] == "passed"


def test_changed_frozen_metadata_fails(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame()
    saved = save(pipeline, data)
    source = json.loads(saved.to_json())
    source["checksum_sha256"] = "0" * 64
    request = BatchRequest(("AAPL",), start=START, end=END, as_of=AS_OF)
    with pytest.raises(ResearchError, match="Frozen source metadata changed"):
        validation.validate_batch_data(pipeline, request, [source], context(data, request))


@pytest.mark.parametrize("provider", ["yahoo", "databento-xnas.itch"])
def test_processed_full_grid_cannot_hide_missing_raw_minute(tmp_path, provider):
    pipeline = DataPipeline(tmp_path)
    data = frame("1m")
    missing = data["timestamp"][3]
    raw = save(pipeline, data.filter(pl.col("timestamp") != missing), timeframe="1m", provider=provider)
    processor = ResampleRegularHoursDaily.for_dates(START.date(), SECOND.date(), incomplete="aggregate_available")
    with warnings.catch_warnings(record=True):
        saved = pipeline.process(raw.dataset_id, processors=[processor])
    output = pipeline.read_dataset(saved.dataset_id)
    assert output.height == 2
    with pytest.raises(ResearchError, match="incomplete raw source coverage"):
        validate(pipeline, saved, output)


def test_processed_validation_checks_pinned_parent_checksums(tmp_path):
    pipeline = DataPipeline(tmp_path)
    raw = save(pipeline, frame())
    saved = pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    output = pipeline.read_dataset(saved.dataset_id)
    (pipeline.store.data_dir / raw.relative_path).write_bytes(b"corrupted")
    with pytest.raises(Exception, match="Checksum mismatch"):
        validate(pipeline, saved, output)


def test_complete_processed_data_passes_without_backtest_causality_policy(tmp_path, monkeypatch):
    class StaticScale(ScalePrices):
        name = "static_scale_validation_fixture"
        causal = False

    pipeline = DataPipeline(tmp_path)
    raw = save(pipeline, frame())
    saved = pipeline.process(raw.dataset_id, processors=[StaticScale(2)])
    registry = discover_processors()
    registry.register_class(StaticScale)
    monkeypatch.setattr(validation, "discover_processors", lambda: registry)
    report = validate(pipeline, saved, pipeline.read_dataset(saved.dataset_id))
    assert report["status"] == "passed"
    assert report["raw_parent_dataset_ids"] == [raw.dataset_id]


def test_unknown_weekly_grid_is_rejected(tmp_path):
    pipeline = DataPipeline(tmp_path)
    saved = save(pipeline, frame())
    with pytest.raises(ResearchError, match="supports"):
        validate(pipeline, saved, frame(), timeframe="1wk")


def test_explicit_supported_calendar_is_used(tmp_path):
    pipeline = DataPipeline(tmp_path)
    data = frame("1h", calendar="XLON")
    saved = save(pipeline, data, timeframe="1h")
    assert validate(pipeline, saved, data, calendar="XLON")["calendar"] == "XLON"
    with pytest.raises(ResearchError, match="Strict data validation failed"):
        validate(pipeline, saved, data, calendar="XNYS")


@pytest.mark.parametrize("required_omission", [False, True])
def test_strict_runner_allows_verified_processed_flat_analysis_only_for_complete_support(tmp_path, required_omission):
    pipeline = DataPipeline(tmp_path / "data")
    data = frame("1m")
    omitted = (data["timestamp"][3] if required_omission else START + timedelta(hours=22))
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(omitted, "invalid_bar: open must be greater than zero"),))
    raw = save(pipeline, data.filter(pl.col("timestamp") != omitted), timeframe="1m",
               provider="databento-xnas.itch", quality=quality)
    processor = ResampleRegularHoursDaily.for_dates(START.date(), SECOND.date(), incomplete="aggregate_available")
    with warnings.catch_warnings(record=True):
        output = pipeline.process(raw.dataset_id, processors=[processor])
    registry = LogicRegistry()
    registry.register_class(FlatLogic)
    store = BatchStore(tmp_path / "results")
    runner = BatchRunner(pipeline, registry, store)
    request = BatchRequest(("AAPL",), provider="databento-xnas.itch", timeframe="1d",
                           layer="processed", pipeline_id=output.pipeline_id,
                           start=START, end=END, as_of=AS_OF, strict_validation=True)
    manifest = runner.execute(request, [LogicSpec("flat", FlatLogic.name, FlatLogic.version,
                                                  {"min_duration": 2, "max_duration": 3, "max_range_pct": 5.})])
    result = store.load_ticker(manifest["run_id"], "AAPL")
    integrity = result["data_integrity"]
    # Original omission provenance is retained even when outside regular hours.
    assert integrity["processed_omission_sources"]
    if required_omission:
        assert result["status"] == "failed"
        assert integrity["validation"]["status"] == "failed"
        assert "incomplete raw source coverage" in result["error"]
        assert result["results"] == {}
    else:
        assert result["status"] == "completed"
        assert integrity["validation"]["status"] == "passed"
        assert integrity["messages"] == []
        assert "flat" in result["results"]
        assert result["errors"] == {}
