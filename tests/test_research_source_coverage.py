"""Strict research verifies the raw bars underlying a complete derived grid."""

from datetime import UTC, date, datetime, time
import warnings

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest, FetchQuality, OmittedBar
from data_pipeline.corporate_actions.models import ActionCoverage, ActionSnapshot
from data_pipeline.processing import Pipeline, ScalePrices
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.price_adjustment import AdjustCorporateActions
from data_pipeline.processing.processors.regular_hours import ResampleRegularHours30m
from research.datasets import prepare_snapshot
from research.errors import PreflightError
from research.instruments import Instrument, expected_bars


START = datetime(2024, 1, 2, tzinfo=UTC)
SECOND = datetime(2024, 1, 3, tzinfo=UTC)
END = datetime(2024, 1, 4, tzinfo=UTC)


def prices(timeframe="1m"):
    bars = expected_bars(Instrument("AAPL"), start=START, end=END, timeframe=timeframe)
    return pl.DataFrame({
        "timestamp": [bar.timestamp for bar in bars], "symbol": ["AAPL"] * len(bars),
        "open": [100.] * len(bars), "high": [102.] * len(bars),
        "low": [99.] * len(bars), "close": [101.] * len(bars), "volume": [10.] * len(bars),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


def adjustment():
    snapshot = ActionSnapshot("verified-fixture", date(2024, 1, 3), END, (
        ActionCoverage("AAPL", date(2024, 1, 2), date(2024, 1, 3),
                       source_basis="unadjusted", security_id="aapl"),
    ), ())
    return AdjustCorporateActions(snapshot)


def save_daily(pipeline, frame, *, provider="yahoo", quality=None, mode="strict", incomplete="aggregate_available"):
    raw = pipeline.store.write_raw(DataRequest("AAPL", START, END, timeframe="1m", provider=provider),
                                   frame, quality=quality)
    processors = [ResampleRegularHoursDaily.for_dates(START.date(), SECOND.date(), incomplete=incomplete)]
    if mode == "adjusted_price":
        processors.insert(0, adjustment())
    with warnings.catch_warnings(record=True):
        return pipeline.process(raw.dataset_id, processors=processors)


def select(pipeline, output, *, provider="yahoo", mode="strict", start=START, **kwargs):
    return prepare_snapshot(pipeline, tickers=["AAPL"], start=start, end=END,
                            timeframe="1d", lookback=0, provider=provider, layer="processed",
                            pipeline_id=output.pipeline_id, research_mode=mode, **kwargs)


@pytest.mark.parametrize("provider,mode", [
    ("yahoo", "strict"), ("databento-xnas.itch", "strict"), ("databento-xnas.itch", "adjusted_price"),
])
@pytest.mark.parametrize("reported", [False, True])
def test_partial_aggregate_fails_research_even_without_reported_omission(tmp_path, provider, reported, mode):
    pipeline = DataPipeline(tmp_path)
    frame = prices()
    missing = frame["timestamp"][3]
    quality = (FetchQuality(status="reported", skip_invalid_bars=True,
                            omitted_bars=(OmittedBar(missing, "invalid_bar: open must be greater than zero"),))
               if reported else None)
    output = save_daily(pipeline, frame.filter(pl.col("timestamp") != missing),
                        provider=provider, quality=quality, mode=mode)
    # Both daily labels exist; checking the derived grid alone conceals the gap.
    assert pipeline.read_dataset(output.dataset_id).height == 2
    with pytest.raises(PreflightError, match="incomplete raw source coverage") as caught:
        select(pipeline, output, provider=provider, mode=mode)
    assert "missing 1 required 1m source bars" in str(caught.value)
    assert missing.isoformat() in str(caught.value)
    if reported:
        assert "open must be greater than zero" in str(caught.value)


def test_leading_gap_option_cannot_hide_incomplete_first_output_bar(tmp_path):
    pipeline = DataPipeline(tmp_path)
    output = save_daily(pipeline, prices().slice(1))
    with pytest.raises(PreflightError, match="incomplete raw source coverage"):
        select(pipeline, output, skip_initial_missing_bars=True)


def test_missing_underlying_warmup_bar_fails_even_when_execution_data_is_complete(tmp_path):
    pipeline = DataPipeline(tmp_path)
    output = save_daily(pipeline, prices().slice(1))
    with pytest.raises(PreflightError, match="incomplete raw source coverage"):
        prepare_snapshot(pipeline, tickers=["AAPL"], start=SECOND, end=END,
                         timeframe="1d", lookback=1, layer="processed", pipeline_id=output.pipeline_id)


def test_intraday_aggregate_cannot_conceal_missing_source_minute(tmp_path):
    pipeline = DataPipeline(tmp_path)
    frame = prices()
    missing = frame["timestamp"][3]
    raw = pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m"),
                                frame.filter(pl.col("timestamp") != missing)).raw
    processor = ResampleRegularHours30m.for_dates(START.date(), SECOND.date(), incomplete="aggregate_available")
    with warnings.catch_warnings(record=True):
        output = pipeline.process(raw.dataset_id, processors=[processor])
    assert pipeline.read_dataset(output.dataset_id).height == 26
    with pytest.raises(PreflightError, match="incomplete raw source coverage"):
        prepare_snapshot(pipeline, tickers=["AAPL"], start=START, end=END,
                         timeframe="30m", lookback=0, layer="processed", pipeline_id=output.pipeline_id)


@pytest.mark.parametrize("mode", ["strict", "adjusted_price"])
def test_complete_sources_pass_despite_skip_policy_and_repaired_omission_metadata(tmp_path, mode):
    pipeline = DataPipeline(tmp_path)
    provider = "databento-xnas.itch" if mode == "adjusted_price" else "yahoo"
    frame = prices()
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(frame["timestamp"][3], "historical omission, repaired"),))
    output = save_daily(pipeline, frame, quality=quality, mode=mode, provider=provider)
    assert select(pipeline, output, mode=mode, provider=provider).report.ok


def test_gaps_outside_selected_output_support_do_not_block_research(tmp_path):
    pipeline = DataPipeline(tmp_path)
    frame = prices()
    missing = frame["timestamp"][3]
    outside = datetime(2024, 1, 3, 22, tzinfo=UTC)
    quality = FetchQuality(status="reported", skip_invalid_bars=True, omitted_bars=(
        OmittedBar(missing, "invalid_bar: open is missing (null)"),
        OmittedBar(outside, "invalid_bar: open is missing (null)"),
    ))
    output = save_daily(pipeline, frame.filter(pl.col("timestamp") != missing), quality=quality)
    assert select(pipeline, output, start=SECOND).report.ok


def test_explicit_initial_history_allowance_still_allows_wholly_missing_earlier_session(tmp_path):
    pipeline = DataPipeline(tmp_path)
    output = save_daily(pipeline, prices().filter(pl.col("timestamp") >= SECOND), incomplete="drop")
    result = select(pipeline, output, skip_initial_missing_bars=True)
    assert result.report.ok
    assert result.effective_starts["AAPL"] == SECOND
    assert any("skipped 1 initial missing bars" in note for note in result.report.notes)


def test_same_timeframe_output_cannot_conceal_missing_pinned_parent_bar(tmp_path):
    pipeline = DataPipeline(tmp_path)
    frame = prices("1d")
    raw = pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1d"), frame.slice(1)).raw
    configured = Pipeline([ScalePrices(1)])
    # Low-level storage accepts valid derived OHLCV with declared lineage;
    # strict research must still verify that every output had a source bar.
    output = pipeline.store.write_processed(raw.dataset_id, frame, pipeline_id=configured.fingerprint,
                                            processors_json=configured.canonical_json)
    with pytest.raises(PreflightError, match="missing 1 required 1d source bars"):
        select(pipeline, output)


@pytest.mark.parametrize("timeframe", ["1d", "30m"])
@pytest.mark.parametrize("window_end,valid", [(time(15, 45), False), (time(16), True),
                                               (time(9, 30), True)])
def test_intermediate_time_filter_cannot_hide_partial_aggregates(tmp_path, timeframe, window_end, valid):
    from data_pipeline.processing.processors.time_window import TimeWindowFilter

    pipeline = DataPipeline(tmp_path)
    frame = prices()
    raw = pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m"), frame).raw
    aggregate = (ResampleRegularHoursDaily if timeframe == "1d" else ResampleRegularHours30m)
    processors = [TimeWindowFilter(time(9, 30), window_end),
                  aggregate.for_dates(START.date(), SECOND.date(), incomplete="aggregate_available")]
    with warnings.catch_warnings(record=True):
        output = pipeline.process(raw.dataset_id, processors=processors)
    # A partial final interval still has its ordinary label, even though 15
    # required minutes were removed from it by the earlier filter.
    assert pipeline.read_dataset(output.dataset_id).height == (2 if timeframe == "1d" else 26)
    options = dict(tickers=["AAPL"], start=START, end=END, timeframe=timeframe, lookback=0,
                   layer="processed", pipeline_id=output.pipeline_id)
    if valid:
        assert prepare_snapshot(pipeline, **options).report.ok
    else:
        with pytest.raises(PreflightError, match="Time-window filter removes 30 required 1m source bars"):
            prepare_snapshot(pipeline, **options)


def test_overnight_filter_window_preserves_required_regular_session(tmp_path):
    from data_pipeline.processing.processors.time_window import TimeWindowFilter

    pipeline = DataPipeline(tmp_path)
    raw = pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m"), prices()).raw
    output = pipeline.process(raw.dataset_id, processors=[
        TimeWindowFilter(time(9), time(8)),
        ResampleRegularHoursDaily.for_dates(START.date(), SECOND.date(), incomplete="aggregate_available"),
    ])
    assert select(pipeline, output).report.ok
