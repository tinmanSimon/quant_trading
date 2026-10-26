"""Flat periods use one calculation contract in Market Data and batch research."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import math

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest
from research.analysis import LogicContext, LogicRunner, LogicSpec, load_registry
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore
from research.analysis.builtins.flat import FlatLogic


START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
SETTINGS = {"min_duration": 3, "max_duration": 200, "max_range_pct": 2.0}


def flat_bars(symbol="TEST"):
    prices = [100.] * 6 + [120.] * 6 + [140.] * 6
    return pl.DataFrame({
        "timestamp": [START + timedelta(minutes=30 * index) for index in range(len(prices))],
        "symbol": [symbol] * len(prices), "open": prices,
        "high": [price + 1 for price in prices], "low": [price - 1 for price in prices],
        "close": prices, "volume": [100.] * len(prices),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


def period_rows(result):
    tables = result.tables if hasattr(result, "tables") else result["tables"]
    return next(table["rows"] for table in tables if table["name"] == "Flat periods")


def summary_row(result):
    tables = result.tables if hasattr(result, "tables") else result["tables"]
    return next(table["rows"][0] for table in tables if table["name"] == "Flat summary")


def test_market_data_and_analysis_discover_same_flat_settings():
    from dashboard.market_data.analysis_controls import discover_logics
    from dashboard.pages.analysis import load_registry as analysis_registry

    definitions = [registry.definition(FlatLogic.name, FlatLogic.version)
                   for registry in (discover_logics(), analysis_registry())]
    assert definitions[0] == definitions[1]
    assert definitions[0].label == "Flat analysis"
    properties = definitions[0].schema["properties"]
    assert set(properties) == {
        "min_duration", "max_duration", "max_range_pct", "band_width_mode", "auto_percentile",
        "auto_multiplier", "long_bar_filter", "long_bar_multiplier",
    }
    assert properties["min_duration"]["default"] == 20
    assert properties["max_duration"]["default"] == 200
    assert properties["max_range_pct"]["default"] == 2.0
    assert properties["max_range_pct"]["exclusiveMinimum"] == 0
    assert properties["max_range_pct"]["exclusiveMaximum"] == 200
    assert properties["band_width_mode"]["default"] == "fixed"
    assert properties["band_width_mode"]["enum"] == ["fixed", "automatic"]
    assert properties["auto_percentile"]["default"] == 90.0
    assert properties["auto_multiplier"]["default"] == 2.0
    assert "auto_lookback" not in properties
    assert properties["long_bar_filter"]["default"] is False
    assert properties["long_bar_multiplier"]["default"] == 2.0
    assert FlatLogic.result_version == "3"
    assert FlatLogic.input_policy == "completed"


@pytest.mark.parametrize("maximum", [2, 3])
def test_maximum_duration_must_exceed_minimum_in_both_dashboard_configuration_paths(maximum):
    from dashboard.market_data.analysis_controls import discover_logics
    from dashboard.pages.analysis import load_registry as analysis_registry

    config = {**SETTINGS, "max_duration": maximum}
    for registry in (discover_logics(), analysis_registry()):
        with pytest.raises(ValueError):
            registry.configure(FlatLogic.name, FlatLogic.version, config)


@pytest.mark.parametrize("width", [0, -1, 200, float("inf"), float("nan")])
def test_invalid_band_width_preserves_previous_calculation_and_display(width):
    registry = load_registry()
    runner = LogicRunner(registry)
    context = LogicContext(flat_bars(), "30m", "TEST", START + timedelta(days=1))
    spec = LogicSpec("flat-instance", FlatLogic.name, FlatLogic.version, SETTINGS)
    applied = runner.apply([spec], context)
    assert not applied.errors
    previous = deepcopy(runner.present(applied).results[spec.instance_id].to_dict())
    with pytest.raises(ValueError):
        runner.apply([LogicSpec(spec.instance_id, spec.name, spec.version,
                                {**SETTINGS, "max_range_pct": width})], context)
    assert runner.present(applied).results[spec.instance_id].to_dict() == previous


@pytest.mark.parametrize("changes", [
    {"band_width_mode": "unknown"}, {"auto_percentile": -1}, {"auto_percentile": 101},
    {"auto_percentile": float("nan")}, {"auto_multiplier": 0},
    {"auto_multiplier": float("inf")}, {"auto_lookback": 100},
    {"long_bar_filter": "yes"}, {"long_bar_multiplier": .5},
    {"long_bar_multiplier": float("nan")},
])
def test_invalid_automatic_or_long_bar_settings_preserve_applied_results(changes):
    runner = LogicRunner(load_registry())
    context = LogicContext(flat_bars(), "30m", "TEST", START + timedelta(days=1))
    spec = LogicSpec("flat-instance", FlatLogic.name, FlatLogic.version, SETTINGS)
    applied = runner.apply([spec], context)
    assert not applied.errors
    previous = deepcopy(runner.present(applied).results[spec.instance_id].to_dict())
    with pytest.raises(ValueError):
        runner.apply([LogicSpec(spec.instance_id, spec.name, spec.version,
                                {**SETTINGS, **changes})], context)
    assert runner.present(applied).results[spec.instance_id].to_dict() == previous


def test_calculation_publishes_disjoint_table_and_matching_bands_before_presentation():
    frame = flat_bars()
    result = FlatLogic().calculate(LogicContext(frame, "30m", "TEST", START + timedelta(days=1)), SETTINGS)
    result.validate()
    rows = period_rows(result)
    assert [(row["start_index"], row["end_index"]) for row in rows] == [(12, 17), (6, 11), (0, 5)]
    assert [row["bars"] for row in rows] == [6, 6, 6]
    assert [row["ends_at_current_bar"] for row in rows] == [True, False, False]
    bands = {overlay["id"]: overlay for overlay in result.overlays if overlay["type"] == "log_line"}
    assert set(bands) == {row["period_id"] + suffix for row in rows for suffix in ("", "-lower", "-upper")}
    for row in rows:
        for suffix, field in (("", "center_price"), ("-lower", "lower_price"), ("-upper", "upper_price")):
            line = bands[row["period_id"] + suffix]
            assert (line["start_index"], line["end_index"]) == (row["start_index"], row["end_index"])
            assert line["slope"] == 0
            assert math.exp(line["intercept"]) == pytest.approx(row[field])
        candles = frame.slice(row["start_index"], row["bars"])
        assert row["lower_price"] == candles["low"].min()
        assert row["upper_price"] == candles["high"].max()
        assert row["range_pct"] == pytest.approx(100 * (row["upper_price"] - row["lower_price"]) / row["center_price"])
        assert {"slope", "slope_percent", "fitted_change_percent"}.isdisjoint(row)
    events = result.data["analysis"]["events"]
    assert [event["id"] for event in events] == [row["_event_id"] for row in rows]
    assert [(event["source_index"], event["end_index"]) for event in events] == [(12, 17), (6, 11), (0, 5)]
    boundaries, = [overlay for overlay in result.overlays if overlay["type"] == "markers"]
    assert boundaries["x"] == [0, 5, 6, 11, 12, 17]
    summary = summary_row(result)
    assert summary["period_count"] == 3
    assert summary["covered_bars"] == summary["source_bars"] == 18
    assert summary["coverage_percent"] == pytest.approx(100.)
    assert summary["current_flat"] is True and summary["current_duration"] == 6


def test_display_toggles_preserve_calculated_periods_and_cached_archive():
    registry = load_registry()
    runner = LogicRunner(registry)
    spec = LogicSpec("flat-instance", FlatLogic.name, FlatLogic.version, SETTINGS)
    context = LogicContext(flat_bars(), "30m", "TEST", START + timedelta(days=1))
    applied = runner.apply([spec], context)
    assert not applied.errors
    original = deepcopy(applied.results[spec.instance_id].to_dict())
    hidden = runner.present(applied, {spec.instance_id: {"show_fit": False, "show_boundaries": False}})
    assert not hidden.errors
    assert hidden.results[spec.instance_id].overlays == []
    assert period_rows(hidden.results[spec.instance_id]) == period_rows(original)
    assert applied.results[spec.instance_id].to_dict() == original
    fits_only = runner.present(applied, {spec.instance_id: {"show_fit": True, "show_boundaries": False}})
    assert not fits_only.errors
    assert {item["type"] for item in fits_only.results[spec.instance_id].overlays} == {"log_line"}
    assert period_rows(fits_only.results[spec.instance_id]) == period_rows(original)


@pytest.mark.parametrize("scale", [1., 1000.])
def test_period_table_reports_entire_envelope_as_percent_of_midpoint(scale):
    frame = flat_bars().head(6).with_columns(
        pl.lit(100. * scale).alias("open"), pl.lit(100. * scale).alias("close"),
        pl.lit(102. * scale).alias("high"), pl.lit(99. * scale).alias("low"),
    )
    result = FlatLogic().calculate(LogicContext(frame, "30m", "TEST", START + timedelta(days=1)),
                                  {**SETTINGS, "max_range_pct": 3.0})
    row, = period_rows(result)
    assert (row["start_index"], row["end_index"]) == (0, 5)
    assert row["lower_price"] == 99. * scale
    assert row["upper_price"] == 102. * scale
    assert row["center_price"] == 100.5 * scale
    assert row["range_pct"] == pytest.approx(100 * 3 / 100.5)


def test_unchanged_closes_do_not_accept_candles_with_a_wide_high_low_range():
    frame = flat_bars().with_columns(
        pl.lit(100.).alias("open"), pl.lit(100.).alias("close"),
        pl.lit(110.).alias("high"), pl.lit(90.).alias("low"),
    )
    result = FlatLogic().calculate(LogicContext(frame, "30m", "TEST", START + timedelta(days=1)), SETTINGS)
    assert period_rows(result) == []
    assert result.overlays == []
    assert summary_row(result)["period_count"] == 0
    assert summary_row(result)["current_flat"] is False


def test_automatic_width_uses_candidate_candles_including_endpoint_without_extra_history():
    frame = flat_bars().head(6).with_columns(
        pl.when(pl.int_range(pl.len()) == 5).then(104.).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.int_range(pl.len()) == 5).then(96.).otherwise(pl.col("low")).alias("low"),
    )
    result = FlatLogic().calculate(LogicContext(frame, "30m", "TEST", START + timedelta(days=1)),
                                  {**SETTINGS, "band_width_mode": "automatic"})
    row, = period_rows(result)
    assert (row["start_index"], row["end_index"]) == (0, 5)
    # Candidate lengths [2, 2, 2, 2, 2, 8]: linear p90 is 5%; multiplier 2 gives 10%.
    assert row["range_pct"] == pytest.approx(8.)
    assert row["max_range_pct"] == pytest.approx(10.)
    assert summary_row(result)["source_bars"] == 6


def test_one_wide_candle_ends_the_current_band_even_when_close_is_unchanged():
    frame = flat_bars().head(6).with_columns(
        pl.when(pl.int_range(pl.len()) == 5).then(110.).otherwise(pl.col("high")).alias("high"))
    result = FlatLogic().calculate(LogicContext(frame, "30m", "TEST", START + timedelta(days=1)), SETTINGS)
    row, = period_rows(result)
    assert (row["start_index"], row["end_index"]) == (0, 4)
    assert row["ends_at_current_bar"] is False
    assert row["upper_price"] == 101.
    assert summary_row(result)["current_flat"] is False


@pytest.mark.parametrize("maximum,intervals,current_duration", [
    (200, [(12, 16), (6, 11), (0, 5)], 5),
    (4, [(13, 16), (8, 11), (2, 5)], 4),
])
def test_processed_batch_preserves_final_periods_and_completed_30m_cutoff(
        tmp_path, maximum, intervals, current_duration):
    pipeline = DataPipeline(tmp_path / "data")
    # The last candle is still incomplete at cutoff; its large wick must not
    # contaminate the latest completed band's observed envelope.
    frame = flat_bars().with_columns(
        pl.when(pl.int_range(pl.len()) == 17).then(1000.).otherwise(pl.col("high")).alias("high"))
    raw_prices = pl.Series("close", [100. + index * 10 for index in range(frame.height)])
    raw = frame.with_columns(raw_prices).with_columns(
        pl.col("close").alias("open"), (pl.col("close") + 1).alias("high"),
        (pl.col("close") - 1).alias("low"),
    )
    parent = pipeline.store.write_raw(DataRequest(
        "TEST", START, START + timedelta(hours=9), provider="databento", timeframe="30m"), raw)
    processors = json.dumps([{"name": "flat_fixture", "version": "1", "config": {}}],
                            sort_keys=True, separators=(",", ":"))
    fingerprint = sha256(processors.encode()).hexdigest()
    processed = pipeline.store.write_processed(parent.dataset_id, frame, pipeline_id=fingerprint,
                                               processors_json=processors)
    cutoff = frame["timestamp"][-1] + timedelta(minutes=15)
    registry = load_registry()
    spec = LogicSpec("flat-instance", FlatLogic.name, FlatLogic.version,
                     {**SETTINGS, "max_duration": maximum})
    expected = LogicRunner(registry).apply([spec], LogicContext(frame, "30m", "TEST", cutoff))
    assert not expected.errors
    store = BatchStore(tmp_path / "results")
    manifest = BatchRunner(pipeline, registry, store).execute(BatchRequest(
        ("TEST",), provider="databento", timeframe="30m", layer="processed",
        pipeline_id=fingerprint, as_of=cutoff), [spec])
    assert manifest["counts"]["completed"] == 1
    assert manifest["sources"]["TEST"][0]["dataset_id"] == processed.dataset_id
    assert manifest["specs"][0]["config"]["max_duration"] == maximum
    assert manifest["specs"][0]["config"]["min_duration"] == 3
    assert manifest["specs"][0]["config"]["max_range_pct"] == 2.0
    saved = BatchStore(store.directory).load_ticker(manifest["run_id"], "TEST")["results"][spec.instance_id]
    assert saved["tables"] == expected.results[spec.instance_id].tables
    assert saved["overlays"] == []
    assert [(row["start_index"], row["end_index"]) for row in period_rows(saved)] == intervals
    assert all(3 <= row["bars"] <= maximum for row in period_rows(saved))
    assert all(row["range_pct"] <= 2.0 for row in period_rows(saved))
    assert period_rows(saved)[0]["upper_price"] == 141.
    assert summary_row(saved)["source_bars"] == 17
    assert summary_row(saved)["current_duration"] == current_duration
