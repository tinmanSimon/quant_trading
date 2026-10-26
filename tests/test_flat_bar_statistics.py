"""Flat-period counts and volatility use one completed, disjoint population."""

from datetime import UTC, datetime, timedelta
from copy import deepcopy
import json

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataRequest
from research import Research
from research.analysis import LogicContext, LogicSpec, load_registry
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore
from research.analysis.batch.aggregation import aggregate_metrics, extract_metrics, infer_metrics
from research.analysis.builtins.flat import FlatLogic
from research.analysis.builtins.flat_bar_statistics import FlatBarStatisticsLogic
from research.analysis.flat import FlatConfig, analyze_summary


START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
SETTINGS = {"min_duration": 3, "max_duration": 20, "max_range_pct": 6., "high_percentage": 102.}


def sample_bars(symbol="TEST"):
    """Five flat candles, one rejected spike, then three flat candles."""
    return pl.DataFrame({
        "timestamp": [START + timedelta(minutes=30 * i) for i in range(9)],
        "symbol": [symbol] * 9,
        "open": [100.] * 5 + [150.] + [200.] * 3,
        "high": [103., 102., 101., 103., 103., 160., 206., 204., 205.],
        "low": [98.] * 5 + [140.] + [196.] * 3,
        "close": [99., 99.5, 99.8, 101., 100., 145., 199., 201., 198.],
        "volume": [100.] * 9,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


def calculate(frame=None, **settings):
    frame = sample_bars() if frame is None else frame
    context = LogicContext(frame, "30m", "test", START + timedelta(days=1))
    return FlatBarStatisticsLogic().calculate(context, {**SETTINGS, **settings})


def summary(result):
    return result.tables[0]["rows"][0]


def recent_summary(result):
    return result.tables[1]["rows"][0]


def calculate_dated(frame, dates):
    frame = frame.with_columns(pl.Series("timestamp", dates))
    return FlatBarStatisticsLogic().calculate(
        LogicContext(frame, "30m", "test", dates[-1] + timedelta(days=1)), SETTINGS)


def test_last_month_clips_crossing_period_and_excludes_return_across_cutoff():
    dates = [datetime.fromisoformat(day).replace(tzinfo=UTC) for day in
             ["2024-01-01", "2024-01-02", "2024-02-28", "2024-02-29", "2024-03-01",
              "2024-03-10", "2024-03-29", "2024-03-30", "2024-03-31"]]
    result = calculate_dated(sample_bars(), dates)
    result.validate()
    recent = recent_summary(result)
    assert summary(result) == summary(calculate())
    assert recent["last_month_cutoff"] == dates[3].isoformat()
    assert recent["last_month_start"] == dates[3].isoformat()
    assert recent["last_month_end"] == dates[-1].isoformat()
    assert {key: value for key, value in recent.items() if key.endswith("_count")} == {
        "last_month_source_bar_count": 6, "last_month_flat_period_count": 2,
        "last_month_flat_bar_count": 5, "last_month_red_bar_count": 2,
        "last_month_red_high_above_count": 2, "last_month_red_high_not_above_count": 0,
        "last_month_all_high_above_count": 4, "last_month_return_count": 3,
    }
    assert recent["last_month_return_volatility_pct"] == pytest.approx(
        np.std([100 / 101 - 1, 201 / 199 - 1, 198 / 201 - 1]) * 100)
    assert recent["last_month_red_high_above_ratio"] == 1.0


@pytest.mark.parametrize("end,cutoff", [
    ("2024-03-31", "2024-02-29"), ("2023-03-31", "2023-02-28"),
    ("2024-01-31", "2023-12-31"), ("2024-05-31", "2024-04-30"),
])
def test_last_month_uses_calendar_month_and_includes_exact_cutoff(end, cutoff):
    end = datetime.fromisoformat(end).replace(hour=14, minute=30, tzinfo=UTC)
    cutoff = datetime.fromisoformat(cutoff).replace(hour=14, minute=30, tzinfo=UTC)
    dates = [cutoff - timedelta(minutes=30), cutoff, end]
    recent = recent_summary(calculate_dated(sample_bars().head(3), dates))
    assert recent["last_month_cutoff"] == recent["last_month_start"] == cutoff.isoformat()
    assert recent["last_month_end"] == end.isoformat()
    assert recent["last_month_flat_bar_count"] == 2
    assert recent["last_month_return_count"] == 1


def test_last_month_single_flat_bar_has_no_return_or_volatility():
    dates = [START + timedelta(days=i) for i in range(8)] + [datetime(2024, 3, 31, tzinfo=UTC)]
    recent = recent_summary(calculate_dated(sample_bars(), dates))
    assert recent["last_month_source_bar_count"] == recent["last_month_flat_bar_count"] == 1
    assert recent["last_month_flat_period_count"] == 1
    assert recent["last_month_red_high_above_count"] == 1
    assert recent["last_month_return_count"] == 0
    assert recent["last_month_return_volatility_pct"] is None


def test_last_month_anchors_to_source_end_even_when_recent_bars_are_not_flat():
    dates = [START + timedelta(days=i) for i in range(5)] + [datetime(2024, 3, 31, tzinfo=UTC)]
    result = calculate_dated(sample_bars().head(6), dates)
    assert summary(result)["flat_bar_count"] == 5
    recent = recent_summary(result)
    assert recent["last_month_end"] == dates[-1].isoformat()
    assert recent["last_month_source_bar_count"] == 1
    assert recent["last_month_flat_bar_count"] == recent["last_month_flat_period_count"] == 0
    assert recent["last_month_return_volatility_pct"] is None


def test_short_history_last_month_matches_full_summary_and_shows_available_start():
    result = calculate()
    recent = recent_summary(result)
    assert {key: recent[f"last_month_{key}"] for key in summary(result)} == summary(result)
    assert recent["last_month_start"] == START.isoformat()
    assert recent["last_month_cutoff"] == (START.replace(month=12, year=2023) + timedelta(hours=4)).isoformat()


@pytest.mark.parametrize("old_version", ["1.0.0-engine.2.2.0", "1.1.0-engine.2.2.0", "1.2.0-engine.2.2.0"])
def test_statistics_upgrade_preserves_configuration_and_applied_results(old_version):
    from dashboard.state.migrations import upgrade_flat_bar_statistics_settings
    from research.analysis.core.registry import LogicRegistry

    old = {"instance_id": "saved-stats", "name": FlatBarStatisticsLogic.name,
           "version": old_version, "config": dict(SETTINGS)}
    draft, applied = [deepcopy(old)], [deepcopy(old)]
    assert not upgrade_flat_bar_statistics_settings(draft, LogicRegistry())
    assert upgrade_flat_bar_statistics_settings(draft, load_registry())
    assert draft == [{**old, "version": FlatBarStatisticsLogic.version}]
    assert applied == [old]
    assert not upgrade_flat_bar_statistics_settings(draft, load_registry())


def test_discovery_reuses_every_flat_setting_and_has_explicit_percentage_semantics():
    registry = load_registry()
    definition = registry.definition(FlatBarStatisticsLogic.name, FlatBarStatisticsLogic.version)
    properties = definition.schema["properties"]
    assert {key: properties[key] for key in FlatLogic.config_schema["properties"]} == FlatLogic.config_schema["properties"]
    assert properties["high_percentage"]["default"] == 102.
    assert "102 means high > open" in properties["high_percentage"]["description"]
    assert "auto_lookback" not in properties
    configured = registry.configure(definition.name, definition.version, SETTINGS)
    assert configured["band_width_mode"] == "fixed"
    assert configured["long_bar_filter"] is False
    assert FlatBarStatisticsLogic.input_policy == "completed"


def test_counts_include_green_and_doji_highs_but_only_close_below_open_is_red():
    frame = sample_bars()
    before = frame.clone()
    result = calculate(frame)
    result.validate()
    row = summary(result)
    assert {key: row[key] for key in row if key != "return_volatility_pct"} == {
        "source_bar_count": 9, "flat_period_count": 2, "flat_bar_count": 8,
        "red_bar_count": 5, "red_high_above_count": 3,
        "red_high_not_above_count": 2, "all_high_above_count": 5, "return_count": 6,
        "red_high_above_ratio": 0.6,
    }
    details = result.tables[2]["rows"]
    assert [(row["start_index"], row["end_index"]) for row in details] == [(6, 8), (0, 4)]
    assert [(row["red_high_above_count"], row["red_high_not_above_count"]) for row in details] == [(2, 0), (1, 2)]
    assert_frame_equal(frame, before)
    assert result.overlays == []
    assert result.data["metadata"]["flat_status"] == "ok"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("percentage,red_above,all_above", [(100., 5, 8), (102., 3, 5), (103., 0, 0)])
def test_high_threshold_is_a_direct_percentage_and_equality_is_not_above(percentage, red_above, all_above):
    row = summary(calculate(high_percentage=percentage))
    assert row["red_high_above_count"] == red_above
    assert row["red_high_not_above_count"] == 5 - red_above
    assert row["all_high_above_count"] == all_above


def test_return_volatility_pools_only_within_period_returns_without_cross_period_jumps():
    result = calculate()
    closes = [[199., 201., 198.], [99., 99.5, 99.8, 101., 100.]]
    samples = [np.array(values[1:]) / values[:-1] - 1 for values in closes]
    expected = np.std(np.concatenate(samples), ddof=0) * 100
    assert summary(result)["return_volatility_pct"] == pytest.approx(expected)
    assert summary(result)["return_count"] == sum(map(len, samples))
    for row, returns in zip(result.tables[2]["rows"], samples):
        assert row["return_volatility_pct"] == pytest.approx(np.std(returns, ddof=0) * 100)
    assert expected != pytest.approx(np.mean([np.std(values) * 100 for values in samples]))
    joined_closes = np.array(closes[1] + closes[0])
    assert expected != pytest.approx(np.std(joined_closes[1:] / joined_closes[:-1] - 1) * 100)


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_detection_matches_shared_flat_package_with_all_custom_settings(automatic, filtered):
    settings = {**SETTINGS, "band_width_mode": "automatic" if automatic else "fixed",
                "auto_percentile": 75., "auto_multiplier": 1.5,
                "long_bar_filter": filtered, "long_bar_multiplier": 1.4, "max_duration": 4}
    frame = sample_bars()
    expected = analyze_summary(frame, config=FlatConfig.from_dict(
        {key: value for key, value in settings.items() if key != "high_percentage"}))
    result = calculate(frame, **settings)
    details = result.tables[2]["rows"]
    assert [(row["start_index"], row["end_index"]) for row in details] == [
        (period.start_index, period.end_index) for period in expected.periods]
    covered = [i for period in expected.periods for i in range(period.start_index, period.end_index + 1)]
    assert summary(result)["flat_bar_count"] == len(covered) == len(set(covered))
    assert summary(result)["return_count"] == len(covered) - len(expected.periods)
    for row, period in zip(details, expected.periods):
        assert row["max_range_pct"] == period.max_range_pct


@pytest.mark.parametrize("case,status", [("empty", "insufficient_data"), ("short", "insufficient_data"),
                                         ("trending", "no_flat_period")])
def test_no_periods_produce_zero_counts_and_null_volatility(case, status):
    frame = sample_bars().head(0 if case == "empty" else 2 if case == "short" else 3)
    if case == "trending":
        frame = frame.with_columns(pl.Series("open", [100., 130., 160.])).with_columns(
            (pl.col("open") + 1).alias("high"), (pl.col("open") - 1).alias("low"),
            pl.col("open").alias("close"))
    result = calculate(frame)
    assert result.data["metadata"]["flat_status"] == status
    assert result.tables[2]["rows"] == []
    row = summary(result)
    assert row["return_volatility_pct"] is None
    assert row["red_high_above_ratio"] is None
    assert all(value == 0 for key, value in row.items()
               if key not in {"return_volatility_pct", "source_bar_count", "red_high_above_ratio"})
    result.validate()
    recent = recent_summary(result)
    assert recent["last_month_flat_bar_count"] == 0
    assert recent["last_month_return_volatility_pct"] is None
    if case == "empty":
        assert recent["last_month_cutoff"] is recent["last_month_start"] is recent["last_month_end"] is None


def test_zero_returns_and_no_red_bars_are_measured_as_zero_not_missing():
    frame = sample_bars().head(4).with_columns(pl.lit(100.).alias("close"))
    row = summary(calculate(frame))
    assert row["red_bar_count"] == row["red_high_above_count"] == row["red_high_not_above_count"] == 0
    assert row["red_high_above_ratio"] is None
    assert row["all_high_above_count"] == 2
    assert row["return_count"] == 3
    assert row["return_volatility_pct"] == 0


def test_single_return_has_zero_population_volatility():
    row = summary(calculate(sample_bars().head(2), min_duration=2))
    assert row["return_count"] == 1
    assert row["return_volatility_pct"] == 0


def test_incomplete_endpoint_is_excluded_from_detection_counts_and_volatility():
    frame = sample_bars()
    future = frame.tail(1).with_columns(
        pl.lit(START + timedelta(minutes=30 * 9)).alias("timestamp"),
        pl.lit(1000.).alias("high"),
    ).cast(frame.schema)
    context = LogicContext(pl.concat([frame, future]), "30m", "test",
                           START + timedelta(minutes=30 * 9 + 15))
    result = FlatBarStatisticsLogic().calculate(context, SETTINGS)
    assert result.tables == calculate(frame).tables


def test_price_scaling_preserves_counts_and_volatility():
    frame = sample_bars()
    scaled = frame.with_columns(pl.col(name) * 1e6 for name in ("open", "high", "low", "close"))
    before, after = summary(calculate(frame)), summary(calculate(scaled))
    assert after == pytest.approx(before)


@pytest.mark.parametrize("percentage", [0, -1, True, "102", None, float("nan"), float("inf"), 10 ** 400])
def test_invalid_high_thresholds_fail_cleanly(percentage):
    with pytest.raises(ValueError, match="high_percentage"):
        FlatBarStatisticsLogic().validate({**SETTINGS, "high_percentage": percentage})


@pytest.mark.parametrize("column", ["open", "close"])
def test_open_and_close_are_required(column):
    with pytest.raises(ValueError, match=column):
        calculate(sample_bars().drop(column))


@pytest.mark.parametrize("size", [5, 9])
def test_summary_metrics_sum_counts_but_do_not_sum_volatilities_or_duplicate_details(size):
    result = calculate(sample_bars().head(size))
    specs = infer_metrics(result.tables)
    assert {spec.table for spec in specs} == {"Flat-bar summary", "Last-month flat-bar summary"}
    assert {spec.name for spec in specs} == set(summary(result)) | {
        f"last_month_{key}" for key in summary(result)}
    for spec in specs:
        assert spec.aggregation == ("none" if spec.name.endswith(("return_volatility_pct", "red_high_above_ratio")) else "sum")
    metrics = extract_metrics(result.tables, specs)
    combined = aggregate_metrics({"A": metrics, "B": metrics}, specs)
    assert combined["red_high_above_count"]["value"] == 2 * summary(result)["red_high_above_count"]
    assert combined["return_volatility_pct"]["value"] is None
    assert combined["last_month_red_high_above_count"]["value"] == 2 * summary(result)["red_high_above_count"]
    assert combined["last_month_return_volatility_pct"]["value"] is None


def test_batch_runs_each_ticker_and_saves_exact_statistics(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    for ticker in ("AAA", "BBB"):
        research.pipeline.ingest_frame(DataRequest(ticker, START, START + timedelta(hours=5),
                                                  provider="databento", timeframe="30m"), sample_bars(ticker))
    store = BatchStore(tmp_path / "results")
    registry = load_registry()
    spec = LogicSpec("flat-stats", FlatBarStatisticsLogic.name, FlatBarStatisticsLogic.version, SETTINGS)
    runner = BatchRunner(research.pipeline, registry, store)
    manifest = runner.execute(BatchRequest(("AAA", "BBB"), provider="databento", timeframe="30m",
                                          as_of=START + timedelta(days=1), workers=2), [spec])
    assert manifest["counts"]["completed"] == 2
    for ticker in ("AAA", "BBB"):
        detail = BatchStore(store.directory).load_ticker(manifest["run_id"], ticker)
        result = detail["results"][spec.instance_id]
        assert result["tables"] == calculate(sample_bars(ticker)).tables
        assert detail["status"] == "completed"
        assert result["data"]["metadata"]["config"]["high_percentage"] == 102.
        assert manifest["tickers"][ticker]["summaries"][spec.instance_id]["values"]["red_high_above_count"] == 3
