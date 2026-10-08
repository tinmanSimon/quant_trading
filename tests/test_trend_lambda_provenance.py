"""Penalty provenance describes actual discovery and never drops overlaps."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import combinations

import numpy as np
import polars as pl
import pytest

from trend_analysis import TrendConfig, analyze_current
from trend_analysis.scoring import volatility
from trend_analysis.segmentation import LinearCosts, candidate_provenance, candidate_starts
from trend_analysis.tails import TailInfo


def prices(values):
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
                      for i in range(len(values))],
        "close": 100 * np.exp(values),
    })


def test_each_reported_lambda_selects_the_boundary_in_exhaustive_partitions():
    values = np.random.default_rng(192).normal(size=24)
    config = TrendConfig(penalty_multiplier=1.7, penalty_grid_step=.5)
    sigma = volatility(values)
    partitions = []
    for boundary_count in range(3):
        for internal in combinations(range(8, 17), boundary_count):
            edges = (0, *internal, len(values))
            if min(np.diff(edges)) < 8:
                continue
            error = 0.0
            for start, end in zip(edges, edges[1:]):
                x = np.arange(end - start)
                fitted = np.polyval(np.polyfit(x, values[start:end], 1), x)
                error += float(np.square(values[start:end] - fitted).sum())
            partitions.append((error / sigma ** 2, boundary_count, edges[-2]))
    full = next(error for error, count, _ in partitions if count == 0)
    penalty = config.penalty_multiplier * np.log(len(values)) * 2 ** -4
    expected = {}
    level = 0
    while True:
        winner = min((error + penalty * count, start) for error, count, start in partitions)[1]
        expected.setdefault(winner, []).append((level, penalty))
        if penalty > full:
            break
        penalty *= 2 ** config.penalty_grid_step
        level += 1
    actual = candidate_provenance(LinearCosts(values), sigma, config)
    assert actual == {start: tuple(entries) for start, entries in expected.items()}
    assert candidate_starts(LinearCosts(values), sigma, config) == sorted(actual)


def test_all_selecting_lambdas_are_preserved_once_per_interval_in_results_and_tables():
    frame = prices(np.arange(100) ** 2 * .0001)
    config = TrendConfig()
    values = np.log(frame["close"].to_numpy())
    provenance = candidate_provenance(LinearCosts(values), volatility(values), config)
    result = analyze_current(frame, config=config)
    assert any(len(entries) > 1 for entries in provenance.values())
    assert len({t.trend_id for t in result.trends}) == len(result.trends)
    for record in (*result.trends, *result.diagnostics):
        entries = provenance.get(record.start_index, ())
        assert record.lambda_levels == tuple(level for level, _ in entries)
        assert record.lambda_values == tuple(value for _, value in entries)
        assert record.origin == ("segmentation" if entries else "tail")
    for table, records in ((result.to_frame(), result.trends),
                           (result.diagnostics_frame(), result.diagnostics)):
        assert table.schema["lambda_values"] == pl.List(pl.Float64)
        assert table.schema["lambda_levels"] == pl.List(pl.Int64)
        assert table["lambda_values"].to_list() == [list(r.lambda_values) for r in records]
        assert table["lambda_levels"].to_list() == [list(r.lambda_levels) for r in records]
        assert table["origin"].to_list() == [r.origin for r in records]


def test_tail_only_candidate_has_no_invented_lambda_and_both_nearby_intervals_survive():
    path = np.r_[np.arange(134) * -.003, -.399 + np.arange(1, 14) * .006]
    result = analyze_current(prices(path))
    segmented = next(t for t in result.trends if t.start_index == 133)
    tail = next(t for t in result.trends if t.start_index == 134)
    assert segmented.direction == tail.direction == 1
    assert segmented.end_index == tail.end_index == 146
    assert segmented.origin == "segmentation" and segmented.lambda_values
    assert tail.origin == "tail"
    assert tail.lambda_values == tail.lambda_levels == ()
    assert all(d.accepted and not d.rejection_reason for d in result.diagnostics
               if d.start_index in (133, 134))


def test_tail_promotion_preserves_an_existing_segmentation_candidates_lambdas(monkeypatch):
    import trend_analysis.api as api

    frame = prices(np.arange(100) ** 2 * .0001)
    expected = next(t for t in analyze_current(frame).trends if t.start_index == 50)
    monkeypatch.setattr(api, "diagnose", lambda start, scorer:
                        TailInfo(promoted_start=50) if start == 0 else TailInfo())
    result = analyze_current(frame)
    actual = next(t for t in result.trends if t.start_index == 50)
    diagnostic = next(d for d in result.diagnostics if d.start_index == 50)
    assert actual.origin == diagnostic.origin == "segmentation"
    assert actual.lambda_values == diagnostic.lambda_values == expected.lambda_values
    assert actual.lambda_levels == diagnostic.lambda_levels == expected.lambda_levels


def test_nearby_same_direction_candidates_are_all_retained_regardless_of_display_threshold():
    frame = prices(np.arange(100) ** 2 * .0001)
    config = TrendConfig()
    result = analyze_current(frame, config=config)
    starts = {trend.start_index for trend in result.trends}
    assert {83, 85, 87, 88, 90, 91, 92} <= starts
    assert all(d.accepted for d in result.diagnostics)
    assert all(not d.rejection_reason for d in result.diagnostics)
    # Legacy duration-ratio suppression and the new display-only threshold
    # must not change which intervals or metric values the engine returns.
    changed = analyze_current(frame, config=replace(config, dedup_ratio=100,
                                                   redundancy_percentage=100,
                                                   redundancy_boundary_rule="or"))
    assert changed.trends == result.trends
    assert changed.diagnostics == result.diagnostics


def test_empty_tables_keep_typed_provenance_columns():
    result = analyze_current(prices(np.zeros(7)))
    for table in (result.to_frame(), result.diagnostics_frame()):
        assert table.schema["lambda_values"] == pl.List(pl.Float64)
        assert table.schema["lambda_levels"] == pl.List(pl.Int64)
        assert table.schema["origin"] == pl.String


def test_old_configuration_uses_percentage_defaults_and_round_trips_without_mutating_input():
    values = {"clarity_threshold": .7, "dedup_ratio": 1.5, "redundancy_threshold_bars": 12}
    config = TrendConfig.from_dict(values)
    assert values["redundancy_threshold_bars"] == 12
    assert config.redundancy_percentage == 5.0
    assert config.redundancy_boundary_rule == "and"
    assert TrendConfig.from_dict(config.to_dict()) == config
    assert TrendConfig(redundancy_percentage=0).redundancy_percentage == 0
    assert TrendConfig(redundancy_percentage=.5, redundancy_boundary_rule="or").redundancy_percentage == .5


@pytest.mark.parametrize("invalid", [-1, 100.1, True, float("nan"), float("inf"), "5"])
def test_redundancy_percentage_requires_finite_number_in_range(invalid):
    with pytest.raises(ValueError, match="redundancy_percentage"):
        TrendConfig(redundancy_percentage=invalid)


@pytest.mark.parametrize("invalid", ["xor", "AND", "", True, None, []])
def test_redundancy_rule_requires_and_or_or(invalid):
    with pytest.raises(ValueError, match="redundancy_boundary_rule"):
        TrendConfig(redundancy_boundary_rule=invalid)
