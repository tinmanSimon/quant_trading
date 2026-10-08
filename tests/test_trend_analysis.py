"""Causal behavior, exact estimators, and meaningful controlled price paths."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from itertools import combinations
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from trend_analysis import ALGORITHM_VERSION, TrendConfig, analyze_current as analyze, replay
from trend_analysis.scoring import Scorer, prefix_robust_slopes, volatility
from trend_analysis.segmentation import LinearCosts, candidate_starts
from trend_analysis.tails import diagnose


def prices(log_path):
    return pl.DataFrame({
        "timestamp": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(len(log_path))],
        "symbol": ["TEST"] * len(log_path), "close": 100 * np.exp(log_path),
    })


def test_config_is_immutable_serializable_and_missing_fields_use_defaults():
    config = TrendConfig.from_dict({"clarity_threshold": 0.72})
    assert config.max_history == 2000
    assert TrendConfig.from_dict(config.to_dict()) == config
    with pytest.raises(FrozenInstanceError):
        config.clarity_threshold = .3


@pytest.mark.parametrize("changes", [
    {"weight_ols": .5}, {"weight_robust": -.1}, {"clarity_threshold": 1.2},
    {"boundary_acceptance": -.1}, {"min_segment_length": 7}, {"max_history": 2001},
    {"min_segment_length": 8.0}, {"max_history": True}, {"magnitude_threshold": 0},
    {"max_tail_fraction": .6}, {"penalty_grid_step": .01}, {"penalty_multiplier": float("nan")},
    {"tail_fit_threshold": "0.3"}, {"min_tail_length": 2001}, {"unknown": 1},
])
def test_invalid_configs_fail_clearly(changes):
    with pytest.raises(ValueError):
        TrendConfig.from_dict(changes)


@pytest.mark.parametrize("direction", [-1, 1])
def test_steady_log_trend_has_full_interval_and_consistent_fit(direction):
    frame = prices(np.arange(147) * .002 * direction)
    original = frame.clone()
    result = analyze(frame)
    assert result.status == "ok"
    assert len(result.trends) == 1
    trend = result.trends[0]
    assert (trend.start_index, trend.end_index, trend.duration, trend.direction) == (0, 146, 147, direction)
    assert trend.clarity == pytest.approx(1)
    assert trend.slope == pytest.approx(.002 * direction)
    assert trend.rate_per_bar == pytest.approx(np.expm1(trend.slope))
    fit = np.exp(trend.intercept + trend.slope * np.arange(trend.duration))
    np.testing.assert_allclose(fit, frame["close"].to_numpy())
    assert trend.tail_status == "No tail change detected"
    assert trend.onset_truncated
    assert result.algorithm_version == ALGORITHM_VERSION
    assert result.to_frame().height == 1
    assert result.diagnostics_frame()["accepted"].sum() == 1
    assert_frame_equal(frame, original)


def test_flat_and_jump_then_flat_are_not_reported_as_ongoing_trends():
    for path in (np.zeros(100), np.r_[np.zeros(50), np.full(50, .3)]):
        result = analyze(prices(path))
        assert result.status == "no_clear_trend"
        assert not result.trends
        assert result.to_frame().is_empty()
        assert result.diagnostics_frame()["rejection_reason"].str.contains("insufficient").any()


def test_broad_decline_and_short_rebound_remain_distinct():
    path = np.r_[np.arange(134) * -.003, -.399 + np.arange(1, 14) * .006]
    result = analyze(prices(path))
    broad = next(t for t in result.trends if t.duration == 147)
    local = next(t for t in result.trends if t.direction == 1)
    assert broad.direction == -1
    assert 13 <= local.duration <= 14
    assert local.parent_id == broad.trend_id
    assert broad.tail_status == "Contains a local countertrend"
    assert broad.tail_start_time == local.start_time
    assert broad.clarity > .8


def test_flat_tail_and_young_shock_do_not_hide_behind_high_clarity():
    paused = analyze(prices(np.r_[np.arange(80) * .01, np.full(20, .79)]))
    assert paused.trends[0].clarity > .9
    assert "slowed or paused" in paused.trends[0].tail_status
    shocked = analyze(prices(np.r_[np.arange(100) * .01, .79, .59]))
    assert shocked.trends[0].direction == 1
    assert shocked.trends[0].tail_status == "Abrupt counter-move; insufficient observations"


def test_level_change_is_detected_even_when_both_robust_slopes_are_flat():
    # Exercise tail diagnosis directly: whole-interval robust slope is nonzero
    # for this split, while the two pieces are flat. No slope gate may hide it.
    values = np.r_[np.zeros(55), np.full(45, .3)]
    config = TrendConfig()
    scorer = Scorer(values, LinearCosts(values), config)
    info = diagnose(0, scorer)
    assert "Level shift" in info.status
    assert info.start == 55
    assert info.slope_change == pytest.approx(0)
    assert info.level_change > config.tail_level_threshold


def test_as_of_replay_is_unchanged_by_future_prices_and_instruments():
    frame = prices(np.arange(80) * .002)
    cutoff = frame["timestamp"][39]
    expected = analyze(frame.head(40))
    assert analyze(frame, as_of=cutoff) == expected
    altered = frame.with_columns(
        pl.when(pl.col("timestamp") > cutoff).then(float("nan")).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("timestamp") > cutoff).then(pl.lit("FUTURE")).otherwise(pl.col("symbol")).alias("symbol"),
    )
    assert analyze(altered, as_of=cutoff) == expected
    snapshots = list(replay(frame, as_of_values=[cutoff, frame["timestamp"][-1]]))
    assert snapshots == [expected, analyze(frame)]


@pytest.mark.parametrize("factor", [1e-4, 1000])
def test_price_unit_invariance_including_symmetric_segmentation_ties(factor):
    frame = prices(np.arange(100) ** 2 * .0001)
    before = analyze(frame)
    after = analyze(frame.with_columns(pl.col("close") * factor))
    assert [(t.start_index, t.direction, t.tail_status) for t in before.trends] == [
        (t.start_index, t.direction, t.tail_status) for t in after.trends]
    for first, second in zip(before.trends, after.trends):
        assert first.clarity == pytest.approx(second.clarity)
        assert first.slope == pytest.approx(second.slope)
        assert first.intercept + np.log(factor) == pytest.approx(second.intercept)


def test_history_cap_preserves_global_ordinals_and_onset_flag():
    frame = prices(np.arange(100) * .001)
    result = analyze(frame, config=TrendConfig(max_history=32))
    assert result.source_offset == 68
    assert result.analysis_start == frame["timestamp"][68]
    assert (result.trends[0].start_index, result.trends[0].end_index) == (68, 99)
    assert result.trends[0].onset_truncated
    earlier = analyze(frame, as_of=frame["timestamp"][79], config=TrendConfig(max_history=32))
    assert (earlier.source_offset, earlier.trends[0].end_index) == (48, 79)


@pytest.mark.parametrize("change", [
    lambda df: df.reverse(),
    lambda df: pl.concat([df, df.tail(1)]),
    lambda df: df.with_columns(pl.lit(-1).alias("close")),
    lambda df: df.with_columns(pl.lit(float("inf")).alias("close")),
    lambda df: df.with_columns(pl.lit(None, dtype=pl.Float64).alias("close")),
    lambda df: df.with_columns(pl.Series("symbol", ["A"] * 19 + ["B"])),
    lambda df: df.with_columns(pl.col("timestamp").cast(pl.String)),
])
def test_invalid_inputs_have_explicit_errors(change):
    with pytest.raises(ValueError):
        analyze(change(prices(np.arange(20) * .01)))


def test_empty_and_short_data_have_a_normal_insufficient_data_result():
    frame = prices(np.arange(7) * .01)
    for short in (frame, frame.head(0)):
        result = analyze(short)
        assert result.status == "insufficient_data"
        assert result.to_frame().is_empty()
        assert result.diagnostics_frame().is_empty()


def test_exact_all_prefix_medians_match_brute_force_with_ties():
    rng = np.random.default_rng(812)
    for values in (rng.normal(size=35), np.repeat([1., 3., 2.], 10)):
        slopes = prefix_robust_slopes(values)
        for end in range(2, len(values) + 1):
            expected = np.median([(values[j] - values[i]) / (j - i)
                                  for i, j in combinations(range(end), 2)])
            assert slopes[end] == pytest.approx(expected, abs=1e-14)


def test_all_boundary_segmentation_matches_exhaustive_partition_costs():
    rng = np.random.default_rng(192)
    values = rng.normal(size=24)
    costs = LinearCosts(values)
    config = TrendConfig()
    sigma = volatility(values)
    full = float(costs.fit(0, len(values))[0]) / sigma ** 2
    penalty = 2. ** -4 * np.log(len(values))
    expected = set()
    while True:
        choices = []
        for boundary_count in range(3):
            for internal in combinations(range(8, len(values) - 7), boundary_count):
                edges = (0, *internal, len(values))
                if min(np.diff(edges)) < 8:
                    continue
                # Use independent least squares, not the implementation's cost.
                error = 0.0
                for start, end in zip(edges, edges[1:]):
                    x = np.arange(end - start)
                    coefficients = np.polyfit(x, values[start:end], 1)
                    error += float(np.square(values[start:end] - np.polyval(coefficients, x)).sum())
                choices.append((error / sigma ** 2 + penalty * boundary_count, edges[-2]))
        expected.add(min(choices)[1])
        if penalty > full:
            break
        penalty *= 2
    assert candidate_starts(costs, sigma, config) == sorted(expected)


def test_maximum_history_analysis_runs_exactly_within_interactive_budget():
    rng = np.random.default_rng(741)
    frame = prices(np.cumsum(rng.normal(.001, .01, 2000)))
    began = perf_counter()
    result = analyze(frame)
    elapsed = perf_counter() - began
    assert result.analysis_start == frame["timestamp"][0]
    assert result.analysis_end == frame["timestamp"][-1]
    assert len(result.diagnostics) > 1
    # Generous regression guard; the reference local run is about 1-2 seconds.
    # No pair sampling, boundary subsampling, or approximate medians are used.
    assert elapsed < 20, f"2000-bar exact analysis took {elapsed:.2f}s"


def test_extreme_penalty_settings_fail_before_allocating_unbounded_work():
    frame = prices(np.arange(40) ** 2 * .001)
    with pytest.raises(ValueError, match="256 levels"):
        analyze(frame, config=TrendConfig(penalty_multiplier=1e-300, penalty_grid_step=.25))
    with pytest.raises(ValueError, match="numerically supported"):
        analyze(frame, config=TrendConfig(penalty_multiplier=1e308))


def test_zero_boundary_fraction_disables_perturbation():
    values = np.r_[np.full(20, 10.), np.arange(8) * .01]
    config = TrendConfig(boundary_fraction=0)
    scorer = Scorer(values, LinearCosts(values), config)
    # A genuinely rising minimum-length suffix supports itself, and no
    # available pre-onset observation is required when perturbations are off.
    assert scorer.boundary(20) == 1
    with_perturbation = Scorer(values, LinearCosts(values), TrendConfig())
    assert with_perturbation.boundary(20) < config.boundary_acceptance
