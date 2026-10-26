"""Adaptive flat detection compared with exhaustive independent candidates."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json

import numpy as np
import polars as pl
import pytest

from research.analysis.flat import (
    FlatConfig, analyze, analyze_current, analyze_history, replay, select_periods,
)
from research.analysis.flat.presentation import summary_result


def bars(lengths, centers=100.):
    lengths = np.asarray(lengths, dtype=float)
    centers = np.broadcast_to(centers, lengths.shape)
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i)
                      for i in range(len(lengths))],
        "high": centers + centers * lengths / 200,
        "low": centers - centers * lengths / 200,
    })


def exhaustive(frame, config, end):
    """Check every start independently; do not assume filter monotonicity."""
    if end + 1 < config.min_duration:
        return None
    highs, lows = frame["high"].to_numpy(), frame["low"].to_numpy()
    lengths = 100 * (highs - lows) / ((highs + lows) / 2)
    tolerance = 1 + 32 * np.finfo(float).eps
    for start in range(max(0, end - config.max_duration + 1), end - config.min_duration + 2):
        high, low = max(highs[start:end + 1]), min(lows[start:end + 1])
        width = 100 * (high - low) / ((high + low) / 2)
        selected = lengths[start:end + 1]
        limit = config.max_range_pct
        if config.band_width_mode == "automatic":
            limit = config.auto_multiplier * np.percentile(selected, config.auto_percentile,
                                                          method="linear")
        maximum, median = max(selected), np.median(selected)
        if width > limit * tolerance:
            continue
        if config.long_bar_filter and maximum > config.long_bar_multiplier * median * tolerance:
            continue
        return start, low, high, width, limit, maximum, median
    return None


@pytest.mark.parametrize("mode", ["fixed", "automatic"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("seed", [8, 91, 432])
def test_every_endpoint_matches_exhaustive_candidates_and_prefixes(mode, filtered, seed):
    rng = np.random.default_rng(seed)
    frame = bars(rng.choice([0, .1, .2, .8, 1.5], 65),
                 centers=100 + np.cumsum(rng.normal(0, .18, 65)))
    config = FlatConfig(min_duration=3, max_duration=12, max_range_pct=1.3,
                        band_width_mode=mode, auto_percentile=75,
                        auto_multiplier=1.6, long_bar_filter=filtered, long_bar_multiplier=2)
    history = analyze_history(frame, config=config)
    by_end = {period.end_index: period for period in history.periods}
    for end in range(len(frame)):
        expected = exhaustive(frame, config, end)
        actual = by_end.get(end)
        assert (actual is None) == (expected is None), (seed, mode, filtered, end)
        current = analyze_current(frame.head(end + 1), config=config)
        assert current.periods == (() if actual is None else (actual,))
        if actual is not None:
            assert actual.start_index == expected[0]
            assert actual.lower_price == expected[1]
            assert actual.upper_price == expected[2]
            assert actual.range_pct == pytest.approx(expected[3])
            assert actual.max_range_pct == pytest.approx(expected[4])
            assert actual.max_bar_range_pct == (pytest.approx(expected[5]) if filtered else None)
            assert actual.median_bar_range_pct == (pytest.approx(expected[6]) if filtered else None)
    assert analyze(frame, config=config).periods == select_periods(history.periods)


def test_candidate_percentile_includes_endpoint_and_does_not_ignore_any_wicks():
    config = FlatConfig(min_duration=5, max_duration=8, band_width_mode="automatic",
                        auto_percentile=90, auto_multiplier=2)
    frame = bars([.2] * 4 + [1.0])
    period, = analyze_current(frame, config=config).periods
    assert period.max_range_pct == pytest.approx(1.36)
    assert period.range_pct == pytest.approx(1)
    assert period.start_index == 0
    # A lower percentile does not let the detector discard the largest wick.
    rejected = analyze_current(frame, config=replace(config, auto_percentile=50))
    assert rejected.status == "no_flat_period"
    assert not rejected.periods


def test_automatic_mode_needs_only_minimum_duration_and_uses_linear_percentile():
    config = FlatConfig(min_duration=2, max_duration=8, band_width_mode="automatic",
                        auto_percentile=75, auto_multiplier=2)
    frame = bars([.1, .2, .3, .4, .2, .2])
    assert analyze(frame.head(1), config=config).status == "insufficient_data"
    assert analyze(frame.head(2), config=config).status == "ok"
    period, = analyze_current(frame.head(4), config=config).periods
    assert period.max_range_pct == pytest.approx(.65)
    assert period.start_index == 0
    assert len(list(replay(frame, config=config))) == 5
    explicit = list(replay(frame, config=config, as_of_values=frame["timestamp"].to_list()[:2]))
    assert [item.status for item in explicit] == ["insufficient_data", "ok"]
    presented = summary_result(analyze(frame.head(1), config=config))
    assert "need at least 2" in presented.messages[0]
    assert "within each candidate period" in presented.messages[1]
    assert "auto_lookback" not in presented.data["metadata"]["config"]


@pytest.mark.parametrize("filtered", [False, True])
def test_automatic_width_rejection_cannot_stop_extension_to_a_longer_candidate(filtered):
    frame = bars([0, 0, 4, 4, 4, 1, 1])
    config = FlatConfig(min_duration=3, max_duration=8, band_width_mode="automatic",
                        auto_percentile=50, auto_multiplier=1.5, long_bar_filter=filtered)
    assert not analyze_current(frame.tail(3), config=config).periods
    period, = analyze_current(frame, config=config).periods
    assert period.start_index == 2
    assert period.max_range_pct == pytest.approx(6)


def test_candidate_start_changes_its_percentile_and_saved_threshold():
    frame = bars([0, 0, 0, 0, 4, 4])
    config = FlatConfig(min_duration=2, max_duration=8, band_width_mode="automatic",
                        auto_percentile=50, auto_multiplier=1)
    period, = analyze_current(frame, config=config).periods
    assert (period.start_index, period.end_index) == (3, 5)
    assert period.max_range_pct == pytest.approx(4)
    assert np.percentile([0, 0, 0, 0, 4, 4], 50) == 0


def test_completed_bars_before_candidate_do_not_change_its_limit_or_filter_statistics():
    config = FlatConfig(min_duration=3, max_duration=4, band_width_mode="automatic",
                        long_bar_filter=True)
    centers = [200.] * 25 + [100.] * 3
    first = analyze_current(bars([10.] * 25 + [.2] * 3, centers), config=config)
    second = analyze_current(bars([40.] * 25 + [.2] * 3, centers), config=config)
    assert first.periods == second.periods
    period, = first.periods
    assert (period.start_index, period.end_index) == (25, 27)
    assert period.max_range_pct == pytest.approx(.4)
    assert period.max_bar_range_pct == pytest.approx(.2)
    assert period.median_bar_range_pct == pytest.approx(.2)


def test_filter_does_not_stop_when_a_shorter_suffix_fails():
    config = FlatConfig(min_duration=3, max_duration=8, max_range_pct=10,
                        long_bar_filter=True, long_bar_multiplier=2)
    frame = bars([4, 4, 4, 1, 1])
    assert not analyze_current(frame.tail(3), config=config).periods
    period, = analyze_current(frame, config=config).periods
    assert period.start_index == 0
    assert period.median_bar_range_pct == pytest.approx(4)


def test_filter_can_restore_a_start_rejected_at_previous_endpoint():
    config = FlatConfig(min_duration=3, max_duration=8, max_range_pct=10,
                        long_bar_filter=True, long_bar_multiplier=2)
    frame = bars([4, 1, 1, 4])
    assert not analyze_current(frame.head(3), config=config).periods
    assert analyze_current(frame, config=config).periods[0].start_index == 0


def test_filter_limit_is_inclusive_customizable_and_allows_small_candles():
    config = FlatConfig(min_duration=5, max_duration=8, max_range_pct=10,
                        long_bar_filter=True, long_bar_multiplier=2)
    assert analyze_current(bars([1, 1, 1, 1, 2]), config=config).periods
    assert not analyze_current(bars([1, 1, 1, 1, 2.01]), config=config).periods
    assert analyze_current(bars([1, 1, 1, 1, 2.01]),
                           config=replace(config, long_bar_multiplier=3)).periods
    assert analyze_current(bars([.01, 1, 1, 1, 1]), config=config).periods


@pytest.mark.parametrize("mode", ["fixed", "automatic"])
def test_zero_length_medians_and_zero_automatic_threshold(mode):
    config = FlatConfig(min_duration=3, max_duration=8, band_width_mode=mode,
                        long_bar_filter=True)
    period, = analyze_current(bars([0] * 5), config=config).periods
    assert period.max_bar_range_pct == period.median_bar_range_pct == 0
    if mode == "automatic":
        assert period.max_range_pct == 0
    assert not analyze_current(bars([0, 0, 0, 0, .1]), config=config).periods
    json.dumps(summary_result(analyze(bars([0] * 5), config=config)).to_dict(), allow_nan=False)


def test_automatic_history_is_causal_and_scale_invariant():
    rng = np.random.default_rng(45)
    frame = bars(rng.uniform(.2, 1., 80), 100 + np.cumsum(rng.normal(0, .2, 80)))
    config = FlatConfig(min_duration=3, max_duration=15, band_width_mode="automatic",
                        long_bar_filter=True)
    history = analyze_history(frame, config=config)
    earlier = analyze_history(frame.head(40), config=config)
    assert earlier.periods == tuple(period for period in history.periods if period.end_index < 40)
    assert earlier == analyze_history(frame, config=config, as_of=frame["timestamp"][39])
    scaled = frame.with_columns(pl.col("high") * 1e10, pl.col("low") * 1e10)
    rescaled = analyze_history(scaled, config=config)
    assert [(p.start_index, p.end_index) for p in history.periods] == [
        (p.start_index, p.end_index) for p in rescaled.periods]
    for original, changed in zip(history.periods, rescaled.periods):
        assert original.max_range_pct == pytest.approx(changed.max_range_pct)
        assert original.max_bar_range_pct == pytest.approx(changed.max_bar_range_pct)
        assert original.median_bar_range_pct == pytest.approx(changed.median_bar_range_pct)


def test_overflowing_automatic_threshold_fails_instead_of_publishing_infinity():
    config = FlatConfig(min_duration=2, max_duration=8, band_width_mode="automatic",
                        auto_multiplier=1e308)
    with pytest.raises(ValueError, match="reduce auto_multiplier"):
        analyze(bars([2.] * 5), config=config)


@pytest.mark.parametrize("filtered", [False, True])
def test_automatic_progress_finishes_after_all_eligible_endpoints(filtered):
    config = FlatConfig(min_duration=2, max_duration=8, band_width_mode="automatic",
                        long_bar_filter=filtered)
    updates = []
    result = analyze(bars([.2] * 5), config=config,
                     progress=lambda done, total: updates.append((done, total)))
    assert result.status == "ok"
    assert updates == [(done, 4) for done in range(5)]


@pytest.mark.parametrize("values", [
    {"band_width_mode": "auto"}, {"band_width_mode": None}, {"band_width_mode": 1},
    {"auto_percentile": -1}, {"auto_percentile": 101}, {"auto_percentile": True},
    {"auto_percentile": float("nan")}, {"auto_multiplier": 0}, {"auto_multiplier": -1},
    {"auto_multiplier": float("inf")}, {"auto_multiplier": "2"},
    {"auto_multiplier": 10 ** 400}, {"auto_lookback": 100},
    {"long_bar_filter": 1}, {"long_bar_filter": "false"},
    {"long_bar_multiplier": .9}, {"long_bar_multiplier": float("nan")},
    {"long_bar_multiplier": True}, {"long_bar_multiplier": 10 ** 400},
])
def test_new_settings_validate_even_when_their_mode_is_disabled(values):
    with pytest.raises(ValueError):
        FlatConfig.from_dict(values)
