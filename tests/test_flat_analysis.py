"""Causal horizontal candle envelopes and the disjoint final summary."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from research.analysis.flat import (
    ALGORITHM_VERSION, FlatConfig, analyze, analyze_current, analyze_history,
    analyze_summary, replay, select_periods,
)


def candles(values, *, width=.1):
    values = np.asarray(values, dtype=float)
    return pl.DataFrame({
        "timestamp": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * i)
                      for i in range(len(values))],
        "symbol": ["TEST"] * len(values), "open": values, "close": values,
        "high": values + np.asarray(width) / 2, "low": values - np.asarray(width) / 2,
    }, schema_overrides={"timestamp": pl.Datetime("us", "UTC"), "symbol": pl.String})


def brute_current(frame, config):
    """Independent backwards extension, stopping at the first invalid candle."""
    last = None
    high, low = -float("inf"), float("inf")
    for start in range(len(frame) - 1, max(-1, len(frame) - config.max_duration - 1), -1):
        high = max(high, frame["high"][start])
        low = min(low, frame["low"][start])
        center = (high + low) / 2
        width_pct = (high - low) / center * 100
        if width_pct > config.max_range_pct * (1 + 32 * np.finfo(float).eps):
            break
        if len(frame) - start >= config.min_duration:
            last = start, low, high, center, width_pct
    return last


def test_settings_are_immutable_serializable_and_use_documented_defaults():
    config = FlatConfig()
    assert config.to_dict() == {
        "min_duration": 20, "max_duration": 200, "max_range_pct": 2.0,
        "band_width_mode": "fixed", "auto_percentile": 90.0, "auto_multiplier": 2.0,
        "long_bar_filter": False, "long_bar_multiplier": 2.0,
    }
    assert FlatConfig.from_dict(config.to_dict()) == config
    assert FlatConfig.from_dict({"min_duration": 3}).max_range_pct == 2
    assert ALGORITHM_VERSION == "2.2.0"
    with pytest.raises(FrozenInstanceError):
        config.min_duration = 10


@pytest.mark.parametrize("values", [
    {"min_duration": 1}, {"min_duration": True}, {"min_duration": 3.0},
    {"max_duration": 20}, {"max_duration": 19}, {"max_duration": True},
    {"max_duration": 30.0}, {"max_duration": float("inf")},
    {"max_range_pct": 0}, {"max_range_pct": -1}, {"max_range_pct": 200},
    {"max_range_pct": 201}, {"max_range_pct": float("inf")},
    {"max_range_pct": float("nan")}, {"max_range_pct": True},
    {"max_range_pct": "2"}, {"max_range_pct": 10 ** 400},
    {"slope_min_pct": -.02}, {"extra": 1}, None, [],
])
def test_invalid_settings_fail(values):
    with pytest.raises(ValueError):
        FlatConfig.from_dict(values)


def test_every_endpoint_matches_backwards_extension_until_first_failure():
    rng = np.random.default_rng(901)
    frame = candles(100 + np.cumsum(rng.normal(0, .6, 173)), width=rng.uniform(0, 1, 173))
    config = FlatConfig(min_duration=4, max_duration=17, max_range_pct=2)
    history = analyze_history(frame, config=config)
    by_end = {period.end_index: period for period in history.periods}
    assert len(by_end) == len(history.periods)
    for end in range(config.min_duration, len(frame) + 1):
        expected = brute_current(frame.head(end), config)
        actual = by_end.get(end - 1)
        assert (actual is None) == (expected is None)
        assert analyze_current(frame.head(end), config=config).periods == ((actual,) if actual else ())
        if actual is not None:
            assert actual.start_index == expected[0]
            assert actual.end_index == end - 1
            assert actual.lower_price == expected[1]
            assert actual.upper_price == expected[2]
            assert actual.center_price == pytest.approx(expected[3])
            assert actual.range_pct == pytest.approx(expected[4])


def test_total_band_width_is_percentage_of_midpoint_and_inclusive():
    frame = candles(np.full(25, 100.), width=2)
    original = frame.clone()
    result = analyze_current(frame)
    assert result.status == "ok"
    period, = result.periods
    assert (period.start_index, period.end_index, period.duration) == (0, 24, 25)
    assert (period.lower_price, period.upper_price, period.center_price, period.range_pct) == (99, 101, 100, 2)
    assert_frame_equal(frame, original)
    assert not analyze_current(candles(np.full(25, 100.), width=2.00000001)).periods


def test_entire_high_low_envelope_matters_even_with_exactly_flat_closes():
    frame = candles(np.full(30, 100.), width=12)
    assert frame["close"].n_unique() == 1
    assert analyze_current(frame).status == "no_flat_period"
    assert not analyze_history(frame).periods
    assert not analyze_summary(frame).periods


def test_candle_heights_need_not_be_uniform_when_all_wicks_fit():
    frame = candles(np.full(20, 100.), width=np.linspace(.01, 1.9, 20))
    period, = analyze_current(frame).periods
    assert period.duration == 20
    assert period.range_pct == pytest.approx(1.9)


def test_returning_to_the_start_price_does_not_cancel_a_large_excursion():
    config = FlatConfig(min_duration=3, max_duration=8)
    frame = candles([100, 110, 120, 110, 100])
    assert frame["close"][0] == frame["close"][-1]
    assert not analyze_current(frame, config=config).periods
    assert not analyze_history(frame, config=config).periods


def test_breakout_ends_old_period_and_recovery_requires_enough_new_candles():
    config = FlatConfig(min_duration=3, max_duration=8)
    frame = candles([100] * 5 + [110] + [100] * 4)
    by_end = {period.end_index: period for period in analyze_history(frame, config=config).periods}
    assert (by_end[4].start_index, by_end[4].end_index) == (0, 4)
    assert all(end not in by_end for end in (5, 6, 7))
    assert (by_end[8].start_index, by_end[8].end_index) == (6, 8)
    assert (by_end[9].start_index, by_end[9].end_index) == (6, 9)
    assert [(p.start_index, p.end_index) for p in analyze(frame, config=config).periods] == [(6, 9), (0, 4)]


def test_single_wide_candle_is_rejected_and_does_not_break_later_windows():
    frame = candles(np.full(8, 100.), width=[.1, .1, .1, 20, .1, .1, .1, .1])
    config = FlatConfig(min_duration=3, max_duration=7)
    assert not analyze_current(frame.head(4), config=config).periods
    assert [(p.start_index, p.end_index) for p in analyze(frame, config=config).periods] == [(4, 7), (0, 2)]


def test_minimum_and_maximum_duration_are_inclusive_and_summary_never_overlaps():
    config = FlatConfig(min_duration=3, max_duration=7)
    frame = candles(np.full(23, 100.))
    for short in (frame.head(0), frame.head(2)):
        result = analyze(short, config=config)
        assert result.status == "insufficient_data"
        assert not result.periods
    assert analyze_current(frame.head(3), config=config).periods[0].duration == 3
    current, = analyze_current(frame, config=config).periods
    assert (current.start_index, current.end_index, current.duration) == (16, 22, 7)
    history = analyze_history(frame, config=config)
    assert [period.duration for period in history.periods] == [3, 4, 5, 6, 7] + [7] * 16
    summary = analyze(frame, config=config)
    assert [(p.start_index, p.end_index) for p in summary.periods] == [(16, 22), (9, 15), (2, 8)]
    assert summary.periods == select_periods(history.periods)


def test_summary_uses_latest_earlier_period_if_current_is_absent_and_keeps_gaps():
    frame = candles([100] * 6 + [105, 110, 115] + [120] * 7 + [125, 130])
    config = FlatConfig(min_duration=4, max_duration=50)
    result = analyze(frame, config=config)
    assert result.analysis_mode == "summary"
    assert not analyze_current(frame, config=config).periods
    assert [(p.start_index, p.end_index) for p in result.periods] == [(9, 15), (0, 5)]
    assert result.periods == select_periods(analyze_history(frame, config=config).periods)
    for newer, older in zip(result.periods, result.periods[1:]):
        assert older.end_index < newer.start_index


def test_random_summary_matches_history_selection_and_progress_finishes():
    rng = np.random.default_rng(38)
    config = FlatConfig(min_duration=5, max_duration=17)
    for frame in (candles(100 + np.cumsum(rng.normal(0, .6, 97))), candles(100 + np.arange(40) * 10)):
        for function in (analyze_history, analyze_summary):
            updates = []
            result = function(frame, config=config, progress=lambda a, b: updates.append((a, b)))
            assert updates == [(done, len(frame) - 4) for done in range(len(frame) - 3)]
            if function is analyze_summary:
                assert result.periods == select_periods(analyze_history(frame, config=config).periods)
    updates = []
    analyze_summary(candles([]), config=config, progress=lambda a, b: updates.append((a, b)))
    assert updates == [(0, 0)]


def test_selection_prefers_longest_at_the_latest_endpoint():
    history = analyze_history(candles(np.full(10, 100.)), config=FlatConfig(min_duration=3))
    longest = history.periods[-1]
    shorter = replace(longest, start_index=4, duration=6, period_id="shorter")
    assert select_periods([shorter, *reversed(history.periods)]) == (longest,)


def test_as_of_checks_prices_and_series_identity_only_through_cutoff():
    frame = candles([100] * 20 + [110] * 20)
    cutoff = frame["timestamp"][19]
    future_bad = frame.with_columns([
        pl.when(pl.col("timestamp") > cutoff).then(float("nan")).otherwise(pl.col(name)).alias(name)
        for name in ("high", "low", "open", "close")
    ]).with_columns(
        pl.when(pl.col("timestamp") > cutoff).then(pl.lit("FUTURE")).otherwise(pl.col("symbol")).alias("symbol"),
    )
    for function in (analyze_current, analyze_history, analyze_summary):
        assert function(future_bad, as_of=cutoff) == function(frame.head(20))
    snapshots = list(replay(frame, as_of_values=[cutoff, frame["timestamp"][-1]]))
    assert snapshots == [analyze_current(frame.head(20)), analyze_current(frame)]


def test_history_records_do_not_change_when_future_candles_are_appended():
    rng = np.random.default_rng(19)
    frame = candles(100 + np.cumsum(rng.normal(0, .2, 90)))
    config = FlatConfig(min_duration=5, max_duration=30)
    earlier = analyze_history(frame.head(45), config=config)
    later = analyze_history(frame, config=config)
    assert earlier.periods == tuple(period for period in later.periods if period.end_index < 45)


@pytest.mark.parametrize("factor", [1e-12, 1e12])
def test_price_scaling_preserves_boundaries_and_relative_width(factor):
    rng = np.random.default_rng(12)
    frame = candles(100 + np.cumsum(rng.normal(0, .3, 77)))
    config = FlatConfig(min_duration=4, max_duration=20)
    before = analyze_history(frame, config=config)
    scaled = frame.with_columns([pl.col(name) * factor for name in ("open", "high", "low", "close")])
    after = analyze_history(scaled, config=config)
    assert [(p.start_index, p.end_index) for p in before.periods] == [
        (p.start_index, p.end_index) for p in after.periods]
    for first, second in zip(before.periods, after.periods):
        assert first.range_pct == pytest.approx(second.range_pct)
        assert first.lower_price * factor == pytest.approx(second.lower_price)
        assert first.upper_price * factor == pytest.approx(second.upper_price)


def test_large_finite_prices_do_not_overflow_and_tiny_threshold_has_no_absolute_tolerance():
    frame = candles(np.full(20, 1e307), width=2e305)
    period, = analyze_current(frame).periods
    assert np.isfinite(period.center_price)
    assert period.range_pct == pytest.approx(2)
    tiny = candles([100, 100], width=0).with_columns(pl.lit(np.nextafter(100., np.inf)).alias("high"))
    config = FlatConfig(min_duration=2, max_duration=3, max_range_pct=1e-18)
    assert not analyze_current(tiny, config=config).periods


@pytest.mark.parametrize("change", [
    lambda df: df.reverse(),
    lambda df: pl.concat([df, df.tail(1)]),
    lambda df: df.with_columns(pl.lit(-1).alias("low")),
    lambda df: df.with_columns(pl.lit(float("inf")).alias("high")),
    lambda df: df.with_columns(pl.lit(None, dtype=pl.Float64).alias("high")),
    lambda df: df.with_columns(pl.lit(101).alias("low")),
    lambda df: df.with_columns(pl.lit(102).alias("open")),
    lambda df: df.with_columns(pl.lit(98).alias("close")),
    lambda df: df.with_columns(pl.lit(float("nan")).alias("close")),
    lambda df: df.with_columns(pl.Series("symbol", ["A"] * 19 + ["B"])),
    lambda df: df.with_columns(pl.col("timestamp").cast(pl.String)),
    lambda df: df.drop("high"),
    lambda df: df.drop("low"),
])
def test_invalid_inputs_fail_clearly(change):
    with pytest.raises(ValueError):
        analyze(change(candles(np.full(20, 100.))))


def test_wrong_configuration_and_cutoff_types_fail_clearly():
    frame = candles(np.full(20, 100.))
    for function in (analyze_current, analyze_history, analyze_summary):
        with pytest.raises(ValueError, match="FlatConfig"):
            function(frame, config={})
        with pytest.raises(ValueError, match="as_of"):
            function(frame, as_of="2020-01-01")
        with pytest.raises(ValueError, match="timezone"):
            function(frame, as_of=datetime(2020, 1, 1))
    for step in (0, True, 1.5):
        with pytest.raises(ValueError, match="step"):
            list(replay(frame, step=step))


def test_high_low_only_custom_names_and_date_inputs_have_stable_table_types():
    frame = candles(np.full(20, 100.))
    full = analyze(frame)
    empty = analyze(frame.head(4))
    assert full.to_frame().schema == empty.to_frame().schema
    assert full.to_frame().columns == [
        "period_id", "start_index", "end_index", "start", "end", "bars",
        "lower_price", "upper_price", "center_price", "range_pct",
        "max_range_pct", "max_bar_range_pct", "median_bar_range_pct",
    ]
    daily = pl.DataFrame({"day": [date(2020, 1, 1) + timedelta(days=i) for i in range(20)],
                          "upper": np.full(20, 101.), "lower": np.full(20, 99.)})
    result = analyze(daily, timestamp_col="day", high_col="upper", low_col="lower")
    assert result.to_frame().schema["start"] == pl.Datetime("us")
    assert result.periods[0].start_time == date(2020, 1, 1)
    assert len(list(replay(frame, step=2, config=FlatConfig(min_duration=4)))) == 9


def test_long_history_uses_all_endpoints_with_global_indices():
    frame = candles(np.full(26208, 100.))
    result = analyze(frame)
    assert result.source_count == 26208
    assert result.analysis_start == frame["timestamp"][0]
    assert result.analysis_end == frame["timestamp"][-1]
    assert result.periods[0].end_index == 26207
    assert result.periods[0].duration == 200
    assert len(result.periods) == 131
