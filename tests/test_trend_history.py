"""Historical records must equal analyses made at their own endpoint."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from trend_analysis import TrendConfig, analyze, analyze_current, analyze_history, replay


def prices(values):
    return pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
                                       for i in range(len(values))],
                         "symbol": ["TEST"] * len(values), "close": 100 * np.exp(values)})


@pytest.mark.parametrize("lookback", [2000, 24, 8])
def test_all_endpoints_equal_independent_prefix_analyses_including_rolling_context(lookback):
    rng = np.random.default_rng(42)
    frame = prices(np.cumsum(rng.normal(.002, .003, 60)))
    config = TrendConfig(max_history=lookback)
    history = analyze(frame, config=config)
    assert history.analysis_mode == "history"
    assert history.source_offset == 0
    assert history.analysis_start == frame["timestamp"][0]
    assert history.analysis_end == frame["timestamp"][-1]
    for end in range(config.min_segment_length - 1, frame.height):
        snapshot = analyze_current(frame.head(end + 1), config=config)
        actual = tuple(trend for trend in history.trends if trend.end_index == end)
        assert actual == snapshot.trends
        assert tuple(d for d in history.diagnostics if d.end_index == end) == snapshot.diagnostics
    assert len({trend.trend_id for trend in history.trends}) == len(history.trends)
    assert len({trend.end_index for trend in history.trends}) > 1
    assert all(trend.duration <= lookback for trend in history.trends)
    table = history.to_frame()
    assert {"start_index", "end_index"} <= set(table.columns)


def test_appending_or_changing_future_prices_cannot_revise_earlier_records():
    old = prices(np.arange(60) * .004)
    extended = prices(np.r_[np.arange(60) * .004, .236 - np.arange(1, 41) * .007])
    config = TrendConfig(max_history=32)
    before = analyze(old, config=config)
    after = analyze(extended, config=config)
    assert tuple(t for t in after.trends if t.end_index < 60) == before.trends
    assert tuple(d for d in after.diagnostics if d.end_index < 60) == before.diagnostics
    changed = extended.with_columns(
        pl.when(pl.col("timestamp") > old["timestamp"][-1])
        .then(pl.col("close") * 100).otherwise(pl.col("close")).alias("close"))
    altered = analyze(changed, config=config)
    assert tuple(t for t in altered.trends if t.end_index < 60) == before.trends
    assert tuple(d for d in altered.diagnostics if d.end_index < 60) == before.diagnostics


def test_as_of_truncates_all_history_and_ignores_invalid_future_prices():
    frame = prices(np.arange(50) * .003)
    cutoff = frame["timestamp"][29]
    changed = frame.with_columns(pl.when(pl.col("timestamp") > cutoff).then(float("nan"))
                                 .otherwise(pl.col("close")).alias("close"))
    assert analyze(changed, as_of=cutoff) == analyze(frame.head(30))


def test_replay_still_returns_individual_current_snapshots_not_nested_histories():
    frame = prices(np.arange(40) * .002)
    cutoffs = [frame["timestamp"][19], frame["timestamp"][-1]]
    results = list(replay(frame, as_of_values=cutoffs))
    assert all(result.analysis_mode == "current" for result in results)
    assert results == [analyze_current(frame, as_of=cutoff) for cutoff in cutoffs]


def test_progress_counts_every_eligible_endpoint_even_when_no_trend_qualifies():
    updates = []
    frame = prices(np.zeros(20))
    result = analyze_history(frame, progress=lambda completed, total: updates.append((completed, total)))
    assert result.status == "no_clear_trend"
    assert updates == [(count, 13) for count in range(14)]
    assert {d.end_index for d in result.diagnostics} == set(range(7, 20))


def test_insufficient_data_has_no_historical_records():
    result = analyze(prices(np.zeros(7)))
    assert result.status == "insufficient_data"
    assert result.analysis_mode == "history"
    assert not result.trends and not result.diagnostics


def test_history_above_former_rolling_work_budget_analyzes_every_eligible_endpoint():
    # Exercise the real 2,000-bar rolling window beyond the former 2,017-bar
    # cutoff. One minimum-length candidate per window keeps this test cheap.
    frame = prices(np.zeros(2025))
    config = TrendConfig(max_history=2000, min_segment_length=2000)
    updates = []
    result = analyze(frame, config=config, progress=lambda done, total: updates.append((done, total)))
    assert result.status == "no_clear_trend"
    assert result.analysis_start == frame["timestamp"][0]
    assert result.analysis_end == frame["timestamp"][-1]
    assert [(d.start_index, d.end_index) for d in result.diagnostics] == [
        (end - 1999, end) for end in range(1999, 2025)
    ]
    assert updates == [(done, 26) for done in range(27)]
