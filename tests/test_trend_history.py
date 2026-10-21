"""Historical records must equal analyses made at their own endpoint."""

from datetime import UTC, datetime, timedelta
from time import perf_counter

import numpy as np
import polars as pl
import pytest

from research.analysis.trend import TrendConfig, analyze, analyze_current, analyze_history, replay


def prices(values):
    return pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
                                       for i in range(len(values))],
                         "symbol": ["TEST"] * len(values), "close": 100 * np.exp(values)})


@pytest.mark.parametrize("lookback", [2000, 24, 8])
def test_all_endpoints_equal_independent_prefix_analyses_including_rolling_context(lookback):
    rng = np.random.default_rng(42)
    frame = prices(np.cumsum(rng.normal(.002, .003, 105)))
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
    assert all(trend.duration <= 2 * lookback for trend in history.trends)
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
    # The initial window can now extend beyond H without discarding its start.
    frame = prices(np.zeros(2025))
    config = TrendConfig(max_history=2000, min_segment_length=2000)
    updates = []
    result = analyze(frame, config=config, progress=lambda done, total: updates.append((done, total)))
    assert result.status == "no_clear_trend"
    assert result.analysis_start == frame["timestamp"][0]
    assert result.analysis_end == frame["timestamp"][-1]
    assert [(d.start_index, d.end_index) for d in result.diagnostics] == [
        (0, end) for end in range(1999, 2025)
    ]
    assert updates == [(done, 26) for done in range(27)]


@pytest.mark.parametrize("path", [
    np.cumsum(np.random.default_rng(519).normal(.002, .01, 70)),
    np.arange(70) * .003,
    np.r_[np.zeros(30), np.ones(20), np.zeros(20)],
])
def test_block_endpoints_match_independent_window_fits(path):
    from research.analysis.trend.api import _describe

    config = TrendConfig(max_history=16)
    frame = prices(path)
    times = frame["timestamp"].to_list()
    values = np.log(frame["close"].to_numpy())
    result = analyze(frame, config=config)
    # Explicit boundaries exercise full windows, shifts and the final partial window.
    for start, endpoints in ((0, range(8, 33)), (16, range(33, 49)),
                             (32, range(49, 65)), (48, range(65, 71))):
        for end in endpoints:
            expected = _describe(times[start:end], values[start:end], start, config)
            assert tuple(t for t in result.trends if t.end_index == end - 1) == expected.trends
            assert tuple(d for d in result.diagnostics if d.end_index == end - 1) == expected.diagnostics


@pytest.mark.parametrize("length", [31, 32, 33, 47, 48, 49])
def test_appending_across_window_shifts_preserves_published_records(length):
    config = TrendConfig(max_history=16)
    past = np.cumsum(np.random.default_rng(15).normal(.003, .01, length))
    before = analyze(prices(past), config=config)
    after = analyze(prices(np.r_[past, [10., -10., 0., 1.] * 12]), config=config)
    assert tuple(t for t in after.trends if t.end_index < length) == before.trends
    assert tuple(d for d in after.diagnostics if d.end_index < length) == before.diagnostics


def test_expired_context_cannot_affect_later_windows():
    config = TrendConfig(max_history=16)
    values = np.cumsum(np.random.default_rng(55).normal(.002, .01, 70))
    before = analyze(prices(values), config=config)
    values[:32] += np.linspace(-10, 10, 32)
    after = analyze(prices(values), config=config)
    assert tuple(t for t in after.trends if t.end_index >= 48) == tuple(
        t for t in before.trends if t.end_index >= 48)
    assert tuple(d for d in after.diagnostics if d.end_index >= 48) == tuple(
        d for d in before.diagnostics if d.end_index >= 48)


def test_only_window_shifts_rebuild_segmentation_and_overlap_is_not_republished(monkeypatch):
    import research.analysis.trend.api as api

    built = []
    original = api.PrefixSegmentations

    def counted(values, config, **kwargs):
        built.append((float(values[0]), len(values)))
        return original(values, config, **kwargs)

    def no_independent_dp(*args, **kwargs):
        raise AssertionError("Historical analysis must reuse its window's prefix segmentation")

    monkeypatch.setattr(api, "PrefixSegmentations", counted)
    monkeypatch.setattr(api, "candidate_provenance", no_independent_dp)
    frame = prices(np.arange(70) * .003)
    updates = []
    result = analyze(frame, config=TrendConfig(max_history=16),
                     progress=lambda done, total: updates.append((done, total)))
    values = np.log(frame["close"].to_numpy())
    assert built == [(float(values[start]), min(32, 70 - start)) for start in (0, 16, 32, 48)]
    assert updates == [(done, 63) for done in range(64)]
    assert len({t.trend_id for t in result.trends}) == len(result.trends)
    assert len({(d.start_index, d.end_index) for d in result.diagnostics}) == len(result.diagnostics)


def test_multiple_window_shifts_share_work_within_interactive_budget():
    frame = prices(np.cumsum(np.random.default_rng(741).normal(.001, .01, 4010)))
    config = TrendConfig(max_history=1000, penalty_grid_levels=5)
    began = perf_counter()
    result = analyze(frame, config=config)
    elapsed = perf_counter() - began
    assert {d.end_index for d in result.diagnostics} == set(range(7, 4010))
    # Includes three shifts and their overlap rebuilds; the old per-bar
    # fallback takes much longer even with this smaller history setting.
    assert elapsed < 30, f"Overlapping-window analysis took {elapsed:.2f}s"
