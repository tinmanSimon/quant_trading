"""Display reduction must stay bounded and preserve the meaning of source bars."""

from datetime import UTC, datetime, timedelta
import math

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from dashboard.chart_data import ChartSeries, MAX_WINDOW_CANDLES


def frame_for(stamps):
    count = len(stamps)
    return pl.DataFrame({
        "timestamp": stamps, "symbol": ["AAPL"] * count,
        "open": [100.0 + index / 100 for index in range(count)],
        "high": [102.0 + index / 100 for index in range(count)],
        "low": [99.0 + index / 100 for index in range(count)],
        "close": [101.0 + index / 100 for index in range(count)],
        "volume": [index + 0.125 for index in range(count)],
    })


def intraday_frame(days=520, bars=16, *, minutes=60):
    start = datetime(2023, 1, 2, 9, tzinfo=UTC)
    return frame_for([start + timedelta(days=day, minutes=bar * minutes)
                      for day in range(days) for bar in range(bars)])


@pytest.mark.parametrize("count", [1, 49, 50, 51, 10000])
def test_default_view_is_latest_fifty_and_transfer_is_bounded(count):
    frame = frame_for([datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
                       for index in range(count)])
    series = ChartSeries(frame, timeframe="1m")
    window = series.window()
    assert window.view_range == (max(0, count - 50) - 0.5, count - 0.5)
    assert window.frame.height <= 150
    assert window.last_indices[-1] == count - 1
    assert window.resolution == "Original 1m bars"
    assert window.source_counts == [1] * window.frame.height
    assert series.total_count == count
    assert series.estimated_size > frame.estimated_size()


def test_pan_both_directions_evicts_distant_data_and_groups_are_stable():
    series = ChartSeries(intraday_frame(), timeframe="1h")
    recent = series.window()
    early = series.window(49.5, 149.5)
    middle = series.window(3999.5, 4099.5)
    assert early.first_indices[0] == 0
    assert early.last_indices[-1] < middle.first_indices[0]
    assert middle.last_indices[-1] < recent.first_indices[0]
    assert_frame_equal(series.window().frame, recent.frame)
    # Equal resolution uses complete, globally anchored groups while panning.
    left = series.window(999.5, 3999.5, width=1000)
    right = series.window(1099.5, 4099.5, width=1000)
    first_groups = dict(zip(left.first_indices, left.last_indices))
    second_groups = dict(zip(right.first_indices, right.last_indices))
    common = set(first_groups) & set(second_groups)
    assert common
    assert all(first_groups[key] == second_groups[key] for key in common)


def test_two_year_overview_has_exact_ohlcv_and_originals_are_unchanged():
    frame = intraday_frame()
    before = frame.clone()
    series = ChartSeries(frame, timeframe="1h")
    window = series.window(-0.5, frame.height - 0.5, width=700)
    assert window.frame.height <= 700
    assert "display only" in window.resolution
    assert sum(window.source_counts) == frame.height
    assert window.first_indices[0] == 0 and window.last_indices[-1] == frame.height - 1
    for index, (first, last) in enumerate(zip(window.first_indices, window.last_indices)):
        chunk = frame.slice(first, last - first + 1)
        row = window.frame.row(index, named=True)
        assert row["open"] == chunk["open"][0]
        assert row["high"] == chunk["high"].max()
        assert row["low"] == chunk["low"].min()
        assert row["close"] == chunk["close"][-1]
        assert row["volume"] == math.fsum(chunk["volume"].to_list())
        assert window.positions[index] == (first + last) / 2
        assert window.widths[index] == last - first + 1
    assert_frame_equal(frame, before)
    assert_frame_equal(series.frame, before)
    external = series.frame
    external.replace_column(external.get_column_index("close"), pl.Series("close", [0.0] * frame.height))
    assert_frame_equal(series.frame, before)


def test_intraday_groups_stop_at_date_boundaries_and_missing_intervals():
    # 600 bars exceed the minimum display budget, but each continuous block
    # has 100 minutes; the 30-minute gaps must not be silently bridged.
    start = datetime(2025, 1, 6, 14, 30, tzinfo=UTC)
    stamps = [start + timedelta(days=day, minutes=block * 130 + index)
              for day in range(3) for block in range(2) for index in range(100)]
    frame = frame_for(stamps)
    omitted = stamps[99] + timedelta(minutes=1)
    series = ChartSeries(frame, timeframe="1m", omitted_timestamps=[omitted, stamps[0], omitted])
    window = series.window(-0.5, len(stamps) - 0.5, width=200)
    assert window.frame.height <= 200
    assert "× 1m" in window.resolution
    assert series.omitted_timestamps == (omitted,)
    assert sum(window.omitted_counts) == 0  # Inter-group omission stays available as a boundary marker.
    for first, last in zip(window.first_indices, window.last_indices):
        assert first // 100 == last // 100
        assert stamps[last] - stamps[first] == timedelta(minutes=last - first)


def test_daily_summaries_retain_gaps_inside_groups_and_volume_fractions():
    start = datetime(2025, 1, 1, 14, tzinfo=UTC)
    stamps = [start + timedelta(days=day, hours=hour) for day in range(120) for hour in (0, 2)]
    missing = tuple(start + timedelta(days=day, hours=1) for day in range(120))
    series = ChartSeries(frame_for(stamps), timeframe="1h", omitted_timestamps=missing)
    window = series.window(-0.5, len(stamps) - 0.5, width=200)
    assert window.resolution == "Daily summaries (display only; America/New_York calendar days)"
    assert window.frame.height == 120
    assert window.omitted_counts == [1] * 120
    assert window.source_counts == [2] * 120
    assert window.frame["volume"][0] == 1.25


def test_week_summaries_keep_daily_session_labels_in_their_original_date():
    start = datetime(2020, 1, 6, tzinfo=UTC)  # Monday, including in canonical daily labels.
    stamps = [start + timedelta(days=day) for day in range(2100)]
    series = ChartSeries(frame_for(stamps), timeframe="1d", session_timezone="America/New_York")
    window = series.window(-0.5, len(stamps) - 0.5, width=200)
    assert window.frame.height <= 200
    assert "calendar-week" in window.resolution.lower()
    assert "session dates" in window.resolution
    assert all(stamp.weekday() == 0 for stamp in window.starts[1:])
    assert sum(window.source_counts) == len(stamps)


def test_dst_groups_use_aware_timestamps_and_retain_extended_hours():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    stamps = [datetime(2025, 3, day, 4, tzinfo=zone).astimezone(UTC) + timedelta(minutes=minute)
              for day in (7, 10) for minute in range(480)]
    series = ChartSeries(frame_for(stamps), timeframe="1m")
    window = series.window(-0.5, len(stamps) - 0.5, width=200)
    assert window.frame.height <= 200
    assert sum(window.source_counts) == len(stamps)
    assert window.starts[0] == datetime(2025, 3, 7, 9, tzinfo=UTC)
    assert any(stamp == datetime(2025, 3, 10, 8, tzinfo=UTC) for stamp in window.starts)
    assert all(first.astimezone(zone).date() == last.astimezone(zone).date()
               for first, last in zip(window.starts, window.ends))


def test_sparse_buffers_cannot_exceed_response_limit():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    # A dense middle view borders sparse runs: those buffers must still be capped.
    sparse = [start + timedelta(minutes=index * 2) for index in range(5000)]
    dense = [sparse[-1] + timedelta(minutes=index + 1) for index in range(10000)]
    tail = [dense[-1] + timedelta(minutes=(index + 1) * 2) for index in range(5000)]
    series = ChartSeries(frame_for(sparse + dense + tail), timeframe="1m")
    for lower, upper in ((-0.5, 19999.5), (4999.5, 14999.5), (999.5, 5999.5), (13999.5, 18999.5)):
        window = series.window(lower, upper, width=1000)
        assert window.frame.height <= MAX_WINDOW_CANDLES
        visible = sum(last + 0.5 > lower and first - 0.5 < upper
                      for first, last in zip(window.first_indices, window.last_indices))
        assert visible <= 1000


def test_source_sorting_does_not_mutate_input_and_utc_conversion_preserves_instants():
    from zoneinfo import ZoneInfo

    stamps = [datetime(2025, 1, 2, 10, tzinfo=ZoneInfo("America/New_York")),
              datetime(2025, 1, 2, 9, tzinfo=ZoneInfo("America/New_York"))]
    source = frame_for(stamps)
    before = source.clone()
    series = ChartSeries(source, timeframe="1h")
    assert series.timestamps == tuple(sorted(stamp.astimezone(UTC) for stamp in stamps))
    assert_frame_equal(source, before)


@pytest.mark.parametrize("start,end", [(float("nan"), 10), (0, float("inf")), (10, 10), (20, 10), (True, 10)])
def test_invalid_view_bounds_are_rejected(start, end):
    with pytest.raises(ValueError, match="range"):
        ChartSeries(intraday_frame(days=1), timeframe="1h").window(start, end)


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf"), True, "1000"])
def test_invalid_width_is_rejected(width):
    with pytest.raises(ValueError, match="width"):
        ChartSeries(intraday_frame(days=1), timeframe="1h").window(width=width)


def test_navigation_outside_data_is_clamped_without_empty_windows():
    series = ChartSeries(intraday_frame(days=1), timeframe="1h")
    assert series.window(-100, -90).view_range == (-0.5, 9.5)
    assert series.window(100, 110).view_range == (5.5, 15.5)
    assert series.window(-100, 100).view_range == (-0.5, 15.5)


def test_invalid_snapshot_is_rejected():
    source = intraday_frame(days=1)
    with pytest.raises(ValueError, match="at least one"):
        ChartSeries(source.head(0), timeframe="1h")
    with pytest.raises(ValueError, match="unique"):
        ChartSeries(pl.concat([source.head(1)] * 2), timeframe="1h")
    with pytest.raises(ValueError, match="timezone-aware"):
        ChartSeries(source.with_columns(pl.col("timestamp").dt.replace_time_zone(None)), timeframe="1h")


def test_overflowing_aggregate_volume_is_rejected():
    frame = intraday_frame(days=1, bars=300, minutes=1).with_columns(pl.lit(1e308).alias("volume"))
    with pytest.raises(ValueError, match="finite"):
        ChartSeries(frame, timeframe="1m").window(-0.5, 299.5, width=200)
