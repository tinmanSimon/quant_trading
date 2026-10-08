from datetime import UTC, datetime

import polars as pl

from dashboard.trend_input import completed_prefix


def test_hourly_completion_excludes_suffix_without_reindexing():
    frame = pl.DataFrame({"timestamp": [datetime(2026, 1, 5, h, tzinfo=UTC) for h in (14, 15, 16)],
                          "close": [100., 101., 102.]})
    result = completed_prefix(frame, timeframe="1h", cutoff=datetime(2026, 1, 5, 16, 30, tzinfo=UTC))
    assert result.equals(frame.head(2))
    assert frame.height == 3


def test_daily_session_labels_wait_for_local_calendar_day_end():
    frame = pl.DataFrame({"timestamp": [datetime(2026, 1, d, tzinfo=UTC) for d in (5, 6)],
                          "close": [100., 101.]})
    # Midnight UTC is still the previous evening in the chart's session zone.
    assert completed_prefix(frame, timeframe="1d", cutoff=datetime(2026, 1, 6, tzinfo=UTC)).is_empty()
    assert completed_prefix(frame, timeframe="1d", cutoff=datetime(2026, 1, 6, 5, tzinfo=UTC)).height == 1


def test_week_and_month_end_bounds():
    week = pl.DataFrame({"timestamp": [datetime(2026, 1, 5, tzinfo=UTC)], "close": [1.]})
    assert completed_prefix(week, timeframe="1wk", cutoff=datetime(2026, 1, 11, 23, tzinfo=UTC)).is_empty()
    assert completed_prefix(week, timeframe="1wk", cutoff=datetime(2026, 1, 12, 5, tzinfo=UTC)).height == 1
    month = pl.DataFrame({"timestamp": [datetime(2025, 12, 1, tzinfo=UTC)], "close": [1.]})
    assert completed_prefix(month, timeframe="1mo", cutoff=datetime(2026, 1, 1, 5, tzinfo=UTC)).height == 1
