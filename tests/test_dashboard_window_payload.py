"""Validate bounded browser payloads independently of storage and chart rendering."""

from datetime import UTC, datetime, timedelta
import json
import math

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from dashboard.chart_component import viewport_request, window_payload
from dashboard.chart_data import ChartSeries


def _frame(stamps):
    count = len(stamps)
    return pl.DataFrame({"timestamp": stamps, "symbol": ["AAPL"] * count,
                         "open": [10.0 + index / 10 for index in range(count)],
                         "high": [12.0 + index / 10 for index in range(count)],
                         "low": [9.0 + index / 10 for index in range(count)],
                         "close": [11.0 + index / 10 for index in range(count)],
                         "volume": [index + 0.125 for index in range(count)]})


def _request(**kwargs):
    return {"chart_id": "snapshot", "request_id": 1, "start": -0.5,
            "end": 149.5, "width": 900, **kwargs}


@pytest.mark.parametrize("value", [None, [], "snapshot", {}, _request(chart_id="stale"),
                                    _request(request_id=0), _request(request_id=-1),
                                    _request(request_id=True), _request(request_id=1.0),
                                    _request(request_id=2**53), _request(start=True),
                                    _request(end=None), _request(width=False),
                                    _request(start=float("nan")), _request(end=float("inf")),
                                    _request(width=float("nan")), _request(start=150),
                                    _request(start=149.5), _request(width=0),
                                    _request(width=10001), _request(start="1"),
                                    _request(start=10**400), _request(end=10**400),
                                    _request(width=10**400)])
def test_bad_or_stale_client_navigation_is_ignored(value):
    assert viewport_request(value, "snapshot") is None


def test_valid_navigation_retains_exact_bounds_and_sequence():
    request = _request(start=37.125, end=100.375, request_id=1234, width=758.5)
    assert viewport_request(request, "snapshot") == request


def test_initial_payload_is_small_and_preserves_exact_latest_source_data():
    start = datetime(2023, 1, 1, 9, tzinfo=UTC)
    source = _frame([start + timedelta(days=index // 16, hours=index % 16) for index in range(8320)])
    before = source.clone()
    series = ChartSeries(source, timeframe="1h")
    payload = window_payload(series, chart_id="snapshot", timeframe="1h", timezone="UTC")
    candles, volumes = payload["figure"]["data"]
    assert payload["view_range"] == (8269.5, 8319.5)
    assert len(candles["x"]) <= 150
    assert candles["x"] == list(range(8170, 8320))
    assert len(payload["overview"]["positions"]) <= 300
    assert payload["total_count"] == 8320
    assert payload["resolution"] == "Original 1h bars"
    assert payload["request_id"] == 0
    assert len(json.dumps(payload).encode()) < 150_000
    for name in ("open", "high", "low", "close"):
        assert candles[name] == source[name].tail(150).to_list()
    assert candles["customdata"] == volumes["y"] == source["volume"].tail(150).to_list()
    assert "Volume: %{customdata}" in candles["hovertemplate"]
    assert_frame_equal(source, before)
    assert_frame_equal(series.frame, before)


def test_overview_hover_explains_source_range_counts_and_aggregated_volume():
    start = datetime(2025, 1, 1, 14, tzinfo=UTC)
    source = _frame([start + timedelta(days=day, hours=hour)
                     for day in range(120) for hour in (0, 2)])
    missing = [start + timedelta(days=day, hours=1) for day in range(120)]
    series = ChartSeries(source, timeframe="1h", omitted_timestamps=missing)
    payload = window_payload(series, chart_id="snapshot", timeframe="1h", timezone="UTC",
                             request=_request(end=239.5, width=200))
    candles, volumes = payload["figure"]["data"]
    assert payload["request_id"] == 1
    assert "Daily summaries" in payload["resolution"]
    assert payload["source_counts"] == [2] * 120
    assert len(candles["x"]) == 120
    for index, hover in enumerate(candles["text"]):
        assert "Display summary:" in hover
        assert "Source bars: 2" in hover
        assert "Recorded omissions within summary: 1" in hover
        assert source["timestamp"][2 * index].isoformat() in hover
        assert source["timestamp"][2 * index + 1].isoformat() in hover
        expected_volume = math.fsum(source["volume"].slice(2 * index, 2).to_list())
        assert candles["customdata"][index] == volumes["y"][index] == expected_volume
    assert payload["figure"]["layout"]["xaxis"]["range"] == [-0.5, 239.5]
    assert payload["figure"]["layout"]["xaxis2"]["rangeslider"]["visible"] is False


def test_raw_omission_markers_stay_between_existing_candles_not_over_them():
    start = datetime(2025, 1, 1, 14, tzinfo=UTC)
    stamps = [start + timedelta(hours=index) for index in (0, 2, 4)]
    missing = [start - timedelta(hours=1), start + timedelta(hours=1),
               start + timedelta(hours=3), start + timedelta(hours=5)]
    series = ChartSeries(_frame(stamps), timeframe="1h", omitted_timestamps=missing)
    payload = window_payload(series, chart_id="snapshot", timeframe="1h", timezone="UTC")
    assert [shape["x0"] for shape in payload["figure"]["layout"]["shapes"]] == [-0.5, 0.5, 1.5, 2.5]
    hover = " ".join(item["hovertext"] for item in payload["figure"]["layout"]["annotations"])
    assert all(stamp.isoformat() in hover for stamp in missing)
    assert payload["figure"]["data"][0]["x"] == [0, 1, 2]
    assert payload["source_counts"] == [1, 1, 1]


def test_many_omissions_cannot_make_annotation_payload_unbounded():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    stamps = [start + timedelta(hours=index) for index in range(100)]
    missing = [stamp + timedelta(minutes=minute) for stamp in stamps for minute in range(1, 60)]
    series = ChartSeries(_frame(stamps), timeframe="1h", omitted_timestamps=missing)
    payload = window_payload(series, chart_id="snapshot", timeframe="1h", timezone="UTC")
    annotations = payload["figure"]["layout"]["annotations"]
    assert len(annotations) <= 80
    visible = [item["x"] for item in annotations if 49.5 <= item["x"] <= 99.5]
    assert len(visible) == 51  # Visible omissions take priority over buffered bars.
    assert "80 of 100" in payload["omission_marker_note"]
    assert all(len(item["hovertext"].split("<br>")) <= 6 for item in annotations)
    assert len(series.omitted_timestamps) == 5900
    assert len(json.dumps(payload).encode()) < 200_000


def test_stale_chart_request_cannot_navigate_a_different_snapshot():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    series = ChartSeries(_frame([start + timedelta(hours=index) for index in range(300)]), timeframe="1h")
    payload = window_payload(series, chart_id="snapshot", timeframe="1h", timezone="UTC",
                             request=_request(chart_id="old-snapshot", end=30))
    assert payload["request_id"] == 0
    assert payload["view_range"] == (249.5, 299.5)
