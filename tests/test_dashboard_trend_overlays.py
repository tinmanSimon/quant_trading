"""Trend fits retain source coordinates through bounded/grouped chart windows."""

from datetime import UTC, datetime, timedelta
import math
from types import SimpleNamespace

import polars as pl
import pytest

from dashboard.chart_component import window_payload
from dashboard.chart_data import ChartSeries
from dashboard.trend_overlays import MAX_TREND_POINTS, trend_descriptors, trend_traces


def trend(*, start=2000, end=2999, **overrides):
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    values = dict(trend_id="T1", start_index=start, end_index=end,
                  start_time=stamp + timedelta(hours=start), end_time=stamp + timedelta(hours=end),
                  duration=end - start + 1, direction=1, clarity=0.86, slope=0.0002,
                  intercept=math.log(140), rate_per_bar=math.expm1(0.0002),
                  boundary_robustness=0.9, magnitude=2.5, scores=(0.8, 0.9, 0.7, 0.9),
                  tail_status="local_countertrend", tail_start_time=stamp + timedelta(hours=end - 12),
                  retracement=0.12, onset_truncated=False,
                  lambda_values=(0.25, 0.5), lambda_levels=(0, 1), origin="segmentation")
    values.update(overrides)
    return SimpleNamespace(**values)


def series(count=3000):
    stamps = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(count)]
    prices = [100 + (index % 17) / 10 for index in range(count)]
    return ChartSeries(pl.DataFrame({"timestamp": stamps, "open": prices, "close": prices,
                                     "high": [value + 0.5 for value in prices],
                                     "low": [value - 0.5 for value in prices],
                                     "volume": [50.0] * count}), timeframe="1h")


def test_overlay_window_coordinates_hover_and_bounded_payload():
    fitted = trend()
    payload = window_payload(series(), chart_id="test", timeframe="1h", timezone="UTC",
                             trends=[fitted], overlay_revision=4)
    candles, volume, line = payload["figure"]["data"]
    assert candles["type"] == "candlestick"
    assert volume["type"] == "bar"
    assert line["type"] == "scatter"
    assert (line["xaxis"], line["yaxis"]) == ("x", "y")
    assert len(line["x"]) <= MAX_TREND_POINTS
    assert line["x"][0] == max(fitted.start_index, payload["loaded_range"][0])
    assert line["x"][-1] == fitted.end_index
    assert line["y"][0] == pytest.approx(math.exp(fitted.intercept + fitted.slope * (line["x"][0] - 2000)))
    assert payload["overlay_revision"] == 4
    assert payload["trend_overlays"][0]["start_index"] == 2000
    assert "Slope (log-price per bar): +0.0002" in line["hovertemplate"]
    assert "Clarity: 0.860" in line["hovertemplate"]
    assert "Boundary robustness: 90.0%" in line["hovertemplate"]
    assert "local countertrend" in line["hovertemplate"]
    assert "Retracement: 12.0%" in line["hovertemplate"]
    assert "Duration: 1000 bars" in line["hovertemplate"]
    assert "Segmentation λ values: 0.25, 0.5" in line["hovertemplate"]
    assert "Color λ: 0.25 (smallest segmentation λ)" in line["hovertemplate"]
    assert line["meta"]["lambda_values"] == [0.25, 0.5]
    assert line["meta"]["primary_lambda"] == 0.25
    assert "λ=0.25" in line["name"]


def test_grouping_changes_candles_but_not_the_fit():
    fitted = trend()
    request = {"chart_id": "test", "request_id": 1, "start": -0.5, "end": 2999.5, "width": 200}
    payload = window_payload(series(), chart_id="test", timeframe="1h", timezone="UTC",
                             request=request, trends=[fitted])
    line = payload["figure"]["data"][2]
    assert max(payload["source_counts"]) > 1
    assert line["x"][0] == 2000
    assert line["x"][-1] == 2999
    assert line["y"][0] == pytest.approx(140)
    assert line["y"][-1] == pytest.approx(math.exp(fitted.intercept + 999 * fitted.slope))
    assert len(line["x"]) == MAX_TREND_POINTS


def test_offscreen_trends_keep_descriptor_without_trace():
    payload = window_payload(series(), chart_id="test", timeframe="1h", timezone="UTC",
                             trends=[trend(start=0, end=50)])
    assert len(payload["figure"]["data"]) == 2
    assert len(payload["trend_overlays"]) == 1


def test_fractional_clipping_and_no_extrapolation():
    descriptor = trend_descriptors([trend(start=10, end=50)])[0]
    line = trend_traces([descriptor], (12.25, 24.75))[0]
    assert line.x[0] == 12.25
    assert line.x[-1] == 24.75
    assert line.y[0] == pytest.approx(math.exp(descriptor["intercept"] + 2.25 * descriptor["slope"]))
    assert trend_traces([descriptor], (51, 60)) == []


def test_hover_escapes_non_numeric_content():
    descriptor = trend_descriptors([trend(trend_id="<img src=x>")])[0]
    assert "<img" not in descriptor["hovertemplate"]
    assert "&lt;img" in descriptor["hovertemplate"]


def test_nested_and_overlapping_trends_all_render_with_lambda_colors():
    fits = [trend(start=10, end=100, trend_id="long", lambda_values=(8.0,), lambda_levels=(5,)),
            trend(start=50, end=80, trend_id="nested", direction=-1, slope=-0.0002,
                  lambda_values=(0.5,), lambda_levels=(1,)),
            trend(start=90, end=120, trend_id="overlapping", lambda_values=(2.0,), lambda_levels=(3,))]
    descriptors = trend_descriptors(fits)
    traces = trend_traces(descriptors, (0, 130))
    assert [trace.meta["trend_id"] for trace in traces] == ["long", "nested", "overlapping"]
    assert [(trace.x[0], trace.x[-1]) for trace in traces] == [(10, 100), (50, 80), (90, 120)]
    assert len({trace.line.color for trace in traces}) == 3
    assert [trace.meta["primary_lambda"] for trace in traces] == [8.0, 0.5, 2.0]


def test_lambda_color_stable_across_direction_order_and_window():
    fits = [trend(trend_id="first", lambda_values=(0.5, 2.0), lambda_levels=(1, 3)),
            trend(trend_id="second", direction=-1, lambda_values=(0.5,), lambda_levels=(1,))]
    descriptors = trend_descriptors(fits)
    reverse = trend_descriptors(fits[::-1])
    expected = descriptors[0]["line"]
    assert all(item["line"] == expected for item in descriptors + reverse)
    assert trend_traces(descriptors, (2050, 2200))[0].line.color == expected["color"]
    assert trend_traces(descriptors, (2500, 2750))[0].line.color == expected["color"]


def test_numeric_lambda_not_grid_level_controls_color():
    descriptors = trend_descriptors([
        trend(lambda_values=(0.5,), lambda_levels=(1,)),
        trend(lambda_values=(0.50001,), lambda_levels=(1,)),
    ])
    assert descriptors[0]["line"]["color"] != descriptors[1]["line"]["color"]


def test_tail_only_trend_is_explicit_and_does_not_invent_lambda():
    descriptor = trend_descriptors([trend(lambda_values=(), lambda_levels=(), origin="tail")])[0]
    assert descriptor["primary_lambda"] is None
    assert descriptor["lambda_values"] == []
    assert "Tail-discovered · no segmentation λ" in descriptor["hovertemplate"]
    assert "Color λ:" not in descriptor["hovertemplate"]
    assert descriptor["line"]["color"] == "#64748b"
    assert descriptor["line"]["dash"] == "dash"
    assert trend_traces([descriptor], (2000, 2999))[0].meta["origin"] == "tail"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True, "1"])
def test_malformed_lambda_rejected(value):
    with pytest.raises(ValueError, match="lambda"):
        trend_descriptors([trend(lambda_values=(value,))])


@pytest.mark.parametrize("overrides", [{"slope": float("nan")}, {"start_index": -1}, {"end_index": 5}])
def test_malformed_fit_rejected(overrides):
    with pytest.raises(ValueError, match="source-bar"):
        trend_descriptors([trend(**overrides)])
