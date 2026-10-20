"""Chart primitive validation, source coordinates, and bounded display output."""

from copy import deepcopy
import math

import pytest

from dashboard.chart_component import window_payload
from dashboard.chart_overlays import MAX_OVERLAY_POINTS, overlay_traces, validate_overlay, window_overlays
from test_dashboard_trend_overlays import series


def line(**changes):
    return {"id": "example:line", "type": "line", "name": "Example", "x": [0, 10, 20],
            "y": [100, 200, 100], "line": {"color": "#334455"}, **changes}


def test_fractional_clipping_interpolates_price_without_extrapolation():
    trace = overlay_traces([line()], [2.5, 12.5])[0]
    assert list(trace.x) == [2.5, 10, 12.5]
    assert list(trace.y) == [125, 200, 175]
    assert trace.meta["overlay_id"] == "example:line"
    assert trace.xaxis == "x" and trace.yaxis == "y"
    assert overlay_traces([line()], [21, 25]) == []


def test_band_boundaries_and_null_breaks_are_preserved():
    descriptor = {"id": "band", "type": "band", "x": [0, 10, 11, 12, 20],
                  "lower": [90, 100, None, 110, 120], "upper": [110, 120, None, 130, 140],
                  "fillcolor": "rgba(0,20,200,0.2)"}
    lower, upper = overlay_traces(window_overlays([descriptor], [2.5, 15]), [2.5, 15])
    assert list(lower.x) == list(upper.x) == [2.5, 10, 11, 12, 15]
    assert list(lower.y) == [92.5, 100, None, 110, 113.75]
    assert list(upper.y) == [112.5, 120, None, 130, 133.75]
    assert upper.fill == "tonexty"
    assert upper.fillcolor == descriptor["fillcolor"]
    assert lower.connectgaps is False and upper.connectgaps is False


def test_markers_are_not_interpolated_at_window_edges():
    marker = line(type="markers", marker={"color": "red", "symbol": "diamond"})
    descriptor = window_overlays([marker], [2.5, 12.5])[0]
    trace = overlay_traces([descriptor], [2.5, 12.5])[0]
    assert list(trace.x) == [10]
    assert list(trace.y) == [200]
    assert trace.mode == "markers"
    assert trace.marker.symbol == "diamond"


def test_generic_log_line_supports_an_analytic_curve_without_trend_metadata():
    descriptor = {"id": "fit", "type": "log_line", "start_index": 10, "end_index": 100,
                  "intercept": math.log(100), "slope": 0.001}
    validate_overlay(descriptor)
    trace = overlay_traces([descriptor], [15.5, 25.5])[0]
    assert trace.x[0] == 15.5 and trace.x[-1] == 25.5
    assert trace.y[0] == pytest.approx(100 * math.exp(0.0055))
    assert "trend_overlay" not in trace.meta


def test_large_arrays_are_bounded_on_the_wire_and_original_results_remain_unchanged():
    positions = list(range(100_000))
    values = [100.0 + index % 10 for index in positions]
    values[35_000], values[67_000] = 1000, -100
    primitive = line(x=positions, y=values)
    before = deepcopy(primitive)
    output = window_overlays([primitive], [-0.5, 99_999.5])[0]
    assert len(output["x"]) <= MAX_OVERLAY_POINTS
    assert min(output["y"]) == -100 and max(output["y"]) == 1000
    assert output["sampled"] is True
    assert output["source_range"] == [0, 99_999]
    assert primitive == before
    narrow = window_overlays([primitive], [4000.25, 4020.75])[0]
    assert len(narrow["x"]) <= 23
    assert narrow["sampled"] is False


def test_reduction_never_draws_a_line_across_a_discarded_gap():
    positions = list(range(10_000))
    values = [None if index % 13 == 0 else float(index) for index in positions]
    output = window_overlays([line(x=positions, y=values)], [0, 9999])[0]
    for index in range(1, len(output["x"])):
        first, last = output["x"][index - 1:index + 1]
        if any(value is None for value in values[first + 1:last]):
            assert output["y"][index] is None
    assert len(output["x"]) <= MAX_OVERLAY_POINTS


def test_payload_includes_generic_coverage_and_only_bounded_original_coordinates():
    primitive = line(x=list(range(3000)), y=[150 + index / 100 for index in range(3000)])
    payload = window_payload(series(), chart_id="generic", timeframe="1h", timezone="UTC",
                             overlays=[primitive], overlay_revision=5)
    assert payload["overlay_revision"] == 5
    assert payload["overlay_coverage"] == payload["loaded_range"]
    descriptor = payload["overlays"][0]
    assert descriptor["source_range"] == [0, 2999]
    assert len(descriptor["x"]) < 200
    trace = payload["figure"]["data"][2]
    assert trace["x"][0] == payload["loaded_range"][0]
    assert trace["x"][-1] == 2999
    assert trace["meta"]["chart_overlay"] is True


@pytest.mark.parametrize("changes", [
    {"type": "unknown"}, {"id": ""}, {"x": [0, 2, 1]}, {"x": [0, math.nan, 2]},
    {"y": [100, math.inf, 100]}, {"y": [100]}, {"y": [True, 100, 100]},
    {"line": {"dash": "invalid"}}, {"name": {"bad": 1}}, {"hovertemplate": {"bad": 1}},
    {"hovertemplate": ["first", "second", "third"]},
    {"type": "band", "lower": [1, None, 3], "upper": [2, 3, 4]},
    {"type": "band", "lower": [1, 4, 3], "upper": [2, 3, 4]},
])
def test_invalid_primitive_is_rejected_before_rendering(changes):
    with pytest.raises(ValueError):
        validate_overlay(line(**changes))


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        window_overlays([line(), line()], [0, 20])


@pytest.mark.parametrize("marker", [
    {"size": [1, 2, 3]}, {"color": ["red", "blue", "green"]},
    {"line": {"color": ["red", "blue", "green"]}}, {"line": {"width": (1, 2, 3)}},
])
def test_per_point_marker_styles_are_rejected_instead_of_bypassing_sampling(marker):
    with pytest.raises(ValueError, match="scalar values"):
        validate_overlay(line(type="markers", marker=marker))
