"""Independent indicator rows, histogram transport, and synchronized navigation."""

from copy import deepcopy
import json

import pytest

from chart_logics.base import LogicResult
from dashboard.chart_component import _javascript, window_payload
from dashboard.chart_overlays import MAX_OVERLAY_POINTS, overlay_traces, validate_overlay, window_overlays
from test_dashboard_chart_browser import _run_browser_document
from test_dashboard_trend_overlays import series


def indicators(count=3000):
    positions = list(range(count))
    rsi = {"id": "rsi-instance", "title": "RSI <14>", "range": [0, 100]}
    macd = {"id": "macd-instance", "title": "MACD (12, 26, 9)"}
    return [
        {"id": "rsi", "type": "line", "panel": rsi, "x": positions,
         "y": [None if i < 14 else 40 + i % 30 for i in positions]},
        {"id": "rsi-level", "type": "line", "panel": rsi, "x": [0, count - 1], "y": [70, 70]},
        {"id": "macd", "type": "line", "panel": macd, "x": positions,
         "y": [None if i < 33 else (i % 5 - 2) * 1000 for i in positions]},
        {"id": "histogram", "name": "MACD histogram", "type": "bar", "panel": macd,
         "x": positions, "y": [None if i < 33 else (i % 7 - 3) * 1000 for i in positions],
         "marker": {"color": "#18a999"}, "hovertemplate": "Histogram: %{y}<extra></extra>"},
    ]


def test_indicator_axes_follow_volume_with_independent_ranges_and_shared_time():
    primitives = indicators()
    LogicResult(overlays=primitives).validate()
    payload = window_payload(series(), chart_id="panels", timeframe="1h", timezone="UTC", overlays=primitives)
    layout, traces = payload["figure"]["layout"], payload["figure"]["data"]
    assert layout["height"] == 1030
    assert layout["yaxis"]["domain"][0] > layout["yaxis2"]["domain"][1]
    assert layout["yaxis2"]["domain"][0] > layout["yaxis3"]["domain"][1]
    assert layout["yaxis3"]["domain"][0] > layout["yaxis4"]["domain"][1]
    assert layout["yaxis3"]["range"] == [0, 100]
    assert [trace["yaxis"] for trace in traces] == ["y", "y2", "y3", "y3", "y4", "y4"]
    assert traces[-1]["type"] == "bar" and traces[-1]["marker"]["color"] == "#18a999"
    assert layout["annotations"][0]["text"] == "RSI &lt;14&gt;"
    for index in range(2, 5):
        axis = layout[f"xaxis{index}"]
        assert axis["matches"] == "x"
        assert axis["range"] == layout["xaxis"]["range"]
        assert axis["showticklabels"] is (index == 4)


def test_bar_sampling_is_bounded_and_never_interpolates_or_connects_gaps():
    primitive = indicators(100_000)[-1]
    primitive["y"][5000] = 999_999
    original = deepcopy(primitive)
    wire = window_overlays([primitive], [0, 99_999])[0]
    assert len(wire["x"]) <= MAX_OVERLAY_POINTS
    assert 999_999 in wire["y"]
    assert primitive == original
    trace = overlay_traces(window_overlays([primitive], [5000.25, 5002.75]), [5000.25, 5002.75])[0]
    assert trace.type == "bar"
    assert list(trace.x) == [5001, 5002]
    assert list(trace.y) == primitive["y"][5001:5003]
    assert trace.xaxis == "x3" and trace.yaxis == "y3"


@pytest.mark.parametrize("panel", [
    {}, {"id": "", "title": "RSI"}, {"id": "x", "title": 12},
    {"id": "x", "title": "RSI", "range": [100, 0]},
    {"id": "x", "title": "RSI", "range": [0, True]},
    {"id": "x", "title": "RSI", "range": [0, float("inf")]},
    {"id": "x", "title": "RSI", "range": None},
])
def test_panel_validation_applies_before_storage_and_rendering(panel):
    primitive = {**indicators()[0], "panel": panel}
    with pytest.raises(ValueError):
        LogicResult(overlays=[primitive]).validate()
    with pytest.raises(ValueError):
        validate_overlay(primitive)


def test_same_panel_requires_consistent_declaration():
    primitives = indicators()[:2]
    primitives[1]["panel"] = {**primitives[1]["panel"], "range": [20, 80]}
    with pytest.raises(ValueError, match="same panel"):
        LogicResult(overlays=primitives).validate()
    with pytest.raises(ValueError, match="same panel"):
        window_overlays(primitives, [0, 2999])


def test_warmup_keeps_an_empty_indicator_panel_without_inventing_values():
    primitive = {**indicators()[0], "y": [None] * 3000}
    trace = overlay_traces([primitive], [100, 150])[0]
    assert trace.meta["panel_placeholder"] is True
    assert trace.xaxis == "x3" and trace.yaxis == "y3"
    assert list(trace.y) == [None]


PANEL_INTERACTIONS = r"""
const result = document.getElementById("result");
console.error = error => { result.textContent = "FAIL: renderer: " + (error?.stack || error); };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(ok, message) { if (!ok) throw new Error(message); }
function close(a, b) { return Math.abs(a - b) < 1e-7; }
async function until(test, message) {
    for (let i = 0; i < 200; i++) { if (test()) return; await pause(25); }
    throw new Error(message);
}
async function test() {
    const host = document.getElementById("host"), calls = [];
    const mount = data => render({parentElement: host, data: structuredClone(data),
        setStateValue: (name, value) => calls.push(value)});
    const cleanup = mount(fixtures.latest), chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "initial fit did not settle");
    await Plotly.relayout(chart, {"xaxis.range": [2970.25, 2990.75], dragmode: "pan"});
    await pause(200);
    const price = chart.layout.yaxis.range.slice(), volume = chart.layout.yaxis2.range.slice();
    const before = calls.length;
    mount(fixtures.added);
    await until(() => chart.data.length === 6 && chart.layout.yaxis4?.range[0] < -3000,
        "indicator panels were not fitted");
    check(chart.layout.yaxis.range.every((value, i) => close(value, price[i])), "indicator changed price scale");
    check(chart.layout.yaxis2.range.every((value, i) => close(value, volume[i])), "indicator changed volume scale");
    check(chart.layout.yaxis3.range[0] === 0 && chart.layout.yaxis3.range[1] === 100, "RSI fixed range ignored");
    check(chart.layout.yaxis2.domain[0] > chart.layout.yaxis3.domain[1], "indicator is above volume");
    check(chart.layout.yaxis3.domain[0] > chart.layout.yaxis4.domain[1], "indicator panels overlap");
    check(chart.style.height === "1030px", "panel height not expanded");
    check(chart.layout.dragmode === "pan" && calls.length === before, "panel addition changed navigation");
    check(chart.querySelector(".annotation").textContent === "RSI <14>", "panel title is not literal text");
    const histogram = chart.data.findIndex(trace => trace.meta?.overlay_id === "histogram");
    check(chart.data[histogram].type === "bar" && chart.data[histogram].yaxis === "y4", "histogram axes lost");
    Plotly.Fx.hover(chart, [{curveNumber: histogram, pointNumber: 30}], "x4y4");
    check(chart.querySelector(".hoverlayer").textContent.includes("Histogram:"), "histogram hover missing");
    Plotly.Fx.unhover(chart);
    await Plotly.relayout(chart, {"xaxis3.range": [2973, 2984]});
    await until(() => chart.layout.xaxis.range[0] === 2973 && chart.layout.xaxis4.range[1] === 2984,
        "indicator pan did not synchronize all rows");
    for (const axis of ["xaxis", "xaxis2", "xaxis3", "xaxis4"])
        check(chart.layout[axis].range[0] === 2973 && chart.layout[axis].range[1] === 2984, "row navigation differs");

    const beforePan = calls.length;
    await Plotly.relayout(chart, {"xaxis.range": [2099.5, 2149.5]});
    await until(() => calls.length > beforePan, "distant pan did not request candles");
    const request = calls.at(-1);
    mount(fixtures.removed);
    await until(() => chart.data.length === 2 && !chart.layout.xaxis3, "panel removal did not remove its axes");
    check(chart.style.height === "650px", "panel removal did not restore height");
    const old = structuredClone(fixtures.earlier);
    old.request_id = request.request_id;
    mount(old);
    await until(() => close(chart.layout.xaxis.range[0], 2099.5), "viewport response was lost");
    check(!chart.layout.xaxis3 && chart.data.length === 2, "old viewport restored removed panel");

    const candles = chart.data[0].x.slice(), beforeRefresh = calls.length;
    const stale = structuredClone(fixtures.added);
    stale.overlay_revision = 3;
    mount(stale);
    await until(() => chart.layout.xaxis4 && calls.length > beforeRefresh, "new panel layout on old viewport lost");
    check(chart.data[0].x.every((value, i) => value === candles[i]), "new panel restored old candles");
    check(close(chart.layout.xaxis4.range[0], 2099.5), "new panel restored old viewport");
    const current = structuredClone(fixtures.earlier);
    current.request_id = calls.at(-1).request_id;
    current.overlay_revision = 3;
    mount(current);
    await until(() => chart.data.length === 6 && chart.layout.yaxis4.range[0] < -3000, "fresh panel samples not drawn");
    check(chart.data[2].x[0] >= fixtures.earlier.loaded_range[0], "panel samples use old coordinates");
    mount(fixtures.removed);
    await pause(200);
    check(chart.data.length === 6 && chart.layout.xaxis4, "old display revision removed newer panels");
    cleanup();
    result.textContent = "PASS";
}
window.addEventListener("unhandledrejection", event => { result.textContent = "FAIL: " + event.reason; });
test().catch(error => { result.textContent = "FAIL: " + error.stack; });
document.querySelectorAll("script").forEach(script => script.remove());
"""


@pytest.mark.browser
def test_indicator_panels_navigation_scaling_and_display_revision_races(tmp_path):
    source = series()
    options = {"chart_id": "indicator-panels", "timeframe": "1h", "timezone": "UTC"}
    fixtures = {
        "latest": window_payload(source, **options),
        "added": window_payload(source, **options, overlays=indicators(), overlay_revision=1),
        "removed": window_payload(source, **options, overlay_revision=2),
        "earlier": window_payload(source, **options, overlays=indicators(), overlay_revision=1,
            request={"chart_id": "indicator-panels", "request_id": 1,
                     "start": 2099.5, "end": 2149.5, "width": 1000}),
    }
    encoded = json.dumps(fixtures).replace("<", "\\u003c")
    document = ("<!doctype html><meta charset='utf-8'><div id='host' style='width:1000px'></div>"
        "<pre id='result'>PENDING</pre><script type='module'>" + _javascript()
        + "\nconst fixtures = " + encoded + ";\n" + PANEL_INTERACTIONS + "</script>")
    _run_browser_document(tmp_path, document)
