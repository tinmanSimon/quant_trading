"""Offline browser coverage for composable chart primitives and revision races."""

import json

import pytest

from dashboard.chart_component import _javascript, window_payload
from test_dashboard_chart_browser import _run_browser_document
from test_dashboard_trend_overlays import series


INTERACTIONS = r"""
const result = document.getElementById("result");
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
    const cleanup = mount(fixtures.latest);
    const chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "initial fit did not settle");
    const localRange = [2970.25, 2990.75];
    await Plotly.relayout(chart, {"xaxis.range": localRange, dragmode: "pan"});
    await until(() => close(chart.layout.xaxis.range[0], localRange[0]), "local pan did not settle");
    const before = calls.length, volume = chart.layout.yaxis2.range.slice();
    mount(fixtures.overlay);
    await until(() => chart.data.length === 6 && chart.layout.yaxis.range[1] > 210, "generic overlays not fitted");
    check(chart.layout.xaxis.range.every((value, i) => close(value, localRange[i])), "apply reset zoom");
    check(chart.layout.dragmode === "pan", "apply reset pan mode");
    check(calls.length === before, "apply fetched a redundant viewport");
    check(chart.layout.yaxis.range[1] < 1000, "offscreen generic values affected visible scale");
    check(chart.layout.yaxis2.range.every((value, i) => close(value, volume[i])), "logic changed volume scale");
    const line = chart.data.find(trace => trace.meta?.overlay_id === "levels");
    check(line.y.includes(null), "line segments joined across an explicit null break");
    check(line.x.length <= 2048, "line payload is unbounded");
    const upper = chart.data.find(trace => trace.meta?.band_upper);
    check(upper.fill === "tonexty" && upper.fillcolor === "rgba(80,140,240,0.15)", "band fill missing");
    const markerIndex = chart.data.findIndex(trace => trace.meta?.overlay_id === "events");
    check(chart.data[markerIndex].mode === "markers" && chart.data[markerIndex].marker.symbol === "diamond",
          "marker style missing");
    Plotly.Fx.hover(chart, [{curveNumber: markerIndex, pointNumber: 0}]);
    check(chart.querySelector(".hoverlayer").textContent.includes("Example event"), "generic hover missing");
    Plotly.Fx.unhover(chart);

    // Navigate while removing an instance; a response carrying its old result
    // must never put the removed result back on the chart.
    const beforePan = calls.length;
    await Plotly.relayout(chart, {"xaxis.range": [2099.5, 2149.5]});
    await until(() => calls.length > beforePan, "distant pan did not request bars");
    const request = calls.at(-1);
    mount(fixtures.removed);
    await until(() => chart.data.length === 2, "remove during navigation ignored");
    const old = structuredClone(fixtures.earlier);
    old.request_id = request.request_id;
    mount(old);
    await until(() => close(chart.layout.xaxis.range[0], 2099.5), "navigation response was lost");
    check(chart.data.length === 2, "old navigation restored removed logic");

    // A new calculation can arrive with an obsolete viewport. Its bounded
    // arrays require a new display-only window, without changing current bars.
    const candles = chart.data[0].x.slice();
    const beforeRefresh = calls.length;
    const newLogic = structuredClone(fixtures.overlay);
    newLogic.overlay_revision = 3;
    mount(newLogic);
    await until(() => calls.length > beforeRefresh, "stale generic sample did not refresh its display window");
    check(chart.data[0].x.every((value, i) => close(value, candles[i])), "new calculation restored old candles");
    check(close(chart.layout.xaxis.range[0], 2099.5), "new calculation restored old zoom");
    const current = structuredClone(fixtures.earlier);
    current.overlay_revision = 3;
    current.request_id = calls.at(-1).request_id;
    mount(current);
    await until(() => chart.data.length === 5 && chart.layout.yaxis.range[1] > 180,
                "fresh generic window did not render");
    check(chart.data[2].x[0] >= fixtures.earlier.loaded_range[0], "generic line uses old source coordinates");
    mount(fixtures.removed);
    await pause(200);
    check(chart.data.length === 5, "old removal overrode newer calculation");

    const empty = structuredClone(fixtures.latest);
    empty.overlay_revision = 4;
    mount(empty);
    await until(() => chart.data.length === 2 && chart.layout.yaxis.range[1] < 110,
                "remove did not refit the visible price range");
    check(close(chart.layout.xaxis.range[0], 2099.5), "remove reset current viewport");
    cleanup();
    result.textContent = "PASS";
}
window.addEventListener("unhandledrejection", event => { result.textContent = "FAIL: " + event.reason; });
test().catch(error => { result.textContent = "FAIL: " + error.stack; });
document.querySelectorAll("script").forEach(script => script.remove());
"""


def browser_document():
    source = series()
    positions = list(range(source.total_count))
    values = [180.0] * source.total_count
    values[0], values[2971] = 1e7, None
    primitives = [
        {"type": "line", "id": "levels", "name": "Example levels", "x": positions, "y": values,
         "line": {"color": "#7c3aed", "width": 2}},
        {"type": "band", "id": "envelope", "name": "Example band", "x": positions,
         "lower": [130.0] * source.total_count, "upper": [145.0] * source.total_count,
         "fillcolor": "rgba(80,140,240,0.15)"},
        {"type": "markers", "id": "events", "name": "Example events", "x": [0, 2982], "y": [1e7, 210],
         "marker": {"symbol": "diamond", "color": "#d97706", "size": 12},
         "hovertemplate": "Example event: %{y}<extra></extra>"},
    ]
    options = {"chart_id": "generic-test", "timeframe": "1h", "timezone": "UTC"}
    fixtures = {
        "latest": window_payload(source, **options),
        "overlay": window_payload(source, **options, overlays=primitives, overlay_revision=1),
        "removed": window_payload(source, **options, overlay_revision=2),
        "earlier": window_payload(source, **options, overlays=primitives, overlay_revision=1,
                                  request={"chart_id": "generic-test", "request_id": 1,
                                           "start": 2099.5, "end": 2149.5, "width": 1000}),
    }
    encoded = json.dumps(fixtures).replace("<", "\\u003c")
    return ("<!doctype html><meta charset='utf-8'><div id='host' style='width:1000px'></div>"
            "<pre id='result'>PENDING</pre><script type='module'>" + _javascript()
            + "\nconst fixtures = " + encoded + ";\n" + INTERACTIONS + "</script>")


@pytest.mark.browser
def test_generic_logic_apply_remove_and_navigation_races(tmp_path):
    _run_browser_document(tmp_path, browser_document())


DENSE_INTERACTIONS = r"""
const result = document.getElementById("result");
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(ok, message) { if (!ok) throw new Error(message); }
async function until(test, message) {
    for (let i = 0; i < 200; i++) { if (test()) return; await pause(25); }
    throw new Error(message);
}
async function test() {
    const host = document.getElementById("host"), calls = [];
    const mount = data => render({parentElement: host, data: structuredClone(data),
                                 setStateValue: (name, value) => calls.push(value)});
    const cleanup = mount(fixtures.latest);
    const chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "initial chart did not settle");
    await pause(350);
    check(calls.length === 0, "sampled overlay caused repeated initial requests");
    await Plotly.relayout(chart, {"xaxis.range": [2970, 2980]});
    await until(() => calls.length > 0, "zoom did not refine dense overlay samples");
    const reply = structuredClone(fixtures.zoomed);
    reply.request_id = calls.at(-1).request_id;
    const count = calls.length;
    mount(reply);
    await until(() => chart.layout.xaxis.range[0] === 2970, "zoom reply was not applied");
    await pause(600);
    check(calls.length === count, "sampled overlay caused an endless identical-window refresh: " + JSON.stringify({calls, view: fixtures.zoomed.view_range, loaded: fixtures.zoomed.loaded_range}));
    check(chart.data[2].x.length <= 2048, "dense overlay samples are unbounded");
    cleanup();
    result.textContent = "PASS";
}
window.addEventListener("unhandledrejection", event => { result.textContent = "FAIL: " + event.reason; });
test().catch(error => { result.textContent = "FAIL: " + error.stack; });
document.querySelectorAll("script").forEach(script => script.remove());
"""


@pytest.mark.browser
def test_dense_primitive_samples_do_not_repeat_the_same_window_request(tmp_path):
    source = series()
    positions = [index / 100 for index in range(299901)]
    primitive = {"type": "line", "id": "dense", "x": positions, "y": [180.0] * len(positions)}
    options = {"chart_id": "dense-test", "timeframe": "1h", "timezone": "UTC", "overlays": [primitive]}
    fixtures = {
        "latest": window_payload(source, **options),
        "zoomed": window_payload(source, **options, request={"chart_id": "dense-test", "request_id": 1,
                                                            "start": 2970, "end": 2980, "width": 1000}),
    }
    assert all(item["overlays"][0]["sampled"] for item in fixtures.values())
    encoded = json.dumps(fixtures).replace("<", "\\u003c")
    document = ("<!doctype html><meta charset='utf-8'><div id='host' style='width:1000px'></div>"
                "<pre id='result'>PENDING</pre><script type='module'>" + _javascript()
                + "\nconst fixtures = " + encoded + ";\n" + DENSE_INTERACTIONS + "</script>")
    _run_browser_document(tmp_path, document)
