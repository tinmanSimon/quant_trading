"""Finding commands cross unloaded windows without replay or stale-result races."""

import json

import pytest

from dashboard.charts.component import _javascript, window_payload
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
    let cleanup = mount(fixtures.latest);
    let chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "initial fit did not settle");
    const width = fixtures.zoom_width;
    await Plotly.relayout(chart, {"xaxis.range": [2800 - width / 2, 2800 + width / 2]});
    await pause(250);
    if (calls.length) {
        const zoomReply = structuredClone(fixtures.zoom_reply);
        zoomReply.request_id = calls.at(-1).request_id;
        mount(zoomReply);
    }
    await until(() => close(chart.layout.xaxis.range[0], 2800 - width / 2), "user zoom did not settle");
    const zoomCalls = calls.length;
    mount(fixtures.point);
    await until(() => calls.length > zoomCalls, "finding outside loaded bars did not request a window");
    check(close(calls.at(-1).start, 2000 - width / 2) && close(calls.at(-1).end, 2000 + width / 2), "finding was not centered");
    check(chart.layout.xaxis.range[0] > 2600, "loading a finding showed an empty chart");
    const firstReply = structuredClone(fixtures.point_reply);
    firstReply.request_id = calls.at(-1).request_id;
    mount(firstReply);
    await until(() => close(chart.layout.xaxis.range[0], 2000 - width / 2), "finding response was not displayed");
    check(["xaxis2", "xaxis3"].every(axis => chart.layout[axis].range.every((x,i) => close(x, chart.layout.xaxis.range[i]))),
          "finding did not synchronize every chart panel");
    check(chart.layout.shapes.some(shape => shape.name === "analysis-navigation-target" && shape.x0 === 2000),
          "selected finding was not highlighted");

    // Rerenders must not replay the last finding or overwrite a user's pan.
    await Plotly.relayout(chart, {"xaxis.range": [1997, 2003]});
    await until(() => close(chart.layout.xaxis.range[0], 1997), "local pan did not settle");
    const before = calls.length;
    mount(fixtures.point);
    const badDataset = structuredClone(fixtures.point);
    badDataset.navigation_command.sequence = 2;
    badDataset.navigation_command.chart_id = "other-dataset";
    mount(badDataset);
    const badResult = structuredClone(fixtures.point);
    badResult.navigation_command.sequence = 2;
    badResult.navigation_command.result_id = "other-result";
    mount(badResult);
    await pause(250);
    check(close(chart.layout.xaxis.range[0], 1997) && calls.length === before, "duplicate/stale command changed the view");

    mount(fixtures.interval);
    await until(() => calls.length > before, "interval did not request an earlier window");
    const intervalRequest = calls.at(-1);
    check(close(intervalRequest.start, 547) && close(intervalRequest.end, 553), "interval navigation changed the user zoom");
    mount(firstReply); // Old viewport reply and old selection arrive in flight.
    await pause(100);
    const intervalReply = structuredClone(fixtures.interval_reply);
    intervalReply.request_id = intervalRequest.request_id;
    mount(intervalReply);
    await until(() => close(chart.layout.xaxis.range[0], 547) && close(chart.layout.xaxis.range[1], 553),
                "interval response was not displayed");
    check(chart.layout.shapes.some(shape => shape.type === "rect" && shape.x0 === 499.6 && shape.x1 === 600.4),
          "interval selection did not highlight its extent");
    check(chart.layout.yaxis.range.every(Number.isFinite), "navigation produced invalid vertical scaling");

    // A new result clears old highlights. A late older result cannot restore them.
    const removed = structuredClone(intervalReply);
    removed.overlay_revision = 2;
    removed.navigation_command = null;
    removed.navigation_result_id = null;
    mount(removed);
    await until(() => !chart.layout.shapes.some(shape => shape.name === "analysis-navigation-target"),
                "new results retained a stale selection highlight");
    mount(firstReply);
    await pause(200);
    check(chart.layout.xaxis.range[1] < 1000, "late result restored old finding");
    cleanup();

    // Persisted acknowledgements prevent a component remount from replaying a
    // command that was already applied before the user manually panned.
    cleanup = mount(removed);
    chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "remount did not settle");
    const beforeRemountCommand = calls.length;
    mount(intervalReply);
    await pause(250);
    check(calls.length === beforeRemountCommand, "remount replayed an acknowledged command");
    cleanup();
    result.textContent = "PASS";
}
window.addEventListener("unhandledrejection", event => { result.textContent = "FAIL: " + event.reason; });
test().catch(error => { result.textContent = "FAIL: " + error.stack; });
document.querySelectorAll("script").forEach(script => script.remove());
"""


@pytest.mark.browser
@pytest.mark.parametrize("zoom_width", [10, 250])
def test_findings_fetch_original_bars_and_preserve_navigation_revisions(tmp_path, zoom_width):
    source = series()
    overlay = {"type": "line", "id": "rsi", "x": list(range(source.total_count)),
               "y": [50.] * source.total_count, "panel": {"id": "rsi", "title": "RSI", "range": [0, 100]}}
    options = {"chart_id": "analysis-navigation", "timeframe": "1h", "timezone": "UTC",
               "overlays": [overlay], "overlay_revision": 1}
    point = {"chart_id": options["chart_id"], "result_id": "result-a", "sequence": 1,
             "event_id": "entries:1", "label": "MA crossing", "source_index": 2000, "end_index": 2000}
    interval = dict(point, sequence=2, event_id="interval:1", source_index=500, end_index=600)
    fixtures = {
        "zoom_width": zoom_width,
        "latest": window_payload(source, **options,
                                 request={"chart_id": options["chart_id"], "request_id": 0,
                                          "start": 2399.5, "end": 2999.5, "width": 1000}),
        "zoom_reply": window_payload(source, **options,
                                     request={"chart_id": options["chart_id"], "request_id": 1,
                                              "start": 2800 - zoom_width / 2, "end": 2800 + zoom_width / 2,
                                              "width": 1000}),
        "point": window_payload(source, **options, navigation_command=point),
        "point_reply": window_payload(source, **options, navigation_command=point,
                                      request={"chart_id": options["chart_id"], "request_id": 1,
                                               "start": 2000 - zoom_width / 2, "end": 2000 + zoom_width / 2, "width": 1000}),
        "interval": window_payload(source, **options, navigation_command=interval),
        "interval_reply": window_payload(source, **options, navigation_command=interval,
                                         request={"chart_id": options["chart_id"], "request_id": 2,
                                                  "start": 547, "end": 553, "width": 1000}),
    }
    encoded = json.dumps(fixtures).replace("<", "\\u003c")
    document = ("<!doctype html><meta charset='utf-8'><div id='host' style='width:1000px'></div>"
                "<pre id='result'>PENDING</pre><script type='module'>" + _javascript()
                + "\nconst fixtures = " + encoded + ";\n" + INTERACTIONS + "</script>")
    _run_browser_document(tmp_path, document)
