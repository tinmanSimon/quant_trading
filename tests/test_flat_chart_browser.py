"""The flat summary draws separate periods and navigates their full extents."""

from datetime import UTC, datetime, timedelta
import json

import polars as pl
import pytest

from dashboard.charts.component import _javascript, window_payload
from dashboard.charts.series import ChartSeries
from research.analysis import LogicContext
from research.analysis.builtins.flat import FlatLogic
from test_dashboard_chart_browser import _run_browser_document


INTERACTIONS = r"""
const result = document.getElementById("result");
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(ok, message) { if (!ok) throw new Error(message); }
async function until(test, message) {
    for (let i = 0; i < 150; i++) { if (test()) return; await pause(25); }
    throw new Error(message);
}
async function test() {
    const host = document.getElementById("host");
    const mount = data => render({parentElement: host, data: structuredClone(data),
                                 setStateValue: () => {}});
    const cleanup = mount(fixtures.initial);
    const chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "chart did not settle");
    const fits = chart.data.filter(trace => trace.meta?.primitive === "log_line");
    check(fits.length === 3 * fixtures.periods.length, "chart and period table differ");
    for (const row of fixtures.periods) {
        const fit = fits.find(trace => trace.meta.overlay_id === row.period_id);
        check(fit && fit.x[0] === row.start_index && fit.x.at(-1) === row.end_index,
              "a flat band bridges periods or misses its endpoints");
        for (const [suffix, field] of [["", "center_price"], ["-lower", "lower_price"],
                                        ["-upper", "upper_price"]]) {
            const line = fits.find(trace => trace.meta.overlay_id === row.period_id + suffix);
            check(line && line.x[0] === row.start_index && line.x.at(-1) === row.end_index,
                  "band edge misses its period boundaries");
            check(line.y.every(y => Number.isFinite(y) && Math.abs(y - row[field]) < 1e-10),
                  "band is not horizontal at the reported price");
        }
        check(fit.hovertemplate.includes("Duration: " + row.bars + " bars") &&
              fit.hovertemplate.includes("Band width:"), "flat hover lacks period details");
    }
    const boundaries = chart.data.find(trace => trace.meta?.overlay_id === "flat-boundaries");
    check(boundaries && boundaries.x.length === 2 * fixtures.periods.length, "flat boundaries are missing");
    mount(fixtures.hidden);
    await until(() => !chart.data.some(trace => trace.meta?.primitive === "log_line"),
                "hiding flat bands retained lines");
    check(chart.data.some(trace => trace.meta?.overlay_id === "flat-boundaries"),
          "hiding bands also removed boundaries");
    mount(fixtures.navigation);
    await until(() => chart.layout.shapes?.some(shape =>
        shape.name === "analysis-navigation-target" && shape.type === "rect"),
        "flat period navigation did not highlight interval");
    const target = chart.layout.shapes.find(shape => shape.name === "analysis-navigation-target");
    check(target.x0 === fixtures.target_start - .4 && target.x1 === fixtures.target_end + .4,
          "flat selection highlights the wrong bars");
    cleanup();
    result.textContent = "PASS";
}
window.addEventListener("unhandledrejection", event => { result.textContent = "FAIL: " + event.reason; });
test().catch(error => { result.textContent = "FAIL: " + error.stack; });
document.querySelectorAll("script").forEach(script => script.remove());
"""


@pytest.mark.browser
def test_flat_periods_draw_and_navigate_without_bridging(tmp_path):
    prices = [100.] * 6 + [120.] * 6 + [140.] * 6
    stamps = [datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=30 * i)
              for i in range(len(prices))]
    frame = pl.DataFrame({"timestamp": stamps, "open": prices, "close": prices,
                          "high": [p + 1 for p in prices], "low": [p - 1 for p in prices],
                          "volume": [100.] * len(prices)})
    context = LogicContext(frame, "30m", "flat-browser", stamps[-1] + timedelta(minutes=30))
    logic = FlatLogic()
    result = logic.calculate(context, {"min_duration": 3, "max_range_pct": 2.0})
    periods = result.tables[0]["rows"]
    assert [(row["start_index"], row["end_index"]) for row in periods] == [(12, 17), (6, 11), (0, 5)]
    source = ChartSeries(frame, timeframe="30m")
    options = {"chart_id": "flat-browser", "timeframe": "30m", "timezone": "UTC"}
    target = result.data["analysis"]["events"][1]
    command = {"chart_id": "flat-browser", "result_id": "flat-result", "sequence": 1,
               "event_id": target["id"], "label": target["label"],
               "source_index": target["source_index"], "end_index": target["end_index"]}
    fixtures = {
        "periods": periods,
        "target_start": target["source_index"], "target_end": target["end_index"],
        "initial": window_payload(source, **options, overlays=result.overlays, overlay_revision=1),
        "hidden": window_payload(source, **options,
                                 overlays=logic.present(result, {"show_fit": False}).overlays,
                                 overlay_revision=2),
        "navigation": window_payload(source, **options, overlays=result.overlays,
                                     overlay_revision=3, navigation_command=command),
    }
    document = ("<!doctype html><meta charset='utf-8'><div id='host' style='width:1000px'></div>"
                "<pre id='result'>PENDING</pre><script type='module'>" + _javascript()
                + "\nconst fixtures = " + json.dumps(fixtures).replace("<", "\\u003c")
                + ";\n" + INTERACTIONS + "</script>")
    _run_browser_document(tmp_path, document)
