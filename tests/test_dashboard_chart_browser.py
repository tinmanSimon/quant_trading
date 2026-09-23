"""Offline tests of the shipped component in real Chrome, including DOM gestures.

Run with: python -m pytest tests/test_dashboard_chart_browser.py --run-browser
Only temporary HTML and synthetic prices are used; no local store is opened.
"""

from datetime import UTC, datetime, timedelta
import json
import re
import shutil
import subprocess
import time

import polars as pl
import pytest
from websockets.sync.client import connect

from dashboard.chart_component import _javascript
from dashboard.charts import CHART_CONFIG, price_chart


INTERACTIONS = r"""
const result = document.getElementById("result");
let asynchronousError;
function check(ok, message) { if (!ok) throw new Error(message); }
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(test, message) {
    for (let i = 0; i < 100; i++) { if (test()) return; await pause(25); }
    throw new Error(message);
}
function close(a, b) { return Math.abs(a - b) < 1e-7; }
function aligned(chart) {
    return chart.layout.xaxis.range.every((value, i) => close(value, chart.layout.xaxis2.range[i]));
}
function priceLabelsFit(chart) {
    const bounds = chart.getBoundingClientRect();
    const labels = [...chart.querySelectorAll(".yaxislayer-above .ytick text")];
    check(labels.length >= 2, "price ticks are missing");
    const title = chart.querySelector(".g-ytitle").getBoundingClientRect();
    const boxes = labels.map(label => label.getBoundingClientRect()).sort((a, b) => a.top - b.top);
    for (let i = 0; i < boxes.length; i++) {
        check(boxes[i].left >= bounds.left && boxes[i].right <= bounds.right, "price digits clipped at chart edge");
        check(boxes[i].left >= title.right, "price ticks overlap the axis title");
        if (i) check(boxes[i].top >= boxes[i - 1].bottom, "price ticks overlap each other");
    }
}
function drag(element, from, to) {
    element.dispatchEvent(new MouseEvent("mousedown", {bubbles: true, clientX: from[0], clientY: from[1], buttons: 1}));
    const capture = document.querySelector(".dragcover") || document;
    capture.dispatchEvent(new MouseEvent("mousemove", {bubbles: true, clientX: to[0], clientY: to[1], buttons: 1}));
    capture.dispatchEvent(new MouseEvent("mouseup", {bubbles: true, clientX: to[0], clientY: to[1]}));
}
async function test() {
    const host = document.getElementById("host");
    const cleanup = render({parentElement: host, data: payload});
    const chart = host.firstElementChild;
    await until(() => chart.layout?.yaxis?.autorange === false, "initial scale not fitted");
    const count = payload.figure.data[0].x.length;
    check(chart.data[0].x.every((value, i) => value === i), "bars are not consecutive");
    check(chart.querySelectorAll(".cartesianlayer .trace.boxes .box").length >= Math.min(count, 50), "candles not rendered");
    check(aligned(chart), "initial volume/price axes differ");
    check(chart.layout.yaxis.range.every(Number.isFinite), "invalid initial price range");
    check(chart.layout.yaxis2.range[1] > 0, "zero-volume range collapsed");
    check(chart.layout.yaxis.range[1] > chart.layout.yaxis.range[0], "flat price range collapsed");
    priceLabelsFit(chart);
    if (count === 1) {
        check([...chart.querySelectorAll(".ytick text")].some(label => label.textContent === "13.5"),
              "13.5 is not shown as a complete price label");
    }
    if (count > 50) {
        check(close(chart.layout.xaxis.range[0], count - 50.5) && close(chart.layout.xaxis.range[1], count - 0.5),
              "initial window does not contain the latest 50 bars");
        check(chart.layout.yaxis.range[1] < 20, "old outlier affected initial price scale");
        check(chart.layout.yaxis2.range[1] < 1000, "old volume affected initial volume scale");
        check(chart.data[0].x.length === count, "older bars were discarded");
        Plotly.Fx.hover(chart, [{curveNumber: 0, pointNumber: count - 1}]);
        check(chart.querySelector(".hoverlayer").textContent.includes("Volume: 100.25"), "candle hover omits volume");
        for (let i = 0; i < 2; i++) {
            const before = chart.layout.xaxis.range[0];
            chart.querySelector('[data-title="Zoom out"]').dispatchEvent(new MouseEvent("click", {bubbles: true}));
            await until(() => chart.layout.xaxis.range[0] < before && aligned(chart), "zoom out did not reveal older bars");
        }
        await until(() => close(chart.layout.xaxis.range[0], -0.5) && chart.layout.yaxis.range[1] > 10000,
                    "full history unavailable after zoom out");
        priceLabelsFit(chart);
        chart.querySelector('[data-title="Reset axes"]').dispatchEvent(new MouseEvent("click", {bubbles: true}));
        await until(() => close(chart.layout.xaxis.range[0], count - 50.5) && chart.layout.yaxis.range[1] < 20,
                    "reset did not restore the latest 50 bars");
        host.style.width = "360px";
        await new Promise(requestAnimationFrame);
        await until(() => chart._fullLayout.width === 360, "narrow layout did not resize");
        priceLabelsFit(chart);
    } else if (count > 1) {
        const initialWidth = chart.querySelector(".cartesianlayer .trace.boxes .box").getBBox().width;
        const initialPriceSpan = chart.layout.yaxis.range[1] - chart.layout.yaxis.range[0];
        // Actual modebar click exercises Plotly's zoom event, not just our math.
        chart.querySelector('[data-title="Zoom in"]').dispatchEvent(new MouseEvent("click", {bubbles: true}));
        await until(() => chart.layout.yaxis.range[1] < 200, "zoom did not exclude remote outliers");
        check(chart.querySelector(".cartesianlayer .trace.boxes .box").getBBox().width > initialWidth, "candles did not widen");
        check(aligned(chart), "zoom split the two panels");
        check(chart.layout.yaxis2.range[1] < 1000, "volume still scaled to off-screen bar");
        check(chart.layout.yaxis.range[1] - chart.layout.yaxis.range[0] < initialPriceSpan, "price did not fit");

        // A drag on the plot zooms only horizontally (price is auto-fitted).
        const plot = chart.querySelector(".nsewdrag");
        const rect = plot.getBoundingClientRect();
        const widthBeforeDrag = chart.layout.xaxis.range[1] - chart.layout.xaxis.range[0];
        drag(plot, [rect.x + rect.width * 0.25, rect.y + rect.height * 0.4],
                   [rect.x + rect.width * 0.75, rect.y + rect.height * 0.6]);
        await until(() => chart.layout.xaxis.range[1] - chart.layout.xaxis.range[0] < widthBeforeDrag,
                    "drag zoom did not change range");

        const beforeWheel = chart.layout.xaxis.range[1] - chart.layout.xaxis.range[0];
        plot.dispatchEvent(new WheelEvent("wheel", {bubbles: true, cancelable: true, deltaY: -100,
            clientX: rect.x + rect.width / 2, clientY: rect.y + rect.height / 2}));
        await until(() => chart.layout.xaxis.range[1] - chart.layout.xaxis.range[0] < beforeWheel,
                    "wheel zoom did not change range");

        // The lower axis is controlled by the range slider; both range event
        // encodings must update the same candle window and price/volume scales.
        await Plotly.relayout(chart, {"xaxis2.range": [4.5, 9.5]});
        await until(() => close(chart.layout.xaxis.range[0], 4.5) && aligned(chart), "slider range not synchronized");
        check(chart.layout.yaxis.range[0] < 104 && chart.layout.yaxis.range[1] > 110, "visible OHLC clipped");
        check(chart.layout.xaxis2.ticktext.every(text => text.includes("2025-")), "timestamps lost on zoom");
        const handle = chart.querySelector(".rangeslider-grabarea-min");
        const handleRect = handle.getBoundingClientRect();
        const beforeSlider = chart.layout.xaxis.range[0];
        drag(handle, [handleRect.x + handleRect.width / 2, handleRect.y + handleRect.height / 2],
                     [handleRect.x + handleRect.width / 2 + 35, handleRect.y + handleRect.height / 2]);
        await until(() => chart.layout.xaxis.range[0] > beforeSlider && aligned(chart), "slider gesture did not fit both panels");
        Plotly.Fx.hover(chart, [{curveNumber: 0, pointNumber: 8}]);
        check(chart.querySelector(".hoverlayer").textContent.includes(payload.figure.layout.meta.bar_labels[8]),
              "hover does not show the real bar time");
        check(chart.querySelector(".hoverlayer").textContent.includes("Volume: 100.25"), "candle hover omits volume");
        await Plotly.relayout(chart, {"xaxis.range[0]": 6.5, "xaxis.range[1]": 11.5});
        await until(() => close(chart.layout.xaxis2.range[0], 6.5), "pan event not synchronized");

        // Actual pan gesture and bounds clamping keep a window on stored bars.
        chart.querySelector('[data-title="Pan"]').dispatchEvent(new MouseEvent("click", {bubbles: true}));
        const beforePan = chart.layout.xaxis.range[0];
        drag(plot, [rect.x + rect.width * 0.6, rect.y + rect.height / 2],
                   [rect.x + rect.width * 0.5, rect.y + rect.height / 2]);
        await until(() => !close(chart.layout.xaxis.range[0], beforePan) && aligned(chart), "pan gesture not handled");
        await Plotly.relayout(chart, {"xaxis.range": [100, 102]});
        await until(() => chart.layout.xaxis.range[1] <= count - 0.5 && aligned(chart), "pan escaped stored bars");
        await Plotly.relayout(chart, {"xaxis.range": [7.1, 7.2]});
        await until(() => chart.layout.xaxis.range[1] - chart.layout.xaxis.range[0] >= 1, "sub-bar zoom not bounded");
        const rangeBeforeResize = chart.layout.xaxis.range.slice();
        host.style.width = "600px";
        await new Promise(requestAnimationFrame);
        await until(() => chart._fullLayout.width === 600, "chart did not resize");
        check(chart.layout.xaxis.range.every((v, i) => close(v, rangeBeforeResize[i])), "resize lost zoom");
        check(aligned(chart), "resize split panels");
        priceLabelsFit(chart);

        chart.querySelector('[data-title="Reset axes"]').dispatchEvent(new MouseEvent("click", {bubbles: true}));
        await until(() => close(chart.layout.xaxis.range[0], -0.5) && close(chart.layout.xaxis.range[1], count - 0.5),
                    "reset did not restore full window");
        await until(() => chart.layout.yaxis.range[1] > 10000, "reset did not restore full price scale");
        await Plotly.relayout(chart, {"xaxis.range": [4.5, 9.5]});
        await until(() => chart.layout.yaxis.range[1] < 200, "second zoom did not finish");
        const resetPlot = chart.querySelector(".nsewdrag");
        const resetRect = resetPlot.getBoundingClientRect();
        const center = [resetRect.x + resetRect.width / 2, resetRect.y + resetRect.height / 2];
        drag(resetPlot, center, center);
        await pause(50);
        drag(resetPlot, center, center);
        await until(() => close(chart.layout.xaxis.range[0], -0.5) && close(chart.layout.xaxis.range[1], count - 0.5),
                    "double-click did not reset window");
    }
    cleanup();
    check(host.children.length === 0, "component did not clean up");
    // Switching to a one-bar dataset must not retain the old window/scales.
    const one = structuredClone(payload);
    for (const trace of one.figure.data) {
        for (const key of ["x", "open", "high", "low", "close", "text", "y", "customdata"])
            if (Array.isArray(trace[key])) trace[key] = trace[key].slice(0, 1);
    }
    one.figure.layout.meta.bar_labels = one.figure.layout.meta.bar_labels.slice(0, 1);
    for (const axis of ["xaxis", "xaxis2"]) {
        one.figure.layout[axis].range = [-0.5, 0.5];
        one.figure.layout[axis].maxallowed = 0.5;
    }
    const cleanupOne = render({parentElement: host, data: one});
    await until(() => host.firstElementChild.layout?.yaxis?.autorange === false, "replacement did not render");
    check(close(host.firstElementChild.layout.xaxis.range[1], 0.5), "dataset change kept old window");
    cleanupOne();
    await pause(250); // Let pending Plotly resize/redraw work finish after cleanup.
    check(!asynchronousError, "late chart error: " + asynchronousError);
    result.textContent = "PASS";
}
window.addEventListener("unhandledrejection", event => {
    asynchronousError = event.reason?.stack || String(event.reason);
    result.textContent = "FAIL: " + asynchronousError;
});
test().catch(error => { result.textContent = "FAIL: " + error.stack; });
document.querySelectorAll("script").forEach(script => script.remove());
"""


def browser_document(*, single=False, latest=False):
    size = 120 if latest else 1 if single else 24
    # Skip nights and weekends; compression depends on stored positions only.
    stamps = [datetime(2025, 1, 3, 14, tzinfo=UTC) + timedelta(days=i // 6 * 3, hours=i % 6)
              for i in range(size)]
    prices = [13.5] if single else [10000.0] + [100.0 + i for i in range(1, size - 1)] + [20000.0]
    if latest:
        prices = [10000.0] + [13.5 + i * 0.001 for i in range(1, size)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * size,
                          "open": prices, "close": prices,
                          "low": prices if single else [p - 1 for p in prices],
                          "high": prices if single else [p + 1 for p in prices],
                          "volume": [0.0] if single else [1e6] + [100.25] * (size - 1)})
    figure = price_chart(frame, timeframe="1h")
    payload = json.dumps({"figure": json.loads(figure.to_json()), "config": CHART_CONFIG}).replace("<", "\\u003c")
    return ("<!doctype html><meta charset='utf-8'><div id='host' style='width:1000px'></div>"
            "<pre id='result'>PENDING</pre><script type='module'>" + _javascript()
            + "\nconst payload = " + payload + ";\n" + INTERACTIONS + "</script>")


@pytest.mark.browser
@pytest.mark.parametrize("single,latest", [(False, False), (True, False), (False, True)],
                         ids=["zoom-pan-resize-reset", "single-flat-zero-volume", "latest-50-with-full-history"])
def test_chart_browser_interactions(tmp_path, single, latest):
    browser = next((found for name in ("google-chrome", "chromium", "chromium-browser")
                    if (found := shutil.which(name))), None)
    assert browser, "Install Chrome/Chromium to run --run-browser tests."
    page = tmp_path / "chart.html"
    page.write_text(browser_document(single=single, latest=latest), encoding="utf-8")
    # Real time (rather than Chrome's virtual-time dump mode) is necessary for
    # animation frames and ResizeObserver. Drive CDP using an existing dependency.
    with (tmp_path / "chrome.log").open("w+") as log:
        process = subprocess.Popen([
            browser, "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
            "--no-first-run", "--disable-background-networking", "--disable-extensions",
            f"--user-data-dir={tmp_path / 'chrome'}", "--remote-debugging-port=0", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=log)
        try:
            deadline = time.monotonic() + 10
            address = None
            while time.monotonic() < deadline:
                log.seek(0)
                output = log.read()
                address = re.search(r"DevTools listening on (ws://\S+)", output)
                if address or process.poll() is not None:
                    break
                time.sleep(0.05)
            assert address, output
            with connect(address[1], proxy=None) as socket:
                sequence = 0
                session = None

                def command(method, **params):
                    nonlocal sequence
                    sequence += 1
                    request = {"id": sequence, "method": method, "params": params}
                    if session:
                        request["sessionId"] = session
                    socket.send(json.dumps(request))
                    while True:
                        response = json.loads(socket.recv(timeout=20))
                        if response.get("id") == sequence:
                            assert "error" not in response, response
                            return response["result"]

                target = command("Target.createTarget", url="about:blank")["targetId"]
                session = command("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]
                command("Page.navigate", url=page.as_uri())
                deadline = time.monotonic() + 15
                result = "PENDING"
                while time.monotonic() < deadline:
                    state = command("Runtime.evaluate", expression='document.getElementById("result")?.textContent',
                                    returnByValue=True)
                    result = state.get("result", {}).get("value", "PENDING")
                    if result != "PENDING":
                        break
                    time.sleep(0.05)
                if result != "PASS":
                    html = command("Runtime.evaluate", expression="document.documentElement.outerHTML", returnByValue=True)
                    (tmp_path / "rendered.html").write_text(html["result"]["value"], encoding="utf-8")
                assert result == "PASS", result
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
