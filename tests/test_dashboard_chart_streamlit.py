"""Offline end-to-end chart navigation through the actual Streamlit callback."""

from datetime import UTC, datetime, timedelta
import http.client
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest
from data_pipeline.schemas import OHLCV_SCHEMA
from test_dashboard_chart_browser import _run_browser_document


STREAMLIT_INTERACTIONS = r"""
(() => {
    const result = document.createElement("pre");
    result.id = "result";
    result.textContent = "PENDING";
    document.body.appendChild(result);
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    const close = (a, b) => Math.abs(a - b) < 1e-7;
    function check(ok, message) { if (!ok) throw new Error(message); }
    async function until(test, message, timeout = 15000) {
        const deadline = performance.now() + timeout;
        while (performance.now() < deadline) { if (test()) return; await pause(25); }
        throw new Error(message + ": " + document.body.textContent.slice(-2500));
    }
    async function test() {
        await until(() => document.querySelector(".js-plotly-plot")?.layout?.yaxis?.autorange === false,
                    "actual Streamlit chart did not initialize");
        const chart = document.querySelector(".js-plotly-plot");
        const end = chart.layout.xaxis.range[1];
        check(end > 8000, "synthetic two-year dataset was not fully indexed");
        check(close(chart.layout.xaxis.range[0], end - 50), "initial Streamlit view is not latest 50");
        check(chart.data[0].x.length < 500, "initial Streamlit payload contains all candles");
        check(window.Plotly, "separately mounted Plotly runtime is unavailable");
        function button(label) {
            const found = [...document.querySelectorAll("button")]
                .find(item => item.textContent.trim() === label);
            check(found, "missing chart control: " + label);
            return found;
        }
        function verify() {
            check(document.querySelector(".js-plotly-plot") === chart,
                  "Streamlit rerun replaced chart DOM rather than updating it");
            check(document.querySelectorAll(".js-plotly-plot").length === 1,
                  "Streamlit callback leaked chart DOM nodes");
            check(chart.data[0].x.length <= 2000, "Streamlit response exceeds bounded window");
            check(chart.layout.yaxis.range.every(Number.isFinite) && chart.layout.yaxis2.range.every(Number.isFinite),
                  "Streamlit response has invalid vertical scales");
            check(chart.layout.xaxis.range.every((value, index) => close(value, chart.layout.xaxis2.range[index])),
                  "price and volume ranges differ after Streamlit callback");
            check(chart.querySelectorAll(".barlayer .bartext").length === 0,
                  "volume bars print hover text inside the plot");
            const tickLabels = [...chart.querySelectorAll(".xaxislayer-above .x2tick text")];
            const title = chart.querySelector(".g-x2title");
            check(tickLabels.length && title, "bottom timestamps or axis title are missing");
            const titleBox = title.getBoundingClientRect();
            check(tickLabels.every(label => label.getBoundingClientRect().bottom <= titleBox.top),
                  "bottom timestamps overlap the x-axis title");
            check(titleBox.bottom <= chart.getBoundingClientRect().bottom,
                  "bottom axis title extends beyond the chart");
        }
        for (let round = 0; round < 2; round++) {
            button("All history").click();
            await until(() => close(chart.layout.xaxis.range[0], -0.5)
                && chart.data[0].x[0] < 50 && chart.data[0].x.length <= 1000,
                "All history did not complete a server round trip");
            check(chart.parentElement.textContent.includes("display only"),
                  "server overview does not disclose display grouping");
            verify();
            if (round === 0) {
                const initialCount = chart.data[0].x.length;
                check(initialCount > 300, "wide overview did not retain available detail");
                chart.parentElement.style.width = "300px";
                await until(() => chart._fullLayout.width === 300 && chart.data[0].x.length <= 300,
                            "narrow resize did not request a smaller display resolution");
                const narrowCount = chart.data[0].x.length;
                verify();
                chart.parentElement.style.width = "1000px";
                await until(() => chart._fullLayout.width === 1000 && chart.data[0].x.length > narrowCount,
                            "wide resize did not restore more detailed display candles");
                check(chart.data[0].x.length === initialCount, "wide resize accumulated old window candles");
                verify();
                chart.parentElement.style.width = "";
            }
            button("Latest 50").click();
            await until(() => close(chart.layout.xaxis.range[0], end - 50)
                && chart.data[0].x[0] > end - 500,
                "Latest 50 did not restore original bars through the server");
            check(chart.parentElement.textContent.includes("Original 1h bars"),
                  "original resolution was not restored");
            verify();
        }
        chart.parentElement.style.width = "200px";
        await until(() => chart._fullLayout.width === 200, "fractional-range test did not resize");
        const controller = chart.parentElement.__quantChart;
        const originalUpdate = controller.update;
        let replies = 0;
        controller.update = (...args) => { replies++; return originalUpdate(...args); };
        // This 200-position span intersects 201 source bars. It must remain
        // grouped at a 200-candle budget rather than trigger endless refetches.
        await Plotly.relayout(chart, {"xaxis.range": [2000, 2200]});
        await until(() => close(chart.layout.xaxis.range[0], 2000)
            && close(chart.layout.xaxis.range[1], 2200)
            && chart.parentElement.textContent.includes("display only")
            && !chart.parentElement.textContent.includes("Loading visible period"),
            "fractional range did not finish its grouped server response");
        const settledReplies = replies;
        await pause(650);
        check(replies === settledReplies, "fractional range repeatedly refetched the same grouped window");
        check(!chart.parentElement.textContent.includes("Loading visible period"),
              "fractional range remained in a refetch loop");
        controller.update = originalUpdate;
        chart.parentElement.style.width = "";
        await until(() => chart._fullLayout.width > 500
            && chart.parentElement.textContent.includes("Original 1h bars")
            && !chart.parentElement.textContent.includes("Loading visible period"),
            "wider chart did not restore original detail after fractional range");
        button("Latest 50").click();
        await until(() => close(chart.layout.xaxis.range[0], end - 50)
            && chart.data[0].x[0] > end - 500,
            "latest bars did not restore after fractional range");
        verify();
        await Plotly.relayout(chart, {"xaxis.range": [999.5, 1049.5]});
        await until(() => close(chart.layout.xaxis.range[0], 999.5)
            && chart.data[0].x[0] < 1000 && chart.data[0].x.at(-1) < 1300,
            "earlier pan did not load and replace the server window");
        verify();
        button("Latest 50").click();
        await until(() => close(chart.layout.xaxis.range[0], end - 50), "latest bars did not restore for trend analysis");
        const trendRange = [end - 20, end];
        await Plotly.relayout(chart, {"xaxis.range": trendRange});
        await until(() => close(chart.layout.xaxis.range[0], trendRange[0]), "trend local zoom did not finish");
        function toggleTrends() {
            const checkbox = document.querySelector('input[aria-label="Analyze trends"]');
            check(checkbox, "missing Analyze trends checkbox");
            checkbox.click();
        }
        function boundaryControl() {
            return [...document.querySelectorAll('[data-testid="stSelectbox"]')]
                .find(item => item.textContent.includes("Boundary matching rule"));
        }
        const selectedBoundaryRule = () => boundaryControl()?.querySelector('[role="combobox"]')?.value;
        toggleTrends();
        await until(() => [...document.querySelectorAll('button')].some(item => item.textContent.trim() === "Apply and analyze"),
                    "trend configuration form did not open");
        await until(() => document.querySelector('input[aria-label="Maximum lookback per endpoint (bars)"]')?.value === "8",
                    "saved configuration did not load");
        await until(() => Number(document.querySelector('input[aria-label="Redundancy tolerance (%)"]')?.value) === 25
            && selectedBoundaryRule() === "Either boundary (OR)",
            "saved redundancy percentage and boundary rule did not load");
        button("Apply and analyze").click();
        await until(() => chart.data?.length > 2 && document.body.textContent.includes("Analysis range:"),
                    "trend form did not produce a chart overlay and result table", 90000);
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])), "analysis reset local zoom");
        check(document.querySelector('[data-testid="stDataFrame"]'), "trend result table is missing");
        const visibleTraces = () => chart.data.slice(2).filter(trace => trace.x.length);
        await until(() => visibleTraces().length > 1 && visibleTraces().every(trace =>
            trace.x[0] >= trendRange[0] && trace.x.at(-1) <= trendRange[1]),
            "latest historical overlays were not sampled for the visible range");
        const curveNumber = chart.data.findIndex((trace, index) => index > 1 && trace.x.length);
        const trendTrace = chart.data[curveNumber];
        check(visibleTraces().every(trace => trace.x.at(-1) <= end - 0.5),
              "historical fit extends past its observed data");
        check(new Set(visibleTraces().map(trace => trace.x.at(-1))).size > 1,
              "historical chart still contains only one final endpoint");
        const intervals = visibleTraces().map(trace => [trace.x[0], trace.x.at(-1)]);
        check(intervals.some((interval, index) => intervals.slice(index + 1).some(other =>
            Math.min(interval[1], other[1]) > Math.max(interval[0], other[0]))),
            "nearby-boundary grouping still suppressed all overlapping intervals");
        check(visibleTraces().every(trace => trace.meta.lambda_values.length > 0
            && trace.meta.primary_lambda === Math.min(...trace.meta.lambda_values)),
            "historical overlays lost segmentation lambda provenance");
        Plotly.Fx.hover(chart, [{curveNumber, pointNumber: Math.floor(trendTrace.x.length / 2)}]);
        check(chart.querySelector(".hoverlayer").textContent.includes("Clarity:")
            && chart.querySelector(".hoverlayer").textContent.includes("Color λ:"),
            "actual dashboard trend hover lacks metrics or lambda labels");
        Plotly.Fx.unhover(chart);
        button("All history").click();
        await until(() => close(chart.layout.xaxis.range[0], -0.5)
            && chart.data[0].x.length <= 1000 && visibleTraces().length > 100,
            "grouped all-history chart did not render its historical trend intervals");
        check(visibleTraces().every(trace => trace.x[0] >= 0 && trace.x.at(-1) <= end - 0.5),
              "all-history trend fit extends outside its observed source bars");
        check(visibleTraces().every(trace => trace.x.length <= 768), "all-history trend traces exceeded their point budget");
        verify();
        await Plotly.relayout(chart, {"xaxis.range": [999.5, 1049.5]});
        await until(() => close(chart.layout.xaxis.range[0], 999.5)
            && chart.data[0].x[0] < 1000 && chart.data[0].x.at(-1) < 1300
            && visibleTraces().length > 1 && visibleTraces().every(trace =>
                trace.x[0] >= 999.5 && trace.x.at(-1) <= 1049.5),
            "historical navigation did not load earlier trend intervals");
        check(new Set(visibleTraces().map(trace => trace.x.at(-1))).size > 1,
              "earlier chart shows only a single endpoint trend");
        check(visibleTraces().every(trace => trace.x[0] >= 999.5 && trace.x.at(-1) <= 1049.5),
              "historical overlays were not clipped to the visible source coordinates");
        verify();
        button("Latest 50").click();
        await until(() => close(chart.layout.xaxis.range[0], end - 50)
            && visibleTraces().length > 1, "latest historical trends did not restore");
        await Plotly.relayout(chart, {"xaxis.range": trendRange});
        await until(() => close(chart.layout.xaxis.range[0], trendRange[0]), "local trend zoom did not restore");
        toggleTrends();
        await until(() => chart.data?.length === 2, "checkbox did not remove trend overlays");
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])), "checkbox reset local zoom");
        toggleTrends();
        await until(() => chart.data?.length > 2, "re-enabled analysis did not reuse its matching result");
        check(Number(document.querySelector('input[aria-label="Redundancy tolerance (%)"]')?.value) === 25
            && selectedBoundaryRule() === "Either boundary (OR)",
            "re-enabled analysis lost its applied redundancy settings");
        button("Restore trend defaults").click();
        await until(() => chart.data?.length === 2
            && document.querySelector('input[aria-label="Maximum lookback per endpoint (bars)"]')?.value === "2000"
            && Number(document.querySelector('input[aria-label="Redundancy tolerance (%)"]')?.value) === 5
            && selectedBoundaryRule() === "Both boundaries (AND)",
            "restoring defaults did not invalidate the old result and restore form values");
        verify();
        check(!document.querySelector('[data-testid="stException"]'), "Streamlit raised an application exception");
        result.textContent = "PASS";
    }
    test().catch(error => { result.textContent = "FAIL: " + error.stack; });
})();
"""


@pytest.mark.browser
def test_chart_streamlit_window_round_trips(tmp_path):
    """Exercise component registration/runtime loading and real session state."""
    stamps = []
    day = datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York"))
    while day.year < 2026:
        if day.weekday() < 5:
            stamps.extend((day + timedelta(hours=hour)).astimezone(UTC) for hour in range(4, 20))
        day += timedelta(days=1)
    count = len(stamps)
    prices = [100.0 + (index % 97) / 10 for index in range(count)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * count,
                          "open": prices, "close": [price + 0.1 for price in prices],
                          "low": [price - 0.2 for price in prices],
                          "high": [price + 0.3 for price in prices],
                          "volume": [100.25 + index for index in range(count)]}, schema=OHLCV_SCHEMA)
    root = tmp_path / "market-data"
    DataPipeline(root).store.write_raw(
        DataRequest("TEST", stamps[0], stamps[-1] + timedelta(hours=1), timeframe="1h"), frame)
    app = tmp_path / "chart_app.py"
    from dashboard.trend_settings import load_trend_settings, save_trend_settings
    from trend_analysis import TrendConfig

    settings_path = tmp_path / "trend_config.json"
    save_trend_settings(TrendConfig(max_history=8, redundancy_percentage=25.0,
                                   redundancy_boundary_rule="or"), settings_path)
    app.write_text(
        "import streamlit as st\nfrom dashboard.app import _data_page\nfrom research import Research\n"
        "from pathlib import Path\nimport dashboard.trend_settings as settings\n"
        f"settings.DEFAULT_SETTINGS_PATH = Path({str(settings_path)!r})\n"
        "st.set_page_config(layout='wide')\n"
        f"_data_page(Research({str(root)!r}, runs_dir={str(tmp_path / 'runs')!r}))\n",
        encoding="utf-8",
    )
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    with (tmp_path / "streamlit.log").open("w+") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(app), "--server.headless=true",
            "--server.address=127.0.0.1", f"--server.port={port}", "--browser.gatherUsageStats=false",
        ], cwd=tmp_path, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            ready = False
            while time.monotonic() < deadline and process.poll() is None:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    connection.request("GET", "/_stcore/health")
                    response = connection.getresponse()
                    ready = response.status == 200
                    response.read()
                except OSError:
                    pass
                finally:
                    connection.close()
                if ready:
                    break
                time.sleep(0.05)
            log.seek(0)
            assert ready, log.read()
            _run_browser_document(tmp_path, url=f"http://127.0.0.1:{port}",
                                  browser_js=STREAMLIT_INTERACTIONS,
                                  screenshot_path=tmp_path / "chart-window.png", timeout_seconds=120)
            assert load_trend_settings(settings_path).config == TrendConfig()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
