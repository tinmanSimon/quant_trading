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
        // Selectors are generated from each module's schema. Native input is
        // needed by the React Aria selectbox used by the actual Streamlit app.
        async function nativeClick(element) {
            check(element, "missing native-click target");
            element.scrollIntoView({block: "center", behavior: "instant"});
            await new Promise(requestAnimationFrame);
            const box = element.getBoundingClientRect();
            window.__testClick = {x: box.x + box.width / 2, y: box.y + box.height / 2};
            await until(() => !window.__testClick, "native click was not delivered");
        }
        function selectControl(title) {
            return [...document.querySelectorAll('[data-testid="stSelectbox"]')]
                .find(item => item.textContent.includes(title));
        }
        async function nativeKey(key, virtualKey) {
            window.__testKey = {key, code: key, windowsVirtualKeyCode: virtualKey};
            await until(() => !window.__testKey, "native key was not delivered: " + key);
            await new Promise(requestAnimationFrame);
        }
        async function choose(title, label) {
            const control = selectControl(title);
            check(control, "missing selectbox: " + title);
            await nativeClick(control.querySelector('button[aria-haspopup="listbox"]'));
            await until(() => [...document.querySelectorAll('[role="option"]')]
                .some(item => item.textContent.trim() === label), "options did not open: " + title);
            const option = [...document.querySelectorAll('[role="option"]')]
                .find(item => item.textContent.trim() === label);
            const position = Number(option.getAttribute("aria-posinset"));
            check(Number.isInteger(position) && position > 0, "option position missing: " + label);
            // Native keyboard navigation follows the virtualized list without
            // relying on coordinates that change while its rows are scrolling.
            await nativeKey("Home", 36);
            for (let index = 1; index < position; index++) await nativeKey("ArrowDown", 40);
            await until(() => document.querySelector('[role="option"][data-focused="true"]')?.textContent.trim() === label,
                "requested option did not receive keyboard focus: " + title);
            await nativeKey("Enter", 13);
            await until(() => !document.querySelector('[role="listbox"]')
                && !document.querySelector('[data-test-script-state="running"]'), "selection did not settle: " + title);
        }
        async function openSettingsGroup(label) {
            const summary = () => [...document.querySelectorAll('[data-testid="stExpander"] summary')]
                .find(item => item.textContent.includes(label));
            check(summary(), "missing settings group: " + label);
            if (!summary().closest("details").open) summary().click();
            await until(() => summary()?.closest("details").open, "settings group did not open: " + label);
        }
        async function setCheckbox(label, value, scope = document) {
            const input = () => [...scope.querySelectorAll('[data-testid="stCheckbox"]')]
                .find(item => item.textContent.trim() === label)?.querySelector('input[type="checkbox"]');
            check(input(), "missing checkbox: " + label);
            if (input().checked !== value) input().click();
            await pause(100);
            await until(() => input()?.checked === value
                && !document.querySelector('[data-test-script-state="running"]'),
                "checkbox did not settle: " + label);
        }
        function displayCard(label) {
            return [...document.querySelectorAll('[data-testid="stExpander"]')]
                .find(item => item.querySelector("summary")?.textContent.includes(label));
        }
        const selectedBoundaryRule = () => selectControl("Boundary matching rule")
            ?.querySelector('[role="combobox"]')?.value;
        await choose("Available chart logic", "Trend analysis");
        button("Add logic").click();
        await until(() => document.querySelector('input[aria-label="Window advance and overlap (bars)"]')?.value === "8",
                    "saved configuration did not load");
        check(!selectControl("Slope filter")
            && !document.querySelector('input[aria-label="Redundancy tolerance (%)"]'),
            "display settings appeared before there was a calculated result");
        button("Apply").click();
        await until(() => chart.data?.length > 2 && document.body.textContent.includes("Analysis range:"),
                    "Apply did not produce a trend chart overlay and result table", 90000);
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])), "analysis reset local zoom");
        check(document.querySelector('[data-testid="stDataFrame"]'), "trend result table is missing");
        await openSettingsGroup("Trend analysis · display (1)");
        await until(() => Number(document.querySelector('input[aria-label="Redundancy tolerance (%)"]')?.value) === 25
            && selectedBoundaryRule() === "or",
            "saved redundancy percentage and boundary rule did not load beside the chart");
        const displayTitle = [...document.querySelectorAll("h3")]
            .find(item => item.textContent.includes("Chart display and filters"));
        check(displayTitle && Boolean(button("Apply").compareDocumentPosition(displayTitle)
            & Node.DOCUMENT_POSITION_FOLLOWING)
            && Boolean(displayTitle.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING),
            "display controls are not between calculation Apply and the chart");
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
        const intervalSignature = () => JSON.stringify(visibleTraces().map(trace =>
            [trace.meta.trend_id, trace.x[0], trace.x.at(-1)]));
        const defaultIntervals = intervalSignature();
        // H=8 block windows can produce disjoint default representatives.
        // Showing the complete archive must still retain overlapping fits.
        await setCheckbox("Show redundant trends", true);
        await until(() => {
            const intervals = visibleTraces().map(trace => [trace.x[0], trace.x.at(-1)]);
            return intervals.some((interval, index) => intervals.slice(index + 1).some(other =>
                Math.min(interval[1], other[1]) > Math.max(interval[0], other[0])));
        }, "showing redundant trends suppressed overlapping intervals");
        await setCheckbox("Show redundant trends", false);
        await until(() => intervalSignature() === defaultIntervals,
            "hiding redundant trends did not restore the default intervals");
        check(visibleTraces().every(trace => trace.meta.lambda_values.length > 0
            && trace.meta.primary_lambda === Math.min(...trace.meta.lambda_values)),
            "historical overlays lost segmentation lambda provenance");
        Plotly.Fx.hover(chart, [{curveNumber, pointNumber: Math.floor(trendTrace.x.length / 2)}]);
        check(chart.querySelector(".hoverlayer").textContent.includes("Clarity:")
            && chart.querySelector(".hoverlayer").textContent.includes("Color λ:"),
            "actual dashboard trend hover lacks metrics or lambda labels");
        Plotly.Fx.unhover(chart);
        const originalTrends = JSON.parse(JSON.stringify(visibleTraces()));
        const ids = traces => traces.map(trace => trace.meta.trend_id).sort().join(",");
        const lambdaControl = () => [...document.querySelectorAll('[data-testid="stMultiSelect"]')]
            .find(item => item.textContent.includes("Selected λ values"));
        check(lambdaControl(), "result-derived lambda multiselect is missing");
        check(!document.querySelector('textarea[aria-label="Selected λ values"]'),
              "lambda selection still requires a JSON list");
        await setCheckbox("Show all λ values", false);
        await until(() => visibleTraces().length === 0, "lambda filter did not apply without recalculation");
        await nativeClick(lambdaControl().querySelector('[role="combobox"]'));
        await until(() => document.querySelector('[role="option"]'), "lambda suggestions did not open");
        const options = [...document.querySelectorAll('[role="option"]')]
            .filter(option => option.textContent.trim() !== "Select all");
        const availableLambdas = options.map(option => Number(option.textContent.trim()));
        check(availableLambdas.length > 0 && availableLambdas.every(value => Number.isFinite(value) && value > 0),
              "lambda suggestions are not numeric values from the accepted results");
        const suggested = options.find(option => originalTrends.some(trace =>
            trace.meta.lambda_values.includes(Number(option.textContent.trim()))));
        check(suggested, "lambda suggestions exclude every lambda on the chart");
        const selectedLambda = Number(suggested.textContent.trim());
        const matchingLambda = originalTrends.filter(trace => trace.meta.lambda_values.includes(selectedLambda));
        suggested.click();
        await until(() => lambdaControl().querySelectorAll('[data-baseweb="tag"]').length === 1,
                    "choosing a suggested lambda did not select exactly that value");
        await nativeClick([...document.querySelectorAll("h3")]
            .find(item => item.textContent.includes("Chart display and filters")));
        await until(() => ids(visibleTraces()) === ids(matchingLambda),
                    "selecting a suggested lambda did not filter the chart immediately");
        check(!document.body.textContent.includes("Unapplied changes"),
              "lambda display edits changed the calculation draft");
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])),
              "lambda display filtering reset local zoom");
        await setCheckbox("Show all λ values", true);
        await until(() => ids(visibleTraces()) === ids(originalTrends),
                    "all-lambda display did not restore cached trends immediately");
        await setCheckbox("Show on chart", false);
        await until(() => chart.data.length === 2, "hiding a plugin did not remove its overlays immediately");
        check(document.querySelector('[data-testid="stDataFrame"]'), "hiding overlays removed the result table");
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])),
              "hiding a plugin reset local zoom");
        await setCheckbox("Show on chart", true);
        await until(() => ids(visibleTraces()) === ids(originalTrends),
                    "showing a plugin did not restore its cached result immediately");
        for (const label of ["negative", "positive", "range", "all"]) {
            const expected = originalTrends.filter(trace => label === "negative"
                ? trace.y.at(-1) < trace.y[0]
                : label === "positive" ? trace.y.at(-1) > trace.y[0] : true);
            await choose("Slope filter", label);
            check(document.querySelector('input[aria-label="Minimum slope"]')
                && document.querySelector('input[aria-label="Maximum slope"]'),
                "schema-generated slope bounds are missing");
            await until(() => ids(visibleTraces()) === ids(expected)
                && !document.body.textContent.includes("Unapplied changes")
                && !document.querySelector('[data-test-script-state="running"]'),
                "slope filter did not update the chart without Apply: " + label);
            check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])),
                  "changing a slope filter reset local zoom");
            verify();
        }
        const lookbackControl = () => document.querySelector('input[aria-label="Window advance and overlap (bars)"]')
            ?.closest('[data-testid="stNumberInput"]');
        await nativeClick(lookbackControl().querySelector('button[data-testid="stNumberInputStepUp"]'));
        await until(() => document.body.textContent.includes("Unapplied changes"),
                    "editing calculation settings did not remain a draft");
        check(ids(visibleTraces()) === ids(originalTrends), "calculation settings changed the chart before Apply");
        await choose("Slope filter", "negative");
        await until(() => ids(visibleTraces()) === ids(originalTrends.filter(trace => trace.y.at(-1) < trace.y[0])),
                    "display filter did not work while calculation changes were pending");
        check(document.body.textContent.includes("Unapplied changes"),
              "editing display settings discarded pending calculation changes");
        await choose("Slope filter", "all");
        await nativeClick(lookbackControl().querySelector('button[data-testid="stNumberInputStepDown"]'));
        await until(() => !document.body.textContent.includes("Unapplied changes")
            && ids(visibleTraces()) === ids(originalTrends),
            "restoring the calculation draft did not preserve the current display");
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
        const originalIds = ids(visibleTraces());
        button("Remove").click();
        await until(() => document.body.textContent.includes("Unapplied changes")
            && !document.querySelector('input[aria-label="Window advance and overlap (bars)"]'),
            "Remove did not update the draft selection");
        check(ids(visibleTraces()) === originalIds, "Remove changed the applied chart before Apply");
        button("Apply").click();
        await until(() => chart.data?.length === 2, "Apply did not remove trend overlays");
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])),
              "removal reset local zoom");
        button("Add logic").click();
        await until(() => document.querySelector('input[aria-label="Window advance and overlap (bars)"]')?.value === "8"
            && !selectControl("Slope filter")
            && !document.querySelector('[data-test-script-state="running"]'),
            "adding another trend instance did not reuse saved defaults without result display controls");
        check(chart.data.length === 2, "adding an instance calculated it before Apply");
        button("Apply").click();
        await until(() => ids(visibleTraces()) === originalIds && chart.data.length > 2,
                    "re-added trend instance did not restore its matching cached result", 90000);
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])),
              "re-adding a cached trend instance reset local zoom");
        await openSettingsGroup("Trend analysis · display (1)");
        check(Number(document.querySelector('input[aria-label="Redundancy tolerance (%)"]')?.value) === 25
            && selectedBoundaryRule() === "or", "new trend instance lost its migrated display defaults");
        verify();
        await choose("Available chart logic", "Open price bands");
        button("Add logic").click();
        await until(() => selectControl("Open reference"), "band calculation settings did not appear");
        await choose("Open reference", "first_bar");
        button("Apply").click();
        const bandTraces = () => chart.data.slice(2).filter(trace => !trace.meta?.trend_overlay);
        const trendTraces = () => chart.data.slice(2).filter(trace => trace.meta?.trend_overlay && trace.x.length);
        await until(() => bandTraces().length === 3 && ids(trendTraces()) === originalIds,
                    "bands and cached trends were not applied together");
        await openSettingsGroup("Open price bands · display (2)");
        await setCheckbox("Show on chart", false, displayCard("Open price bands · display (2)"));
        await until(() => bandTraces().length === 0 && ids(trendTraces()) === originalIds,
                    "generic band visibility changed other plugins or required Apply");
        await setCheckbox("Show on chart", true, displayCard("Open price bands · display (2)"));
        await until(() => bandTraces().length === 3, "showing bands did not restore their cached outputs");
        await setCheckbox("Show open reference", false, displayCard("Open price bands · display (2)"));
        await until(() => bandTraces().length === 2 && !bandTraces().some(trace => trace.name === "Open"),
                    "band display option did not hide its reference line without Apply");
        check(ids(trendTraces()) === originalIds, "band display settings changed trend outputs");
        check(chart.layout.xaxis.range.every((value, index) => close(value, trendRange[index])),
              "combining plugins or changing their display reset local zoom");
        check(!document.body.textContent.includes("Unapplied changes"),
              "band display settings were treated as calculation changes");
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
    from research.analysis.trend import TrendConfig

    settings_path = tmp_path / "trend_config.json"
    saved_settings = TrendConfig(max_history=8, redundancy_percentage=25.0,
                                 redundancy_boundary_rule="or")
    save_trend_settings(saved_settings, settings_path)
    original_settings = settings_path.read_bytes()
    calls_path = tmp_path / "analysis-calls.txt"
    app.write_text(
        "import streamlit as st\nfrom dashboard.app import _data_page\nfrom research import Research\n"
        "from pathlib import Path\nimport dashboard.trend_settings as settings\nimport dashboard.workspace as workspace\n"
        "from research.analysis import loader as analysis_loader\n"
        f"analysis_loader.project_root = lambda: Path({str(tmp_path / 'private-test-project')!r})\n"
        f"settings.DEFAULT_SETTINGS_PATH = Path({str(settings_path)!r})\n"
        f"workspace.DEFAULT_DIRECTORY = Path({str(tmp_path / 'workspaces')!r})\n"
        "import research.analysis.builtins.trend as trend_plugin\n"
        "if not getattr(trend_plugin.analyze, '_browser_counter', False):\n"
        "    def instrument(original):\n"
        "        def counted(*args, **kwargs):\n"
        f"            with Path({str(calls_path)!r}).open('a') as calls: calls.write('analysis\\n')\n"
        "            return original(*args, **kwargs)\n"
        "        counted._browser_counter = True\n"
        "        return counted\n"
        "    trend_plugin.analyze = instrument(trend_plugin.analyze)\n"
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
                                  screenshot_path=tmp_path / "chart-window.png", timeout_seconds=240, native_clicks=True)
            assert load_trend_settings(settings_path).config == saved_settings
            assert settings_path.read_bytes() == original_settings
            # Display filters and visibility never recalculate accepted trends.
            # Navigation, calculation drafts, and a new identical instance reuse results.
            assert calls_path.read_text().splitlines() == ["analysis"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
