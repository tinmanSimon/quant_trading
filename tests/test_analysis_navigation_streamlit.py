"""Finding and viewport interactions must not rerun the expensive market page."""

from datetime import UTC, datetime, timedelta
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import polars as pl
import pytest

from dashboard.state.workspace import MarketMemory
from data_pipeline import DataPipeline, DataRequest
from data_pipeline.schemas import OHLCV_SCHEMA
from test_dashboard_chart_browser import _run_browser_document


INTERACTIONS = r"""
(() => {
    const output = document.createElement('pre');
    output.id = 'result'; output.textContent = 'PENDING'; document.body.appendChild(output);
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    function check(value, message) { if (!value) throw new Error(message); }
    async function until(fn, message) {
        const deadline = performance.now() + 20000;
        while (performance.now() < deadline) { if (fn()) return; await pause(20); }
        throw new Error(message + ': ' + document.body.textContent.slice(-1500));
    }
    const button = name => [...document.querySelectorAll('button')].find(item => item.textContent.trim() === name);
    const fullPasses = () => Number(document.body.textContent.match(/Full page passes: (\d+)/)?.[1]);
    async function test() {
        await until(() => document.querySelector('.js-plotly-plot')?.layout?.yaxis?.autorange === false,
                    'chart did not initialize');
        const chart = document.querySelector('.js-plotly-plot');
        button('Apply').click();
        await until(() => button('Next') && chart.layout.xaxis3, 'private findings did not apply');
        await pause(250);
        const before = fullPasses(), timings = [];
        async function jump(label, index) {
            const start = performance.now();
            button(label).click();
            await until(() => chart.layout.xaxis.range[0] < index && chart.layout.xaxis.range[1] > index
                && chart.layout.shapes?.some(shape => shape.name === 'analysis-navigation-target' && shape.x0 === index),
                'finding did not load: ' + index);
            timings.push(performance.now() - start);
            check(document.querySelector('.js-plotly-plot') === chart, 'navigation remounted chart');
            check(['xaxis2', 'xaxis3'].every(axis => chart.layout[axis].range.every((v,i) => v === chart.layout.xaxis.range[i])),
                  'indicator panels did not follow the finding');
            check(fullPasses() === before, 'finding navigation reran the market page');
        }
        await jump('Next', 1200);
        await jump('Next', 4200);
        await jump('Next', 92000);
        await jump('Previous', 4200);
        button('All history').click();
        await until(() => chart.layout.xaxis.range[0] === -0.5 && chart.data[0].x[0] < 500,
                    'full-history viewport did not load');
        button('Latest 50').click();
        await until(() => chart.layout.xaxis.range[0] > 99000 && chart.data[0].x[0] > 99000,
                    'latest viewport did not load');
        check(fullPasses() === before, 'viewport callback reran the market page');

        // A genuine page-setting change still refreshes source/analysis controls.
        const originalBars = [...document.querySelectorAll('[data-testid="stCheckbox"]')]
            .find(item => item.textContent.trim() === 'Show original bars').querySelector('input');
        originalBars.click();
        await until(() => fullPasses() > before && document.body.textContent.includes('Original rows'),
                    'page setting did not trigger a full refresh');
        await pause(250);
        check(chart.layout.xaxis.range[0] > 99000, 'full refresh replayed an old finding command');
        check(document.querySelector('.js-plotly-plot') === chart, 'full refresh remounted chart');
        check(!document.querySelector('[data-testid="stException"]'), 'Streamlit exception');
        window.navigationMetrics = {milliseconds: timings, pageRunsBeforeNavigation: before,
                                     pageRunsAfterRefresh: fullPasses()};
        output.textContent = 'PASS';
    }
    test().catch(error => { output.textContent = 'FAIL: ' + error.stack; });
})();
"""


@pytest.mark.browser
def test_navigation_reuses_market_snapshot_in_actual_streamlit(tmp_path):
    count = 100_000
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(count)]
    prices = [100. + i % 97 / 10 for i in range(count)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * count,
                          "open": prices, "close": prices, "low": [p - 1 for p in prices],
                          "high": [p + 1 for p in prices], "volume": [100.] * count}, schema=OHLCV_SCHEMA)
    root = tmp_path / "data"
    DataPipeline(root).store.write_raw(DataRequest("TEST", stamps[0], stamps[-1] + timedelta(minutes=1),
                                                   timeframe="1m"), frame)
    scripts = tmp_path / "secrete_analysis" / "analyses"
    scripts.mkdir(parents=True)
    (scripts / "example.py").write_text('''
import polars as pl
from research.analysis import analysis
@analysis("navigation_test")
def run(ctx):
    data = ctx.selected(ctx.features(average=ctx.sma(20), momentum=ctx.rsi(14)))
    findings = data.filter(pl.col("source_index").is_in([1200, 4200, 92000]))
    return (ctx.result().text("Navigation test")
            .signals("Entries", findings)
            .line("Average", data, value="average")
            .line("RSI", data, value="momentum", panel={"id":"rsi", "title":"RSI", "range":[0,100]}))
''')
    directory = tmp_path / "workspaces"
    memory = MarketMemory(str(root.resolve()), directory=directory)
    memory.values["chart_logics"] = {"version": 1, "draft": [
        {"instance_id": "example", "name": "navigation_test", "version": "1", "config": {}}],
        "applied": [], "defaults": {}}
    memory.save()
    calls = tmp_path / "page-runs.txt"
    app = tmp_path / "app.py"
    app.write_text(f'''
from pathlib import Path
import streamlit as st
from dashboard.app import _data_page
from dashboard.state import workspace
from research.analysis import loader
from research import Research
loader.project_root = lambda: Path({str(tmp_path)!r})
workspace.DEFAULT_DIRECTORY = Path({str(directory)!r})
st.set_page_config(layout="wide")
with Path({str(calls)!r}).open("a") as file:
    file.write("page\\n")
st.caption("Full page passes: " + str(len(Path({str(calls)!r}).read_text().splitlines())))
_data_page(Research({str(root)!r}, runs_dir={str(tmp_path / "runs")!r}))
''')
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    with (tmp_path / "streamlit.log").open("w+") as log:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(app),
                                    "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}",
                                    "--browser.gatherUsageStats=false"], cwd=tmp_path, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
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
                time.sleep(.05)
            log.seek(0)
            assert ready, log.read()
            metrics = _run_browser_document(tmp_path, url=f"http://127.0.0.1:{port}",
                                            browser_js=INTERACTIONS, timeout_seconds=90,
                                            metrics_expression="window.navigationMetrics")
            assert metrics["pageRunsBeforeNavigation"] == 2
            assert metrics["pageRunsAfterRefresh"] == 3
            assert calls.read_text().splitlines() == ["page"] * 3
            print("100k-bar navigation:", json.dumps(metrics))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
