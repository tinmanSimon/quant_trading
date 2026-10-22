"""The live batch page completes, preserves history, and reopens without running scripts."""

from datetime import UTC, datetime, timedelta
import http.client
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
from research.analysis.batch import BatchStore
from test_dashboard_chart_browser import _run_browser_document


INTERACTIONS = r"""
(() => {
    const output = document.createElement('pre');
    output.id = 'result'; output.textContent = 'PENDING'; document.body.appendChild(output);
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    const button = name => [...document.querySelectorAll('button')].find(item => item.textContent.trim() === name);
    function check(value, message) { if (!value) throw new Error(message); }
    async function until(fn, message) {
        const deadline = performance.now() + 60000;
        while (performance.now() < deadline) { if (fn()) return; await pause(50); }
        throw new Error(message + ': ' + document.body.textContent.slice(-2000));
    }
    async function test() {
        await until(() => button('Run analysis') && !button('Run analysis').disabled, 'run controls did not load');
        for (let run = 1; run <= 2; run++) {
            button('Run analysis').click();
            await until(() => document.body.textContent.includes('Executed tickers: ' + (run * 101))
                && document.body.textContent.includes('Completed: 102 of 102 tickers finished')
                && document.body.textContent.includes('completed: 100')
                && document.body.textContent.includes('no data: 1')
                && !button('Stop after current tickers')
                && button('Run analysis') && !button('Run analysis').disabled, 'batch did not finish');
            check(document.body.textContent.includes('completed: 100'), 'successful ticker count is missing');
            check(document.body.textContent.includes('failed: 1'), 'failed ticker count is missing');
            check(document.body.textContent.includes('no data: 1'), 'missing-data ticker count is missing');
            await until(() => button('Save analysis') && !button('Save analysis').disabled, 'save unavailable');
            button('Save analysis').click();
            await until(() => !button('Save analysis') && button('Delete analysis'), 'analysis was not saved');
            check(!document.querySelector('.js-plotly-plot'), 'batch page created a chart');
            check(!document.querySelector('[data-testid="stException"]'), 'batch page raised an exception');
        }
        button('Reload analyses').click();
        await pause(1000);
        check(document.body.textContent.includes('Executed tickers: 202'), 'discovery reran analyses');
        check(document.body.textContent.includes('Ticker summaries'), 'saved summary disappeared');
        output.textContent = 'PASS';
    }
    test().catch(error => { output.textContent = 'FAIL: ' + error.stack; });
})();
"""

REOPEN = r"""
(() => {
    const output = document.createElement('pre');
    output.id = 'result'; output.textContent = 'PENDING'; document.body.appendChild(output);
    const deadline = performance.now() + 20000;
    const timer = setInterval(() => {
        if (document.querySelector('[data-testid="stException"]')) {
            clearInterval(timer); output.textContent = 'FAIL: ' + document.body.textContent;
        } else if (document.body.textContent.includes('Overall summary')
                   && document.body.textContent.includes('Executed tickers: 202')) {
            clearInterval(timer); output.textContent = 'PASS';
        } else if (performance.now() > deadline) {
            clearInterval(timer); output.textContent = 'FAIL: Saved batch did not reopen';
        }
    }, 100);
})();
"""


@pytest.mark.browser
def test_hundred_ticker_batches_in_actual_streamlit_and_reopen(tmp_path):
    root, runs = tmp_path / 'data', tmp_path / 'runs'
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(4)]
    pipeline = DataPipeline(root)
    for ticker in ['FAIL', *[f'T{i:03}' for i in range(100)]]:
        frame = pl.DataFrame({'timestamp': stamps, 'symbol': [ticker] * 4, 'open': [100.] * 4,
                              'high': [101.] * 4, 'low': [99.] * 4, 'close': [100.] * 4,
                              'volume': [100.] * 4}, schema=OHLCV_SCHEMA)
        pipeline.store.write_raw(DataRequest(ticker, stamps[0], stamps[-1] + timedelta(days=1), timeframe='1d'), frame)
    calls = tmp_path / 'calls.txt'
    scripts = tmp_path / 'secrete_analysis' / 'analyses'
    scripts.mkdir(parents=True)
    (scripts / 'counts.py').write_text(f'''
from pathlib import Path
import polars as pl
from research.analysis import analysis
@analysis("batch_counts", label="Batch counts")
def run(ctx):
    ticker = ctx.bars["symbol"][0]
    with Path({str(calls)!r}).open("a") as file:
        file.write(ticker + "\\n")
    if ticker == "FAIL":
        raise ValueError("Deliberate isolated ticker failure")
    wins = int(ticker[1:]) + 1
    return (ctx.result().text("Summary for " + ticker)
            .table("Summary", pl.DataFrame({{"outcome": ["win", "loss", "inconclusive", "total"],
                                            "count": [wins, 1, 0, wins + 1]}})))
''')
    scope = 'batch-analysis:' + repr((str(root.resolve()), str(runs.resolve())))
    memory = MarketMemory(scope, directory=runs / 'analysis' / 'workspaces')
    memory.values['widgets'].update({'batch-all-tickers': True, 'batch-extra-tickers': 'MISSING'})
    memory.values['analyses'] = [{'instance_id': 'counts', 'name': 'batch_counts', 'version': '1', 'config': {}}]
    memory.save()
    app = tmp_path / 'app.py'
    app.write_text(f'''
from pathlib import Path
import streamlit as st
from dashboard.pages.analysis import _analysis_page
from research.analysis.scripting import discovery
from research import Research
st.set_page_config(layout="wide")
discovery.project_root = lambda: Path({str(tmp_path)!r})
calls = Path({str(calls)!r})
st.caption("Executed tickers: " + str(len(calls.read_text().splitlines()) if calls.exists() else 0))
_analysis_page(Research({str(root)!r}, runs_dir={str(runs)!r}))
''')
    with socket.socket() as available:
        available.bind(('127.0.0.1', 0))
        port = available.getsockname()[1]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    with (tmp_path / 'streamlit.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(app),
                                    '--server.headless=true', '--server.address=127.0.0.1', f'--server.port={port}',
                                    '--browser.gatherUsageStats=false'], cwd=tmp_path, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            ready = False
            while time.monotonic() < deadline and process.poll() is None:
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                try:
                    connection.request('GET', '/_stcore/health')
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
            url = f'http://127.0.0.1:{port}'
            _run_browser_document(tmp_path, url=url, browser_js=INTERACTIONS, timeout_seconds=150)
            store = BatchStore(runs / 'analysis')
            saved = store.list_runs()
            assert len(saved) == 2
            for manifest in saved:
                assert manifest['status'] == 'completed'
                assert manifest['counts']['completed'] == 100
                assert manifest['counts']['failed'] == manifest['counts']['no_data'] == 1
                summaries = [v['summaries']['counts']['values'] for v in manifest['tickers'].values()
                             if v['status'] == 'completed']
                assert sum(v['win'] for v in summaries) == 5050
                assert sum(v['loss'] for v in summaries) == 100
                assert sum(v['total'] for v in summaries) == 5150
            reopened = tmp_path / 'reopened'
            reopened.mkdir()
            _run_browser_document(reopened, url=url, browser_js=REOPEN, timeout_seconds=30)
            assert len(calls.read_text().splitlines()) == 202
            assert len(store.list_runs()) == 2
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
