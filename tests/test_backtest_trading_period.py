"""Clock-aligned hourly execution preserves real bar durations and early closes."""
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from data_pipeline import DataRequest
from data_pipeline.processing.processors.time_window import TimeWindowFilter
from research import Research, PreflightError
from research.datasets import prepare_snapshot
from research.instruments import Instrument, expected_bars
from test_initial_split import EveryBar

NY = ZoneInfo('America/New_York')
START = datetime(2024, 11, 27, tzinfo=UTC)
END = datetime(2024, 12, 3, tzinfo=UTC)


def source(tmp_path, *, missing=None, after_close=True):
    r = Research(tmp_path / 'data', tmp_path / 'runs')
    stamps = [datetime(2024, month, day, hour, tzinfo=NY).astimezone(UTC)
              for month, day in [(11, 26), (11, 27), (11, 29), (12, 2)]
              for hour in range(9, 17)
              if (month, day, hour) != missing and (after_close or day != 29 or hour < 13)]
    f = pl.DataFrame({'timestamp': stamps, 'symbol': ['A'] * len(stamps),
                      **{name: [10.] * len(stamps) for name in ('open','high','low','close')},
                      'volume': [100.] * len(stamps)})
    raw = r.pipeline.ingest_frame(DataRequest('A', stamps[0], END, timeframe='1h'), f).raw
    from datetime import time
    processed = r.pipeline.process(raw.dataset_id, processors=[TimeWindowFilter(end_time=time(16,0,1))])
    return r, processed


@pytest.mark.parametrize('after_close', [False, True])
def test_processed_early_close_warns_and_does_not_trade_after_close(tmp_path, after_close):
    r, processed = source(tmp_path, after_close=after_close)
    run = r.backtest(['A'], strategies=[EveryBar()], start=START, end=END, timeframe='1h',
                     layer='processed', pipeline_id=processed.pipeline_id, trading_period='09:00-15:00')
    restored = r.load_run(run.run_id)
    assert restored.manifest['trading_period'] == '09:00-15:00'
    warning = restored.manifest['warnings'][0]
    assert warning['date'] == '2024-11-29'
    assert f"Excluded {3 if after_close else 0}" in warning['message']
    selected = prepare_snapshot(r.pipeline, tickers=['A'], start=START, end=END, timeframe='1h',
                                lookback=1, trading_period='09:00-15:00')
    early = [b for b in selected.bars['A'] if b.timestamp.astimezone(NY).day == 29]
    assert [b.timestamp.astimezone(NY).hour for b in early] == [9,10,11,12]
    assert early[-1].close.astimezone(NY).hour == 13
    assert len(selected.frames['A']) == len(selected.bars['A'])
    normal = [b for b in selected.bars['A'] if b.timestamp.astimezone(NY).day == 27]
    assert normal[-1].open.astimezone(NY).hour == 15
    assert normal[-1].close.astimezone(NY).hour == 16


def test_missing_bar_before_early_close_still_fails(tmp_path):
    r, _ = source(tmp_path, missing=(11,29,11))
    with pytest.raises(PreflightError) as caught:
        prepare_snapshot(r.pipeline, tickers=['A'], start=START, end=END, timeframe='1h',
                         lookback=1, trading_period='09:00-15:00')
    assert caught.value.report.issues[0].code == 'missing_bars'
    assert caught.value.report.issues[0].timestamps == (datetime(2024,11,29,16,tzinfo=UTC),)


def test_regular_mode_remains_strict_and_new_mode_tracks_dst(tmp_path):
    r, _ = source(tmp_path)
    with pytest.raises(PreflightError):
        prepare_snapshot(r.pipeline, tickers=['A'], start=START, end=END, timeframe='1h', lookback=1)
    bars = expected_bars(Instrument('A'), start=datetime(2024,11,1,tzinfo=UTC),
                         end=datetime(2024,11,5,tzinfo=UTC), timeframe='1h', trading_period='09:00-15:00')
    assert [b.open.hour for b in bars if b.open.astimezone(NY).hour == 9] == [13,14]
    with pytest.raises(Exception, match='requires 1h'):
        expected_bars(Instrument('A'), start=START, end=END, timeframe='1d', trading_period='09:00-15:00')


def test_dashboard_period_persists_and_displays_warnings(tmp_path, monkeypatch):
    from test_dashboard_processed_backtest import _open
    from datetime import date
    r, processed = source(tmp_path)
    app = _open(monkeypatch, r)
    app.radio(key='backtest-layer').set_value('processed').run()
    app.radio(key='backtest-trading-period').set_value('09:00-15:00').run()
    app.date_input(key='backtest-start').set_value(date(2024,11,27))
    app.date_input(key='backtest-end').set_value(date(2024,12,3))
    next(f for f in app.number_input if f.label == 'Fast moving average (bars)').set_value(1)
    next(f for f in app.number_input if f.label == 'Slow moving average (bars)').set_value(2)
    next(b for b in app.button if b.label == 'Validate all data and run').click().run()
    assert not app.exception and not app.error
    assert any('Early-close sessions' in w.value for w in app.warning)
    assert app.session_state['backtest-run'].manifest['trading_period'] == '09:00-15:00'
    restarted = _open(monkeypatch, r)
    assert restarted.radio(key='backtest-trading-period').value == '09:00-15:00'


def test_missing_1600_bar_is_not_required(tmp_path):
    r, _ = source(tmp_path, missing=(11,27,16))
    selected = prepare_snapshot(r.pipeline, tickers=['A'], start=START, end=END, timeframe='1h',
                                lookback=1, trading_period='09:00-15:00')
    assert selected.report.ok
    assert all(stamp.astimezone(NY).hour <= 15 for stamp in selected.frames['A']['timestamp'])


def test_previous_saved_period_is_migrated(tmp_path, monkeypatch):
    from dashboard.state.workspace import MarketMemory
    from test_dashboard_processed_backtest import _open
    r, _ = source(tmp_path)
    memory = MarketMemory('backtest:' + str(r.pipeline.store.data_dir.resolve()))
    memory.values['widgets'].update({'backtest-timeframe': '1h', 'backtest-trading-period': '09:00-16:00'})
    memory.save()
    app = _open(monkeypatch, r)
    assert app.radio(key='backtest-trading-period').value == '09:00-15:00'
    restored = MarketMemory(memory.scope)
    assert restored.values['widgets']['backtest-trading-period'] == '09:00-15:00'
