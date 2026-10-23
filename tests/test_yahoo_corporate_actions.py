"""Offline tests for free Yahoo event evidence, without market-data requests."""
from datetime import UTC, date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

from data_pipeline.corporate_actions import other_events_report
from data_pipeline.corporate_actions.providers import YahooActionProvider
from data_pipeline.corporate_actions.providers.yahoo import _schedule

NY = ZoneInfo('America/New_York')
START, END = date(2020, 8, 6), date(2020, 8, 10)


def history(start=START, end=END, events=None, tail=True):
    days = [t.date() for t in _schedule(start, end).index]
    if events:
        days.extend(date.fromisoformat(day) for day in events)
    if tail:
        now = datetime.now(UTC)
        recent = _schedule(start, now.astimezone(NY).date())
        completed = recent[recent['close'] <= pd.Timestamp(now)]
        days.extend(stamp.date() for stamp in completed.index)
    frame = pd.DataFrame({'Close': 100., 'Dividends': 0., 'Stock Splits': 0.},
                         index=pd.DatetimeIndex(sorted(set(days)), tz=NY))
    for day, values in (events or {}).items():
        for name, value in values.items():
            frame.loc[pd.Timestamp(day, tz=NY), name] = value
    return frame


def ticker(frame, *, symbol='AAPL', currency='USD', dividends=None):
    obj = Mock()
    obj.history.return_value = frame
    obj.get_history_metadata.return_value = dict(symbol=symbol, instrumentType='EQUITY',
        currency=currency, exchangeTimezoneName='America/New_York', exchangeName='NMS')
    obj.get_dividends.return_value = (frame.loc[frame['Dividends'] != 0, 'Dividends']
                                    if dividends is None else dividends)
    return obj


def bars(stamp='2020-08-06T19:59:00+00:00', close=100., symbol='AAPL'):
    return pl.DataFrame({'timestamp': [datetime.fromisoformat(stamp)], 'symbol': [symbol], 'close': [close]})


def fetch(obj, *, start=START, end=END, raw=None, symbols=('AAPL',), **kwargs):
    return YahooActionProvider(ticker_factory=lambda _: obj).fetch(symbols, start, end,
        source_basis='unadjusted', reference_bars=raw, **kwargs)


def test_restore_cash_using_split_after_selected_reference_and_raw_close():
    frame = history(events={'2020-08-07': {'Dividends': .205}, '2020-08-31': {'Stock Splits': 4}})
    obj = ticker(frame)
    snap = fetch(obj, raw={'AAPL': bars(close=455.61)})
    event, = snap.events
    assert event.cash_amount == pytest.approx(.82)
    assert event.reference_close == 455.61
    assert event.price_factor == pytest.approx((455.61 - .82) / 455.61)
    assert event.status == 'apply'
    assert '2020-08-31=4' in event.details
    assert snap.coverage[0].status == 'complete'
    assert snap.coverage[0].other_events_status == 'unknown'
    assert 'ticker-based' in snap.coverage[0].details
    obj.history.assert_called_once_with(period='max', interval='1d', auto_adjust=False,
        back_adjust=False, actions=True, repair=False, keepna=True, raise_errors=True)
    obj.get_dividends.assert_called_once_with(period='max')
    assert other_events_report(snap, ['AAPL'])[0]['status'] == 'unknown'


def test_same_date_split_cash_is_pending_but_split_retained():
    frame = history(events={'2020-08-07': {'Dividends': .5, 'Stock Splits': 2}})
    snap = fetch(ticker(frame), raw={'AAPL': bars()})
    cash = next(e for e in snap.events if e.event_type == 'cash_dividend')
    split = next(e for e in snap.events if e.event_type == 'forward_split')
    assert cash.status == 'pending'
    assert 'Same-date' in cash.details
    assert split.status == 'apply' and split.share_ratio == 2


def test_missing_previous_close_preserves_cash_and_splits_without_yahoo_price_fallback():
    frame = history(events={'2020-08-07': {'Dividends': 1}, '2020-08-10': {'Stock Splits': 2}})
    snap = fetch(ticker(frame), raw={'AAPL': bars('2020-08-06T20:01:00+00:00')})
    cash = next(e for e in snap.events if e.event_type == 'cash_dividend')
    assert cash.status == 'pending'
    assert cash.price_factor is None
    assert 'final regular-session minute' in cash.details
    assert any(e.share_ratio == 2 for e in snap.events)


def test_early_close_then_weekend_uses_actual_previous_session_final_minute():
    start, end = date(2024, 11, 29), date(2024, 12, 2)
    frame = history(start, end, {'2024-12-02': {'Dividends': 1}})
    raw = pl.concat([bars('2024-11-29T17:59:00+00:00', 50), bars('2024-11-29T20:59:00+00:00', 999)])
    snap = fetch(ticker(frame), start=start, end=end, raw={'AAPL': raw})
    assert snap.events[0].reference_close == 50
    assert snap.events[0].price_factor == .98


def test_holiday_is_not_a_missing_session_and_previous_close_uses_wednesday():
    start, end = date(2024, 11, 27), date(2024, 11, 29)
    frame = history(start, end, {'2024-11-29': {'Dividends': 1}})
    snap = fetch(ticker(frame), start=start, end=end,
                 raw={'AAPL': bars('2024-11-27T20:59:00+00:00', 50)})
    assert snap.events[0].price_factor == .98


def test_reverse_split_adjusts_reported_ratio():
    frame = history(events={'2020-08-07': {'Stock Splits': .1}})
    loader = Mock(side_effect=AssertionError('Split-only history must not load minute data'))
    event, = fetch(ticker(frame), raw=loader).events
    assert (event.event_type, event.share_ratio, event.price_factor) == ('reverse_split', .1, 10)
    loader.assert_not_called()


def test_loader_exception_leaves_pending_cash_with_safe_details():
    frame = history(events={'2020-08-07': {'Dividends': 1}})
    snap = fetch(ticker(frame), raw=Mock(side_effect=RuntimeError('SECRET')))
    assert snap.events[0].status == 'pending'
    assert 'SECRET' not in str(snap.to_dict())


def test_each_ticker_failure_isolated_and_no_events_is_success_only_after_history_check():
    good = ticker(history())
    bad = Mock()
    bad.history.side_effect = RuntimeError('https://secret-key.example')
    provider = YahooActionProvider(ticker_factory=lambda s: good if s == 'AAPL' else bad)
    snap = provider.fetch(['AAPL', 'BAD'], START, END)
    assert [(c.symbol, c.status) for c in snap.coverage] == [('AAPL', 'complete'), ('BAD', 'error')]
    assert 'secret-key' not in str(snap.to_dict())
    assert not snap.events


@pytest.mark.parametrize('alter', [
    lambda f: f.iloc[:0],
    lambda f: f.drop(pd.Timestamp('2020-08-07', tz=NY)),
    lambda f: f.drop(columns='Stock Splits'),
    lambda f: f.assign(Dividends=float('nan')),
    lambda f: f.set_axis(f.index.tz_localize(None)),
])
def test_incomplete_or_invalid_history_is_not_event_free(alter):
    snap = fetch(ticker(alter(history())))
    assert snap.coverage[0].status == 'error'


def test_non_us_or_non_usd_metadata_rejected():
    snap = fetch(ticker(history(), currency='CAD'))
    assert snap.coverage[0].status == 'error'


def test_foreign_cash_currency_is_pending_even_for_usd_equity():
    frame = history(events={'2020-08-07': {'Dividends': 1}})
    divs = frame.loc[frame['Dividends'] > 0, ['Dividends']].assign(currency='CAD')
    event, = fetch(ticker(frame, dividends=divs), raw={'AAPL': bars()}).events
    assert event.status == 'pending' and event.currency == 'CAD'


def test_inconsistent_second_dividend_response_does_not_apply():
    frame = history(events={'2020-08-07': {'Dividends': 1}})
    divs = frame.loc[frame['Dividends'] > 0, 'Dividends'] * 2
    event, = fetch(ticker(frame, dividends=divs), raw={'AAPL': bars()}).events
    assert event.status == 'pending' and 'inconsistent' in event.details


@pytest.mark.parametrize('ratio', [1.281, 1.253, 1.324, 1.5])
def test_ambiguous_yahoo_split_cannot_adjust_volume_and_is_reported(ratio):
    frame = history(events={'2020-08-07': {'Stock Splits': ratio}})
    snap = fetch(ticker(frame))
    split = next(e for e in snap.events if e.event_type == 'forward_split')
    assert split.status == 'pending' and split.price_factor is None
    assert any(e.event_type == 'unclassified_yahoo_action' for e in snap.events)
    assert any(r['event_type'] == 'unclassified_yahoo_action' for r in other_events_report(snap, ['AAPL']))


def test_ambiguous_later_than_reference_split_prevents_cash_unrebasing():
    frame = history(events={'2020-08-07': {'Dividends': 1}, '2020-08-31': {'Stock Splits': 1.281}})
    event, = fetch(ticker(frame), raw={'AAPL': bars()}).events
    assert event.status == 'pending' and event.price_factor is None
    assert 'ambiguous factor' in event.details


def test_stale_max_history_cannot_establish_future_split_chain():
    frame = history(events={'2020-08-07': {'Dividends': 1}}, tail=False)
    event, = fetch(ticker(frame), raw={'AAPL': bars()}).events
    assert event.status == 'pending' and 'stale or truncated' in event.details


def test_gap_after_reference_prevents_cash_share_basis_verification():
    frame = history(events={'2020-08-07': {'Dividends': 1}})
    frame = frame.drop(pd.Timestamp('2020-08-31', tz=NY))
    event, = fetch(ticker(frame), raw={'AAPL': bars()}).events
    assert event.status == 'pending' and 'stale or truncated' in event.details


def test_public_yfinance_dividend_wrapper_signature():
    # PriceHistory accepts repair=False, but Ticker.get_dividends does not.
    frame = history(events={'2020-08-07': {'Dividends': 1}})
    obj = ticker(frame)
    def public_get_dividends(period='max'):
        assert period == 'max'
        return frame.loc[frame['Dividends'] > 0, 'Dividends']
    obj.get_dividends = public_get_dividends
    event, = fetch(obj, raw={'AAPL': bars()}).events
    assert event.status == 'apply'


def test_per_symbol_start_uses_stored_history_instead_of_prelisting_dates():
    frame = history(start=date(2020, 8, 7))
    snap = fetch(ticker(frame), coverage_starts={'AAPL': date(2020, 8, 7)})
    assert snap.coverage[0].status == 'complete'
    assert snap.coverage[0].start == date(2020, 8, 7)
