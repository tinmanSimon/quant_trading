"""Free, retrospective Yahoo split/dividend evidence via yfinance.

Yahoo's daily OHLC and historical cash amounts are split-restated even with
``auto_adjust=False``. Restore cash to its ex-date share basis using the full
subsequent split history, then use the user's raw previous-session close.
Yahoo provides no comprehensive other-action or historical identity coverage.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
import importlib
import math
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl

from ..models import ActionCoverage, ActionSnapshot, CorporateAction

NY = ZoneInfo('America/New_York')
_US_EXCHANGES = {'NYQ', 'NMS', 'NGM', 'NCM', 'ASE', 'PCX', 'BTS'}


class YahooActionProvider:
    def __init__(self, *, ticker_factory=None):
        self._ticker_factory = ticker_factory

    def fetch(self, tickers, start: date, reference_date: date,
              source_basis: str = 'unknown', reference_bars=None, coverage_starts=None) -> ActionSnapshot:
        """Fetch once on explicit request; processing later uses only the snapshot.

        ``reference_bars`` is a ticker-to-unadjusted-1m-frame mapping or callable.
        ``coverage_starts`` optionally narrows each ticker to the first selected
        raw date, so newer listings need not supply pre-listing history.
        No Yahoo quote is substituted for a missing raw regular-session close.
        Missing cash references leave pending events while retaining splits.
        """
        if type(start) is not date or type(reference_date) is not date or start > reference_date:
            raise ValueError('Choose a valid inclusive date range.')
        symbols = tuple(dict.fromkeys(s.strip().upper() for s in tickers if s.strip()))
        if not symbols or 'ALL_SYMBOLS' in symbols:
            raise ValueError('Choose explicit ticker symbols.')
        ActionCoverage(symbols[0], start, reference_date, source_basis=source_basis)
        fetched_at = datetime.now(UTC)
        if reference_date > fetched_at.astimezone(NY).date():
            raise ValueError('Yahoo cannot verify future corporate-action coverage.')
        factory = self._ticker_factory or importlib.import_module('yfinance').Ticker
        coverage, events = [], []
        for symbol in symbols:
            symbol_start = max(start, (coverage_starts or {}).get(symbol, start))
            if type(symbol_start) is not date or symbol_start > reference_date:
                raise ValueError(f'Invalid coverage start for {symbol}.')
            try:
                ticker = factory(symbol)
                history = ticker.history(period='max', interval='1d', auto_adjust=False,
                                         back_adjust=False, actions=True, repair=False,
                                         keepna=True, raise_errors=True)
                metadata = ticker.get_history_metadata()
                _validate_metadata(metadata, symbol)
                history = _validate_history(history, symbol_start, reference_date)
                cash = _cash_evidence(ticker, history)
                normalized = _events(history, cash, symbol, symbol_start, reference_date,
                                     source_basis, reference_bars, fetched_at)
                events.extend(normalized)
                coverage.append(ActionCoverage(
                    symbol, symbol_start, reference_date, 'complete',
                    'Yahoo reported splits/cash dividends checked; best-effort, not a guarantee of completeness. '
                    'Regular/special cash classifications and other corporate actions are not supplied. '
                    'Identity is ticker-based; historical ticker reuse/listing continuity is not verified.',
                    source_basis, f'yahoo:{symbol}', other_events_status='unknown'))
            except _EvidenceError as exc:
                coverage.append(ActionCoverage(symbol, symbol_start, reference_date, 'error', str(exc),
                                               source_basis, other_events_status='unknown'))
            except Exception as exc:
                # Vendor exceptions can contain request URLs or credentials.
                coverage.append(ActionCoverage(
                    symbol, symbol_start, reference_date, 'error',
                    f'Yahoo event lookup failed ({type(exc).__name__}); check symbol, connectivity or rate limits.',
                    source_basis, other_events_status='unknown'))
        return ActionSnapshot('yahoo-finance', reference_date, fetched_at, tuple(coverage), tuple(events))


class _EvidenceError(ValueError):
    pass


def _validate_metadata(metadata, symbol):
    if (not isinstance(metadata, Mapping) or metadata.get('symbol', '').upper() != symbol
            or metadata.get('instrumentType') != 'EQUITY'
            or metadata.get('currency') != 'USD'
            or metadata.get('exchangeTimezoneName') != 'America/New_York'
            or metadata.get('exchangeName') not in _US_EXCHANGES):
        raise _EvidenceError('Yahoo requires a matching US-listed equity, USD currency and New York exchange timezone.')


@lru_cache(maxsize=32)
def _schedule(start, end):
    import exchange_calendars as xcals
    return xcals.get_calendar('XNYS', start=str(start - timedelta(days=14)),
                              end=str(end + timedelta(days=7))).schedule.loc[str(start):str(end)]


def _validate_history(history, start, end):
    columns = {'Close', 'Dividends', 'Stock Splits'}
    if not isinstance(history, pd.DataFrame) or history.empty or not columns.issubset(history.columns):
        raise _EvidenceError('Yahoo returned empty or incomplete action history; absence of events is unverified.')
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise _EvidenceError('Yahoo action dates must have an exchange timezone.')
    history = history.copy()
    history.index = history.index.tz_convert(NY)
    if any(t.time() != time.min for t in history.index) or history.index.has_duplicates:
        raise _EvidenceError('Yahoo returned ambiguous or duplicate daily action dates.')
    history = history.sort_index()
    for column in ('Dividends', 'Stock Splits'):
        if any(not _finite(v) or float(v) < 0 for v in history[column]):
            raise _EvidenceError('Yahoo returned invalid split/dividend values; event coverage is unverified.')
    expected = {t.date() for t in _schedule(start, end).index}
    available = {t.date() for t, row in history.iterrows() if _positive(row['Close'])}
    if not expected or expected - available:
        raise _EvidenceError('Yahoo daily history does not cover every requested trading session; shorten the range or import verified coverage.')
    if 'Capital Gains' in history.columns and any(
        not _finite(v) or float(v) != 0 for v in history['Capital Gains']
    ):
        raise _EvidenceError('Yahoo returned capital-gain distributions requiring manual classification.')
    return history


def _cash_evidence(ticker, history):
    """Retain currency metadata discarded from yfinance's history dataframe."""
    relevant = history[history['Dividends'] > 0]
    if relevant.empty:
        return {}
    try:
        # Ticker's public wrapper has no repair keyword (PriceHistory does).
        # Its default delegates to unrepaired history.
        dividends = ticker.get_dividends(period='max')
        if not isinstance(dividends, (pd.Series, pd.DataFrame)) or dividends.empty:
            return None
        if not isinstance(dividends.index, pd.DatetimeIndex) or dividends.index.tz is None:
            return None
        records = {}
        for stamp, row in dividends.items() if isinstance(dividends, pd.Series) else dividends.iterrows():
            day = stamp.tz_convert(NY).date()
            amount = row if isinstance(dividends, pd.Series) else row['Dividends']
            currency = 'USD' if isinstance(dividends, pd.Series) else row.get('currency', 'USD')
            if day in records or not _positive(amount):
                return None
            records[day] = (float(amount), currency)
        # Separate queries must agree before their currency/basis evidence is combined.
        reported = {stamp.date(): float(value) for stamp, value in relevant['Dividends'].items()}
        if set(records) != set(reported) or any(
            not math.isclose(records[day][0], amount, rel_tol=1e-10, abs_tol=1e-12)
            for day, amount in reported.items()
        ):
            return None
        return records
    except Exception:
        return None


def _events(history, cash, symbol, start, end, source_basis, reference_bars, fetched_at):
    splits = [(stamp.date(), float(row['Stock Splits'])) for stamp, row in history.iterrows()
              if row['Stock Splits'] != 0]
    selected = history.loc[(history.index.date >= start) & (history.index.date <= end)]
    today = fetched_at.astimezone(NY).date()
    later_sessions = _schedule(start, today)
    completed = later_sessions[later_sessions['close'] <= pd.Timestamp(fetched_at)]
    available = {stamp.date() for stamp, row in history.iterrows() if _positive(row['Close'])}
    tail_complete = (not completed.empty
                     and {stamp.date() for stamp in completed.index} <= available
                     and not any(d > today for d, _ in splits))
    loaded = None
    load_error = ''
    if (selected['Dividends'] > 0).any():
        try:
            loaded = reference_bars(symbol) if callable(reference_bars) else (reference_bars or {}).get(symbol)
        except Exception:
            load_error = 'Could not read selected unadjusted minute data.'
    for stamp, row in selected.iterrows():
        day = stamp.date()
        common = dict(symbol=symbol, effective_at=datetime.combine(day, time.min, NY),
                      security_id=f'yahoo:{symbol}', source_ref=f'https://finance.yahoo.com/quote/{symbol}/history/')
        ratio = float(row['Stock Splits'])
        if ratio:
            ambiguous = not _simple_split(ratio)
            yield CorporateAction(event_id=f'yahoo:{symbol}:{day}:split',
                event_type='forward_split' if ratio >= 1 else 'reverse_split',
                share_ratio=ratio, price_factor=None if ambiguous else 1 / ratio,
                status='pending' if ambiguous else 'apply',
                details=('Yahoo also places spin-off adjustment factors in Stock Splits. '
                         'Non-integer or non-reciprocal ratio requires manual classification.'
                         if ambiguous else 'Yahoo reported simple share split; classification is best-effort.'), **common)
            if ambiguous:
                yield CorporateAction(event_id=f'yahoo:{symbol}:{day}:unclassified',
                    event_type='unclassified_yahoo_action', status='pending',
                    details=f'Yahoo Stock Splits={ratio:g}; may represent a spin-off or complex split. '
                            'No automatic price/volume adjustment; import verified classification.', **common)
        amount = float(row['Dividends'])
        if not amount:
            continue
        # Yahoo restates old cash amounts for subsequent splits, even those after
        # the user's chosen reference date. Never use a truncated split history.
        later_splits = [(d, r) for d, r in splits if d > day]
        restored = amount * math.prod(r for _, r in later_splits)
        reason = load_error
        currency = 'USD'
        if cash is None or day not in cash:
            reason = 'Dividend currency/basis evidence is missing or inconsistent between Yahoo responses.'
        elif cash[day][1] != 'USD':
            currency = str(cash[day][1] or 'UNKNOWN')
            reason = 'Dividend currency differs from USD; import a verified cash adjustment.'
        elif ratio:
            reason = 'Same-date split and cash dividend share basis is ambiguous in Yahoo; import a verified per-event factor.'
        elif not tail_complete:
            reason = 'Yahoo full history is stale or truncated; subsequent split coverage through the latest completed session is unverified.'
        elif any(not _simple_split(r) for d, r in splits if d > day):
            reason = 'Later Yahoo split history includes an ambiguous factor; cannot restore historical cash share basis safely.'
        elif source_basis != 'unadjusted':
            reason = 'Dividend factors require verified unadjusted minute input.'
        close = None
        if not reason:
            try:
                close = _previous_close(loaded, symbol, day)
            except _EvidenceError as exc:
                reason = str(exc)
        if not _positive(restored):
            reason = 'Restored cash amount is invalid; verify Yahoo split history.'
            restored = None
        factor = None
        if not reason:
            factor = (close - restored) / close
            if not 0 < factor < 1:
                reason = 'Cash amount must be below the preceding raw regular-session close; verify distribution and share basis.'
                factor = None
        split_evidence = ', '.join(f'{d.isoformat()}={r:g}' for d, r in later_splits) or 'none'
        detail = (f'Yahoo reported cash={amount:g}; candidate historical amount calculated using subsequent split ratios '
                  f'(new/old): {split_evidence}. '
                  'Yahoo does not distinguish regular from special cash dividends. ')
        yield CorporateAction(event_id=f'yahoo:{symbol}:{day}:cash', event_type='cash_dividend',
            cash_amount=restored, reference_close=close, price_factor=factor, currency=currency,
            status='pending' if reason else 'apply', details=detail + reason, **common)


def _previous_close(frame, symbol, exdate):
    if not isinstance(frame, pl.DataFrame) or not {'timestamp', 'symbol', 'close'} <= set(frame.columns):
        raise _EvidenceError('Missing unadjusted 1m data for the preceding regular-session close.')
    sessions = _schedule(exdate - timedelta(days=14), exdate - timedelta(days=1))
    if sessions.empty:
        raise _EvidenceError('Cannot determine the preceding exchange session.')
    expected = sessions.iloc[-1]['close'].to_pydatetime() - timedelta(minutes=1)
    if frame.schema['timestamp'].base_type() != pl.Datetime or frame.schema['timestamp'].time_zone != 'UTC':
        raise _EvidenceError('Reference minute timestamps must be timezone-aware UTC.')
    match = frame.filter((pl.col('symbol') == symbol) & (pl.col('timestamp') == expected))
    if match.height != 1 or not _positive(match['close'][0]):
        raise _EvidenceError(f'Missing or ambiguous final regular-session minute at {expected.isoformat()}; import a verified factor or include the preceding session.')
    return float(match['close'][0])


def _finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive(value):
    return _finite(value) and float(value) > 0


def _simple_split(ratio):
    """Conservative heuristic, not proof of corporate-action classification."""
    value = ratio if ratio >= 1 else 1 / ratio
    return value >= 2 and math.isclose(value, round(value), rel_tol=1e-10, abs_tol=1e-10)
