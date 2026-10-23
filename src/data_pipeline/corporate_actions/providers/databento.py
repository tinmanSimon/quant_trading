"""Explicit Databento Reference API retrieval; never called by a processor.

Schema: https://databento.com/docs/schemas/corporate-actions
Factors: https://databento.com/docs/datasets/adjustment-factors

Only unambiguous US/USD listings, mandatory split benefits and cash dividends
are normalized for adjustment. Other events are retained for reporting. PIT
revisions are selected by event_unique_id, not event_id (which joins elections).
A missing entitlement, ambiguous listing or partial history fails closed.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
import importlib
import json
import math
import os
import re
from zoneinfo import ZoneInfo

from ..models import ActionCoverage, ActionSnapshot, CorporateAction, SUPPORTED_TYPES, SPLIT_TYPES

HISTORY_START = date(2018, 5, 1)
NY = ZoneInfo('America/New_York')


class ReferenceRequestError(RuntimeError):
    """Safe request diagnostics without vendor URLs, payloads or credentials."""

_TYPE = {'FSPLT': 'forward_split', 'SD': 'forward_split', 'RSPLT': 'reverse_split',
         'CONSD': 'reverse_split', 'DMRGR': 'spin_off', 'RTS': 'rights_issue',
         'MRGR': 'merger', 'TKOVR': 'acquisition', 'CONV': 'share_exchange',
         'DRCHG': 'adr_ratio_change', 'CURRD': 'currency_redenomination',
         'LCC': 'symbol_change', 'DELIST': 'delisting', 'RCAP': 'return_of_capital'}


class DatabentoActionProvider:
    def __init__(self, *, api_key=None, client=None):
        self._api_key = api_key
        self._client = client

    def _get_client(self):
        if self._client is None:
            key = self._api_key or os.environ.get('DATABENTO_API_KEY', '').strip()
            if not key:
                raise ValueError('Set DATABENTO_API_KEY or import a verified snapshot.')
            try:
                self._client = importlib.import_module('databento').Reference(key=key)
            except Exception:
                raise ValueError('Could not initialize Databento Reference client.') from None
        return self._client

    def fetch(self, tickers: Iterable[str], start: date, reference_date: date,
              source_basis: str = 'unknown') -> ActionSnapshot:
        """Fetch a current evidence snapshot, independently reporting ticker errors.

        The selected reference date limits effective events, not knowledge time.
        This is retrospective evidence. Calls may incur reference-data charges;
        callers must expose this explicit action, never fetch during transforms.
        """
        if type(start) is not date or type(reference_date) is not date or start > reference_date:
            raise ValueError('Choose a valid inclusive date range.')
        symbols = tuple(dict.fromkeys(s.strip().upper() for s in tickers if s.strip()))
        if not symbols or any(s == 'ALL_SYMBOLS' for s in symbols):
            raise ValueError('Choose explicit ticker symbols.')
        # Validate the basis even if every vendor request subsequently fails.
        ActionCoverage(symbols[0], start, reference_date, source_basis=source_basis)
        client = self._get_client()
        fetched_at = datetime.now(UTC)
        coverage, events = [], []
        for symbol in symbols:
            if start < HISTORY_START:
                coverage.append(ActionCoverage(symbol, start, reference_date, 'partial',
                    'Databento reference history begins 2018-05-01; import verified older coverage.', source_basis))
                continue
            try:
                kwargs = dict(symbols=[symbol], stype_in='raw_symbol', countries=['US'])
                # Identity history catches ticker reuse. No latest-only inference.
                security = _records(_request(client.security_master,
                    start=HISTORY_START, end=reference_date + timedelta(days=1), **kwargs))
                identity = _identity(security, symbol)
                common = dict(start=start, end=reference_date + timedelta(days=1), **kwargs)
                # event_date retains non-distribution events with null ex_date;
                # ex_date additionally captures actions whose event_date differs.
                raw = _records(_request(client.corporate_actions, index='event_date', pit=True, flatten=False, **common))
                raw += _records(_request(client.corporate_actions, index='ex_date', pit=True, flatten=False, **common))
                try:
                    factors = _records(_request(client.adjustment_factors, **common))
                except RuntimeError:
                    # Event lookup succeeded: retain known other events and leave
                    # cash adjustments pending instead of erasing the report.
                    factors = []
                normalized = normalize_records(raw, factors, symbol=symbol, security_id=identity['security_id'],
                                               listing_id=identity['listing_id'], exchange=identity['exchange'],
                                               start=start, reference_date=reference_date, fetched_at=fetched_at, allow_incomplete=True)
                events.extend(normalized)
                coverage.append(ActionCoverage(symbol, start, reference_date, 'complete',
                    'All event types checked; current snapshot, not point-in-time trading evidence.',
                    source_basis, identity['security_id']))
            except ReferenceRequestError as exc:
                coverage.append(ActionCoverage(symbol, start, reference_date, 'error', str(exc), source_basis))
            except ValueError as exc:
                coverage.append(ActionCoverage(symbol, start, reference_date, 'error', str(exc), source_basis))
            except Exception:
                # SDK exceptions can expose credentials or signed URLs.
                coverage.append(ActionCoverage(symbol, start, reference_date, 'error',
                    'Databento reference request failed; check credentials, coverage and reference-data entitlement.', source_basis))
        return ActionSnapshot('databento-reference', reference_date, fetched_at, tuple(coverage), tuple(events))


def _request(api, **kwargs):
    try:
        return api.get_range(**kwargs)
    except Exception as exc:
        # Preserve structured error codes only; arbitrary exception text or
        # vendor payloads can contain credentials and signed URLs.
        status = getattr(exc, 'http_status', None)
        body = getattr(exc, 'json_body', None)
        detail = body.get('detail', {}) if isinstance(body, dict) else {}
        case = detail.get('case', '') if isinstance(detail, dict) else ''
        case = case if isinstance(case, str) and re.fullmatch(r'[a-z_]{1,100}', case) else ''
        message = 'Databento reference request failed'
        if type(status) is int:
            message += f' (HTTP {status}' + (f'; {case}' if case else '') + ')'
        if case == 'license_reference_dataset_no_subscription':
            payload = detail.get('payload', {})
            dataset = payload.get('reference_dataset') if isinstance(payload, dict) else None
            dataset = dataset if dataset in {'security master', 'corporate actions', 'adjustment factors'} else 'requested reference'
            message += f': a subscription is required for the {dataset} dataset.'
        else:
            message += '. Check reference-data access and network connectivity.'
        raise ReferenceRequestError(message) from None


def _records(frame) -> list[dict]:
    if isinstance(frame, list):
        return [_clean(row) for row in frame]
    if frame.empty:
        return []
    if frame.index.name is not None and frame.index.name not in frame.columns:
        frame = frame.reset_index()
    return [_clean(row) for row in frame.to_dict(orient='records')]


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if value is None or str(value) in {'NaT', '<NA>'}:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _identity(rows, symbol):
    matches = [r for r in rows if r.get('symbol') == symbol and r.get('trading_currency') == 'USD'
               and r.get('exchange') and r.get('exchange') == r.get('primary_exchange')]
    identities = {(str(r.get('security_id') or ''), str(r.get('listing_id') or ''), r.get('exchange')) for r in matches}
    if len(identities) != 1 or not all(next(iter(identities), ('', '', ''))):
        raise ValueError('Security identity missing or ambiguous across requested history; import verified listing mapping.')
    security_id, listing_id, exchange = next(iter(identities))
    return dict(security_id=security_id, listing_id=listing_id, exchange=exchange)


def _timestamp(value):
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Vendor revision timestamps must be timezone aware.')
    return result


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    return value if type(value) is date else date.fromisoformat(str(value))


def normalize_records(records, factors, *, symbol, security_id, listing_id, exchange,
                      start, reference_date, fetched_at, allow_incomplete=False) -> tuple[CorporateAction, ...]:
    """Normalize captured SDK rows offline; preserve latest revision and audit trail."""
    latest, revisions = {}, defaultdict(list)
    for row in records:
        if str(row.get('listing_id')) != listing_id:
            continue
        if str(row.get('security_id')) != security_id or row.get('symbol') != symbol:
            raise ValueError('Corporate action does not match verified security mapping.')
        uid = row.get('event_unique_id')
        if not uid or not row.get('event_id'):
            raise ValueError('Corporate action is missing record/event identity.')
        stamp = _timestamp(row.get('ts_record'))
        if stamp > fetched_at:
            continue
        revisions[uid].append({'ts_record': stamp.isoformat(), 'event_action': row.get('event_action')})
        if uid in latest and stamp == _timestamp(latest[uid]['ts_record']) and row != latest[uid]:
            raise ValueError('Conflicting corporate-action revisions.')
        if uid not in latest or stamp >= _timestamp(latest[uid]['ts_record']):
            latest[uid] = row
    groups = defaultdict(list)
    for uid, row in latest.items():
        groups[str(row['event_id'])].append((uid, row))
    output = []
    for event_id, options in groups.items():
        active = [(uid, r) for uid, r in options if r.get('event_action') not in {'C', 'D', 'P', 'Q'}]
        # Multiple serial benefits or shareholder elections need explicit review.
        if len(active) > 1:
            raise ValueError(f'Event {event_id} has multiple benefits/elections; import a verified selection.')
        selected = active or options[:1]
        uid, row = selected[0]
        action = row.get('event_action')
        if action not in {'I', 'U', 'C', 'D', 'P', 'Q'}:
            raise ValueError('Unknown corporate-action revision status.')
        status = 'cancelled' if action in {'C', 'D', 'P', 'Q'} else 'apply'
        info, rates = row.get('event_info') or {}, row.get('rate_info') or {}
        event_type = _classify(row)
        effective = row.get('ex_date') if event_type in SUPPORTED_TYPES else (row.get('effective_date') or row.get('event_date') or row.get('ex_date'))
        if effective is None:
            raise ValueError(f'Event {event_id} is missing its effective date.')
        effective = _day(effective)
        if not start <= effective <= reference_date:
            continue
        ratio, cash, price_factor = None, None, None
        unresolved = ''
        try:
            if status == 'apply' and event_type in SUPPORTED_TYPES:
                if row.get('trading_currency') != 'USD':
                    raise ValueError('Only verified USD trading currency is supported.')
                if event_type in SPLIT_TYPES:
                    old, new = row.get('ratio_old'), row.get('ratio_new')
                    if old is None or new is None or float(old) <= 0 or float(new) <= 0:
                        raise ValueError(f'Event {event_id} has no verified share ratio.')
                    ratio = float(new) / float(old)
                    price_factor = 1 / ratio
                else:
                    cash = rates.get('gross_dividend', row.get('gross_dividend'))
                    if row.get('rate_currency') != 'USD':
                        raise ValueError('Dividend currency is not verified as USD.')
                    factor = _cash_factor(factors, row, effective, fetched_at, exchange)
                    status = {'A': 'apply', 'R': 'pending', 'P': 'pending'}[factor['status']]
                    if status == 'apply':
                        price_factor = float(factor['factor'])
                        if not 0 < price_factor < 1:
                            raise ValueError('Cash dividend factor must be between zero and one; placeholder factors are not usable.')
                    if cash is not None:
                        cash = float(cash)
                        if cash <= 0:
                            raise ValueError('Cash dividend amount must be positive.')
        except (ValueError, TypeError) as exc:
            if not allow_incomplete:
                raise
            status, price_factor, ratio, cash = 'pending', None, None, None
            unresolved = str(exc)
        audit = {'vendor_event': row.get('event'), 'subtype': row.get('event_subtype'),
                 'record_id': uid, 'listing_id': listing_id, 'revisions': revisions[uid],
                 'payment_date': row.get('payment_date'), 'option_id': row.get('option_id'),
                 'serial_id': row.get('serial_id'), 'event_info': info, 'unresolved': unresolved,
                 'adjustment_records': [f for f in factors if str(f.get('event_id')) == event_id
                                        and str(f.get('security_id')) == security_id]}
        output.append(CorporateAction(event_id, symbol, event_type,
            datetime.combine(effective, time(), NY), security_id, price_factor, ratio, cash,
            status=status, details=json.dumps(audit, default=str, sort_keys=True),
            source_ref='https://databento.com/docs/schemas/corporate-actions'))
    return tuple(output)


def _classify(row):
    code, subtype = row.get('event'), row.get('event_subtype')
    if code == 'FSPLT' and subtype in {'DIV', 'BON'}:
        return ('stock_dividend' if row.get('outturn_security_id') == row.get('security_id')
                else 'unsupported_stock_distribution')
    if code in _TYPE:
        return _TYPE[code]
    if code == 'DIV':
        if subtype == 'CAPDIST':
            return 'return_of_capital'
        if row.get('payment_type') == 'C' and subtype is None:
            return 'special_cash_dividend' if (row.get('event_info') or {}).get('marker') == 'SPL' else 'cash_dividend'
        if row.get('payment_type') == 'S' and subtype == 'DIV' and row.get('outturn_security_id') == row.get('security_id'):
            return 'stock_dividend'
        return 'unsupported_dividend'
    return str(code or 'unknown').lower()


def _cash_factor(factors, row, effective, fetched_at, exchange):
    # Rescinds remove a factor value; a replacement can have another value.
    candidates = [f for f in factors if str(f.get('event_id')) == str(row['event_id'])
                  and str(f.get('security_id')) == str(row['security_id'])
                  and f.get('exchange') == exchange and _day(f.get('ex_date')) == effective
                  and _timestamp(f.get('ts_created')) <= fetched_at]
    if not candidates:
        raise ValueError(f"Missing verified cash-dividend factor for event {row['event_id']}.")
    if any(int(f.get('reason', -1)) != 17 or f.get('dividend_currency') != 'USD' for f in candidates):
        raise ValueError('Dividend factor combines unsupported events or currencies.')
    if len({f.get('option') for f in candidates}) != 1:
        raise ValueError('Dividend has multiple factor options; import a verified selection.')
    by_value = {}
    for factor in sorted(candidates, key=lambda f: _timestamp(f['ts_created'])):
        if factor.get('status') not in {'A', 'P', 'R'}:
            raise ValueError('Unknown dividend factor status.')
        key = float(factor['factor'])
        old = by_value.get(key)
        if old and old['ts_created'] == factor['ts_created'] and old['status'] != factor['status']:
            raise ValueError('Conflicting dividend factor revisions.')
        by_value[key] = factor
    active = [f for f in by_value.values() if f['status'] != 'R']
    if len(active) > 1:
        raise ValueError('Ambiguous unreconciled dividend factor revisions.')
    return active[0] if active else max(by_value.values(), key=lambda f: _timestamp(f['ts_created']))
