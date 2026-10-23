"""Immutable, serializable corporate-action evidence for US equity adjustments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from zoneinfo import ZoneInfo
from collections.abc import Mapping

SPLIT_TYPES = frozenset({'forward_split', 'reverse_split', 'stock_dividend'})
CASH_DIVIDEND_TYPES = frozenset({'cash_dividend', 'special_cash_dividend'})
SUPPORTED_TYPES = SPLIT_TYPES | CASH_DIVIDEND_TYPES


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string.')


def _aware(value, name):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{name} must be timezone aware.')


def _date(value, name):
    if type(value) is not date:
        raise ValueError(f'{name} must be a date.')


@dataclass(frozen=True)
class CorporateAction:
    """One resolved event/benefit, with a verified *per-event* price multiplier.

    ``share_ratio`` means new shares / old shares. ``reference_close`` must be
    the preceding regular-session close in the dividend's currency/share basis.
    ``effective_at`` is midnight of the exchange-local ex-date, including the
    premarket boundary. Event IDs must uniquely identify the selected benefit;
    alternative elections must be resolved before creating a snapshot.
    """
    event_id: str
    symbol: str
    event_type: str
    effective_at: datetime
    security_id: str = ''
    price_factor: float | None = None
    share_ratio: float | None = None
    cash_amount: float | None = None
    reference_close: float | None = None
    currency: str = 'USD'
    status: str = 'apply'
    details: str = ''
    source_ref: str = ''

    def __post_init__(self):
        for name in ('event_id', 'symbol', 'event_type', 'currency'):
            _text(getattr(self, name), name)
        object.__setattr__(self, 'symbol', self.symbol.strip().upper())
        _aware(self.effective_at, 'effective_at')
        if self.event_type in SUPPORTED_TYPES:
            local = self.effective_at.astimezone(ZoneInfo('America/New_York'))
            if any((local.hour, local.minute, local.second, local.microsecond)):
                raise ValueError('Supported event effective_at must be midnight America/New_York on its ex-date.')
        if self.status not in {'apply', 'pending', 'cancelled'}:
            raise ValueError('Unknown event status.')
        for name in ('price_factor', 'share_ratio', 'cash_amount', 'reference_close'):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive.')
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class ActionCoverage:
    """Inclusive dates checked for supported adjustment events.

    ``other_events_status`` separately describes coverage of report-only events.
    ``None`` preserves legacy snapshots whose status covered all event types.
    Neither scope treats unknown coverage as evidence that no events occurred.
    """
    symbol: str
    start: date
    end: date
    status: str = 'complete'
    details: str = ''
    source_basis: str = 'unknown'
    security_id: str = ''
    other_events_status: str | None = None

    def __post_init__(self):
        _text(self.symbol, 'symbol')
        object.__setattr__(self, 'symbol', self.symbol.strip().upper())
        _date(self.start, 'start')
        _date(self.end, 'end')
        if self.end < self.start:
            raise ValueError('Coverage end precedes start.')
        if self.status not in {'complete', 'partial', 'unknown', 'error'}:
            raise ValueError('Unknown coverage status.')
        if self.other_events_status is not None and self.other_events_status not in {'complete', 'partial', 'unknown', 'error'}:
            raise ValueError('Unknown other-events coverage status.')
        if self.source_basis not in {'unadjusted', 'split_adjusted', 'dividend_adjusted', 'vendor_adjusted', 'unknown'}:
            raise ValueError('Unknown source basis.')


@dataclass(frozen=True)
class ActionSnapshot:
    source: str
    reference_date: date
    fetched_at: datetime
    coverage: tuple[ActionCoverage, ...]
    events: tuple[CorporateAction, ...]
    methodology_version: str = '1'

    def __post_init__(self):
        _text(self.source, 'source')
        _text(self.methodology_version, 'methodology_version')
        _date(self.reference_date, 'reference_date')
        _aware(self.fetched_at, 'fetched_at')
        if any(not isinstance(item, ActionCoverage) for item in self.coverage):
            raise ValueError('Invalid coverage record.')
        if any(not isinstance(item, CorporateAction) for item in self.events):
            raise ValueError('Invalid event record.')
        object.__setattr__(self, 'coverage', tuple(sorted(self.coverage, key=lambda c: c.symbol)))
        object.__setattr__(self, 'events', tuple(sorted(self.events, key=lambda e: (e.symbol, e.effective_at, e.event_id))))
        if len({c.symbol for c in self.coverage}) != len(self.coverage):
            raise ValueError('Duplicate coverage symbol.')
        identities = [(e.symbol, e.event_id) for e in self.events]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate event identity; resolve revisions/options first.')

    def to_dict(self) -> dict:
        def encode(value):
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            return value
        fields = asdict(self)
        # Persisted snapshots are content-addressed. Preserve their original
        # canonical representation when this optional scope was not supplied.
        for coverage in fields['coverage']:
            if coverage['other_events_status'] is None:
                del coverage['other_events_status']
        return encode(fields)

    @classmethod
    def from_dict(cls, value: Mapping) -> ActionSnapshot:
        if not isinstance(value, Mapping):
            raise ValueError('Snapshot must be an object.')
        try:
            fields = dict(value)
            fields['reference_date'] = date.fromisoformat(fields['reference_date'])
            fields['fetched_at'] = datetime.fromisoformat(fields['fetched_at'])
            fields['coverage'] = tuple(ActionCoverage(**{
                **c, 'start': date.fromisoformat(c['start']), 'end': date.fromisoformat(c['end']),
            }) for c in fields['coverage'])
            fields['events'] = tuple(CorporateAction(**{
                **e, 'effective_at': datetime.fromisoformat(e['effective_at']),
            }) for e in fields['events'])
            return cls(**fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid corporate-action snapshot: {exc}') from exc

    @property
    def checksum(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
