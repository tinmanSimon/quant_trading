"""Batch-wide reporting, including requested symbols without selected bars."""
from collections.abc import Iterable
from zoneinfo import ZoneInfo
from .models import ActionSnapshot, SUPPORTED_TYPES


def other_events_report(snapshot: ActionSnapshot, tickers: Iterable[str]) -> list[dict]:
    rows = []
    coverage = {c.symbol: c for c in snapshot.coverage}
    for symbol in dict.fromkeys(t.strip().upper() for t in tickers):
        cov = coverage.get(symbol)
        other_status = (cov.other_events_status or cov.status) if cov else 'unknown'
        base = {'ticker': symbol, 'coverage': other_status,
                'supported_events_coverage': cov.status if cov else 'unknown',
                'checked_start': cov.start.isoformat() if cov else '',
                'checked_end': cov.end.isoformat() if cov else '', 'source': snapshot.source}
        events = [e for e in snapshot.events if e.symbol == symbol and e.event_type not in SUPPORTED_TYPES
                  and e.effective_at.astimezone(ZoneInfo('America/New_York')).date() <= snapshot.reference_date]
        for event in events:
            rows.append({**base, 'effective_date': event.effective_at.astimezone(ZoneInfo('America/New_York')).date().isoformat(),
                         'event_type': event.event_type, 'event_id': event.event_id, 'details': event.details,
                         'handling': 'Cancelled; not adjusted' if event.status == 'cancelled' else 'Reported; not adjusted',
                         'status': event.status, 'source_ref': event.source_ref})
        if not events or other_status != 'complete':
            complete = other_status == 'complete'
            rows.append({**base, 'effective_date': '', 'event_type': 'no_other_events' if complete else 'coverage_unknown',
                         'event_id': '', 'details': cov.details if cov else 'No event coverage supplied for this ticker.',
                         'handling': 'No other events found in checked range' if complete else 'Event coverage incomplete; cannot establish absence',
                         'status': 'complete' if complete else 'unknown', 'source_ref': ''})
    return rows
