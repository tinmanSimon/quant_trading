"""Invocation-local coverage and source-basis checks."""

from zoneinfo import ZoneInfo

from ....corporate_actions.models import SUPPORTED_TYPES
from ....exceptions import ProcessingError


EXCHANGE_ZONE = ZoneInfo("America/New_York")


def relevant_events(snapshot, symbol, earliest):
    return tuple(event for event in snapshot.events
                 if event.symbol == symbol and event.status != "cancelled"
                 and earliest < event.effective_at
                 and event.effective_at.astimezone(EXCHANGE_ZONE).date() <= snapshot.reference_date)


def validate_symbol(snapshot, symbol, earliest, latest, source_basis):
    first_date = earliest.astimezone(EXCHANGE_ZONE).date()
    last_date = latest.astimezone(EXCHANGE_ZONE).date()
    if last_date > snapshot.reference_date:
        raise ProcessingError(f"{symbol}: input contains bars after the adjustment reference date.")
    coverage = [item for item in snapshot.coverage if item.symbol == symbol]
    if len(coverage) != 1:
        raise ProcessingError(f"{symbol}: exactly one corporate-action coverage record is required.")
    coverage = coverage[0]
    if coverage.status != "complete" or coverage.start > first_date or coverage.end < snapshot.reference_date:
        raise ProcessingError(
            f"{symbol}: complete event coverage is required from {first_date} through {snapshot.reference_date}."
        )
    if coverage.source_basis not in ("unknown", "unadjusted") or (source_basis == "unknown" and coverage.source_basis != "unadjusted"):
        raise ProcessingError(f"{symbol}: verify the source price and volume basis as unadjusted before adjusting.")
    if not coverage.security_id:
        raise ProcessingError(f"{symbol}: verified security_id mapping is required in event coverage.")
    events = relevant_events(snapshot, symbol, earliest)
    seen = set()
    for event in events:
        if event.security_id != coverage.security_id:
            raise ProcessingError(f"{symbol}: event {event.event_id!r} security_id does not match coverage.")
        if event.event_type in SUPPORTED_TYPES and event.status != "apply":
            raise ProcessingError(f"{symbol}: supported event {event.event_id!r} is unresolved ({event.status}).")
        if event.event_type in SUPPORTED_TYPES:
            signature = (event.event_type, event.effective_at, event.share_ratio,
                         event.cash_amount, event.price_factor, event.reference_close)
            if signature in seen:
                raise ProcessingError(f"{symbol}: duplicate supported event terms; resolve duplicate source records first.")
            seen.add(signature)
    return events
