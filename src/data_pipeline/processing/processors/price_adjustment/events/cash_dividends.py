"""Cash distributions change price basis, never historical share volume."""

import math

from .....exceptions import ProcessingError


def cash_dividend_factors(event, *, simultaneous_split: bool = False) -> tuple[float, float]:
    """Use an event-only verified factor, or a same-share-basis reference close.

    reference_close is the previous regular-session close, in USD and in the
    same share units as cash_amount. Combined split/dividend events must supply
    an explicit factor to avoid guessing which share units the amount uses.
    """
    if event.currency != "USD":
        raise ProcessingError(f"Cash dividend {event.event_id!r} requires USD currency in this version.")
    amount = event.cash_amount
    if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0):
        raise ProcessingError(f"Cash dividend {event.event_id!r} requires a positive finite cash_amount.")
    if event.price_factor is not None:
        factor = event.price_factor
    else:
        if simultaneous_split:
            raise ProcessingError(
                f"Cash dividend {event.event_id!r} shares a date with a split; supply a verified event-only price_factor."
            )
        close = event.reference_close
        if amount is None or type(close) not in (int, float) or not math.isfinite(close) or close <= 0:
            raise ProcessingError(
                f"Cash dividend {event.event_id!r} requires cash_amount and a verified same-basis reference_close, "
                "or a verified price_factor."
            )
        factor = (close - amount) / close
    if type(factor) not in (int, float) or not math.isfinite(factor) or not 0 < factor < 1:
        raise ProcessingError(f"Cash dividend {event.event_id!r} price_factor must be strictly between zero and one.")
    return float(factor), 1.0
