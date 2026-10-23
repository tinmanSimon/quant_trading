"""Splits express historical trades in post-event share units."""

import math

from .....exceptions import ProcessingError


def split_factors(event) -> tuple[float, float]:
    """Return price/volume factors; share_ratio is new shares / old shares."""
    ratio = event.share_ratio
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
        raise ProcessingError(f"Split {event.event_id!r} requires a positive finite share_ratio.")
    if event.event_type in ("forward_split", "stock_dividend") and ratio <= 1:
        raise ProcessingError(f"Split {event.event_id!r} requires share_ratio greater than one.")
    if event.event_type == "reverse_split" and ratio >= 1:
        raise ProcessingError(f"Reverse split {event.event_id!r} requires share_ratio below one.")
    factor = 1.0 / ratio
    if event.price_factor is not None and not math.isclose(event.price_factor, factor, rel_tol=1e-12):
        raise ProcessingError(f"Split {event.event_id!r} price_factor conflicts with its share_ratio.")
    return factor, float(ratio)
