"""Event factor composition and efficient timestamp lookup."""

from collections import defaultdict
from datetime import UTC
import math

import polars as pl

from ....corporate_actions.models import SPLIT_TYPES, CASH_DIVIDEND_TYPES
from ....exceptions import ProcessingError
from .events.cash_dividends import cash_dividend_factors
from .events.splits import split_factors
from .validation import EXCHANGE_ZONE


def event_factors(events):
    splits = {event.effective_at.astimezone(EXCHANGE_ZONE).date()
              for event in events if event.event_type in SPLIT_TYPES}
    # Separate cash distributions on the same date need verified event-only
    # factors: multiplying independently derived (P-D)/P factors would be wrong.
    cash_counts = defaultdict(int)
    for event in events:
        if event.event_type in CASH_DIVIDEND_TYPES:
            cash_counts[event.effective_at.astimezone(EXCHANGE_ZONE).date()] += 1
    result = []
    for event in sorted(events, key=lambda value: (value.effective_at, value.event_id)):
        day = event.effective_at.astimezone(EXCHANGE_ZONE).date()
        if event.event_type in SPLIT_TYPES:
            factors = split_factors(event)
        elif event.event_type in CASH_DIVIDEND_TYPES:
            if cash_counts[day] > 1 and event.price_factor is None:
                raise ProcessingError(
                    f"Multiple cash dividends on {day} require verified sequential event-only price factors."
                )
            factors = cash_dividend_factors(event, simultaneous_split=day in splits)
        else:
            continue
        result.append((event.effective_at.astimezone(UTC), *factors))
    return result


def attach_factors(frame, factors):
    """Join post-boundary cumulative factors; bars exactly on a boundary exclude it."""
    if not factors:
        return frame.with_columns(pl.lit(1.).alias("_price_factor"), pl.lit(1.).alias("_volume_factor"))
    grouped = defaultdict(list)
    for stamp, price, volume in factors:
        grouped[stamp].append((price, volume))
    price = volume = 1.
    rows = []
    for stamp, values in sorted(grouped.items(), reverse=True):
        rows.append((stamp, price, volume))
        price *= math.prod(pair[0] for pair in values)
        volume *= math.prod(pair[1] for pair in values)
        if not all(math.isfinite(factor) and factor > 0 for factor in (price, volume)):
            raise ProcessingError("Cumulative adjustment factors overflowed or underflowed.")
    schedule = pl.DataFrame(rows[::-1], schema={"_effective": pl.Datetime("ms", "UTC"),
                           "_price_factor": pl.Float64, "_volume_factor": pl.Float64}, orient="row")
    return frame.join_asof(schedule, left_on="timestamp", right_on="_effective", strategy="backward").with_columns(
        pl.col("_price_factor").fill_null(price), pl.col("_volume_factor").fill_null(volume))
