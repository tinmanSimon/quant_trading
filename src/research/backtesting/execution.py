"""Exact decimal accounting and conservative affordability checks."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_UP

from .models import ExecutionSettings

ZERO = Decimal(0)
ONE = Decimal(1)
BPS = Decimal(10000)
FEE_QUANTUM = Decimal("0.00000001")
MAX_SHARES = 2**63 - 1


@dataclass(frozen=True, slots=True)
class Fill:
    side: str
    requested_quantity: int
    quantity: int
    price: Decimal
    fees: Decimal
    cash: Decimal
    holdings: int
    status: str
    reason: str


def fee(notional: Decimal, settings: ExecutionSettings) -> Decimal:
    return (settings.commission_fixed + notional * settings.commission_bps / BPS).quantize(FEE_QUANTUM, rounding=ROUND_UP)


def rebalance(cash: Decimal, holdings: int, open_price: Decimal, target: float, settings: ExecutionSettings) -> Fill | None:
    """Size at the execution bar's open; shrink buys to their affordable size.

    Strategy allocation is decided earlier. Execution-time sizing is allowed
    to inspect the open only, never the bar's close/high/low/volume.
    """
    equity = cash + holdings * open_price
    desired = int((Decimal(str(target)) * equity / open_price).to_integral_value(rounding=ROUND_FLOOR))
    if desired > MAX_SHARES:
        raise ValueError("Requested share count exceeds supported Int64 range.")
    delta = desired - holdings
    if delta == 0:
        return None
    side = "buy" if delta > 0 else "sell"
    requested = abs(delta)
    price = open_price * (ONE + settings.slippage_bps / BPS if delta > 0 else ONE - settings.slippage_bps / BPS)
    if delta > 0:
        # Binary search includes actual rounded commission in every candidate.
        low, high = 0, requested
        while low < high:
            mid = (low + high + 1) // 2
            cost = mid * price + fee(mid * price, settings)
            if cost <= cash:
                low = mid
            else:
                high = mid - 1
        quantity = low
        if not quantity:
            return Fill(side, requested, 0, price, ZERO, cash, holdings, "rejected", "insufficient_cash")
        fees = fee(quantity * price, settings)
        after_cash = cash - quantity * price - fees
        after_holdings = holdings + quantity
    else:
        quantity = min(requested, holdings)
        fees = fee(quantity * price, settings)
        after_cash = cash + quantity * price - fees
        if after_cash < ZERO:
            return Fill(side, requested, 0, price, ZERO, cash, holdings, "rejected", "sale_proceeds_cannot_cover_fees")
        after_holdings = holdings - quantity
    if after_cash < ZERO or after_holdings < 0:
        raise ArithmeticError("Execution violated the nonnegative cash/holdings invariant.")
    status = "filled" if quantity == requested else "reduced"
    return Fill(side, requested, quantity, price, fees, after_cash, after_holdings, status, "affordability_limit" if status == "reduced" else "")
