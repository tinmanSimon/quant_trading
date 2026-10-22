"""Exact decimal accounting and conservative affordability checks."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_UP, localcontext

from .models import ExecutionSettings

ZERO = Decimal(0)
ONE = Decimal(1)
BPS = Decimal(10000)
FEE_QUANTUM = Decimal("0.00000001")
SHARE_QUANTUM = Decimal("0.00000001")
MAX_SHARES = 2**63 - 1


@dataclass(frozen=True, slots=True)
class Fill:
    side: str
    requested_quantity: int | Decimal
    quantity: int | Decimal
    price: Decimal
    fees: Decimal
    cash: Decimal
    holdings: int | Decimal
    status: str
    reason: str


def fee(notional: Decimal, settings: ExecutionSettings) -> Decimal:
    return (settings.commission_fixed + notional * settings.commission_bps / BPS).quantize(FEE_QUANTUM, rounding=ROUND_UP)


def rebalance(cash: Decimal, holdings: int | Decimal, open_price: Decimal, target: float, settings: ExecutionSettings) -> Fill | None:
    """Size at the execution bar's open; shrink buys to their affordable size.

    Strategy allocation is decided earlier. Execution-time sizing is allowed
    to inspect the open only, never the bar's close/high/low/volume.
    """
    if settings.fractional_shares:
        with localcontext() as context:
            context.prec = 800
            return _rebalance_fractional(cash, holdings, open_price, target, settings)
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


def _rebalance_fractional(cash: Decimal, holdings: int | Decimal, open_price: Decimal, target: float, settings: ExecutionSettings) -> Fill | None:
    holdings = Decimal(holdings)
    target_value = Decimal(str(target)) * (cash + holdings * open_price)
    desired = (target_value / open_price).quantize(SHARE_QUANTUM, rounding=ROUND_FLOOR)
    if desired > MAX_SHARES:
        raise ValueError("Requested share count exceeds supported Int64 range.")
    delta = desired - holdings
    if delta == ZERO:
        return None
    side = "buy" if delta > ZERO else "sell"
    requested = abs(delta)
    price = open_price * (ONE + settings.slippage_bps / BPS if delta > ZERO else ONE - settings.slippage_bps / BPS)
    if delta > ZERO:
        # The new allocation includes all costs. The minimum is a purchase
        # budget, not a cash reserve or a restriction on closing a position.
        budget = min(cash, target_value - holdings * open_price)
        if budget < settings.minimum_trade_budget:
            return Fill(side, requested, ZERO, price, ZERO, cash, holdings, "rejected", "below_minimum_trade_budget")
        low, high = 0, int(requested / SHARE_QUANTUM)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = mid * SHARE_QUANTUM
            if candidate * price + fee(candidate * price, settings) <= budget:
                low = mid
            else:
                high = mid - 1
        quantity = low * SHARE_QUANTUM
        if quantity == ZERO:
            return Fill(side, requested, ZERO, price, ZERO, cash, holdings, "rejected", "insufficient_cash")
        fees = fee(quantity * price, settings)
        after_cash = cash - quantity * price - fees
        after_holdings = holdings + quantity
    else:
        quantity = min(requested, holdings)
        fees = fee(quantity * price, settings)
        after_cash = cash + quantity * price - fees
        if after_cash < ZERO:
            return Fill(side, requested, ZERO, price, ZERO, cash, holdings, "rejected", "sale_proceeds_cannot_cover_fees")
        after_holdings = holdings - quantity
    if after_cash < ZERO or after_holdings < ZERO:
        raise ArithmeticError("Execution violated the nonnegative cash/holdings invariant.")
    status = "filled" if quantity == requested else "reduced"
    return Fill(side, requested, quantity, price, fees, after_cash, after_holdings, status, "affordability_limit" if status == "reduced" else "")
