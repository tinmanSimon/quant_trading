"""Configuration and outputs of long-only account simulations."""

from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real

import polars as pl


def as_decimal(value, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite number.")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError(f"{name} must be a finite number.")
    return number


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """No borrowing, no shorts, entries at the next bar open.

    Allocation strategies also exit at the next open; bracket strategies use
    their fixed intrabar stop/target execution rules.

    Whole shares are the default; fractional shares round down to 8 decimal
    places and require the configured minimum purchase budget (including fees).
    Cash arithmetic uses Decimal; fees round upward to 8 decimal places.
    Slippage is a deterministic adverse basis-point adjustment, not measured
    historical spread. There are no dividend payments or interest accruals.
    """

    initial_cash: Decimal | float = Decimal("10000")
    commission_fixed: Decimal | float = Decimal("0")
    commission_bps: Decimal | float = Decimal("0")
    slippage_bps: Decimal | float = Decimal("0")
    fractional_shares: bool = False
    minimum_trade_budget: Decimal | float = Decimal("5")

    def __post_init__(self):
        if not isinstance(self.fractional_shares, bool):
            raise ValueError("fractional_shares must be a boolean.")
        for name in ("initial_cash", "commission_fixed", "commission_bps", "slippage_bps", "minimum_trade_budget"):
            number = as_decimal(getattr(self, name), name)
            if number < 0 or (name == "initial_cash" and number == 0):
                raise ValueError(f"{name} must be {'positive' if name == 'initial_cash' else 'nonnegative'}.")
            if number > Decimal("1e18"):
                raise ValueError(f"{name} is outside the supported numeric range.")
            if len(number.as_tuple().digits) > 28:
                raise ValueError(f"{name} must have at most 28 significant decimal digits.")
            if number.as_tuple().exponent < -28:
                raise ValueError(f"{name} supports at most 28 decimal places.")
            object.__setattr__(self, name, number)
        if self.slippage_bps >= 10000:
            raise ValueError("slippage_bps must be less than 10000.")

    def to_dict(self) -> dict:
        return {
            **{name: str(getattr(self, name)) for name in ("initial_cash", "commission_fixed", "commission_bps", "slippage_bps", "minimum_trade_budget")},
            "fractional_shares": self.fractional_shares,
        }


@dataclass(slots=True)
class BacktestResult:
    symbol: str
    strategy_spec: dict
    settings: ExecutionSettings
    equity: pl.DataFrame
    trades: pl.DataFrame
    orders: pl.DataFrame
    metrics: dict
    metadata: dict = field(default_factory=dict)
    completed_trades: pl.DataFrame = field(default_factory=pl.DataFrame)
    open_positions: pl.DataFrame = field(default_factory=pl.DataFrame)
