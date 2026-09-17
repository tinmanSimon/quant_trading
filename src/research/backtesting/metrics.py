"""Metrics use the initial cash baseline, including all entry costs."""

from decimal import Decimal, localcontext
import math

import polars as pl


def calculate_metrics(equity: pl.DataFrame, trades: pl.DataFrame) -> dict:
    with localcontext() as context:
        context.prec = 800
        values = [Decimal(value) for value in equity["equity_exact"]]
        initial, final = values[0], values[-1]
        peak = initial
        drawdown = Decimal(0)
        for value in values:
            peak = max(peak, value)
            drawdown = max(drawdown, (peak - value) / peak)
        fees = sum((Decimal(value) for value in trades["fees_exact"]), Decimal(0))
        result = {
            "initial_cash": float(initial),
            "final_equity": float(final),
            "final_equity_exact": str(final),
            "total_return": float(final / initial - 1),
            "max_drawdown": float(drawdown),
            "trade_count": trades.height,
            "total_fees": float(fees),
            "total_fees_exact": str(fees),
            "open_shares": equity["holdings"][-1],
            "return_basis": "price_return_excluding_dividends",
        }
        if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
            raise ValueError("Backtest metrics exceed the supported Float64 display range.")
        return result
