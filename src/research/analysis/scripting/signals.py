"""Small, explicit Polars conditions for research scripts."""

import polars as pl
import math


def _expr(value):
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str) and value:
        return pl.col(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return pl.lit(value)
    raise ValueError("A crossover operand must be a column name, expression, or number")


def crossed_above(left, right) -> pl.Expr:
    """True only where left > right and previous left <= previous right.

    Missing current or previous observations produce False. Evaluate over the
    working history before trimming initialization rows from your findings.
    """
    left, right = _expr(left), _expr(right)
    return ((left > right) & (left <= right).shift(1)).fill_null(False)


def crossed_below(left, right) -> pl.Expr:
    """True only where left < right and previous left >= previous right."""
    left, right = _expr(left), _expr(right)
    return ((left < right) & (left >= right).shift(1)).fill_null(False)
