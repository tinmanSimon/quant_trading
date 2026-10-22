"""Compare separate strategy simulations, including shared portfolios."""

import json

import polars as pl

from .models import BacktestResult


def compare_results(results: list[BacktestResult]) -> pl.DataFrame:
    """Return one metric row per simulation; never sum separate equity curves."""
    return pl.DataFrame([
        {"symbol": result.symbol, "strategy": result.strategy_spec["name"], "version": result.strategy_spec["version"], "config": json.dumps(result.strategy_spec["config"], sort_keys=True), **dict(sorted(result.metrics.items()))}
        for result in results
    ], infer_schema_length=None)
