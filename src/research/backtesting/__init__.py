"""Backtest engine with isolated strategy state and explicit execution rules."""

from .compare import compare_results
from .engine import run_backtest
from .models import BacktestResult, ExecutionSettings

__all__ = ["ExecutionSettings", "BacktestResult", "run_backtest", "compare_results"]
