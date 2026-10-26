"""Provider- and dashboard-independent horizontal candle-range analysis."""

from .api import analyze, analyze_current, analyze_history, analyze_summary, replay, select_periods
from .models import ALGORITHM_VERSION, FlatAnalysisResult, FlatConfig, FlatPeriod

__all__ = [
    "ALGORITHM_VERSION", "FlatAnalysisResult", "FlatConfig", "FlatPeriod", "analyze",
    "analyze_current", "analyze_history", "analyze_summary", "replay", "select_periods",
]
