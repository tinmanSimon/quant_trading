"""Pure retrospective trend descriptions for a single ordered Polars dataframe."""

from .api import analyze, analyze_current, analyze_history, replay
from .models import ALGORITHM_VERSION, AnalysisResult, CandidateDiagnostic, Trend, TrendConfig
from .segmentation import penalty_grid

__all__ = ["ALGORITHM_VERSION", "AnalysisResult", "CandidateDiagnostic", "Trend", "TrendConfig",
           "analyze", "analyze_current", "analyze_history", "replay", "penalty_grid"]
