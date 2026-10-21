"""Technical indicators for ordered Polars market data, usable without a UI.

Every function preserves input rows and timestamps. Initialization values are
null, never filled. Supply earlier bars in the input when they should initialize
recursive indicators, then slice the returned frame to the research/display
range. Results depend on that supplied starting point and never on future bars.
Each input must describe one time series: optional symbol, ticker, provider,
timeframe, and interval columns must each contain one non-null identity.
"""

from .api import adx, atr, bollinger_bands, macd, moving_average, obv, rsi, stochastic
from .models import ALGORITHM_VERSION, BACKEND_VERSION, IndicatorResult

__all__ = [
    "ALGORITHM_VERSION", "BACKEND_VERSION", "IndicatorResult", "moving_average",
    "rsi", "macd", "atr", "bollinger_bands", "adx", "stochastic", "obv",
]
