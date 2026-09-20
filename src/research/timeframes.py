"""Research bar durations and provider-compatible workflow choices."""

from datetime import timedelta
from types import MappingProxyType

from data_pipeline import DataQuery

from .errors import ResearchError


INTRADAY_DURATIONS = MappingProxyType({
    "1m": timedelta(minutes=1),
    "2m": timedelta(minutes=2),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "90m": timedelta(minutes=90),
})
# Keep the existing daily default in interactive controls.
BACKTEST_TIMEFRAMES = ("1d", "1h", "1m", "2m", "5m", "15m", "30m", "90m")


def normalize_timeframe(value: str) -> str:
    """Use the same aliases as ingestion and storage (notably 60m -> 1h)."""
    return DataQuery(timeframe=value).timeframe


def bar_duration(timeframe: str) -> timedelta | None:
    """None denotes daily session labels; all other supported bars have a duration."""
    timeframe = normalize_timeframe(timeframe)
    if timeframe not in BACKTEST_TIMEFRAMES:
        raise ResearchError("Backtesting supports: " + ", ".join(BACKTEST_TIMEFRAMES) + ".")
    return INTRADAY_DURATIONS.get(timeframe)


def fetch_timeframes(provider_timeframes) -> tuple[str, ...]:
    """Offer only intervals both the provider and the research workflow support."""
    supported = {normalize_timeframe(value) for value in provider_timeframes}
    return tuple(value for value in BACKTEST_TIMEFRAMES if value in supported)
