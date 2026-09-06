from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import polars as pl

# Standard columns expected by your backtester and feature pipeline
OHLCV_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

class BaseDataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = "1h"
    ) -> pl.DataFrame:
        """
        Fetches OHLCV data and returns a Polars DataFrame 
        conforming exactly to CANONICAL_SCHEMA.
        """
        pass

