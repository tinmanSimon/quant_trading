from abc import ABC, abstractmethod
import polars as pl

from .schemas.ohlcv import OHLCV_SCHEMA


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
        Fetch OHLCV data as a Polars DataFrame conforming to ``OHLCV_SCHEMA``.

        The public provider interface will accept ``DataRequest`` in a later
        step; this legacy signature remains unchanged while the contract is
        introduced and tested.
        """
        pass
