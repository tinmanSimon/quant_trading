"""Compatibility imports for the original Yahoo fetcher module."""

from .base_fetcher import BaseDataProvider, OHLCV_SCHEMA
from .providers.yahoo import YFinanceProvider, yf


# Legacy class imports remain supported; fetching now checks raw source volume.
__all__ = ["BaseDataProvider", "OHLCV_SCHEMA", "YFinanceProvider"]
