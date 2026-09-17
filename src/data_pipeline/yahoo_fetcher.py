"""Compatibility imports for the original Yahoo fetcher module."""

from .base_fetcher import BaseDataProvider, OHLCV_SCHEMA
from .providers.yahoo import YFinanceProvider, yf


# Keep the historical yf.download monkeypatch path working as well.
__all__ = ["BaseDataProvider", "OHLCV_SCHEMA", "YFinanceProvider"]
