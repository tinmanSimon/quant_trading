"""Compatibility imports for the original fetcher module."""

from .providers.base import BaseDataProvider
from .schemas.ohlcv import OHLCV_SCHEMA


__all__ = ["BaseDataProvider", "OHLCV_SCHEMA"]
