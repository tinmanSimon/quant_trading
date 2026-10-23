"""Backward adjustment of US-equity prices using pinned corporate actions."""

from .processor import AdjustCorporateActions, PriceAdjustmentWarning

__all__ = ["AdjustCorporateActions", "PriceAdjustmentWarning"]
