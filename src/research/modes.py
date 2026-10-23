"""Explicit execution assumptions for strict and adjusted-price research."""

from .errors import ResearchError


RESEARCH_MODES = ("strict", "adjusted_price")
ADJUSTED_PRICE_BASIS = "split_and_cash_dividend_adjusted"
TRANSFORMED_ADJUSTED_PRICE_BASIS = "processed_transformed_adjusted_prices"
ADJUSTED_RETURN_BASIS = "adjusted_price_return_no_additional_dividend_credit"
ADJUSTED_PRICE_NOTE = (
    "Adjusted-price research uses retrospective split and cash-dividend adjustments. "
    "Prices and shares use adjusted units; absolute price thresholds, whole-share sizing "
    "and per-share fees therefore use those units. Cash dividends are reflected in the "
    "adjusted series, with no additional dividend cash credit or corporate-action position "
    "accounting. Other corporate actions are reported only and remain unadjusted."
)


def validate_research_mode(research_mode: str, *, layer: str | None = None) -> str:
    if not isinstance(research_mode, str) or research_mode not in RESEARCH_MODES:
        raise ResearchError("research_mode must be 'strict' or 'adjusted_price'.")
    if research_mode == "adjusted_price" and layer is not None and layer != "processed":
        raise ResearchError("Adjusted-price research requires processed data with corporate-action adjustments.")
    return research_mode
