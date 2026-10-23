"""Explicit opt-in and saved-result context for retrospective adjusted prices."""

import streamlit as st


RESEARCH_MODE_LABELS = {
    "strict": "Strict (causal processors only)",
    "adjusted_price": "Adjusted-price research",
}
ADJUSTED_PRICE_EXPLANATION = (
    "Adjusted-price research uses retrospective split and cash-dividend adjustments, including events after "
    "the simulated date. Trades use adjusted price and share units; cash dividends are reflected in the "
    "price series without additional cash credits. This does not reproduce historical share counts or "
    "dividend cash payments. Absolute price thresholds and whole-share rounding can change results."
)


def research_mode_control(memory, *, key):
    mode = memory.widget(
        st.radio, "Research mode", options=list(RESEARCH_MODE_LABELS),
        format_func=RESEARCH_MODE_LABELS.get, key=key,
        help="Adjusted-price research requires a pipeline containing Adjust prices: splits and cash dividends. "
             "Other processors that use future information remain blocked.",
    )
    if mode == "adjusted_price":
        st.info(ADJUSTED_PRICE_EXPLANATION)
    return mode


def show_research_mode(manifest, *, ui=st):
    mode = manifest.get("research_mode", "strict")
    ui.caption(f"Research mode: {RESEARCH_MODE_LABELS.get(mode, mode)}.")
    if mode == "adjusted_price":
        ui.info(ADJUSTED_PRICE_EXPLANATION)

