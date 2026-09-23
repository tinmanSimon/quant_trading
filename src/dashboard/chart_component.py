"""Local Plotly renderer with browser-side fitting of visible candles."""

from functools import lru_cache
from importlib.resources import files
import json

from plotly.offline import get_plotlyjs
import streamlit as st

from .charts import CHART_CONFIG


@lru_cache(maxsize=1)
def _javascript():
    # Use the installed Plotly bundle: no CDN or additional frontend dependency.
    javascript = files("dashboard").joinpath("bar_chart.js").read_text(encoding="utf-8")
    return get_plotlyjs() + "\n" + javascript


def render_price_chart(figure, *, key="market-price-chart"):
    """Mount a chart whose zoom/pan events stay entirely in the browser."""
    component = st.components.v2.component("bar_chart", js=_javascript(), isolate_styles=False)
    component(data={"figure": json.loads(figure.to_json()), "config": CHART_CONFIG},
              key=key, height=650, width="stretch")
