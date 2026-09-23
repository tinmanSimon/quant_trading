"""Local Plotly renderer serving bounded, display-only candle windows."""

from functools import lru_cache
from bisect import bisect_left
from collections import defaultdict
from importlib.resources import files
import json
import math

from plotly.offline import get_plotlyjs
import streamlit as st

from .charts import CHART_CONFIG, window_price_chart


@lru_cache(maxsize=1)
def _plotly_bundle():
    return get_plotlyjs()


def _javascript():
    """Self-contained bundle for offline browser tests."""
    return _plotly_bundle() + "\n" + _chart_javascript()


def _chart_javascript():
    return files("dashboard").joinpath("bar_chart.js").read_text(encoding="utf-8")


def _renderer():
    # The large bundle has its own constant Streamlit message, so changing a
    # window does not resend it with each figure. All assets stay local.
    loader = st.components.v2.component("chart_plotly_runtime", js=_plotly_bundle() +
        '\nexport default function() { window.dispatchEvent(new Event("quant-plotly-ready")); }')
    loader(key="chart-plotly-runtime", height=0)
    return st.components.v2.component("bar_chart", js=_chart_javascript(), isolate_styles=False)


def render_price_chart(figure, *, key="market-price-chart"):
    """Mount a chart whose zoom/pan events stay entirely in the browser."""
    component = _renderer()
    component(data={"figure": json.loads(figure.to_json()), "config": CHART_CONFIG},
              key=key, height=650, width="stretch")


def viewport_request(value, chart_id):
    """Validate client navigation before allowing it to select a server window."""
    if not isinstance(value, dict) or value.get("chart_id") != chart_id:
        return None
    sequence = value.get("request_id")
    if type(sequence) is not int or not 0 < sequence < 2**53:
        return None
    for name in ("start", "end", "width"):
        number = value.get(name)
        if (isinstance(number, bool) or not isinstance(number, (float, int))
                or not -2**53 < number < 2**53 or not math.isfinite(number)):
            return None
    if value["start"] >= value["end"] or not 0 < value["width"] <= 10000:
        return None
    return value


def window_payload(series, *, chart_id, timeframe, timezone, title="", request=None):
    request = viewport_request(request, chart_id)
    window = series.window(request["start"] if request else None, request["end"] if request else None,
                           width=request["width"] if request else 900)
    markers = defaultdict(list)
    for stamp in series.omitted_timestamps:
        boundary = bisect_left(series.timestamps, stamp) - 0.5
        if window.loaded_range[0] <= boundary <= window.loaded_range[1]:
            group = bisect_left(window.last_indices, boundary)
            # Between groups, mark the compressed boundary. Only an omission
            # inside a grouped candle belongs on that summary's midpoint.
            position = (window.positions[group] if group < len(window.positions)
                        and window.first_indices[group] < boundary < window.last_indices[group]
                        else boundary)
            markers[position].append(stamp.isoformat())
    # Count every omission, but bound the number and length of annotations.
    positions = sorted(markers, key=lambda position: (
        not window.view_range[0] <= position <= window.view_range[1], position))[:80]
    gaps = [(position, len(markers[position]), markers[position][:5]
             + (["More timestamps in omission details"] if len(markers[position]) > 5 else []))
            for position in positions]
    figure = window_price_chart(window, timeframe=timeframe, timezone=timezone, title=title, gap_markers=gaps)
    overview = series.window(-0.5, series.total_count - 0.5, width=300)
    return {"figure": json.loads(figure.to_json()), "config": CHART_CONFIG, "chart_id": chart_id,
            "request_id": request["request_id"] if request else 0, "total_count": series.total_count,
            "width_budget": max(200, min(1000, int(request["width"] if request else 900))),
            "view_range": window.view_range, "loaded_range": window.loaded_range,
            "resolution": window.resolution, "source_counts": window.source_counts,
            "omission_marker_note": (f"Showing 80 of {len(markers)} omission markers; see omission details."
                                     if len(markers) > 80 else ""),
            "first_indices": window.first_indices, "last_indices": window.last_indices,
            "overview": {"positions": overview.positions, "close": overview.frame["close"].to_list(),
                         "label": "Full history · grouped closing prices",
                         "first_label": series.timestamps[0].isoformat(),
                         "last_label": series.timestamps[-1].isoformat()}}


def render_series_chart(series, *, chart_id, timeframe, timezone, title=""):
    key = "market-price-chart-" + chart_id
    state = st.session_state.get(key, {})
    request = state.get("viewport") if isinstance(state, dict) else getattr(state, "viewport", None)
    payload = window_payload(series, chart_id=chart_id, timeframe=timeframe, timezone=timezone,
                             title=title, request=request)
    component = _renderer()
    component(data=payload, key=key, height=820, width="stretch", default={"viewport": None},
              on_viewport_change=lambda: None)
