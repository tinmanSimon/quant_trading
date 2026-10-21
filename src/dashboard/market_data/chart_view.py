"""Rerun only the navigation controls and bounded chart window on interaction."""

import streamlit as st

from dashboard.charts.component import render_series_chart
from .navigation import render_analysis_navigation


@st.fragment
def render_chart_view(series, *, applied, event_data, chart_id, timeframe, timezone,
                      title, overlays, overlay_revision, memory=None,
                      render_chart=render_series_chart):
    """Navigate the verified snapshot without rebuilding analysis controls/results.

    Source selection, Apply, display settings, and result-table selections still
    run the page and supply a fresh snapshot. Navigation buttons and viewport
    callbacks rerun this fragment using the already prepared chart layers.
    """
    with st.container():
        navigation = render_analysis_navigation(applied, series, chart_id, memory,
                                                event_data=event_data)
    options = {"navigation_command": navigation} if navigation is not None else {}
    render_chart(series, chart_id=chart_id, timeframe=timeframe, timezone=timezone,
                 title=title, overlays=overlays, overlay_revision=overlay_revision, **options)
