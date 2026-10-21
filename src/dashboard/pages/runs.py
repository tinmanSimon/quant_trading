"""Open and compare saved research runs."""

import polars as pl
import streamlit as st

from dashboard.pages.backtest import _show_result
from research import Research, ResearchError

def _runs_page(research: Research):
    st.title("Saved research runs")
    runs = research.list_runs()
    if not runs:
        st.info("No saved runs yet. Complete a backtest to record one.")
        return
    st.dataframe(pl.DataFrame(runs), hide_index=True)
    ids = [item["run_id"] for item in runs]
    chosen = st.selectbox("Open saved run", ids)
    _show_result(research.load_run(chosen))
    compare = st.multiselect("Compare saved runs", ids)
    if len(compare) > 1:
        try:
            st.dataframe(research.compare_runs(compare), hide_index=True)
        except (ResearchError, ValueError) as error:
            st.warning(str(error))


