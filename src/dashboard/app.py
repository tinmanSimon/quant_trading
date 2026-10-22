"""Dashboard startup and routing. Run: streamlit run src/dashboard/app.py."""

from pathlib import Path

import streamlit as st

from dashboard.charts import bar_table
from dashboard.charts.component import render_series_chart
from dashboard.pages import market_data, backtest
from dashboard.pages.fetch import _fetch_page
from dashboard.pages.runs import _runs_page
from dashboard.pages.processing import processing_page
from dashboard.pages.deletion import deletion_page
from dashboard.pages.analysis import _analysis_page
from dashboard.state.workspace import market_memory
from research import Research

# Keep existing entry points and caller-supplied render/export hooks compatible.
_chart_csv = market_data._chart_csv


def _data_page(research):
    return market_data._data_page(
        research, render_chart=render_series_chart, csv_export=_chart_csv, table_factory=bar_table)


def _quality(*args, **kwargs):
    return market_data._quality(*args, **kwargs, ui=st)


def _show_result(run):
    return backtest._show_result(run, ui=st)


_backtest_page = backtest._backtest_page


def main():
    st.set_page_config(page_title="Quant research", page_icon="📈", layout="wide")
    with st.sidebar:
        st.title("Quant research")
        directories = market_memory("dashboard-directories")
        # Empty text inputs resolve to cwd, so two cleared fields accidentally
        # select the same store. Restore defaults before rendering/persisting.
        for key, default in (("data-directory", "data"), ("runs-directory", "runs")):
            value = directories.values["widgets"].get(key)
            if isinstance(value, str) and not value.strip():
                directories.values["widgets"][key] = default
        data_dir = directories.widget(st.text_input, "Data directory", value="data", key="data-directory")
        runs_dir = directories.widget(st.text_input, "Research runs directory", value="runs", key="runs-directory")
        directories.save()
        if directories.warning:
            st.warning(directories.warning)
        st.caption(f"Data: {Path(data_dir).expanduser().resolve()}")
        st.caption(f"Results: {Path(runs_dir).expanduser().resolve()}")
        page = st.radio("Navigate", ["Market data", "Analysis", "Fetch", "Process data", "Backtest", "Saved runs", "Delete data"], key="dashboard-page")
        st.caption("Local data · reproducible research")
    # A store switch must not leave results from another directory on screen.
    identity = (str(Path(data_dir).expanduser().resolve()), str(Path(runs_dir).expanduser().resolve()))
    if st.session_state.get("store-identity") != identity:
        for key in ("fetch-report", "backtest-run"):
            st.session_state.pop(key, None)
        st.session_state["store-identity"] = identity
    data_path, runs_path = map(Path, identity)
    if data_path == runs_path or data_path in runs_path.parents or runs_path in data_path.parents:
        st.error("Choose separate Data directory and Research runs directory folders in the sidebar. "
                 "Neither folder can be inside the other.")
        st.code(f"Data: {data_path}\nResults: {runs_path}")
        return
    try:
        research = Research(data_dir=data_dir, runs_dir=runs_dir)
        {"Market data": _data_page, "Analysis": _analysis_page, "Fetch": _fetch_page, "Process data": processing_page,
         "Backtest": _backtest_page, "Saved runs": _runs_page, "Delete data": deletion_page}[page](research)
    except Exception as error:
        # Domain errors are displayed while preserving the local store; full
        # traceback remains available for diagnosing a failed operation.
        st.error(f"{type(error).__name__}: {error}")
        with st.expander("Error details"):
            st.exception(error)


if __name__ == "__main__":
    main()
