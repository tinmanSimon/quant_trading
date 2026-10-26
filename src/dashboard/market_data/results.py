"""Text and paginated tables produced by research analyses."""

from functools import partial

import polars as pl
import streamlit as st

from dashboard.market_data import analysis_controls as controls
from dashboard.market_data.navigation import request_analysis_event


def _selected_finding(key, instance_id, event_ids, source_key, result_key):
    """Map a displayed table row back to its stable event, including pagination."""
    selected = st.session_state.get(key, {}).get("selection", {}).get("rows", [])
    if len(selected) == 1 and type(selected[0]) is int and 0 <= selected[0] < len(event_ids):
        event_id = event_ids[selected[0]]
        if event_id is not None:
            request_analysis_event(instance_id, event_id, source_key=source_key, result_key=result_key)


def render_logic_results(run, registry=None, memory=None):
    """Render standard messages and tables without knowing the analysis type."""
    if run is None:
        return
    registry = registry if registry is not None else controls.discover_logics()
    definitions = {(item.name, item.version): item for item in registry.definitions()}
    for index, spec in enumerate(run.specs):
        definition = definitions.get((spec.name, spec.version))
        label = definition.label if definition else spec.name
        result = run.results.get(spec.instance_id)
        if spec.instance_id in run.errors:
            st.error(f"{label} ({index + 1}) failed: {run.errors[spec.instance_id]}")
            continue
        if result is None:
            continue
        with st.expander(f"{label} · results ({index + 1})", expanded=True):
            for message in result.messages:
                st.caption(message)
            for text in result.data.get("analysis", {}).get("text", []):
                st.markdown(text)
            events = {event["id"] for event in result.data.get("analysis", {}).get("events", [])}
            for table_index, table in enumerate(result.tables):
                st.markdown(f"**{table['name']}**")
                if table["rows"]:
                    rows = table["rows"]
                    first = 0
                    if len(rows) > 500:
                        pages = (len(rows) + 499) // 500
                        default_page = table.get("default_page", pages)
                        if type(default_page) is not int or not 1 <= default_page <= pages:
                            default_page = pages
                        kwargs = dict(min_value=1, max_value=pages, value=default_page, step=1,
                                      key=f"chart-result-page-{spec.instance_id}-{table_index}")
                        label = f"{table['name']} page"
                        page = (memory.widget(st.number_input, label, **kwargs) if memory else
                                st.number_input(label, **kwargs))
                        first = (page - 1) * 500
                        st.caption(f"Rows {first + 1:,}–{min(first + 500, len(rows)):,} of {len(rows):,}.")
                        rows = rows[first:first + 500]
                    try:
                        frame = pl.DataFrame(rows, infer_schema_length=None, strict=False)
                        event_ids = [row.get("_event_id") if row.get("_event_id") in events else None
                                     for row in rows]
                        options = {}
                        if any(event_id is not None for event_id in event_ids):
                            revision = run.keys.get(spec.instance_id, "")[:12]
                            key = f"chart-result-table-{spec.instance_id}-{revision}-{table_index}-{first}"
                            options = {"key": key, "selection_mode": "single-row",
                                       "on_select": partial(_selected_finding, key, spec.instance_id, event_ids,
                                                            run.source_key, run.keys.get(spec.instance_id)),
                                       "column_config": {"_event_id": None}}
                            st.caption("Select a finding to navigate to it on the chart.")
                        st.dataframe(frame, hide_index=True, width="stretch", **options)
                    except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
                        st.error(f"Could not display the {table['name']!r} table: {error}")
                        st.json(rows)
                else:
                    st.caption("No rows.")
            with st.expander("Applied settings"):
                st.caption(f"Logic: {spec.name} · version {spec.version}")
                st.json(spec.config)


