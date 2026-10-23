"""Acquire and review a frozen corporate-action snapshot before processing."""

import csv
from datetime import date
from io import StringIO
import json

import streamlit as st


def snapshot_controls(label, value, memory, key, bounds, context):
    from data_pipeline.corporate_actions.providers import DatabentoActionProvider, YahooActionProvider, import_snapshot

    context = context or {}
    st.markdown(f"**{label}**")
    st.caption("Splits and cash dividends are adjusted. Other events are reported. "
               "The snapshot and reference date are saved in the recipe; processing never refreshes them silently.")
    tickers = context.get("tickers", [])
    event_provider = memory.widget(st.selectbox, "Corporate-action provider",
                                  options=["Yahoo Finance (free)", "Databento Reference (subscription)"],
                                  key=key + "-provider")
    first, last = context.get("action_bounds") or bounds or (date.today(), date.today())
    left, right = st.columns(2)
    start = memory.widget(left.date_input, "Action coverage start", value=first, key=key + "-start")
    reference = memory.widget(right.date_input, "Adjustment reference date", value=last, key=key + "-reference")
    st.caption("Coverage must span the selected bars through the reference date. "
               "These dates apply to the next fetch; the active snapshot below keeps its own reference date. "
               "Verified JSON imports also work offline.")
    yahoo = event_provider == "Yahoo Finance (free)"
    if yahoo:
        st.caption("Yahoo supplies reported splits and cash dividends. Dividend factors use the previous "
                   "regular-session closing minute from your selected unadjusted 1m batches. "
                   "Other-event coverage and historical ticker identity are not verified by Yahoo.")
    else:
        st.caption("Databento requires subscriptions for Security Master, Corporate Actions and Adjustment Factors.")
    if st.button("Fetch corporate actions from " + ("Yahoo" if yahoo else "Databento"),
                 key=key + "-fetch", disabled=not tickers):
        try:
            provider = context.get("provider", "")
            basis = "unadjusted" if provider.startswith("databento-") or provider == "massive" else "unknown"
            with st.spinner("Fetching corporate actions and factors for all requested tickers…"):
                if yahoo:
                    snapshot = YahooActionProvider().fetch(
                        tickers, start, reference, source_basis=basis,
                        reference_bars=context.get("reference_bars"),
                        coverage_starts=context.get("coverage_starts"))
                else:
                    snapshot = DatabentoActionProvider().fetch(tickers, start, reference, source_basis=basis)
            memory.values["widgets"][key] = json.dumps(snapshot.to_dict(), indent=2)
        except Exception as error:
            st.error(f"Corporate-action fetch failed: {error}")
    uploaded = st.file_uploader("Import verified corporate-action snapshot", type=["json"], key=key + "-upload")
    if st.button("Use imported snapshot", key=key + "-import", disabled=uploaded is None):
        try:
            snapshot = import_snapshot(uploaded.getvalue())
            memory.values["widgets"][key] = json.dumps(snapshot.to_dict(), indent=2)
        except Exception as error:
            st.error(f"Invalid snapshot: {error}")
    text = memory.widget(st.text_area, "Corporate-action snapshot (JSON)",
                         value=json.dumps(value or {}, indent=2), key=key, height=220)
    try:
        snapshot = import_snapshot(text)
    except Exception as error:
        raise ValueError(f"Load a valid corporate-action snapshot: {error}") from error
    st.caption(f"Source: {snapshot.source} · Reference date: {snapshot.reference_date} · "
               f"Snapshot: {snapshot.checksum[:12]}")
    st.dataframe(snapshot.to_dict()["coverage"], hide_index=True)
    for coverage in snapshot.coverage:
        if coverage.status != "complete":
            st.warning(f"{coverage.symbol}: {coverage.details}")
    if snapshot.events:
        st.dataframe(snapshot.to_dict()["events"], hide_index=True)
    return snapshot.to_dict()


def render_other_events(rows, *, key):
    st.subheader("Other corporate actions")
    st.caption("These events are reported without automatic adjustment. Coverage is shown for every requested ticker.")
    st.dataframe(rows, hide_index=True)
    stream = StringIO()
    fields = list(dict.fromkeys(field for row in rows for field in row))
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    left, right = st.columns(2)
    left.download_button("Download other events CSV", stream.getvalue(), "other_corporate_actions.csv",
                         "text/csv", key=key + "-csv")
    right.download_button("Download other events JSON", json.dumps(rows, indent=2), "other_corporate_actions.json",
                          "application/json", key=key + "-json")
