"""Browse verified local bars and apply research analyses."""

from datetime import UTC, datetime, timedelta
from functools import partial
from hashlib import sha256
from pathlib import Path
import json

import polars as pl
import streamlit as st

from data_pipeline import DataQuery
from dashboard.charts import bar_table
from dashboard.charts.cache import BoundedChartCache
from dashboard.charts.component import render_series_chart
from dashboard.components.source_controls import _date_fields, _provider_choice
from dashboard.pages.processing import pipeline_label
from dashboard.market_data.analysis_controls import render_logic_controls
from dashboard.market_data.display_controls import render_logic_display_controls
from dashboard.market_data.results import render_logic_results
from dashboard.market_data.navigation import analysis_events
from dashboard.market_data.chart_view import render_chart_view
from dashboard.state.workspace import market_memory
from research.analysis import LogicContext
from research.analysis.core.omissions import omission_metadata
from research import Research
from research.timeframes import INTRADAY_DURATIONS

def _metadata_rows(items):
    return [{
        "symbol": item.request.symbol, "provider": item.request.provider,
        "timeframe": item.request.timeframe, "layer": item.layer,
        "first_bar_utc": item.first_timestamp.isoformat(),
        "last_bar_utc": item.last_timestamp.isoformat(), "rows": item.row_count,
        "active": item.active, "dataset_id": item.dataset_id,
        "pipeline_id": item.pipeline_id,
    } for item in items]


def _quality(items, start, end, *, present_timestamps=(), show_details=True, ui=st,
             quality_metadata=None):
    """Show recorded omissions without treating old metadata as proof of completeness."""
    st = ui
    omitted = []
    processed = []
    unknown = []
    present = set(present_timestamps)
    for item in items:
        quality = getattr(item, "quality", None)
        if quality is None or quality.status == "unknown":
            unknown.append(item.dataset_id)
        if quality is None:
            continue
        if getattr(item, "layer", "raw") == "processed":
            processed.extend((bar, item.dataset_id) for bar in quality.omitted_bars)
            continue
        for bar in quality.omitted_bars:
            if start <= bar.timestamp < end:
                omitted.append((bar, item.dataset_id))
    if omitted:
        absent = {bar.timestamp for bar, _ in omitted if bar.timestamp not in present}
        restored = {bar.timestamp for bar, _ in omitted if bar.timestamp in present}
        if absent:
            st.warning(f"{len(absent)} recorded omitted bars are still absent in this window. "
                       "Known omissions in the loaded chart window are marked; hover their labels for timestamps.")
        if restored:
            st.info(f"{len(restored)} previously omitted bars are present in the selected data. Their omission records are historical.")
        if show_details:
            st.dataframe(pl.DataFrame([{
                "timestamp_utc": bar.timestamp, "reason": bar.reason, "dataset_id": dataset_id,
                "current_state": "present in selected data" if bar.timestamp in present else "absent from selected data",
            } for bar, dataset_id in omitted]), hide_index=True)
    if processed:
        # Parent timestamps and resampled output labels are different domains.
        # Only verified raw parent contents can establish a historical repair.
        unresolved = ({stamp for source in quality_metadata.get("processed_omission_sources", [])
                       for stamp in source["omitted_bar_timestamps"]}
                      if quality_metadata is not None else
                      {bar.timestamp.isoformat() for bar, _ in processed})
        if unresolved:
            st.warning(f"Processed data has {len(unresolved)} unresolved omitted raw-input bars. "
                       "Their effect on output bars has not been verified. Flat analysis is unavailable "
                       "until those source omissions are resolved; output timestamps do not establish repairs.")
        else:
            st.info("Previously omitted source bars are present in the verified raw parents. "
                    "Their omission records are historical.")
        if show_details:
            st.dataframe(pl.DataFrame([{
                "raw_timestamp_utc": bar.timestamp, "reason": bar.reason, "dataset_id": dataset_id,
                "current_state": ("unresolved in raw inputs" if bar.timestamp.isoformat() in unresolved
                                  else "present in verified raw inputs"),
            } for bar, dataset_id in processed]), hide_index=True)
    if unknown:
        st.caption(f"Omission history is unavailable for {len(unknown)} selected revision(s). "
                   "Valid stored values and first/last bounds do not establish complete coverage.")
    elif not omitted and not processed:
        st.caption("No ingestion omissions were recorded in this window. Backtest preflight checks expected bar coverage separately.")
    return sorted({bar.timestamp for bar, _ in omitted if bar.timestamp not in present})


def _chart_csv(pipeline, query, dataset_id=None):
    """Read and verify again at download time, including after storage changes."""
    frame = pipeline.read(query) if dataset_id is None else pipeline.read_dataset(dataset_id)
    timestamp = pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    if query.start is not None:
        frame = frame.filter(timestamp >= pl.lit(query.start, dtype=pl.Datetime("us", "UTC")))
    if query.end is not None:
        frame = frame.filter(timestamp < pl.lit(query.end, dtype=pl.Datetime("us", "UTC")))
    return frame.write_csv()


def _data_page(research: Research, **render_options):
    memory = market_memory(str(Path(research.pipeline.store.data_dir).resolve()))
    try:
        _render_data_page(research, memory, **render_options)
    finally:
        memory.save()
        if memory.warning:
            st.warning(memory.warning)


def _render_data_page(research: Research, memory, *, render_chart=render_series_chart,
                      csv_export=_chart_csv, table_factory=bar_table):
    st.title("Market data")
    st.caption("Explore stored bars and their revisions. All chart values come from the verified local data.")
    layer = memory.widget(st.radio, "Layer", options=["raw", "processed"], horizontal=True, key="browse-layer")
    include_history = memory.widget(st.checkbox, "Include historical revisions", key="browse-history")
    items = research.pipeline.list_datasets(DataQuery(layer=layer, include_history=include_history))
    if not items:
        st.info("No matching local datasets. Open Fetch to download your first batch.")
        return
    provider = _provider_choice((item.request.provider for item in items), key="browse-provider", memory=memory)
    items = [item for item in items if item.request.provider == provider]
    search = memory.widget(st.text_input, "Find a ticker", placeholder="AAPL", key="browse-search")
    symbols = sorted({item.request.symbol for item in items if search.upper() in item.request.symbol.upper()})
    if not symbols:
        st.info("No tickers match this search.")
        return
    symbol = memory.widget(st.selectbox, "Ticker", options=symbols, key="browse-symbol")
    symbol_items = [item for item in items if item.request.symbol == symbol]
    identities = sorted({(item.request.provider, item.request.timeframe, item.pipeline_id) for item in symbol_items})
    identity = memory.widget(st.selectbox, "Data series", options=identities, key="browse-series",
                            format_func=lambda value: f"{value[0]} · {value[1]}" + (f" · {pipeline_label(research, value[2], symbol_items)}" if value[2] else ""))
    matching = [item for item in symbol_items
                if (item.request.provider, item.request.timeframe, item.pipeline_id) == identity]
    mode = memory.widget(st.radio, "Read", options=["Active batches together", "One exact revision"],
                        horizontal=True, key="browse-read")
    if mode == "One exact revision":
        selected_id = memory.widget(st.selectbox, "Revision", options=[item.dataset_id for item in matching], key="browse-revision",
                                  format_func=lambda value: next(
                                      f"{item.dataset_id[:12]} · {item.first_timestamp.date()} to {item.last_timestamp.date()} · "
                                      f"{'active' if item.active else 'historical'}"
                                      for item in matching if item.dataset_id == value))
        selected = [item for item in matching if item.dataset_id == selected_id]
    else:
        selected = [item for item in matching if item.active]
        if not selected:
            st.info("This series has no active revisions; select an exact historical revision to view it.")
            return
    lower = min(item.first_timestamp for item in selected)
    upper = max(item.last_timestamp for item in selected)
    window_key = f"browse-{symbol}-{identity}-{mode}-{selected[0].dataset_id if mode == 'One exact revision' else 'active'}"
    start, end = _date_fields(window_key, first=lower.date(), last=(upper + timedelta(days=1)).date(), memory=memory)
    timezone = memory.widget(st.selectbox, "Display timezone (intraday bars)",
                             options=["America/New_York", "UTC", "Asia/Shanghai"], key="browse-timezone")
    if start >= end:
        st.error("The end must be later than the start.")
        return
    st.caption("Drag to zoom, scroll to zoom, or use the overview navigator. Double-click to reset. "
               "Candles are equally spaced: periods without stored bars are compressed. "
               "Price and volume scales fit the visible bars automatically. "
               "Daily and longer bars retain their session-date labels in every timezone.")
    if identity[1] in INTRADAY_DURATIONS:
        st.caption("Each candle is labeled by its start. The final bar of a trading session may be shorter than the selected interval.")
    query = DataQuery(layer=layer, provider=identity[0], symbol=symbol, timeframe=identity[1],
                      pipeline_id=identity[2] or None, start=start, end=end)
    dataset_id = selected[0].dataset_id if mode == "One exact revision" else None
    quality_metadata = omission_metadata([json.loads(item.to_json()) for item in selected],
                                         pipeline=research.pipeline)
    omitted = [bar.timestamp for item in selected if item.quality is not None and item.layer == "raw"
               for bar in item.quality.omitted_bars if start <= bar.timestamp < end]
    cache = st.session_state.get("market-chart-cache")
    if not isinstance(cache, BoundedChartCache):
        cache = BoundedChartCache()
        st.session_state["market-chart-cache"] = cache
    session_timezone = "America/New_York"
    series = cache.load(research.pipeline, query, dataset_id=dataset_id, timeframe=identity[1],
                        session_timezone=session_timezone, omitted_timestamps=omitted, include_support=True)
    with memory.widget(st.expander, "Dataset metadata", key="browse-metadata-open"):
        st.dataframe(pl.DataFrame(_metadata_rows(selected)), hide_index=True)
    has_omissions = bool(omitted) or any(item.layer == "processed" and item.quality.omitted_bars for item in selected)
    show_details = has_omissions and memory.widget(st.checkbox, "Show omission details", key="browse-omission-details")
    _quality(selected, start, end, present_timestamps=series.timestamps if series else (),
             show_details=show_details, quality_metadata=quality_metadata)
    if series is None:
        st.info("No saved bars fall inside this window.")
        return
    chart_id = sha256(repr((str(research.pipeline.store.data_dir), query, dataset_id,
                           tuple(item.to_json() for item in sorted(selected, key=lambda item: item.dataset_id)),
                           timezone, session_timezone)).encode()).hexdigest()
    st.caption(f"{series.total_count:,} original bars available · new views start on the latest {min(series.total_count, 50)} · "
               "zoom out for a grouped overview or zoom in for original bars. Stored values are unchanged; gaps are not filled.")
    st.caption("Display grouping uses America/New_York trading dates for U.S. stocks, including stored extended-hours bars. "
               "Daily and calendar-week summaries are labelled on the chart.")
    # Calendar start/end selectors filter session labels, not UTC completion
    # instants. Intraday bounds are instants and also limit the completion cutoff.
    cutoff = datetime.now(UTC)
    if not identity[1].endswith(("d", "wk", "mo")):
        cutoff = min(cutoff, end)
    logic_identity = (str(research.pipeline.store.data_dir), query, dataset_id,
                      tuple(item.to_json() for item in sorted(selected, key=lambda item: item.dataset_id)))
    context = LogicContext(frame=series.frame, timeframe=identity[1], identity=logic_identity,
                           cutoff=cutoff, session_timezone=session_timezone, support_frame=series.support_frame,
                           metadata=quality_metadata)
    # A stable container prevents conditional controls from moving the chart's
    # Streamlit element path, unmounting it, and losing browser-local navigation.
    with st.container():
        applied = render_logic_controls(context, memory)
    with st.container():
        displayed = render_logic_display_controls(applied, memory)
    overlays = displayed.overlays if displayed is not None else []
    # Overlay changes have their own sequence. Browser navigation requests keep
    # their sequence and range even when applied logic changes.
    # A new calculation can keep identical chart layers but replace findings.
    # Advance the revision too, so the browser clears an obsolete selection
    # and rejects delayed commands from the preceding analysis result.
    signature = repr((chart_id, overlays, applied.source_key if applied else None,
                      applied.keys if applied else {}))
    if st.session_state.get("chart-logic-overlay-signature") != signature:
        st.session_state["chart-logic-overlay-signature"] = signature
        st.session_state["chart-logic-overlay-revision"] = st.session_state.get("chart-logic-overlay-revision", 0) + 1
    render_chart_view(series, applied=applied, event_data=analysis_events(applied, series),
                      chart_id=chart_id, timeframe=identity[1], timezone=timezone,
                      title=f"{symbol} · {identity[1]}", overlays=overlays, memory=memory,
                      overlay_revision=st.session_state.get("chart-logic-overlay-revision", 0),
                      render_chart=render_chart)
    render_logic_results(displayed, memory=memory)
    if memory.widget(st.checkbox, "Show original bars", key="browse-show-original-bars"):
        pages = (series.total_count + 499) // 500
        page = memory.widget(st.number_input,"Original bar page", min_value=1, max_value=pages, value=pages,
                               step=1, key=f"browse-original-page-{chart_id}")
        first_row = (page - 1) * 500
        frame = series.frame.slice(first_row, 500)
        st.caption(f"Original rows {first_row + 1:,}–{first_row + frame.height:,} of {series.total_count:,}.")
        st.dataframe(table_factory(frame, timeframe=identity[1], timezone=timezone), hide_index=True,
                     column_config={column: st.column_config.NumberColumn(format="plain")
                                    for column in ("open", "high", "low", "close", "volume")})
    st.download_button("Download original bars in selected date range as CSV",
                       data=partial(csv_export, research.pipeline, query, dataset_id),
                       file_name=f"{symbol}-{identity[1]}.csv", mime="text/csv",
                       key=f"browse-csv-{chart_id}", on_click="ignore")
