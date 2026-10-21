"""Run locally with: python -m streamlit run src/dashboard/app.py."""

from datetime import UTC, date, datetime, time, timedelta
from functools import partial
from hashlib import sha256
from pathlib import Path

import polars as pl
import streamlit as st

from data_pipeline import DataQuery
from dashboard.charts import CHART_CONFIG, bar_table, line_chart
from dashboard.chart_cache import BoundedChartCache
from dashboard.chart_component import render_series_chart
from dashboard.deletion import deletion_page
from dashboard.processing import processing_page, pipeline_label, backtest_memory
from dashboard.strategy_controls import strategy_specs
from chart_logics import LogicContext
from dashboard.logic_controls import render_logic_controls, render_logic_display_controls, render_logic_results
from dashboard.workspace import market_memory
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from research.timeframes import BACKTEST_TIMEFRAMES, INTRADAY_DURATIONS, fetch_timeframes


def _utc_date(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _tickers(text: str, provider: str = "yahoo") -> list[str]:
    items = text.replace(",", " ").split()
    return list(dict.fromkeys(item.upper() if provider == "yahoo" else item for item in items))


def _provider_choice(names, *, key, result_key=None, memory=None):
    names = sorted(set(names))
    if not names:
        st.info("No providers are available here. Fetching requires a registered provider; browsing and backtesting require local datasets.")
        return None
    provider = (memory.widget(st.selectbox, "Provider", options=names, key=key) if memory else
                st.selectbox("Provider", names, key=key))
    if result_key and st.session_state.get(f"{key}-previous") != provider:
        st.session_state.pop(result_key, None)
    st.session_state[f"{key}-previous"] = provider
    return provider


def _date_fields(prefix: str, *, first: date | None = None, last: date | None = None,
                 intraday: bool = False, memory=None):
    left, right = st.columns(2)
    first = first or date.today() - timedelta(days=30)
    last = last or date.today()
    if memory:
        start = memory.widget(left.date_input, "Start date (UTC, inclusive)", value=first, key=f"{prefix}-start")
        end = memory.widget(right.date_input, "End date (UTC, exclusive)", value=last, key=f"{prefix}-end")
    else:
        start = left.date_input("Start date (UTC, inclusive)", first, key=f"{prefix}-start")
        end = right.date_input("End date (UTC, exclusive)", last, key=f"{prefix}-end")
    if intraday:
        start_time = (memory.widget(left.time_input, "Start time (UTC)", value=time.min, step=60, key=f"{prefix}-start-time") if memory else
                      left.time_input("Start time (UTC)", time.min, step=60, key=f"{prefix}-start-time"))
        end_time = (memory.widget(right.time_input, "End time (UTC)", value=time.min, step=60, key=f"{prefix}-end-time") if memory else
                    right.time_input("End time (UTC)", time.min, step=60, key=f"{prefix}-end-time"))
        return datetime.combine(start, start_time, UTC), datetime.combine(end, end_time, UTC)
    return _utc_date(start), _utc_date(end)


def _metadata_rows(items):
    return [{
        "symbol": item.request.symbol, "provider": item.request.provider,
        "timeframe": item.request.timeframe, "layer": item.layer,
        "first_bar_utc": item.first_timestamp.isoformat(),
        "last_bar_utc": item.last_timestamp.isoformat(), "rows": item.row_count,
        "active": item.active, "dataset_id": item.dataset_id,
        "pipeline_id": item.pipeline_id,
    } for item in items]


def _quality(items, start, end, *, present_timestamps=(), show_details=True):
    """Show recorded omissions without treating old metadata as proof of completeness."""
    omitted = []
    unknown = []
    present = set(present_timestamps)
    for item in items:
        quality = getattr(item, "quality", None)
        if quality is None or quality.status == "unknown":
            unknown.append(item.dataset_id)
        if quality is None:
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
    if unknown:
        st.caption(f"Omission history is unavailable for {len(unknown)} selected revision(s). "
                   "Valid stored values and first/last bounds do not establish complete coverage.")
    elif not omitted:
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


def _data_page(research: Research):
    memory = market_memory(str(Path(research.pipeline.store.data_dir).resolve()))
    try:
        _render_data_page(research, memory)
    finally:
        memory.save()
        if memory.warning:
            st.warning(memory.warning)


def _render_data_page(research: Research, memory):
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
    omitted = [bar.timestamp for item in selected if item.quality is not None
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
    show_details = bool(omitted) and memory.widget(st.checkbox, "Show omission details", key="browse-omission-details")
    _quality(selected, start, end, present_timestamps=series.timestamps if series else (),
             show_details=show_details)
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
                           cutoff=cutoff, session_timezone=session_timezone, support_frame=series.support_frame)
    # A stable container prevents conditional controls from moving the chart's
    # Streamlit element path, unmounting it, and losing browser-local navigation.
    with st.container():
        applied = render_logic_controls(context, memory)
    with st.container():
        displayed = render_logic_display_controls(applied, memory)
    overlays = displayed.overlays if displayed is not None else []
    # Overlay changes have their own sequence. Browser navigation requests keep
    # their sequence and range even when applied logic changes.
    signature = repr((chart_id, overlays))
    if st.session_state.get("chart-logic-overlay-signature") != signature:
        st.session_state["chart-logic-overlay-signature"] = signature
        st.session_state["chart-logic-overlay-revision"] = st.session_state.get("chart-logic-overlay-revision", 0) + 1
    render_series_chart(series, chart_id=chart_id, timeframe=identity[1], timezone=timezone,
                        title=f"{symbol} · {identity[1]}", overlays=overlays,
                        overlay_revision=st.session_state.get("chart-logic-overlay-revision", 0))
    render_logic_results(displayed, memory=memory)
    if memory.widget(st.checkbox, "Show original bars", key="browse-show-original-bars"):
        pages = (series.total_count + 499) // 500
        page = memory.widget(st.number_input,"Original bar page", min_value=1, max_value=pages, value=pages,
                               step=1, key=f"browse-original-page-{chart_id}")
        first_row = (page - 1) * 500
        frame = series.frame.slice(first_row, 500)
        st.caption(f"Original rows {first_row + 1:,}–{first_row + frame.height:,} of {series.total_count:,}.")
        st.dataframe(bar_table(frame, timeframe=identity[1], timezone=timezone), hide_index=True,
                     column_config={column: st.column_config.NumberColumn(format="plain")
                                    for column in ("open", "high", "low", "close", "volume")})
    st.download_button("Download original bars in selected date range as CSV",
                       data=partial(_chart_csv, research.pipeline, query, dataset_id),
                       file_name=f"{symbol}-{identity[1]}.csv", mime="text/csv",
                       key=f"browse-csv-{chart_id}", on_click="ignore")


def _fetch_page(research: Research):
    st.title("Fetch market data")
    st.caption("Each ticker is attempted independently; a failure does not stop the remaining downloads.")
    provider = _provider_choice(research.pipeline.providers.names(), key="fetch-provider", result_key="fetch-report")
    if provider is None:
        return
    adapter = research.pipeline.providers.get(provider)
    intervals = fetch_timeframes(adapter.supported_timeframes)
    if not intervals:
        st.warning("This provider declares no fetch intervals supported by the dashboard.")
        return
    timeframe = st.selectbox("Bar size", intervals, key="fetch-timeframe")
    with st.form("fetch-data"):
        tickers = st.text_area("Tickers (commas, spaces, or new lines)", "AAPL, MSFT")
        start, end = _date_fields("fetch", intraday=timeframe in INTRADAY_DURATIONS)
        fetch_options = {}
        if provider == "massive":
            fetch_options["massive_request_interval_seconds"] = st.number_input(
                "Minimum seconds between API requests", min_value=0.0, value=0.0, step=1.0,
                key="fetch-massive-request-interval",
            )
            st.caption("Massive only: its free plan allows 5 requests per minute. About 13 seconds "
                       "helps stay within that limit when no other downloads share your API key. "
                       "0 disables deliberate pacing.")
        overlap_choice = st.selectbox(
            "Existing data",
            ["Fail on overlap", "Skip covered requests; refresh overlapping ranges",
             "Fetch missing ranges; merge locally"],
            key="fetch-overlap-policy",
        )
        st.caption("Both optional modes skip fully covered requests without contacting the provider. "
                   "Refresh downloads the combined range again. Fetch missing ranges downloads only "
                   "uncovered portions and merges them locally, preserving existing bars. "
                   "Historical revisions are retained; processed data derived from replaced inputs "
                   "becomes inactive and needs rebuilding.")
        skip = None
        if provider == "yahoo":
            skip = st.checkbox("Skip bars where every OHLC price is missing", value=False)
            st.caption("Skipped timestamps and reasons are recorded. Other invalid values still fail validation.")
        submitted = st.form_submit_button("Fetch and save", type="primary")
    if submitted:
        if not _tickers(tickers, provider) or start >= end:
            st.error("Provide at least one ticker and an end after the start.")
            return
        progress_area = st.container()
        ticker_progress = {}

        def show_progress(event):
            index = event["index"]
            if index not in ticker_progress:
                ticker_progress[index] = progress_area.empty()
            lines = [f'**{event["ticker"]} ({index}/{event["total"]}) · {event["status"]}**']
            if event["fetch_ranges"]:
                lines.append("Vendor request ranges (UTC; end exclusive):")
                lines.extend(f'- {item["start"]} → {item["end"]} · {item["status"]}'
                             for item in event["fetch_ranges"])
            elif event["status"] == "skipped":
                lines.append("Already covered; no vendor request.")
            ticker_progress[index].markdown("\n\n".join(lines))

        overlap_policies = {
            "Fail on overlap": "error",
            "Skip covered requests; refresh overlapping ranges": "refresh_overlap",
            "Fetch missing ranges; merge locally": "fetch_missing",
        }
        with st.spinner("Fetching and verifying each ticker…"):
            report = research.pipeline.fetch_many(_tickers(tickers, provider), provider=provider, timeframe=timeframe,
                                                  start=start, end=end, skip_missing_ohlc=skip,
                                                  overlap_policy=overlap_policies[overlap_choice],
                                                  progress=show_progress, **fetch_options)
        st.session_state["fetch-report"] = report
    report = st.session_state.get("fetch-report")
    if report is not None:
        st.subheader("Most recent fetch report")
        st.dataframe(report.to_frame(), hide_index=True)
        st.caption("row_count is the saved dataset size; fetched_row_count counts newly downloaded bars. "
                   "fetch_ranges shows each planned vendor request and its status. fetch_start/end spans "
                   "the plan and can include stored data that was not downloaded.")
        counts = {status: sum(item.status == status for item in report.outcomes)
                  for status in ("saved", "refreshed", "merged", "skipped", "empty", "failed")}
        completed = [f"{counts[status]} ticker(s) {status}" for status in ("saved", "refreshed", "merged")
                     if counts[status]]
        if completed:
            st.success("; ".join(completed) + ".")
        if counts["skipped"]:
            st.info(f'{counts["skipped"]} ticker(s) already covered; no vendor request was made for them.')
        if counts["empty"]:
            st.info(f'{counts["empty"]} ticker(s) returned no bars; no datasets were changed for those requests.')
        if counts["failed"]:
            st.warning(f'{counts["failed"]} ticker(s) failed. Review their errors above; the remaining tickers were attempted.')
        for outcome in report.outcomes:
            if getattr(outcome, "warning_message", None):
                st.warning(f"{outcome.ticker}: {outcome.warning_message}")


def _show_result(run):
    st.subheader(f"Run {run.run_id}")
    if run.manifest.get("price_basis") == "processed_transformed_prices":
        st.info("This run executes at transformed prices. Cash, trade sizes, and account values use those transformed price units.")
    st.caption("Each ticker has an independent account. Price returns exclude dividends; these are not shared-capital portfolio results.")
    st.dataframe(run.comparison, hide_index=True)
    st.caption("Return and drawdown metrics are fractions: 0.10 means 10%.")
    if not run.results:
        return
    selected = st.selectbox("Inspect result", list(range(len(run.results))),
                            format_func=lambda index: f"{run.results[index].symbol} · {run.results[index].strategy_spec['name']}",
                            key=f"result-{run.run_id}")
    result = run.results[selected]
    equity = result.equity
    # A closing valuation is known at the bar's close, not at its source
    # start/date label. Keep the original labels in the account table below.
    curve = equity.with_columns(pl.col("valuation_time").alias("timestamp"))
    st.caption("Equity and drawdown are plotted at their actual valuation times in UTC.")
    st.plotly_chart(line_chart(curve, columns=["equity"], title="Account equity", timeframe="1h"),
                    config=CHART_CONFIG, width="stretch", key=f"equity-{run.run_id}-{selected}")
    drawdown = curve.select("timestamp", (pl.col("equity") / pl.col("equity").cum_max() - 1).alias("drawdown"))
    st.plotly_chart(line_chart(drawdown, columns=["drawdown"], title="Drawdown", timeframe="1h",
                               y_title="Fraction below prior peak"), config=CHART_CONFIG, width="stretch",
                    key=f"drawdown-{run.run_id}-{selected}")
    with st.expander("Trades and orders"):
        st.write("Executed trades")
        st.dataframe(result.trades, hide_index=True)
        st.write("Orders")
        st.dataframe(result.orders, hide_index=True)
    with st.expander("Account values and metrics"):
        st.dataframe(result.equity, hide_index=True)
        st.json(result.metrics)
    with st.expander("Reproducibility manifest"):
        st.json(run.manifest)


def _backtest_page(research: Research):
    memory = backtest_memory(research)
    try:
        _render_backtest_page(research, memory)
    finally:
        memory.save()
        if memory.warning:
            st.warning(memory.warning)


def _render_backtest_page(research: Research, memory):
    st.title("Backtest strategies")
    st.caption("All selected tickers must pass local-data and warm-up checks before any strategy runs. "
               "Orders use the next bar's open; this screen never fetches missing data.")
    registry = research.strategy_registry
    layer = memory.widget(st.radio, "Data source", options=["raw", "processed"],
                          horizontal=True, key="backtest-layer", format_func=str.title)
    local = research.pipeline.list_datasets(DataQuery(layer=layer))
    if not local:
        st.session_state.pop("backtest-run", None)
        st.info("No active processed datasets. Open Process data to create one." if layer == "processed"
                else "No local raw datasets. Open Fetch to download data.")
        return
    provider = _provider_choice((item.request.provider for item in local),
                                key="backtest-provider", result_key="backtest-run", memory=memory)
    if provider is None:
        return
    local = [item for item in local if item.request.provider == provider]
    pipeline_id = None
    if layer == "processed":
        pipeline_ids = sorted({item.pipeline_id for item in local})
        pipeline_labels = {value: pipeline_label(research, value, items=local) for value in pipeline_ids}
        pipeline_id = memory.widget(st.selectbox, "Processing pipeline", options=pipeline_ids,
                                    key="backtest-pipeline", format_func=pipeline_labels.get)
        local = [item for item in local if item.pipeline_id == pipeline_id]
        st.caption(f"Pipeline ID: {pipeline_id}. Preflight verifies its processors and raw-source lineage.")
    timeframes = (list(BACKTEST_TIMEFRAMES) if layer == "raw" else
                  [value for value in BACKTEST_TIMEFRAMES if any(
                      item.request.timeframe == value for item in local)])
    if not timeframes:
        st.session_state.pop("backtest-run", None)
        st.info("This pipeline has no output intervals supported by backtesting.")
        return
    timeframe = memory.widget(st.selectbox, "Bar size", options=timeframes, key="backtest-timeframe")
    symbols = sorted({item.request.symbol for item in local
                      if layer == "raw" or item.request.timeframe == timeframe})
    tickers = memory.widget(st.multiselect, "Local tickers", options=symbols,
                            default=symbols[:1], key="backtest-tickers")
    extra = memory.widget(st.text_input, "Additional required tickers", key="backtest-extra",
                          help="A missing ticker will be reported by preflight and abort the run.")
    start, end = _date_fields("backtest", intraday=timeframe in INTRADAY_DURATIONS, memory=memory)
    available = sorted({item.request.symbol for item in local if item.request.timeframe == timeframe})
    st.caption(f"Local {timeframe} data: {', '.join(available) if available else 'none'}. "
               "Preflight requires this exact interval, including warm-up; other intervals are not substituted.")
    if layer == "processed":
        st.caption("Every ticker uses this same pipeline. Keep its raw parent files for lineage verification. "
                   "Choose a start date with enough earlier processed bars for strategy warm-up.")
    chosen = list(dict.fromkeys(tickers + _tickers(extra, provider)))
    source_identity = (layer, provider, pipeline_id, timeframe, tuple(chosen), start, end,
                       tuple(sorted(item.dataset_id for item in local
                                    if item.request.timeframe == timeframe and item.request.symbol in chosen)))
    if st.session_state.get("backtest-source-identity") != source_identity:
        st.session_state.pop("backtest-run", None)
        st.session_state["backtest-source-identity"] = source_identity
    specs = strategy_specs(registry)
    st.caption("Strategy lookbacks count bars at the selected interval. Download history before the test start; "
               "preflight checks the longest selected requirement for every ticker.")
    st.caption("After changing private strategy Python files, restart the dashboard to load the new code.")
    st.caption("Calendar: U.S. regular equity sessions (XNYS), including holidays and early closes.")
    st.caption("Use USD-quoted equities on this calendar. Exchange and currency are explicit assumptions; "
               "they are not inferred from ticker names.")
    cash = st.number_input("Initial cash per ticker", min_value=1.0, value=10000.0)
    first, second, third = st.columns(3)
    fixed = first.number_input("Fixed commission per trade", min_value=0.0, value=0.0)
    bps = second.number_input("Commission (basis points)", min_value=0.0, value=0.0)
    slippage = third.number_input("Slippage (basis points)", min_value=0.0, max_value=9999.0, value=0.0)
    submitted = st.button("Validate all data and run", type="primary", disabled=specs is None)
    if submitted:
        st.session_state.pop("backtest-run", None)
        if not chosen or not specs or start >= end:
            st.error("Choose tickers, at least one strategy, and a valid date range.")
            return
        settings = ExecutionSettings(initial_cash=cash, commission_fixed=fixed,
                                     commission_bps=bps, slippage_bps=slippage)
        try:
            with st.spinner("Checking all datasets, then running strategies…"):
                run = research.backtest(chosen, strategies=specs, start=start, end=end,
                                        timeframe=timeframe, provider=provider, settings=settings,
                                        calendar="XNYS", layer=layer, pipeline_id=pipeline_id)
            st.session_state["backtest-run"] = run
        except PreflightError as error:
            st.session_state.pop("backtest-run", None)
            st.error("Backtest aborted: the required data did not pass preflight.")
            report = getattr(error, "report", None)
            if report is not None:
                st.dataframe(report.to_frame(), hide_index=True)
            else:
                st.error(str(error))
    if st.session_state.get("backtest-run") is not None:
        _show_result(st.session_state["backtest-run"])


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


def main():
    st.set_page_config(page_title="Quant research", page_icon="📈", layout="wide")
    with st.sidebar:
        st.title("Quant research")
        directories = market_memory("dashboard-directories")
        data_dir = directories.widget(st.text_input, "Data directory", value="data", key="data-directory")
        runs_dir = directories.widget(st.text_input, "Research runs directory", value="runs", key="runs-directory")
        directories.save()
        if directories.warning:
            st.warning(directories.warning)
        st.caption(f"Data: {Path(data_dir).expanduser().resolve()}")
        page = st.radio("Navigate", ["Market data", "Fetch", "Process data", "Backtest", "Saved runs", "Delete data"], key="dashboard-page")
        st.caption("Local data · reproducible research")
    # A store switch must not leave results from another directory on screen.
    identity = (str(Path(data_dir).expanduser().resolve()), str(Path(runs_dir).expanduser().resolve()))
    if st.session_state.get("store-identity") != identity:
        for key in ("fetch-report", "backtest-run"):
            st.session_state.pop(key, None)
        st.session_state["store-identity"] = identity
    try:
        research = Research(data_dir=data_dir, runs_dir=runs_dir)
        {"Market data": _data_page, "Fetch": _fetch_page, "Process data": processing_page,
         "Backtest": _backtest_page, "Saved runs": _runs_page, "Delete data": deletion_page}[page](research)
    except Exception as error:
        # Domain errors are displayed while preserving the local store; full
        # traceback remains available for diagnosing a failed operation.
        st.error(f"{type(error).__name__}: {error}")
        with st.expander("Error details"):
            st.exception(error)


if __name__ == "__main__":
    main()
