"""Run locally with: python -m streamlit run src/dashboard/app.py."""

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import streamlit as st

from data_pipeline import DataQuery
from dashboard.charts import CHART_CONFIG, bar_table, line_chart, price_chart
from dashboard.deletion import deletion_page
from dashboard.strategy_controls import strategy_specs
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from research.timeframes import BACKTEST_TIMEFRAMES, INTRADAY_DURATIONS, fetch_timeframes


def _utc_date(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _tickers(text: str, provider: str = "yahoo") -> list[str]:
    items = text.replace(",", " ").split()
    return list(dict.fromkeys(item.upper() if provider == "yahoo" else item for item in items))


def _provider_choice(names, *, key, result_key=None):
    names = sorted(set(names))
    if not names:
        st.info("No providers are available here. Fetching requires a registered provider; browsing and backtesting require local datasets.")
        return None
    provider = st.selectbox("Provider", names, key=key)
    if result_key and st.session_state.get(f"{key}-previous") != provider:
        st.session_state.pop(result_key, None)
    st.session_state[f"{key}-previous"] = provider
    return provider


def _date_fields(prefix: str, *, first: date | None = None, last: date | None = None,
                 intraday: bool = False):
    left, right = st.columns(2)
    first = first or date.today() - timedelta(days=30)
    last = last or date.today()
    start = left.date_input("Start date (UTC, inclusive)", first, key=f"{prefix}-start")
    end = right.date_input("End date (UTC, exclusive)", last, key=f"{prefix}-end")
    if intraday:
        start_time = left.time_input("Start time (UTC)", time.min, step=60, key=f"{prefix}-start-time")
        end_time = right.time_input("End time (UTC)", time.min, step=60, key=f"{prefix}-end-time")
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


def _quality(items, start, end, *, present_timestamps=()):
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
                omitted.append({"timestamp_utc": bar.timestamp, "reason": bar.reason,
                                "dataset_id": item.dataset_id,
                                "current_state": "present in selected data" if bar.timestamp in present
                                                 else "absent from selected data"})
    if omitted:
        absent = {item["timestamp_utc"] for item in omitted if item["timestamp_utc"] not in present}
        restored = {item["timestamp_utc"] for item in omitted if item["timestamp_utc"] in present}
        if absent:
            st.warning(f"{len(absent)} recorded omitted bars are still absent in this window. Dotted lines mark them on the chart.")
        if restored:
            st.info(f"{len(restored)} previously omitted bars are present in the selected data. Their omission records are historical.")
        st.dataframe(pl.DataFrame(omitted), hide_index=True)
    if unknown:
        st.caption(f"Omission history is unavailable for {len(unknown)} selected revision(s). "
                   "Valid stored values and first/last bounds do not establish complete coverage.")
    elif not omitted:
        st.caption("No ingestion omissions were recorded in this window. Backtest preflight checks expected bar coverage separately.")
    return sorted({item["timestamp_utc"] for item in omitted if item["timestamp_utc"] not in present})


def _data_page(research: Research):
    st.title("Market data")
    st.caption("Explore stored bars and their revisions. All chart values come from the verified local data.")
    layer = st.radio("Layer", ["raw", "processed"], horizontal=True)
    include_history = st.checkbox("Include historical revisions")
    items = research.pipeline.list_datasets(DataQuery(layer=layer, include_history=include_history))
    if not items:
        st.info("No matching local datasets. Open Fetch to download your first batch.")
        return
    provider = _provider_choice((item.request.provider for item in items), key="browse-provider")
    items = [item for item in items if item.request.provider == provider]
    search = st.text_input("Find a ticker", placeholder="AAPL")
    symbols = sorted({item.request.symbol for item in items if search.upper() in item.request.symbol.upper()})
    if not symbols:
        st.info("No tickers match this search.")
        return
    symbol = st.selectbox("Ticker", symbols)
    symbol_items = [item for item in items if item.request.symbol == symbol]
    identities = sorted({(item.request.provider, item.request.timeframe, item.pipeline_id) for item in symbol_items})
    identity = st.selectbox("Data series", identities,
                            format_func=lambda value: f"{value[0]} · {value[1]}" + (f" · pipeline {value[2][:12]}" if value[2] else ""))
    matching = [item for item in symbol_items
                if (item.request.provider, item.request.timeframe, item.pipeline_id) == identity]
    mode = st.radio("Read", ["Active batches together", "One exact revision"], horizontal=True)
    if mode == "One exact revision":
        selected_id = st.selectbox("Revision", [item.dataset_id for item in matching],
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
    start, end = _date_fields(window_key, first=lower.date(), last=(upper + timedelta(days=1)).date())
    timezone = st.selectbox("Display timezone (intraday bars)", ["America/New_York", "UTC", "Asia/Shanghai"])
    if start >= end:
        st.error("The end must be later than the start.")
        return
    st.caption("Drag to zoom, scroll to zoom, or use the range slider. Double-click to reset. "
               "Daily and longer bars retain their session-date labels in every timezone.")
    if identity[1] in INTRADAY_DURATIONS:
        st.caption("Each candle is labeled by its start. The final bar of a trading session may be shorter than the selected interval.")
    if mode == "One exact revision":
        frame = research.pipeline.read_dataset(selected[0].dataset_id).filter(
            (pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
    else:
        frame = research.pipeline.read(DataQuery(layer=layer, provider=identity[0], symbol=symbol,
                                               timeframe=identity[1], pipeline_id=identity[2] or None, start=start, end=end))
    with st.expander("Dataset metadata"):
        st.dataframe(pl.DataFrame(_metadata_rows(selected)), hide_index=True)
    omitted = _quality(selected, start, end, present_timestamps=frame["timestamp"].to_list())
    if frame.is_empty():
        st.info("No saved bars fall inside this window.")
        return
    st.caption(f"{frame.height:,} original bars shown · no downsampling or gap filling")
    if frame.height > 20_000:
        st.warning("This window contains more than 20,000 bars and may render slowly. Choose a shorter date range if needed.")
    st.plotly_chart(price_chart(frame, timeframe=identity[1], timezone=timezone,
                               title=f"{symbol} · {identity[1]}", omitted_timestamps=omitted),
                    config=CHART_CONFIG, width="stretch")
    with st.expander("OHLCV rows"):
        st.dataframe(bar_table(frame, timeframe=identity[1], timezone=timezone), hide_index=True,
                     column_config={column: st.column_config.NumberColumn(format="plain")
                                    for column in ("open", "high", "low", "close", "volume")})
        st.download_button("Download displayed bars as CSV", frame.write_csv(),
                           file_name=f"{symbol}-{identity[1]}.csv", mime="text/csv")


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
        skip = None
        if provider == "yahoo":
            skip = st.checkbox("Skip bars where every OHLC price is missing", value=False)
            st.caption("Skipped timestamps and reasons are recorded. Other invalid values still fail validation.")
        submitted = st.form_submit_button("Fetch and save", type="primary")
    if submitted:
        if not _tickers(tickers, provider) or start >= end:
            st.error("Provide at least one ticker and an end after the start.")
            return
        with st.spinner("Fetching and verifying each ticker…"):
            report = research.pipeline.fetch_many(_tickers(tickers, provider), provider=provider, timeframe=timeframe,
                                                  start=start, end=end, skip_missing_ohlc=skip, **fetch_options)
        st.session_state["fetch-report"] = report
    report = st.session_state.get("fetch-report")
    if report is not None:
        st.subheader("Most recent fetch report")
        st.dataframe(report.to_frame(), hide_index=True)
        failed = sum(item.status == "failed" for item in report.outcomes)
        if failed:
            st.warning(f"{failed} ticker(s) failed. Review their errors above; the remaining tickers were attempted.")
        else:
            st.success("All ticker requests completed.")


def _show_result(run):
    st.subheader(f"Run {run.run_id}")
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
    st.title("Backtest strategies")
    st.caption("All selected tickers must pass local-data and warm-up checks before any strategy runs. "
               "Orders use the next bar's open; this screen never fetches missing data.")
    # Discover strategies on this page even when the store has no raw data yet.
    registry = research.strategy_registry
    local = research.pipeline.list_datasets(DataQuery(layer="raw"))
    provider = _provider_choice((item.request.provider for item in local),
                                key="backtest-provider", result_key="backtest-run")
    if provider is None:
        return
    local = [item for item in local if item.request.provider == provider]
    symbols = sorted({item.request.symbol for item in local})
    tickers = st.multiselect("Local tickers", symbols, default=symbols[:1])
    extra = st.text_input("Additional required tickers", help="A missing ticker will be reported by preflight and abort the run.")
    timeframe = st.selectbox("Bar size", BACKTEST_TIMEFRAMES, key="backtest-timeframe")
    start, end = _date_fields("backtest", intraday=timeframe in INTRADAY_DURATIONS)
    available = sorted({item.request.symbol for item in local if item.request.timeframe == timeframe})
    st.caption(f"Local {timeframe} data: {', '.join(available) if available else 'none'}. "
               "Preflight requires this exact interval, including warm-up; other intervals are not substituted.")
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
        chosen = list(dict.fromkeys(tickers + _tickers(extra, provider)))
        if not chosen or not specs or start >= end:
            st.error("Choose tickers, at least one strategy, and a valid date range.")
            return
        settings = ExecutionSettings(initial_cash=cash, commission_fixed=fixed,
                                     commission_bps=bps, slippage_bps=slippage)
        try:
            with st.spinner("Checking all datasets, then running strategies…"):
                run = research.backtest(chosen, strategies=specs, start=start, end=end,
                                        timeframe=timeframe, provider=provider, settings=settings, calendar="XNYS")
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
        data_dir = st.text_input("Data directory", "data")
        runs_dir = st.text_input("Research runs directory", "runs")
        st.caption(f"Data: {Path(data_dir).expanduser().resolve()}")
        page = st.radio("Navigate", ["Market data", "Fetch", "Backtest", "Saved runs", "Delete data"])
        st.caption("Local data · reproducible research")
    # A store switch must not leave results from another directory on screen.
    identity = (str(Path(data_dir).expanduser().resolve()), str(Path(runs_dir).expanduser().resolve()))
    if st.session_state.get("store-identity") != identity:
        for key in ("fetch-report", "backtest-run"):
            st.session_state.pop(key, None)
        st.session_state["store-identity"] = identity
    try:
        research = Research(data_dir=data_dir, runs_dir=runs_dir)
        {"Market data": _data_page, "Fetch": _fetch_page,
         "Backtest": _backtest_page, "Saved runs": _runs_page, "Delete data": deletion_page}[page](research)
    except Exception as error:
        # Domain errors are displayed while preserving the local store; full
        # traceback remains available for diagnosing a failed operation.
        st.error(f"{type(error).__name__}: {error}")
        with st.expander("Error details"):
            st.exception(error)


if __name__ == "__main__":
    main()
