"""Configure local backtests and present their results."""

import polars as pl
import streamlit as st

from data_pipeline import DataQuery
from dashboard.charts import CHART_CONFIG, line_chart
from dashboard.components.source_controls import _date_fields, _provider_choice, _tickers
from dashboard.components.strategy_controls import strategy_specs
from dashboard.pages.processing import backtest_memory, pipeline_label
from research import PreflightError, Research
from research.backtesting import ExecutionSettings
from research.timeframes import BACKTEST_TIMEFRAMES, INTRADAY_DURATIONS

def _show_result(run, *, ui=st):
    st = ui
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


