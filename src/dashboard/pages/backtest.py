"""Configure local backtests and present their results."""

import os

import polars as pl
import streamlit as st

from data_pipeline import DataQuery
from dashboard.charts import CHART_CONFIG, line_chart
from dashboard.components.source_controls import _date_fields, _provider_choice, _tickers
from dashboard.components.strategy_controls import strategy_specs
from dashboard.pages.processing import backtest_memory, pipeline_label
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from research.timeframes import BACKTEST_TIMEFRAMES, INTRADAY_DURATIONS

def _show_result(run, *, ui=st):
    st = ui
    st.subheader(f"Run {run.run_id}")
    warnings = run.manifest.get("warnings", [])
    if warnings:
        st.warning("Early-close sessions were handled with a shorter trading period. See warnings below.")
        with st.expander("Backtest warnings", expanded=True):
            st.dataframe(pl.DataFrame(warnings), hide_index=True)
    ticker_errors = run.manifest.get("ticker_errors", [])
    if ticker_errors:
        count = len({issue["ticker"] for issue in ticker_errors})
        st.warning(f"Skipped {count} problematic ticker(s). Results and capital allocation include only tickers that passed validation.")
        st.subheader("Skipped ticker errors")
        st.dataframe(pl.DataFrame(ticker_errors).select("ticker", "code", "message"), hide_index=True)
    if run.manifest.get("price_basis") == "processed_transformed_prices":
        st.info("This run executes at transformed prices. Cash, trade sizes, and account values use those transformed price units.")
    portfolio_results = [result for result in run.results
                         if result.metadata.get("account_model", "").startswith("shared_cash_long_only")]
    initial_split = run.manifest.get("allocation_mode") == "initial_split"
    if initial_split:
        st.caption("Each strategy splits its total initial capital equally across tickers once. "
                   "Each ticker keeps its cash and profits; funds are never transferred between tickers.")
        summaries = run.manifest.get("allocation_summary", [])
        if summaries:
            st.subheader("All-ticker totals")
            st.dataframe(pl.DataFrame(summaries, infer_schema_length=None).select(
                pl.exclude("strategy_index", "^.*_exact$")
            ), hide_index=True)
        st.subheader("Per-ticker results")
    elif portfolio_results and len(portfolio_results) == len(run.results):
        st.caption("Each strategy uses one shared cash account across all selected tickers. Price returns exclude dividends.")
    elif portfolio_results:
        st.caption("Portfolio strategies share cash across selected tickers; allocation strategies use an independent account per ticker. "
                   "Each result is a separate simulation. Price returns exclude dividends.")
    else:
        st.caption("Each ticker has an independent account. Price returns exclude dividends; these are not shared-capital portfolio results.")
    st.dataframe(run.comparison, hide_index=True)
    st.caption("Return and drawdown metrics are fractions: 0.10 means 10%.")
    if run.manifest.get("skip_initial_missing_bars") or run.manifest.get("skip_problematic_tickers"):
        delayed = {symbol: stamp for symbol, stamp in run.manifest.get("effective_starts", {}).items()
                   if stamp > run.manifest["start"]}
        skipped = run.manifest.get("skipped_tickers", {})
        if delayed or skipped:
            with st.expander("Data availability"):
                if delayed:
                    st.write("These tickers begin trading after their available history provides enough warm-up bars.")
                    st.dataframe(pl.DataFrame({"ticker": list(delayed), "trading_start": list(delayed.values())}),
                                 hide_index=True)
                if skipped:
                    st.write("These tickers were excluded from the run and its capital allocation.")
                    st.dataframe(pl.DataFrame({"ticker": list(skipped), "reason": list(skipped.values())}),
                                 hide_index=True)
    if not run.results:
        return
    selected = st.selectbox("Inspect result", list(range(len(run.results))),
                            format_func=lambda index: f"{run.results[index].symbol} · {run.results[index].strategy_spec['name']}",
                            key=f"result-{run.run_id}")
    result = run.results[selected]
    if "wins" in result.metrics:
        if initial_split:
            st.caption("This ticker trades only its initial allocation and its own subsequent sale proceeds.")
        elif run.manifest.get("allocation_mode") == "fixed_tenth":
            st.caption(f"Each entry budgets {run.manifest.get('entry_percentage', '10')}% of initial capital, including costs. Unfunded signals are skipped; "
                       "later signals can trade after sales replenish cash. Simultaneous entries use ticker order.")
        else:
            st.caption("Entry budget = available cash ÷ eligible tickers not currently held. "
                       "Same-open entries use the same cash snapshot; existing holdings are not rebalanced.")
        counts = st.columns(3)
        for column, label, key in zip(counts, ("Wins", "Losses", "Inconclusive stops"),
                                      ("wins", "losses", "inconclusive")):
            column.metric(label, result.metrics[key])
        profits = st.columns(3)
        for column, label, key in zip(profits, ("Realized P&L", "Unrealized P&L", "Total P&L"),
                                      ("realized_pnl", "unrealized_pnl", "total_pnl")):
            column.metric(label, f"${result.metrics[key]:,.2f}")
        st.caption("Inconclusive stops are counted separately and sold at the modeled stop price. "
                   "P&L includes commissions and slippage. Open positions are valued at their final close.")
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
        if result.completed_trades.width:
            st.write("Completed trades")
            st.dataframe(result.completed_trades, hide_index=True)
        if result.open_positions.width:
            st.write("Open positions")
            st.dataframe(result.open_positions, hide_index=True)
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
    st.caption("Local-data and warm-up checks run before any strategy. You can exclude tickers that fail these checks. "
               "Entries use the next bar's open; stop/target strategies can exit within a bar. "
               "This screen never fetches missing data.")
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
    trading_period = "regular"
    if timeframe == "1h":
        if memory.values["widgets"].get("backtest-trading-period") == "09:00-16:00":
            memory.values["widgets"]["backtest-trading-period"] = "09:00-15:00"
        trading_period = memory.widget(
            st.radio, "Trading period", options=["regular", "09:00-15:00"], key="backtest-trading-period",
            format_func={"regular": "Regular session (09:30–16:00)",
                         "09:00-15:00": "Hourly bar starts 09:00–15:00 (New York)"}.get,
            help="Includes bar starts 09:00 through 15:00; the final bar ends at 16:00. Bars outside this window are excluded. Early-close days end at the official close; later bars are excluded and reported as warnings.",
        )
    symbols = sorted({item.request.symbol for item in local
                      if layer == "raw" or item.request.timeframe == timeframe})
    tickers = memory.widget(st.multiselect, "Local tickers", options=symbols,
                            default=symbols[:1], key="backtest-tickers")
    extra = memory.widget(st.text_input, "Additional required tickers", key="backtest-extra",
                          help="Missing tickers abort preflight unless skipping problematic tickers or allowing missing history at start is enabled.")
    start, end = _date_fields("backtest", intraday=timeframe in INTRADAY_DURATIONS, memory=memory)
    skip_initial_missing_bars = memory.widget(
        st.checkbox, "Allow missing history at start", value=False, key="backtest-skip-initial-missing",
        help="Late-starting tickers begin after enough warm-up bars. Gaps after their first available bar still fail validation.",
    )
    skip_problematic_tickers = memory.widget(
        st.checkbox, "Skip problematic tickers", value=False, key="backtest-skip-problematic",
        help="Exclude tickers that fail data validation and report their errors in the summary. "
             "Capital is allocated only among valid tickers. If none pass, the run stops. Strategy execution errors still stop the run.",
    )
    available = sorted({item.request.symbol for item in local if item.request.timeframe == timeframe})
    st.caption(f"Local {timeframe} data: {', '.join(available) if available else 'none'}. "
               "Preflight requires this exact interval, including warm-up; other intervals are not substituted.")
    if layer == "processed":
        st.caption("Every ticker uses this same pipeline. Keep its raw parent files for lineage verification. "
                   "Choose a start date with enough earlier processed bars for strategy warm-up.")
    chosen = list(dict.fromkeys(tickers + _tickers(extra, provider)))
    source_identity = (layer, provider, pipeline_id, timeframe, tuple(chosen), start, end, skip_initial_missing_bars,
                       skip_problematic_tickers, trading_period,
                       tuple(sorted(item.dataset_id for item in local
                                    if item.request.timeframe == timeframe and item.request.symbol in chosen)))
    if st.session_state.get("backtest-source-identity") != source_identity:
        st.session_state.pop("backtest-run", None)
        st.session_state["backtest-source-identity"] = source_identity
    specs = strategy_specs(registry, memory=memory)
    st.caption("Strategy lookbacks count bars at the selected interval. Download history before the test start; "
               "preflight checks the longest selected requirement for every ticker.")
    st.caption("After changing private strategy Python files, restart the dashboard to load the new code.")
    st.caption("Calendar: U.S. regular equity sessions (XNYS), including holidays and early closes.")
    st.caption("Use USD-quoted equities on this calendar. Exchange and currency are explicit assumptions; "
               "they are not inferred from ticker names.")
    allocation_mode = memory.widget(
        st.radio, "Capital allocation", options=["per_entry", "initial_split", "fixed_tenth"],
        format_func={"per_entry": "Allocate at each entry (shared cash)",
                     "initial_split": "Split once at start (parallel tickers)",
                     "fixed_tenth": "Fixed percentage of initial capital per entry"}.get,
        key="backtest-allocation-mode",
    )
    entry_percentage = 10.0
    if allocation_mode == "fixed_tenth":
        entry_percentage = memory.widget(st.number_input, "Initial capital per entry (%)", min_value=0.01,
                                         max_value=100.0, value=10.0, step=1.0,
                                         key="backtest-entry-percentage")
    max_workers = memory.widget(st.number_input, "Parallel workers", min_value=1, max_value=32,
                                value=min(4, os.cpu_count() or 1), step=1,
                                disabled=allocation_mode != "initial_split", key="backtest-max-workers")
    cash = memory.widget(st.number_input, "Initial capital", min_value=1.0, value=10000.0,
                         key="backtest-initial-capital")
    if allocation_mode == "initial_split":
        st.caption("This is the total initial capital per strategy, split equally across included tickers once. "
                   "Each ticker keeps its own cash and profits, so tickers can run in parallel.")
    elif allocation_mode == "fixed_tenth":
        st.caption(f"Stop/target strategies budget {entry_percentage:g}% of initial capital per entry, including fees and slippage. "
                   "Skip signals when cash cannot cover the full allocation; later signals can use sale proceeds. "
                   "Simultaneous signals are considered in ticker order. Shared cash runs sequentially.")
    else:
        st.caption("Stop/target strategies use this initial cash once across all selected tickers, allocating free cash at each entry. "
                   "Other strategies retain a separate full-size account for each ticker. Shared cash runs sequentially.")
    fractional_shares = memory.widget(st.checkbox, "Allow fractional shares", value=True,
                                      key="backtest-fractional-shares",
                                      help="New purchases need at least a $5 budget, including fees. No $5 cash reserve is required.")
    first, second, third = st.columns(3)
    fixed = memory.widget(first.number_input, "Fixed commission per trade", min_value=0.0, value=0.0,
                          key="backtest-commission-fixed")
    bps = memory.widget(second.number_input, "Commission (basis points)", min_value=0.0, value=0.0,
                        key="backtest-commission-bps")
    slippage = memory.widget(third.number_input, "Slippage (basis points)", min_value=0.0, max_value=9999.0, value=0.0,
                             key="backtest-slippage-bps")
    submitted = st.button("Validate all data and run", type="primary", disabled=specs is None)
    if submitted:
        st.session_state.pop("backtest-run", None)
        if not chosen or not specs or start >= end:
            st.error("Choose tickers, at least one strategy, and a valid date range.")
            return
        try:
            settings = ExecutionSettings(initial_cash=cash, commission_fixed=fixed,
                                         commission_bps=bps, slippage_bps=slippage,
                                         fractional_shares=fractional_shares)
            with st.spinner("Checking all datasets, then running strategies…"):
                run = research.backtest(chosen, strategies=specs, start=start, end=end,
                                        timeframe=timeframe, provider=provider, settings=settings,
                                        calendar="XNYS", layer=layer, pipeline_id=pipeline_id,
                                        allocation_mode=allocation_mode, max_workers=int(max_workers), entry_percentage=entry_percentage,
                                        skip_initial_missing_bars=skip_initial_missing_bars,
                                        skip_problematic_tickers=skip_problematic_tickers, trading_period=trading_period)
            st.session_state["backtest-run"] = run
        except PreflightError as error:
            st.session_state.pop("backtest-run", None)
            st.error("Backtest aborted: the required data did not pass preflight.")
            report = getattr(error, "report", None)
            if report is not None:
                st.dataframe(report.to_frame(), hide_index=True)
                if report.warnings:
                    st.warning("Early-close warnings")
                    st.dataframe(pl.DataFrame(report.warnings), hide_index=True)
            else:
                st.error(str(error))
        except (ResearchError, ValueError, TypeError) as error:
            st.session_state.pop("backtest-run", None)
            st.error(f"Backtest aborted: {error}")
    if st.session_state.get("backtest-run") is not None:
        _show_result(st.session_state["backtest-run"])


