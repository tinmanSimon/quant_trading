"""Factor research workspace and inspectable shared-account rebalances."""

from datetime import timedelta
from pathlib import Path

import polars as pl
import streamlit as st

from dashboard.charts import CHART_CONFIG, line_chart
from dashboard.components.research_mode import research_mode_control, show_research_mode
from dashboard.components.source_controls import _date_fields, _provider_choice, _tickers
from dashboard.pages.processing import pipeline_label
from dashboard.state.workspace import market_memory
from data_pipeline import DataQuery
from research import PreflightError, ResearchError
from research.backtesting import ExecutionSettings
from research.timeframes import BACKTEST_TIMEFRAMES, INTRADAY_DURATIONS


FACTOR_LABELS = {
    "momentum": "Momentum",
    "volatility": "Return volatility",
    "red_bar_high_percentage": "Red-bar high percentage",
}
POLICY_LABELS = {
    "equal_weight": "Top-N equal weight",
    "score_weight": "Top-N score weight",
    "inverse_volatility": "Top-N inverse volatility",
}
SCHEDULE_LABELS = {
    "every_n_bars": "Every N bars", "daily": "Daily", "weekly": "Weekly", "monthly": "Monthly",
}


def factor_memory(research):
    """Different stores must never share a remembered result or source selection."""
    scope = f"factor-backtest:{Path(research.pipeline.store.data_dir).resolve()}:{research.runs_dir.resolve()}"
    return market_memory(scope)


def factor_backtest_page(research):
    memory = factor_memory(research)
    try:
        _render_page(research, memory)
    finally:
        memory.save()
        if memory.warning:
            st.warning(memory.warning)


def _source_controls(research, memory):
    layer = memory.widget(st.radio, "Data source", options=["raw", "processed"], horizontal=True,
                          key="factor-layer", format_func=str.title)
    local = research.pipeline.list_datasets(DataQuery(layer=layer))
    if not local:
        st.info("No active processed datasets. Open Process data to create one." if layer == "processed"
                else "No local raw datasets. Open Fetch to download data.")
        return None
    provider = _provider_choice((item.request.provider for item in local), key="factor-provider", memory=memory)
    local = [item for item in local if item.request.provider == provider]
    pipeline_id = None
    research_mode = "strict"
    if layer == "processed":
        pipelines = sorted({item.pipeline_id for item in local})
        labels = {value: pipeline_label(research, value, items=local) for value in pipelines}
        pipeline_id = memory.widget(st.selectbox, "Processing pipeline", options=pipelines,
                                    format_func=labels.get, key="factor-pipeline")
        local = [item for item in local if item.pipeline_id == pipeline_id]
        st.caption("Preflight checks this pipeline's raw-source lineage. All tickers use the same pipeline.")
        research_mode = research_mode_control(memory, key="factor-research-mode")
    timeframes = [value for value in BACKTEST_TIMEFRAMES if any(
        item.request.timeframe == value for item in local)]
    if not timeframes:
        st.info("These datasets have no bar sizes supported by backtesting.")
        return None
    timeframe = memory.widget(st.selectbox, "Bar size", options=timeframes, key="factor-timeframe")
    local = [item for item in local if item.request.timeframe == timeframe]
    symbols = sorted({item.request.symbol for item in local})
    chosen = memory.widget(st.multiselect, "Local tickers", options=symbols, default=symbols,
                           key="factor-tickers")
    extra = memory.widget(st.text_input, "Additional required tickers", key="factor-extra")
    first = min(item.first_timestamp for item in local).date()
    last = max(item.last_timestamp for item in local).date() + timedelta(days=1)
    start, end = _date_fields("factor", first=first, last=last,
                              intraday=timeframe in INTRADAY_DURATIONS, memory=memory)
    trading_period = "regular"
    if timeframe == "1h":
        trading_period = memory.widget(
            st.radio, "Trading period", options=["regular", "09:00-15:00"], key="factor-trading-period",
            format_func={"regular": "Regular session (09:30–16:00)",
                         "09:00-15:00": "Hourly bar starts 09:00–15:00 (New York)"}.get)
    allow_late = memory.widget(st.checkbox, "Allow missing history at start", value=False,
                              key="factor-allow-late",
                              help="A ticker joins once its available history supplies all factor lookbacks. Interior gaps still fail validation.")
    skip_bad = memory.widget(st.checkbox, "Skip problematic tickers", value=False,
                            key="factor-skip-problematic")
    if skip_bad:
        st.warning("Excluding a ticker because of data problems later in the test can introduce selection bias. "
                   "Skipped tickers and their errors will appear in the report.")
    st.caption("Calendar: XNYS, including holidays and early closes. Use USD equities. "
               "Factor lookbacks count bars; choose a start with enough earlier data for warm-up.")
    return dict(tickers=list(dict.fromkeys(chosen + _tickers(extra, provider))), provider=provider,
                start=start, end=end, timeframe=timeframe, calendar="XNYS", layer=layer,
                pipeline_id=pipeline_id, research_mode=research_mode, trading_period=trading_period,
                skip_initial_missing_bars=allow_late, skip_problematic_tickers=skip_bad)


def _spec_controls(memory):
    # Import lazily so non-factor pages remain independent of factor discovery.
    from research.factors import ComponentSpec, FactorBacktestSpec, FactorSpec

    st.subheader("Factors and scoring")
    names = memory.widget(st.multiselect, "Factors", options=list(FACTOR_LABELS),
                          default=["momentum"], format_func=FACTOR_LABELS.get, key="factor-names")
    factors = []
    for name in names:
        with st.expander(FACTOR_LABELS[name], expanded=True):
            lookback = memory.widget(st.number_input, "Lookback (bars)", min_value=2, max_value=10000,
                                     value=20, step=1, key=f"factor-{name}-lookback")
            weight = memory.widget(st.number_input, "Score contribution", min_value=0.01,
                                   value=1.0, step=0.1, key=f"factor-{name}-weight")
            direction = memory.widget(st.selectbox, "Preferred factor value", options=[1, -1],
                                      index=1 if name == "volatility" else 0,
                                      format_func={1: "Higher", -1: "Lower"}.get,
                                      key=f"factor-{name}-direction")
            factors.append(FactorSpec(name=name, lookback=int(lookback), weight=weight, direction=direction))
    st.caption("Available factor values are ranked across eligible tickers, then combined using these contributions. "
               "Factors use completed bars only; orders execute at the next eligible open.")
    st.subheader("Portfolio and rebalancing")
    policy = memory.widget(st.selectbox, "Portfolio policy", options=list(POLICY_LABELS),
                           format_func=POLICY_LABELS.get, key="factor-policy")
    top_n = memory.widget(st.number_input, "Maximum selected holdings", min_value=1, max_value=10000,
                          value=20, step=1, key="factor-top-n")
    portfolio_config = {"top_n": int(top_n)}
    if policy == "inverse_volatility":
        if "volatility" in names:
            st.caption("Position sizing uses the selected volatility factor and its lookback.")
        else:
            volatility_lookback = memory.widget(
                st.number_input, "Volatility lookback for sizing (bars)", min_value=2, max_value=10000,
                value=20, step=1, key="factor-sizing-volatility-lookback")
            portfolio_config["volatility_lookback"] = int(volatility_lookback)
            st.caption("Sizing volatility changes allocation weights without contributing to stock scores.")
    schedule = memory.widget(st.selectbox, "Rebalance schedule", options=list(SCHEDULE_LABELS),
                             index=1, format_func=SCHEDULE_LABELS.get, key="factor-schedule")
    schedule_config = {}
    if schedule == "every_n_bars":
        bars = memory.widget(st.number_input, "Bars between decisions", min_value=1, max_value=10000,
                             value=1, step=1, key="factor-schedule-bars")
        schedule_config["bars"] = int(bars)
    left, right = st.columns(2)
    reserve = memory.widget(left.number_input, "Cash reserve (%)", min_value=0.0, max_value=100.0,
                            value=0.0, step=1.0, key="factor-reserve")
    cap = memory.widget(right.number_input, "Maximum position weight (%)", min_value=0.01, max_value=100.0,
                        value=100.0, step=1.0, key="factor-cap")
    drift = memory.widget(left.number_input, "Minimum weight change to trade (%)", min_value=0.0,
                          max_value=100.0, value=0.0, step=0.1, key="factor-drift")
    turnover = memory.widget(right.number_input, "Maximum turnover per rebalance (%)", min_value=0.0,
                             max_value=200.0, value=100.0, step=5.0, key="factor-turnover",
                             help="Sum of absolute allocation changes. Replacing a fully invested portfolio can total 200%. "
                                  "Mandatory reductions for position caps or the cash reserve can exceed this budget.")
    funding = memory.widget(st.selectbox, "When purchases exceed available cash", options=["pro_rata", "score_priority"],
                            format_func={"pro_rata": "Scale purchases proportionally", "score_priority": "Fund higher scores first"}.get,
                            key="factor-funding")
    proceeds = memory.widget(st.selectbox, "Sale proceeds available for purchases", options=["same_open", "next_open"],
                             format_func={"same_open": "After successful sales at the same open", "next_open": "From the next opening event"}.get,
                             key="factor-proceeds")
    st.caption("One shared long-only account. Targets are fractions of total portfolio equity. "
               "Only changes in holdings are traded; reductions execute before purchases. "
               "Unfilled orders expire after their intended opening event.")
    if not factors:
        st.info("Select at least one factor.")
        return None
    return FactorBacktestSpec(factors=tuple(factors), portfolio=ComponentSpec(policy, config=portfolio_config),
                              schedule=ComponentSpec(schedule, config=schedule_config), cash_reserve=reserve / 100,
                              max_position_weight=cap / 100, min_weight_change=drift / 100,
                              max_turnover=turnover / 100, sale_proceeds=proceeds, funding=funding)


def _execution_controls(memory):
    st.subheader("Execution")
    cash = memory.widget(st.number_input, "Initial capital", min_value=1.0, value=10000.0,
                         key="factor-initial-capital")
    fractional = memory.widget(st.checkbox, "Allow fractional shares", value=True, key="factor-fractional",
                               help="Purchases need a $5 budget including fees; closing a small position is allowed.")
    left, center, right = st.columns(3)
    fixed = memory.widget(left.number_input, "Fixed commission per trade", min_value=0.0, value=0.0,
                          key="factor-commission-fixed")
    bps = memory.widget(center.number_input, "Commission (basis points)", min_value=0.0, value=0.0,
                        key="factor-commission-bps")
    slippage = memory.widget(right.number_input, "Slippage (basis points)", min_value=0.0, max_value=9999.0,
                             value=0.0, key="factor-slippage-bps")
    return ExecutionSettings(initial_cash=cash, fractional_shares=fractional, commission_fixed=fixed,
                             commission_bps=bps, slippage_bps=slippage, minimum_trade_budget=5)


def _render_page(research, memory):
    st.title("Factor backtest")
    st.caption("Rank stocks, build target portfolios, and rebalance one shared account using local data. "
               "Configuration and your latest saved run survive navigation and restarts.")
    st.info("Prices are used as stored. A historical membership universe is not modeled; "
            "today's ticker list can bias multi-year results. Processed data can use adjusted-price research mode.")
    source = _source_controls(research, memory)
    if source:
        try:
            spec = _spec_controls(memory)
            settings = _execution_controls(memory)
        except (ValueError, TypeError) as error:
            st.error(str(error))
            spec = None
        if st.button("Validate data and run factor backtest", type="primary", key="factor-submit", disabled=spec is None):
            if not source["tickers"] or source["start"] >= source["end"]:
                st.error("Choose tickers and a valid date range.")
            else:
                try:
                    with st.spinner("Validating local data and simulating portfolio rebalances…"):
                        run = research.factor_backtest(**source, spec=spec, settings=settings)
                    st.session_state["factor-run"] = run
                    memory.values["last_run_id"] = run.run_id
                except PreflightError as error:
                    st.error("Factor backtest aborted: the required data did not pass preflight.")
                    if getattr(error, "report", None) is not None:
                        st.dataframe(error.report.to_frame(), hide_index=True)
                    else:
                        st.error(str(error))
                except (ResearchError, ValueError, TypeError) as error:
                    st.error(f"Factor backtest aborted: {error}")
    # Only the run ID is persisted; reload validated, versioned research artifacts.
    run_id = memory.values.get("last_run_id")
    run = st.session_state.get("factor-run")
    if run_id and (run is None or run.run_id != run_id):
        try:
            run = research.load_run(run_id)
            st.session_state["factor-run"] = run
        except (ResearchError, OSError, ValueError) as error:
            st.warning(f"The last saved factor run could not be opened: {error}")
            run = None
    if run is not None and run_id == run.run_id:
        st.divider()
        st.caption("Latest completed run. Its saved settings are shown below; edited controls apply to the next run.")
        show_factor_result(run)


def _audit_frame(ui, frame, title):
    ui.write(title)
    if frame.height:
        ui.dataframe(frame.head(5000), hide_index=True)
        if frame.height > 5000:
            ui.caption(f"Showing the first 5,000 of {frame.height:,} records. All records are retained in the saved run.")
    else:
        ui.caption("No records.")


def show_factor_result(run, *, ui=st):
    ui.subheader(f"Factor run {run.run_id}")
    show_research_mode(run.manifest, ui=ui)
    ui.caption("One shared long-only account. Return and drawdown values are fractions; 0.10 means 10%.")
    for field, title in (("ticker_errors", "Skipped ticker errors"), ("warnings", "Run warnings")):
        records = run.manifest.get(field, [])
        if records:
            ui.warning(title)
            ui.dataframe(pl.DataFrame(records, infer_schema_length=None), hide_index=True)
    if (run.manifest.get("research_mode", "strict") != "adjusted_price"
            and run.manifest.get("price_basis") == "processed_transformed_prices"):
        ui.info("Execution and account values use the transformed price units of the selected processing pipeline.")
    delayed = {symbol: stamp for symbol, stamp in run.manifest.get("effective_starts", {}).items()
               if stamp > run.manifest.get("start", stamp)}
    if delayed:
        with ui.expander("Tickers joining after warm-up"):
            ui.dataframe(pl.DataFrame({"ticker": list(delayed), "eligible_from": list(delayed.values())}),
                         hide_index=True)
    ui.dataframe(run.comparison, hide_index=True)
    if not run.results:
        return
    result = run.results[0]
    equity = result.equity
    if equity.height:
        curve = equity.with_columns(pl.col("valuation_time").alias("timestamp"))
        ui.plotly_chart(line_chart(curve, columns=["equity"], title="Account equity", timeframe="1h"),
                        config=CHART_CONFIG, width="stretch", key=f"factor-equity-{run.run_id}")
        drawdown = curve.select("timestamp", (pl.col("equity") / pl.col("equity").cum_max() - 1).alias("drawdown"))
        ui.plotly_chart(line_chart(drawdown, columns=["drawdown"], title="Drawdown", timeframe="1h",
                                  y_title="Fraction below prior peak"), config=CHART_CONFIG, width="stretch",
                        key=f"factor-drawdown-{run.run_id}")
    ui.subheader("Rebalance decisions")
    _audit_frame(ui, result.rebalances, "Decision summary")
    if result.rebalances.height and "rebalance_id" in result.rebalances.columns:
        ids = result.rebalances["rebalance_id"].unique(maintain_order=True).to_list()
        selected = ui.selectbox("Inspect rebalance", ids, key=f"factor-rebalance-{run.run_id}")
        for title, frame in (("Factor values and scores", result.factor_values), ("Proposed and accepted targets", result.targets),
                             ("Requested orders and outcomes", result.orders), ("Executed trades", result.trades),
                             ("Holdings at execution and subsequent closes", result.positions)):
            if "rebalance_id" in frame.columns:
                frame = frame.filter(pl.col("rebalance_id") == selected)
            _audit_frame(ui, frame, title)
    with ui.expander("All trades, orders, and factor records"):
        for title, frame in (("Executed trades", result.trades), ("Orders", result.orders),
                             ("Factor values", result.factor_values), ("Portfolio targets", result.targets),
                             ("Holdings", result.positions)):
            _audit_frame(ui, frame, title)
    with ui.expander("Account values and metrics"):
        ui.dataframe(equity, hide_index=True)
        ui.json(result.metrics)
    with ui.expander("Saved configuration and reproducibility manifest"):
        ui.json(run.manifest)
