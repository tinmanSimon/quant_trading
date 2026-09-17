"""Causal, independent-account backtesting over verified local OHLCV bars."""

from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from collections.abc import Mapping
from dataclasses import replace
import math

import polars as pl

from data_pipeline.schemas import validate_ohlcv
from research.strategies.base import Strategy, strategy_spec, validate_weight

from .execution import rebalance
from .metrics import calculate_metrics
from .models import BacktestResult, ExecutionSettings

TIMESTAMP = pl.Datetime("us", "UTC")
EQUITY_SCHEMA = {"timestamp": TIMESTAMP, "valuation_time": TIMESTAMP, "phase": pl.String, "cash": pl.Float64, "holdings": pl.Int64, "price": pl.Float64, "equity": pl.Float64, "cash_exact": pl.String, "equity_exact": pl.String}
TRADE_SCHEMA = {"timestamp": TIMESTAMP, "decision_timestamp": TIMESTAMP, "bar_timestamp": TIMESTAMP, "decision_bar_timestamp": TIMESTAMP, "symbol": pl.String, "side": pl.String, "quantity": pl.Int64, "price": pl.Float64, "fees": pl.Float64, "cash": pl.Float64, "holdings": pl.Int64, "target_weight": pl.Float64, "price_exact": pl.String, "fees_exact": pl.String, "cash_exact": pl.String}
ORDER_SCHEMA = {**TRADE_SCHEMA, "requested_quantity": pl.Int64, "status": pl.String, "reason": pl.String}


def _bound(value, name):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime.")
    return value.astimezone(UTC)


def _display_number(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Simulation result exceeds the supported Float64 display range.")
    return result


def run_backtest(frame: pl.DataFrame, strategy: Strategy, *, start: datetime, end: datetime, settings: ExecutionSettings | None = None, bar_times: Mapping[datetime, tuple[datetime, datetime]] | None = None) -> BacktestResult:
    """Simulate a fresh strategy with no future-row access and no borrowing.

    Frames include warm-up observations. The caller must first verify session
    coverage and exclude unfinished bars. Bar timestamps label bar starts;
    With bar_times, trade timestamps use actual calendar opens/closes while
    bar_timestamp retains the source label. Without bar_times timestamps are
    labels and the caller is responsible for checking bar completion.
    No decision can execute on that same bar. Remaining holdings are
    marked at the final included close, not fictionally liquidated.
    """
    start, end = _bound(start, "start"), _bound(end, "end")
    if start >= end:
        raise ValueError("start must precede end.")
    if settings is None:
        settings = ExecutionSettings()
    if not isinstance(settings, ExecutionSettings):
        raise TypeError("settings must be ExecutionSettings.")
    spec = strategy_spec(strategy)
    worker = deepcopy(strategy)
    canonical = validate_ohlcv(frame)
    if canonical["symbol"].n_unique() != 1:
        raise ValueError("Each backtest account requires exactly one symbol.")
    symbol = canonical["symbol"][0]
    # Python datetime comparisons retain precise query boundaries. Never rely
    # on a Polars literal cast down to the millisecond storage dtype.
    rows = list(canonical.iter_rows(named=True))
    times = {}
    if bar_times is not None:
        for row in rows:
            label = row["timestamp"]
            if label not in bar_times:
                raise ValueError(f"Missing calendar times for bar {label.isoformat()}.")
            opened, closed = bar_times[label]
            opened, closed = _bound(opened, "bar open"), _bound(closed, "bar close")
            if opened >= closed:
                raise ValueError("Bar open must precede bar close.")
            times[label] = (opened, closed)
        for previous, current in zip(rows, rows[1:]):
            if times[previous["timestamp"]][1] > times[current["timestamp"]][0]:
                raise ValueError("Calendar bars overlap: previous bar is not complete at the next open.")
    selected = [index for index, row in enumerate(rows) if start <= row["timestamp"] < end]
    if not selected:
        raise ValueError("No bars in the requested backtest interval.")
    first, last = selected[0], selected[-1]
    if first < worker.lookback:
        raise ValueError(f"Backtest requires {worker.lookback} completed warm-up bars before its first execution bar.")
    records = []
    trades = []
    orders = []
    cash = settings.initial_cash
    holdings = 0
    records.append({"timestamp": start, "valuation_time": start, "phase": "initial", "cash": float(cash), "holdings": 0, "price": None, "equity": float(cash), "cash_exact": str(cash), "equity_exact": str(cash)})
    with localcontext() as context:
        # Float64 spans ~632 decimal orders of magnitude. This precision plus
        # bounded settings/Int64 positions prevents ledger subtraction from
        # rounding even at supported extreme input magnitudes.
        context.prec = 800
        for index in range(first, last + 1):
            row = rows[index]
            previous = rows[index - 1]
            fill_time = times[row["timestamp"]][0] if times else row["timestamp"]
            valuation_time = times[row["timestamp"]][1] if times else row["timestamp"]
            decision_time = times[previous["timestamp"]][1] if times else previous["timestamp"]
            if not start <= fill_time < end or valuation_time > end:
                raise ValueError("Backtest interval includes an incomplete or out-of-range execution bar.")
            # Clone ensures even an in-place mutation in user strategy code
            # cannot alter the source, another strategy, or execution prices.
            history = canonical.slice(0, index).clone()
            target = validate_weight(worker.target_weight(history))
            if strategy_spec(worker) != spec:
                raise ValueError("Strategy changed its declared configuration during simulation.")
            fill = rebalance(cash, holdings, Decimal(str(row["open"])), target, settings)
            if fill is not None:
                # Zero reported volume cannot support a simulated execution.
                # This is an execution-availability rule, never strategy input;
                # positive-volume fills still assume unlimited capacity.
                if row["volume"] == 0:
                    fill = replace(fill, quantity=0, fees=Decimal(0), cash=cash,
                                   holdings=holdings, status="rejected", reason="no_reported_volume")
                cash, holdings = fill.cash, fill.holdings
                trade = {"timestamp": fill_time, "decision_timestamp": decision_time, "bar_timestamp": row["timestamp"], "decision_bar_timestamp": previous["timestamp"], "symbol": symbol, "side": fill.side, "quantity": fill.quantity, "price": _display_number(fill.price), "fees": _display_number(fill.fees), "cash": _display_number(cash), "holdings": holdings, "target_weight": target, "price_exact": str(fill.price), "fees_exact": str(fill.fees), "cash_exact": str(cash)}
                orders.append({**trade, "requested_quantity": fill.requested_quantity, "status": fill.status, "reason": fill.reason})
                if fill.quantity:
                    trades.append(trade)
            price = Decimal(str(row["close"]))
            equity = cash + holdings * price
            if cash < 0 or holdings < 0 or equity < 0:
                raise ArithmeticError("Negative cash, holdings, or equity.")
            records.append({"timestamp": row["timestamp"], "valuation_time": valuation_time, "phase": "close", "cash": _display_number(cash), "holdings": holdings, "price": float(price), "equity": _display_number(equity), "cash_exact": str(cash), "equity_exact": str(equity)})
    equity_frame = pl.DataFrame(records, schema=EQUITY_SCHEMA)
    trade_frame = pl.DataFrame(trades, schema=TRADE_SCHEMA)
    return BacktestResult(
        symbol=symbol,
        strategy_spec=spec,
        settings=settings,
        equity=equity_frame,
        trades=trade_frame,
        orders=pl.DataFrame(orders, schema=ORDER_SCHEMA),
        metrics=calculate_metrics(equity_frame, trade_frame),
        metadata={"start": start.isoformat(), "end": end.isoformat(), "warmup_bars": first, "account_model": "independent_long_only_whole_shares", "execution": "next_bar_open", "decision_timestamp_semantics": "actual_calendar_bar_close" if times else "start_label_of_completed_decision_bar", "fees_rounding": "upward_to_0.00000001", "cash_arithmetic": "Decimal_800_significant_digits", "end_position_policy": "mark_to_market_at_last_close", "dividends": "excluded", "liquidity_model": "unlimited_at_slipped_open_on_positive_volume_bars", "settings": settings.to_dict()},
    )
