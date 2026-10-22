"""Causal factor decisions and a chronological, shared-cash portfolio ledger."""

from collections import defaultdict
from datetime import datetime
from decimal import Decimal, localcontext

import polars as pl

from data_pipeline.schemas import validate_ohlcv
from research.factors import FactorBacktestSpec, build_factor_panel, score_factor_panel
from research.portfolio import PortfolioRegistry, build_targets, should_rebalance, validate_schedule

from .engine import TIMESTAMP, _bound, _display_number
from .factor_execution import execute_targets
from .metrics import calculate_metrics
from .models import BacktestResult, ExecutionSettings

ZERO = Decimal(0)
EQUITY_SCHEMA = {"timestamp": TIMESTAMP, "valuation_time": TIMESTAMP, "phase": pl.String,
                 "cash": pl.Float64, "cash_exact": pl.String, "holdings": pl.Float64,
                 "holdings_exact": pl.String, "equity": pl.Float64, "equity_exact": pl.String}
ORDER_SCHEMA = {"rebalance_id": pl.String, "timestamp": TIMESTAMP, "decision_timestamp": TIMESTAMP,
                "bar_timestamp": TIMESTAMP, "decision_bar_timestamp": TIMESTAMP,
                "symbol": pl.String, "side": pl.String,
                **{key: dtype for name in ("requested_quantity", "quantity", "price", "fees", "cash", "holdings")
                   for key, dtype in ((name, pl.Float64), (name + "_exact", pl.String))},
                "target_weight": pl.Float64, "status": pl.String, "reason": pl.String}
REBALANCE_SCHEMA = {"rebalance_id": pl.String, "decision_timestamp": TIMESTAMP,
                    "decision_bar_timestamp": TIMESTAMP, "execution_timestamp": TIMESTAMP,
                    "status": pl.String, "reason": pl.String, "cash_before": pl.Float64,
                    "cash_after": pl.Float64, "equity_before": pl.Float64, "turnover": pl.Float64,
                    "fees": pl.Float64, "selected_count": pl.Int64}
TARGET_SCHEMA = {"rebalance_id": pl.String, "symbol": pl.String, "score": pl.Float64,
                 "decision_weight": pl.Float64, "proposed_weight": pl.Float64,
                 "target_weight": pl.Float64, "execution_target_weight": pl.Float64,
                 "achieved_weight": pl.Float64, "frozen": pl.Boolean, "reason": pl.String}
POSITION_SCHEMA = {"rebalance_id": pl.String, "timestamp": TIMESTAMP, "valuation_time": TIMESTAMP,
                   "phase": pl.String, "symbol": pl.String, "quantity": pl.Float64,
                   "quantity_exact": pl.String, "price": pl.Float64, "value": pl.Float64,
                   "value_exact": pl.String, "weight": pl.Float64}


class _FixedTarget:
    """Recheck constraints at the fill open without rerunning stock selection."""
    def __init__(self, target):
        self.target = target

    def propose(self, scores, snapshot):
        return self.target


def _timeline(frames, bar_times):
    if not frames:
        raise ValueError("Factor backtesting requires at least one ticker.")
    if bar_times is None:
        raise ValueError("Factor backtesting requires explicit calendar bar_times.")
    canonical, by_label, times, observations = {}, defaultdict(dict), {}, {}
    for symbol, frame in sorted(frames.items()):
        canonical[symbol] = checked = validate_ohlcv(frame)
        if checked.is_empty() or set(checked["symbol"]) != {symbol}:
            raise ValueError(f"Factor frame must contain exactly its declared symbol: {symbol}.")
        previous_close = None
        for index, row in enumerate(checked.iter_rows(named=True)):
            label = row["timestamp"]
            if symbol not in bar_times or label not in bar_times[symbol]:
                raise ValueError(f"Missing calendar times for {symbol} at {label.isoformat()}.")
            opened, closed = (_bound(value, "calendar boundary") for value in bar_times[symbol][label])
            if opened >= closed or (previous_close is not None and previous_close > opened):
                raise ValueError("Calendar bars must have positive duration and must not overlap.")
            if label in times and times[label] != (opened, closed):
                raise ValueError("Factor portfolios require a common session calendar and bar grid.")
            previous_close = closed
            times[label] = opened, closed
            by_label[label][symbol] = row
            observations[symbol, label] = index + 1
    labels = sorted(times)
    if any(times[a][1] > times[b][0] for a, b in zip(labels, labels[1:])):
        raise ValueError("Factor portfolios require one non-overlapping session grid.")
    return canonical, by_label, times, observations, labels


def run_factor_backtest(frames, spec, *, start: datetime, end: datetime, settings=None,
                        bar_times=None, effective_starts=None, factor_registry=None,
                        portfolio_registry=None, session_timezone="America/New_York"):
    """Select from completed bars, fill once at the next open, then mark closes.

    The caller verifies complete data coverage. Registries are trusted Python
    extensions: custom factors must be causal, as the built-ins are. No policy
    receives a future price or the eventual execution bar's high/low/close.
    """
    start, end = _bound(start, "start"), _bound(end, "end")
    if start >= end:
        raise ValueError("start must precede end.")
    if not isinstance(spec, FactorBacktestSpec):
        raise TypeError("spec must be a FactorBacktestSpec.")
    settings = ExecutionSettings(fractional_shares=True) if settings is None else settings
    if not isinstance(settings, ExecutionSettings):
        raise TypeError("settings must be ExecutionSettings.")
    validate_schedule(spec.schedule)
    registry = PortfolioRegistry() if portfolio_registry is None else portfolio_registry
    policy = registry.load(spec.portfolio)
    canonical, by_label, times, observations, labels = _timeline(frames, bar_times)
    effective = {symbol: _bound(value, "effective start") for symbol, value in (effective_starts or {}).items()}
    # If every requested listing begins late, its early observations are still
    # warm-up, even though their labels fall inside the requested test period.
    first_execution = min((max(start, effective.get(symbol, start)) for symbol in canonical), default=start)
    selected = [index for index, label in enumerate(labels) if first_execution <= label < end]
    if not selected:
        raise ValueError("No bars in the requested factor backtest interval.")
    if selected[0] == 0:
        raise ValueError("Factor backtesting requires completed warm-up bars before execution.")
    for index in selected:
        opened, closed = times[labels[index]]
        if not start <= opened < end or closed > end:
            raise ValueError("Factor interval includes an incomplete or out-of-range execution bar.")
    panel = build_factor_panel(canonical, spec, factor_registry)
    # Partition once; avoid scanning a multi-million-row panel at every event.
    panels = {key[0]: value for key, value in panel.partition_by("timestamp", as_dict=True).items()}
    factor_schema = {**panel.schema, "rebalance_id": pl.String, "decision_timestamp": TIMESTAMP}
    equity_rows, orders, rebalance_rows, target_rows, position_rows, factor_parts = [], [], [], [], [], []
    cash, holdings, last_rebalance = settings.initial_cash, {}, None

    def mark(prices):
        if set(holdings) - set(prices):
            raise ValueError("A held ticker is missing its current valuation bar.")
        return cash + sum((quantity * Decimal(str(prices[symbol])) for symbol, quantity in holdings.items()), ZERO)

    def snapshot(prices):
        equity = mark(prices)
        return {symbol: float(quantity * Decimal(str(prices[symbol])) / equity) if equity else 0.0
                for symbol, quantity in holdings.items()}

    def record_positions(label, at, phase, prices, rebalance_id):
        equity = mark(prices)
        for symbol, quantity in sorted(holdings.items()):
            price = Decimal(str(prices[symbol]))
            value = quantity * price
            position_rows.append({"rebalance_id": rebalance_id, "timestamp": label, "valuation_time": at,
                                  "phase": phase, "symbol": symbol, "quantity": _display_number(quantity),
                                  "quantity_exact": str(quantity), "price": _display_number(price),
                                  "value": _display_number(value), "value_exact": str(value),
                                  "weight": float(value / equity) if equity else 0.0})

    def decision(label, execution_label, event_index):
        closed_at = times[label][1]
        following_close = times[execution_label][1] if execution_label is not None else None
        # A truncated series does not prove a day/week/month has ended.
        if following_close is None and spec.schedule.name != "every_n_bars":
            return None
        if not should_rebalance(spec.schedule, index=event_index, closed_at=closed_at,
                                next_closed_at=following_close, session_timezone=session_timezone):
            return None
        rows = panels[label]
        eligible = [symbol for symbol in rows["symbol"]
                    if observations[symbol, label] >= spec.lookback
                    and (execution_label is None or effective.get(symbol, start) <= execution_label)]
        scores = score_factor_panel(rows.filter(pl.col("symbol").is_in(eligible)), spec)
        weights = snapshot({symbol: row["close"] for symbol, row in by_label[label].items()})
        target = build_targets(scores, weights, spec, registry, policy=policy)
        rebalance_id = f"rebalance-{len(rebalance_rows) + 1:06d}"
        factor_parts.append(scores.with_columns(pl.lit(rebalance_id).alias("rebalance_id"),
                                               pl.lit(closed_at, dtype=TIMESTAMP).alias("decision_timestamp")))
        score_map = {row["symbol"]: row["score"] for row in scores.iter_rows(named=True) if row["score"] is not None}
        audit = {"rebalance_id": rebalance_id, "decision_timestamp": closed_at,
                 "decision_bar_timestamp": label, "execution_timestamp": None,
                 "status": "no_change" if target.no_change else "expired",
                 "reason": "policy requested no change" if target.no_change else "no next execution bar in interval",
                 "cash_before": float(cash), "cash_after": float(cash), "equity_before": None,
                 "turnover": 0.0, "fees": 0.0,
                 "selected_count": sum(weight > 0 for weight in target.weights.values())}
        rebalance_rows.append(audit)
        audits = []
        for symbol in sorted(set(target.weights) | set(weights)):
            item = {"rebalance_id": rebalance_id, "symbol": symbol, "score": score_map.get(symbol),
                    "decision_weight": weights.get(symbol, 0.0),
                    "proposed_weight": target.proposed_weights.get(symbol, 0.0),
                    "target_weight": target.weights.get(symbol, 0.0), "execution_target_weight": None,
                    "achieved_weight": weights.get(symbol, 0.0) if target.no_change else None,
                    "frozen": symbol in target.frozen_symbols, "reason": target.reasons.get(symbol, "")}
            target_rows.append(item)
            audits.append(item)
        return target, scores, score_map, audit, audits

    with localcontext() as context:
        context.prec = 800
        equity_rows.append({"timestamp": start, "valuation_time": start, "phase": "initial",
                            "cash": float(cash), "cash_exact": str(cash), "holdings": 0.0,
                            "holdings_exact": "0", "equity": float(cash), "equity_exact": str(cash)})
        for event_index, index in enumerate(selected):
            label, previous = labels[index], labels[index - 1]
            opened, closed = times[label]
            opening_prices = {symbol: row["open"] for symbol, row in by_label[label].items()}
            pending = decision(previous, label, event_index)
            if pending is not None:
                target, scores, score_map, audit, audits = pending
                last_rebalance = audit["rebalance_id"]
                if not target.no_change:
                    # Keep selection fixed, but enforce risk limits against the
                    # common opening snapshot after overnight price gaps.
                    execution_target = build_targets(scores, snapshot(opening_prices), spec, registry,
                                                     policy=_FixedTarget(target))
                    result = execute_targets(cash, holdings, opening_prices, execution_target,
                                             settings=settings, spec=spec, scores=score_map,
                                             unavailable=[s for s, row in by_label[label].items() if row["volume"] == 0])
                    cash, holdings = result.cash, result.holdings
                    for order in result.orders:
                        orders.append({**order, "rebalance_id": last_rebalance, "timestamp": opened,
                                       "decision_timestamp": times[previous][1], "bar_timestamp": label,
                                       "decision_bar_timestamp": previous})
                    audit.update(execution_timestamp=opened, status="executed", reason="",
                                 cash_after=float(cash), equity_before=float(result.equity_before),
                                 turnover=float(result.turnover), fees=sum(order["fees"] for order in result.orders))
                    for item in audits:
                        symbol = item["symbol"]
                        item["execution_target_weight"] = execution_target.weights.get(symbol, 0.0)
                        item["achieved_weight"] = result.achieved_weights.get(symbol, 0.0)
                        if abs(item["execution_target_weight"] - item["target_weight"]) > 1e-12:
                            item["reason"] += "; opening risk check: " + execution_target.reasons.get(symbol, "")
                    record_positions(label, opened, "rebalance", opening_prices, last_rebalance)
            closing_prices = {symbol: row["close"] for symbol, row in by_label[label].items()}
            equity = mark(closing_prices)
            quantity = sum(holdings.values(), ZERO)
            equity_rows.append({"timestamp": label, "valuation_time": closed, "phase": "close",
                                "cash": _display_number(cash), "cash_exact": str(cash),
                                "holdings": _display_number(quantity), "holdings_exact": str(quantity),
                                "equity": _display_number(equity), "equity_exact": str(equity)})
            record_positions(label, closed, "close", closing_prices, last_rebalance)
        # Record a scheduled final decision without inventing a future fill.
        decision(labels[selected[-1]], None, len(selected))
    equity_frame = pl.DataFrame(equity_rows, schema=EQUITY_SCHEMA)
    order_frame = pl.DataFrame(orders, schema=ORDER_SCHEMA)
    trades = order_frame.filter(pl.col("quantity") > 0)
    positions = pl.DataFrame(position_rows, schema=POSITION_SCHEMA)
    metrics = calculate_metrics(equity_frame, trades)
    metrics.update(rebalance_count=sum(row["status"] == "executed" for row in rebalance_rows),
                   gross_turnover=sum(row["turnover"] for row in rebalance_rows),
                   final_cash=float(cash), open_position_count=len(holdings))
    final_positions = positions.filter((pl.col("timestamp") == labels[selected[-1]]) & (pl.col("phase") == "close"))
    return BacktestResult(
        symbol="PORTFOLIO", strategy_spec={"name": "factor_portfolio", "version": "1", "config": spec.to_dict()},
        settings=settings, equity=equity_frame, trades=trades, orders=order_frame, metrics=metrics,
        metadata={"flow": "factor", "account_model": "shared_cash_long_only_factor",
                  "start": start.isoformat(), "end": end.isoformat(), "settings": settings.to_dict(),
                  "execution": "next_bar_open", "decision_timestamp_semantics": "actual_calendar_bar_close",
                  "cash_arithmetic": "Decimal_800_significant_digits", "sale_proceeds": spec.sale_proceeds,
                  "funding": spec.funding, "unfilled_orders": "expire_at_intended_open",
                  "end_position_policy": "mark_to_market_at_last_close", "dividends": "excluded",
                  "liquidity_model": "unlimited_at_slipped_open_on_positive_volume_bars"},
        open_positions=final_positions, factor_values=pl.concat(factor_parts) if factor_parts else pl.DataFrame(schema=factor_schema),
        rebalances=pl.DataFrame(rebalance_rows, schema=REBALANCE_SCHEMA),
        targets=pl.DataFrame(target_rows, schema=TARGET_SCHEMA), positions=positions,
    )
