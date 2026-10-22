"""Shared cash, next-open entries and fixed stop/target exits across tickers."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR, localcontext

import polars as pl

from data_pipeline.schemas import validate_ohlcv
from research.strategies.base import strategy_spec
from research.strategies.bracket import BracketStrategy

from .engine import TIMESTAMP, _bound, _display_number, quantity_fields, result_schemas
from .execution import MAX_SHARES, Fill, fee, rebalance
from .metrics import calculate_metrics
from .models import BacktestResult, ExecutionSettings, as_decimal


ENTRY_SCHEMA = {
    "record_id": pl.Int64, "symbol": pl.String,
    "signal_index": pl.Int64, "signal_timestamp": TIMESTAMP,
    "entry_index": pl.Int64, "entry_timestamp": TIMESTAMP,
    "entry_bar_timestamp": TIMESTAMP, "quantity": pl.Int64,
}
ENTRY_MONEY = ("entry_open", "entry_price", "stop_price", "win_price", "entry_budget", "entry_fees")
EXIT_MONEY = ("exit_price", "exit_fees", "fees", "gross_pnl", "net_pnl")
OPEN_MONEY = ("last_price", "market_value", "unrealized_pnl")


def _money_schema(names):
    return {key: dtype for name in names for key, dtype in
            ((name, pl.Float64), (name + "_exact", pl.String))}


COMPLETED_SCHEMA = {
    **ENTRY_SCHEMA, **_money_schema(ENTRY_MONEY), "outcome": pl.String,
    "exit_index": pl.Int64, "exit_timestamp": TIMESTAMP, "exit_bar_timestamp": TIMESTAMP,
    **_money_schema(EXIT_MONEY),
}
OPEN_SCHEMA = {**ENTRY_SCHEMA, **_money_schema(ENTRY_MONEY), **_money_schema(OPEN_MONEY)}
FILL_EXTRA = {"record_id": pl.Int64, "outcome": pl.String, **_money_schema(("entry_budget",))}


def _money(values):
    return {key: result for name, value in values.items() for key, result in
            ((name, _display_number(value)), (name + "_exact", str(value)))}


def _schedule(frames, strategy, start, end, bar_times, effective_starts):
    """Validate all inputs before execution and group opening/closing events."""
    data, rows, times, events, warmup = {}, {}, {}, defaultdict(list), {}
    if not frames:
        raise ValueError("A shared portfolio requires at least one ticker.")
    for symbol in sorted(frames):
        canonical = validate_ohlcv(frames[symbol])
        if canonical["symbol"].n_unique() != 1 or canonical["symbol"][0] != symbol:
            raise ValueError("Portfolio frame must contain exactly its keyed ticker.")
        data[symbol] = canonical
        rows[symbol] = list(canonical.iter_rows(named=True))
        times[symbol] = []
        for row in rows[symbol]:
            label = row["timestamp"]
            if bar_times is None:
                opened = closed = label
            else:
                try:
                    opened, closed = bar_times[symbol][label]
                except KeyError:
                    raise ValueError(f"Missing calendar times for {symbol}: {label.isoformat()}.") from None
                opened, closed = _bound(opened, "bar open"), _bound(closed, "bar close")
                if opened >= closed:
                    raise ValueError("Bar open must precede bar close.")
                if times[symbol] and times[symbol][-1][1] > opened:
                    raise ValueError("Calendar bars overlap: preceding bar is incomplete at the next open.")
            times[symbol].append((opened, closed))
        effective_start = _bound(effective_starts.get(symbol, start), "effective start")
        if not start <= effective_start < end:
            raise ValueError("Effective ticker start must lie within the backtest interval.")
        selected = [i for i, row in enumerate(rows[symbol]) if effective_start <= row["timestamp"] < end]
        if not selected:
            raise ValueError(f"No bars in the requested backtest interval for {symbol}.")
        warmup[symbol] = selected[0]
        if selected[0] < strategy.lookback:
            raise ValueError(f"Backtest requires {strategy.lookback} completed warm-up bars for {symbol}.")
        for index in selected:
            opened, closed = times[symbol][index]
            if not start <= opened < end or closed > end:
                raise ValueError("Backtest interval includes an incomplete or out-of-range execution bar.")
            # Real calendars: completed exits precede simultaneous next opens.
            # Without a calendar, each timestamp denotes a discrete whole bar.
            events[(opened, 1 if bar_times is not None else 0, "open")].append((symbol, index))
            events[(closed, 0 if bar_times is not None else 1, "close")].append((symbol, index))
    return data, rows, times, events, warmup


def validate_entry_percentage(value):
    value = as_decimal(value, "entry_percentage")
    if not 0 < value <= 100:
        raise ValueError("entry_percentage must be greater than 0 and at most 100.")
    return value


def run_portfolio_backtest(frames, strategy, *, start, end, settings=None, bar_times=None, effective_starts=None,
                           allocation_mode="per_entry", entry_percentage=10):
    """Run one shared account with equal free-cash or fixed initial-capital budgets.

    Thresholds use raw entry open; fills apply configured adverse slippage.
    Intrabar exits are recognized at bar close because OHLC cannot establish
    their exact time. Proceeds cannot finance earlier or same-bar open entries.
    """
    if allocation_mode not in ("per_entry", "fixed_tenth"):
        raise ValueError("Shared portfolios support per_entry or fixed_tenth allocation.")
    entry_percentage = validate_entry_percentage(entry_percentage)
    start, end = _bound(start, "start"), _bound(end, "end")
    if start >= end:
        raise ValueError("start must precede end.")
    if not isinstance(strategy, BracketStrategy):
        raise TypeError("Shared portfolio execution requires a BracketStrategy.")
    settings = settings if settings is not None else ExecutionSettings()
    if not isinstance(settings, ExecutionSettings):
        raise TypeError("settings must be ExecutionSettings.")
    equity_schema, trade_schema, order_schema = result_schemas(settings)
    detail_quantities = ({"quantity": pl.Float64, "quantity_exact": pl.String}
                         if settings.fractional_shares else {})
    spec = strategy_spec(strategy)
    stop = as_decimal(strategy.stop_percentage, "stop_percentage")
    win = as_decimal(strategy.win_percentage, "win_percentage")
    if not 0 < stop < 1 < win:
        raise ValueError("Require 0 < stop_percentage < 1 < win_percentage.")
    data, rows, times, events, warmup = _schedule(frames, strategy, start, end, bar_times, effective_starts or {})
    workers = {symbol: deepcopy(strategy) for symbol in data}
    prepared = {}
    for symbol, worker in workers.items():
        signals = worker.prepare_entry_signals(data[symbol].clone())
        if signals is not None:
            if not isinstance(signals, pl.Series) or signals.dtype != pl.Boolean or signals.null_count() or len(signals) != data[symbol].height:
                raise ValueError("Prepared entry signals must be one non-null boolean per source bar.")
            prepared[symbol] = signals.to_list()
        if strategy_spec(worker) != spec:
            raise ValueError("Strategy changed its declared configuration during signal preparation.")
    eligible_from = {symbol: times[symbol][warmup[symbol]][0] for symbol in data}
    cash, positions, prices = settings.initial_cash, {}, {}
    records, trades, orders, completed = [], [], [], []
    next_id = 1

    def held(position):
        return (Decimal(position["quantity_exact"]) if settings.fractional_shares
                else position["quantity"])

    def record_equity(stamp, phase):
        equity = cash + sum((held(position) * prices[symbol]
                             for symbol, position in positions.items()), Decimal(0))
        if cash < 0 or equity < 0:
            raise ArithmeticError("Negative cash or equity in shared portfolio.")
        holdings = sum((held(p) for p in positions.values()), Decimal(0) if settings.fractional_shares else 0)
        if holdings > MAX_SHARES:
            raise ValueError("Aggregate portfolio share count exceeds supported Int64 range.")
        records.append({"timestamp": stamp, "valuation_time": stamp, "phase": phase,
                        "cash": _display_number(cash), "cash_exact": str(cash),
                        "equity": _display_number(equity), "equity_exact": str(equity),
                        **quantity_fields(settings, holdings=holdings),
                        "position_count": len(positions), "price": None})

    def record_fill(fill, symbol, index, stamp, decision_index, record_id, budget, outcome=None):
        trade = {"timestamp": stamp, "decision_timestamp": times[symbol][decision_index][1],
                 "bar_timestamp": rows[symbol][index]["timestamp"],
                 "decision_bar_timestamp": rows[symbol][decision_index]["timestamp"],
                 "symbol": symbol, "side": fill.side,
                 **quantity_fields(settings, quantity=fill.quantity, holdings=fill.holdings), "target_weight": None,
                 "record_id": record_id, "outcome": outcome,
                 **_money({"price": fill.price, "fees": fill.fees, "cash": fill.cash,
                           "entry_budget": budget})}
        orders.append({**trade, **quantity_fields(settings, requested_quantity=fill.requested_quantity),
                       "status": fill.status, "reason": fill.reason})
        if fill.quantity:
            trades.append(trade)

    with localcontext() as context:
        context.prec = 800
        record_equity(start, "initial")
        for (stamp, _, phase), items in sorted(events.items()):
            if phase == "open":
                unheld = {symbol for symbol in data if eligible_from[symbol] <= stamp} - set(positions)
                # A single pre-entry snapshot makes simultaneous allocations
                # independent of ticker ordering. Round down, never borrow.
                with localcontext() as allocation_context:
                    allocation_context.rounding = ROUND_FLOOR
                    budget = (settings.initial_cash * entry_percentage / 100 if allocation_mode == "fixed_tenth" else
                              cash / len(unheld) if unheld else Decimal(0))
                for symbol, index in items:
                    row = rows[symbol][index]
                    prices[symbol] = Decimal(str(row["open"]))
                    if symbol not in unheld:
                        continue
                    worker = workers[symbol]
                    signal = (prepared[symbol][index - 1] if symbol in prepared else
                              worker.entry_signal(data[symbol].slice(0, index).clone()))
                    if type(signal) is not bool:
                        raise ValueError("Entry signal must be a boolean.")
                    if symbol not in prepared and strategy_spec(worker) != spec:
                        raise ValueError("Strategy changed its declared configuration during simulation.")
                    if not signal:
                        continue
                    entry_open = prices[symbol]
                    stop_price, win_price = entry_open * stop, entry_open * win
                    _display_number(stop_price)
                    _display_number(win_price)
                    if allocation_mode == "fixed_tenth" and cash < budget:
                        # Skip this signal, without queuing it or borrowing. At
                        # simultaneous opens the existing sorted ticker order
                        # determines which affordable entries execute first.
                        fill = Fill("buy", 0, 0, entry_open, Decimal(0), cash, 0,
                                    "rejected", "insufficient_cash_for_fixed_allocation")
                        record_fill(fill, symbol, index, stamp, index - 1, None, budget)
                        continue
                    fill = rebalance(budget, 0, entry_open, 1.0, settings)
                    if fill is None:
                        fill = Fill("buy", 0, 0, entry_open, Decimal(0), budget, 0,
                                    "rejected", "insufficient_cash")
                    if row["volume"] == 0:
                        fill = replace(fill, quantity=0, fees=Decimal(0), cash=budget,
                                       holdings=0, status="rejected", reason="no_reported_volume")
                    fill = replace(fill, cash=cash - (budget - fill.cash))
                    cash = fill.cash
                    record_fill(fill, symbol, index, stamp, index - 1, next_id if fill.quantity else None, budget)
                    if not fill.quantity:
                        continue
                    positions[symbol] = {
                        "record_id": next_id, "symbol": symbol, "signal_index": index - 1,
                        "signal_timestamp": rows[symbol][index - 1]["timestamp"],
                        "entry_index": index, "entry_timestamp": stamp,
                        "entry_bar_timestamp": row["timestamp"], **quantity_fields(settings, quantity=fill.quantity),
                        **_money({"entry_open": entry_open, "entry_price": fill.price,
                                  "stop_price": stop_price, "win_price": win_price,
                                  "entry_budget": budget, "entry_fees": fill.fees}),
                    }
                    next_id += 1
            else:
                exits = []
                for symbol, index in items:
                    row = rows[symbol][index]
                    prices[symbol] = Decimal(str(row["close"]))
                    position = positions.get(symbol)
                    if position is None:
                        continue
                    stop_price, win_price = (Decimal(position[name + "_exact"])
                                             for name in ("stop_price", "win_price"))
                    hit_win = Decimal(str(row["high"])) > win_price
                    hit_stop = Decimal(str(row["low"])) < stop_price
                    if not hit_win and not hit_stop:
                        continue
                    outcome = "inconclusive" if hit_win and hit_stop else "loss" if hit_stop else "win"
                    threshold = stop_price if hit_stop else win_price
                    price = threshold * (1 - settings.slippage_bps / Decimal(10000))
                    notional = held(position) * price
                    net_proceeds = notional - fee(notional, settings)
                    exits.append((symbol, index, position, outcome, threshold, net_proceeds))
                # Same-time sale proceeds form one funding pool. When that pool
                # cannot cover every cash-consuming exit, reject all such exits
                # rather than favoring whichever ticker happens to sort first.
                executable = [item for item in exits if rows[item[0]][item[1]]["volume"] > 0]
                sale_cash = cash + sum((item[5] for item in executable if item[5] >= 0), Decimal(0))
                exit_cost = -sum((item[5] for item in executable if item[5] < 0), Decimal(0))
                cover_exit_costs = sale_cash >= exit_cost
                for symbol, index, position, outcome, threshold, net_proceeds in sorted(
                    exits, key=lambda item: (item[5] < 0, item[0])
                ):
                    row = rows[symbol][index]
                    fill = rebalance(cash, held(position), threshold, 0.0, settings)
                    if row["volume"] == 0:
                        fill = replace(fill, quantity=0, fees=Decimal(0), cash=cash,
                                       holdings=held(position), status="rejected", reason="no_reported_volume")
                    elif net_proceeds < 0 and not cover_exit_costs:
                        fill = replace(fill, quantity=0, fees=Decimal(0), cash=cash,
                                       holdings=held(position), status="rejected",
                                       reason="sale_proceeds_cannot_cover_fees")
                    cash = fill.cash
                    record_fill(fill, symbol, index, stamp, position["signal_index"],
                                position["record_id"], Decimal(position["entry_budget_exact"]), outcome)
                    if not fill.quantity:
                        continue
                    gross = (fill.price - Decimal(position["entry_price_exact"])) * fill.quantity
                    fees = fill.fees + Decimal(position["entry_fees_exact"])
                    completed.append({**position, "outcome": outcome, "exit_index": index,
                                      "exit_timestamp": stamp, "exit_bar_timestamp": row["timestamp"],
                                      **_money({"exit_price": fill.price, "exit_fees": fill.fees,
                                                "fees": fees, "gross_pnl": gross, "net_pnl": gross - fees})})
                    del positions[symbol]
                record_equity(stamp, "close")

        open_records = []
        for symbol, position in sorted(positions.items()):
            market_value = held(position) * prices[symbol]
            unrealized = (market_value - held(position) * Decimal(position["entry_price_exact"])
                          - Decimal(position["entry_fees_exact"]))
            open_records.append({**position, **_money({"last_price": prices[symbol],
                                                      "market_value": market_value, "unrealized_pnl": unrealized})})
        equity_frame = pl.DataFrame(records, schema={**equity_schema, "position_count": pl.Int64})
        trade_frame = pl.DataFrame(trades, schema={**trade_schema, **FILL_EXTRA})
        metrics = calculate_metrics(equity_frame, trade_frame)
        realized = sum((Decimal(p["net_pnl_exact"]) for p in completed), Decimal(0))
        unrealized = sum((Decimal(p["unrealized_pnl_exact"]) for p in open_records), Decimal(0))
        total = Decimal(records[-1]["equity_exact"]) - settings.initial_cash
        if total != realized + unrealized:
            raise ArithmeticError("Portfolio profit does not reconcile to its cash ledger.")
        metrics.update(wins=sum(p["outcome"] == "win" for p in completed),
                       losses=sum(p["outcome"] == "loss" for p in completed),
                       inconclusive=sum(p["outcome"] == "inconclusive" for p in completed),
                       completed_trade_count=len(completed), open_positions=len(positions),
                       **_money({"realized_pnl": realized, "unrealized_pnl": unrealized, "total_pnl": total}))
    return BacktestResult(
        symbol="PORTFOLIO", strategy_spec=spec, settings=settings, equity=equity_frame,
        trades=trade_frame, orders=pl.DataFrame(orders, schema={**order_schema, **FILL_EXTRA}),
        metrics=metrics, completed_trades=pl.DataFrame(completed, schema={**COMPLETED_SCHEMA, **detail_quantities}),
        open_positions=pl.DataFrame(open_records, schema={**OPEN_SCHEMA, **detail_quantities}),
        metadata={"start": start.isoformat(), "end": end.isoformat(), "tickers": sorted(data),
                  "warmup_bars": warmup, "account_model": ("shared_cash_long_only_fractional_shares" if settings.fractional_shares
                                                         else "shared_cash_long_only_whole_shares"),
                  "share_precision": 8 if settings.fractional_shares else 0,
                  "effective_starts": {symbol: eligible_from[symbol].isoformat() for symbol in data},
                  "allocation_mode": allocation_mode,
                  "entry_percentage": format(entry_percentage.normalize(), "f") if allocation_mode == "fixed_tenth" else None,
                  "allocation": ("fixed_percentage_initial_capital_per_entry_skip_if_unfunded" if allocation_mode == "fixed_tenth"
                                 else "cash_divided_by_eligible_nonheld_tickers_at_open"),
                  "simultaneous_entry_priority": "ticker_ascending",
                  "execution": "next_bar_open_entry_strict_intrabar_threshold_exit",
                  "exit_price_model": "stop_or_target_threshold_with_adverse_slippage_including_gaps",
                  "exit_timestamp_semantics": "bar_close_recognition_intrabar_time_unknown",
                  "ambiguous_exit_policy": "inconclusive_stop_at_stop_price",
                  "simultaneous_exit_fees": "pool_positive_volume_sale_proceeds_then_all_or_none_cash_consuming_exits",
                  "threshold_basis": "raw_entry_open", "end_position_policy": "mark_to_market_at_last_close",
                  "dividends": "excluded", "liquidity_model": "unlimited_on_positive_volume_bars",
                  "cash_arithmetic": "Decimal_800_significant_digits", "settings": settings.to_dict()},
    )
