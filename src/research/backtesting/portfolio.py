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
from .execution import MAX_SHARES, SHARE_QUANTUM, Fill, fee, rebalance
from .metrics import calculate_metrics
from .models import BacktestResult, ExecutionSettings, as_decimal

DEFAULT_HOLDING_SLOTS = 25


def validate_holding_slots(value):
    if type(value) is not int or value < 1:
        raise ValueError("initial_available_holdings must be a positive integer.")
    return value


ENTRY_SCHEMA = {
    "record_id": pl.Int64, "symbol": pl.String, "direction": pl.String,
    "signal_index": pl.Int64, "signal_timestamp": TIMESTAMP,
    "entry_index": pl.Int64, "entry_timestamp": TIMESTAMP,
    "entry_bar_timestamp": TIMESTAMP, "quantity": pl.Int64,
}
ENTRY_MONEY = ("entry_open", "entry_price", "stop_price", "win_price", "entry_budget", "entry_fees",
               "reserved_margin", "short_proceeds")
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
FILL_EXTRA = {"record_id": pl.Int64, "outcome": pl.String, "direction": pl.String,
              "position_action": pl.String, **_money_schema(("entry_budget",))}


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


def _short_entry(budget, opened, settings):
    """Reserve one dollar of collateral per dollar of raw entry notional.

    Sale proceeds remain restricted until cover. Size against collateral plus
    actual rounded entry fees, with the same precision/minimum as long entries.
    """
    unit = SHARE_QUANTUM if settings.fractional_shares else Decimal(1)
    requested = (budget / opened).quantize(unit, rounding=ROUND_FLOOR)
    if requested > MAX_SHARES:
        raise ValueError("Requested share count exceeds supported Int64 range.")
    price = opened * (1 - settings.slippage_bps / 10000)
    reason = "insufficient_cash"
    low, high = 0, int(requested / unit)
    if settings.fractional_shares and budget < settings.minimum_trade_budget:
        high, reason = 0, "below_minimum_trade_budget"
    while low < high:
        mid = (low + high + 1) // 2
        quantity = mid * unit
        if quantity * opened + fee(quantity * price, settings) <= budget:
            low = mid
        else:
            high = mid - 1
    quantity = low * unit if settings.fractional_shares else low
    requested = requested if settings.fractional_shares else int(requested)
    fees = fee(quantity * price, settings) if quantity else Decimal(0)
    status = "rejected" if not quantity else "filled" if quantity == requested else "reduced"
    return Fill("sell", requested, quantity, price, fees, budget - quantity * opened - fees,
                -quantity, status, reason if not quantity else "affordability_limit" if status == "reduced" else "")


def run_portfolio_backtest(frames, strategy, *, start, end, settings=None, bar_times=None, effective_starts=None,
                           allocation_mode="per_entry", entry_percentage=10,
                           initial_available_holdings=DEFAULT_HOLDING_SLOTS):
    """Run one shared account with free-cash, fixed-capital or holding-slot budgets.

    Thresholds use raw entry open; fills apply configured adverse slippage.
    Intrabar exits are recognized at bar close because OHLC cannot establish
    their exact time. Proceeds cannot finance earlier or same-bar open entries.
    """
    if allocation_mode not in ("per_entry", "fixed_tenth", "holding_slots"):
        raise ValueError("Shared portfolios support per_entry, fixed_tenth or holding_slots allocation.")
    initial_available_holdings = validate_holding_slots(initial_available_holdings)
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
    prepared, prepared_shorts = {}, {}
    for symbol, worker in workers.items():
        signals = worker.prepare_entry_signals(data[symbol].clone())
        if signals is not None:
            if not isinstance(signals, pl.Series) or signals.dtype != pl.Boolean or signals.null_count() or len(signals) != data[symbol].height:
                raise ValueError("Prepared entry signals must be one non-null boolean per source bar.")
            prepared[symbol] = signals.to_list()
        short_signals = worker.prepare_short_entry_signals(data[symbol].clone())
        if short_signals is None and type(worker).short_entry_signal is BracketStrategy.short_entry_signal:
            # Existing long-only strategies need no per-bar history copy just
            # to evaluate the base class's unconditional False.
            short_signals = pl.Series([False] * data[symbol].height, dtype=pl.Boolean)
        if short_signals is not None:
            if not isinstance(short_signals, pl.Series) or short_signals.dtype != pl.Boolean or short_signals.null_count() or len(short_signals) != data[symbol].height:
                raise ValueError("Prepared short entry signals must be one non-null boolean per source bar.")
            prepared_shorts[symbol] = short_signals.to_list()
        if strategy_spec(worker) != spec:
            raise ValueError("Strategy changed its declared configuration during signal preparation.")
    eligible_from = {symbol: times[symbol][warmup[symbol]][0] for symbol in data}
    cash, positions, prices = settings.initial_cash, {}, {}
    records, trades, orders, completed = [], [], [], []
    next_id = 1
    has_shorts = bool(getattr(strategy, "enable_shorts", False))

    def held(position):
        return (Decimal(position["quantity_exact"]) if settings.fractional_shares
                else position["quantity"])

    def record_equity(stamp, phase):
        margin = sum((Decimal(p["reserved_margin_exact"]) for p in positions.values()), Decimal(0))
        proceeds = sum((Decimal(p["short_proceeds_exact"]) for p in positions.values()), Decimal(0))
        equity = cash + margin + proceeds + sum((held(position) * prices[symbol]
                             * (-1 if position["direction"] == "short" else 1)
                             for symbol, position in positions.items()), Decimal(0))
        if not has_shorts and (cash < 0 or equity < 0):
            raise ArithmeticError("Negative cash or equity in shared portfolio.")
        gross_holdings = sum((held(p) for p in positions.values()), Decimal(0) if settings.fractional_shares else 0)
        holdings = sum((held(p) * (-1 if p["direction"] == "short" else 1) for p in positions.values()),
                       Decimal(0) if settings.fractional_shares else 0)
        if gross_holdings > MAX_SHARES:
            raise ValueError("Aggregate portfolio share count exceeds supported Int64 range.")
        records.append({"timestamp": stamp, "valuation_time": stamp, "phase": phase,
                        "cash": _display_number(cash), "cash_exact": str(cash),
                        "equity": _display_number(equity), "equity_exact": str(equity),
                        **_money({"reserved_margin": margin, "short_proceeds": proceeds}),
                        **quantity_fields(settings, holdings=holdings),
                        "position_count": len(positions), "price": None,
                        **({"available_holdings": initial_available_holdings - len(positions)}
                           if allocation_mode == "holding_slots" else {})})

    def record_fill(fill, symbol, index, stamp, decision_index, record_id, budget, outcome=None,
                    direction="long", position_action="entry"):
        trade = {"timestamp": stamp, "decision_timestamp": times[symbol][decision_index][1],
                 "bar_timestamp": rows[symbol][index]["timestamp"],
                 "decision_bar_timestamp": rows[symbol][decision_index]["timestamp"],
                 "symbol": symbol, "side": fill.side,
                 **quantity_fields(settings, quantity=fill.quantity, holdings=fill.holdings), "target_weight": None,
                 "record_id": record_id, "outcome": outcome,
                 "direction": direction, "position_action": position_action,
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
                # per_entry uses a single pre-entry snapshot. holding_slots
                # instead recomputes its budget at each signal below.
                with localcontext() as allocation_context:
                    allocation_context.rounding = ROUND_FLOOR
                    budget = (settings.initial_cash * entry_percentage / 100 if allocation_mode == "fixed_tenth" else
                              max(cash, Decimal(0)) / len(unheld) if unheld else Decimal(0))
                for symbol, index in items:
                    row = rows[symbol][index]
                    prices[symbol] = Decimal(str(row["open"]))
                    if symbol not in unheld:
                        continue
                    worker = workers[symbol]
                    signal = (prepared[symbol][index - 1] if symbol in prepared else
                              worker.entry_signal(data[symbol].slice(0, index).clone()))
                    short_signal = (prepared_shorts[symbol][index - 1] if symbol in prepared_shorts else
                                    worker.short_entry_signal(data[symbol].slice(0, index).clone()))
                    if type(signal) is not bool or type(short_signal) is not bool:
                        raise ValueError("Entry signal must be a boolean.")
                    if (symbol not in prepared or symbol not in prepared_shorts) and strategy_spec(worker) != spec:
                        raise ValueError("Strategy changed its declared configuration during simulation.")
                    if signal and short_signal:
                        raise ValueError("Long and short entry signals cannot both be true on one bar.")
                    if not signal and not short_signal:
                        continue
                    direction = "short" if short_signal else "long"
                    has_shorts = has_shorts or short_signal
                    entry_open = prices[symbol]
                    if allocation_mode == "holding_slots":
                        # Derive the counter from filled positions: failed entries
                        # and failed exits never consume or release a slot.
                        available_holdings = initial_available_holdings - len(positions)
                        if available_holdings == 0:
                            fill = Fill("sell" if short_signal else "buy", 0, 0, entry_open,
                                        Decimal(0), cash, 0, "rejected", "no_available_holdings")
                            record_fill(fill, symbol, index, stamp, index - 1, None, Decimal(0), direction=direction)
                            continue
                        with localcontext() as allocation_context:
                            allocation_context.rounding = ROUND_FLOOR
                            budget = max(cash, Decimal(0)) / available_holdings
                    if short_signal:
                        # Bound repeating reciprocals so later cash arithmetic
                        # remains exact within the ledger's 800-digit context.
                        with localcontext() as price_context:
                            price_context.prec = 28
                            stop_price, win_price = entry_open / stop, entry_open / win
                    else:
                        stop_price, win_price = entry_open * stop, entry_open * win
                    _display_number(stop_price)
                    _display_number(win_price)
                    if cash <= 0 or (allocation_mode == "fixed_tenth" and cash < budget):
                        # Skip this signal, without queuing it or borrowing. At
                        # simultaneous opens the existing sorted ticker order
                        # determines which affordable entries execute first.
                        fill = Fill("sell" if short_signal else "buy", 0, 0, entry_open, Decimal(0), cash, 0,
                                    "rejected", "insufficient_cash_for_fixed_allocation" if allocation_mode == "fixed_tenth" else "insufficient_cash")
                        record_fill(fill, symbol, index, stamp, index - 1, None, budget, direction=direction)
                        continue
                    fill = _short_entry(budget, entry_open, settings) if short_signal else rebalance(budget, 0, entry_open, 1.0, settings)
                    if fill is None:
                        fill = Fill("buy", 0, 0, entry_open, Decimal(0), budget, 0,
                                    "rejected", "insufficient_cash")
                    if row["volume"] == 0:
                        fill = replace(fill, quantity=0, fees=Decimal(0), cash=budget,
                                       holdings=0, status="rejected", reason="no_reported_volume")
                    fill = replace(fill, cash=cash - (budget - fill.cash))
                    cash = fill.cash
                    record_fill(fill, symbol, index, stamp, index - 1, next_id if fill.quantity else None, budget, direction=direction)
                    if not fill.quantity:
                        continue
                    positions[symbol] = {
                        "record_id": next_id, "symbol": symbol, "direction": direction, "signal_index": index - 1,
                        "signal_timestamp": rows[symbol][index - 1]["timestamp"],
                        "entry_index": index, "entry_timestamp": stamp,
                        "entry_bar_timestamp": row["timestamp"], **quantity_fields(settings, quantity=fill.quantity),
                        **_money({"entry_open": entry_open, "entry_price": fill.price,
                                  "stop_price": stop_price, "win_price": win_price,
                                  "entry_budget": budget, "entry_fees": fill.fees,
                                  "reserved_margin": fill.quantity * entry_open if short_signal else Decimal(0),
                                  "short_proceeds": fill.quantity * fill.price if short_signal else Decimal(0)}),
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
                    short = position["direction"] == "short"
                    hit_win = Decimal(str(row["low"])) < win_price if short else Decimal(str(row["high"])) > win_price
                    hit_stop = Decimal(str(row["high"])) > stop_price if short else Decimal(str(row["low"])) < stop_price
                    if not hit_win and not hit_stop:
                        continue
                    outcome = "inconclusive" if hit_win and hit_stop else "loss" if hit_stop else "win"
                    threshold = stop_price if hit_stop else win_price
                    price = threshold * (1 + (1 if short else -1) * settings.slippage_bps / Decimal(10000))
                    notional = held(position) * price
                    net_proceeds = ((Decimal(position["reserved_margin_exact"]) + Decimal(position["short_proceeds_exact"])
                                     - notional - fee(notional, settings)) if short else notional - fee(notional, settings))
                    exits.append((symbol, index, position, outcome, threshold, net_proceeds))
                # Same-time exits form one funding pool. Short covers are
                # mandatory even at a deficit. Cash-consuming long sales keep
                # the existing all-or-none fee policy within the funding pool.
                executable = [item for item in exits if rows[item[0]][item[1]]["volume"] > 0]
                sale_cash = cash + sum((item[5] for item in executable if item[5] >= 0), Decimal(0))
                exit_cost = -sum((item[5] for item in executable if item[5] < 0), Decimal(0))
                cover_exit_costs = sale_cash >= exit_cost
                for symbol, index, position, outcome, threshold, net_proceeds in sorted(
                    exits, key=lambda item: (item[5] < 0, item[0])
                ):
                    row = rows[symbol][index]
                    short = position["direction"] == "short"
                    price = threshold * (1 + (1 if short else -1) * settings.slippage_bps / 10000)
                    quantity = held(position)
                    fill = Fill("buy" if short else "sell", quantity, quantity, price,
                                fee(quantity * price, settings), cash + net_proceeds, 0, "filled", "")
                    if row["volume"] == 0:
                        fill = replace(fill, quantity=0, fees=Decimal(0), cash=cash,
                                       holdings=-quantity if short else quantity, status="rejected", reason="no_reported_volume")
                    elif not short and net_proceeds < 0 and not cover_exit_costs:
                        fill = replace(fill, quantity=0, fees=Decimal(0), cash=cash,
                                       holdings=held(position), status="rejected",
                                       reason="sale_proceeds_cannot_cover_fees")
                    cash = fill.cash
                    record_fill(fill, symbol, index, stamp, position["signal_index"],
                                position["record_id"], Decimal(position["entry_budget_exact"]), outcome,
                                direction=position["direction"], position_action="exit")
                    if not fill.quantity:
                        continue
                    gross = (fill.price - Decimal(position["entry_price_exact"])) * fill.quantity * (-1 if short else 1)
                    fees = fill.fees + Decimal(position["entry_fees_exact"])
                    completed.append({**position, "outcome": outcome, "exit_index": index,
                                      "exit_timestamp": stamp, "exit_bar_timestamp": row["timestamp"],
                                      **_money({"exit_price": fill.price, "exit_fees": fill.fees,
                                                "fees": fees, "gross_pnl": gross, "net_pnl": gross - fees})})
                    del positions[symbol]
                record_equity(stamp, "close")

        open_records = []
        for symbol, position in sorted(positions.items()):
            sign = -1 if position["direction"] == "short" else 1
            market_value = sign * held(position) * prices[symbol]
            unrealized = (market_value - sign * held(position) * Decimal(position["entry_price_exact"])
                          - Decimal(position["entry_fees_exact"]))
            open_records.append({**position, **_money({"last_price": prices[symbol],
                                                      "market_value": market_value, "unrealized_pnl": unrealized})})
        equity_frame = pl.DataFrame(records, schema={**equity_schema, "position_count": pl.Int64,
                                                    **({"available_holdings": pl.Int64} if allocation_mode == "holding_slots" else {}),
                                                    **_money_schema(("reserved_margin", "short_proceeds"))})
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
        if has_shorts:
            metrics.update(long_trades=sum(p["direction"] == "long" for p in completed),
                           short_trades=sum(p["direction"] == "short" for p in completed),
                           open_long_positions=sum(p["direction"] == "long" for p in positions.values()),
                           open_short_positions=sum(p["direction"] == "short" for p in positions.values()),
                           cash_deficit=max(0.0, -_display_number(cash)))
    return BacktestResult(
        symbol="PORTFOLIO", strategy_spec=spec, settings=settings, equity=equity_frame,
        trades=trade_frame, orders=pl.DataFrame(orders, schema={**order_schema, **FILL_EXTRA}),
        metrics=metrics, completed_trades=pl.DataFrame(completed, schema={**COMPLETED_SCHEMA, **detail_quantities}),
        open_positions=pl.DataFrame(open_records, schema={**OPEN_SCHEMA, **detail_quantities}),
        metadata={"start": start.isoformat(), "end": end.isoformat(), "tickers": sorted(data),
                  "warmup_bars": warmup, "account_model": ("shared_cash_long_short" if has_shorts else "shared_cash_long_only")
                                                        + ("_fractional_shares" if settings.fractional_shares else "_whole_shares"),
                  "short_accounting": "reserve_raw_entry_notional_and_lock_sale_proceeds_until_cover",
                  "short_borrow_costs": "not_modeled", "short_borrow_availability": "assumed",
                  "margin_calls": "not_modeled", "short_cover_deficit": "honor_full_exit_and_block_unfunded_entries",
                  "short_threshold_significant_digits": 28,
                  "share_precision": 8 if settings.fractional_shares else 0,
                  "effective_starts": {symbol: eligible_from[symbol].isoformat() for symbol in data},
                  "allocation_mode": allocation_mode,
                  **({"initial_available_holdings": initial_available_holdings,
                      "available_holdings": initial_available_holdings - len(positions)} if allocation_mode == "holding_slots" else {}),
                  "entry_percentage": format(entry_percentage.normalize(), "f") if allocation_mode == "fixed_tenth" else None,
                  "allocation": ("fixed_percentage_initial_capital_per_entry_skip_if_unfunded" if allocation_mode == "fixed_tenth"
                                 else "current_cash_divided_by_available_holdings" if allocation_mode == "holding_slots"
                                 else "cash_divided_by_eligible_nonheld_tickers_at_open"),
                  "simultaneous_entry_priority": "ticker_ascending",
                  "execution": "next_bar_open_entry_strict_intrabar_threshold_exit",
                  "exit_price_model": "stop_or_target_threshold_with_adverse_slippage_including_gaps",
                  "exit_timestamp_semantics": "bar_close_recognition_intrabar_time_unknown",
                  "ambiguous_exit_policy": "inconclusive_stop_at_stop_price",
                  "simultaneous_exit_fees": ("pool_exits_mandatory_short_covers_then_all_or_none_cash_consuming_long_sales" if has_shorts
                                             else "pool_positive_volume_sale_proceeds_then_all_or_none_cash_consuming_exits"),
                  "threshold_basis": "raw_entry_open", "end_position_policy": "mark_to_market_at_last_close",
                  "dividends": "excluded", "liquidity_model": "unlimited_on_positive_volume_bars",
                  "cash_arithmetic": "Decimal_800_significant_digits", "settings": settings.to_dict()},
    )
