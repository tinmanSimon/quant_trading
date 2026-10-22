"""Portfolio-wide target sizing and jointly funded next-open fills.

Selection is already frozen. Only execution opens are used to size targets;
the account is marked once before any order is sized or filled.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext

from .execution import MAX_SHARES, SHARE_QUANTUM, fee

D = lambda value: Decimal(str(value))
ZERO = Decimal(0)


@dataclass
class RebalanceExecution:
    cash: Decimal
    holdings: dict
    orders: list[dict]
    equity_before: Decimal
    achieved_weights: dict
    turnover: Decimal


def execute_targets(cash, holdings, prices, target, *, settings, spec, scores=None, unavailable=()):
    """Return fills without mutating inputs; skipped targets expire this open.

    ``next_open`` funding restricts this event's buys to its starting cash.
    The cash ledger still records sales immediately; their proceeds become
    usable at the following event. Default pro-rata funding is order invariant.
    """
    with localcontext() as context:
        context.prec = 800
        return _execute(cash, holdings, prices, target, settings, spec, scores or {}, set(unavailable))


def _execute(cash, holdings, prices, target, settings, spec, scores, unavailable):
    cash = D(cash)
    holdings = {symbol: D(value) for symbol, value in holdings.items() if value}
    prices = {symbol: D(value) for symbol, value in prices.items()}
    if any(value <= 0 or not value.is_finite() for value in prices.values()):
        raise ValueError("Execution requires positive finite opening prices.")
    if any(symbol not in prices for symbol in holdings):
        raise ValueError("A held ticker has no current opening price; cannot value the portfolio.")
    if cash < 0 or any(value < 0 for value in holdings.values()):
        raise ValueError("Factor execution requires nonnegative cash and long-only holdings.")
    equity = cash + sum((quantity * prices[symbol] for symbol, quantity in holdings.items()), ZERO)
    if target.no_change:
        return RebalanceExecution(cash, holdings, [], equity,
                                  {symbol: float(value * prices[symbol] / equity) if equity else 0.0
                                   for symbol, value in holdings.items()}, ZERO)
    starting_cash = cash
    reserve = equity * D(spec.cash_reserve)
    quantum = SHARE_QUANTUM if settings.fractional_shares else Decimal(1)
    frozen = set(target.frozen_symbols)
    orders, buys, sells = [], {}, {}

    def quantity(value):
        result = (value / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum
        if result > MAX_SHARES:
            raise ValueError("Requested position exceeds the supported share count.")
        return result

    def record(symbol, side, requested, filled, price, fees, reason=""):
        orders.append({"symbol": symbol, "side": side, "requested_quantity": float(requested),
                       "requested_quantity_exact": str(requested), "quantity": float(filled),
                       "quantity_exact": str(filled), "price": float(price), "price_exact": str(price),
                       "fees": float(fees), "fees_exact": str(fees), "cash": float(cash),
                       "cash_exact": str(cash), "holdings": float(holdings.get(symbol, ZERO)),
                       "holdings_exact": str(holdings.get(symbol, ZERO)),
                       "target_weight": float(target.weights.get(symbol, 0)),
                       "status": "rejected" if filled == 0 else "filled" if filled == requested else "reduced",
                       "reason": reason})

    # All desired quantities use the same pre-trade total equity, including
    # frozen positions. An unchanged target is traded only by its delta.
    for symbol in sorted(set(target.weights) | set(holdings)):
        if symbol in frozen:
            continue
        weight = D(target.weights.get(symbol, 0))
        if weight < 0 or weight > 1 or not weight.is_finite():
            raise ValueError("Invalid long-only target weight.")
        if symbol not in prices:
            if weight:
                record(symbol, "buy", ZERO, ZERO, ZERO, ZERO, "missing_execution_bar")
            continue
        current = holdings.get(symbol, ZERO)
        desired = quantity(weight * equity / prices[symbol])
        delta = desired - current
        if delta == 0:
            continue
        side = "buy" if delta > 0 else "sell"
        requested = abs(delta)
        price = prices[symbol] * (1 + settings.slippage_bps / 10000 if delta > 0
                                  else 1 - settings.slippage_bps / 10000)
        if symbol in unavailable:
            record(symbol, side, requested, ZERO, price, ZERO, "no_reported_volume")
            continue
        if delta > 0:
            buys[symbol] = (requested, price)
            continue
        sells[symbol] = (requested, price)
    # Positive proceeds can pay for a small residual closure whose commission
    # exceeds its sale value. Tie-breaking is deterministic by symbol.
    for symbol in sorted(sells, key=lambda item: (
            -(sells[item][0] * sells[item][1] - fee(sells[item][0] * sells[item][1], settings)), item)):
        requested, price = sells[symbol]
        fees = fee(requested * price, settings)
        after_cash = cash + requested * price - fees
        if after_cash < 0:
            record(symbol, "sell", requested, ZERO, price, ZERO, "sale_proceeds_cannot_cover_fees")
            continue
        cash = after_cash
        holdings[symbol] -= requested
        record(symbol, "sell", requested, requested, price, fees)

    buying_power = max(ZERO, min(cash, starting_cash) - reserve) if spec.sale_proceeds == "next_open" else max(ZERO, cash - reserve)
    costs = {symbol: requested * price + fee(requested * price, settings)
             for symbol, (requested, price) in buys.items()}
    total_cost = sum(costs.values(), ZERO)
    budgets = ({symbol: min(cost, buying_power * cost / total_cost) for symbol, cost in costs.items()}
               if total_cost and spec.funding == "pro_rata" else {})
    order = sorted(buys, key=lambda symbol: (-scores.get(symbol, 0), symbol)) if spec.funding == "score_priority" else sorted(buys)
    for symbol in order:
        requested, price = buys[symbol]
        budget = min(buying_power, costs[symbol]) if spec.funding == "score_priority" else budgets[symbol]
        if budget < settings.minimum_trade_budget:
            record(symbol, "buy", requested, ZERO, price, ZERO, "below_minimum_trade_budget")
            continue
        low, high = 0, int(requested / quantum)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = mid * quantum
            if candidate * price + fee(candidate * price, settings) <= budget:
                low = mid
            else:
                high = mid - 1
        filled = low * quantum
        if not filled:
            record(symbol, "buy", requested, ZERO, price, ZERO, "insufficient_cash")
            continue
        fees = fee(filled * price, settings)
        spent = filled * price + fees
        cash -= spent
        buying_power -= spent
        holdings[symbol] = holdings.get(symbol, ZERO) + filled
        record(symbol, "buy", requested, filled, price, fees,
               "affordability_limit" if filled != requested else "")
    if cash < 0 or any(value < 0 for value in holdings.values()):
        raise ArithmeticError("Factor execution violated nonnegative cash/holdings.")
    marked = cash + sum((value * prices[symbol] for symbol, value in holdings.items()), ZERO)
    achieved = {symbol: float(holdings.get(symbol, ZERO) * prices.get(symbol, ZERO) / marked) if marked else 0.0
                for symbol in set(target.weights) | set(holdings)}
    traded = sum((D(order["quantity_exact"]) * D(order["price_exact"]) for order in orders), ZERO)
    return RebalanceExecution(cash, {symbol: value for symbol, value in holdings.items() if value},
                              orders, equity, achieved, traded / equity if equity else ZERO)
