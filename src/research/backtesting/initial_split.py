"""Independent ticker accounts funded once from one initial capital budget."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR, localcontext
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
from threading import Lock

from research.strategies.bracket import BracketStrategy
from research.strategies.loader import load_registry

from .engine import run_backtest
from .models import ExecutionSettings
from .portfolio import run_portfolio_backtest


MAX_WORKERS = 32
_SPAWN_LOCK = Lock()


def validate_allocation_options(allocation_mode, max_workers):
    if allocation_mode not in ("per_entry", "initial_split", "fixed_tenth"):
        raise ValueError("allocation_mode must be 'per_entry', 'initial_split', or 'fixed_tenth'.")
    if type(max_workers) is not int or not 1 <= max_workers <= MAX_WORKERS:
        raise ValueError(f"max_workers must be an integer from 1 to {MAX_WORKERS}.")


def split_initial_cash(settings, ticker_count):
    """Retain the sub-eight-decimal remainder as idle portfolio cash."""
    if not ticker_count:
        raise ValueError("Initial capital splitting requires at least one ticker.")
    with localcontext() as context:
        context.prec = 800
        allocation = (settings.initial_cash / ticker_count).quantize(
            Decimal("0.00000001"), rounding=ROUND_FLOOR,
        )
        remainder = settings.initial_cash - allocation * ticker_count
    if allocation <= 0:
        raise ValueError("Initial cash is too small to split across the selected tickers.")
    return allocation, remainder


def _private_roots(strategies):
    # Private classes use generated module names. Import the same trusted local
    # packages before unpickling them in a fresh, spawned process.
    roots = set()
    pending = list(strategies)
    seen = set()
    while pending:
        strategy = pending.pop()
        if id(strategy) in seen:
            continue
        seen.add(id(strategy))
        pending.extend(getattr(strategy, "strategies", ()))
        name = type(strategy).__module__.split(".", 1)[0]
        module = sys.modules.get(name)
        if name.startswith("_quant_private_"):
            for package_path in getattr(module, "__path__", ()):
                roots.add(str(Path(package_path).resolve().parent))
    return tuple(sorted(roots))


def _initialize_worker(private_roots):
    for root in private_roots:
        load_registry(project_dir=root)


def _run_ticker(task):
    symbol, frame, strategy, strategy_index, start, end, settings, times, total, remainder = task
    if isinstance(strategy, BracketStrategy):
        result = run_portfolio_backtest(
            {symbol: frame}, strategy, start=start, end=end, settings=settings,
            bar_times=None if times is None else {symbol: times},
        )
        result.symbol = symbol
    else:
        result = run_backtest(frame, strategy, start=start, end=end, settings=settings, bar_times=times)
    short_enabled = result.metadata.get("account_model", "").startswith("shared_cash_long_short")
    result.metadata.update(
        account_model="initial_split_long_short" if short_enabled else "initial_split_long_only",
        allocation_mode="initial_split",
        allocation="equal_initial_capital_per_ticker_no_transfers", strategy_index=strategy_index,
        portfolio_initial_cash_exact=str(total), unallocated_cash_exact=str(remainder),
    )
    return result


def _run_pickled_ticker(payload):
    # Bytes are created locally in the parent, never loaded from user artifacts.
    return _run_ticker(pickle.loads(payload))


def run_initial_split_backtests(frames, strategies, *, start, end, settings=None,
                               bar_times=None, effective_starts=None, max_workers=1):
    """Run each ticker with an equal fixed budget, optionally in spawned processes.

    Results always follow sorted ticker order then strategy input order. Profits
    and idle cash stay in their ticker's account for the entire simulation.
    """
    validate_allocation_options("initial_split", max_workers)
    settings = settings if settings is not None else ExecutionSettings()
    allocation, remainder = split_initial_cash(settings, len(frames))
    ticker_settings = replace(settings, initial_cash=allocation)
    tasks = [
        (symbol, frames[symbol], strategy, strategy_index,
         (effective_starts or {}).get(symbol, start), end, ticker_settings,
         None if bar_times is None else bar_times[symbol], settings.initial_cash, remainder)
        for symbol in sorted(frames) for strategy_index, strategy in enumerate(strategies)
    ]
    if max_workers == 1 or len(tasks) < 2:
        return [_run_ticker(task) for task in tasks]
    try:
        # Fail before any simulations start if a custom local class cannot be
        # transferred. Serialize the larger bar frames only as workers need them.
        for strategy in strategies:
            pickle.dumps(strategy, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, AttributeError, TypeError) as exc:
        raise ValueError("Parallel backtests require importable strategy classes. "
                         "Use max_workers=1 for a locally defined strategy.") from exc
    worker_count = min(max_workers, len(tasks))
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker, initargs=(_private_roots(strategies),),
    ) as pool:
        # Spawn must inherit this before Polars is imported. Bound each process
        # to one Polars thread rather than creating a machine-wide pool in every
        # worker. Restore the caller's environment immediately after spawning.
        def submit(task):
            payload = pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
            with _SPAWN_LOCK:
                previous = os.environ.get("POLARS_MAX_THREADS")
                os.environ["POLARS_MAX_THREADS"] = "1"
                try:
                    return pool.submit(_run_pickled_ticker, payload)
                finally:
                    if previous is None:
                        os.environ.pop("POLARS_MAX_THREADS", None)
                    else:
                        os.environ["POLARS_MAX_THREADS"] = previous

        remaining = iter(enumerate(tasks))
        pending = {submit(task): index for index, task in
                   (next(remaining) for _ in range(min(len(tasks), worker_count * 2)))}
        results = [None] * len(tasks)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                results[pending.pop(future)] = future.result()
                next_task = next(remaining, None)
                if next_task is not None:
                    index, task = next_task
                    pending[submit(task)] = index
        return results


def summarize_initial_split(results):
    """Aggregate independent slices of the same capital, separately per strategy."""
    grouped = {}
    for result in results:
        if result.metadata.get("allocation_mode") != "initial_split":
            continue
        key = result.metadata["strategy_index"]
        grouped.setdefault(key, []).append(result)
    summaries = []
    with localcontext() as context:
        context.prec = 800
        for strategy_index, group in sorted(grouped.items()):
            first = group[0]
            initial = Decimal(first.metadata["portfolio_initial_cash_exact"])
            reserve = Decimal(first.metadata["unallocated_cash_exact"])
            final = reserve + sum((Decimal(item.metrics["final_equity_exact"]) for item in group), Decimal(0))
            summary = {
                "strategy_index": strategy_index, "strategy": first.strategy_spec["name"],
                "config": json.dumps(first.strategy_spec["config"], sort_keys=True),
                "tickers": len(group), "initial_cash": float(initial),
                "final_equity": float(final), "total_pnl": float(final - initial),
                "total_pnl_exact": str(final - initial), "total_return": float(final / initial - 1),
                "unallocated_cash_exact": str(reserve),
            }
            for name in ("wins", "losses", "inconclusive", "completed_trade_count", "open_positions"):
                if all(name in item.metrics for item in group):
                    summary[name] = sum(item.metrics[name] for item in group)
            for name in ("realized_pnl", "unrealized_pnl"):
                if all(name + "_exact" in item.metrics for item in group):
                    amount = sum((Decimal(item.metrics[name + "_exact"]) for item in group), Decimal(0))
                    summary[name], summary[name + "_exact"] = float(amount), str(amount)
            summaries.append(summary)
    return summaries
