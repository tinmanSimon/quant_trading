"""Independent entry/exit observations, without trading or portfolio simulation."""

import polars as pl

from research.analysis import analysis, crossed_above, crossed_below
from ..helpers.validation import require_ordered_periods


@analysis(
    "ma_rsi_example",
    label="Example: MA crossover and RSI",
    parameters={
        "fast": {"title": "Fast MA period", "minimum": 2, "maximum": 100_000},
        "slow": {"title": "Slow MA period", "minimum": 2, "maximum": 100_000},
        "rsi_period": {"minimum": 2, "maximum": 100_000},
        "entry_rsi_limit": {"minimum": 0, "maximum": 100},
        "exit_rsi_level": {"minimum": 0, "maximum": 100},
        "confirm_trend": {"description": "Require an upward trend observed at the entry bar. This adds trend-history calculations."},
        "trend_history": {"minimum": 8, "maximum": 100_000},
    },
)
def run(ctx, fast: int = 20, slow: int = 60, rsi_period: int = 14,
        entry_rsi_limit: float = 70, exit_rsi_level: float = 70,
        confirm_trend: bool = False, trend_history: int = 2000):
    """Find MA crossovers with an RSI condition and optional causal trend confirmation."""
    require_ordered_periods(fast, slow)
    data = ctx.features(fast_ma=ctx.sma(fast), slow_ma=ctx.sma(slow), rsi=ctx.rsi(rsi_period))

    # Compute shifts/crossovers on history BEFORE trimming the selected range.
    # Null indicator warmup values never create a signal.
    data = data.with_columns(
        entry=(crossed_above("fast_ma", "slow_ma") & (pl.col("rsi") < entry_rsi_limit)).fill_null(False),
        exit=crossed_below("fast_ma", "slow_ma") | crossed_above("rsi", exit_rsi_level),
    )
    if confirm_trend:
        # A historical trend becomes observable at its end_index. Its fitted
        # start must never be used as the time at which an entry was known.
        trends = ctx.trends(max_history=trend_history)
        observed_up = sorted({item.end_index for item in trends.trends if item.direction > 0})
        data = data.with_columns(entry=pl.col("entry") & pl.col("source_index").is_in(observed_up))

    selected = ctx.selected(data)
    columns = ["source_index", "timestamp", "close", "fast_ma", "slow_ma", "rsi"]
    entries = selected.filter("entry").select(columns).with_columns(
        reason=pl.lit("Fast MA crossed above slow MA; RSI below entry limit"
                      + ("; upward trend observed" if confirm_trend else "")))
    exits = selected.filter("exit").select(columns).with_columns(
        reason=pl.lit("Fast MA crossed below slow MA or RSI crossed above exit level"))

    result = ctx.result()
    result.text(f"**{entries.height} entry observations** and **{exits.height} exit observations** "
                f"across {selected.height} completed bars. "
                f"SMA {fast}/{slow}; RSI {rsi_period}. These are independent observations, not paired trades.")
    result.signals("Entries", entries, label="reason", color="#16a34a", symbol="triangle-up")
    result.signals("Exits", exits, label="reason", color="#dc2626", symbol="triangle-down")
    result.line(f"SMA {fast}", selected, value="fast_ma", color="#2563eb")
    result.line(f"SMA {slow}", selected, value="slow_ma", color="#f97316")
    result.line("RSI", selected, value="rsi", color="#8b5cf6",
                panel={"id": "rsi", "title": f"RSI ({rsi_period})", "range": [0, 100]})
    return result
