"""Validated array calculations backed by TA-Lib's default conventions."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import polars as pl
import talib

from .models import ALGORITHM_VERSION, BACKEND_VERSION, IndicatorResult


def _period(value: int, name: str, *, minimum: int = 2) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= 100_000:
        raise ValueError(f"{name} must be an integer between {minimum} and 100000")
    return int(value)


def _deviation(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 100_000:
        raise ValueError(f"{name} must be a finite number between 0 and 100000")
    return float(value)


def _inputs(frame: pl.DataFrame, columns: tuple[str, ...], *, unstable: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    if not isinstance(frame, pl.DataFrame):
        raise ValueError("Input must be a Polars DataFrame")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Price column must be a nonempty string")
    missing = set(("timestamp", *columns)) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    timestamps = frame.get_column("timestamp")
    if timestamps.dtype.base_type() not in (pl.Date, pl.Datetime):
        raise ValueError("timestamp must have Date or Datetime dtype")
    if timestamps.null_count() or not timestamps.is_sorted() or timestamps.n_unique() != frame.height:
        raise ValueError("timestamp must be non-null, increasing, and unique")
    for column in ("symbol", "ticker", "provider", "timeframe", "interval"):
        if column in frame.columns and frame.height:
            identity = frame.get_column(column)
            if identity.null_count() or identity.n_unique() != 1:
                raise ValueError(f"Indicator input requires a single non-null {column}")
    arrays = {}
    for column in dict.fromkeys(columns):
        values = frame.get_column(column)
        if not values.dtype.is_numeric() or values.null_count():
            raise ValueError(f"{column} must contain finite numeric values without nulls")
        array = np.ascontiguousarray(values.to_numpy(), dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{column} must contain finite numeric values without nulls")
        arrays[column] = array
    if "volume" in arrays and np.any(arrays["volume"] < 0):
        raise ValueError("volume must be nonnegative")
    if "high" in arrays and "low" in arrays and np.any(arrays["high"] < arrays["low"]):
        raise ValueError("high must be greater than or equal to low")
    # TA-Lib settings are process-global. Refuse externally changed conventions
    # instead of silently changing research results or mutating another caller's
    # settings; the package's cache identity describes the default conventions.
    if talib.get_compatibility() != 0 or any(talib.get_unstable_period(name) != 0 for name in unstable):
        raise ValueError("research.analysis.indicators requires TA-Lib default compatibility and zero unstable periods")
    return arrays


def _result(
    frame: pl.DataFrame,
    indicator: str,
    parameters: dict[str, Any],
    values: dict[str, np.ndarray],
    required_inputs: tuple[str, ...],
    units: dict[str, str],
    warmup: dict[str, int],
    initialization: str,
) -> IndicatorResult:
    for name, array in values.items():
        if np.isinf(array).any() or np.isnan(array[min(warmup[name], len(array)):]).any():
            raise ValueError(f"{indicator} produced non-finite {name} values; check input magnitudes")
    output = frame.select("timestamp").with_row_index("source_index").with_columns(
        [pl.Series(name, array, dtype=pl.Float64, nan_to_null=True) for name, array in values.items()]
    )
    timestamp = frame.get_column("timestamp")
    return IndicatorResult(output, {
        "indicator": indicator,
        "parameters": parameters,
        "required_inputs": ["timestamp", *dict.fromkeys(required_inputs)],
        "units": units,
        "warmup_bars": warmup,
        "initialization": initialization,
        "history_start": timestamp[0].isoformat() if len(timestamp) else None,
        "history_end": timestamp[-1].isoformat() if len(timestamp) else None,
        "input_rows": frame.height,
        "backend": "TA-Lib",
        "backend_version": BACKEND_VERSION,
        "backend_core_version": talib.__ta_version__.decode("ascii").split()[0],
        "algorithm_version": ALGORITHM_VERSION,
    })


def moving_average(frame: pl.DataFrame, *, kind: str = "sma", period: int = 20, price_col: str = "close") -> IndicatorResult:
    """SMA, SMA-seeded EMA, or linearly weighted MA (newest has most weight)."""
    if not isinstance(kind, str) or kind.lower() not in ("sma", "ema", "wma"):
        raise ValueError("kind must be 'sma', 'ema', or 'wma'")
    kind = kind.lower()
    period = _period(period, "period")
    columns = (price_col,)
    inputs = _inputs(frame, columns, unstable=("EMA",) if kind == "ema" else ())
    values = getattr(talib, kind.upper())(inputs[price_col], timeperiod=period)
    return _result(frame, "moving_average", {"kind": kind, "period": period, "price_col": price_col},
                   {"value": values}, columns, {"value": "price"}, {"value": period - 1},
                   "EMA starts with the first period-bar simple average; SMA/WMA use complete trailing periods.")


def rsi(frame: pl.DataFrame, *, period: int = 14, price_col: str = "close") -> IndicatorResult:
    """Wilder RSI, initialized with period price changes (period + 1 bars).

    TA-Lib returns zero for no gains and no losses, and 100 for gains with no
    losses. These values are retained without replacement or interpolation.
    """
    period = _period(period, "period")
    columns = (price_col,)
    inputs = _inputs(frame, columns, unstable=("RSI",))
    values = talib.RSI(inputs[price_col], timeperiod=period)
    return _result(frame, "rsi", {"period": period, "price_col": price_col}, {"value": values}, columns,
                   {"value": "percent"}, {"value": period},
                   "Wilder smoothing seeded from the first period price changes; flat prices produce zero.")


def macd(frame: pl.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "close") -> IndicatorResult:
    """TA-Lib MACD with EMA signal; all outputs share the signal warmup.

    Both price EMAs are seeded at the slow EMA's first endpoint, following
    TA-Lib's MACD convention. Consequently MACD need not equal a subtraction of
    two independently initialized full-series EMA calls during initialization.
    """
    fast, slow, signal = _period(fast, "fast"), _period(slow, "slow"), _period(signal, "signal", minimum=1)
    if fast >= slow:
        raise ValueError("fast must be less than slow")
    columns = (price_col,)
    inputs = _inputs(frame, columns, unstable=("EMA",))
    line, signal_line, histogram = talib.MACD(inputs[price_col], fastperiod=fast, slowperiod=slow, signalperiod=signal)
    values = {"macd": line, "signal": signal_line, "histogram": histogram}
    return _result(frame, "macd", {"fast": fast, "slow": slow, "signal": signal, "price_col": price_col},
                   values, columns, dict.fromkeys(values, "price"), dict.fromkeys(values, slow + signal - 2),
                   "TA-Lib SMA-seeded fast/slow EMAs aligned at the slow seed; SMA-seeded signal EMA.")


def atr(frame: pl.DataFrame, *, period: int = 14, normalized: bool = False) -> IndicatorResult:
    """Wilder ATR, optionally NATR (100 * ATR / close), in one forward pass."""
    period = _period(period, "period", minimum=1)
    if not isinstance(normalized, bool):
        raise ValueError("normalized must be a boolean")
    columns = ("high", "low", "close")
    inputs = _inputs(frame, columns, unstable=("ATR",))
    values = talib.ATR(inputs["high"], inputs["low"], inputs["close"], timeperiod=period)
    if normalized:
        # Normalize explicitly: TA-Lib's NATR period=1 fast path returns raw TR
        # rather than percent. Retain its zero-close convention after warmup.
        normalized_values = np.zeros_like(values)
        np.divide(values, inputs["close"], out=normalized_values, where=inputs["close"] != 0)
        normalized_values *= 100
        normalized_values[:period] = np.nan
        values = normalized_values
    return _result(frame, "atr", {"period": period, "normalized": normalized}, {"value": values}, columns,
                   {"value": "percent" if normalized else "price"}, {"value": period},
                   "Wilder smoothing seeded from period true ranges after the first bar; TA-Lib zero-close convention.")


def bollinger_bands(
    frame: pl.DataFrame, *, period: int = 20, upper_deviation: float = 2.0,
    lower_deviation: float = 2.0, price_col: str = "close",
) -> IndicatorResult:
    """SMA-centered bands using trailing population standard deviation."""
    period = _period(period, "period")
    upper_deviation, lower_deviation = _deviation(upper_deviation, "upper_deviation"), _deviation(lower_deviation, "lower_deviation")
    columns = (price_col,)
    inputs = _inputs(frame, columns)
    upper, middle, lower = talib.BBANDS(inputs[price_col], timeperiod=period, nbdevup=upper_deviation, nbdevdn=lower_deviation, matype=0)
    values = {"middle": middle, "upper": upper, "lower": lower}
    return _result(frame, "bollinger_bands", {"period": period, "upper_deviation": upper_deviation,
                   "lower_deviation": lower_deviation, "price_col": price_col}, values, columns,
                   dict.fromkeys(values, "price"), dict.fromkeys(values, period - 1),
                   "Complete trailing periods; SMA center and population standard deviation.")


def adx(frame: pl.DataFrame, *, period: int = 14) -> IndicatorResult:
    """Wilder ADX and +/- DI; DI becomes available before ADX."""
    period = _period(period, "period")
    columns = ("high", "low", "close")
    inputs = _inputs(frame, columns, unstable=("ADX", "PLUS_DI", "MINUS_DI"))
    args = (inputs["high"], inputs["low"], inputs["close"])
    values = {"adx": talib.ADX(*args, timeperiod=period), "plus_di": talib.PLUS_DI(*args, timeperiod=period),
              "minus_di": talib.MINUS_DI(*args, timeperiod=period)}
    return _result(frame, "adx", {"period": period}, values, columns, dict.fromkeys(values, "percent"),
                   {"adx": 2 * period - 1, "plus_di": period, "minus_di": period},
                   "TA-Lib Wilder directional-movement smoothing; ADX begins after averaging period DX values.")


def stochastic(frame: pl.DataFrame, *, period: int = 14, k_smoothing: int = 3, d_period: int = 3) -> IndicatorResult:
    """Slow stochastic: simple moving averages for both %K and %D."""
    period = _period(period, "period", minimum=1)
    k_smoothing, d_period = _period(k_smoothing, "k_smoothing", minimum=1), _period(d_period, "d_period", minimum=1)
    columns = ("high", "low", "close")
    inputs = _inputs(frame, columns)
    k, d = talib.STOCH(inputs["high"], inputs["low"], inputs["close"], fastk_period=period,
                       slowk_period=k_smoothing, slowk_matype=0, slowd_period=d_period, slowd_matype=0)
    values = {"k": k, "d": d}
    return _result(frame, "stochastic", {"period": period, "k_smoothing": k_smoothing, "d_period": d_period},
                   values, columns, dict.fromkeys(values, "percent"), dict.fromkeys(values, period + k_smoothing + d_period - 3),
                   "Slow %K and %D use simple averages and share %D's warmup; a zero price range produces zero.")


def obv(frame: pl.DataFrame, *, price_col: str = "close") -> IndicatorResult:
    """Cumulative signed volume, seeded with the first bar's volume."""
    columns = (price_col, "volume")
    inputs = _inputs(frame, columns)
    values = talib.OBV(inputs[price_col], inputs["volume"])
    return _result(frame, "obv", {"price_col": price_col}, {"value": values}, columns,
                   {"value": "volume"}, {"value": 0},
                   "Seeded with first volume; unchanged price adds zero, higher/lower prices add/subtract volume.")
