"""Independent numerical references and the dashboard/research input contract."""

from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

import market_indicators as indicators


def bars(prices, *, volume=None):
    close = np.asarray(prices, dtype=float)
    return pl.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(len(close))],
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": np.arange(1, len(close) + 1, dtype=float) if volume is None else volume,
    }, schema_overrides={"timestamp": pl.Datetime("us", "UTC")})


def rolling_mean(values, period):
    return np.array([np.nan if i < period - 1 else np.mean(values[i-period+1:i+1]) for i in range(len(values))])


def reference_ema(values, period, start=None):
    start = period - 1 if start is None else start
    result = np.full(len(values), np.nan)
    if start >= len(values):
        return result
    result[start] = np.mean(values[start-period+1:start+1])
    for i in range(start + 1, len(values)):
        result[i] = result[i-1] + 2 / (period + 1) * (values[i] - result[i-1])
    return result


def assert_values(actual, expected):
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-11, atol=1e-10, equal_nan=True)


ALL_FUNCTIONS = [indicators.moving_average, indicators.rsi, indicators.macd, indicators.atr,
                 indicators.bollinger_bands, indicators.adx, indicators.stochastic, indicators.obv]


@pytest.mark.parametrize("kind", ["sma", "ema", "wma"])
def test_moving_averages_against_independent_formula(kind):
    prices = np.array([2, 5, 1, 7, 8, 3, 9, 6], dtype=float)
    expected = rolling_mean(prices, 3)
    if kind == "ema":
        expected = reference_ema(prices, 3)
    elif kind == "wma":
        expected = np.array([np.nan, np.nan, *[np.dot(prices[i-2:i+1], [1, 2, 3]) / 6 for i in range(2, len(prices))]])
    result = indicators.moving_average(bars(prices), kind=kind, period=3)
    assert_values(result.frame["value"], expected)
    assert result.metadata["warmup_bars"] == {"value": 2}


def test_rsi_against_wilder_reference():
    prices = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                       46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
    changes = np.diff(prices)
    gain = sum(max(change, 0) for change in changes[:14]) / 14
    loss = sum(max(-change, 0) for change in changes[:14]) / 14
    expected = [np.nan] * 14 + [100 * gain / (gain + loss)]
    for change in changes[14:]:
        gain = (13 * gain + max(change, 0)) / 14
        loss = (13 * loss + max(-change, 0)) / 14
        expected.append(100 * gain / (gain + loss))
    result = indicators.rsi(bars(prices))
    assert result.frame["value"][14] == pytest.approx(70.4641350211)
    assert_values(result.frame["value"], expected)


@pytest.mark.parametrize("signal", [1, 3])
def test_macd_against_aligned_seed_ema_formula(signal):
    prices = np.array([3, 8, 4, 9, 7, 5, 12, 7, 8, 13, 9, 14, 8, 11], dtype=float)
    line = reference_ema(prices, 2, start=4) - reference_ema(prices, 5)
    signal_values = np.concatenate([np.full(4, np.nan), reference_ema(line[4:], signal)])
    line[:4 + signal - 1] = np.nan
    result = indicators.macd(bars(prices), fast=2, slow=5, signal=signal)
    assert_values(result.frame["macd"], line)
    assert_values(result.frame["signal"], signal_values)
    assert_values(result.frame["histogram"], line - signal_values)


@pytest.mark.parametrize("period", [1, 3])
@pytest.mark.parametrize("normalized", [False, True])
def test_atr_against_true_range_and_wilder_formula(period, normalized):
    frame = bars([11, 12, 13, 15, 14])
    ranges = np.array([2., 2., 3., 2.])
    expected = np.full(frame.height, np.nan)
    expected[period] = np.mean(ranges[:period])
    for i in range(period + 1, frame.height):
        expected[i] = (expected[i-1] * (period - 1) + ranges[i-1]) / period
    if normalized:
        expected = 100 * expected / frame["close"].to_numpy()
    result = indicators.atr(frame, period=period, normalized=normalized)
    assert_values(result.frame["value"], expected)


def test_normalized_atr_zero_close_is_zero_after_warmup():
    result = indicators.atr(bars([0, 2, 0, 3, 0]), period=2, normalized=True)
    assert result.frame["value"].to_list() == [None, None, 0, pytest.approx(116.6666666667), 0]


def test_bands_against_population_deviation_with_asymmetric_multipliers():
    prices = np.array([1, 4, 2, 7, 6, 3], dtype=float)
    mean = rolling_mean(prices, 3)
    deviation = np.array([np.nan, np.nan, *[np.std(prices[i-2:i+1], ddof=0) for i in range(2, len(prices))]])
    result = indicators.bollinger_bands(bars(prices), period=3, upper_deviation=1.5, lower_deviation=2.5)
    assert_values(result.frame["middle"], mean)
    assert_values(result.frame["upper"], mean + 1.5 * deviation)
    assert_values(result.frame["lower"], mean - 2.5 * deviation)


def test_adx_directional_values_on_a_linear_trend():
    result = indicators.adx(bars(np.arange(1, 11)), period=3)
    assert result.frame["adx"].to_list() == [None] * 5 + [100.] * 5
    assert_values(result.frame["plus_di"], [np.nan] * 3 + [50.] * 7)
    assert_values(result.frame["minus_di"], [np.nan] * 3 + [0.] * 7)


def test_stochastic_against_rolling_extrema_and_simple_smoothing():
    frame = bars([2, 5, 1, 7, 8, 3, 9, 6])
    fast_k = np.full(frame.height, np.nan)
    for i in range(2, frame.height):
        low = min(frame["low"][i-2:i+1])
        high = max(frame["high"][i-2:i+1])
        fast_k[i] = 100 * (frame["close"][i] - low) / (high - low)
    k = rolling_mean(fast_k, 2)
    d = rolling_mean(k, 2)
    k[:4] = np.nan
    result = indicators.stochastic(frame, period=3, k_smoothing=2, d_period=2)
    assert_values(result.frame["k"], k)
    assert_values(result.frame["d"], d)


def test_obv_accumulation_and_first_volume_seed():
    result = indicators.obv(bars([2, 5, 5, 1, 3], volume=[10, 20, 30, 40, 50]))
    assert result.frame["value"].to_list() == [10, 30, 30, -10, 40]


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_future_bars_do_not_change_existing_results(function):
    random = np.random.default_rng(71)
    frame = bars(100 + np.cumsum(random.normal(size=130)))
    prefix = function(frame.head(80)).frame
    assert_frame_equal(prefix, function(frame).frame.head(80))


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_schema_alignment_metadata_input_immutability_and_insufficient_history(function):
    frame = bars([10., 11.])
    original = frame.clone()
    result = function(frame)
    assert_frame_equal(frame, original)
    assert result.frame.height == frame.height
    assert result.frame["timestamp"].equals(frame["timestamp"])
    assert result.frame["source_index"].to_list() == [0, 1]
    assert all(dtype == pl.Float64 for name, dtype in result.frame.schema.items() if name not in ("timestamp", "source_index"))
    assert result.metadata["backend_version"] == indicators.BACKEND_VERSION
    assert result.metadata["algorithm_version"] == indicators.ALGORITHM_VERSION
    assert result.metadata["history_start"] == frame["timestamp"][0].isoformat()
    json.dumps(result.metadata, allow_nan=False)
    if function is not indicators.obv:
        assert all(result.frame[name].null_count() == 2 for name in result.metadata["units"])


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_empty_frame_has_typed_results_and_no_initialization_values(function):
    result = function(bars([]))
    assert result.frame.height == 0
    assert result.metadata["history_start"] is None
    assert result.metadata["history_end"] is None
    assert result.metadata["input_rows"] == 0


@pytest.mark.parametrize("function", [indicators.rsi, indicators.adx, indicators.stochastic])
def test_flat_zero_range_convention(function):
    frame = bars([10.] * 60).with_columns(pl.lit(10.).alias("high"), pl.lit(10.).alias("low"))
    result = function(frame)
    for column in result.metadata["units"]:
        assert set(result.frame[column].drop_nulls()) == {0.}


def test_price_source_and_date_timestamps_are_supported():
    frame = bars([1, 2, 3, 4]).with_columns(
        pl.Series("timestamp", [date(2025, 1, i) for i in range(1, 5)]),
        pl.Series("adjusted_close", [10., 20., 30., 40.]),
    )
    result = indicators.moving_average(frame, period=2, price_col="adjusted_close")
    assert result.frame["value"].to_list() == [None, 15, 25, 35]
    assert result.frame["timestamp"].dtype == pl.Date


def test_only_required_columns_are_validated_and_volume_is_optional():
    frame = bars(range(50)).drop("volume")
    indicators.rsi(frame)
    indicators.atr(frame)
    with pytest.raises(ValueError, match="volume"):
        indicators.obv(frame)
    indicators.rsi(frame.with_columns(pl.lit(float("nan")).alias("high")))
    indicators.rsi(frame.with_columns(pl.lit(-1.).alias("volume")))


@pytest.mark.parametrize("column", ["symbol", "ticker", "provider", "timeframe", "interval"])
@pytest.mark.parametrize("values", [["first", "second", "first"], ["first", None, "first"], [None] * 3])
def test_mixed_or_null_optional_series_identity_is_rejected(column, values):
    frame = bars([1, 2, 3]).with_columns(pl.Series(column, values, dtype=pl.String))
    with pytest.raises(ValueError, match=f"single non-null {column}"):
        indicators.rsi(frame)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_single_series_metadata_is_accepted_but_mixed_symbols_are_not(function):
    frame = bars(range(50))
    identified = frame.with_columns(
        pl.lit("TEST").alias("symbol"), pl.lit("TEST").alias("ticker"),
        pl.lit("local").alias("provider"), pl.lit("1h").alias("timeframe"),
        pl.lit("1h").alias("interval"),
    )
    assert_frame_equal(function(frame).frame, function(identified).frame)
    assert function(identified.head(0)).frame.is_empty()
    mixed = identified.with_columns(pl.Series("symbol", ["FIRST"] * 25 + ["SECOND"] * 25))
    with pytest.raises(ValueError, match="single non-null symbol"):
        function(mixed)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_or_missing_required_price_is_rejected(bad):
    frame = bars([1, 2, 3]).with_columns(pl.Series("close", [1., bad, 3.], dtype=pl.Float64))
    with pytest.raises(ValueError, match="close.*finite"):
        indicators.rsi(frame)


def test_bad_timestamp_numeric_columns_and_bar_bounds_are_rejected():
    frame = bars([1, 2, 3])
    for altered in [frame.reverse(), frame.with_columns(pl.lit(frame["timestamp"][0]).alias("timestamp")),
                    frame.with_columns(pl.lit(None, dtype=pl.Datetime).alias("timestamp"))]:
        with pytest.raises(ValueError, match="timestamp"):
            indicators.rsi(altered)
    with pytest.raises(ValueError, match="Date or Datetime"):
        indicators.rsi(frame.with_columns(pl.col("timestamp").cast(pl.String)))
    with pytest.raises(ValueError, match="numeric"):
        indicators.rsi(frame.with_columns(pl.col("close").cast(pl.String)))
    with pytest.raises(ValueError, match="nonnegative"):
        indicators.obv(frame.with_columns(pl.lit(-1.).alias("volume")))
    with pytest.raises(ValueError, match="high"):
        indicators.atr(frame.with_columns(pl.lit(-10.).alias("high")))


@pytest.mark.parametrize("function,kwargs", [
    (indicators.moving_average, {"kind": "invalid"}), (indicators.moving_average, {"period": True}),
    (indicators.rsi, {"period": 1}), (indicators.rsi, {"period": 100_001}),
    (indicators.rsi, {"period": 2.5}), (indicators.rsi, {"price_col": None}),
    (indicators.macd, {"fast": 26, "slow": 12}), (indicators.macd, {"signal": 0}),
    (indicators.atr, {"normalized": "yes"}), (indicators.atr, {"period": 0}),
    (indicators.bollinger_bands, {"upper_deviation": float("inf")}),
    (indicators.bollinger_bands, {"lower_deviation": -1}),
    (indicators.adx, {"period": 1}), (indicators.stochastic, {"k_smoothing": 0}),
])
def test_parameter_validation(function, kwargs):
    with pytest.raises(ValueError):
        function(bars(range(50)), **kwargs)


def test_external_backend_setting_changes_are_not_silently_used(monkeypatch):
    import talib
    monkeypatch.setattr(talib, "get_unstable_period", lambda name: 5)
    with pytest.raises(ValueError, match="default compatibility"):
        indicators.rsi(bars(range(50)))


def test_all_indicators_handle_100000_bars_in_a_single_batch():
    # Avoid fragile wall-clock assertions: the batch path and usable outputs at
    # this size are regression coverage; sustained timing is measured separately.
    frame = bars(100 + np.sin(np.arange(100_000) / 25) + np.arange(100_000) / 100_000)
    for function in ALL_FUNCTIONS:
        result = function(frame)
        assert result.frame.height == 100_000
        for column in result.metadata["units"]:
            assert result.frame[column][-1] is not None
            assert np.isfinite(result.frame[column][-1])
