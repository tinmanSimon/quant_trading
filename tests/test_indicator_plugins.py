"""Indicator adapters preserve calculations and render independent live panels."""

from datetime import UTC, datetime, timedelta
import json
import math

import polars as pl
import pytest

import market_indicators
from chart_logics import LogicContext, LogicRegistry, LogicResult, LogicRunner, LogicSpec, discover_logics
from chart_logics.plugins.indicators import (
    ADXLogic, ATRLogic, BollingerBandsLogic, MACDLogic, MovingAverageLogic,
    OBVLogic, RSILogic, StochasticLogic,
)
from dashboard.chart_overlays import validate_overlay


LOGICS = (MovingAverageLogic, RSILogic, MACDLogic, ATRLogic, BollingerBandsLogic,
          ADXLogic, StochasticLogic, OBVLogic)


def bars(count=120):
    close = [100 + index * .04 + 4 * math.sin(index / 3) for index in range(count)]
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=index)
                      for index in range(count)],
        "open": [value - .3 for value in close], "close": close,
        "high": [value + 1 for value in close], "low": [value - 1 for value in close],
        "volume": [1000. + 10 * index for index in range(count)],
    }, schema={"timestamp": pl.Datetime("us", "UTC"), "open": pl.Float64, "close": pl.Float64,
               "high": pl.Float64, "low": pl.Float64, "volume": pl.Float64})


def context(frame, support=None, cutoff=None):
    return LogicContext(frame, "1m", "TEST", cutoff or datetime(2025, 1, 1, tzinfo=UTC),
                        support_frame=support)


def registry():
    result = LogicRegistry()
    for cls in LOGICS:
        result.register_class(cls)
    return result


def rows(frame):
    output = frame.to_dicts()
    for row in output:
        row["timestamp"] = row["timestamp"].isoformat()
    return output


def presented(logic, result, **config):
    return registry().present(logic.name, logic.version, result, config)


@pytest.mark.parametrize("cls", LOGICS)
def test_full_resolution_archive_and_panel_match_independent_package(cls):
    frame = bars()
    before = frame.clone()
    logic = cls()
    config = logic.validate({})
    expected = getattr(market_indicators, logic.function)(frame, **config)
    calls = []
    raw = logic.calculate(context(frame), config, progress=lambda *args: calls.append(args))
    assert calls == [(1, 1)]
    assert raw.tables[0]["rows"] == rows(expected.frame)
    assert frame.equals(before)
    assert not raw.overlays  # Values are archived exactly once, in the table.
    assert raw.data["metadata"]["algorithm_version"] in logic.version
    assert raw.data["metadata"]["initialization_bars"] == 0
    assert raw.data["metadata"]["selected_bars"] == frame.height
    archive = raw.to_dict()
    result = presented(logic, raw)
    assert result.overlays
    for overlay in result.overlays:
        assert overlay["panel"] == raw.data["panel"]
        validate_overlay(overlay)
        if overlay["id"].startswith(("reference_", "histogram_")):
            continue
        assert overlay["x"] == expected.frame["source_index"].to_list()
        assert overlay["y"] == expected.frame[overlay["id"]].to_list()
    assert raw.to_dict() == archive
    restored = LogicResult.from_dict(json.loads(json.dumps(archive)))
    assert presented(logic, restored).to_dict() == result.to_dict()


def test_all_indicator_classes_discover_with_separate_config_and_display_schemas():
    found = discover_logics()
    assert not found.issues
    assert {cls.name for cls in LOGICS} <= {item.name for item in found.definitions()}
    for cls in LOGICS:
        logic = cls()
        assert logic.input_policy == "completed"
        assert not set(logic.config_schema["properties"]) & set(logic.display_schema["properties"])
        assert found.configure(logic.name, logic.version, {}) == found.defaults(logic.name, logic.version)


@pytest.mark.parametrize("cls,config", [
    (MovingAverageLogic, {"kind": "ema", "period": 7, "price_col": "open"}),
    (MovingAverageLogic, {"kind": "wma", "period": 9}),
    (RSILogic, {"period": 7, "price_col": "high"}),
    (MACDLogic, {"fast": 3, "slow": 7, "signal": 1, "price_col": "low"}),
    (ATRLogic, {"period": 5, "normalized": True}),
    (BollingerBandsLogic, {"period": 8, "upper_deviation": 3., "lower_deviation": 1.5}),
    (ADXLogic, {"period": 7}),
    (StochasticLogic, {"period": 5, "k_smoothing": 2, "d_period": 4}),
    (OBVLogic, {"price_col": "open"}),
])
def test_custom_parameters_route_to_shared_calculations(cls, config):
    frame = bars()
    logic = cls()
    config = logic.validate(config)
    raw = logic.calculate(context(frame), config)
    expected = getattr(market_indicators, logic.function)(frame, **config)
    assert raw.tables[0]["rows"] == rows(expected.frame)
    assert raw.data["metadata"]["parameters"] == expected.metadata["parameters"]
    assert raw.data["panel"]["title"] == logic.title(config)


@pytest.mark.parametrize("cls", LOGICS)
def test_supporting_history_initializes_values_then_rebases_selected_rows(cls):
    frame = bars()
    selected = frame.slice(80, 20)
    # Overlapping and later support bars must not replace selected data or
    # influence any output; all earlier support bars remain available.
    support = frame.with_columns(pl.when(pl.arange(0, frame.height) >= 80)
                                 .then(pl.col("close") * 3).otherwise(pl.col("close")).alias("close"))
    support_before = support.clone()
    logic = cls()
    config = logic.validate({})
    raw = logic.calculate(context(selected, support), config)
    expected = getattr(market_indicators, logic.function)(frame.head(100), **config).frame.slice(80)
    expected = expected.with_columns((pl.col("source_index") - 80).alias("source_index"))
    assert raw.tables[0]["rows"] == rows(expected)
    assert raw.data["metadata"]["initialization_bars"] == 80
    assert raw.data["metadata"]["history_start"] == frame["timestamp"][0].isoformat()
    assert raw.data["metadata"]["selected_bars"] == 20
    assert any("80 earlier completed bars" in message for message in raw.messages)
    assert support.equals(support_before)


def test_display_colors_reference_levels_and_visibility_never_recalculate(monkeypatch):
    logic = RSILogic()
    raw = logic.calculate(context(bars()), {"period": 7})
    archive = raw.to_dict()
    monkeypatch.setattr(market_indicators, "rsi", lambda *a, **k: pytest.fail("Display recalculated RSI"))
    shown = presented(logic, raw, value_color="#123456", line_width=3,
                      reference_levels=[25, 75], reference_color="#654321")
    assert shown.overlays[0]["line"] == {"color": "#123456", "width": 3}
    assert [item["y"] for item in shown.overlays[1:]] == [[25, 25], [75, 75]]
    hidden = presented(logic, raw, show_value=False)
    assert not hidden.overlays
    assert hidden.tables == raw.tables
    assert any("All indicator series are hidden" in message for message in hidden.messages)
    assert raw.to_dict() == archive
    assert len(presented(logic, raw, show_reference_levels=False).overlays) == 1


def test_macd_histogram_uses_independently_configurable_scalar_colors():
    logic = MACDLogic()
    raw = logic.calculate(context(bars()), {})
    shown = presented(logic, raw, show_signal=False, positive_color="#001122", negative_color="#334455")
    by_id = {item["id"]: item for item in shown.overlays}
    assert "signal" not in by_id
    assert by_id["histogram_positive"]["marker"] == {"color": "#001122"}
    assert by_id["histogram_negative"]["marker"] == {"color": "#334455"}
    for index, row in enumerate(raw.tables[0]["rows"]):
        positive = by_id["histogram_positive"]["y"][index]
        negative = by_id["histogram_negative"]["y"][index]
        assert positive is None or positive >= 0
        assert negative is None or negative < 0
        assert (positive if positive is not None else negative) == row["histogram"]
    without_histogram = presented(logic, raw, show_histogram=False)
    assert all(item["type"] != "bar" for item in without_histogram.overlays)


def test_instances_cache_independently_and_restore_without_recalculation(monkeypatch):
    runner = LogicRunner(registry(), output_validator=validate_overlay)
    ctx = context(bars())
    calls = []
    original = market_indicators.rsi

    def calculate(*args, **kwargs):
        calls.append(kwargs["period"])
        return original(*args, **kwargs)

    monkeypatch.setattr(market_indicators, "rsi", calculate)
    specs = [LogicSpec("rsi-a", RSILogic.name, RSILogic.version, {"period": 7}),
             LogicSpec("rsi-b", RSILogic.name, RSILogic.version, {"period": 14})]
    run = runner.apply(specs, ctx)
    assert not run.errors
    assert calls == [7, 14]
    again = runner.apply(specs + [LogicSpec("rsi-c", RSILogic.name, RSILogic.version, {"period": 7})], ctx)
    assert not again.errors
    assert calls == [7, 14]
    displayed = runner.present(run, visibility={"rsi-a": False})
    assert not displayed.results["rsi-a"].overlays
    assert displayed.results["rsi-b"].overlays
    assert len({item["panel"]["id"] for item in runner.present(run).overlays}) == 2
    snapshot = json.loads(json.dumps(runner.snapshot(run)))
    restarted = LogicRunner(registry(), output_validator=validate_overlay)
    restored = restarted.restore(snapshot, ctx)
    assert restored is not None
    assert not restarted.present(restored).errors
    assert calls == [7, 14]
    assert restarted.snapshot(restored) == snapshot


@pytest.mark.parametrize("cls", LOGICS)
def test_incomplete_last_bar_is_excluded_and_future_bars_do_not_change_prior_values(cls):
    frame = bars()
    logic = cls()
    cutoff = frame["timestamp"][-1] + timedelta(seconds=30)
    earlier = logic.calculate(context(frame, cutoff=cutoff), {})
    later = logic.calculate(context(frame), {})
    assert earlier.tables[0]["rows"] == later.tables[0]["rows"][:-1]
    assert earlier.data["metadata"]["selected_bars"] == frame.height - 1


@pytest.mark.parametrize("cls", LOGICS)
def test_empty_input_has_table_and_clear_message(cls):
    logic = cls()
    raw = logic.calculate(context(bars(0)), {})
    assert raw.tables[0]["rows"] == []
    assert raw.data["panel"]["id"] == "indicator"
    assert any("No completed bars" in message for message in raw.messages)
    assert not presented(logic, raw).overlays


@pytest.mark.parametrize("cls", [MovingAverageLogic, RSILogic, MACDLogic, ATRLogic,
                               BollingerBandsLogic, ADXLogic, StochasticLogic])
def test_insufficient_history_retains_nulls_and_indicator_panel(cls):
    logic = cls()
    raw = logic.calculate(context(bars(2)), {})
    assert len(raw.tables[0]["rows"]) == 2
    assert any("Insufficient completed history" in message for message in raw.messages)
    shown = presented(logic, raw)
    assert shown.overlays
    assert all(value is None for overlay in shown.overlays if not overlay["id"].startswith("reference_")
               for value in overlay["y"])


@pytest.mark.parametrize("cls,config", [
    (MovingAverageLogic, {"period": 1}), (RSILogic, {"period": 2.0}),
    (RSILogic, {"period": True}), (RSILogic, {"period": 100_001}),
    (MACDLogic, {"fast": 26, "slow": 12}), (ATRLogic, {"period": 0}),
    (BollingerBandsLogic, {"lower_deviation": -1}), (ADXLogic, {"period": 1}),
    (StochasticLogic, {"k_smoothing": 0}), (OBVLogic, {"price_col": "missing"}),
    (RSILogic, {"value_color": "#123456"}),
])
def test_invalid_parameters_fail_before_calculation(cls, config):
    with pytest.raises(ValueError):
        registry().configure(cls.name, cls.version, config)


@pytest.mark.parametrize("config", [
    {"line_width": 0}, {"value_color": "red"}, {"reference_levels": [101]},
    {"reference_levels": [30, 30]}, {"period": 7},
])
def test_invalid_live_settings_are_rejected(config):
    logic = RSILogic()
    raw = logic.calculate(context(bars()), {})
    with pytest.raises(ValueError):
        presented(logic, raw, **config)


def test_configured_price_source_does_not_require_close_or_volume():
    frame = bars().select("timestamp", "open")
    result = RSILogic().calculate(context(frame), {"price_col": "open"})
    assert result.tables[0]["rows"] == rows(market_indicators.rsi(frame, price_col="open").frame)
    with pytest.raises(ValueError, match="volume"):
        OBVLogic().calculate(context(frame), {"price_col": "open"})
    with pytest.raises(ValueError, match="close"):
        RSILogic().calculate(context(frame), {})


@pytest.mark.parametrize("field,value", [("symbol", "OTHER"), ("provider", "OTHER"), ("timeframe", "1h")])
def test_mismatched_supporting_data_fails_clearly(field, value):
    frame = bars().with_columns(pl.lit("TEST").alias("symbol"), pl.lit("source").alias("provider"),
                                pl.lit("1m").alias("timeframe"))
    support = frame.head(80).with_columns(pl.lit(value).alias(field))
    with pytest.raises(ValueError, match="different|timeframe"):
        RSILogic().calculate(context(frame.tail(40), support), {})


def test_mixed_instrument_and_unsorted_history_are_not_silently_stripped():
    frame = bars(50).with_columns(pl.Series("symbol", ["ONE"] * 25 + ["TWO"] * 25))
    with pytest.raises(ValueError, match="single non-null symbol"):
        RSILogic().calculate(context(frame), {})
    with pytest.raises(ValueError, match="increasing"):
        RSILogic().calculate(context(bars(50).reverse()), {})
    with pytest.raises(ValueError, match="increasing"):
        RSILogic().calculate(context(bars(50).tail(10), bars(50).head(40).reverse()), {})
