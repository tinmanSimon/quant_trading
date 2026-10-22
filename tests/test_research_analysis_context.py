"""Script authoring boundaries, alignment, and compatibility of research APIs."""

from datetime import UTC, date, datetime, timedelta
import json
import subprocess
import sys

import polars as pl
import pytest

from research.analysis.core.base import LogicContext
from research.analysis.scripting.context import AnalysisContext, Feature
from research.analysis.scripting.signals import crossed_above, crossed_below
from research.analysis import indicators, trend


def bars(values=(4., 3., 2., 3., 5., 7., 6., 8.)):
    return pl.DataFrame({
        "timestamp": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(values))],
        "open": values, "high": [v + 1 for v in values], "low": [v - 1 for v in values],
        "close": values, "volume": [100.] * len(values), "symbol": ["TEST"] * len(values),
    })


def context(frame=None, *, start=0, cache=None, cutoff=None):
    frame = bars() if frame is None else frame
    logic = LogicContext(frame.slice(start), "1d", "TEST", cutoff or datetime(2030, 1, 1, tzinfo=UTC),
                         support_frame=frame.head(start), metadata={"source": "test"})
    return AnalysisContext.from_logic(logic, shared_cache=cache)


def test_research_analysis_import_does_not_load_ui_or_backtesting():
    result = subprocess.run([sys.executable, "-B", "-c", "import research.analysis.scripting.context, sys, json; "
                             "print(json.dumps([name for name in sys.modules if name.startswith("
                             "('dashboard', 'streamlit', 'research.backtesting', 'research.strategies'))]))"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


def test_support_initializes_first_selected_crossover():
    ctx = context(start=4)
    data = ctx.features(fast=ctx.sma(2), slow=ctx.sma(3)).with_columns(
        crossing=crossed_above("fast", "slow"))
    selected = ctx.selected(data)
    assert selected["source_index"].to_list() == [0, 1, 2, 3]
    assert data["source_index"].to_list() == list(range(-4, 4))
    assert selected["crossing"].to_list() == [True, False, False, False]
    assert selected["slow"][0] == pytest.approx(10 / 3)
    assert ctx.metadata["initialization_bars"] == 4
    assert len(ctx.metadata["calculations"]) == 2


def test_multi_output_features_and_explicit_output_selection():
    ctx = context(bars(tuple(float(i + 10) for i in range(60))))
    result = ctx.features(momentum=ctx.macd(), main=ctx.macd().select("macd"))
    assert {"momentum_macd", "momentum_signal", "momentum_histogram", "main"} <= set(result.columns)
    assert result["momentum_macd"].equals(result["main"])
    with pytest.raises(ValueError, match="feature outputs"):
        ctx.macd().select("wrong")
    with pytest.raises(ValueError, match="conflicts"):
        ctx.features(close=ctx.sma(2))


def test_warmup_values_stay_null():
    ctx = context()
    result = ctx.features(rsi=ctx.rsi(14), average=ctx.sma(3))
    assert result["rsi"].null_count() == result.height
    assert result["average"].null_count() == 2


def test_calculation_cache_shared_without_mutable_outputs(monkeypatch):
    calls = []
    original = indicators.rsi
    def counted(frame, **kwargs):
        calls.append(kwargs)
        return original(frame, **kwargs)
    monkeypatch.setattr(indicators, "rsi", counted)
    cache = {}
    first, second = context(start=3, cache=cache), context(start=3, cache=cache)
    returned = first.rsi(2)
    returned.frame.drop_in_place("value")
    returned.metadata["parameters"]["period"] = 99
    assert second.rsi(2).frame.columns == ["source_index", "timestamp", "value"]
    assert first.rsi(2).metadata["parameters"]["period"] == 2
    assert len(calls) == 1
    changed = bars().with_columns((pl.col("close") + 1).alias("close"))
    context(changed, start=3, cache=cache).rsi(2)
    assert len(calls) == 2


def test_frames_and_metadata_are_copies():
    ctx = context(start=2)
    ctx.bars.drop_in_place("close")
    ctx.history.drop_in_place("timestamp")
    ctx.metadata["source"] = "changed"
    assert "close" in ctx.bars.columns
    assert "timestamp" in ctx.history.columns
    assert ctx.metadata["source"] == "test"


def test_mismatched_and_reordered_features_fail_instead_of_silent_alignment():
    ctx = context(start=2)
    feature = ctx.sma(2).frame
    with pytest.raises(ValueError, match="exactly align"):
        ctx.features(bad=feature.reverse())
    with pytest.raises(ValueError, match="exactly align"):
        ctx.features(bad=feature.with_columns((pl.col("source_index") + 1).alias("source_index")))
    with pytest.raises(ValueError, match="unknown or mismatched"):
        ctx.selected(ctx.history.with_columns((pl.col("source_index") + 1).alias("source_index")))
    with pytest.raises(ValueError, match="timestamp and source_index"):
        ctx.selected(ctx.bars.drop("timestamp"))
    repeated = pl.concat([ctx.bars.head(1), ctx.bars.head(1)])
    assert ctx.selected(repeated).height == 2


def test_at_restricts_history_and_preserves_selected_identities():
    ctx = context(start=2)
    restricted = ctx.at(2)
    assert restricted.bars["source_index"].to_list() == [0, 1, 2]
    assert restricted.history.height == 5
    assert restricted.history["timestamp"][-1] == ctx.bars["timestamp"][2]
    assert restricted.history.equals(ctx.at(ctx.bars["timestamp"][2]).history)
    assert ctx.at(-1).bars.is_empty()
    assert ctx.at(-1).history.height == 2
    with pytest.raises(ValueError, match="available bar"):
        restricted.at(3)
    with pytest.raises(ValueError, match="source_index or timestamp"):
        ctx.at(True)
    original = context(bars().head(5), start=2)
    assert restricted.rsi(2).frame.equals(original.rsi(2).frame)
    assert restricted.metadata["as_of"] == ctx.bars["timestamp"][2].isoformat()


def test_features_do_not_change_when_future_rows_are_appended():
    frame = bars(tuple(10. + (i % 7) for i in range(60)))
    earlier = context(frame.head(40), start=10)
    later = context(frame, start=10)
    a = earlier.features(average=earlier.ema(3), rsi=earlier.rsi(3), momentum=earlier.macd(3, 6, 3))
    b = later.features(average=later.ema(3), rsi=later.rsi(3), momentum=later.macd(3, 6, 3))
    assert a.equals(b.head(40))


@pytest.mark.parametrize("helper,expected", [(crossed_above, [False, True, False, False, False, True]),
                                            (crossed_below, [False, False, False, True, False, False])])
def test_crossovers_include_equality_nulls_and_scalar_operands(helper, expected):
    frame = pl.DataFrame({"value": [2., 3., 2., 1., 2., 3.], "threshold": [2.] * 6})
    assert frame.select(helper("value", 2).alias("signal"))["signal"].to_list() == expected
    assert frame.select(helper("value", "threshold").alias("signal"))["signal"].to_list() == expected
    frame = pl.DataFrame({"value": [None, 3., None, 1.]})
    assert frame.select(helper("value", 2).alias("signal"))["signal"].to_list() == [False] * 4


def test_trends_preserve_observation_endpoints_and_prefix_results():
    frame = bars(tuple(10 * 1.02 ** i for i in range(28)))
    config = trend.TrendConfig(max_history=16, min_segment_length=8, min_tail_length=8)
    earlier = context(frame.head(22), start=10)
    later = context(frame, start=10)
    result = earlier.trends(config)
    assert result.trends
    assert all(item.end_index >= 0 for item in result.trends)
    assert any(item.start_index < 0 for item in result.trends)
    for item in result.trends:
        assert item.end_time == earlier.bars["timestamp"][item.end_index]
    assert result.trends == tuple(item for item in later.trends(config).trends if item.end_index <= 11)
    assert earlier.at(11).trends(config).trends == result.trends
    assert any(item.end_index < 0 for item in earlier.trends(config, include_history=True).trends)
    assert earlier.metadata["calculations"][-1]["observation_endpoint"] == "end_index/end_time"


def test_date_labels_and_incomplete_suffix():
    frame = bars().with_columns(pl.col("timestamp").cast(pl.Date))
    ctx = context(frame, cutoff=datetime(2020, 1, 3, 5, tzinfo=UTC))
    assert ctx.bars["timestamp"].to_list() == [date(2020, 1, 1), date(2020, 1, 2)]
    assert ctx.features(average=ctx.sma(2)).height == 2


def test_support_after_range_start_is_ignored():
    frame = bars()
    logic = LogicContext(frame.slice(3), "1d", "TEST", datetime(2030, 1, 1, tzinfo=UTC), support_frame=frame)
    ctx = AnalysisContext.from_logic(logic)
    assert ctx.history.height == frame.height
    assert ctx.metadata["initialization_bars"] == 3


def test_invalid_mixed_or_unsorted_inputs_fail():
    with pytest.raises(ValueError, match="increasing"):
        context(bars().reverse())
    with pytest.raises(ValueError, match="single non-null symbol"):
        context(bars().with_columns(pl.Series("symbol", ["A"] * 4 + ["B"] * 4)))
    frame = bars()
    logic = LogicContext(frame.slice(3), "1d", "TEST", datetime(2030, 1, 1, tzinfo=UTC),
                         support_frame=frame.head(3).with_columns(pl.lit("OTHER").alias("symbol")))
    with pytest.raises(ValueError, match="different symbol"):
        AnalysisContext.from_logic(logic)


@pytest.mark.parametrize("timeframe,stamp,expected", [
    ("1m", datetime(2020, 1, 1, 13, tzinfo=UTC), datetime(2020, 1, 1, 13, 1, tzinfo=UTC)),
    ("1d", date(2020, 1, 1), datetime(2020, 1, 2, 5, tzinfo=UTC)),
    ("1wk", date(2020, 1, 1), datetime(2020, 1, 6, 5, tzinfo=UTC)),
    ("1mo", date(2020, 1, 1), datetime(2020, 2, 1, 5, tzinfo=UTC)),
])
def test_as_of_cutoff_is_the_observed_bars_completion_bound(timeframe, stamp, expected):
    frame = pl.DataFrame({"timestamp": [stamp], "close": [100.]})
    original = LogicContext(frame, timeframe, "TEST", datetime(2030, 1, 1, tzinfo=UTC))
    ctx = AnalysisContext.from_logic(original).at(0)
    assert ctx.cutoff == expected
    assert ctx.metadata["cutoff"] == expected.isoformat()
    assert ctx.metadata["source_cutoff"] == original.cutoff.isoformat()
