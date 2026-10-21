"""Script findings retain exact original bars through text and chart outputs."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from research.analysis.results import AnalysisResult


def context():
    return SimpleNamespace(bars=pl.DataFrame({
        "source_index": [0, 1, 2],
        "timestamp": [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(3)],
        "close": [100., 102., 101.],
    }), metadata={"initialization_bars": 20, "selected_bars": 3})


def test_signals_link_custom_table_markers_and_navigation_without_changing_input():
    ctx = context()
    findings = ctx.bars.filter(pl.col("source_index") != 1).with_columns(pl.lit("MA crossing").alias("reason"))
    result = AnalysisResult(ctx).text("**Candidates**, not simulated trades.").signals(
        "Entries", findings, label="reason", symbol="triangle-up").to_logic_result()
    assert result.data["analysis"]["text"] == ["**Candidates**, not simulated trades."]
    assert result.overlays[0]["x"] == [0, 2]
    assert result.overlays[0]["y"] == [100, 101]
    events = result.data["analysis"]["events"]
    assert [event["label"] for event in events] == ["MA crossing", "MA crossing"]
    rows = result.tables[0]["rows"]
    assert [row["_event_id"] for row in rows] == [event["id"] for event in events]
    assert rows[0]["timestamp"] == ctx.bars["timestamp"][0].isoformat()
    assert "_event_id" not in findings.columns
    assert result.data["analysis"]["layers"] == [
        {key: result.overlays[0][key] for key in ("id", "name", "type")}]
    result.overlays[0]["x"][0] = 99
    assert findings["source_index"].to_list() == [0, 2]


@pytest.mark.parametrize("frame", [
    pl.DataFrame({"source_index": [-1], "close": [1.]}),
    pl.DataFrame({"source_index": [3], "close": [1.]}),
    pl.DataFrame({"source_index": [True], "close": [1.]}),
    pl.DataFrame({"source_index": [1], "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)], "close": [1.]}),
    pl.DataFrame({"close": [1.]}),
])
def test_chart_outputs_reject_unselected_or_misaligned_coordinates(frame):
    with pytest.raises(ValueError):
        AnalysisResult(context()).markers("Findings", frame)


def test_timestamp_only_custom_output_maps_back_to_original_indices_and_sorts():
    ctx = context()
    frame = ctx.bars.reverse().drop("source_index")
    result = AnalysisResult(ctx).text("Custom output").line("Prices", frame).to_logic_result()
    assert result.overlays[0]["x"] == [0, 1, 2]
    assert result.overlays[0]["y"] == [100, 102, 101]


def test_intervals_duplicate_findings_and_as_of_context_preserve_original_indices():
    ctx = context()
    ctx.bars = ctx.bars.slice(1)
    frame = pl.DataFrame({"begin": [1, 1], "finish": [2, 2], "reason": ["a", "b"]})
    builder = AnalysisResult(ctx).text("Intervals").events("Setups", frame, start="begin", end="finish", label="reason")
    events = builder.to_logic_result().data["analysis"]["events"]
    assert len({event["id"] for event in events}) == 2
    assert [event["source_index"] for event in events] == [1, 1]
    assert all(event["end_index"] == 2 for event in events)
    with pytest.raises(ValueError, match="end at or after"):
        AnalysisResult(ctx).events("Backward", pl.DataFrame({"source_index": [2], "end_index": [1]}))


def test_no_findings_is_valid_and_metadata_captures_later_calculations():
    ctx = context()
    builder = AnalysisResult(ctx)
    ctx.metadata["calculations"] = [{"indicator": "rsi", "period": 14}]
    result = builder.signals("Entries", ctx.bars.head(0)).to_logic_result()
    assert result.tables[0]["rows"] == []
    assert "source_index" in result.tables[0]["columns"]
    assert result.data["analysis"]["events"] == []
    assert result.data["metadata"]["calculations"][0]["indicator"] == "rsi"
    with pytest.raises(ValueError, match="text or at least one table"):
        AnalysisResult(ctx).markers("Points", ctx.bars).to_logic_result()


def test_independent_chart_panel_warmup_gaps_and_band_validation():
    ctx = context()
    frame = ctx.bars.with_columns(pl.Series("value", [None, 40., 60.]),
                                  pl.Series("lower", [None, 30., 50.]),
                                  pl.Series("upper", [None, 50., 70.]))
    panel = {"id": "oscillator", "title": "Custom oscillator", "range": [0, 100]}
    result = (AnalysisResult(ctx).table("Values", frame)
              .line("Oscillator", frame, value="value", panel=panel)
              .band("Range", frame, lower="lower", upper="upper", panel=panel).to_logic_result())
    assert result.overlays[0]["y"] == [None, 40., 60.]
    assert result.overlays[1]["lower"] == [None, 30., 50.]
    with pytest.raises(ValueError, match="share null gaps"):
        AnalysisResult(ctx).band("Invalid", frame.with_columns(pl.lit(10.).alias("lower")), lower="lower", upper="upper")


def test_duplicate_names_and_nonfinite_custom_data_fail_clearly():
    ctx = context()
    builder = AnalysisResult(ctx).table("Findings", ctx.bars)
    with pytest.raises(ValueError, match="already exists"):
        builder.table("Findings", ctx.bars)
    with pytest.raises(ValueError, match="finite"):
        builder.line("Invalid", ctx.bars.with_columns(pl.lit(float("nan")).alias("close")))
    with pytest.raises(ValueError):
        builder.table("Invalid", pl.DataFrame({"value": [float("inf")]}))
