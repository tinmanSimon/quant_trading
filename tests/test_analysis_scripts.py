"""Ordinary research functions, typed settings, and private-source reloads."""

from datetime import UTC, date, datetime, timedelta
from enum import Enum
import json
import math
import os
from pathlib import Path
import shutil
from typing import Literal

import polars as pl
import pytest

from research.analysis import AnalysisContext, LogicContext, LogicRegistry, LogicRunner, LogicSpec, analysis, run_script
from research.analysis.loader import load_registry
from research.analysis.scripts import script_class


class Choice(Enum):
    FIRST = "first"
    SECOND = "second"


def source_frame(count=100):
    values = [100 + i * .2 + 10 * math.sin(i / 6) for i in range(count)]
    return pl.DataFrame({
        "timestamp": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(count)],
        "open": values, "high": [v + 1 for v in values], "low": [v - 1 for v in values],
        "close": values, "volume": [100.] * count,
    })


def logic_context(frame=None, start=0):
    frame = source_frame() if frame is None else frame
    return LogicContext(frame.slice(start), "1d", "TEST", datetime(2030, 1, 1, tzinfo=UTC),
                        support_frame=frame.head(start))


def write_private(tmp_path, files):
    root = tmp_path / "secrete_analysis"
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


SIMPLE_SCRIPT = '''from research.analysis import analysis
@analysis("private_demo")
def run(ctx, value: int = 2):
    return ctx.result().text(f"value={value}")
'''


def test_decorator_preserves_ordinary_function_and_validates_typed_settings():
    def original(ctx, period: int = 3, mode: Literal["fast", "slow"] = "fast", enabled: bool = True,
                 choice: Choice = Choice.FIRST, when: date = date(2020, 1, 1)):
        assert isinstance(choice, Choice)
        assert isinstance(when, date)
        return ctx.result().text(f"{period}/{mode}/{enabled}/{choice.value}/{when.isoformat()}")
    decorated = analysis("typed", parameters={"period": {"minimum": 2, "maximum": 60, "title": "Period"}})(original)
    assert decorated is original
    direct = original(AnalysisContext.from_logic(logic_context())).to_logic_result()
    assert run_script(original, logic_context()).to_dict() == direct.to_dict()
    result = run_script(original, logic_context(), period=5, mode="slow", choice="second", when="2022-03-04")
    assert result.data["analysis"]["text"] == ["5/slow/True/second/2022-03-04"]
    registry = LogicRegistry()
    registry.register_class(script_class(original, "typed-source"))
    properties = registry.definition("typed", "1").schema["properties"]
    assert properties["period"] == {"type": "integer", "default": 3, "minimum": 2, "maximum": 60, "title": "Period"}
    assert properties["mode"]["enum"] == ["fast", "slow"]
    for config in ({"period": 1}, {"period": True}, {"mode": "bad"}, {"enabled": 1}, {"typo": 3}):
        with pytest.raises((TypeError, ValueError)):
            run_script(original, logic_context(), **config)


def test_required_and_inferred_parameters_and_mutable_defaults():
    @analysis("required")
    def required(ctx, required: int, *, values: list[int] = [1], title="hello"):
        values.append(required)
        return ctx.result().text(f"{title}:{values}")
    with pytest.raises(ValueError, match="required"):
        run_script(required, logic_context())
    assert run_script(required, logic_context(), required=2).data["analysis"]["text"] == ["hello:[1, 2]"]
    assert run_script(required, logic_context(), required=3).data["analysis"]["text"] == ["hello:[1, 3]"]
    schema = script_class(required, "source").config_schema
    assert schema["properties"]["title"]["type"] == "string"
    assert schema["required"] == ["required"]


def test_invalid_authoring_and_return_types_fail_explicitly():
    with pytest.raises(ValueError, match="identifiers"):
        analysis("bad name")
    with pytest.raises(TypeError, match="first analysis argument"):
        analysis("empty")(lambda: None)
    with pytest.raises(TypeError, match="named parameters"):
        analysis("args")(lambda ctx, *args: None)
    with pytest.raises(ValueError, match="declared function"):
        analysis("unknown", parameters={"typo": {"minimum": 1}})(lambda ctx, period=2: None)
    async def asynchronous(ctx):
        return ctx.result().text("wrong")
    with pytest.raises(TypeError, match="synchronous"):
        analysis("async")(asynchronous)
    @analysis("wrong_return")
    def wrong_return(ctx):
        return {"message": "Not a result"}
    with pytest.raises(TypeError, match="must return ctx.result"):
        run_script(wrong_return, logic_context())
    with pytest.raises(TypeError, match="Use @analysis"):
        run_script(lambda ctx: None, logic_context())


def test_private_loader_is_optional_and_includes_builtins(tmp_path):
    registry = load_registry(tmp_path)
    assert not registry.issues
    assert {"trend", "moving_average", "rsi"} <= {item.name for item in registry.definitions()}


def test_private_loader_fingerprints_helper_edits_even_same_size_same_mtime(tmp_path):
    root = write_private(tmp_path, {"analyses/demo.py": '''from research.analysis import analysis
@analysis("helper_demo")
def run(ctx):
    from ..helpers.value import VALUE
    return ctx.result().text(str(VALUE))
''', "helpers/value.py": "VALUE = 11\n"})
    first = load_registry(tmp_path)
    assert not first.issues
    initial = first.definition("helper_demo", "1").implementation_fingerprint
    assert load_registry(tmp_path).create("helper_demo", "1").__class__ is first.create("helper_demo", "1").__class__
    helper = root / "helpers/value.py"
    original_time = helper.stat().st_mtime_ns
    helper.write_text("VALUE = 22\n")
    os.utime(helper, ns=(original_time, original_time))
    second = load_registry(tmp_path)
    assert second.definition("helper_demo", "1").implementation_fingerprint != initial
    # Deferred imports still read each generation's captured helper snapshot.
    assert first.create("helper_demo", "1").calculate(logic_context(), {}).data["analysis"]["text"] == ["11"]
    assert second.create("helper_demo", "1").calculate(logic_context(), {}).data["analysis"]["text"] == ["22"]
    third = load_registry(tmp_path, reload=True)
    assert third.create("helper_demo", "1").__class__ is not second.create("helper_demo", "1").__class__
    assert third.definition("helper_demo", "1").implementation_fingerprint == second.definition("helper_demo", "1").implementation_fingerprint


def test_import_failures_duplicates_and_unrelated_analyses_are_isolated(tmp_path):
    write_private(tmp_path, {"analyses/good.py": SIMPLE_SCRIPT,
                             "analyses/broken.py": "raise RuntimeError('bad import')\n",
                             "analyses/one.py": SIMPLE_SCRIPT.replace("private_demo", "duplicate"),
                             "analyses/two.py": SIMPLE_SCRIPT.replace("private_demo", "duplicate"),
                             "analyses/three.py": SIMPLE_SCRIPT.replace("private_demo", "duplicate"),
                             "analyses/_not_discovered.py": "raise RuntimeError('should not import')\n",
                             "helpers/not_imported.py": "raise RuntimeError('should not import')\n"})
    registry = load_registry(tmp_path)
    names = {item.name for item in registry.definitions()}
    assert "private_demo" in names and "trend" in names
    assert "duplicate" not in names
    assert any("bad import" in issue for issue in registry.issues)
    assert any("Duplicate analysis" in issue for issue in registry.issues)
    assert not any("should not import" in issue for issue in registry.issues)


def test_missing_import_can_be_repaired_and_discovered_on_reload(tmp_path):
    write_private(tmp_path, {"analyses/demo.py": "from ..helpers.missing import VALUE\n" + SIMPLE_SCRIPT})
    assert load_registry(tmp_path).issues
    write_private(tmp_path, {"helpers/missing.py": "VALUE = 1\n"})
    registry = load_registry(tmp_path, reload=True)
    assert not registry.issues
    assert registry.create("private_demo", "1").calculate(logic_context(), {"value": 4}).data["analysis"]["text"] == ["value=4"]


def test_runner_cache_restore_rejects_changed_private_source(tmp_path):
    root = write_private(tmp_path, {"analyses/demo.py": SIMPLE_SCRIPT})
    registry = load_registry(tmp_path)
    runner = LogicRunner(registry)
    specs = [LogicSpec("demo", "private_demo", "1", {"value": 5})]
    context = logic_context()
    applied = runner.apply(specs, context)
    assert not applied.errors
    snapshot = runner.snapshot(applied)
    assert LogicRunner(load_registry(tmp_path)).restore(snapshot, context) is not None
    (root / "analyses/demo.py").write_text(SIMPLE_SCRIPT.replace("value={value}", "updated={value}"))
    changed = LogicRunner(load_registry(tmp_path))
    assert changed.restore(snapshot, context) is None
    assert changed.apply(specs, context).results["demo"].data["analysis"]["text"] == ["updated=5"]


def test_multiple_scripts_share_calculations_and_runtime_failures_are_isolated(monkeypatch):
    from research.analysis import indicators
    calls = []
    original = indicators.rsi
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(indicators, "rsi", counted)
    @analysis("first")
    def first(ctx):
        ctx.rsi(14)
        return ctx.result().text("first")
    @analysis("second")
    def second(ctx):
        ctx.rsi(14)
        return ctx.result().text("second")
    @analysis("failed")
    def failed(ctx):
        raise ValueError("Example independent failure")
    registry = LogicRegistry()
    for function in (first, second, failed):
        registry.register_class(script_class(function, "source"))
    runner = LogicRunner(registry)
    applied = runner.apply([LogicSpec(name, name, "1", {}) for name in ("first", "failed", "second")], logic_context())
    assert set(applied.results) == {"first", "second"}
    assert applied.errors == {"failed": "Example independent failure"}
    assert len(calls) == 1


def test_private_chart_layers_are_optional_and_visibility_preserves_tables():
    @analysis("layers")
    def layers(ctx):
        result = ctx.result().signals("Findings", ctx.bars.head(2))
        return result.line("Average", ctx.selected(ctx.features(average=ctx.sma(3))), value="average")
    registry = LogicRegistry()
    registry.register_class(script_class(layers, "source"))
    runner = LogicRunner(registry)
    applied = runner.apply([LogicSpec("example", "layers", "1", {})], logic_context())
    assert not applied.errors
    shown = runner.present(applied)
    assert len(shown.overlays) == 2
    hidden = runner.present(applied, {"example": {"visible_layers": []}})
    assert not hidden.overlays
    assert hidden.results["example"].tables == applied.results["example"].tables
    assert len(applied.results["example"].overlays) == 2


def test_tracked_example_loads_runs_and_survives_snapshot_roundtrip(tmp_path):
    template = Path(__file__).resolve().parents[1] / "examples" / "secrete_analysis"
    shutil.copytree(template, tmp_path / "secrete_analysis")
    registry = load_registry(tmp_path)
    assert not registry.issues
    runner = LogicRunner(registry)
    context = logic_context(start=20)
    spec = LogicSpec("example", "ma_rsi_example", "1", {"fast": 3, "slow": 8, "rsi_period": 3,
                                                        "entry_rsi_limit": 100, "exit_rsi_level": 70})
    applied = runner.apply([spec], context)
    assert not applied.errors
    result = applied.results["example"]
    assert result.data["metadata"]["initialization_bars"] == 20
    assert len(result.tables) == 2
    assert result.data["analysis"]["events"]
    assert len(result.overlays) == 5
    assert runner.restore(runner.snapshot(applied), context).results["example"].to_dict() == result.to_dict()
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_example_trend_confirmation_uses_observation_endpoint_and_stays_causal(tmp_path):
    template = Path(__file__).resolve().parents[1] / "examples" / "secrete_analysis"
    shutil.copytree(template, tmp_path / "secrete_analysis")
    registry = load_registry(tmp_path)
    logic = registry.create("ma_rsi_example", "1")
    config = registry.configure("ma_rsi_example", "1", {"fast": 3, "slow": 16, "rsi_period": 3,
                                                        "entry_rsi_limit": 100, "confirm_trend": True, "trend_history": 32})
    frame = source_frame(140)
    earlier = logic.calculate(logic_context(frame.head(100), start=20), config)
    later = logic.calculate(logic_context(frame, start=20), config)
    entries = next(table["rows"] for table in earlier.tables if table["name"] == "Entries")
    later_entries = next(table["rows"] for table in later.tables if table["name"] == "Entries")
    assert entries
    assert entries == [row for row in later_entries if row["source_index"] < 80]
    observed = {item.end_index for item in AnalysisContext.from_logic(logic_context(frame.head(100), start=20)).trends(max_history=32).trends
                if item.direction > 0}
    assert all(row["source_index"] in observed for row in entries)


def test_cutoff_dependent_scripts_opt_in_to_exact_clock_cache_keys():
    from dataclasses import replace
    calls = []
    @analysis("clock", uses_cutoff=True)
    def clock(ctx):
        calls.append(ctx.cutoff)
        return ctx.result().text(ctx.cutoff.isoformat())
    @analysis("bars_only")
    def bars_only(ctx):
        calls.append("bars")
        return ctx.result().text(str(ctx.bars.height))
    registry = LogicRegistry()
    for function in (clock, bars_only):
        registry.register_class(script_class(function, "source"))
    runner = LogicRunner(registry)
    specs = [LogicSpec(name, name, "1", {}) for name in ("clock", "bars_only")]
    context = logic_context()
    first = runner.apply(specs, context)
    second = runner.apply(specs, replace(context, cutoff=context.cutoff + timedelta(minutes=1)))
    assert len(calls) == 3
    assert first.keys["bars_only"] == second.keys["bars_only"]
    assert first.keys["clock"] != second.keys["clock"]
    assert first.results["clock"].data["analysis"]["text"] != second.results["clock"].data["analysis"]["text"]
    with pytest.raises(ValueError, match="boolean"):
        analysis("invalid", uses_cutoff=1)


def test_layer_labels_cannot_hide_another_layer_with_the_same_display_name():
    @analysis("ambiguous_labels")
    def ambiguous(ctx):
        return (ctx.result().text("Ambiguous names")
                .line("A", ctx.bars.head(2))
                .markers("A", ctx.bars.head(2))
                .line("A (line)", ctx.bars.head(2)))
    registry = LogicRegistry()
    registry.register_class(script_class(ambiguous, "source"))
    runner = LogicRunner(registry)
    applied = runner.apply([LogicSpec("example", "ambiguous_labels", "1", {})], logic_context())
    defaults = registry.display_defaults("ambiguous_labels", "1", applied.results["example"])
    assert len(defaults["visible_layers"]) == 3
    assert len(runner.present(applied).overlays) == 3
