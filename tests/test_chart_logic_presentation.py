"""Display settings create views of immutable calculations, never new analyses."""

from copy import deepcopy
from datetime import UTC, datetime

import polars as pl

from chart_logics import BaseChartLogic, LogicContext, LogicRegistry, LogicResult, LogicRunner, LogicSpec


class Levels(BaseChartLogic):
    name = "levels"
    calls = 0
    config_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"offset": {"type": "number", "default": 5}},
    }
    display_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"minimum": {"type": "number", "default": 0},
                       "chosen": {"type": "array", "items": {"type": "number"}, "default": []}},
    }

    def calculate(self, context, config, progress=None):
        type(self).calls += 1
        return LogicResult(
            overlays=[{"id": str(index), "type": "markers", "x": [index], "y": [value + config["offset"]]}
                      for index, value in enumerate(context.frame["close"])],
            data={"available": [1, 2, 3]},
        )

    def display_schema_for(self, result):
        schema = deepcopy(self.display_schema)
        schema["properties"]["chosen"]["items"]["enum"] = result.data["available"]
        return schema

    def present(self, result, config):
        # Deliberately mutate the supplied copy to verify framework isolation.
        result.overlays[:] = [item for item in result.overlays if item["y"][0] >= config["minimum"]]
        result.data["available"].append(99)
        return result


def setup():
    Levels.calls = 0
    registry = LogicRegistry()
    registry.register_class(Levels)
    runner = LogicRunner(registry)
    context = LogicContext(
        pl.DataFrame({"timestamp": [datetime(2024, 1, index + 1, tzinfo=UTC) for index in range(3)],
                      "close": [100, 200, 300]}),
        "1d", "source", datetime(2025, 1, 1, tzinfo=UTC))
    run = runner.apply([LogicSpec("one", "levels", "1", {}), LogicSpec("two", "levels", "1", {})], context)
    return registry, runner, context, run


def test_display_changes_do_not_calculate_or_mutate_snapshot_and_cache():
    _, runner, context, run = setup()
    assert Levels.calls == 1
    before = runner.snapshot(run)
    cache_before = deepcopy(runner._cache)
    high = runner.present(run, {"one": {"minimum": 250}})
    assert [item["y"] for item in high.results["one"].overlays] == [[305]]
    assert len(high.results["two"].overlays) == 3
    low = runner.present(run, {"one": {"minimum": 0}})
    assert len(low.results["one"].overlays) == 3
    assert runner.snapshot(run) == before
    assert runner._cache == cache_before
    assert Levels.calls == 1
    assert runner.apply(run.specs, context).overlays == run.overlays
    assert Levels.calls == 1


def test_choices_always_use_complete_result_and_persisted_archive():
    registry, runner, context, run = setup()
    runner.present(run, {"one": {"minimum": 500}})
    assert registry.display_schema("levels", "1", run.results["one"])["properties"]["chosen"]["items"]["enum"] == [1, 2, 3]
    restarted = LogicRunner(registry)
    restored = restarted.restore(runner.snapshot(run), context)
    assert restored is not None
    assert len(restarted.present(restored, {"one": {"minimum": 0}}).overlays) == 6
    assert Levels.calls == 1


def test_static_display_schema_does_not_copy_result_but_returns_isolated_controls():
    class StaticControls(BaseChartLogic):
        name = "static_controls"
        display_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {"levels": {"type": "array", "items": {"type": "number"}, "default": [30, 70]}},
        }

        def calculate(self, context, config, progress=None):
            return LogicResult()

    class ArchiveWithoutCopy(LogicResult):
        def __deepcopy__(self, memo):
            raise AssertionError("Static display schemas must not copy the result archive")

    registry = LogicRegistry()
    registry.register_class(StaticControls)
    archive = ArchiveWithoutCopy(tables=[{"name": "Values", "rows": [{"value": 42}]}])
    controls = registry.display_schema("static_controls", "1", archive)
    controls["properties"]["levels"]["default"].append(50)
    assert StaticControls.display_schema["properties"]["levels"]["default"] == [30, 70]
    assert registry.display_defaults("static_controls", "1", archive) == {"levels": [30, 70]}


def test_dynamic_display_schema_receives_an_isolated_result_copy():
    class MutatingControls(Levels):
        name = "mutating_controls"

        def display_schema_for(self, result):
            result.data["available"].append(4)
            return super().display_schema_for(result)

    registry = LogicRegistry()
    registry.register_class(MutatingControls)
    archive = LogicResult(data={"available": [1, 2, 3]})
    controls = registry.display_schema("mutating_controls", "1", archive)
    assert controls["properties"]["chosen"]["items"]["enum"] == [1, 2, 3, 4]
    assert archive.data["available"] == [1, 2, 3]
    controls["properties"]["chosen"]["items"]["enum"].append(5)
    assert "enum" not in MutatingControls.display_schema["properties"]["chosen"]["items"]


def test_visibility_is_per_instance_and_invalid_display_recovers_without_apply():
    _, runner, _, run = setup()
    hidden = runner.present(run, visibility={"one": False})
    assert hidden.results["one"].overlays == []
    assert len(hidden.results["two"].overlays) == 3
    invalid = runner.present(run, {"one": {"chosen": [100]}})
    assert "one" in invalid.errors
    assert "two" in invalid.results
    assert "one" not in run.errors
    recovered = runner.present(run, {"one": {"chosen": [2]}})
    assert not recovered.errors and len(recovered.overlays) == 6
    assert Levels.calls == 1


def test_old_partial_snapshots_require_apply_and_result_format_invalidates_cache():
    registry, runner, context, run = setup()
    old = runner.snapshot(run)
    old["version"] = 1
    assert runner.restore(old, context) is None
    previous = Levels.result_version
    try:
        Levels.result_version = "2"
        assert not runner.matches(run, context)
        assert LogicRunner(registry).restore(runner.snapshot(run), context) is None
    finally:
        Levels.result_version = previous


def test_metadata_signature_does_not_construct_optional_plugins():
    from dashboard.logic_controls import _signature

    class BrokenConstructor(BaseChartLogic):
        name = "broken_constructor"
        result_version = "archive-2"
        def __init__(self):
            raise RuntimeError("Plugin unavailable")
        def calculate(self, context, config, progress=None):
            raise AssertionError("Never called")

    registry = LogicRegistry()
    registry.register_class(BrokenConstructor)
    assert registry.definition("broken_constructor", "1").result_version == "archive-2"
    assert isinstance(_signature(registry), str)


def test_saved_trend_archive_uses_earliest_endpoint_without_recalculation(monkeypatch):
    from dataclasses import replace
    import json

    from chart_logics.plugins import trend as module
    from trend_analysis import analyze

    frame = pl.DataFrame({
        "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in range(1, 13)],
        "close": [100.0 + index for index in range(12)],
    })
    analysis = analyze(frame)
    earliest_end = min(item.end_index for item in analysis.trends)
    early = replace(next(item for item in analysis.trends if item.end_index == earliest_end), clarity=.8)
    late = replace(next(item for item in analysis.trends
                        if item.end_index == earliest_end + 1 and item.start_index == early.start_index),
                   clarity=.99)
    analysis = replace(analysis, trends=(late, early))
    calls = []

    def calculate_once(*args, **kwargs):
        calls.append(1)
        return analysis

    monkeypatch.setattr(module, "analyze", calculate_once)
    registry = LogicRegistry()
    registry.register_class(module.TrendLogic)
    runner = LogicRunner(registry)
    context = LogicContext(frame, "1d", "chronological", datetime(2025, 1, 1, tzinfo=UTC))
    run = runner.apply([LogicSpec("trend", module.TrendLogic.name, module.TrendLogic.version, {})], context)
    assert not run.errors
    assert calls == [1]
    snapshot = json.loads(json.dumps(runner.snapshot(run)))

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Presenting a saved archive must not recalculate trends")

    monkeypatch.setattr(module.TrendLogic, "calculate", unexpected_calculation)
    restarted = LogicRunner(registry)
    restored = restarted.restore(snapshot, context)
    assert restored is not None
    displayed = restarted.present(restored)
    assert not displayed.errors
    result = displayed.results["trend"]
    assert [overlay["id"] for overlay in result.overlays] == [early.trend_id]
    rows = {row["trend_id"]: row for row in result.tables[0]["rows"]}
    assert rows[early.trend_id]["on_chart"] is True
    assert rows[early.trend_id]["redundant_to"] is None
    assert rows[late.trend_id]["status"] == "Redundant"
    assert rows[late.trend_id]["redundant_to"] == early.trend_id
    expanded = restarted.present(restored, {"trend": {"show_redundant": True}})
    assert not expanded.errors
    assert len(expanded.results["trend"].overlays) == 2
    assert restarted.snapshot(restored) == snapshot
    assert calls == [1]
