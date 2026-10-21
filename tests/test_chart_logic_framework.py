"""Extension contract tests use new modules with no dashboard registration edits."""

from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json

import polars as pl
import pytest

from research.analysis import BaseChartLogic, LogicContext, LogicRegistry, LogicResult, LogicRunner, LogicSpec, discover_logics


def context():
    stamps = [datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=i) for i in range(3)]
    return LogicContext(pl.DataFrame({"timestamp": stamps, "open": [100., 101., 102.],
                                     "close": [101., 102., 103.]}),
                        "1m", ("test", "revision-1"), stamps[-1] + timedelta(minutes=1))


class CountingLogic(BaseChartLogic):
    name = "counting"
    calls = 0
    config_schema = {"type": "object", "properties": {
        "offset": {"type": "number", "default": 5, "minimum": 0}},
        "additionalProperties": False}

    def calculate(self, context, config, progress=None):
        type(self).calls += 1
        if progress:
            progress(1, 1)
        return LogicResult(overlays=[{"type": "line", "id": "levels", "name": "Levels",
                                     "x": [0, 1, 2], "y": [price + config["offset"]
                                                           for price in context.frame["open"]]}])


class BrokenLogic(BaseChartLogic):
    name = "broken"

    def calculate(self, context, config, progress=None):
        raise RuntimeError("Expected failure")


@pytest.fixture
def runner():
    CountingLogic.calls = 0
    registry = LogicRegistry()
    registry.register_class(CountingLogic)
    registry.register_class(BrokenLogic)
    return LogicRunner(registry)


def spec(identifier="one", offset=5):
    return LogicSpec(identifier, "counting", "1", {"offset": offset})


def test_independent_cache_duplicate_instances_and_removal(runner):
    progress = []
    run = runner.apply([spec(), spec("two", 10)], context(), lambda *args: progress.append(args))
    assert CountingLogic.calls == 2
    assert progress == [("one", 1, 1), ("two", 1, 1)]
    assert [item["id"] for item in run.overlays] == ["one:levels", "two:levels"]
    changed = runner.apply([spec(), spec("two", 20)], context())
    assert CountingLogic.calls == 3
    assert changed.results["two"].overlays[0]["y"] == [120, 121, 122]
    duplicate = runner.apply([spec(), spec("duplicate")], context())
    assert CountingLogic.calls == 3
    duplicate.results["one"].overlays[0]["y"][0] = -99
    assert duplicate.results["duplicate"].overlays[0]["y"][0] == 105
    assert runner.apply([spec()], context()).overlays[0]["y"][0] == 105
    assert runner.apply([], context()).overlays == []


def test_instance_panel_ids_are_namespaced_without_changing_archived_descriptors():
    class PanelLogic(CountingLogic):
        name = "panel"

        def calculate(self, context, config, progress=None):
            result = super().calculate(context, config, progress)
            result.overlays[0]["panel"] = {"id": "indicator", "title": "Example"}
            return result

    registry = LogicRegistry()
    registry.register_class(PanelLogic)
    runner = LogicRunner(registry)
    run = runner.apply([LogicSpec(name, "panel", "1", {}) for name in ("first", "second")], context())
    assert not run.errors
    assert [item["panel"]["id"] for item in run.overlays] == ["first:indicator", "second:indicator"]
    assert run.results["first"].overlays[0]["panel"]["id"] == "indicator"
    restored = runner.restore(runner.snapshot(run), context())
    assert restored.overlays == run.overlays


def test_validation_is_atomic_calculation_errors_are_isolated(runner):
    with pytest.raises(ValueError, match="offset"):
        runner.apply([spec(), spec("bad", -1)], context())
    assert CountingLogic.calls == 0
    run = runner.apply([spec(), LogicSpec("bad", "broken", "1", {})], context())
    assert list(run.results) == ["one"]
    assert run.errors == {"bad": "Expected failure"}
    with pytest.raises(ValueError, match="unique"):
        runner.apply([spec(), spec()], context())


def test_source_and_version_changes_invalidate_and_unrelated_clock_does_not(runner):
    original = context()
    run = runner.apply([spec()], original)
    assert runner.matches(run, replace(original, cutoff=original.cutoff + timedelta(seconds=20)))
    assert not runner.matches(run, replace(original, identity=("test", "revision-2")))
    changed = replace(original, frame=original.frame.with_columns((pl.col("open") + 1).alias("open")))
    assert not runner.matches(run, changed)
    runner.apply([spec()], changed)
    assert CountingLogic.calls == 2
    del runner.registry._classes["counting", "1"]
    del runner.registry._definitions["counting", "1"]
    assert not runner.matches(run, original)


def test_completed_input_cache_changes_only_with_completed_bars():
    class Completed(CountingLogic):
        name = "completed"
        input_policy = "completed"
        calls = 0
    registry = LogicRegistry()
    registry.register_class(Completed)
    runner = LogicRunner(registry)
    original = context()
    incomplete = replace(original, cutoff=original.frame["timestamp"][-1])
    specs = [LogicSpec("one", "completed", "1", {})]
    runner.apply(specs, incomplete)
    runner.apply(specs, replace(incomplete, cutoff=incomplete.cutoff + timedelta(seconds=20)))
    assert Completed.calls == 1
    runner.apply(specs, original)
    assert Completed.calls == 2


def test_snapshot_roundtrip_and_rejection(runner):
    original = context()
    run = runner.apply([spec(), LogicSpec("bad", "broken", "1", {})], original)
    snapshot = json.loads(json.dumps(runner.snapshot(run)))
    restored_runner = LogicRunner(runner.registry)
    restored = restored_runner.restore(snapshot, original)
    assert restored.overlays == run.overlays and restored.errors == run.errors
    restored_runner.apply([spec()], original)
    assert CountingLogic.calls == 1
    assert restored_runner.restore(snapshot, replace(original, identity="other")) is None
    snapshot["results"]["one"]["overlays"][0]["y"][0] = float("nan")
    assert restored_runner.restore(snapshot, original) is None


def test_bounded_cache_eviction(runner):
    runner.cache_limit = 1
    runner.apply([spec()], context())
    runner.apply([spec("two", 2)], context())
    runner.apply([spec()], context())
    assert CountingLogic.calls == 3
    assert len(runner._cache) == 1


def test_adding_new_module_discovers_without_registration(tmp_path, monkeypatch):
    package = tmp_path / "extra_chart_logics"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert discover_logics("extra_chart_logics").definitions() == ()
    (package / "example.py").write_text('''
from research.analysis import BaseChartLogic, LogicResult
class Example(BaseChartLogic):
    name = "new_example"
    label = "New example"
    config_schema = {"type":"object", "properties":{
        "offset":{"type":"number","default":5}}, "additionalProperties":False}
    def calculate(self, context, config, progress=None):
        return LogicResult(overlays=[{"type":"line","id":"example","x":[0,2],
            "y":[100 + config["offset"], 100 + config["offset"]]}])
''')
    registry = discover_logics("extra_chart_logics")
    assert registry.issues == ()
    definition, = registry.definitions()
    assert definition.name == "new_example"
    assert registry.configure("new_example", "1", {}) == {"offset": 5}
    run = LogicRunner(registry).apply([LogicSpec("custom", "new_example", "1", {})], context())
    assert run.overlays[0]["y"] == [105, 105]
    from dashboard.chart_component import window_payload
    from dashboard.chart_data import ChartSeries
    frame = context().frame.with_columns(pl.col("close").alias("high"), pl.col("open").alias("low"),
                                         pl.lit(10.).alias("volume"))
    series = ChartSeries(frame, timeframe="1m")
    payload = window_payload(series, chart_id="example", timeframe="1m", timezone="UTC",
                             overlays=run.overlays)
    assert any(trace.get("name") == "custom:example" or trace.get("meta", {}).get("overlay_id") == "custom:example"
               or trace.get("meta", {}).get("id") == "custom:example"
               for trace in payload["figure"]["data"])


def test_broken_import_and_duplicate_identity_do_not_hide_other_logics(tmp_path, monkeypatch):
    package = tmp_path / "conflicting_chart_logics"
    package.mkdir()
    (package / "__init__.py").write_text("")
    template = '''
from research.analysis import BaseChartLogic, LogicResult
class Example(BaseChartLogic):
    name = {name!r}
    def calculate(self, context, config, progress=None):
        return LogicResult()
'''
    for module, name in (("a", "duplicate"), ("b", "duplicate"), ("c", "healthy"), ("d", "duplicate")):
        (package / f"{module}.py").write_text(template.format(name=name))
    (package / "broken.py").write_text("raise RuntimeError('broken import')")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = discover_logics("conflicting_chart_logics")
    assert [definition.name for definition in registry.definitions()] == ["healthy"]
    assert len(registry.issues) == 3


@pytest.mark.parametrize("overlay", [
    {"id": "bad", "type": "line", "x": [1, 0], "y": [1, 2]},
    {"id": "bad", "type": "line", "x": [0], "y": [float("inf")]},
    {"id": "bad", "type": "band", "x": [0], "lower": [5], "upper": [4]},
    {"id": "bad", "type": "log_line", "start_index": 0, "end_index": 1, "slope": 1000, "intercept": 1},
])
def test_malformed_outputs_are_rejected(overlay):
    with pytest.raises(ValueError):
        LogicResult(overlays=[overlay]).validate()
