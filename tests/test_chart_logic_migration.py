"""Reuse legacy results only when their verified inputs still match."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import math

import polars as pl

from chart_logics import LogicContext, LogicRunner, LogicSpec, discover_logics
from chart_logics.plugins.trend import TrendLogic
from dashboard.logic_migration import restore_legacy_analysis
from dashboard.trends import _result_key
from dashboard.workspace import MarketMemory
from trend_analysis import TrendConfig, analyze


def saved_legacy(tmp_path):
    stamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(24)]
    frame = pl.DataFrame({"timestamp": stamps, "close": [100 * math.exp(.002 * i) for i in range(24)]})
    context = LogicContext(frame, "1d", ("revision", 1), datetime(2025, 1, 1, tzinfo=UTC))
    config = TrendConfig()
    old = analyze(context.completed_frame, config=config)
    key = _result_key(context.completed_frame, timeframe="1d", identity=context.identity,
                      as_of=None, config=config)
    memory = MarketMemory("legacy", tmp_path)
    memory.save_analysis(key, old)
    registry = discover_logics()
    settings = registry.configure("trend", TrendLogic.version,
                                  {key: value for key, value in config.to_dict().items()
                                   if key in TrendLogic.config_schema["properties"]})
    spec = LogicSpec("legacy-trend-analysis", "trend", TrendLogic.version, settings)
    selection = {"version": 1, "applied": [spec.to_dict()], "draft": [spec.to_dict()]}
    return memory, selection, LogicRunner(registry), context, old


def test_compatible_legacy_snapshot_reuses_analysis_and_primes_cache(tmp_path, monkeypatch):
    memory, selection, runner, context, old = saved_legacy(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("Migration must not recalculate trends")

    monkeypatch.setattr("chart_logics.plugins.trend.analyze", fail)
    run = restore_legacy_analysis(memory, selection, runner, context)
    assert run is not None
    assert len(run.results["legacy-trend-analysis"].tables[0]["rows"]) == len(old.trends)
    assert memory.chart_analysis_path.exists()
    assert runner.apply(run.specs, context).overlays == run.overlays
    restarted = LogicRunner(runner.registry).restore(memory.load_chart_analysis(), context)
    assert restarted.overlays == run.overlays


def test_changed_source_or_parameters_do_not_restore_old_analysis(tmp_path):
    memory, selection, runner, context, _ = saved_legacy(tmp_path)
    assert restore_legacy_analysis(memory, selection, runner, replace(context, identity="changed")) is None
    selection["applied"][0]["config"]["penalty_multiplier"] = 2
    assert restore_legacy_analysis(memory, selection, runner, context) is None
    assert not memory.chart_analysis_path.exists()


def test_existing_new_snapshot_never_falls_back_to_legacy(tmp_path):
    memory, selection, runner, context, _ = saved_legacy(tmp_path)
    memory.save_chart_analysis({"version": 1, "specs": []})
    assert restore_legacy_analysis(memory, selection, runner, context) is None
