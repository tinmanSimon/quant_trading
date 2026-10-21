"""Compatibility for saved trend selections and former trend-only results."""

from research.analysis import AppliedRun, LogicSpec
from research.analysis.runner import source_key


def upgrade_trend_windows(selection, registry):
    """Stage the known v4-to-v5 settings upgrade without reusing old results."""
    version = "1.0.0-engine.5.0.0"
    if not any(item.name == "trend" and item.version == version for item in registry.definitions()):
        return False
    changed = False
    for spec in selection["draft"]:
        if (isinstance(spec, dict) and spec.get("name") == "trend"
                and spec.get("version") == "1.0.0-engine.4.0.0"):
            spec["version"] = version
            changed = True
    # Leave applied specs and archives on their original version until Apply.
    return changed


def restore_legacy_analysis(memory, selection, runner, context):
    """Reuse a verified old result without recalculating or changing its inputs."""
    saved = selection.get("applied", [])
    if (len(saved) != 1 or saved[0].get("instance_id") != "legacy-trend-analysis"
            or memory.chart_analysis_path.exists()):
        return None
    try:
        from dashboard.market_data.legacy_trends import _result_key
        from research.analysis.trend import TrendConfig

        spec, = runner.validate([LogicSpec.from_dict(saved[0])])
        if spec.name != "trend":
            return None
        values = {**TrendConfig().to_dict(), **spec.config}
        display = selection.get("display", {}).get(spec.instance_id, {}).get("config", {})
        values.update({key: value for key, value in display.items() if key in values})
        config = TrendConfig.from_dict(values)
        key = _result_key(context.completed_frame, timeframe=context.timeframe,
                          identity=context.identity, as_of=None, config=config)
        if memory.values.get("analysis_key") != key:
            return None
        legacy = memory.load_analysis(key, config)
        if legacy is None:
            return None
        result = runner.registry.create(spec.name, spec.version).from_analysis(legacy, spec.config)
        runner._validate_output(result)
        run = AppliedRun(source_key(context), [spec], {spec.instance_id: result}, {},
                         {spec.instance_id: runner._key(spec, context)})
        # Restoration revalidates the new contract and primes the shared cache.
        snapshot = runner.snapshot(run)
        restored = runner.restore(snapshot, context)
        if restored is not None:
            memory.save_chart_analysis(snapshot)
            selection["applied"] = [spec.to_dict()]
        return restored
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        # Settings remain migrated even when the old result needs a fresh Apply.
        return None
