"""Compatibility for saved analysis selections and former trend-only results."""

from research.analysis import AppliedRun, LogicSpec
from research.analysis.core.runner import source_key


def upgrade_flat_bar_statistics_settings(specs, registry):
    """Preserve selected settings when extending the statistics summaries."""
    name, version = "flat_bar_statistics", "1.3.0-engine.2.2.0"
    if not any(item.name == name and item.version == version for item in registry.definitions()):
        return False
    changed = False
    for spec in specs:
        if (isinstance(spec, dict) and spec.get("name") == name
                and spec.get("version") in {"1.0.0-engine.2.2.0", "1.1.0-engine.2.2.0", "1.2.0-engine.2.2.0"}):
            spec["version"] = version
            changed = True
    return changed


def upgrade_flat_settings(specs, registry, *, saved_defaults=None):
    """Stage known flat settings and defaults; never rewrite applied results."""
    version = "1.1.0-engine.2.2.0"
    if not any(item.name == "flat" and item.version == version for item in registry.definitions()):
        return False
    defaults = registry.defaults("flat", version)
    changed = False
    default_config = saved_defaults.get("flat") if isinstance(saved_defaults, dict) else None
    if isinstance(default_config, dict) and "auto_lookback" in default_config:
        default_config.pop("auto_lookback")
        changed = True
    for spec in specs:
        if (not isinstance(spec, dict) or spec.get("name") != "flat"
                or spec.get("version") not in {
                    "1.0.0-engine.1.0.0", "1.0.0-engine.1.1.0", "1.0.0-engine.2.0.0",
                    "1.0.0-engine.2.1.0", "1.0.0-engine.2.2.0"}
                or not isinstance(spec.get("config"), dict)):
            continue
        config = spec["config"]
        if spec["version"] in {"1.0.0-engine.1.0.0", "1.0.0-engine.1.1.0"}:
            minimum = config.get("min_duration", defaults["min_duration"])
            maximum = defaults["max_duration"]
            if type(minimum) is int:
                maximum = max(maximum, minimum + 1)
            config.setdefault("max_duration", maximum)
            config.pop("slope_min_pct", None)
            config.pop("slope_max_pct", None)
            config.setdefault("max_range_pct", defaults["max_range_pct"])
        if spec["version"] == "1.0.0-engine.2.1.0":
            config.pop("auto_lookback", None)
        for name in ("band_width_mode", "auto_percentile", "auto_multiplier",
                     "long_bar_filter", "long_bar_multiplier"):
            config.setdefault(name, defaults[name])
        # Unknown fields must survive for normal configuration validation.
        spec["version"] = version
        changed = True
    return changed


def flat_settings_upgrade_message(specs, *, action, saved_defaults=None):
    """Describe the versions being staged before upgrade_flat_settings mutates them."""
    if any(isinstance(spec, dict) and spec.get("name") == "flat"
           and spec.get("version") == "1.0.0-engine.2.2.0" for spec in specs):
        return ("Flat analysis now separates periods at recorded missing bars. "
                f"Click {action} to recalculate. Settings and saved results are preserved.")
    legacy_slope = any(isinstance(spec, dict) and spec.get("name") == "flat"
                       and spec.get("version") in {"1.0.0-engine.1.0.0", "1.0.0-engine.1.1.0"}
                       for spec in specs)
    default_config = saved_defaults.get("flat") if isinstance(saved_defaults, dict) else None
    previous_automatic = (any(isinstance(spec, dict) and spec.get("name") == "flat"
                              and spec.get("version") == "1.0.0-engine.2.1.0" for spec in specs)
                          or isinstance(default_config, dict) and "auto_lookback" in default_config)
    if legacy_slope:
        change = ("Flat analysis now requires every candle's high and low to fit inside a price band. "
                  "The previous slope settings have been replaced with a fixed maximum total band width "
                  "of 2% by default. Review the band width and duration limits. ")
    elif not previous_automatic:
        change = "Flat analysis now supports automatic band width and an optional long-bar filter. "
    else:
        change = ""
    if previous_automatic:
        change += ("Automatic width now uses the candles within each candidate flat period, including both endpoints; "
                   "the separate lookback has been removed. Your mode, percentile, multipliers, and filter settings "
                   "are preserved. ")
    else:
        change += "Automatic width and the long-bar filter are off by default. "
    return (change +
            f"Click {action} to calculate with the updated settings. Saved results are unchanged.")


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
