"""Schema-driven chart logic selection, explicit execution, and result display.

The panel knows only the shared chart-logic contract. All analysis settings and
outputs belong to discovered modules; adding a module never needs a UI branch.
"""

from copy import deepcopy
from hashlib import sha256
import json
from time import monotonic
from uuid import uuid4

import streamlit as st

from research.analysis import LogicRunner, LogicSpec
from dashboard.components.analysis_selection import (
    _clear_flat_lookback_widgets, _configuration, _settings_object, _needs_json_controls,
)
from dashboard.charts.overlays import validate_overlay


_STATE_PREFIX = "_chart_logic_controls_"


def discover_logics():
    """Load built-in and private research analyses without calculating them."""
    from research.analysis.scripting.discovery import load_registry
    return load_registry()


def _legacy_selection(memory, registry):
    """One-time migration of the former trend-only workspace and defaults."""
    # The compatibility path is intentionally isolated from normal selection,
    # execution, rendering, caching, and persistence.
    draft, applied, defaults = [], [], {}
    widgets = memory.values.get("widgets", {})
    try:
        definition = next(item for item in registry.definitions() if item.name == "trend")
    except StopIteration:
        return {"version": 1, "draft": draft, "applied": applied, "defaults": defaults}
    from dashboard.state.trend_settings import load_trend_settings

    loaded = load_trend_settings()
    if loaded.warning:
        st.warning(loaded.warning)
    settings = loaded.config.to_dict()
    saved = memory.values.get("applied_config")
    if isinstance(saved, dict):
        settings.update(saved)
    base_defaults = {**registry.defaults(definition.name, definition.version),
                     **registry.display_defaults(definition.name, definition.version)}
    config = _legacy_trend_config(base_defaults, settings, widgets)
    defaults[definition.name] = config
    if widgets.get("analyze-trends"):
        instance_id = "legacy-trend-analysis"
        applied.append(LogicSpec(instance_id, definition.name, definition.version, config).to_dict())
        draft_settings = dict(settings)
        for key, value in widgets.items():
            if key.startswith("_trend_setting_"):
                draft_settings[key.removeprefix("_trend_setting_")] = value
        draft_config = _legacy_trend_config(base_defaults, draft_settings, widgets)
        draft.append(LogicSpec(instance_id, definition.name, definition.version, draft_config).to_dict())
    selection = {"version": 1, "draft": draft, "applied": applied, "defaults": defaults}
    _split_saved_display(selection, memory, registry)
    return selection


def _legacy_trend_config(defaults, settings, widgets):
    """Adapt old trend settings to the plugin schema without widget coupling."""
    config = deepcopy(defaults)
    config.update({key: value for key, value in settings.items() if key in config})
    for name, old_key in (("show_redundant", "show-redundant-trends"),
                          ("show_all_lambdas", "show-all-trend-lambdas"),
                          ("slope_mode", "selected-trend-slope-mode"),
                          ("slope_min", "_trend-slope-minimum"),
                          ("slope_max", "_trend-slope-maximum")):
        if name in config and old_key in widgets:
            config[name] = widgets[old_key]
    if "selected_lambdas" in config and widgets.get("show-all-trend-lambdas") is False:
        chosen = widgets.get("selected-trend-lambdas", [])
        config["selected_lambdas"] = [value for value in chosen if isinstance(value, (int, float))]
        config["include_without_lambda"] = "No segmentation λ (tail-derived)" in chosen
    return config


def _saved_selection(memory, registry):
    if "chart_logics" not in memory.values:
        memory.values["chart_logics"] = _legacy_selection(memory, registry)
    selection = memory.values["chart_logics"]
    if (not isinstance(selection, dict) or selection.get("version") != 1
            or not isinstance(selection.get("draft"), list)
            or not isinstance(selection.get("applied"), list)):
        st.error("Saved chart logic selections are invalid. They were left unchanged.")
        return None
    from dashboard.state.migrations import (
        flat_settings_upgrade_message, upgrade_flat_settings, upgrade_trend_windows,
    )

    if upgrade_trend_windows(selection, registry):
        st.info("Trend analysis now uses overlapping windows of twice the configured history value. "
                "Your settings are preserved. Click Apply to calculate the updated analysis.")
    migration_message = flat_settings_upgrade_message(
        selection["draft"], action="Apply", saved_defaults=selection.get("defaults"))
    if upgrade_flat_settings(selection["draft"], registry, saved_defaults=selection.get("defaults")):
        st.info(migration_message)
        _clear_flat_lookback_widgets(selection["draft"], memory, "chart-logic")
    _split_saved_display(selection, memory, registry)
    return selection


def _split_saved_display(selection, memory, registry):
    """Move former mixed settings into independent per-instance display state."""
    definitions = {(item.name, item.version): item for item in registry.definitions()}
    display = selection.setdefault("display", {})
    if not isinstance(display, dict):
        st.warning("Saved chart display settings are invalid; using module defaults.")
        display = selection["display"] = {}
    # Draft is visited first so the last applied presentation wins on migration.
    for source in ("draft", "applied"):
        for saved in selection[source]:
            if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
                continue
            definition = definitions.get((saved.get("name"), saved.get("version")))
            if definition is None or not isinstance(saved.get("instance_id"), str):
                continue
            properties = (set(definition.display_schema.get("properties", {}))
                          - set(definition.schema.get("properties", {})))
            moved = {name: saved["config"].pop(name) for name in properties if name in saved["config"]}
            if not moved:
                continue
            identifier = saved["instance_id"]
            entry = display.setdefault(identifier, {"config": {}, "visible": True})
            if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
                entry = display[identifier] = {"config": {}, "visible": True}
            entry["config"].update(moved)
            # The old controls were calculation drafts; initialize new display
            # widgets from the last applied display settings instead.
            for name in moved:
                old_prefix = f"chart-logic-{identifier}-{name}"
                for key in list(memory.values["widgets"]):
                    if key == old_prefix or key.startswith(old_prefix + "-"):
                        memory.values["widgets"].pop(key, None)
                        st.session_state.pop(key, None)
    defaults = selection.get("defaults", {})
    if isinstance(defaults, dict):
        for definition in definitions.values():
            config = defaults.get(definition.name)
            if isinstance(config, dict):
                properties = (set(definition.display_schema.get("properties", {}))
                              - set(definition.schema.get("properties", {})))
                moved = {name: config.pop(name) for name in properties if name in config}
                if moved:
                    display_defaults = selection.setdefault("display_defaults", {})
                    if isinstance(display_defaults, dict):
                        current = display_defaults.get(definition.name, {})
                        display_defaults[definition.name] = {**(current if isinstance(current, dict) else {}), **moved}


def _signature(registry):
    values = [(item.name, item.version, item.schema, item.display_schema,
               item.result_version, getattr(item, "implementation_fingerprint", ""))
              for item in registry.definitions()]
    return sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def _state(memory, registry, context, selection):
    key = _STATE_PREFIX + sha256(memory.scope.encode()).hexdigest()
    signature = _signature(registry)
    state = st.session_state.get(key)
    if state is None or state["registry_signature"] != signature:
        runner = LogicRunner(registry, output_validator=validate_overlay)
        snapshot = memory.load_chart_analysis()
        compatible = snapshot is not None and snapshot.get("specs") == selection["applied"]
        run = runner.restore(snapshot, context) if compatible else None
        if snapshot is not None and run is None and snapshot.get("version") == 1:
            st.info("Saved chart results need one Apply to support live display filters. "
                    "Your selected calculations and display settings have been preserved.")
        if run is None and snapshot is None:
            from dashboard.state.migrations import restore_legacy_analysis
            run = restore_legacy_analysis(memory, selection, runner, context)
        state = {"registry_signature": signature, "runner": runner, "run": run}
        st.session_state[key] = state
    return state


def render_logic_controls(context, memory, registry=None):
    """Return applied results for this source; editing drafts never executes logic."""
    if registry is None:
        if st.button("Reload analyses", key="chart-logic-reload",
                     help="Discover private scripts and load edits. Apply to calculate updated results."):
            from research.analysis.scripting.discovery import load_registry
            registry = load_registry(reload=True)
        else:
            registry = discover_logics()
    selection = _saved_selection(memory, registry)
    st.subheader("Chart logic")
    st.caption("Add analyses and configure their settings, then click Apply to calculate them together. "
               "Edits and removals stay pending until Apply. Zooming and panning only change the view.")
    for issue in registry.issues:
        st.warning(f"Chart logic discovery: {issue}")
    if selection is None:
        return None
    state = _state(memory, registry, context, selection)
    runner = state["runner"]
    definitions = registry.definitions()
    choices = {(item.name, item.version): item for item in definitions}
    draft = deepcopy(selection["draft"])
    if choices:
        choice = memory.widget(st.selectbox, "Available chart logic", options=list(choices),
                               format_func=lambda value: choices[value].label,
                               key="chart-logic-add-choice")
        if st.button("Add logic", key="chart-logic-add"):
            definition = choices[choice]
            defaults = selection.get("defaults", {})
            if not isinstance(defaults, dict):
                st.warning("Saved chart logic defaults are invalid; using module defaults for this instance.")
                defaults = {}
            config = defaults.get(definition.name, registry.defaults(definition.name, definition.version))
            if not isinstance(config, dict):
                st.warning("Saved chart logic defaults are invalid; using module defaults for this instance.")
                config = registry.defaults(definition.name, definition.version)
            identifier = uuid4().hex
            draft.append(LogicSpec(identifier, definition.name, definition.version, deepcopy(config)).to_dict())
            display_defaults = selection.get("display_defaults", {})
            display_config = display_defaults.get(definition.name, {}) if isinstance(display_defaults, dict) else {}
            if isinstance(display_config, dict):
                selection["display"][identifier] = {"config": deepcopy(display_config), "visible": True}
    else:
        st.info("No chart logic modules are available. Add a script under secrete_analysis/analyses to make it selectable.")

    updated = []
    valid = True
    instance_ids = set()
    for index, saved in enumerate(draft):
        try:
            spec = LogicSpec.from_dict(saved)
            if spec.instance_id in instance_ids:
                raise ValueError("The same instance ID was saved more than once.")
            instance_ids.add(spec.instance_id)
        except (ValueError, TypeError, KeyError) as error:
            st.warning(f"Saved chart logic #{index + 1} is invalid: {error}")
            if not st.button("Remove invalid logic", key=f"chart-logic-remove-invalid-{index}"):
                updated.append(saved)
                valid = False
            continue
        definition = choices.get((spec.name, spec.version))
        label = definition.label if definition else f"{spec.name} (version {spec.version})"
        with memory.widget(st.expander, f"{index + 1}. {label}", expanded=True,
                           key=f"chart-logic-card-{spec.instance_id}"):
            if st.button("Remove", key=f"chart-logic-remove-{spec.instance_id}", help=f"Remove {label} when Apply is clicked"):
                continue
            if definition is None:
                st.warning("This saved chart logic module or version is unavailable. Restore the module or remove "
                           "this selection before applying. Its configuration is preserved.")
                st.json(spec.config)
                updated.append(saved)
                valid = False
                continue
            st.caption(definition.description)
            try:
                config = _configuration(definition, memory, f"chart-logic-{spec.instance_id}", spec.config)
                if spec.name == "flat":
                    # Preserve unknown restored fields so validation can report them.
                    config = {**spec.config, **config}
            except (ValueError, TypeError, OverflowError, AttributeError) as error:
                st.error(f"Invalid saved settings for {label}: {error}")
                valid = False
                updated.append(saved)
            else:
                updated.append(LogicSpec(spec.instance_id, spec.name, spec.version, config).to_dict())
    selection["draft"] = updated
    current = state["run"] if state["run"] is not None and runner.matches(state["run"], context) else None
    pending = selection["draft"] != selection["applied"]
    pending_notice = st.empty()
    status_notice = st.empty()
    if pending:
        pending_notice.caption("Unapplied changes")
    if updated and current is None:
        status_notice.info("Click Apply to calculate the selected chart logic for the current market data.")
    elif not updated and current is None:
        status_notice.caption("No chart logic selected. The original price and volume chart remains available.")
    if st.button("Apply", key="chart-logic-apply", type="primary", disabled=not valid):
        specs = [LogicSpec.from_dict(item) for item in selection["draft"]]
        indicator = st.empty()
        last_update = 0.0
        labels = {spec.instance_id: choices[(spec.name, spec.version)].label for spec in specs}

        def progress(instance_id, done, total):
            nonlocal last_update
            now = monotonic()
            if done == 0 or done == total or now - last_update >= .25:
                fraction = min(1., max(0., done / total)) if total else 0.
                indicator.progress(fraction, text=f"{labels[instance_id]}: {done:,} of {total:,}")
                last_update = now

        try:
            with st.spinner("Calculating chart logic…"):
                applied = runner.apply(specs, context, progress=progress)
        except (ValueError, TypeError) as error:
            st.error(f"Invalid chart logic settings: {error}. The last applied results are unchanged.")
        else:
            pending_notice.empty()
            status_notice.empty()
            state["run"] = current = applied
            selection["applied"] = [spec.to_dict() for spec in applied.specs]
            selection["draft"] = deepcopy(selection["applied"])
            try:
                memory.save_chart_analysis(runner.snapshot(applied))
            except (OSError, ValueError, TypeError) as error:
                st.warning(f"Results are available in this session but could not be saved for reopening: {error}")
            if applied.errors:
                st.warning(f"Applied {len(applied.results)} chart logic instance(s); {len(applied.errors)} failed. "
                           "Successful results are shown; failed results have been cleared.")
            elif specs:
                st.success(f"Applied {len(specs)} chart logic instance(s).")
            else:
                st.success("Chart logic cleared.")
        finally:
            indicator.empty()
    return current


# Historical imports remain available from dashboard.logic_controls.
def render_logic_display_controls(*args, **kwargs):
    from dashboard.market_data.display_controls import render_logic_display_controls as render
    return render(*args, **kwargs)


def render_logic_results(*args, **kwargs):
    from dashboard.market_data.results import render_logic_results as render
    return render(*args, **kwargs)
