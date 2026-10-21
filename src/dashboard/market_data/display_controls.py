"""Presentation settings for already calculated analysis results."""

from copy import deepcopy
from types import SimpleNamespace

import streamlit as st

from research.analysis import LogicRunner
from dashboard.charts.overlays import validate_overlay
from dashboard.market_data import analysis_controls as controls

def render_logic_display_controls(run, memory, registry=None):
    """Present already calculated results; display edits never enter Apply/cache."""
    if run is None or not run.specs:
        return run
    registry = registry if registry is not None else controls.discover_logics()
    selection = controls._saved_selection(memory, registry)
    if selection is None:
        return run
    instances = []
    for index, spec in enumerate(run.specs):
        result = run.results.get(spec.instance_id)
        if result is None:
            continue
        analysis = result.data.get("analysis")
        if isinstance(analysis, dict) and not analysis.get("layers"):
            continue
        instances.append((index, spec))
    if not instances:
        return run
    st.subheader("Chart display and filters")
    st.caption("These controls update the chart immediately using calculated results. "
               "Change calculation settings above and click Apply only when new results are needed.")
    definitions = {(item.name, item.version): item for item in registry.definitions()}
    configs, visibility = {}, {}
    for index, spec in instances:
        definition = definitions.get((spec.name, spec.version))
        result = run.results.get(spec.instance_id)
        if definition is None or result is None:
            continue
        prefix = f"chart-display-{spec.instance_id}"
        saved = selection["display"].get(spec.instance_id, {})
        if not isinstance(saved, dict):
            saved = {}
        entry = {"config": saved.get("config", {}), "visible": saved.get("visible", True)}
        if not isinstance(entry["config"], dict):
            entry["config"] = {}
        previous = entry["config"]
        with memory.widget(st.expander, f"{definition.label} · display ({index + 1})",
                           expanded=True, key=prefix + "-card"):
            visibility[spec.instance_id] = memory.widget(
                st.checkbox, "Show on chart", value=entry["visible"] is not False,
                key=prefix + "-visible",
            )
            try:
                schema = registry.display_schema(spec.name, spec.version, result)
                defaults = registry.display_defaults(spec.name, spec.version, result)
                values = {**defaults, **entry["config"]}
                # Saved options can disappear after new data or settings are
                # applied. Keep remaining choices without inventing new ones.
                for name, field in schema.get("properties", {}).items():
                    if not isinstance(field, dict):
                        continue
                    items = field.get("items", {})
                    options = field.get("x-ui-options", items.get("enum") if isinstance(items, dict) else None)
                    if field.get("type") == "array" and isinstance(options, list):
                        chosen = values.get(name, [])
                        values[name] = [item for item in chosen if item in options] if isinstance(chosen, list) else []
                try:
                    previous = registry.configure_display(spec.name, spec.version, values, result)
                except ValueError:
                    previous = defaults
                if schema.get("properties") or schema.get("additionalProperties") is not False:
                    config = controls._configuration(SimpleNamespace(schema=schema), memory, prefix, values)
                else:
                    config = {}
                config = registry.configure_display(spec.name, spec.version, config, result)
            except (ValueError, TypeError, OverflowError, AttributeError) as error:
                st.error(f"{definition.label}: {error}. The last valid display settings remain in use.")
                config = previous
            configs[spec.instance_id] = config
            selection["display"][spec.instance_id] = {
                "config": deepcopy(config), "visible": visibility[spec.instance_id],
            }
    return LogicRunner(registry, output_validator=validate_overlay).present(run, configs, visibility)


