"""Shared schema-driven configuration for built-in and private analyses."""

from copy import deepcopy
from uuid import uuid4

import streamlit as st

from research.analysis import LogicSpec
from dashboard.components.schema_controls import _MISSING, _field, _json_editor


def _needs_json_controls(schema):
    # Boolean property schemas are valid JSON Schema, but have no field UI.
    if not isinstance(schema, dict):
        return True
    return (any(_needs_json_controls(value) for value in schema.get("properties", {}).values())
            or any(_needs_json_controls(value) for key in ("anyOf", "oneOf")
                   for value in schema.get(key, [])))


def _configuration(definition, memory, prefix, values):
    schema = definition.schema
    properties = schema.get("properties", {})
    groups = schema.get("x-ui-groups", [])
    if not isinstance(groups, list):
        raise ValueError("Chart logic x-ui-groups must be a list.")
    assigned = set()
    for group in groups:
        if (not isinstance(group, dict) or not isinstance(group.get("title"), str)
                or not group["title"].strip() or not isinstance(group.get("fields"), list)
                or not all(isinstance(name, str) for name in group["fields"])
                or type(group.get("expanded", False)) is not bool):
            raise ValueError("Each chart settings group requires a title, a list of field names, "
                             "and an optional boolean expanded value.")
        for name in group["fields"]:
            if name not in properties:
                raise ValueError(f"Chart settings group {group['title']!r} references unknown field {name!r}.")
            if name in assigned:
                raise ValueError(f"Chart setting {name!r} appears in more than one group.")
            assigned.add(name)
    if not properties and schema.get("additionalProperties") is False:
        st.caption("This logic has no configurable settings.")
        return {}
    if schema.get("additionalProperties") is not False or _needs_json_controls(schema):
        return _json_editor("Configuration", schema, values, memory, prefix + "-json")
    if not groups:
        return _settings_object(schema, values, memory, prefix)

    result = {}
    for index, group in enumerate(groups):
        if not group["fields"]:
            continue
        subset = {**schema, "properties": {name: properties[name] for name in group["fields"]}}
        with memory.widget(st.expander, group["title"], expanded=group.get("expanded", False),
                           key=f"{prefix}-group-{index}"):
            result.update(_settings_object(subset, values, memory, prefix))
    ungrouped = {name: field for name, field in properties.items() if name not in assigned}
    if ungrouped:
        st.markdown("**Other settings**")
        result.update(_settings_object({**schema, "properties": ungrouped}, values, memory, prefix))
    return result


def _settings_object(schema, values, memory, prefix):
    """Render plugin fields, including searchable result-derived choices."""
    result, errors = {}, []
    for name, field in schema.get("properties", {}).items():
        label = field.get("title", name.replace("_", " ").capitalize())
        key = prefix + "-" + name
        if name not in schema.get("required", []) and "default" not in field:
            enabled = memory.widget(st.checkbox, f"Configure {label}", value=name in values,
                                    key=key + "-configure")
            if not enabled:
                continue
        value = values.get(name, field.get("default", _MISSING))
        try:
            items = field.get("items", {})
            options = field.get("x-ui-options", items.get("enum") if isinstance(items, dict) else None)
            if field.get("type") == "array" and isinstance(options, list):
                chosen = memory.values["widgets"].get(key, value)
                chosen = chosen if isinstance(chosen, list) else []
                valid = [item for item in chosen if item in options]
                if valid != chosen:
                    st.caption(f"{label}: removed selections that are unavailable in these results.")
                if key in memory.values["widgets"]:
                    memory.values["widgets"][key] = valid
                if not options:
                    st.caption(f"{label}: no choices are available in these results.")
                result[name] = memory.widget(
                    st.multiselect, label, options=options, default=valid, key=key,
                    help=field.get("description"), disabled=not options,
                )
            elif (field.get("type") == "object" and field.get("properties")
                  and field.get("additionalProperties") is False):
                if value is not _MISSING and value is not None and not isinstance(value, dict):
                    raise ValueError(f"{label}: settings must be an object.")
                st.markdown(f"**{label}**")
                result[name] = _settings_object(
                    field, value if isinstance(value, dict) else {}, memory, key)
            else:
                result[name] = _field(label, field, value, memory, key, None)
        except (ValueError, TypeError) as error:
            st.error(str(error))
            errors.append(str(error))
    if errors:
        raise ValueError("Fix the highlighted configuration fields.")
    return result


def select_analyses(registry, memory, *, prefix="batch-analysis"):
    """Edit independent instances without executing any analysis."""
    choices = {(item.name, item.version): item for item in registry.definitions()}
    draft = deepcopy(memory.values.get("analyses", []))
    for issue in registry.issues:
        st.warning(f"Analysis discovery: {issue}")
    if choices:
        chosen = memory.widget(st.selectbox, "Available analysis", options=list(choices),
                               format_func=lambda value: choices[value].label,
                               key=prefix + "-choice")
        if st.button("Add analysis", key=prefix + "-add"):
            name, version = chosen
            draft.append(LogicSpec(uuid4().hex, name, version,
                                   registry.defaults(name, version)).to_dict())
    else:
        st.info("No analyses are available. Add a decorated script under secrete_analysis/analyses.")
    updated, valid = [], True
    for index, saved in enumerate(draft):
        try:
            spec = LogicSpec.from_dict(saved)
        except (ValueError, TypeError, KeyError) as error:
            st.warning(f"Saved analysis {index + 1} is invalid: {error}")
            if not st.button("Remove invalid analysis", key=f"{prefix}-invalid-{index}"):
                updated.append(saved)
                valid = False
            continue
        definition = choices.get((spec.name, spec.version))
        label = definition.label if definition else spec.name
        with memory.widget(st.expander, f"{index + 1}. {label}", expanded=True,
                           key=f"{prefix}-card-{spec.instance_id}"):
            if st.button("Remove analysis", key=f"{prefix}-remove-{spec.instance_id}"):
                continue
            if definition is None:
                st.warning("This analysis version is unavailable. Reload or remove it before running.")
                st.json(spec.config)
                updated.append(saved)
                valid = False
                continue
            st.caption(definition.description)
            try:
                config = _configuration(definition, memory, f"{prefix}-{spec.instance_id}", spec.config)
                updated.append(LogicSpec(spec.instance_id, spec.name, spec.version, config).to_dict())
            except (ValueError, TypeError, OverflowError, AttributeError) as error:
                st.error(f"Invalid settings for {label}: {error}")
                updated.append(saved)
                valid = False
    memory.values["analyses"] = updated
    return [LogicSpec.from_dict(item) for item in updated] if valid else None


