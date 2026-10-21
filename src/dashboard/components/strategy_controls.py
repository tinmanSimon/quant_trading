"""Configuration controls built from registry descriptions, including private code."""

import json
import math

import streamlit as st


def _configuration(entry):
    key = json.dumps(entry.key)
    defaults = entry.default_config
    fields = entry.parameters
    structured = fields is not None and set(fields) == set(defaults) and all(
        field["type"] != "json" for field in fields.values())
    # JavaScript number inputs cannot faithfully represent larger integers.
    if structured:
        structured = all(not (field["type"] in ("integer", "number") and any(
            type(value) is int and abs(value) > 2**53 - 1 for value in
            [defaults[name]] + [field[k] for k in ("minimum", "maximum", "step") if k in field]))
            for name, field in fields.items())
    use_json = not structured
    if structured:
        use_json = st.checkbox("Edit configuration as JSON", key=f"strategy-json-mode-{key}")
    if use_json:
        raw = st.text_area("Configuration (JSON)", json.dumps(defaults, indent=2), key=f"strategy-json-{key}")
        try:
            return entry.validate_config(json.loads(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{entry.label}: invalid configuration: {exc}") from exc
    config = {}
    for name, field in fields.items():
        label = field.get("label", name)
        kwargs = {"key": f"strategy-param-{key}-{name}", "help": field.get("help")}
        value = defaults[name]
        if "choices" in field:
            choices = field["choices"]
            index = next(i for i, choice in enumerate(choices) if type(choice) is type(value) and choice == value)
            config[name] = st.selectbox(label, choices, index=index, **kwargs)
        elif field["type"] == "boolean":
            config[name] = st.checkbox(label, value=value, **kwargs)
        elif field["type"] == "string":
            config[name] = st.text_input(label, value=value, **kwargs)
        else:
            number = int if field["type"] == "integer" else float
            bounds = {target: number(field[source]) for source, target in
                      (("minimum", "min_value"), ("maximum", "max_value")) if source in field}
            config[name] = st.number_input(label, value=number(value),
                                          step=number(field.get("step", 1 if number is int else 0.01)),
                                          **bounds, **kwargs)
    return entry.validate_config(config)


def strategy_specs(registry):
    """Return selected specifications, or None when user input is invalid."""
    entries = {entry.key: entry for entry in registry.definitions()}
    if not entries:
        st.info("No strategies are registered.")
        return []
    selected = st.multiselect("Strategies", list(entries), default=[next(iter(entries))],
                             format_func=lambda key: f"{entries[key].label} (v{key[1]}) · {key[0]}")
    specs = []
    invalid = False
    for key in selected:
        entry = entries[key]
        st.markdown(f"**{entry.label} (v{entry.version})**")
        if entry.description:
            st.caption(entry.description)
        try:
            config = _configuration(entry)
            specs.append({"name": entry.name, "version": entry.version, "config": config})
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            invalid = True
    combine = st.checkbox("Also test a weighted combination", disabled=len(selected) < 2 or ("weighted", "1") not in entries)
    if combine and len(selected) > 1 and not invalid:
        st.caption("Weights are relative and normalized to sum to one. The combination uses one account.")
        values = [st.number_input(f"Weight: {entries[key].label} (v{key[1]})", min_value=0.0,
                                  max_value=1.0, value=1.0, step=0.1, key=f"strategy-weight-{json.dumps(key)}")
                  for key in selected]
        total = math.fsum(values)
        if total <= 0:
            st.error("At least one combination weight must be positive.")
            invalid = True
        else:
            specs.append({"name": "weighted", "version": "1", "config": {
                "strategies": list(specs), "weights": [value / total for value in values],
            }})
    return None if invalid else specs
