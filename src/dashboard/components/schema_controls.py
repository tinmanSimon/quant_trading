"""Schema-driven processor controls; processor names never select UI behavior."""
from datetime import date, time
import json
import math

import streamlit as st

from dashboard.components.session_schedules import CUSTOM_HOURS, REGULAR_HOURS, calendar_options, generate_schedule


_MISSING = object()


def _json_editor(label, schema, value, memory, key):
    default = ([] if schema.get("type") == "array" else {}) if value is _MISSING else value
    text = memory.widget(st.text_area, label + " (JSON)", value=json.dumps(default, indent=2),
                         key=key, help=schema.get("description"))
    try:
        return json.loads(text)
    except ValueError as error:
        raise ValueError(f"{label}: enter valid JSON ({error}).") from error


def _sessions(label, schema, value, memory, key, bounds):
    import exchange_calendars as xcals
    from zoneinfo import available_timezones

    st.markdown(f"**{label}**")
    st.caption("Generate an explicit exchange schedule, or edit its JSON. Bars must match the supplied "
               "session grid; each processor describes how it handles out-of-session input. "
               "The schedule is saved with the pipeline.")
    mode = memory.widget(st.radio, "Session hours", options=[REGULAR_HOURS, CUSTOM_HOURS],
                         key=key + "-mode", horizontal=True)
    first, last = bounds or (date.today(), date.today())
    left, right = st.columns(2)
    first = memory.widget(left.date_input, "First session date", value=first, key=key + "-first")
    last = memory.widget(right.date_input, "Last session date (inclusive)", value=last, key=key + "-last")
    with st.spinner("Checking available calendars…"):
        options, excluded = calendar_options(first, last)
    calendar = None
    timezone = None
    opening, closing, opening_day = time(4), time(20), "Session date"
    if options:
        lookup = {option.name: option for option in options}
        # Preserve old free-text calendar aliases when switching to a dropdown.
        saved = memory.values["widgets"].get(key + "-calendar")
        if saved and saved not in lookup:
            try:
                resolved = xcals.resolve_alias(saved.strip())
            except Exception:
                resolved = None
            if resolved in lookup:
                memory.values["widgets"][key + "-calendar"] = resolved
            else:
                st.warning(f"Previously selected calendar {saved!r} is unavailable for these dates. "
                           "Choose an available calendar; the existing schedule JSON is unchanged.")
        calendar = memory.widget(st.selectbox, "Exchange calendar", options=list(lookup),
                                 format_func=lambda name: lookup[name].label, key=key + "-calendar")
        st.caption("The list includes installed calendars with continuous sessions for the selected dates. "
                   "Aliases are shown together; 24/5 and 24/7 are generic weekday and everyday calendars.")
        if mode == CUSTOM_HOURS:
            default_zone = lookup[calendar].timezone
            zones = sorted(available_timezones() | {default_zone})
            timezone = memory.widget(st.selectbox, "Session timezone", options=zones,
                                     index=zones.index(default_zone), key=key + "-timezone-" + calendar)
            left, right = st.columns(2)
            opening = memory.widget(left.time_input, "Session opening time (local)", value=time(4),
                                    step=60, key=key + "-open")
            closing = memory.widget(right.time_input, "Session closing time (local)", value=time(20),
                                    step=60, key=key + "-close")
            opening_day = memory.widget(st.selectbox, "Opening day", options=["Session date", "Previous calendar day"],
                                        key=key + "-opening-day")
            st.caption("Custom hours use the calendar's trading dates and holidays, but replace its opening and "
                       "closing times, including early closes. Daylight-saving changes follow the selected timezone. "
                       "A closing time at or before the opening time means the following day; equal times mean a full day. "
                       "For overnight sessions ending on the session date, choose Previous calendar day.")
        else:
            st.caption(f"Uses {calendar}'s actual opening and closing times, including early closes. "
                       f"Exchange timezone: {lookup[calendar].timezone}.")
    else:
        st.warning("No compatible calendars are available for these dates. Adjust the dates or edit the schedule JSON.")
    if excluded:
        with st.expander("Calendars unavailable for these dates"):
            for name, reason in excluded:
                st.write(f"{name}: {reason}")
    st.caption("Click Generate session schedule to apply these settings. The JSON below is the schedule used "
               "by the processor; changing generator controls alone does not replace it. Output timestamps are UTC.")
    st.caption("These calendar choices apply to processing. Dashboard backtests still use XNYS and require matching session boundaries.")
    if st.button("Generate session schedule", key=key + "-generate", disabled=calendar is None):
        try:
            sessions = generate_schedule(calendar, first, last, mode=mode, timezone=timezone,
                                         opening=opening, closing=closing, opening_day=opening_day)
            memory.values["widgets"][key] = json.dumps(sessions, indent=2)
        except Exception as error:
            st.error(f"Could not generate sessions: {error}")
    return _json_editor(label, schema, value, memory, key)


def _numeric_kwargs(schema, value):
    """Choose renderable initial values within inclusive or exclusive bounds."""
    integer = schema.get("type") == "integer"
    cast = int if integer else float
    lower, upper = None, None
    if "minimum" in schema:
        lower = math.ceil(schema["minimum"]) if integer else float(schema["minimum"])
    if "exclusiveMinimum" in schema:
        exclusive = (math.floor(schema["exclusiveMinimum"]) + 1 if integer
                     else math.nextafter(float(schema["exclusiveMinimum"]), math.inf))
        lower = max(lower, exclusive) if lower is not None else exclusive
    if "maximum" in schema:
        upper = math.floor(schema["maximum"]) if integer else float(schema["maximum"])
    if "exclusiveMaximum" in schema:
        exclusive = (math.ceil(schema["exclusiveMaximum"]) - 1 if integer
                     else math.nextafter(float(schema["exclusiveMaximum"]), -math.inf))
        upper = min(upper, exclusive) if upper is not None else exclusive
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("The configured numeric bounds contain no valid value.")
    initial = cast(1 if value is _MISSING or value is None else value)
    if lower is not None:
        initial = max(lower, initial)
    if upper is not None:
        initial = min(upper, initial)
    kwargs = {"value": initial, "step": 1 if integer else 0.1}
    if not integer:
        kwargs["format"] = "%.12g"
    if lower is not None:
        kwargs["min_value"] = lower
    if upper is not None:
        kwargs["max_value"] = upper
    return kwargs


def _field(label, schema, value, memory, key, bounds):
    """Render common JSON Schema types; complex unions/maps remain editable JSON."""
    if value is _MISSING:
        value = schema.get("default", _MISSING)
    variants = schema.get("anyOf", schema.get("oneOf", []))
    if variants and len(variants) == 2 and any(v.get("type") == "null" for v in variants):
        enabled = memory.widget(st.checkbox, f"Set {label}", value=value is not _MISSING and value is not None, key=key + "-enabled")
        if not enabled:
            return None
        schema = {**{name: setting for name, setting in schema.items() if name not in ("anyOf", "oneOf")},
                  **next(v for v in variants if v.get("type") != "null")}
    if schema.get("format") == "trading_sessions":
        return _sessions(label, schema, value, memory, key, bounds)
    if "const" in schema:
        st.caption(f"{label}: {schema['const']}")
        return schema["const"]
    if "enum" in schema:
        options = schema["enum"]
        index = options.index(value) if value in options else 0
        return memory.widget(st.selectbox, label, options=options, index=index, key=key,
                             help=schema.get("description"))
    kind = schema.get("type")
    if kind == "boolean":
        return memory.widget(st.checkbox, label, value=bool(value) if value is not _MISSING else False, key=key, help=schema.get("description"))
    if kind in ("integer", "number"):
        return memory.widget(st.number_input, label, key=key, help=schema.get("description"),
                             **_numeric_kwargs(schema, value))
    if kind == "null":
        st.caption(f"{label}: null")
        return None
    if kind == "string":
        return memory.widget(st.text_input, label, value="" if value is _MISSING or value is None else str(value), key=key,
                             help=schema.get("description") or (
                                 f"ISO {schema['format']}" if schema.get("format") else None))
    if kind == "object" and schema.get("properties") and schema.get("additionalProperties") is False:
        st.markdown(f"**{label}**")
        return _object(schema, {} if value is _MISSING or value is None else value, memory, key, bounds)
    return _json_editor(label, schema, value, memory, key)


def _object(schema, values, memory, prefix, bounds):
    result = {}
    errors = []
    for name, field in schema.get("properties", {}).items():
        label = field.get("title", name.replace("_", " ").capitalize())
        key = prefix + "-" + name
        # Omitted optional values preserve constructor defaults, even when they have no schema default.
        if name not in schema.get("required", []) and "default" not in field:
            enabled = memory.widget(st.checkbox, f"Configure {label}", value=name in values, key=key + "-configure")
            if not enabled:
                continue
        try:
            result[name] = _field(label, field, values.get(name, _MISSING), memory, key, bounds)
        except ValueError as error:
            st.error(str(error))
            errors.append(str(error))
    if errors:
        raise ValueError("Fix the highlighted configuration fields.")
    return result


def processor_controls(definition, memory, prefix, values=None, bounds=None):
    schema = definition.schema
    if not schema.get("properties") and schema.get("additionalProperties") is False:
        st.info("This processor doesn’t have any configurable variables.")
        return {}
    if schema.get("additionalProperties") is not False:
        return _json_editor("Configuration", schema, values or {}, memory, prefix + "-json")
    return _object(schema, values or {}, memory, prefix, bounds)
