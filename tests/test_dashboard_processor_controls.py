"""Schema-driven processor forms remain usable as processors add new settings."""

import json
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline.processing import discover_processors


def object_schema(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


def controls_app(tmp_path, schema, values=None):
    script = f'''
import json
import streamlit as st
from pathlib import Path
from types import SimpleNamespace
from dashboard.processor_controls import processor_controls
from dashboard.workspace import MarketMemory
schema = json.loads({json.dumps(schema)!r})
values = json.loads({json.dumps(values)!r})
if "memory" not in st.session_state:
    st.session_state["memory"] = MarketMemory("test-controls", Path({str(tmp_path)!r}))
memory = st.session_state["memory"]
st.session_state["settings"] = None
try:
    st.session_state["settings"] = processor_controls(SimpleNamespace(schema=schema), memory, "custom", values=values)
except ValueError as error:
    st.warning(str(error))
memory.save()
'''
    return AppTest.from_string(script, default_timeout=15).run(), script


def test_processor_without_settings_displays_the_requested_message(tmp_path):
    app, _ = controls_app(tmp_path, object_schema({}))
    assert not app.exception
    assert app.session_state["settings"] == {}
    assert [element.value for element in app.info] == ["This processor doesn’t have any configurable variables."]
    assert not app.number_input and not app.text_input and not app.checkbox


def test_primitive_nested_array_and_map_settings_are_editable_and_sticky(tmp_path):
    schema = object_schema({
        "count": {"type": "integer", "default": 2, "minimum": 1},
        "factor": {"type": "number", "default": 0.5},
        "enabled": {"type": "boolean", "default": False},
        "label": {"type": "string", "default": "initial"},
        "mode": {"type": "string", "enum": ["first", "last"], "default": "last"},
        "window": {**object_schema({"size": {"type": "integer", "default": 3}}), "default": {}},
        "weights": {"type": "array", "items": {"type": "number"}, "default": [1.0]},
        "mapping": {"type": "object", "additionalProperties": {"type": "integer"}, "default": {}},
        "constant": {"const": "fixed"},
    })
    app, script = controls_app(tmp_path, schema)
    assert not app.exception and not app.error
    app.number_input(key="custom-count").set_value(5)
    app.number_input(key="custom-factor").set_value(0.00001234)
    app.checkbox(key="custom-enabled").check()
    app.text_input(key="custom-label").set_value("edited")
    app.selectbox(key="custom-mode").select("first")
    app.number_input(key="custom-window-size").set_value(8)
    app.text_area(key="custom-weights").set_value("[2, 3.5]")
    app.text_area(key="custom-mapping").set_value('{"first": 4}')
    app.run()
    expected = {"count": 5, "factor": 0.00001234, "enabled": True, "label": "edited", "mode": "first",
                "window": {"size": 8}, "weights": [2, 3.5], "mapping": {"first": 4}, "constant": "fixed"}
    assert not app.exception and app.session_state["settings"] == expected
    restored = AppTest.from_string(script, default_timeout=15).run()
    assert not restored.exception and restored.session_state["settings"] == expected


def test_explicit_none_does_not_get_replaced_by_a_nonnull_default(tmp_path):
    schema = object_schema({"maybe": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 7}})
    app, script = controls_app(tmp_path, schema, {"maybe": None})
    assert not app.exception
    assert app.checkbox(key="custom-maybe-enabled").value is False
    assert app.session_state["settings"] == {"maybe": None}
    app.checkbox(key="custom-maybe-enabled").check().run()
    app.number_input(key="custom-maybe").set_value(4).run()
    assert app.session_state["settings"] == {"maybe": 4}
    app.checkbox(key="custom-maybe-enabled").uncheck().run()
    restored = AppTest.from_string(script, default_timeout=15).run()
    assert restored.session_state["settings"] == {"maybe": None}


def test_absent_nullable_setting_uses_its_nonnull_default(tmp_path):
    schema = object_schema({"maybe": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 7}})
    app, _ = controls_app(tmp_path, schema)
    assert not app.exception
    assert app.checkbox(key="custom-maybe-enabled").value is True
    assert app.session_state["settings"] == {"maybe": 7}


def test_optional_settings_without_defaults_are_omitted_until_configured(tmp_path):
    schema = object_schema({"note": {"type": "string"}}, required=[])
    app, _ = controls_app(tmp_path, schema)
    assert app.session_state["settings"] == {}
    app.checkbox(key="custom-note-configure").check().run()
    app.text_input(key="custom-note").set_value("new note").run()
    assert app.session_state["settings"] == {"note": "new note"}


@pytest.mark.parametrize("field,expected", [
    ({"type": "number", "maximum": -0.5}, -0.5),
    ({"type": "integer", "maximum": -0.5}, -1),
    ({"type": "integer", "minimum": 2.5}, 3),
    ({"type": "integer", "exclusiveMinimum": 3}, 4),
    ({"type": "integer", "exclusiveMaximum": 0}, -1),
    ({"type": "number", "minimum": 0.2, "maximum": 0.3}, 0.3),
])
def test_numeric_defaults_fit_declared_bounds(tmp_path, field, expected):
    app, _ = controls_app(tmp_path, object_schema({"number": field}))
    assert not app.exception and not app.error
    assert app.session_state["settings"] == {"number": expected}


def test_exclusive_decimal_bounds_and_unsatisfiable_constraints(tmp_path):
    app, _ = controls_app(tmp_path / "valid", object_schema({
        "number": {"type": "number", "exclusiveMinimum": 2.0, "exclusiveMaximum": 3.0},
    }))
    assert not app.exception and not app.error
    assert 2.0 < app.session_state["settings"]["number"] < 3.0
    invalid, _ = controls_app(tmp_path / "invalid", object_schema({
        "number": {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
    }))
    assert not invalid.exception
    assert any("no valid value" in message.value for message in invalid.error)
    assert invalid.session_state["settings"] is None


def test_complex_union_retains_explicit_json_null_and_reports_malformed_json(tmp_path):
    schema = object_schema({"choice": {"anyOf": [{"type": "null"}, {"type": "integer"}, {"type": "string"}]}})
    app, _ = controls_app(tmp_path, schema, {"choice": None})
    assert not app.exception
    assert app.text_area(key="custom-choice").value == "null"
    assert app.session_state["settings"] == {"choice": None}
    app.text_area(key="custom-choice").set_value("[").run()
    assert not app.exception and app.session_state["settings"] is None
    assert any("enter valid JSON" in message.value for message in app.error)


def test_typed_iso_dates_times_and_datetimes_preserve_offsets(tmp_path):
    settings = {"day": "2024-01-02", "time": "09:30:00+08:00", "moment": "2024-01-02T09:30:00+08:00"}
    schema = object_schema({name: {"type": "string", "format": kind}
                            for name, kind in (("day", "date"), ("time", "time"), ("moment", "date-time"))})
    app, _ = controls_app(tmp_path, schema, settings)
    assert not app.exception and app.session_state["settings"] == settings
    app.text_input(key="custom-time").set_value("10:00:00+08:00").run()
    assert app.session_state["settings"]["time"] == "10:00:00+08:00"


def test_actual_builtin_schemas_render_and_validate(tmp_path):
    registry = discover_processors()
    for definition in registry.definitions():
        settings = {"factor": 2.0} if definition.name == "scale_prices" else {
            "sessions": [{"label": "2024-01-02", "open": "2024-01-02T14:30:00+00:00", "close": "2024-01-02T21:00:00+00:00"}],
            "incomplete": "raise", "target_timeframe": "1d",
        }
        app, _ = controls_app(tmp_path / definition.name, definition.schema, settings)
        assert not app.exception and not app.error
        assert registry.configure(definition.name, definition.version, app.session_state["settings"]) == settings


def test_session_generation_supports_a_single_trading_day_and_reports_holidays(tmp_path):
    definition = discover_processors().definition("resample_ohlcv", "1")
    app, _ = controls_app(tmp_path, definition.schema)
    app.date_input(key="custom-sessions-first").set_value(date(2024, 1, 2))
    app.date_input(key="custom-sessions-last").set_value(date(2024, 1, 2))
    app.button(key="custom-sessions-generate").click().run()
    assert not app.exception and not app.error
    sessions = app.session_state["settings"]["sessions"]
    assert sessions == [{"label": "2024-01-02", "open": "2024-01-02T14:30:00+00:00", "close": "2024-01-02T21:00:00+00:00"}]
    app.date_input(key="custom-sessions-first").set_value(date(2024, 1, 1))
    app.date_input(key="custom-sessions-last").set_value(date(2024, 1, 1))
    app.button(key="custom-sessions-generate").click().run()
    assert not app.exception
    assert any("No trading sessions" in error.value for error in app.error)
    assert app.session_state["settings"]["sessions"] == sessions


def test_calendar_dropdown_custom_hours_and_new_session_restore(tmp_path):
    from datetime import time
    definition = discover_processors().definition("resample_ohlcv", "1")
    app, script = controls_app(tmp_path, definition.schema)
    assert not app.exception and not app.error
    assert app.selectbox(key="custom-sessions-calendar").value == "XNYS"
    assert any(option.startswith("XLON ·") for option in app.selectbox(key="custom-sessions-calendar").options)
    assert not any(option.startswith("XHKG ·") for option in app.selectbox(key="custom-sessions-calendar").options)
    assert not any(field.label == "Exchange calendar" for field in app.text_input)
    app.radio(key="custom-sessions-mode").set_value("Custom session hours").run()
    assert app.selectbox(key="custom-sessions-timezone-XNYS").value == "America/New_York"
    app.date_input(key="custom-sessions-first").set_value(date(2024, 10, 11))
    app.date_input(key="custom-sessions-last").set_value(date(2024, 10, 11))
    app.time_input(key="custom-sessions-open").set_value(time(4))
    app.time_input(key="custom-sessions-close").set_value(time(20))
    app.button(key="custom-sessions-generate").click().run()
    assert not app.exception and not app.error
    expected = [{"label": "2024-10-11", "open": "2024-10-11T08:00:00+00:00", "close": "2024-10-12T00:00:00+00:00"}]
    assert app.session_state["settings"]["sessions"] == expected
    restored = AppTest.from_string(script, default_timeout=15).run()
    assert restored.radio(key="custom-sessions-mode").value == "Custom session hours"
    assert restored.time_input(key="custom-sessions-open").value == time(4)
    assert restored.time_input(key="custom-sessions-close").value == time(20)
    assert restored.session_state["settings"]["sessions"] == expected
    restored.radio(key="custom-sessions-mode").set_value("Exchange regular hours").run()
    # Mode controls are drafts until the user explicitly generates a schedule.
    assert restored.session_state["settings"]["sessions"] == expected
    restored.button(key="custom-sessions-generate").click().run()
    assert restored.session_state["settings"]["sessions"] == [{
        "label": "2024-10-11", "open": "2024-10-11T13:30:00+00:00", "close": "2024-10-11T20:00:00+00:00",
    }]
    restored.radio(key="custom-sessions-mode").set_value("Custom session hours").run()
    assert restored.time_input(key="custom-sessions-open").value == time(4)
    restored.selectbox(key="custom-sessions-calendar").set_value("XLON").run()
    assert restored.selectbox(key="custom-sessions-timezone-XLON").value == "Europe/London"


def test_existing_calendar_alias_is_restored_to_canonical_dropdown_option(tmp_path):
    from dashboard.workspace import MarketMemory
    memory = MarketMemory("test-controls", tmp_path)
    memory.values["widgets"].update({"custom-sessions-calendar": "XNAS",
                                      "custom-sessions-first": date(2024, 10, 11),
                                      "custom-sessions-last": date(2024, 10, 11)})
    memory.save()
    app, _ = controls_app(tmp_path, discover_processors().definition("resample_ohlcv", "1").schema)
    assert not app.exception and not app.error
    assert app.selectbox(key="custom-sessions-calendar").value == "XNYS"
