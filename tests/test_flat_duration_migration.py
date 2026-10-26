"""Saved slope selections stage the band definition without rewriting results."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.state.migrations import upgrade_flat_settings
from research.analysis import LogicRegistry, LogicSpec, discover_logics
from research.analysis.builtins.flat import FlatLogic


NEW_DEFAULTS = {"band_width_mode": "fixed", "auto_percentile": 90.0, "auto_multiplier": 2.0,
                "long_bar_filter": False, "long_bar_multiplier": 2.0}


def previous_spec(minimum=20, version="1.0.0-engine.1.0.0", maximum=None):
    config = {
        "min_duration": minimum, "slope_min_pct": -.01, "slope_max_pct": .01,
    }
    if maximum is not None:
        config["max_duration"] = maximum
    return LogicSpec("saved-flat", "flat", version, config).to_dict()


@pytest.mark.parametrize("version,minimum,previous_maximum,maximum", [
    ("1.0.0-engine.1.0.0", 20, None, 200),
    ("1.0.0-engine.1.0.0", 500, None, 501),
    ("1.0.0-engine.1.1.0", 20, 80, 80),
    ("1.0.0-engine.1.1.0", 500, 700, 700),
])
def test_market_workspace_stages_band_definition_without_upgrading_applied_results(
        version, minimum, previous_maximum, maximum, monkeypatch):
    from dashboard.market_data.analysis_controls import _saved_selection
    import streamlit as st

    spec = previous_spec(minimum, version, previous_maximum)
    selection = {"version": 1, "draft": [deepcopy(spec)], "applied": [deepcopy(spec)],
                 "display": {"saved-flat": {"visible": False, "config": {"show_fit": False}}}}
    before = deepcopy(selection)
    memory = SimpleNamespace(values={"chart_logics": selection, "widgets": {}})
    registry = discover_logics()
    messages = []
    monkeypatch.setattr(st, "info", messages.append)
    staged = _saved_selection(memory, registry)
    assert staged["draft"][0]["version"] == FlatLogic.version
    assert staged["draft"][0]["config"] == {
        "min_duration": minimum, "max_duration": maximum, "max_range_pct": 2.0, **NEW_DEFAULTS,
    }
    assert staged["applied"] == before["applied"]
    assert staged["display"] == before["display"]
    assert registry.configure("flat", FlatLogic.version, staged["draft"][0]["config"])["max_duration"] == maximum
    assert any("every candle's high and low" in message and "Apply" in message for message in messages)
    assert not upgrade_flat_settings(staged["draft"], registry)


def test_fixed_band_upgrade_preserves_original_limits_and_applied_archive(monkeypatch):
    from dashboard.market_data.analysis_controls import _saved_selection
    import streamlit as st

    spec = LogicSpec("saved-flat", "flat", "1.0.0-engine.2.0.0",
                     {"min_duration": 7, "max_duration": 60, "max_range_pct": .8}).to_dict()
    selection = {"version": 1, "draft": [deepcopy(spec)], "applied": [deepcopy(spec)],
                 "display": {"saved-flat": {"visible": False, "config": {"show_fit": False}}}}
    before = deepcopy(selection)
    memory = SimpleNamespace(values={"chart_logics": selection, "widgets": {}})
    messages = []
    monkeypatch.setattr(st, "info", messages.append)
    staged = _saved_selection(memory, discover_logics())
    assert staged["draft"][0]["version"] == FlatLogic.version
    assert staged["draft"][0]["config"] == {**spec["config"], **NEW_DEFAULTS}
    assert staged["applied"] == before["applied"]
    assert staged["display"] == before["display"]
    assert any("automatic band width" in message and "Apply" in message for message in messages)
    assert all("slope" not in message for message in messages)
    assert not upgrade_flat_settings(staged["draft"], discover_logics())


@pytest.mark.parametrize("mode", ["fixed", "automatic"])
def test_lookback_upgrade_preserves_custom_settings_defaults_and_applied_results(mode, monkeypatch):
    from dashboard.market_data.analysis_controls import _saved_selection
    import streamlit as st

    config = {"min_duration": 7, "max_duration": 60, "max_range_pct": .8,
              "band_width_mode": mode, "auto_percentile": 75., "auto_multiplier": 1.5,
              "auto_lookback": 50, "long_bar_filter": True, "long_bar_multiplier": 1.25,
              "unknown_setting": 17}
    spec = LogicSpec("saved-flat", "flat", "1.0.0-engine.2.1.0", config).to_dict()
    selection = {"version": 1, "draft": [deepcopy(spec)], "applied": [deepcopy(spec)],
                 "defaults": {"flat": deepcopy(config), "other": {"auto_lookback": 123}},
                 "display": {"saved-flat": {"visible": False, "config": {"show_fit": False}}}}
    widgets = {"chart-logic-saved-flat-auto_lookback": 50,
               "chart-logic-saved-flat-auto_percentile": 75., "unrelated-auto_lookback": 123}
    before = deepcopy(selection)
    memory = SimpleNamespace(values={"chart_logics": selection, "widgets": widgets})
    messages = []
    monkeypatch.setattr(st, "info", messages.append)
    staged = _saved_selection(memory, discover_logics())
    expected_config = {key: value for key, value in config.items() if key != "auto_lookback"}
    assert staged["draft"][0]["version"] == FlatLogic.version
    assert staged["draft"][0]["config"] == expected_config
    assert staged["defaults"]["flat"] == expected_config
    assert staged["defaults"]["other"] == before["defaults"]["other"]
    assert staged["applied"] == before["applied"]
    assert staged["display"] == before["display"]
    assert widgets == {"chart-logic-saved-flat-auto_percentile": 75., "unrelated-auto_lookback": 123}
    assert any("within each candidate flat period" in message and "lookback has been removed" in message
               for message in messages)
    assert not upgrade_flat_settings(staged["draft"], discover_logics(), saved_defaults=staged["defaults"])
    with pytest.raises(ValueError, match="unknown_setting"):
        discover_logics().configure("flat", FlatLogic.version, expected_config)


def test_saved_flat_defaults_wait_for_available_upgrade_and_preserve_unrelated_values():
    defaults = {"flat": {**NEW_DEFAULTS, "auto_lookback": 100, "unknown_setting": 17},
                "other": {"auto_lookback": 50}}
    before = deepcopy(defaults)
    assert not upgrade_flat_settings([], LogicRegistry(), saved_defaults=defaults)
    assert defaults == before
    assert upgrade_flat_settings([], discover_logics(), saved_defaults=defaults)
    assert defaults["flat"] == {**NEW_DEFAULTS, "unknown_setting": 17}
    assert defaults["other"] == before["other"]


def test_upgrade_only_changes_the_known_available_flat_version():
    known = previous_spec()
    unknown = {**deepcopy(known), "version": "unknown"}
    unrelated = {**deepcopy(known), "name": "other"}
    specs = [known, unknown, unrelated]
    before = deepcopy(specs)
    assert not upgrade_flat_settings(specs, LogicRegistry())
    assert specs == before
    assert upgrade_flat_settings(specs, discover_logics())
    assert specs[1:] == before[1:]


def test_unknown_and_invalid_settings_survive_migration_for_validation():
    spec = previous_spec(version="1.0.0-engine.1.1.0", maximum=10)
    spec["config"]["unrecognized_setting"] = 17
    registry = discover_logics()
    assert upgrade_flat_settings([spec], registry)
    assert spec["config"]["unrecognized_setting"] == 17
    assert spec["config"]["max_duration"] == 10
    with pytest.raises(ValueError, match="unrecognized_setting"):
        registry.configure("flat", FlatLogic.version, spec["config"])
    del spec["config"]["unrecognized_setting"]
    with pytest.raises(ValueError, match="max_duration"):
        registry.configure("flat", FlatLogic.version, spec["config"])


def saved_batch_settings_app():
    import streamlit as st
    from dashboard.components.analysis_selection import select_analyses
    from dashboard.workspace import market_memory
    from research.analysis import discover_logics

    memory = market_memory("flat-duration-migration")
    if "analyses" not in memory.values:
        memory.values["analyses"] = [{
            "instance_id": "saved-flat", "name": "flat", "version": "1.0.0-engine.1.0.0",
            "config": {"min_duration": 20, "slope_min_pct": -.01, "slope_max_pct": .01},
        }]
    if st.session_state.get("include_unknown"):
        memory.values["analyses"][0]["config"]["unrecognized_setting"] = 17
    specs = select_analyses(discover_logics(), memory)
    st.session_state["migrated"] = [spec.to_dict() for spec in specs]


def test_batch_saved_draft_exposes_band_width_without_removing_and_readding_analysis():
    app = AppTest.from_function(saved_batch_settings_app).run()
    assert not app.exception and not app.error and not app.warning
    assert app.number_input(key="batch-analysis-saved-flat-max_duration").value == 200
    assert app.number_input(key="batch-analysis-saved-flat-max_range_pct").value == 2.0
    assert not any(item.key.endswith("-slope_min_pct") or item.key.endswith("-slope_max_pct")
                   for item in app.number_input)
    assert any("every candle's high and low" in message.value and "Run analysis" in message.value
               for message in app.info)
    migrated, = app.session_state["migrated"]
    assert migrated["version"] == FlatLogic.version
    assert migrated["config"]["max_duration"] == 200
    assert migrated["config"]["max_range_pct"] == 2.0
    assert "slope_min_pct" not in migrated["config"] and "slope_max_pct" not in migrated["config"]
    app.number_input(key="batch-analysis-saved-flat-max_duration").set_value(80).run()
    assert not app.exception and not app.error
    assert app.session_state["migrated"][0]["config"]["max_duration"] == 80


def test_batch_ui_preserves_unknown_restored_keys_until_validation():
    app = AppTest.from_function(saved_batch_settings_app)
    app.session_state["include_unknown"] = True
    app.run()
    assert not app.exception and not app.error
    migrated, = app.session_state["migrated"]
    assert migrated["config"]["unrecognized_setting"] == 17
    with pytest.raises(ValueError, match="unrecognized_setting"):
        discover_logics().configure("flat", FlatLogic.version, migrated["config"])


def saved_automatic_batch_settings_app():
    import streamlit as st
    from dashboard.components.analysis_selection import select_analyses
    from dashboard.workspace import market_memory
    from research.analysis import discover_logics

    memory = market_memory("flat-lookback-migration")
    if "analyses" not in memory.values:
        memory.values["analyses"] = [{
            "instance_id": "saved-flat", "name": "flat", "version": "1.0.0-engine.2.1.0",
            "config": {"min_duration": 3, "max_duration": 20, "max_range_pct": .8,
                       "band_width_mode": "automatic", "auto_percentile": 75., "auto_multiplier": 1.5,
                       "auto_lookback": 60, "long_bar_filter": True, "long_bar_multiplier": 1.25},
        }]
        memory.values["widgets"]["batch-analysis-saved-flat-auto_lookback"] = 60
    specs = select_analyses(discover_logics(), memory)
    st.session_state["migrated"] = [spec.to_dict() for spec in specs]
    st.session_state["saved_widgets"] = memory.values["widgets"]


def test_batch_automatic_upgrade_removes_lookback_and_preserves_custom_controls():
    app = AppTest.from_function(saved_automatic_batch_settings_app).run()
    assert not app.exception and not app.error and not app.warning
    assert app.selectbox(key="batch-analysis-saved-flat-band_width_mode").value == "automatic"
    assert app.number_input(key="batch-analysis-saved-flat-auto_percentile").value == 75.
    assert app.number_input(key="batch-analysis-saved-flat-auto_multiplier").value == 1.5
    assert app.checkbox(key="batch-analysis-saved-flat-long_bar_filter").value is True
    assert app.number_input(key="batch-analysis-saved-flat-long_bar_multiplier").value == 1.25
    assert not any(item.key.endswith("-auto_lookback") for item in app.number_input)
    assert "batch-analysis-saved-flat-auto_lookback" not in app.session_state["saved_widgets"]
    migrated, = app.session_state["migrated"]
    assert migrated["version"] == FlatLogic.version
    assert "auto_lookback" not in migrated["config"]
    assert any("lookback has been removed" in item.value for item in app.info)


def saved_automatic_defaults_app():
    from datetime import UTC, datetime, timedelta
    import polars as pl
    import streamlit as st
    from dashboard.market_data.analysis_controls import render_logic_controls
    from dashboard.workspace import market_memory
    from research.analysis import LogicContext, discover_logics

    memory = market_memory("flat-lookback-defaults")
    if "chart_logics" not in memory.values:
        memory.values["chart_logics"] = {
            "version": 1, "draft": [], "applied": [], "defaults": {
                "flat": {"min_duration": 2, "max_duration": 20, "max_range_pct": .8,
                         "band_width_mode": "automatic", "auto_percentile": 75., "auto_multiplier": 1.5,
                         "auto_lookback": 60, "long_bar_filter": True, "long_bar_multiplier": 1.25}}}
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame({"timestamp": [start + timedelta(minutes=30 * i) for i in range(3)],
                          "open": [100.] * 3, "close": [100.] * 3,
                          "high": [101.] * 3, "low": [99.] * 3})
    context = LogicContext(frame, "30m", "TEST", start + timedelta(days=1))
    run = render_logic_controls(context, memory, discover_logics())
    st.session_state["saved_selection"] = memory.values["chart_logics"]
    st.session_state["applied_run"] = run


def test_market_data_can_add_and_apply_flat_from_saved_automatic_defaults():
    app = AppTest.from_function(saved_automatic_defaults_app).run()
    assert not app.exception and not app.error
    assert "auto_lookback" not in app.session_state["saved_selection"]["defaults"]["flat"]
    app.selectbox(key="chart-logic-add-choice").select((FlatLogic.name, FlatLogic.version))
    app.button(key="chart-logic-add").click().run()
    assert not app.exception and not app.error
    assert not any(item.key.endswith("-auto_lookback") for item in app.number_input)
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error
    run = app.session_state["applied_run"]
    assert run is not None and not run.errors
    spec, = run.specs
    assert spec.config["band_width_mode"] == "automatic"
    assert spec.config["auto_percentile"] == 75.
    assert spec.config["auto_multiplier"] == 1.5
    assert spec.config["long_bar_filter"] is True
    assert spec.config["long_bar_multiplier"] == 1.25
    assert "auto_lookback" not in spec.config
