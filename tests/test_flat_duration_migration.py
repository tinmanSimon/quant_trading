"""Saved slope selections stage the band definition without rewriting results."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.state.migrations import upgrade_flat_settings
from research.analysis import LogicRegistry, LogicSpec, discover_logics
from research.analysis.builtins.flat import FlatLogic


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
        "min_duration": minimum, "max_duration": maximum, "max_range_pct": 2.0,
    }
    assert staged["applied"] == before["applied"]
    assert staged["display"] == before["display"]
    assert registry.configure("flat", FlatLogic.version, staged["draft"][0]["config"])["max_duration"] == maximum
    assert any("every candle's high and low" in message and "Apply" in message for message in messages)
    assert not upgrade_flat_settings(staged["draft"], registry)


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
