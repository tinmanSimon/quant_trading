"""Recover blank saved paths without allowing overlapping data/result stores."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.state.workspace import MarketMemory


@pytest.fixture
def startup(tmp_path, monkeypatch):
    from dashboard import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "_data_page", lambda research: st.info("Dashboard ready"))
    return "from dashboard.app import main\nmain()"


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_saved_directories_restore_defaults(startup, value):
    memory = MarketMemory("dashboard-directories")
    memory.values["widgets"] = {"data-directory": value, "runs-directory": value}
    memory.save()
    app = AppTest.from_string(startup).run()
    assert not app.exception and not app.error
    assert app.text_input(key="data-directory").value == "data"
    assert app.text_input(key="runs-directory").value == "runs"
    restored = MarketMemory("dashboard-directories")
    assert restored.values["widgets"] == {"data-directory": "data", "runs-directory": "runs"}
    app.text_input(key="data-directory").set_value("").run()
    app.text_input(key="runs-directory").set_value("  ").run()
    assert not app.exception and not app.error
    assert app.text_input(key="data-directory").value == "data"
    assert app.text_input(key="runs-directory").value == "runs"


def test_explicit_custom_directories_are_preserved(startup):
    memory = MarketMemory("dashboard-directories")
    memory.values["widgets"] = {"data-directory": "custom-data", "runs-directory": "custom-results"}
    memory.save()
    app = AppTest.from_string(startup).run()
    assert not app.exception and not app.error
    assert app.text_input(key="data-directory").value == "custom-data"
    assert app.text_input(key="runs-directory").value == "custom-results"


@pytest.mark.parametrize("data,runs", [("data", "data"), ("data", "data/runs"), ("runs/data", "runs")])
def test_overlap_has_actionable_message_and_can_be_corrected(startup, data, runs):
    memory = MarketMemory("dashboard-directories")
    memory.values["widgets"] = {"data-directory": data, "runs-directory": runs}
    memory.save()
    app = AppTest.from_string(startup).run()
    assert not app.exception
    assert any("Neither folder can be inside the other" in item.value for item in app.error)
    assert not any("Dashboard ready" in item.value for item in app.info)
    app.text_input(key="data-directory").set_value("separate-data").run()
    app.text_input(key="runs-directory").set_value("separate-results").run()
    assert not app.exception and not app.error
    assert any("Dashboard ready" in item.value for item in app.info)
