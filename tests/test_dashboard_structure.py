"""The reorganized dashboard preserves existing imports and saved locations."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(("old", "new"), [
    ("dashboard.processing", "dashboard.pages.processing"),
    ("dashboard.deletion", "dashboard.pages.deletion"),
    ("dashboard.processor_controls", "dashboard.components.schema_controls"),
    ("dashboard.strategy_controls", "dashboard.components.strategy_controls"),
    ("dashboard.session_schedules", "dashboard.components.session_schedules"),
    ("dashboard.workspace", "dashboard.state.workspace"),
    ("dashboard.logic_migration", "dashboard.state.migrations"),
    ("dashboard.trend_settings", "dashboard.state.trend_settings"),
    ("dashboard.trends", "dashboard.market_data.legacy_trends"),
    ("dashboard.logic_controls", "dashboard.market_data.analysis_controls"),
    ("dashboard.trend_input", "research.analysis.input"),
    ("dashboard.trend_selection", "research.analysis.trend.selection"),
])
def test_legacy_imports_reference_the_canonical_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_default_trend_settings_stay_at_the_existing_project_location():
    from dashboard.state import trend_settings
    project = Path(__file__).resolve().parents[1]
    assert trend_settings.DEFAULT_SETTINGS_PATH == project / ".local" / "trend_config.json"


@pytest.mark.parametrize(("rows", "expected"), [
    ([0], "event-on-this-page"), ([1], None), ([2], None), ([-1], None), ([True], None), ([], None),
])
def test_result_table_selection_uses_only_the_current_page_events(monkeypatch, rows, expected):
    from types import SimpleNamespace
    from dashboard.market_data import results

    calls = []
    monkeypatch.setattr(results, "st", SimpleNamespace(session_state={"table": {"selection": {"rows": rows}}}))
    monkeypatch.setattr(results, "request_analysis_event", lambda *args, **kwargs: calls.append((args, kwargs)))
    results._selected_finding("table", "instance", ["event-on-this-page", None], "source", "result")
    assert calls == ([(('instance', expected), {"source_key": "source", "result_key": "result"})]
                     if expected else [])
