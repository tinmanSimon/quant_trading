"""Market data restores widgets and validated results after navigation and restarts."""

from datetime import UTC, date, datetime, timedelta
import math
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.workspace import MarketMemory
from data_pipeline import DataRequest
from research import Research


def test_workspace_round_trip_types_scopes_and_unchanged_writes(tmp_path):
    memory = MarketMemory("one", tmp_path)
    memory.values["widgets"] = {"date": date(2024, 1, 2), "series": ("yahoo", "1h", ""),
                                "lambdas": [1., 2.], "enabled": True, "bounds": -.001}
    memory.save()
    assert memory.warning is None
    modified = memory.path.stat().st_mtime_ns
    memory.save()
    assert memory.path.stat().st_mtime_ns == modified
    restored = MarketMemory("one", tmp_path)
    assert restored.values == memory.values
    assert MarketMemory("two", tmp_path).values == {"widgets": {}}


@pytest.mark.parametrize("document", ["{broken", "[]", '{"version": 999}',
                                      '{"version": 1, "scope": "one", "values": []}'])
def test_bad_workspace_does_not_crash_or_overwrite_file(tmp_path, document):
    memory = MarketMemory("one", tmp_path)
    memory.path.write_text(document)
    restored = MarketMemory("one", tmp_path)
    assert restored.warning
    restored.save()
    assert restored.path.read_text() == document


def test_workspace_save_failure_keeps_session_values(tmp_path, monkeypatch):
    from dashboard import workspace
    memory = MarketMemory("one", tmp_path)
    memory.values["widgets"]["symbol"] = "TEST"

    def fail(*args):
        raise OSError("read-only")

    monkeypatch.setattr(workspace, "_atomic_write", fail)
    memory.save()
    assert "Could not save" in memory.warning
    assert memory.values["widgets"]["symbol"] == "TEST"


@pytest.fixture
def market(tmp_path, monkeypatch):
    from dashboard import app as dashboard, trend_settings
    from chart_logics.plugins import trend as trends
    research = Research(tmp_path / "data", tmp_path / "runs")
    for symbol, timeframe, count in (("AAA", "1d", 30), ("TEST", "1d", 30), ("TEST", "1h", 800)):
        delta = timedelta(hours=1) if timeframe == "1h" else timedelta(days=1)
        stamps = [datetime(2024, 1, 1, tzinfo=UTC) + i * delta for i in range(count)]
        prices = [100 * math.exp(i * .001) for i in range(count)]
        frame = pl.DataFrame({"timestamp": stamps, "symbol": [symbol] * count,
                              "open": prices, "close": prices, "high": [p * 1.01 for p in prices],
                              "low": [p * .99 for p in prices], "volume": [100.] * count})
        research.pipeline.ingest_frame(DataRequest(symbol, stamps[0], stamps[-1] + delta,
                                                  timeframe=timeframe), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", tmp_path / "trends.json")
    calls = []
    analyze = trends.analyze

    def capture(*args, **kwargs):
        calls.append(kwargs["config"])
        return analyze(*args, **kwargs)

    monkeypatch.setattr(trends, "analyze", capture)
    script = "from dashboard.app import main\nmain()"
    return research, calls, script


def _table(app):
    return next(table.value for table in app.dataframe if "trend_id" in table.value.columns)


def _apply(app):
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error


def _add_trend(app):
    from chart_logics.plugins.trend import TrendLogic
    app.selectbox(key="chart-logic-add-choice").set_value(("trend", TrendLogic.version)).run()
    app.button(key="chart-logic-add").click().run()
    instance_id = next(button.key.removeprefix("chart-logic-remove-")
                       for button in app.button if button.key.startswith("chart-logic-remove-"))
    return "chart-logic-" + instance_id + "-"


def test_market_navigation_and_new_session_restore_exact_workspace(market, monkeypatch):
    research, calls, script = market
    app = AppTest.from_string(script, default_timeout=30).run()
    app.text_input(key="data-directory").set_value(str(research.pipeline.store.data_dir)).run()
    app.text_input(key="browse-search").set_value("TEST").run()
    app.selectbox(key="browse-series").set_value(("yahoo", "1h", "")).run()
    app.checkbox(key="browse-history").check().run()
    app.radio(key="browse-read").set_value("One exact revision").run()
    app.date_input[0].set_value(date(2024, 1, 2))
    app.date_input[1].set_value(date(2024, 1, 30)).run()
    app.selectbox(key="browse-timezone").set_value("Asia/Shanghai").run()
    prefix = _add_trend(app)
    app.number_input(key=prefix + "max_history").set_value(16)
    app.number_input(key=prefix + "penalty_grid_levels").set_value(4)
    _apply(app)
    display = prefix.replace("chart-logic-", "chart-display-")
    app.checkbox(key=display + "show_redundant").check()
    app.checkbox(key=display + "show_all_lambdas").uncheck()
    app.multiselect(key=display + "selected_lambdas").set_value([1.0])
    app.selectbox(key=display + "slope_mode").set_value("range")
    app.number_input(key=display + "slope_min").set_value(0.)
    app.number_input(key=display + "slope_max").set_value(.01).run()
    app.checkbox(key="browse-show-original-bars").check().run()
    next(item for item in app.number_input if item.label == "Original bar page").set_value(1).run()
    expected = _table(app)
    assert expected["on_chart"].any()
    assert len(calls) == 1
    # Draft edits are saved immediately but do not change the applied result.
    app.number_input(key=prefix + "max_history").set_value(24).run()
    assert len(calls) == 1
    app.sidebar.radio[0].set_value("Saved runs").run()
    app.run()  # Allow Streamlit to clean up every absent widget.
    app.sidebar.radio[0].set_value("Market data").run()

    def verify(restored):
        assert not restored.exception and not restored.error
        assert restored.text_input(key="data-directory").value == str(research.pipeline.store.data_dir)
        assert restored.text_input(key="browse-search").value == "TEST"
        assert restored.selectbox(key="browse-symbol").value == "TEST"
        assert restored.selectbox(key="browse-series").value == ("yahoo", "1h", "")
        assert restored.checkbox(key="browse-history").value
        assert restored.radio(key="browse-read").value == "One exact revision"
        assert [item.value for item in restored.date_input] == [date(2024, 1, 2), date(2024, 1, 30)]
        assert restored.selectbox(key="browse-timezone").value == "Asia/Shanghai"
        assert restored.number_input(key=prefix + "max_history").value == 24
        saved = MarketMemory(str(research.pipeline.store.data_dir.resolve())).values["chart_logics"]
        assert saved["applied"][0]["config"]["max_history"] == 16
        assert saved["draft"][0]["config"]["max_history"] == 24
        assert restored.checkbox(key=display + "show_redundant").value
        assert restored.multiselect(key=display + "selected_lambdas").value == [1.0]
        assert restored.selectbox(key=display + "slope_mode").value == "range"
        assert restored.number_input(key=display + "slope_min").value == 0.
        assert restored.number_input(key=display + "slope_max").value == .01
        assert next(item for item in restored.number_input if item.label == "Original bar page").value == 1
        assert _table(restored).equals(expected)
        assert len(calls) == 1

    verify(app)
    fresh = AppTest.from_string(script, default_timeout=30).run()
    verify(fresh)
    # Even unchanged metadata cannot make a snapshot match different source bars.
    original_read = research.pipeline.read_dataset
    with monkeypatch.context() as patch:
        patch.setattr(research.pipeline, "read_dataset", lambda *args, **kwargs:
                      original_read(*args, **kwargs).with_columns(pl.col("close") + .01))
        changed = AppTest.from_string(script, default_timeout=30).run()
        assert not changed.exception and not changed.error
        assert not any("trend_id" in table.value.columns for table in changed.dataframe)
        assert len(calls) == 1
    # Changed date windows cannot restore a result computed for other bars.
    fresh.date_input[0].set_value(date(2024, 1, 3)).run()
    assert not fresh.exception and not fresh.error
    assert not any("trend_id" in table.value.columns for table in fresh.dataframe)
    assert any("Click Apply" in item.value for item in fresh.info)
    assert len(calls) == 1


def test_missing_selection_falls_back_and_corrupt_analysis_requires_apply(market, monkeypatch):
    research, calls, script = market
    app = AppTest.from_string(script).run()
    app.selectbox(key="browse-symbol").set_value("TEST").run()
    app.selectbox(key="browse-series").set_value(("yahoo", "1h", "")).run()
    prefix = _add_trend(app)
    app.number_input(key=prefix + "max_history").set_value(8)
    _apply(app)
    memory = MarketMemory(str(research.pipeline.store.data_dir.resolve()))
    assert memory.values["chart_logics"]["applied"]
    memory.chart_analysis_path.write_bytes(b"corrupted snapshot")
    fresh = AppTest.from_string(script).run()
    assert not fresh.exception and not fresh.error
    assert any("Saved chart results" in warning.value for warning in fresh.warning)
    assert not any("trend_id" in table.value.columns for table in fresh.dataframe)
    assert len(calls) == 1
    original_list = research.pipeline.list_datasets
    monkeypatch.setattr(research.pipeline, "list_datasets", lambda query: [
        item for item in original_list(query) if item.request.timeframe == "1d"])
    fresh.run()
    assert not fresh.exception and not fresh.error
    assert fresh.selectbox(key="browse-series").value == ("yahoo", "1d", "")
    assert len(calls) == 1


def test_old_dates_and_expanded_sections_restore_in_new_sessions():
    memory = MarketMemory("old-dates")
    memory.values["widgets"].update({"old-date": date(2001, 1, 2), "details": True})
    memory.save()
    script = (
        "from datetime import date\nimport streamlit as st\n"
        "from dashboard.workspace import market_memory\n"
        "m = market_memory('old-dates')\n"
        "m.widget(st.date_input, 'Old date', value=date(2001, 1, 1), key='old-date')\n"
        "with m.widget(st.expander, 'Details', key='details'):\n    st.write('Saved details')\n"
        "m.save()\n"
    )
    for _ in range(2):
        app = AppTest.from_string(script).run()
        assert not app.exception and not app.error
        assert app.date_input[0].value == date(2001, 1, 2)
        assert app.expander[0].proto.expanded


def test_unreadable_workspace_remains_usable_in_session(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("workspace is unreadable")

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(Path, "exists", denied)
    memory = MarketMemory("denied", tmp_path)
    assert memory.warning and memory.blocked
    memory.values["widgets"]["symbol"] = "TEST"
    memory.save()
    assert memory.values["widgets"]["symbol"] == "TEST"


def test_saved_dropdown_overrides_default_without_duplicate_state_warning(monkeypatch):
    from streamlit.elements.lib import policies
    from unittest.mock import Mock

    warning = Mock()
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    monkeypatch.setattr(policies._LOGGER, "warning", warning)
    script = (
        "import streamlit as st\n"
        "from dashboard.workspace import market_memory\n"
        "memory = market_memory('dropdown-default')\n"
        "memory.widget(st.selectbox, 'Timezone', "
        "options=['UTC', 'America/New_York', 'Asia/Shanghai'], index=1, key='timezone')\n"
        "memory.save()\n"
    )
    app = AppTest.from_string(script).run()
    assert app.selectbox(key="timezone").value == "America/New_York"
    app.selectbox(key="timezone").set_value("Asia/Shanghai").run()
    app.run()
    fresh = AppTest.from_string(script).run()
    for restored in (app, fresh):
        assert not restored.exception and not restored.error
        assert restored.selectbox(key="timezone").value == "Asia/Shanghai"
    warning.assert_not_called()
