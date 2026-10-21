"""Chart logic modules drive their own controls and only run on explicit Apply."""

from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from research.analysis import BaseChartLogic, LogicRegistry, LogicResult, LogicSpec
from dashboard.workspace import MarketMemory


@pytest.fixture
def logic_panel(monkeypatch):
    from dashboard import logic_controls
    calls = []

    class SampleLogic(BaseChartLogic):
        name = "sample"
        label = "Sample levels"
        description = "A module-defined configurable chart calculation."
        config_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "factor": {"type": "number", "default": 5.0, "minimum": 0., "maximum": 20., "title": "Offset"},
                "fail": {"type": "boolean", "default": False, "title": "Simulate failure"},
            },
        }

        display_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "selected_levels": {"type": "array", "items": {"type": "number"},
                                    "default": [], "title": "Selected levels"},
                "show_all": {"type": "boolean", "default": True, "title": "Show all levels"},
            },
        }

        def display_schema_for(self, result):
            schema = deepcopy(self.display_schema)
            schema["properties"]["selected_levels"]["x-ui-options"] = [row["value"] for row in result.tables[0]["rows"]]
            return schema

        def present(self, result, config):
            if not config["show_all"]:
                result.overlays = [overlay for overlay in result.overlays
                                   if overlay["y"][0] in config["selected_levels"]]
            return result

        def validate(self, config):
            if config["factor"] == 13:
                raise ValueError("Offset 13 is unavailable")
            return config

        def calculate(self, context, config, progress=None):
            calls.append(deepcopy(config))
            if config["fail"]:
                raise RuntimeError("Calculation unavailable")
            if progress:
                progress(1, 1)
            y = [value + config["factor"] for value in context.frame["close"].to_list()]
            return LogicResult(
                overlays=[{"id": "level", "type": "line", "x": list(range(len(y))), "y": y}],
                tables=[{"name": "Calculated levels", "rows": [{"value": value} for value in y]}],
                messages=["Original bars were used."],
            )

    registry = LogicRegistry()
    registry.register_class(SampleLogic)
    monkeypatch.setattr(logic_controls, "discover_logics", lambda: registry)
    script = '''
from datetime import UTC, datetime
import polars as pl
import streamlit as st
from research.analysis import LogicContext
from dashboard.logic_controls import render_logic_controls, render_logic_display_controls, render_logic_results
from dashboard.workspace import market_memory
source = st.selectbox('Source', ['First', 'Second'], key='source')
frame = pl.DataFrame({'timestamp': [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
                      'close': [100., 102.] if source == 'First' else [200., 202.]})
context = LogicContext(frame, '1d', source, datetime(2024, 2, 1, tzinfo=UTC))
memory = market_memory('chart-logic-test')
st.session_state['test-memory'] = memory
with st.container():
    raw = render_logic_controls(context, memory)
with st.container():
    run = render_logic_display_controls(raw, memory)
st.session_state['raw-overlays'] = raw.overlays if raw is not None else []
st.session_state['visible-overlays'] = run.overlays if run is not None else []
st.session_state['visible-errors'] = run.errors if run is not None else {}
render_logic_results(run)
memory.save()
'''
    return AppTest.from_string(script, default_timeout=15).run(), calls, script, registry


def _add(app):
    app.button(key="chart-logic-add").click().run()
    assert not app.exception and not app.error
    return app.session_state["test-memory"].values["chart_logics"]["draft"][-1]["instance_id"]


def _apply(app):
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception


def test_module_settings_generated_and_duplicate_instances_apply_independently(logic_panel):
    app, calls, _, _ = logic_panel
    assert not app.exception and not app.error
    assert app.selectbox(key="chart-logic-add-choice").options == ["Sample levels"]
    first = _add(app)
    second = _add(app)
    app.number_input(key=f"chart-logic-{second}-factor").set_value(10).run()
    assert calls == [] and app.session_state["visible-overlays"] == []
    _apply(app)
    assert not app.error
    assert len(calls) == 2
    overlays = app.session_state["visible-overlays"]
    assert [item["y"] for item in overlays] == [[105., 107.], [110., 112.]]
    assert len({item["id"] for item in overlays}) == 2
    assert len(app.dataframe) == 2
    assert any("Original bars" in item.value for item in app.caption)
    # Edits remain drafts and unchanged modules reuse their previous calculation.
    app.number_input(key=f"chart-logic-{second}-factor").set_value(8).run()
    assert len(calls) == 2 and app.session_state["visible-overlays"] == overlays
    assert any("Unapplied changes" in item.value for item in app.caption)
    _apply(app)
    assert len(calls) == 3
    assert app.session_state["visible-overlays"][0] == overlays[0]
    assert app.session_state["visible-overlays"][1]["y"] == [108., 110.]
    # Removals are also drafts until Apply, including removal of the last logic.
    app.button(key=f"chart-logic-remove-{first}").click().run()
    assert len(app.session_state["visible-overlays"]) == 2
    _apply(app)
    assert len(app.session_state["visible-overlays"]) == 1 and len(calls) == 3
    app.button(key=f"chart-logic-remove-{second}").click().run()
    assert len(app.session_state["visible-overlays"]) == 1
    _apply(app)
    assert app.session_state["visible-overlays"] == [] and len(calls) == 3


def test_invalid_configuration_preserves_results_and_runtime_failure_clears_only_failed_logic(logic_panel):
    app, calls, _, _ = logic_panel
    first = _add(app)
    second = _add(app)
    _apply(app)
    assert len(calls) == 1  # Identical logic/config uses the per-logic cache.
    previous = app.session_state["visible-overlays"]
    app.number_input(key=f"chart-logic-{first}-factor").set_value(13).run()
    _apply(app)
    assert len(calls) == 1
    assert app.session_state["visible-overlays"] == previous
    assert any("Offset 13" in item.value for item in app.error)
    app.number_input(key=f"chart-logic-{first}-factor").set_value(5)
    app.checkbox(key=f"chart-logic-{first}-fail").check().run()
    _apply(app)
    assert len(calls) == 2
    assert list(app.session_state["visible-errors"]) == [first]
    assert len(app.session_state["visible-overlays"]) == 1
    assert app.session_state["visible-overlays"][0]["instance_id"] == second
    assert any("Calculation unavailable" in item.value for item in app.error)


def test_draft_and_applied_results_survive_restart_but_other_data_needs_apply(logic_panel):
    app, calls, script, _ = logic_panel
    instance = _add(app)
    _apply(app)
    previous = app.session_state["visible-overlays"]
    app.number_input(key=f"chart-logic-{instance}-factor").set_value(7).run()
    fresh = AppTest.from_string(script, default_timeout=15).run()
    assert not fresh.exception and not fresh.error
    assert fresh.number_input(key=f"chart-logic-{instance}-factor").value == 7
    assert fresh.session_state["visible-overlays"] == previous and len(calls) == 1
    fresh.selectbox(key="source").set_value("Second").run()
    assert fresh.session_state["visible-overlays"] == [] and len(calls) == 1
    _apply(fresh)
    assert fresh.session_state["visible-overlays"][0]["y"] == [207., 209.] and len(calls) == 2


def test_missing_saved_module_preserves_configuration_and_can_be_removed(logic_panel):
    app, _, script, _ = logic_panel
    missing = LogicSpec("unavailable", "missing", "9", {"custom": 5}).to_dict()
    memory = MarketMemory("chart-logic-test")
    memory.values["chart_logics"] = {"version": 1, "draft": [missing], "applied": []}
    memory.save()
    fresh = AppTest.from_string(script, default_timeout=15).run()
    assert not fresh.exception
    assert any("unavailable" in item.value for item in fresh.warning)
    assert fresh.button(key="chart-logic-apply").disabled
    assert fresh.session_state["test-memory"].values["chart_logics"]["draft"] == [missing]
    fresh.button(key="chart-logic-remove-unavailable").click().run()
    assert not fresh.button(key="chart-logic-apply").disabled
    _apply(fresh)
    assert fresh.session_state["visible-overlays"] == []


def test_legacy_trend_settings_migrate_draft_and_applied_separately(tmp_path, monkeypatch):
    from research.analysis import discover_logics
    from dashboard import trend_settings
    from dashboard.logic_controls import _legacy_selection
    from research.analysis.trend import TrendConfig

    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", tmp_path / "trend.json")
    memory = MarketMemory("legacy-trend", tmp_path / "memory")
    config = TrendConfig(max_history=16).to_dict()
    memory.values["applied_config"] = config
    memory.values["widgets"].update({
        "analyze-trends": True, "_trend_setting_max_history": 24,
        "show-redundant-trends": True, "show-all-trend-lambdas": False,
        "selected-trend-lambdas": [1., "No segmentation λ (tail-derived)"],
        "selected-trend-slope-mode": "range", "_trend-slope-minimum": -.1,
        "_trend-slope-maximum": .2,
    })
    migrated = _legacy_selection(memory, discover_logics())
    assert migrated["applied"][0]["config"]["max_history"] == 16
    assert migrated["draft"][0]["config"]["max_history"] == 24
    for selection in (migrated["draft"], migrated["applied"]):
        assert selection[0]["name"] == "trend"
        assert "show_redundant" not in selection[0]["config"]
        assert "selected_lambdas" not in selection[0]["config"]
    display = migrated["display"]["legacy-trend-analysis"]["config"]
    assert display["show_redundant"] is True
    assert display["show_all_lambdas"] is False
    assert display["selected_lambdas"] == [1.]
    assert display["include_without_lambda"] is True
    assert display["slope_mode"] == "range"
    assert display["slope_min"] == -.1
    assert display["slope_max"] == .2


def test_market_page_combines_builtin_logics_and_keeps_chart_stable(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta
    import math

    import polars as pl
    from research.analysis.builtins.trend import TrendLogic
    from dashboard import app as dashboard, trend_settings
    from data_pipeline import DataRequest
    from research import Research

    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", tmp_path / "trend.json")
    research = Research(tmp_path / "data", tmp_path / "runs")
    stamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(24)]
    prices = [100 * math.exp(index * .002) for index in range(24)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * 24,
                          "open": prices, "high": [p * 1.01 for p in prices],
                          "low": [p * .99 for p in prices], "close": prices,
                          "volume": [100.] * 24})
    research.pipeline.ingest_frame(DataRequest("TEST", stamps[0], stamps[-1] + timedelta(days=1),
                                              timeframe="1d"), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    renders = []
    render_chart = dashboard.render_series_chart

    def capture_chart(*args, **kwargs):
        renders.append(deepcopy(kwargs))
        return render_chart(*args, **kwargs)

    monkeypatch.setattr(dashboard, "render_series_chart", capture_chart)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=30).run()
    assert not app.exception and not app.error
    app.selectbox(key="chart-logic-add-choice").set_value(("open_price_bands", "1.0.0")).run()
    app.button(key="chart-logic-add").click().run()
    reference = next(item for item in app.selectbox if item.label == "Open reference")
    reference.set_value("first_bar").run()
    app.selectbox(key="chart-logic-add-choice").set_value(("trend", TrendLogic.version)).run()
    app.button(key="chart-logic-add").click().run()
    assert renders[-1]["overlays"] == []
    _apply(app)
    assert not app.error
    overlays = renders[-1]["overlays"]
    assert {item["type"] for item in overlays} == {"line", "log_line"}
    assert len({item["instance_id"] for item in overlays}) == 2
    assert any("trend_id" in table.value.columns for table in app.dataframe)
    previous = renders[-1]
    next(item for item in app.number_input if item.label == "Upper offset (%)").set_value(10).run()
    assert renders[-1] == previous
    _apply(app)
    assert not app.error
    assert renders[-1]["chart_id"] == previous["chart_id"]
    assert renders[-1]["overlay_revision"] > previous["overlay_revision"]
    assert [item for item in renders[-1]["overlays"] if item["type"] == "log_line"] == [
        item for item in overlays if item["type"] == "log_line"]
    app.date_input[0].set_value(stamps[4].date()).run()
    assert not app.exception and not app.error
    assert renders[-1]["overlays"] == []


def test_live_display_is_separate_from_calculation_and_persists(logic_panel):
    app, calls, script, _ = logic_panel
    first, second = _add(app), _add(app)
    assert not any(widget.label == "Show on chart" for widget in app.checkbox)
    _apply(app)
    raw = deepcopy(app.session_state["raw-overlays"])
    assert len(calls) == 1
    first_prefix, second_prefix = f"chart-display-{first}-", f"chart-display-{second}-"
    assert app.multiselect(key=first_prefix + "selected_levels").options == ["105.0", "107.0"]
    app.checkbox(key=first_prefix + "visible").uncheck().run()
    assert [overlay["instance_id"] for overlay in app.session_state["visible-overlays"]] == [second]
    assert not any("Unapplied changes" in caption.value for caption in app.caption)
    assert app.session_state["raw-overlays"] == raw
    assert len(calls) == 1
    app.checkbox(key=second_prefix + "show_all").uncheck().run()
    assert app.session_state["visible-overlays"] == []
    app.multiselect(key=second_prefix + "selected_levels").set_value([105.]).run()
    assert len(app.session_state["visible-overlays"]) == 1
    # Calculation drafts stay pending even while live filters change the chart.
    app.number_input(key=f"chart-logic-{second}-factor").set_value(8).run()
    app.checkbox(key=first_prefix + "visible").check().run()
    assert app.session_state["visible-overlays"] == raw
    assert len(calls) == 1
    assert any("Unapplied changes" in caption.value for caption in app.caption)
    state = app.session_state["test-memory"].values["chart_logics"]
    assert state["display"][second]["config"]["selected_levels"] == [105.]
    assert state["applied"][1]["config"]["factor"] == 5.
    assert "show_all" not in state["draft"][1]["config"]
    fresh = AppTest.from_string(script, default_timeout=15).run()
    assert not fresh.exception and not fresh.error
    assert len(calls) == 1
    assert fresh.multiselect(key=second_prefix + "selected_levels").value == [105.]
    assert fresh.session_state["visible-overlays"] == raw
    _apply(fresh)
    assert len(calls) == 2
    assert fresh.multiselect(key=second_prefix + "selected_levels").options == ["108.0", "110.0"]
    assert fresh.multiselect(key=second_prefix + "selected_levels").value == []
    assert [overlay["instance_id"] for overlay in fresh.session_state["visible-overlays"]] == [first]
    assert len(fresh.session_state["raw-overlays"]) == 2


def test_mixed_saved_configs_migrate_applied_display_and_preserve_unknown(logic_panel):
    _, _, _, registry = logic_panel
    from dashboard.logic_controls import _saved_selection
    memory = MarketMemory("mixed-display")
    first = LogicSpec("first", "sample", registry.definitions()[0].version, {"factor": 5., "show_all": False,
                                                    "selected_levels": [105.]}).to_dict()
    draft = deepcopy(first)
    draft["config"]["show_all"] = True
    missing = LogicSpec("missing", "unavailable", "1", {"show_all": False}).to_dict()
    memory.values["chart_logics"] = {
        "version": 1, "draft": [draft, missing], "applied": [first],
        "defaults": {"sample": {"factor": 5., "show_all": False}},
    }
    memory.values["widgets"]["chart-logic-first-show_all"] = True
    selection = _saved_selection(memory, registry)
    assert selection["draft"][0]["config"] == {"factor": 5.}
    assert selection["applied"][0]["config"] == {"factor": 5.}
    assert selection["display"]["first"]["config"] == {"show_all": False, "selected_levels": [105.]}
    assert selection["draft"][1] == missing
    assert selection["defaults"]["sample"] == {"factor": 5.}
    assert "chart-logic-first-show_all" not in memory.values["widgets"]
