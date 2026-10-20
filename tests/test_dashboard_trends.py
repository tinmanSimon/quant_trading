"""Offline trend form lifecycle, durable settings, and stale-result protection."""

from datetime import UTC, datetime, timedelta
from textwrap import indent
from types import SimpleNamespace

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.trend_settings import load_trend_settings, save_trend_settings
from dashboard.trend_selection import TrendSelection
from trend_analysis import TrendConfig


def _element(elements, label):
    return next(item for item in elements if item.label == label)


@pytest.fixture
def trend_page(tmp_path, monkeypatch):
    import dashboard.trends as trends
    source = SimpleNamespace(
        frame=pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i)
                                          for i in range(40)],
                            "close": [100.0 + i for i in range(40)]}),
        identity=("test", "AAPL", "raw"), path=tmp_path / "settings.json", calls=[], error=None,
        progress_updates=[], selection=None, diagnostics=pl.DataFrame({"accepted": [True]}),
        trends=(SimpleNamespace(trend_id="T1", start_index=0, end_index=39,
                                duration=40, clarity=.9, direction=1),),
    )

    def fake_analyze(frame, *, as_of, config, progress):
        source.calls.append((frame.clone(), as_of, config))
        total = max(0, len(frame) - config.min_segment_length + 1)
        for done in (0, total):
            progress(done, total)
            source.progress_updates.append((done, total))
        if source.error:
            raise source.error
        return SimpleNamespace(
            status="ok", analysis_start=frame["timestamp"][0], analysis_end=frame["timestamp"][-1],
            config=config, algorithm_version="test", analysis_mode="history",
            trends=source.trends,
            to_frame=lambda: pl.DataFrame([
                {"trend_id": item.trend_id, "clarity": item.clarity,
                 "start_index": item.start_index, "end_index": item.end_index,
                 "lambda_values": [.1, .2], "lambda_levels": [0, 1], "origin": "segmentation"}
                for item in source.trends
            ]),
            diagnostics_frame=lambda: source.diagnostics,
        )

    monkeypatch.setattr(trends, "analyze", fake_analyze)
    monkeypatch.setattr(trends, "_test_source", source, raising=False)
    script = (
        "import streamlit as st\nimport dashboard.trends as t\n"
        "source = t._test_source\n"
        "result = t.render_trend_controls(source.frame, timeframe='1h', identity=source.identity, "
        "settings_path=source.path)\n"
        "st.session_state['_test_result'] = result\n"
        "t.render_trend_results(result, selection=source.selection)\n"
    )
    app = AppTest.from_string(script).run()
    assert not app.exception
    return app, source, script


def _enable(app):
    _element(app.checkbox, "Analyze trends").check().run()
    assert not app.exception


def _apply(app):
    _element(app.button, "Apply and analyze").click().run()
    assert not app.exception


def test_only_apply_computes_and_ordinary_reruns_reuse_result(trend_page):
    app, source, _ = trend_page
    assert not source.calls
    assert not source.path.exists()
    _enable(app)
    _element(app.number_input, "Minimum clarity").set_value(0.8)
    _apply(app)
    assert len(source.calls) == 1
    assert source.calls[0][2].clarity_threshold == 0.8
    assert load_trend_settings(source.path).config.clarity_threshold == 0.8
    assert app.session_state["_test_result"] is not None
    assert source.progress_updates == [(0, 33), (33, 33)]
    app.run()
    _apply(app)
    assert len(source.calls) == 1
    assert not app.exception


def test_restart_and_widget_removal_preserve_applied_config(trend_page):
    app, source, script = trend_page
    _enable(app)
    _element(app.number_input, "Maximum lookback per endpoint (bars)").set_value(500)
    _element(app.number_input, "Number of λ levels").set_value(4)
    _element(app.number_input, "Minimum redundancy resolution (bars)").set_value(9)
    _element(app.number_input, "Redundancy tolerance (%)").set_value(12)
    _element(app.selectbox, "Boundary matching rule").select("or")
    _apply(app)
    _element(app.checkbox, "Analyze trends").uncheck().run()
    assert app.session_state["_test_result"] is None
    app.run()  # Give Streamlit a full hidden run to discard inactive widgets.
    _enable(app)
    assert _element(app.number_input, "Maximum lookback per endpoint (bars)").value == 500
    assert _element(app.number_input, "Minimum redundancy resolution (bars)").value == 9
    assert _element(app.number_input, "Redundancy tolerance (%)").value == 12
    assert _element(app.selectbox, "Boundary matching rule").value == "or"
    assert app.session_state["_test_result"] is not None
    fresh = AppTest.from_string(script).run()
    _enable(fresh)
    assert _element(fresh.number_input, "Maximum lookback per endpoint (bars)").value == 500
    assert _element(fresh.number_input, "Number of λ levels").value == 4
    assert _element(fresh.number_input, "Minimum redundancy resolution (bars)").value == 9
    assert load_trend_settings(source.path).config.redundancy_min_bars == 9
    assert _element(fresh.number_input, "Redundancy tolerance (%)").value == 12
    assert _element(fresh.selectbox, "Boundary matching rule").value == "or"
    assert load_trend_settings(source.path).config.redundancy_boundary_rule == "or"
    assert fresh.session_state["_test_result"] is None
    assert len(source.calls) == 1


def test_fixed_grid_controls_preview_and_persistence(trend_page):
    app, source, _ = trend_page
    _enable(app)
    assert _element(app.number_input, "Number of λ levels").value == 20
    _element(app.number_input, "Starting λ").set_value(2.1)
    _element(app.number_input, "Penalty grid step").set_value(3.0)
    _element(app.number_input, "Number of λ levels").set_value(4)
    _apply(app)
    assert any(item.value == "2.1, 16.8, 134.4, 1075.2" for item in app.code)
    saved = load_trend_settings(source.path).config
    assert saved.penalty_multiplier == 2.1
    assert saved.penalty_grid_step == 3
    assert saved.penalty_grid_levels == 4
    assert source.calls[-1][2] == saved


def test_page_switch_restores_percentage_and_boundary_rule(trend_page):
    _, source, script = trend_page
    page_script = ("import streamlit as st\n"
                   "if st.radio('Page', ('Market data', 'Elsewhere')) == 'Market data':\n"
                   + indent(script, "    "))
    app = AppTest.from_string(page_script).run()
    _enable(app)
    _element(app.number_input, "Redundancy tolerance (%)").set_value(12.5)
    _element(app.selectbox, "Boundary matching rule").select("or")
    _apply(app)
    _element(app.radio, "Page").set_value("Elsewhere").run()
    app.run()
    _element(app.radio, "Page").set_value("Market data").run()
    _enable(app)
    assert _element(app.number_input, "Redundancy tolerance (%)").value == 12.5
    assert _element(app.selectbox, "Boundary matching rule").value == "or"
    assert app.session_state["_test_result"].config.redundancy_boundary_rule == "or"
    assert len(source.calls) == 1


def test_source_content_or_identity_changes_hide_stale_results(trend_page):
    app, source, _ = trend_page
    _enable(app)
    _apply(app)
    source.frame = source.frame.with_columns(pl.col("close") + 1)
    app.run()
    assert not app.exception
    assert app.session_state["_test_result"] is None
    assert len(source.calls) == 1
    _apply(app)
    assert len(source.calls) == 2
    source.identity = ("test", "MSFT", "raw")
    app.run()
    assert app.session_state["_test_result"] is None
    assert len(source.calls) == 2


def test_invalid_config_does_not_save_or_relabel_previous_result(trend_page):
    app, source, _ = trend_page
    _enable(app)
    _apply(app)
    previous = source.path.read_bytes()
    _element(app.number_input, "Regression weight").set_value(0.5)
    _apply(app)
    assert len(source.calls) == 1
    assert source.path.read_bytes() == previous
    assert any("Invalid trend settings" in item.value for item in app.error)
    assert app.session_state["_test_result"].config == TrendConfig()


def test_analysis_failure_saves_settings_but_clears_old_result(trend_page):
    app, source, _ = trend_page
    _enable(app)
    _apply(app)
    source.error = ValueError("synthetic analysis failure")
    _element(app.number_input, "Minimum clarity").set_value(0.85)
    _element(app.number_input, "Redundancy tolerance (%)").set_value(0)
    _element(app.number_input, "Minimum redundancy resolution (bars)").set_value(0)
    _apply(app)
    assert app.session_state["_test_result"] is None
    assert load_trend_settings(source.path).config.clarity_threshold == 0.85
    assert any("synthetic analysis failure" in item.value for item in app.error)


def test_restore_defaults_saves_and_invalidates_previous_analysis(trend_page):
    app, source, script = trend_page
    _enable(app)
    _element(app.number_input, "Minimum clarity").set_value(0.85)
    _element(app.number_input, "Redundancy tolerance (%)").set_value(0)
    _element(app.number_input, "Minimum redundancy resolution (bars)").set_value(0)
    _element(app.selectbox, "Boundary matching rule").select("or")
    _apply(app)
    _element(app.button, "Restore trend defaults").click().run()
    assert not app.exception
    assert app.session_state["_test_result"] is None
    assert load_trend_settings(source.path).config == TrendConfig()
    assert _element(app.number_input, "Minimum clarity").value == TrendConfig().clarity_threshold
    assert _element(app.number_input, "Redundancy tolerance (%)").value == 5
    assert _element(app.number_input, "Minimum redundancy resolution (bars)").value == 5
    assert _element(app.selectbox, "Boundary matching rule").value == "and"
    fresh = AppTest.from_string(script).run()
    _enable(fresh)
    assert _element(fresh.number_input, "Minimum clarity").value == TrendConfig().clarity_threshold
    assert _element(fresh.number_input, "Minimum redundancy resolution (bars)").value == 5
    assert _element(fresh.number_input, "Redundancy tolerance (%)").value == 5
    assert _element(fresh.selectbox, "Boundary matching rule").value == "and"


def test_save_failure_retains_applied_settings_and_analysis(trend_page, monkeypatch):
    import dashboard.trends as trends
    app, source, _ = trend_page
    _enable(app)
    _apply(app)

    def fail_save(*args):
        raise OSError("read-only settings directory")

    monkeypatch.setattr(trends, "save_trend_settings", fail_save)
    _element(app.number_input, "Minimum clarity").set_value(0.75)
    _apply(app)
    assert len(source.calls) == 1
    assert app.session_state["_test_result"].config == TrendConfig()
    assert load_trend_settings(source.path).config == TrendConfig()
    assert any("analysis was not started" in item.value for item in app.error)


def test_corrupt_settings_warn_until_explicit_save(trend_page):
    app, source, script = trend_page
    source.path.write_text("{broken")
    fresh = AppTest.from_string(script).run()
    _enable(fresh)
    assert any("saved file has not been changed" in item.value for item in fresh.warning)
    assert source.path.read_text() == "{broken"
    _apply(fresh)
    assert load_trend_settings(source.path).warning is None


def test_integer_json_values_can_populate_float_widgets(trend_page):
    _, source, script = trend_page
    save_trend_settings(TrendConfig(penalty_multiplier=2, redundancy_percentage=12), source.path)
    fresh = AppTest.from_string(script).run()
    _enable(fresh)
    assert _element(fresh.number_input, "Starting λ").value == 2.0
    assert _element(fresh.number_input, "Redundancy tolerance (%)").value == 12.0


def test_cache_is_bounded_and_recent_selection_can_be_reused(trend_page):
    app, source, _ = trend_page
    _enable(app)
    for index in range(6):
        source.identity = ("test", f"SYMBOL{index}", "raw")
        app.run()
        _apply(app)
    assert len(source.calls) == 6
    assert len(app.session_state["_trend_controls_state"]["cache"]) == 4
    source.identity = ("test", "SYMBOL4", "raw")
    app.run()
    _apply(app)
    assert len(source.calls) == 6
    source.identity = ("test", "SYMBOL0", "raw")
    app.run()
    _apply(app)
    assert len(source.calls) == 7
    assert len(app.session_state["_trend_controls_state"]["cache"]) == 4


def test_real_engine_result_renders_rates_in_percent(trend_page, monkeypatch):
    import dashboard.trends as trends
    from trend_analysis import analyze
    app, source, _ = trend_page
    monkeypatch.setattr(trends, "analyze", analyze)
    _enable(app)
    _element(app.number_input, "Maximum lookback per endpoint (bars)").set_value(16)
    _element(app.number_input, "Redundancy tolerance (%)").set_value(25.0)
    _apply(app)
    result = app.session_state["_test_result"]
    assert result.trends
    assert result.analysis_mode == "history"
    displayed = app.dataframe[0].value
    assert len(displayed) == len(result.trends)
    assert displayed["end_index"].min() == 7
    assert displayed["end_index"].max() == 39
    assert displayed["end_index"].nunique() == 33
    assert displayed["bars"].max() <= 16
    assert result.analysis_start == source.frame["timestamp"][0]
    assert displayed.iloc[0]["rate_per_bar"] == pytest.approx(result.trends[0].rate_per_bar * 100)
    assert displayed.iloc[0]["trend_id"] == result.trends[0].trend_id
    assert set(displayed["status"]) == {"On chart", "Redundant"}
    assert any("that interval's own endpoint" in item.value for item in app.caption)


def test_chart_selection_marks_rows_without_filtering_history_or_recomputing(trend_page):
    app, source, _ = trend_page
    source.selection = TrendSelection((), {"T1": "T2"})
    _enable(app)
    _apply(app)
    assert len(app.dataframe[0].value) == 1
    assert not app.dataframe[0].value.iloc[0]["on_chart"]
    assert app.dataframe[0].value.iloc[0]["status"] == "Redundant"
    assert app.dataframe[0].value.iloc[0]["redundant_to"] == "T2"
    assert any("0 intervals on chart; 1 redundant intervals" in item.value for item in app.caption)
    source.selection = None
    app.run()
    assert not app.exception
    assert len(source.calls) == 1
    assert len(app.dataframe[0].value) == 1
    assert app.dataframe[0].value.iloc[0]["on_chart"]
    assert app.dataframe[0].value.iloc[0]["status"] == "On chart"


def test_boundary_rule_changes_selection_only_when_applied(trend_page):
    app, source, _ = trend_page
    source.trends += (SimpleNamespace(trend_id="T2", start_index=20, end_index=39,
                                     duration=20, clarity=.8, direction=1),)
    _enable(app)
    assert _element(app.number_input, "Redundancy tolerance (%)").value == 5.0
    assert _element(app.selectbox, "Boundary matching rule").value == "and"
    _apply(app)
    assert list(app.dataframe[0].value["status"]) == ["On chart", "On chart"]
    assert any("both start and end differences are within the larger of 5 bars and 5%" in item.value for item in app.caption)

    _element(app.selectbox, "Boundary matching rule").select("or")
    app.run()
    assert app.session_state["_test_result"].config.redundancy_boundary_rule == "and"
    assert list(app.dataframe[0].value["status"]) == ["On chart", "On chart"]
    _apply(app)
    assert list(app.dataframe[0].value["status"]) == ["On chart", "Redundant"]
    assert app.dataframe[0].value.iloc[1]["redundant_to"] == "T1"
    assert len(app.dataframe[0].value) == 2
    assert load_trend_settings(source.path).config.redundancy_boundary_rule == "or"
    assert any("either the start or end difference is within the larger of 5 bars and 5%" in item.value for item in app.caption)

    _element(app.selectbox, "Boundary matching rule").select("and")
    _apply(app)
    assert list(app.dataframe[0].value["status"]) == ["On chart", "On chart"]
    assert len(source.calls) == 2  # Returning to the same applied settings uses the cached analysis.


def test_uncertain_candidates_have_separate_status_reason_and_lambda_labels(trend_page):
    app, source, _ = trend_page
    source.diagnostics = pl.DataFrame({
        "accepted": [True, False, False], "clarity": [.9, .4, .3],
        "start_index": [0, 4, 12], "end_index": [39, 39, 39],
        "rejection_reason": ["", "insufficient directional clarity", "insufficient magnitude"],
        "lambda_values": [[.1, .2], [.2], []], "lambda_levels": [[0, 1], [1], []],
        "origin": ["segmentation", "segmentation", "tail"],
    })
    _enable(app)
    _apply(app)
    accepted, uncertain = [item.value for item in app.dataframe]
    assert accepted.iloc[0]["status"] == "On chart"
    assert accepted.iloc[0]["lambda_label"] == "λ=0.1 (level 0); λ=0.2 (level 1)"
    assert len(uncertain) == 2
    assert set(uncertain["status"]) == {"Uncertain"}
    assert not uncertain["accepted"].any()
    assert uncertain.iloc[0]["rejection_reason"] == "insufficient directional clarity"
    assert uncertain.iloc[0]["lambda_label"] == "λ=0.2 (level 1)"
    assert uncertain.iloc[1]["lambda_label"] == "Tail-derived; no λ"
    assert any("2 uncertain candidates" in item.value for item in app.caption)


def test_retired_dedup_setting_is_preserved_but_not_exposed_in_form(trend_page):
    _, source, script = trend_page
    save_trend_settings(TrendConfig(dedup_ratio=1.75), source.path)
    fresh = AppTest.from_string(script).run()
    _enable(fresh)
    assert "Duplicate duration ratio" not in {item.label for item in fresh.number_input}
    assert _element(fresh.number_input, "Redundancy tolerance (%)").value == 5
    _element(fresh.number_input, "Redundancy tolerance (%)").set_value(3)
    _apply(fresh)
    saved = load_trend_settings(source.path).config
    assert saved.dedup_ratio == 1.75
    assert saved.redundancy_percentage == 3


def test_hot_reload_reloads_saved_settings_and_discards_old_cached_result(trend_page):
    app, source, _ = trend_page
    _enable(app)
    _element(app.number_input, "Redundancy tolerance (%)").set_value(7)
    _element(app.selectbox, "Boundary matching rule").select("or")
    _apply(app)
    assert app.session_state["_test_result"] is not None
    state = app.session_state["_trend_controls_state"]
    state["algorithm_version"] = "old-version"
    # An older live session may hold an object from before the field existed.
    state["applied"] = SimpleNamespace(to_dict=lambda: {"max_history": 2000})
    app.run()
    assert not app.exception
    assert app.session_state["_test_result"] is None
    assert not app.session_state["_trend_controls_state"]["cache"]
    assert _element(app.number_input, "Redundancy tolerance (%)").value == 7
    assert load_trend_settings(source.path).config.redundancy_percentage == 7
    assert _element(app.selectbox, "Boundary matching rule").value == "or"
    assert load_trend_settings(source.path).config.redundancy_boundary_rule == "or"
    assert len(source.calls) == 1


def test_minimum_resolution_changes_selection_and_is_applied_only_on_submit(trend_page):
    app, source, _ = trend_page
    source.trends += (SimpleNamespace(trend_id="T2", start_index=4, end_index=43,
                                     duration=40, clarity=.8, direction=1),)
    _enable(app)
    _apply(app)
    assert list(app.dataframe[0].value["status"]) == ["On chart", "Redundant"]
    _element(app.number_input, "Minimum redundancy resolution (bars)").set_value(0)
    app.run()
    assert list(app.dataframe[0].value["status"]) == ["On chart", "Redundant"]
    _apply(app)
    assert list(app.dataframe[0].value["status"]) == ["On chart", "On chart"]
    assert load_trend_settings(source.path).config.redundancy_min_bars == 0


def test_earlier_lower_clarity_trend_is_the_display_representative(trend_page):
    app, source, _ = trend_page
    source.trends += (SimpleNamespace(trend_id="early", start_index=1, end_index=38,
                                     duration=38, clarity=.8, direction=1),)
    _enable(app)
    _apply(app)
    displayed = app.dataframe[0].value.set_index("trend_id")
    assert displayed.loc["early", "status"] == "On chart"
    assert displayed.loc["T1", "status"] == "Redundant"
    assert displayed.loc["T1", "redundant_to"] == "early"
    assert any("Selection proceeds by earliest ending bar" in item.value for item in app.caption)
    app.run()
    assert not app.exception
    assert len(source.calls) == 1
    assert app.dataframe[0].value.set_index("trend_id").loc["T1", "redundant_to"] == "early"
