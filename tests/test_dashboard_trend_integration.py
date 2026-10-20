"""Use the real Market data page and source store for the generic logic workflow."""

from datetime import UTC, datetime, timedelta
import json
import math

import polars as pl
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from research import Research
from trend_analysis import TrendConfig


def test_market_page_applies_saved_settings_and_hides_stale_results(tmp_path, monkeypatch):
    from chart_logics.plugins import trend as trend_plugin
    from dashboard import app as dashboard
    from dashboard import trend_settings

    analysis_calls = []
    analyze = trend_plugin.analyze

    def capture_analysis(*args, **kwargs):
        analysis_calls.append(kwargs["config"])
        return analyze(*args, **kwargs)

    monkeypatch.setattr(trend_plugin, "analyze", capture_analysis)
    settings_path = tmp_path / "trend_config.json"
    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", settings_path)
    # Existing saved defaults become the first trend instance's parameters.
    trend_settings.save_trend_settings(TrendConfig(penalty_grid_levels=3), settings_path)
    research = Research(tmp_path / "data", tmp_path / "runs")
    stamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(60)]
    close = [100 * math.exp(i * .002) for i in range(60)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * 60,
                          "open": close, "high": [p * 1.01 for p in close],
                          "low": [p * .99 for p in close], "close": close,
                          "volume": [100.] * 60})
    research.pipeline.ingest_frame(DataRequest("TEST", stamps[0], stamps[-1] + timedelta(days=1),
                                             timeframe="1d"), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    rendered_overlays = []
    render_chart = dashboard.render_series_chart

    def capture_chart(*args, **kwargs):
        rendered_overlays.append(tuple(kwargs.get("overlays", ())))
        return render_chart(*args, **kwargs)

    monkeypatch.setattr(dashboard, "render_series_chart", capture_chart)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=30).run()
    assert not app.exception and not app.error
    assert not rendered_overlays[-1]
    app.selectbox(key="chart-logic-add-choice").select((trend_plugin.TrendLogic.name, trend_plugin.TrendLogic.version))
    app.button(key="chart-logic-add").click().run()
    assert not app.exception and not app.error
    # Per-instance keys are generated; derive the prefix from a schema field.
    widget = next(item for item in app.number_input if item.key.endswith("-max_history"))
    prefix = widget.key.removesuffix("-max_history")
    instance_id = prefix.removeprefix("chart-logic-")
    display = "chart-display-" + instance_id

    def apply():
        app.button(key="chart-logic-apply").click().run()
        assert not app.exception and not app.error

    def accepted():
        return next(table.value for table in app.dataframe if "trend_id" in table.value.columns)

    def visible_ids():
        return {item["trend_id"] for item in rendered_overlays[-1] if item["type"] == "log_line"}

    def selection():
        documents = [json.loads(path.read_text()) for path in (tmp_path / "market_workspaces").glob("*.json")]
        return next(document["values"]["chart_logics"] for document in documents
                    if "chart_logics" in document["values"])

    assert not analysis_calls
    assert app.number_input(key=prefix + "-penalty_grid_levels").value == 3
    apply()
    assert len(analysis_calls) == 1
    results = accepted()
    assert results["end_index"].nunique() > 1
    assert results["on_chart"].sum() < len(results)
    assert set(results["status"]) == {"On chart", "Redundant"}
    assert {"lambda_values", "lambda_levels", "origin", "redundant_to"} <= set(results.columns)
    plotted = rendered_overlays[-1]
    assert visible_ids() == set(results.loc[results["on_chart"], "trend_id"])
    assert len(plotted) > 1
    assert any(a["start_index"] < b["end_index"] and b["start_index"] < a["end_index"]
               for i, a in enumerate(plotted) for b in plotted[i + 1:])
    assert all(item["lambda_values"] for item in plotted)
    assert app.number_input(key=display + "-redundancy_percentage").value == 5.0
    assert app.selectbox(key=display + "-redundancy_boundary_rule").value == "and"
    assert "redundancy_boundary_rule" not in selection()["applied"][0]["config"]
    assert not any(item.key == prefix + "-show_redundant" for item in app.checkbox)
    full_lambdas = sorted({value for row in results["lambda_values"] for value in row})
    assert app.multiselect(key=display + "-selected_lambdas").options == [str(value) for value in full_lambdas]

    # Redundancy is also presentation: update chart/table without discovering fits.
    app.selectbox(key=display + "-redundancy_boundary_rule").select("or").run()
    or_results = accepted()
    assert len(rendered_overlays[-1]) == or_results["on_chart"].sum() == 1
    display_columns = ["on_chart", "status", "redundant_to"]
    assert or_results.drop(columns=display_columns).equals(results.drop(columns=display_columns))
    assert selection()["display"][instance_id]["config"]["redundancy_boundary_rule"] == "or"
    assert len(analysis_calls) == 1
    assert not any("Unapplied changes" in item.value for item in app.caption)
    original_signature = app.session_state["chart-logic-overlay-signature"]
    original_revision = app.session_state["chart-logic-overlay-revision"]

    app.checkbox(key=display + "-show_redundant").check().run()
    all_results = accepted()
    assert all_results["on_chart"].all()
    assert len(rendered_overlays[-1]) == len(all_results)
    assert visible_ids() == set(all_results["trend_id"])
    assert all_results.drop(columns="on_chart").equals(or_results.drop(columns="on_chart"))
    assert app.session_state["chart-logic-overlay-revision"] > original_revision
    all_trends = rendered_overlays[-1]

    app.checkbox(key=display + "-show_all_lambdas").uncheck().run()
    assert not rendered_overlays[-1]
    # Suggestions use all accepted fits, even when all fits are hidden.
    assert app.multiselect(key=display + "-selected_lambdas").options == [str(value) for value in full_lambdas]
    selected_lambda = all_trends[0]["lambda_values"][0]
    app.multiselect(key=display + "-selected_lambdas").set_value([selected_lambda]).run()
    expected_ids = {item["trend_id"] for item in all_trends if selected_lambda in item["lambda_values"]}
    assert visible_ids() == expected_ids
    filtered = accepted()
    assert set(filtered.loc[filtered["on_chart"], "trend_id"]) == expected_ids
    assert len(filtered) == len(all_results)
    assert filtered["redundant_to"].equals(all_results["redundant_to"])

    for mode, expected in (("positive", expected_ids), ("negative", set())):
        app.selectbox(key=display + "-slope_mode").select(mode).run()
        assert len(analysis_calls) == 1
        assert visible_ids() == expected
        visible = accepted()
        assert set(visible.loc[visible["on_chart"], "trend_id"]) == expected
        assert visible.drop(columns=["on_chart", "status"]).equals(all_results.drop(columns=["on_chart", "status"]))
    assert any("No trends match" in item.value for item in app.caption)

    # Exact inclusive bounds remain combined with the lambda filter.
    slope = all_trends[0]["slope"]
    app.selectbox(key=display + "-slope_mode").select("range")
    app.number_input(key=display + "-slope_min").set_value(slope)
    app.number_input(key=display + "-slope_max").set_value(slope).run()
    exact_ids = {item["trend_id"] for item in all_trends
                 if item["slope"] == slope and selected_lambda in item["lambda_values"]}
    assert exact_ids
    assert visible_ids() == exact_ids
    assert set(accepted().loc[accepted()["on_chart"], "slope"]) == {slope}

    # Invalid display input preserves the last valid presentation and full result.
    before = rendered_overlays[-1]
    app.number_input(key=display + "-slope_min").set_value(slope + .01).run()
    assert not app.exception
    assert any("Minimum slope" in item.value for item in app.error)
    assert rendered_overlays[-1] == before
    assert len(analysis_calls) == 1
    app.number_input(key=display + "-slope_min").set_value(-.01)
    app.number_input(key=display + "-slope_max").set_value(.01).run()
    assert visible_ids() == expected_ids
    app.selectbox(key=display + "-slope_mode").select("all").run()
    app.selectbox(key=display + "-slope_mode").select("range").run()
    assert app.number_input(key=display + "-slope_min").value == -.01
    assert app.number_input(key=display + "-slope_max").value == .01

    app.selectbox(key=display + "-slope_mode").select("all")
    app.checkbox(key=display + "-show_all_lambdas").check()
    app.multiselect(key=display + "-selected_lambdas").set_value([]).run()
    assert rendered_overlays[-1] == all_trends
    app.checkbox(key=display + "-show_redundant").uncheck().run()
    assert len(rendered_overlays[-1]) == 1
    assert app.session_state["chart-logic-overlay-signature"] == original_signature
    assert len(analysis_calls) == 1
    revision = app.session_state["chart-logic-overlay-revision"]
    app.run()
    assert app.session_state["chart-logic-overlay-revision"] == revision
    assert len(analysis_calls) == 1

    # Calculation settings still wait for Apply while display edits remain live.
    app.number_input(key=prefix + "-max_history").set_value(24).run()
    assert any("Unapplied changes" in item.value for item in app.caption)
    app.checkbox(key=display + "-visible").uncheck().run()
    assert not rendered_overlays[-1]
    assert not accepted()["on_chart"].any()
    app.checkbox(key=display + "-visible").check().run()
    assert len(rendered_overlays[-1]) == 1
    assert len(analysis_calls) == 1
    assert selection()["applied"][0]["config"]["max_history"] != 24
    # Restore the original calculation draft before testing removal/cache reuse.
    app.number_input(key=prefix + "-max_history").set_value(selection()["applied"][0]["config"]["max_history"]).run()
    calls = len(analysis_calls)
    # Removing is also pending until Apply, then clears outputs and tables.
    instance_id = prefix.removeprefix("chart-logic-")
    app.button(key="chart-logic-remove-" + instance_id).click().run()
    assert len(rendered_overlays[-1]) == 1
    apply()
    assert not rendered_overlays[-1]
    assert app.session_state["chart-logic-overlay-revision"] > revision
    assert not any("trend_id" in table.value.columns for table in app.dataframe)
    assert len(analysis_calls) == calls

    # The same data/config under another instance shares the per-logic cache.
    app.button(key="chart-logic-add").click().run()
    apply()
    assert rendered_overlays[-1]
    assert len(analysis_calls) == calls
    first_new = selection()["applied"][0]["instance_id"]
    app.button(key="chart-logic-add").click().run()
    apply()
    second_new = selection()["applied"][1]["instance_id"]
    assert len(analysis_calls) == calls
    assert {item["instance_id"] for item in rendered_overlays[-1]} == {first_new, second_new}
    app.checkbox(key=f"chart-display-{first_new}-visible").uncheck().run()
    app.checkbox(key=f"chart-display-{second_new}-show_all_lambdas").uncheck().run()
    assert not rendered_overlays[-1]
    app.multiselect(key=f"chart-display-{second_new}-selected_lambdas").set_value([selected_lambda]).run()
    assert {item["instance_id"] for item in rendered_overlays[-1]} == {second_new}
    assert len(analysis_calls) == calls
    app.date_input[0].set_value(stamps[10].date()).run()
    assert not app.exception and not app.error
    assert not rendered_overlays[-1]
    assert not any("trend_id" in table.value.columns for table in app.dataframe)
    assert any("Click Apply" in message.value for message in app.info)
    assert len(analysis_calls) == calls
