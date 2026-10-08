"""Use the real Market data page and source store for the trend workflow."""

from datetime import UTC, datetime, timedelta
import math

import polars as pl
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from research import Research


def test_market_page_applies_saved_settings_and_hides_stale_results(tmp_path, monkeypatch):
    from dashboard import app as dashboard
    from dashboard import trend_settings

    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", tmp_path / "trend_config.json")
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
    rendered_trends = []
    render_chart = dashboard.render_series_chart

    def capture_chart(*args, **kwargs):
        rendered_trends.append(tuple(kwargs.get("trends", ())))
        return render_chart(*args, **kwargs)

    monkeypatch.setattr(dashboard, "render_series_chart", capture_chart)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=30).run()
    assert not app.exception and not app.error
    app.checkbox(key="analyze-trends").check().run()
    next(button for button in app.button if button.label == "Apply and analyze").click().run()
    assert not app.exception and not app.error
    assert (tmp_path / "trend_config.json").exists()
    assert any("Clarity" in str(table.value.columns) or "clarity" in table.value.columns
               for table in app.dataframe)
    results = next(table.value for table in app.dataframe if "clarity" in table.value.columns)
    assert results["end_index"].nunique() > 1
    assert results["on_chart"].sum() < len(results)
    assert set(results["status"]) == {"On chart", "Redundant"}
    assert {"lambda_values", "lambda_levels", "origin", "redundant_to"} <= set(results.columns)
    plotted = rendered_trends[-1]
    assert {trend.trend_id for trend in plotted} == set(results.loc[results["on_chart"], "trend_id"])
    assert len(plotted) > 1
    assert any(a.start_index < b.end_index and b.start_index < a.end_index
               for i, a in enumerate(plotted) for b in plotted[i + 1:])
    assert all(trend.lambda_values for trend in plotted)
    assert app.number_input(key="_trend_setting_redundancy_percentage").value == 5.0
    assert app.selectbox(key="_trend_setting_redundancy_boundary_rule").value == "and"
    saved = trend_settings.load_trend_settings(tmp_path / "trend_config.json").config
    assert saved.redundancy_percentage == 5.0
    assert saved.redundancy_boundary_rule == "and"
    # Every interval here starts at zero: OR keeps just the clearest one while
    # retaining the same independently computed interval metrics in the table.
    app.selectbox(key="_trend_setting_redundancy_boundary_rule").select("or")
    next(button for button in app.button if button.label == "Apply and analyze").click().run()
    assert not app.exception and not app.error
    or_results = next(table.value for table in app.dataframe if "trend_id" in table.value.columns)
    assert len(rendered_trends[-1]) == or_results["on_chart"].sum() == 1
    display_columns = ["on_chart", "status", "redundant_to"]
    assert or_results.drop(columns=display_columns).equals(results.drop(columns=display_columns))
    assert trend_settings.load_trend_settings(tmp_path / "trend_config.json").config.redundancy_boundary_rule == "or"
    revision = app.session_state["trend-overlay-revision"]
    signature = app.session_state["trend-overlay-signature"]
    assert "Trend(" in signature or "trend_id" in signature
    # Ordinary reruns reuse the result and do not change overlay generation.
    app.run()
    assert app.session_state["trend-overlay-revision"] == revision
    app.checkbox(key="analyze-trends").uncheck().run()
    assert app.session_state["trend-overlay-revision"] > revision
    assert not any(title.value == "Trend analysis" for title in app.subheader)
    app.checkbox(key="analyze-trends").check().run()
    app.date_input[0].set_value(stamps[10].date()).run()
    assert not app.exception and not app.error
    assert not any(title.value == "Trend analysis" for title in app.subheader)
    assert any("Apply and analyze" in message.value for message in app.info)
