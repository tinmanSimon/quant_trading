"""Indicator controls, initialization, tables, and persistence in Market data."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from streamlit.testing.v1 import AppTest

from chart_logics.plugins.indicators import MACDLogic, RSILogic
from dashboard.workspace import MarketMemory
from data_pipeline import DataRequest
from research import Research


def test_market_indicators_customize_add_remove_and_restore(tmp_path, monkeypatch):
    import market_indicators
    from dashboard import app as dashboard, trend_settings

    research = Research(tmp_path / "data", tmp_path / "runs")
    count = 620
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(count)]
    close = 100 + np.sin(np.arange(count) / 5) * 8 + np.arange(count) * .02
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * count,
                          "open": close, "close": close, "high": close + 1,
                          "low": close - 1, "volume": np.arange(count) + 100.})
    research.pipeline.ingest_frame(DataRequest("TEST", stamps[0], stamps[-1] + timedelta(days=1),
                                              timeframe="1d"), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", tmp_path / "trend.json")
    rsi = market_indicators.rsi
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["period"])
        return rsi(*args, **kwargs)

    monkeypatch.setattr(market_indicators, "rsi", counted)
    rendered = []
    render = dashboard.render_series_chart

    def capture(*args, **kwargs):
        rendered.append(kwargs["overlays"])
        return render(*args, **kwargs)

    monkeypatch.setattr(dashboard, "render_series_chart", capture)
    script = "from dashboard.app import main\nmain()"
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception and not app.error
    app.date_input[0].set_value(stamps[20].date()).run()
    scope = str(research.pipeline.store.data_dir.resolve())

    def memory():
        return MarketMemory(scope)

    def selection():
        return memory().values["chart_logics"]

    def add(logic):
        app.selectbox(key="chart-logic-add-choice").set_value((logic.name, logic.version))
        app.button(key="chart-logic-add").click().run()
        assert not app.exception and not app.error
        return selection()["draft"][-1]["instance_id"]

    def apply():
        app.button(key="chart-logic-apply").click().run()
        assert not app.exception and not app.error

    first = add(RSILogic)
    second = add(RSILogic)
    app.number_input(key=f"chart-logic-{second}-period").set_value(7).run()
    third = add(MACDLogic)
    assert calls == [] and rendered[-1] == []
    apply()
    assert calls == [14, 7]
    panels = {item["panel"]["id"] for item in rendered[-1]}
    assert panels == {f"{identifier}:indicator" for identifier in (first, second, third)}
    assert any(item["type"] == "bar" for item in rendered[-1])

    archive = memory().load_chart_analysis()
    rows = archive["results"][first]["tables"][0]["rows"]
    expected = rsi(frame, period=14).frame.slice(20)
    assert len(rows) == 600
    assert rows[0]["source_index"] == 0 and rows[-1]["source_index"] == 599
    np.testing.assert_allclose([row["value"] for row in rows], expected["value"].to_numpy())
    assert archive["results"][first]["data"]["metadata"]["initialization_bars"] == 20
    assert archive["results"][first]["overlays"] == []
    assert all(len(table.value) <= 500 for table in app.dataframe)
    page_key = f"chart-result-page-{first}-0"
    assert app.number_input(key=page_key).value == 2
    app.number_input(key=page_key).set_value(1).run()
    assert calls == [14, 7]
    assert any(len(table.value) == 500 for table in app.dataframe)

    # Styling and visibility use existing values; archived tables remain exact.
    app.text_input(key=f"chart-display-{first}-value_color").set_value("#123456").run()
    assert not app.error and not app.exception
    assert next(item for item in rendered[-1] if item["id"] == f"{first}:value")["line"]["color"] == "#123456"
    app.checkbox(key=f"chart-display-{second}-visible").uncheck().run()
    assert all(item["panel"]["id"] != f"{second}:indicator" for item in rendered[-1])
    assert calls == [14, 7]
    assert memory().load_chart_analysis() == archive

    # A pending calculation edit cannot silently change the visible result.
    app.number_input(key=f"chart-logic-{first}-period").set_value(21).run()
    assert calls == [14, 7]
    assert any(item["panel"]["title"] == "RSI (14, close)" for item in rendered[-1])
    apply()
    assert calls == [14, 7, 21]
    assert any(item["panel"]["title"] == "RSI (21, close)" for item in rendered[-1])
    shown = rendered[-1]

    fresh = AppTest.from_string(script, default_timeout=30).run()
    assert not fresh.exception and not fresh.error
    assert fresh.number_input(key=f"chart-logic-{first}-period").value == 21
    assert fresh.text_input(key=f"chart-display-{first}-value_color").value == "#123456"
    assert not fresh.checkbox(key=f"chart-display-{second}-visible").value
    assert rendered[-1] == shown and calls == [14, 7, 21]
    app = fresh
    app.button(key=f"chart-logic-remove-{first}").click().run()
    assert any(item["panel"]["id"] == f"{first}:indicator" for item in rendered[-1])
    apply()
    assert {item["panel"]["id"] for item in rendered[-1]} == {f"{third}:indicator"}
    assert calls == [14, 7, 21]
