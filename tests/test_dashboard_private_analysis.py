"""Private scripts use the normal Market data Apply and display workflow."""

from datetime import UTC, datetime, timedelta
import polars as pl
from streamlit.testing.v1 import AppTest

from dashboard.workspace import MarketMemory
from data_pipeline import DataRequest
from research import Research


def test_private_analysis_reload_display_navigation_and_failures(tmp_path, monkeypatch):
    from dashboard import app as dashboard
    from research.analysis.scripting import discovery as loader

    project = tmp_path / "private-project"
    scripts = project / "secrete_analysis" / "analyses"
    scripts.mkdir(parents=True)
    (scripts.parent / "__init__.py").write_text("")
    (scripts / "__init__.py").write_text("")
    calls_path = tmp_path / "analysis-calls.txt"

    def write_script(revision):
        (scripts / "demo.py").write_text(f'''
from pathlib import Path
import polars as pl
from research.analysis import analysis

@analysis("private_demo", label="Private demonstration")
def run(ctx, every: int = 3, fail: bool = False):
    with Path({str(calls_path)!r}).open("a") as log:
        log.write({revision!r} + ":" + str(every) + "\\n")
    if fail:
        raise ValueError("Requested example failure")
    found = ctx.bars.filter(pl.col("source_index") % every == 0)
    result = ctx.result()
    result.text("**{revision} findings**")
    result.signals("Entries", found, price="close", label="Example entry")
    return result
''')

    write_script("Original")
    (scripts / "text_only.py").write_text('''
from research.analysis import analysis

@analysis("text_only", label="Text report")
def run(ctx):
    return ctx.result().text("A text-only report.")
''')
    monkeypatch.setattr(loader, "project_root", lambda: project)
    loader.load_registry(reload=True)
    research = Research(tmp_path / "data", tmp_path / "runs")
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(18)]
    close = [100.0 + i for i in range(len(stamps))]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["TEST"] * len(stamps),
                          "open": close, "close": close, "high": [v + 1 for v in close],
                          "low": [v - 1 for v in close], "volume": [1000.] * len(stamps)})
    research.pipeline.ingest_frame(DataRequest("TEST", stamps[0], stamps[-1] + timedelta(days=1),
                                              timeframe="1d"), frame)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    rendered = []
    monkeypatch.setattr(dashboard, "render_series_chart", lambda *args, **kwargs: rendered.append(kwargs))
    script = "from dashboard.app import main\nmain()"
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception and not app.error
    scope = str(research.pipeline.store.data_dir.resolve())

    def selection():
        return MarketMemory(scope).values["chart_logics"]

    def calls():
        return calls_path.read_text().splitlines() if calls_path.exists() else []

    def add(name):
        definition = next(item for item in loader.load_registry().definitions() if item.name == name)
        app.selectbox(key="chart-logic-add-choice").set_value((definition.name, definition.version))
        app.button(key="chart-logic-add").click().run()
        assert not app.exception and not app.error
        return selection()["draft"][-1]["instance_id"]

    first = add("private_demo")
    text_only = add("text_only")
    app.number_input(key=f"chart-logic-{first}-every").set_value(2).run()
    assert calls() == []
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error
    assert calls() == ["Original:2"]
    assert any("Original findings" in item.value for item in app.markdown)
    assert any("A text-only report" in item.value for item in app.markdown)
    assert not any(item.key == f"chart-display-{text_only}-visible" for item in app.checkbox)
    assert len(rendered[-1]["overlays"]) == 1
    assert any(item.label == "Analysis event" for item in app.selectbox)

    layers = f"chart-display-{first}-visible_layers"
    app.multiselect(key=layers).set_value([]).run()
    assert rendered[-1]["overlays"] == [] and calls() == ["Original:2"]
    app.multiselect(key=layers).set_value(["Entries"]).run()
    assert len(rendered[-1]["overlays"]) == 1 and calls() == ["Original:2"]

    app.checkbox(key=f"chart-display-{first}-visible").uncheck().run()
    assert rendered[-1]["overlays"] == []
    assert any(item.label == "Analysis event" for item in app.selectbox)
    next(item for item in app.button if item.label == "Next").click().run()
    assert rendered[-1]["navigation_command"]["source_index"] == 0
    assert calls() == ["Original:2"]
    original_revision = rendered[-1]["overlay_revision"]

    write_script("Updated")
    app.button(key="chart-logic-reload").click().run()
    assert not app.exception and not app.error
    assert app.number_input(key=f"chart-logic-{first}-every").value == 2
    assert calls() == ["Original:2"]
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error
    assert calls() == ["Original:2", "Updated:2"]
    # Text/code changed while the chart layers stayed hidden and identical.
    # The browser must still retire selection commands from the old result.
    assert rendered[-1]["overlay_revision"] > original_revision
    assert "navigation_command" not in rendered[-1]
    assert any("Updated findings" in item.value for item in app.markdown)
    assert not app.checkbox(key=f"chart-display-{first}-visible").value

    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert calls() == ["Original:2", "Updated:2"]
    app = reopened
    failed = add("private_demo")
    app.checkbox(key=f"chart-logic-{failed}-fail").check().run()
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception
    assert any("Requested example failure" in item.value for item in app.error)
    assert any("Updated findings" in item.value for item in app.markdown)
    assert calls() == ["Original:2", "Updated:2", "Updated:3"]
    app.button(key=f"chart-logic-remove-{failed}").click().run()
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error
    assert calls() == ["Original:2", "Updated:2", "Updated:3"]
