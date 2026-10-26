"""Exercise standalone flat analysis through both real dashboard workflows."""

from copy import deepcopy
from datetime import timedelta

import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from research import Research
from research.analysis.batch import BatchStore
from research.analysis.builtins.flat import FlatLogic
from dashboard.batch_analysis.state import BatchJob

from test_flat_logic import START, flat_bars


def _research(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    frame = flat_bars()
    research.pipeline.ingest_frame(DataRequest(
        "TEST", START, START + timedelta(hours=9), provider="databento", timeframe="30m"), frame)
    return research


def _periods(app):
    return next(item.value for item in app.dataframe if "period_id" in item.value.columns)


def _set_calculation(app, prefix, maximum=200):
    app.number_input(key=prefix + "-min_duration").set_value(3)
    app.number_input(key=prefix + "-max_duration").set_value(maximum)
    app.number_input(key=prefix + "-max_range_pct").set_value(2.0)


@pytest.mark.parametrize("maximum,intervals", [
    (200, [(12, 17), (6, 11), (0, 5)]),
    (4, [(14, 17), (8, 11), (2, 5)]),
])
def test_market_data_flat_analysis_draws_final_periods_and_toggles_without_recalculation(
        tmp_path, monkeypatch, maximum, intervals):
    from dashboard import app as dashboard, trend_settings

    monkeypatch.setattr(trend_settings, "DEFAULT_SETTINGS_PATH", tmp_path / "trend.json")
    research = _research(tmp_path)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    calculations = []
    calculate = FlatLogic.calculate

    def capture_calculation(self, context, config, progress=None):
        calculations.append(deepcopy(config))
        return calculate(self, context, config, progress=progress)

    monkeypatch.setattr(FlatLogic, "calculate", capture_calculation)
    rendered = []
    render_chart = dashboard.render_series_chart

    def capture_chart(*args, **kwargs):
        rendered.append(deepcopy(kwargs.get("overlays", [])))
        return render_chart(*args, **kwargs)

    monkeypatch.setattr(dashboard, "render_series_chart", capture_chart)
    app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=30).run()
    assert not app.exception and not app.error
    app.selectbox(key="chart-logic-add-choice").select((FlatLogic.name, FlatLogic.version))
    app.button(key="chart-logic-add").click().run()
    widget = next(item for item in app.number_input if item.key.endswith("-min_duration"))
    prefix = widget.key.removesuffix("-min_duration")
    _set_calculation(app, prefix, maximum)
    assert calculations == []
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error
    assert len(calculations) == 1
    periods = _periods(app)
    assert list(zip(periods["start_index"], periods["end_index"])) == intervals
    assert periods["bars"].between(3, maximum).all()
    assert periods["range_pct"].between(0, 2.0).all()
    assert periods["lower_price"].tolist() == [139., 119., 99.]
    assert periods["upper_price"].tolist() == [141., 121., 101.]
    assert periods["center_price"].tolist() == [140., 120., 100.]
    bands = [overlay for overlay in rendered[-1] if overlay["type"] == "log_line"]
    instance_id = prefix.removeprefix("chart-logic-")
    assert {overlay["id"] for overlay in bands} == {
        f"{instance_id}:{period_id}{suffix}" for period_id in periods["period_id"]
        for suffix in ("", "-lower", "-upper")}
    centers = {f"{instance_id}:{period_id}" for period_id in periods["period_id"]}
    assert [(item["start_index"], item["end_index"]) for item in bands if item["id"] in centers] == intervals
    assert all(line["slope"] == 0 for line in bands)
    display = "chart-display-" + instance_id
    app.checkbox(key=display + "-show_boundaries").uncheck().run()
    assert not app.exception and not app.error
    assert {overlay["type"] for overlay in rendered[-1]} == {"log_line"}
    app.checkbox(key=display + "-show_fit").uncheck().run()
    assert not app.exception and not app.error
    assert rendered[-1] == []
    assert _periods(app)[["period_id", "start_index", "end_index"]].equals(
        periods[["period_id", "start_index", "end_index"]])
    assert len(calculations) == 1
    app.checkbox(key=display + "-show_fit").check().run()
    assert len(calculations) == 1
    assert len(rendered[-1]) == 9
    # Invalid cross-field settings must preserve the last successfully applied view.
    previous_overlays = deepcopy(rendered[-1])
    app.number_input(key=prefix + "-max_duration").set_value(3)
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and app.error
    assert any("max_duration" in message.value for message in app.error)
    assert len(calculations) == 1
    assert rendered[-1] == previous_overlays
    assert _periods(app)[["period_id", "start_index", "end_index"]].equals(
        periods[["period_id", "start_index", "end_index"]])


@pytest.mark.parametrize("maximum,intervals", [
    (200, [(12, 17), (6, 11), (0, 5)]),
    (4, [(14, 17), (8, 11), (2, 5)]),
])
def test_analysis_page_runs_flat_periods_and_reopens_saved_tables(tmp_path, maximum, intervals):
    research = _research(tmp_path)
    script = f"""
from dashboard.pages.analysis import _analysis_page
from research import Research
_analysis_page(Research({str(research.pipeline.store.data_dir)!r}, {str(research.runs_dir)!r}))
"""
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception and not app.error
    app.checkbox(key="batch-all-tickers").check().run()
    app.selectbox(key="batch-analysis-choice").select((FlatLogic.name, FlatLogic.version))
    app.button(key="batch-analysis-add").click().run()
    widget = next(item for item in app.number_input if item.key.endswith("-min_duration"))
    _set_calculation(app, widget.key.removesuffix("-min_duration"), maximum)
    app.button(key="batch-run").click().run()
    jobs = [item for item in app.session_state.filtered_state.values() if isinstance(item, BatchJob)]
    assert len(jobs) == 1
    jobs[0].thread.join(timeout=10)
    assert not jobs[0].running and jobs[0].error is None
    app.run()
    assert not app.exception and not app.error
    periods = _periods(app)
    assert list(zip(periods["start_index"], periods["end_index"])) == intervals
    assert periods["bars"].between(3, maximum).all()
    assert periods["range_pct"].tolist() == pytest.approx([100 * 2 / 140, 100 * 2 / 120, 2.0])
    assert {"lower_price", "upper_price", "center_price", "range_pct"} <= set(periods.columns)
    app.button(key="batch-save").click().run()
    assert not app.exception and not app.error
    store = BatchStore(research.runs_dir / "analysis")
    manifest, = store.list_runs()
    assert manifest["counts"]["completed"] == 1
    assert manifest["specs"][0]["name"] == "flat"
    assert manifest["specs"][0]["config"]["max_duration"] == maximum
    assert manifest["specs"][0]["config"]["max_range_pct"] == 2.0
    assert {"slope_min_pct", "slope_max_pct"}.isdisjoint(manifest["specs"][0]["config"])
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert reopened.selectbox(key="batch-saved-run").value == manifest["run_id"]
    assert _periods(reopened).equals(periods)


def _flat_pagination_app():
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from research.analysis import LogicContext, LogicRunner, LogicSpec, load_registry
    from research.analysis.builtins.flat import FlatLogic
    from dashboard.market_data.results import render_logic_results

    start = datetime(2024, 1, 1, tzinfo=UTC)
    prices = [100. + index // 2 for index in range(1002)]
    frame = pl.DataFrame({
        "timestamp": [start + timedelta(minutes=30 * index) for index in range(1002)],
        "open": prices, "high": prices, "low": prices, "close": prices,
    })
    registry = load_registry()
    spec = LogicSpec("flat-pagination", FlatLogic.name, FlatLogic.version,
                     {"min_duration": 2, "max_duration": 200, "max_range_pct": .001})
    run = LogicRunner(registry).apply([spec], LogicContext(frame, "30m", "pagination",
                                                        start + timedelta(days=30)))
    render_logic_results(run, registry=registry)


def test_newest_first_flat_table_opens_first_page_and_preserves_order_on_next_page():
    app = AppTest.from_function(_flat_pagination_app, default_timeout=30).run()
    assert not app.exception and not app.error
    page = app.number_input(key="chart-result-page-flat-pagination-0")
    assert page.value == 1
    newest = _periods(app)
    assert len(newest) == 500
    assert newest["start_index"].tolist() == list(range(1000, 0, -2))
    page.set_value(2).run()
    assert not app.exception and not app.error
    oldest = _periods(app)
    assert oldest["start_index"].tolist() == [0]
    assert oldest["end_index"].tolist() == [1]
