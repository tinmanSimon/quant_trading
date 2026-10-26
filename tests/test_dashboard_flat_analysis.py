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


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("maximum,intervals", [
    (200, [(12, 17), (6, 11), (0, 5)]),
    (4, [(14, 17), (8, 11), (2, 5)]),
])
def test_market_data_flat_analysis_draws_final_periods_and_toggles_without_recalculation(
        tmp_path, monkeypatch, maximum, intervals, automatic):
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
    if automatic:
        app.selectbox(key=prefix + "-band_width_mode").select("automatic").run()
        app.checkbox(key=prefix + "-long_bar_filter").check().run()
    assert calculations == []
    app.button(key="chart-logic-apply").click().run()
    assert not app.exception and not app.error
    assert len(calculations) == 1
    assert calculations[0]["band_width_mode"] == ("automatic" if automatic else "fixed")
    assert calculations[0]["long_bar_filter"] is automatic
    periods = _periods(app)
    assert list(zip(periods["start_index"], periods["end_index"])) == intervals
    assert periods["bars"].between(3, maximum).all()
    assert periods["range_pct"].between(0, 2.0).all()
    if automatic:
        assert periods["max_range_pct"].tolist() == pytest.approx(2 * periods["range_pct"])
        assert periods["max_bar_range_pct"].tolist() == pytest.approx(periods["range_pct"])
        assert periods["median_bar_range_pct"].tolist() == pytest.approx(periods["range_pct"])
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


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("maximum,intervals", [
    (200, [(12, 17), (6, 11), (0, 5)]),
    (4, [(14, 17), (8, 11), (2, 5)]),
])
def test_analysis_page_runs_flat_periods_and_reopens_saved_tables(tmp_path, maximum, intervals, automatic):
    research = _research(tmp_path)
    script = f"""
from dashboard.pages.analysis import _analysis_page
from research import Research
_analysis_page(Research({str(research.pipeline.store.data_dir)!r}, {str(research.runs_dir)!r}))
"""
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception and not app.error
    # This synthetic flat-pattern fixture is intentionally not a full session.
    app.checkbox(key="batch-strict-validation-databento").uncheck().run()
    app.checkbox(key="batch-all-tickers").check().run()
    app.selectbox(key="batch-analysis-choice").select((FlatLogic.name, FlatLogic.version))
    app.button(key="batch-analysis-add").click().run()
    widget = next(item for item in app.number_input if item.key.endswith("-min_duration"))
    prefix = widget.key.removesuffix("-min_duration")
    _set_calculation(app, prefix, maximum)
    if automatic:
        app.selectbox(key=prefix + "-band_width_mode").select("automatic").run()
        app.number_input(key=prefix + "-auto_percentile").set_value(75.)
        app.number_input(key=prefix + "-auto_multiplier").set_value(1.5)
        app.checkbox(key=prefix + "-long_bar_filter").check().run()
        app.number_input(key=prefix + "-long_bar_multiplier").set_value(1.25)
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
    if automatic:
        assert periods["max_range_pct"].tolist() == pytest.approx(1.5 * periods["range_pct"])
    assert {"lower_price", "upper_price", "center_price", "range_pct"} <= set(periods.columns)
    app.button(key="batch-save").click().run()
    assert not app.exception and not app.error
    store = BatchStore(research.runs_dir / "analysis")
    manifest, = store.list_runs()
    assert manifest["counts"]["completed"] == 1
    assert manifest["specs"][0]["name"] == "flat"
    assert manifest["specs"][0]["config"]["max_duration"] == maximum
    assert manifest["specs"][0]["config"]["max_range_pct"] == 2.0
    config = manifest["specs"][0]["config"]
    assert config["band_width_mode"] == ("automatic" if automatic else "fixed")
    assert config["long_bar_filter"] is automatic
    assert "auto_lookback" not in config
    if automatic:
        assert config["auto_percentile"] == 75.
        assert config["auto_multiplier"] == 1.5
        assert config["long_bar_multiplier"] == 1.25
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


def _flat_options_app():
    import streamlit as st
    from dashboard.components.analysis_selection import _configuration
    from dashboard.workspace import market_memory
    from research.analysis import discover_logics
    from research.analysis.builtins.flat import FlatLogic

    registry = discover_logics()
    definition = registry.definition("flat", FlatLogic.version)
    memory = market_memory("flat-options")
    values = memory.values.setdefault("config", registry.defaults("flat", FlatLogic.version))
    if st.session_state.get("invalid_auto"):
        values["auto_percentile"] = 101.
    try:
        memory.values["config"] = _configuration(definition, memory, "flat-options", values)
    except ValueError:
        pass
    st.session_state["config"] = memory.values["config"]


def test_flat_mode_and_filter_controls_preserve_inactive_values():
    app = AppTest.from_function(_flat_options_app).run()
    assert not app.exception and not app.error
    assert not app.number_input(key="flat-options-max_range_pct").disabled
    assert app.number_input(key="flat-options-auto_percentile").disabled
    assert app.number_input(key="flat-options-auto_multiplier").disabled
    assert not any(item.key.endswith("-auto_lookback") for item in app.number_input)
    assert app.number_input(key="flat-options-long_bar_multiplier").disabled
    app.number_input(key="flat-options-max_range_pct").set_value(.7)
    app.selectbox(key="flat-options-band_width_mode").select("automatic").run()
    assert app.number_input(key="flat-options-max_range_pct").disabled
    assert not app.number_input(key="flat-options-auto_percentile").disabled
    app.number_input(key="flat-options-auto_percentile").set_value(80.)
    app.number_input(key="flat-options-auto_multiplier").set_value(1.5)
    app.checkbox(key="flat-options-long_bar_filter").check().run()
    assert not app.number_input(key="flat-options-long_bar_multiplier").disabled
    app.number_input(key="flat-options-long_bar_multiplier").set_value(3.)
    app.selectbox(key="flat-options-band_width_mode").select("fixed")
    app.checkbox(key="flat-options-long_bar_filter").uncheck().run()
    assert not app.exception and not app.error
    assert app.number_input(key="flat-options-max_range_pct").value == .7
    assert app.number_input(key="flat-options-long_bar_multiplier").disabled
    app.selectbox(key="flat-options-band_width_mode").select("automatic")
    app.checkbox(key="flat-options-long_bar_filter").check().run()
    assert not app.exception and not app.error
    assert app.number_input(key="flat-options-auto_percentile").value == 80.
    assert app.number_input(key="flat-options-auto_multiplier").value == 1.5
    assert "auto_lookback" not in app.session_state["config"]
    assert app.number_input(key="flat-options-long_bar_multiplier").value == 3.
    assert app.session_state["config"]["max_range_pct"] == .7


def test_disabled_automatic_controls_report_invalid_saved_values():
    app = AppTest.from_function(_flat_options_app)
    app.session_state["invalid_auto"] = True
    app.run()
    assert not app.exception
    assert any("saved inactive value is invalid" in item.value for item in app.error)
    assert app.session_state["config"]["auto_percentile"] == 101.


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
