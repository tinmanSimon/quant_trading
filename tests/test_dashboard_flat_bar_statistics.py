"""The Analysis page discovers, runs, summarizes, and saves flat-bar statistics."""

from datetime import timedelta

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.batch_analysis.state import BatchJob
from data_pipeline import DataRequest
from research import Research
from research.analysis.batch import BatchStore
from research.analysis.builtins.flat_bar_statistics import FlatBarStatisticsLogic

from test_flat_bar_statistics import START, sample_bars


def _ticker_summary(app):
    return next(item.value for item in app.dataframe
                if "source_bar_count" in item.value.columns and "ticker" not in item.value.columns)


def _period_details(app):
    return next(item.value for item in app.dataframe if "period_id" in item.value.columns)


def _recent_summary(app):
    return next(item.value for item in app.dataframe
                if "last_month_cutoff" in item.value.columns and "ticker" not in item.value.columns)


@pytest.mark.parametrize("automatic,percentage,red_above,all_above", [
    (False, 102., 3, 5), (True, 103., 0, 0),
])
def test_analysis_page_runs_new_statistics_and_reopens_saved_results(
        tmp_path, automatic, percentage, red_above, all_above):
    research = Research(tmp_path / "data", tmp_path / "runs")
    for ticker in ("AAA", "BBB"):
        research.pipeline.ingest_frame(DataRequest(
            ticker, START, START + timedelta(hours=5), provider="databento", timeframe="30m"),
            sample_bars(ticker))
    script = f"""
from dashboard.pages.analysis import _analysis_page
from research import Research
_analysis_page(Research({str(research.pipeline.store.data_dir)!r}, {str(research.runs_dir)!r}))
"""
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception and not app.error
    # This synthetic screening fixture is intentionally not a full session.
    app.checkbox(key="batch-strict-validation-databento").uncheck().run()
    app.checkbox(key="batch-all-tickers").check().run()
    app.selectbox(key="batch-analysis-choice").select((FlatBarStatisticsLogic.name, FlatBarStatisticsLogic.version))
    app.button(key="batch-analysis-add").click().run()
    widget = next(item for item in app.number_input if item.key.endswith("-high_percentage"))
    prefix = widget.key.removesuffix("-high_percentage")
    assert widget.value == 102.
    app.number_input(key=prefix + "-high_percentage").set_value(percentage)
    app.number_input(key=prefix + "-min_duration").set_value(3)
    app.number_input(key=prefix + "-max_duration").set_value(20)
    app.number_input(key=prefix + "-max_range_pct").set_value(6.)
    if automatic:
        app.selectbox(key=prefix + "-band_width_mode").select("automatic").run()
        assert app.number_input(key=prefix + "-max_range_pct").disabled
        app.number_input(key=prefix + "-auto_percentile").set_value(75.)
        app.number_input(key=prefix + "-auto_multiplier").set_value(1.5)
        app.checkbox(key=prefix + "-long_bar_filter").check().run()
        app.number_input(key=prefix + "-long_bar_multiplier").set_value(1.4)
    assert not any(item.key.endswith("-auto_lookback") for item in app.number_input)
    app.button(key="batch-run").click().run()
    jobs = [item for item in app.session_state.filtered_state.values() if isinstance(item, BatchJob)]
    assert len(jobs) == 1
    jobs[0].thread.join(timeout=10)
    assert not jobs[0].running and jobs[0].error is None
    app.run()
    assert not app.exception and not app.error
    summary = _ticker_summary(app)
    assert summary["flat_bar_count"].tolist() == [8]
    assert summary["red_bar_count"].tolist() == [5]
    assert summary["red_high_above_count"].tolist() == [red_above]
    assert summary["red_high_not_above_count"].tolist() == [5 - red_above]
    assert summary["all_high_above_count"].tolist() == [all_above]
    assert summary["return_count"].tolist() == [6]
    assert summary["return_volatility_pct"].iloc[0] > 0
    recent = _recent_summary(app)
    assert recent["last_month_flat_bar_count"].tolist() == [8]
    assert recent["last_month_red_high_above_count"].tolist() == [red_above]
    assert recent["last_month_return_volatility_pct"].tolist() == summary["return_volatility_pct"].tolist()
    assert recent["last_month_start"].tolist() == [START.isoformat()]
    details = _period_details(app)
    assert list(zip(details["start_index"], details["end_index"])) == [(6, 8), (0, 4)]
    ranking = next(item.value for item in app.dataframe if "ticker" in item.value.columns)
    assert set(ranking["ticker"]) == {"AAA", "BBB"}
    assert ranking["red_high_above_count"].tolist() == [red_above, red_above]
    assert ranking["last_month_red_high_above_count"].tolist() == [red_above, red_above]

    app.button(key="batch-save").click().run()
    assert not app.exception and not app.error
    store = BatchStore(research.runs_dir / "analysis")
    manifest, = store.list_runs()
    assert manifest["counts"]["completed"] == 2
    settings = manifest["specs"][0]["config"]
    assert settings["high_percentage"] == percentage
    assert settings["band_width_mode"] == ("automatic" if automatic else "fixed")
    assert settings["long_bar_filter"] is automatic
    if automatic:
        assert settings["auto_percentile"] == 75.
        assert settings["auto_multiplier"] == 1.5
        assert settings["long_bar_multiplier"] == 1.4
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert reopened.selectbox(key="batch-saved-run").value == manifest["run_id"]
    assert _ticker_summary(reopened).equals(summary)
    assert _period_details(reopened).equals(details)
    assert _recent_summary(reopened).equals(recent)
