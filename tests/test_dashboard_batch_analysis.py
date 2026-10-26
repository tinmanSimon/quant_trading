"""The Analysis page runs once, then ranks and reopens persisted text results."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from research import Research
from research.analysis import BaseChartLogic, LogicRegistry, LogicResult
from research.analysis.batch import BatchStore
from dashboard.batch_analysis.state import BatchJob


@pytest.fixture
def batch_page(tmp_path, monkeypatch):
    from dashboard.pages import analysis as page

    calls = []

    class Outcomes(BaseChartLogic):
        name = "outcomes"
        label = "Example outcomes"
        config_schema = {"type": "object", "additionalProperties": False,
                         "properties": {"factor": {"type": "integer", "default": 1, "minimum": 1}}}

        def calculate(self, context, config, progress=None):
            symbol = context.frame["symbol"][0]
            calls.append((symbol, config["factor"]))
            wins = (2 if symbol == "AAA" else 5) * config["factor"]
            return LogicResult(
                tables=[{"name": "Outcome summary", "columns": ["outcome", "count"], "rows": [
                    {"outcome": "win", "count": wins}, {"outcome": "loss", "count": 1},
                    {"outcome": "inconclusive", "count": 0}, {"outcome": "total", "count": wins + 1},
                ]}, {"name": "Detail", "columns": ["duration"],
                     "rows": [{"duration": 2}, {"duration": 8}]}],
                data={"analysis": {"text": [f"Saved report for {symbol}."]}},
            )

    registry = LogicRegistry()
    registry.register_class(Outcomes)
    monkeypatch.setattr(page, "load_registry", lambda **kwargs: registry)
    research = Research(tmp_path / "data", tmp_path / "runs")
    stamps = [datetime(2024, 1, 2, 15, tzinfo=UTC) + timedelta(minutes=index) for index in range(5)]
    for symbol in ["AAA", "BBB"]:
        frame = pl.DataFrame({"timestamp": stamps, "symbol": [symbol] * 5,
                              "open": [10.] * 5, "high": [11.] * 5,
                              "low": [9.] * 5, "close": [10.] * 5, "volume": [100.] * 5})
        research.pipeline.ingest_frame(DataRequest(symbol, stamps[0], stamps[-1] + timedelta(minutes=1),
                                                  timeframe="1m"), frame)
    script = f"""
from dashboard.pages.analysis import _analysis_page
from research import Research
_analysis_page(Research({str(research.pipeline.store.data_dir)!r}, {str(research.runs_dir)!r}))
"""
    return AppTest.from_string(script, default_timeout=30).run(), calls, script, research


def _finish(app):
    jobs = [value for value in app.session_state.filtered_state.values() if isinstance(value, BatchJob)]
    assert len(jobs) == 1
    jobs[0].thread.join(timeout=10)
    assert not jobs[0].running and jobs[0].error is None
    app.run()
    assert not app.exception and not app.error, [item.value for item in app.error]


def _add_complete_session(research, provider):
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    stamps = [start + timedelta(minutes=index) for index in range(390)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["FULL"] * 390,
                          "open": [10.] * 390, "high": [11.] * 390,
                          "low": [9.] * 390, "close": [10.] * 390, "volume": [100.] * 390})
    research.pipeline.ingest_frame(DataRequest("FULL", start, start + timedelta(minutes=390),
                                               provider=provider, timeframe="1m"), frame)


def test_strict_validation_defaults_and_overrides_persist_per_provider(batch_page):
    app, _, script, research = batch_page
    assert app.checkbox(key="batch-strict-validation-yahoo").value is False
    _add_complete_session(research, "databento-xnas.itch")
    app.run()
    app.selectbox(key="batch-provider").set_value("databento-xnas.itch").run()
    assert app.checkbox(key="batch-strict-validation-databento-xnas.itch").value is True
    assert app.text_input(key="batch-validation-calendar-databento-xnas.itch").value == "XNYS"
    app.checkbox(key="batch-strict-validation-databento-xnas.itch").uncheck().run()
    app.text_input(key="batch-session-timezone-databento-xnas.itch").set_value("UTC").run()
    app.selectbox(key="batch-provider").set_value("yahoo").run()
    assert app.checkbox(key="batch-strict-validation-yahoo").value is False
    app.checkbox(key="batch-strict-validation-yahoo").check().run()
    app.text_input(key="batch-validation-calendar-yahoo").set_value("XNAS").run()
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert reopened.checkbox(key="batch-strict-validation-yahoo").value is True
    assert reopened.text_input(key="batch-validation-calendar-yahoo").value == "XNAS"
    assert reopened.text_input(key="batch-session-timezone-yahoo").value == "America/New_York"
    reopened.selectbox(key="batch-provider").set_value("databento-xnas.itch").run()
    assert reopened.checkbox(key="batch-strict-validation-databento-xnas.itch").value is False
    assert reopened.text_input(key="batch-session-timezone-databento-xnas.itch").value == "UTC"


@pytest.mark.parametrize("provider,strict_validation", [
    ("yahoo", True), ("databento-xnas.itch", False), ("databento-xnas.itch", True),
])
def test_analysis_validation_policy_survives_run_save_and_reopen(
    batch_page, provider, strict_validation,
):
    app, calls, script, research = batch_page
    _add_complete_session(research, provider)
    app.run()
    app.selectbox(key="batch-provider").set_value(provider).run()
    app.multiselect(key="batch-tickers").set_value(["FULL"]).run()
    app.checkbox(key=f"batch-strict-validation-{provider}").set_value(strict_validation).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    app.button(key="batch-save").click().run()
    manifest = BatchStore(research.runs_dir / "analysis").list_runs()[0]
    assert manifest["request"]["strict_validation"] is strict_validation
    assert manifest["request"]["calendar"] == "XNYS"
    assert manifest["request"]["session_timezone"] == "America/New_York"
    assert manifest["counts"]["completed"] == 1
    assert calls == [("FULL", 1)]
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert reopened.checkbox(key=f"batch-strict-validation-{provider}").value is strict_validation
    assert reopened.selectbox(key="batch-saved-run").value == manifest["run_id"]
    assert any("Saved report for FULL" in item.value for item in reopened.markdown)
    assert calls == [("FULL", 1)]


def test_screening_omission_warning_survives_save_and_reopen(batch_page):
    from data_pipeline import FetchQuality, OmittedBar

    app, calls, script, research = batch_page
    source = next(item for item in research.pipeline.list_datasets() if item.request.symbol == "AAA")
    frame = research.pipeline.read_dataset(source.dataset_id)
    omitted = frame["timestamp"][2]
    research.pipeline.store.replace_raw(
        source.dataset_id, frame.filter(pl.col("timestamp") != omitted), confirm=source.dataset_id,
        quality=FetchQuality(status="reported", skip_missing_ohlc=False, skip_invalid_bars=True,
                             omitted_bars=(OmittedBar(omitted, "invalid_bar: open exceeds high"),)),
    )
    app.run()
    assert app.checkbox(key="batch-strict-validation-yahoo").value is False
    app.multiselect(key="batch-tickers").set_value(["AAA"]).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    warnings = [item.value for item in app.warning if "recorded omitted bars" in item.value]
    assert len(warnings) == 1
    assert omitted.isoformat() in warnings[0]
    assert "Screening results use incomplete history" in warnings[0]
    ranking = next(item.value for item in app.dataframe if "ticker" in item.value.columns)
    assert ranking.set_index("ticker").loc["AAA", "data_validation"] == "not_checked"
    assert ranking.set_index("ticker").loc["AAA", "data_warnings"] == 1
    app.button(key="batch-save").click().run()
    store = BatchStore(research.runs_dir / "analysis")
    manifest = store.list_runs()[0]
    assert manifest["tickers"]["AAA"]["status"] == "completed"
    assert manifest["tickers"]["AAA"]["data_validation"] == {"status": "not_checked"}
    assert manifest["tickers"]["AAA"]["data_warnings"] == warnings
    detail = store.load_ticker(manifest["run_id"], "AAA")
    integrity = detail["results"][manifest["specs"][0]["instance_id"]]["data"]["data_integrity"]
    assert integrity["unresolved_omitted_bar_timestamps"] == [omitted.isoformat()]
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert [item.value for item in reopened.warning if "recorded omitted bars" in item.value] == warnings
    ranking = next(item.value for item in reopened.dataframe if "ticker" in item.value.columns)
    assert ranking.set_index("ticker").loc["AAA", "data_validation"] == "not_checked"
    assert ranking.set_index("ticker").loc["AAA", "data_warnings"] == 1
    assert reopened.selectbox(key="batch-saved-run").value == manifest["run_id"]
    assert calls == [("AAA", 1)]


def test_batch_page_runs_ranks_customizes_and_reopens_without_execution(batch_page):
    app, calls, script, research = batch_page
    assert not app.exception and not app.error
    app.checkbox(key="batch-all-tickers").check().run()
    app.text_area(key="batch-extra-tickers").set_value("MISSING, AAA").run()
    app.button(key="batch-analysis-add").click().run()
    assert not app.exception and calls == []
    app.button(key="batch-run").click().run()
    _finish(app)
    app.button(key="batch-save").click().run()
    assert calls == [("AAA", 1), ("BBB", 1)]
    store = BatchStore(research.runs_dir / "analysis")
    manifest = store.list_runs()[0]
    assert manifest["counts"]["completed"] == 2
    assert manifest["counts"]["no_data"] == 1
    spec = manifest["specs"][0]
    prefix = f"batch-result-{manifest['run_id']}-{spec['instance_id']}"
    overall = next(item.value for item in app.dataframe if "contributors" in item.value.columns)
    assert overall.set_index("metric").loc["win", "value"] == 7
    assert overall.set_index("metric").loc["total", "value"] == 9
    app.selectbox(key=prefix + "-rank").set_value("win").run()
    ranking = next(item.value for item in app.dataframe if "ticker" in item.value.columns)
    assert ranking["ticker"].tolist() == ["BBB", "AAA", "MISSING"]
    app.selectbox(key=prefix + "-detail-ticker").set_value("BBB").run()
    assert any("Saved report for BBB" in item.value for item in app.markdown)
    app.selectbox(key=prefix + "-aggregation-0-win").set_value("mean").run()
    overall = next(item.value for item in app.dataframe if "contributors" in item.value.columns)
    assert overall.set_index("metric").loc["win", "value"] == 3.5
    assert len(calls) == 2
    # A new session restores the selected run and summary preferences from disk.
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert reopened.selectbox(key="batch-saved-run").value == manifest["run_id"]
    assert reopened.selectbox(key=prefix + "-aggregation-0-win").value == "mean"
    assert len(calls) == 2


def test_batch_page_duplicate_analysis_instances_have_independent_settings(batch_page):
    app, calls, _, research = batch_page
    app.multiselect(key="batch-tickers").set_value(["AAA"]).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-analysis-add").click().run()
    factors = [item for item in app.number_input if item.key.endswith("-factor")]
    assert len(factors) == 2
    factors[1].set_value(3).run()
    app.button(key="batch-run").click().run()
    _finish(app)
    app.button(key="batch-save").click().run()
    assert calls == [("AAA", 1), ("AAA", 3)]
    manifest = BatchStore(research.runs_dir / "analysis").list_runs()[0]
    assert len(manifest["specs"]) == 2
    assert len({spec["instance_id"] for spec in manifest["specs"]}) == 2


def test_batch_history_remains_accessible_without_available_data(batch_page, monkeypatch):
    app, calls, script, research = batch_page
    app.multiselect(key="batch-tickers").set_value(["AAA"]).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    app.button(key="batch-save").click().run()
    from data_pipeline import DataPipeline
    monkeypatch.setattr(DataPipeline, "list_datasets", lambda *args, **kwargs: [])
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception and not reopened.error
    assert reopened.selectbox(key="batch-saved-run").value
    assert any("Saved report for AAA" in item.value for item in reopened.markdown)
    assert calls == [("AAA", 1)]


def test_custom_metric_can_be_added_removed_and_reset_without_execution(batch_page):
    app, calls, _, research = batch_page
    app.checkbox(key="batch-all-tickers").check().run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    app.button(key="batch-save").click().run()
    store = BatchStore(research.runs_dir / "analysis")
    manifest = store.list_runs()[0]
    prefix = f"batch-result-{manifest['run_id']}-{manifest['specs'][0]['instance_id']}"
    app.checkbox(key=prefix + "-custom").check().run()
    app.selectbox(key=prefix + "-table").set_value("Detail").run()
    app.text_input(key=prefix + "-metric-name").set_value("average_duration")
    app.selectbox(key=prefix + "-reducer").set_value("mean")
    app.selectbox(key=prefix + "-aggregate").set_value("mean")
    next(button for button in app.button if button.label == "Add summary metric").click().run()
    assert not app.exception and not app.error
    overall = next(item.value for item in app.dataframe if "contributors" in item.value.columns)
    assert overall.set_index("metric").loc["average_duration", "value"] == 5
    # A newly added control is rendered on the following ordinary rerun.
    app.run()
    next(button for button in app.button if button.label == "Remove average_duration").click().run()
    overall = next(item.value for item in app.dataframe if "contributors" in item.value.columns)
    assert "average_duration" not in overall["metric"].tolist()
    app.selectbox(key=prefix + "-aggregation-0-win").set_value("mean").run()
    app.button(key=prefix + "-reset-metrics").click().run()
    assert not app.exception and not app.error
    overall = next(item.value for item in app.dataframe if "contributors" in item.value.columns)
    assert overall.set_index("metric").loc["win", "value"] == 7
    assert len(calls) == 2


def test_bad_summary_settings_do_not_hide_original_reports(batch_page):
    app, calls, _, research = batch_page
    app.multiselect(key="batch-tickers").set_value(["AAA"]).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    app.button(key="batch-save").click().run()
    store = BatchStore(research.runs_dir / "analysis")
    manifest = store.list_runs()[0]
    instance = manifest["specs"][0]["instance_id"]
    store.save_aggregation(manifest["run_id"], {"instances": {instance: [{"name": "invalid"}]}})
    app.run()
    assert not app.exception
    assert any("Could not summarize" in item.value for item in app.error)
    assert any("Saved report for AAA" in item.value for item in app.markdown)
    app.button(key=f"batch-result-{manifest['run_id']}-{instance}-clear-invalid").click().run()
    assert not app.exception and not app.error and len(calls) == 1


def test_custom_metric_failure_is_isolated_and_finished_ticker_values_are_cached(tmp_path):
    from copy import deepcopy
    from dashboard.batch_analysis.results import metric_values
    from research.analysis.batch.aggregation import MetricSpec, extract_metrics, infer_metrics

    tables = [{"name": "Summary", "rows": [{"outcome": "win", "count": 3}]},
              {"name": "Detail", "rows": [{"duration": 2}, {"duration": 8}]}]
    defaults = infer_metrics(tables)
    status = {"detail_file": "detail", "detail_sha256": "checksum", "summaries": {
        "instance": {"metrics": [spec.to_dict() for spec in defaults], "values": extract_metrics(tables, defaults)},
    }}
    manifest = {"run_id": "run", "tickers": {"AAA": status}}

    class Store:
        directory = tmp_path
        calls = []

        def load_ticker(self, run_id, ticker):
            self.calls.append(ticker)
            return {"results": {"instance": {"tables": tables}}}

    store = Store()
    specs = [*defaults, MetricSpec("bad_duration", "Detail", "duration", reducer="single")]
    cache = {}
    values, errors = metric_values(store, manifest, "instance", specs, cache=cache)
    assert values["AAA"] == {"win": 3, "bad_duration": None}
    assert "single" in errors["AAA"]
    # Progress on another ticker does not reread an unchanged saved detail file.
    manifest["tickers"]["BBB"] = deepcopy(status)
    values, errors = metric_values(store, manifest, "instance", specs, cache=cache)
    assert store.calls == ["AAA", "BBB"]
    assert values["BBB"]["win"] == 3 and "single" in errors["BBB"]


def test_analysis_is_temporary_until_saved_and_can_be_deleted(batch_page):
    app, calls, script, research = batch_page
    app.multiselect(key="batch-tickers").set_value(["AAA"]).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    store = BatchStore(research.runs_dir / "analysis")
    assert store.list_runs() == []
    assert any("Saved report for AAA" in item.value for item in app.markdown)
    reopened = AppTest.from_string(script, default_timeout=30).run()
    assert not reopened.exception
    assert not any("Saved report for AAA" in item.value for item in reopened.markdown)
    app.button(key="batch-save").click().run()
    assert not app.exception and not app.error
    manifest = store.list_runs()[0]
    assert store.load_ticker(manifest["run_id"], "AAA")["ticker"] == "AAA"
    app.button(key="batch-delete").click().run()
    assert not app.exception and not app.error
    assert store.list_runs() == []
    assert not (store.directory / manifest["run_id"]).exists()
    assert calls == [("AAA", 1)]


def test_starting_another_run_discards_unsaved_results(batch_page):
    app, calls, _, research = batch_page
    app.multiselect(key="batch-tickers").set_value(["AAA"]).run()
    app.button(key="batch-analysis-add").click().run()
    app.button(key="batch-run").click().run()
    _finish(app)
    previous = next(value for value in app.session_state.filtered_state.values() if isinstance(value, BatchJob))
    temporary_path = previous.runner.store.directory
    app.button(key="batch-run").click().run()
    _finish(app)
    assert not temporary_path.exists()
    assert BatchStore(research.runs_dir / "analysis").list_runs() == []
    assert calls == [("AAA", 1), ("AAA", 1)]
