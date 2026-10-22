"""The Databento page reviews costs before authorizing resumable downloads."""

from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import BatchFetchReport
from data_pipeline.providers import BaseDataProvider, ProviderRegistry
from research import Research


class CatalogProvider(BaseDataProvider):
    name = "databento"
    supported_timeframes = frozenset({"1m", "1h"})

    def __init__(self):
        self.catalog_calls = 0

    def list_datasets(self):
        self.catalog_calls += 1
        return ["XNAS.ITCH", "EQUS.MINI"]

    def fetch(self, request):
        raise AssertionError("The dashboard must use the reviewed batch job, not individual fetches")


@pytest.fixture
def dashboard_batch(tmp_path, monkeypatch):
    import dashboard.app as dashboard

    adapter = CatalogProvider()
    research = Research(tmp_path / "data", tmp_path / "runs", providers=ProviderRegistry({"databento": adapter}))
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    plans, estimates, advances = {}, [], []

    def estimate(**config):
        estimates.append(deepcopy(config))
        plan_id = f"plan-{len(estimates)}"
        plan = {"id": plan_id, "dataset": config["dataset"], "timeframe": config["timeframe"],
                "tickers": config["tickers"], "start": config["start"].isoformat(),
                "end": config["end"].isoformat(), "estimated_cost": 2.50, "status": "estimated",
                "warnings": [], "groups": [{"id": "group-1", "job_id": None, "status": "estimated",
                                             "symbols": config["tickers"], "start": config["start"].isoformat(),
                                             "end": config["end"].isoformat(), "estimated_cost": 2.50}]}
        plans[plan_id] = plan
        return deepcopy(plan)

    def advance(plan_id, *, submit=False, progress=None):
        advances.append((plan_id, submit))
        plan = plans[plan_id]
        plan["status"] = "running" if submit else "completed"
        plan["groups"][0].update(status="submitted" if submit else "decoded", job_id="vendor-job-1",
                                 remote_status="queued" if submit else "done")
        if not submit and progress is not None:
            progress(dict(downloaded_bytes=1024, total_bytes=2048, completed_files=1, total_files=2))
            progress(dict(downloaded_bytes=2048, total_bytes=2048, completed_files=2, total_files=2))
        return deepcopy(plan)

    pipeline = research.pipeline
    monkeypatch.setattr(pipeline, "estimate_databento", estimate, raising=False)
    monkeypatch.setattr(pipeline, "advance_databento", advance, raising=False)
    monkeypatch.setattr(pipeline, "databento_plan", lambda plan_id: deepcopy(plans[plan_id]), raising=False)
    monkeypatch.setattr(pipeline, "databento_plans", lambda: list(deepcopy(plans).values()), raising=False)
    monkeypatch.setattr(pipeline, "databento_report", lambda plan_id: BatchFetchReport(()), raising=False)

    def open_page():
        app = AppTest.from_string("from dashboard.app import main\nmain()").run()
        app.sidebar.radio[0].set_value("Fetch").run()
        return app

    return open_page, adapter, plans, estimates, advances


def prepare_estimate(app, tickers="AAPL, MSFT"):
    app.button(key="db-load-datasets").click().run()
    app.selectbox(key="db-dataset").set_value("XNAS.ITCH").run()
    app.text_area(key="db-tickers").set_value(tickers)
    app.button(key="db-estimate").click().run()
    assert not app.exception


def test_open_page_and_estimate_never_submit_data_requests(dashboard_batch):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    app = open_page()
    assert not app.exception
    assert adapter.catalog_calls == 0
    assert not estimates and not advances
    assert app.button(key="db-estimate").disabled
    prepare_estimate(app, ",".join(f"TICK{i}" for i in range(500)))
    assert adapter.catalog_calls == 1
    assert len(estimates[0]["tickers"]) == 500
    assert estimates[0]["timeframe"] == "1m"
    assert estimates[0]["overlap_policy"] == "fetch_missing"
    assert estimates[0]["max_estimated_cost"] == 10.0
    assert not advances
    assert app.metric[0].value == "$2.5000"
    assert not app.button(key="db-submit-plan-1").disabled
    app.button(key="db-submit-plan-1").click().run()
    assert not app.exception
    assert advances == [("plan-1", True)]


@pytest.mark.parametrize("field,value", [("db-timeframe", "1h"), ("db-tickers", "NVDA"),
                                        ("db-cost-limit", 1.0), ("db-dataset", "EQUS.MINI")])
def test_changed_inputs_invalidate_estimate(dashboard_batch, field, value):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    widgets = app.text_area if field == "db-tickers" else app.number_input if field == "db-cost-limit" else app.selectbox
    widgets(key=field).set_value(value).run()
    assert not app.exception
    assert app.button(key="db-submit-plan-1").disabled
    assert not advances


def test_new_dashboard_session_resumes_authorized_job_without_resubmission(dashboard_batch):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    app.button(key="db-submit-plan-1").click().run()
    reopened = open_page()
    assert not reopened.exception
    assert advances == [("plan-1", True), ("plan-1", False)]
    assert adapter.catalog_calls == 1
    assert any("Download complete" in item.value for item in reopened.success)
    reopened.run()
    assert len(advances) == 2


def test_live_transfer_progress_separates_vendor_state_from_local_stage(dashboard_batch):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    app.button(key="db-submit-plan-1").click().run()
    app.run()
    assert not app.exception
    progress = app.get("progress")
    assert len(progress) == 1
    assert progress[0].proto.value == 100
    assert "2/2 verified" in progress[0].proto.text
    assert "2.0 KiB / 2.0 KiB" in progress[0].proto.text
    table = app.dataframe[0].value
    assert table["databento_state"].tolist() == ["done"]
    assert "local_stage" in table
    assert advances == [("plan-1", True), ("plan-1", False)]


def test_queued_job_explains_why_transfer_progress_has_not_started(dashboard_batch):
    open_page, _, _, _, _ = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    app.button(key="db-submit-plan-1").click().run()
    assert not app.exception
    assert any("Waiting for Databento: **queued**" in item.value
               and "File transfer has not started" in item.value for item in app.markdown)
    assert not app.get("progress")


@pytest.mark.parametrize("status,plan_status", [
    ("downloaded", "running"), ("decoded", "running"), ("decoded", "completed"),
])
def test_verified_file_totals_remain_visible_after_transfer_and_rerun(dashboard_batch, status, plan_status):
    open_page, _, plans, _, advances = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    plan = plans["plan-1"]
    plan["status"] = plan_status
    plan["groups"][0].update(status=status, remote_status="done", download_progress={
        "downloaded_bytes": 2048, "total_bytes": 2048, "completed_files": 2, "total_files": 2,
    })
    app.session_state["db-auto-plan-1"] = False
    for _ in range(2):
        app.run()
        assert not app.exception
        progress = app.get("progress")
        assert len(progress) == 1
        assert progress[0].proto.value == 100
        assert "2/2 verified" in progress[0].proto.text
        assert "2.0 KiB / 2.0 KiB" in progress[0].proto.text
    assert not advances


def test_reopened_draft_requires_a_current_estimate(dashboard_batch):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    reopened = open_page()
    assert not reopened.exception
    assert reopened.button(key="db-submit-plan-1").disabled
    assert not advances


def test_estimate_over_budget_cannot_be_submitted(dashboard_batch):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    app = open_page()
    prepare_estimate(app)
    plans["plan-1"]["max_estimated_cost"] = 1.0
    app.run()
    assert not app.exception
    assert app.button(key="db-submit-plan-1").disabled
    assert any("exceeds your maximum" in item.value for item in app.warning)
    assert not advances


def test_metadata_failure_is_displayed_without_submitting(dashboard_batch, monkeypatch):
    open_page, adapter, plans, estimates, advances = dashboard_batch
    def fail():
        raise ValueError("Set DATABENTO_API_KEY before requesting data")
    monkeypatch.setattr(adapter, "list_datasets", fail)
    app = open_page()
    app.button(key="db-load-datasets").click().run()
    assert not app.exception
    assert any("DATABENTO_API_KEY" in item.value for item in app.error)
    assert not advances
