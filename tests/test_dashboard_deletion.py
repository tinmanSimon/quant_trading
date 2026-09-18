"""Physical deletion requires an exact preview and explicit UI confirmation."""

from datetime import UTC, date, datetime, time
from types import SimpleNamespace

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataQuery, DataRequest
from data_pipeline.exceptions import StorageError
from research import Research


def _element(elements, label):
    return next(element for element in elements if element.label == label)


class FakePipeline:
    def __init__(self, root):
        self.store = SimpleNamespace(data_dir=root)
        request = DataRequest(symbol="AAPL", timeframe="1h",
                              start=datetime(2024, 1, 2, tzinfo=UTC),
                              end=datetime(2024, 1, 3, tzinfo=UTC))
        self.items = [SimpleNamespace(layer=layer, pipeline_id=pipeline_id, request=request,
                                     first_timestamp=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                                     last_timestamp=datetime(2024, 1, 2, 16, 30, tzinfo=UTC))
                      for layer, pipeline_id in (("raw", ""), ("processed", "pipeline-a"),
                                                  ("processed", "pipeline-b"))]
        self.previews, self.deletions, self.reports = [], [], {}
        self.removed_rows = 1
        self.delete_error = None

    def list_datasets(self, query):
        assert query.include_history
        return [item for item in self.items if item.layer == query.layer]

    def list_deletions(self, *, pending_only):
        assert pending_only
        return [report for report in self.reports.values() if report.status == "pending"]

    def plan_delete(self, query):
        self.previews.append(query)
        return SimpleNamespace(operation_id=f"plan-{len(self.previews)}", query=query,
                               removed_rows=self.removed_rows, retained_rows=2, source_bytes=1000,
                               estimated_temporary_bytes=2000,
                               to_frame=lambda: pl.DataFrame({"dataset_id": ["source"], "removed_rows": [1]}))

    def report(self, operation_id, status="completed"):
        fields = dict(operation_id=operation_id, status=status, removed_rows=1,
                      deleted_ids=("source",), replacement_ids=("left", "right"),
                      removed_bytes=1000, written_bytes=500)
        result = SimpleNamespace(**fields, to_dict=lambda: fields.copy())
        self.reports[operation_id] = result
        return result

    def delete(self, plan, *, confirm):
        assert confirm == plan.operation_id
        self.deletions.append(plan)
        if self.delete_error is not None:
            raise self.delete_error
        return self.report(plan.operation_id)

    def deletion_status(self, operation_id):
        return self.reports[operation_id]

    def recover(self):
        for operation_id in self.reports:
            self.report(operation_id)
        return []


@pytest.fixture
def fake_page(tmp_path, monkeypatch):
    import dashboard.deletion as deletion
    pipeline = FakePipeline(tmp_path / "data")
    monkeypatch.setattr(deletion, "_test_research", SimpleNamespace(pipeline=pipeline), raising=False)
    app = AppTest.from_string(
        "import dashboard.deletion as deletion\ndeletion.deletion_page(deletion._test_research)"
    ).run()
    assert not app.exception
    return app, pipeline


def _preview(app):
    _element(app.button, "Preview deletion").click().run()
    assert not app.exception


def _confirm(app):
    _element(app.checkbox, "I understand this permanently deletes the rows shown above.").check().run()
    _element(app.button, "Delete permanently").click().run()
    assert not app.exception


def test_confirmation_required_and_exact_utc_interval_passed(fake_page):
    app, pipeline = fake_page
    assert not any(button.label == "Delete permanently" for button in app.button)
    _element(app.time_input, "Deletion start time (UTC)").set_value(time(15, 30))
    _element(app.date_input, "Deletion end date (UTC, exclusive)").set_value(date(2024, 1, 2))
    _element(app.time_input, "Deletion end time (UTC)").set_value(time(16, 30)).run()
    _preview(app)
    assert _element(app.button, "Delete permanently").disabled
    assert pipeline.previews[-1] == DataQuery(
        layer="raw", provider="yahoo", symbol="AAPL", timeframe="1h", include_history=True,
        start=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 16, 30, tzinfo=UTC))
    assert not pipeline.deletions
    _confirm(app)
    assert len(pipeline.deletions) == 1
    assert "Physical deletion completed" in app.success[0].value


def test_filter_change_discards_preview_and_confirmation(fake_page):
    app, pipeline = fake_page
    _preview(app)
    _element(app.checkbox, "I understand this permanently deletes the rows shown above.").check().run()
    _element(app.time_input, "Deletion start time (UTC)").set_value(time(1)).run()
    assert not any(button.label == "Delete permanently" for button in app.button)
    _preview(app)
    assert _element(app.button, "Delete permanently").disabled
    assert not pipeline.deletions


def test_data_root_change_cannot_execute_old_plan(fake_page, tmp_path, monkeypatch):
    import dashboard.deletion as deletion
    app, old = fake_page
    _preview(app)
    fresh = FakePipeline(tmp_path / "other-data")
    monkeypatch.setattr(deletion, "_test_research", SimpleNamespace(pipeline=fresh))
    app.run()
    assert not app.exception
    assert not any(button.label == "Delete permanently" for button in app.button)
    assert not old.deletions
    assert not fresh.deletions


def test_processed_selection_is_an_exact_pipeline_and_layer(fake_page):
    app, pipeline = fake_page
    _element(app.radio, "Delete from layer").set_value("processed").run()
    _element(app.selectbox, "Processor pipeline").set_value("pipeline-b").run()
    _preview(app)
    query = pipeline.previews[-1]
    assert query.layer == "processed"
    assert query.pipeline_id == "pipeline-b"
    assert query.include_history
    _confirm(app)
    assert pipeline.deletions[0].query == query


def test_no_match_or_invalid_period_cannot_be_confirmed(fake_page):
    app, pipeline = fake_page
    pipeline.removed_rows = 0
    _preview(app)
    assert any("Nothing will be deleted" in item.value for item in app.info)
    assert not app.checkbox
    _element(app.date_input, "Deletion end date (UTC, exclusive)").set_value(date(2024, 1, 1)).run()
    assert any("end must be later" in item.value for item in app.error)
    assert not any(button.label == "Preview deletion" for button in app.button)
    assert not pipeline.deletions


def test_stale_preview_failure_is_visible_and_consumes_confirmation(fake_page):
    app, pipeline = fake_page
    pipeline.delete_error = StorageError("Storage changed after this preview")
    _preview(app)
    _confirm(app)
    assert not app.success
    assert any("fresh preview" in item.value for item in app.error)
    app.run()
    assert not app.checkbox
    assert not any(button.label == "Delete permanently" for button in app.button)
    _preview(app)
    assert _element(app.button, "Delete permanently").disabled


def test_interrupted_cleanup_does_not_claim_success_and_can_be_recovered(fake_page):
    app, pipeline = fake_page
    _preview(app)
    error = StorageError("unlink failed")
    error.operation_id = "plan-1"
    pipeline.delete_error = error
    pipeline.report("plan-1", status="pending")
    # A previously previewed delete fails after publication, as if cleanup just failed.
    _element(app.checkbox, "I understand this permanently deletes the rows shown above.").check()
    _element(app.button, "Delete permanently").click().run()
    # Existing pending work also blocks a new operation even with a saved checkbox.
    assert not app.success
    assert any("physical cleanup is still pending" in item.value for item in app.warning)
    pipeline.items.clear()
    app.run()
    assert any("No local raw datasets" in item.value for item in app.info)
    _element(app.button, "Retry physical cleanup").click().run()
    assert not app.exception
    assert "Physical deletion completed" in app.success[0].value


def test_cleanup_error_reports_operation_without_claiming_completion(fake_page):
    app, pipeline = fake_page
    _preview(app)
    original = pipeline.delete

    def interrupted(plan, *, confirm):
        pipeline.report(plan.operation_id, "pending")
        error = StorageError("file in use")
        error.operation_id = plan.operation_id
        pipeline.delete_error = error
        return original(plan, confirm=confirm)

    pipeline.delete = interrupted
    _confirm(app)
    assert len(pipeline.deletions) == 1
    assert not app.success
    assert any("incomplete" in item.value for item in app.warning)
    assert any("plan-1" in item.value for item in app.caption)
    app.run()
    assert _element(app.button, "Preview deletion").disabled
    assert any(button.label == "Retry physical cleanup" for button in app.button)


def test_dashboard_deletes_only_selected_hourly_bars_physically(tmp_path, monkeypatch, sample_ohlcv_frame):
    import dashboard.app as dashboard
    from data_pipeline.processing import ScalePrices

    research = Research(tmp_path / "data", tmp_path / "runs")
    request = DataRequest(symbol="AAPL", timeframe="1h", start=datetime(2024, 1, 2, tzinfo=UTC),
                          end=datetime(2024, 1, 3, tzinfo=UTC))
    raw = research.pipeline.ingest_frame(request, sample_ohlcv_frame).raw
    processed = research.pipeline.process(raw.dataset_id, processors=[ScalePrices(factor=2)])
    processed_frame = research.pipeline.read_dataset(processed.dataset_id)
    raw_path = research.pipeline.store.data_dir / raw.relative_path
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Delete data").run()
    assert not app.exception
    assert not app.error
    _element(app.time_input, "Deletion start time (UTC)").set_value(time(15, 30))
    _element(app.date_input, "Deletion end date (UTC, exclusive)").set_value(date(2024, 1, 2))
    _element(app.time_input, "Deletion end time (UTC)").set_value(time(16, 30)).run()
    _preview(app)
    assert raw_path.exists()
    _confirm(app)
    assert not app.error
    assert "Physical deletion completed" in app.success[0].value
    assert not raw_path.exists()
    assert_frame_equal(research.pipeline.read(DataQuery(symbol="AAPL", timeframe="1h")),
                       sample_ohlcv_frame[[0, 2]])
    assert_frame_equal(research.pipeline.read_dataset(processed.dataset_id), processed_frame)


def test_empty_delete_page_is_available_from_navigation(tmp_path, monkeypatch):
    import dashboard.app as dashboard
    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Delete data").run()
    assert not app.exception
    assert not app.error
    assert any("No local raw datasets" in item.value for item in app.info)
    assert not any(button.label == "Delete permanently" for button in app.button)


def test_dashboard_processed_delete_keeps_raw_and_other_pipeline(tmp_path, monkeypatch, sample_ohlcv_frame):
    import dashboard.app as dashboard
    from data_pipeline.processing import ScalePrices

    research = Research(tmp_path / "data", tmp_path / "runs")
    request = DataRequest(symbol="AAPL", timeframe="1h", start=datetime(2024, 1, 2, tzinfo=UTC),
                          end=datetime(2024, 1, 3, tzinfo=UTC))
    raw = research.pipeline.ingest_frame(request, sample_ohlcv_frame).raw
    first = research.pipeline.process(raw.dataset_id, processors=[ScalePrices(factor=2)])
    second = research.pipeline.process(raw.dataset_id, processors=[ScalePrices(factor=3)])
    original_second = research.pipeline.read_dataset(second.dataset_id)
    removed_path = research.pipeline.store.data_dir / first.relative_path
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Delete data").run()
    _element(app.radio, "Delete from layer").set_value("processed").run()
    _element(app.selectbox, "Processor pipeline").set_value(first.pipeline_id).run()
    _preview(app)
    _confirm(app)
    assert not app.error
    assert "Physical deletion completed" in app.success[0].value
    assert not removed_path.exists()
    assert_frame_equal(research.pipeline.read_dataset(raw.dataset_id), sample_ohlcv_frame)
    assert_frame_equal(research.pipeline.read_dataset(second.dataset_id), original_second)


def test_dashboard_recovers_abandoned_preparation_without_claiming_rows_deleted(
    tmp_path, monkeypatch, sample_ohlcv_frame,
):
    import dashboard.app as dashboard
    from data_pipeline.storage import deletion
    from data_pipeline.exceptions import DatasetNotFoundError

    research = Research(tmp_path / "data", tmp_path / "runs")
    pipeline = research.pipeline
    raw = pipeline.ingest_frame(DataRequest(
        symbol="AAPL", timeframe="1h", start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 3, tzinfo=UTC)), sample_ohlcv_frame).raw
    plan = pipeline.plan_delete(DataQuery(
        provider="yahoo", symbol="AAPL", timeframe="1h",
        start=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 16, 30, tzinfo=UTC)))
    stage = deletion._stage_survivor

    def interrupted(*args, **kwargs):
        stage(*args, **kwargs)
        raise RuntimeError("interrupted while preparing survivors")

    # Leave a real owned preparation journal and file as after abrupt process
    # termination, bypassing normal error cleanup solely for this fixture.
    with monkeypatch.context() as failure:
        failure.setattr(deletion, "_stage_survivor", interrupted)
        failure.setattr(deletion, "_discard_preparation", lambda *args: None)
        with pytest.raises(StorageError, match="interrupted"):
            pipeline.delete(plan, confirm=plan.operation_id)
    assert pipeline.deletion_status(plan.operation_id).phase == "preparation"
    assert len(list(pipeline.store.data_dir.rglob("*.parquet"))) > 1
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Delete data").run()
    assert any("before any selected rows were deleted" in item.value for item in app.warning)
    _element(app.button, "Retry physical cleanup").click().run()
    assert not app.exception
    assert not app.error
    assert not app.success
    assert any("Preparation cleanup completed" in item.value for item in app.info)
    assert not pipeline.list_deletions(pending_only=True)
    with pytest.raises(DatasetNotFoundError):
        pipeline.deletion_status(plan.operation_id)
    assert list(pipeline.store.data_dir.rglob("*.parquet")) == [pipeline.store.data_dir / raw.relative_path]
    assert_frame_equal(pipeline.read_dataset(raw.dataset_id), sample_ohlcv_frame)
