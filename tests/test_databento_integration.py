"""Dashboard → real batch journal → SDK adapter → real local catalog, offline."""

from datetime import UTC, date, datetime, timedelta
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import databento_dbn as dbn
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataQuery
from data_pipeline.providers import DatabentoProvider, ProviderRegistry
from research import Research


class OfflineSDK:
    """Fake only vendor endpoints; keep real SDK decoding and pipeline storage."""

    def __init__(self):
        self.jobs = {}
        self.key = "offline-integration-key"
        self.metadata = SimpleNamespace(
            list_datasets=Mock(return_value=["XNAS.ITCH", "EQUS.MINI"]),
            list_schemas=Mock(return_value=["ohlcv-1m", "ohlcv-1h"]),
            get_dataset_range=Mock(return_value={
                "start": "2020-01-01T00:00:00Z", "end": "2030-01-01T00:00:00Z",
                "schema": {"ohlcv-1m": {"start": "2020-01-01T00:00:00Z", "end": "2030-01-01T00:00:00Z"}},
            }),
            get_dataset_condition=Mock(return_value=[{
                "date": "2024-11-01", "condition": "degraded", "last_modified_date": None,
            }]),
            get_cost=Mock(return_value=2.5),
        )
        self.symbology = SimpleNamespace(resolve=Mock(side_effect=self.resolve))
        self.batch = SimpleNamespace(
            submit_job=Mock(side_effect=self.submit),
            get_job_details=Mock(side_effect=lambda job_id: {
                "id": job_id, "state": "done", "progress": 100,
                "cost_usd": 2.5, **self.jobs[job_id]["request"],
            }),
            list_files=Mock(side_effect=self.files),
            download=Mock(side_effect=self.download),
        )
        self.timeseries = SimpleNamespace(get_range=Mock(side_effect=AssertionError("Use batch jobs")))

    def resolve(self, **kwargs):
        return {
            "result": {symbol: [{"s": str(index)}]
                       for index, symbol in enumerate(kwargs["symbols"]) if symbol != "MISSING"},
            "partial": [symbol for symbol in kwargs["symbols"] if symbol == "AAPL"],
            "not_found": [symbol for symbol in kwargs["symbols"] if symbol == "MISSING"],
        }

    def submit(self, **kwargs):
        job_id = f"XNAS-{len(self.jobs) + 1}"
        symbols = kwargs["symbols"]
        start, end = kwargs["start"], kwargs["end"]
        mappings = [SimpleNamespace(raw_symbol=symbol, intervals=[SimpleNamespace(
            start_date=start.date(), end_date=end.date() + timedelta(days=1), symbol=str(index + 1),
        )]) for index, symbol in enumerate(symbols)]
        metadata = dbn.Metadata(
            dataset=kwargs["dataset"], schema=dbn.Schema.OHLCV_1M,
            stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
            start=pd.Timestamp(start).value, end=pd.Timestamp(end).value,
            symbols=symbols, mappings=mappings,
        )
        timestamp = datetime(2024, 11, 1, 13, 30, tzinfo=UTC)
        records = [dbn.OHLCVMsg(
            rtype=dbn.RType.OHLCV_1M, publisher_id=2, instrument_id=index + 1,
            ts_event=pd.Timestamp(timestamp).value,
            open=100_250_000_000, high=90_000_000_000 if symbol == "BAD" else 102_750_000_000,
            low=99_000_000_000, close=101_500_000_000, volume=1000,
        ) for index, symbol in enumerate(symbols)]
        self.jobs[job_id] = {"request": kwargs, "bytes": metadata.encode() + b"".join(bytes(row) for row in records)}
        return {"id": job_id, "state": "queued", "api_key": "should-not-be-saved"}

    def files(self, *, job_id):
        content = self.jobs[job_id]["bytes"]
        return [{"filename": "bars.dbn", "size": len(content),
                 "hash": "sha256:" + hashlib.sha256(content).hexdigest(),
                 "urls": {"https": f"https://downloads.databento.com/{job_id}/bars.dbn"}}]

    def download(self, **kwargs):
        raise AssertionError("Direct manifest downloads must not call SDK batch.download")

    def http_get(self, url, **kwargs):
        job_id = url.split("/")[-2]
        content = self.jobs[job_id]["bytes"]
        return OfflineResponse(content)


class OfflineResponse:
    status_code = 200
    headers = {}

    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content


@pytest.fixture
def integrated_dashboard(tmp_path, monkeypatch):
    import dashboard.app as dashboard

    sdk = OfflineSDK()
    session = Mock()
    session.get = sdk.http_get
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("data_pipeline.providers.databento_download.requests.Session", lambda: session)
    providers = ProviderRegistry({"databento": DatabentoProvider(client=sdk, chunk_size=37)})
    state = {}

    def restart():
        state["research"] = Research(tmp_path / "data", tmp_path / "runs", providers=providers)

    restart()
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: state["research"])

    def open_page():
        app = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=15).run()
        app.sidebar.radio[0].set_value("Fetch").run()
        assert not app.exception
        return app

    return open_page, sdk, state, restart


def estimate(app, tickers):
    app.button(key="db-load-datasets").click().run()
    app.selectbox(key="db-dataset").set_value("XNAS.ITCH")
    app.date_input(key="db-fetch-start").set_value(date(2024, 11, 1))
    app.date_input(key="db-fetch-end").set_value(date(2024, 11, 2))
    app.text_area(key="db-tickers").set_value(",".join(tickers))
    app.button(key="db-estimate").click().run()
    assert not app.exception
    assert not app.error


def finish(app, state, *, maximum=45):
    for _ in range(maximum):
        plan = state["research"].pipeline.databento_plans()[0]
        if plan["status"] in {"completed", "completed_with_errors"}:
            return plan
        app.run()
        assert not app.exception
        assert not app.error
    pytest.fail(f"Download did not finish: {plan['status']}")


def test_hundreds_of_tickers_real_plan_restart_and_idempotent_storage(integrated_dashboard):
    open_page, sdk, state, restart = integrated_dashboard
    app = open_page()
    assert not sdk.metadata.list_datasets.called
    assert not sdk.batch.submit_job.called
    tickers = [f"TICK{i:03d}" for i in range(250)]
    estimate(app, tickers)
    pipeline = state["research"].pipeline
    plan = pipeline.databento_plans()[0]
    assert plan["status"] == "estimated"
    assert plan["estimated_cost"] == 2.5
    assert len(plan["groups"]) == 1
    assert len(sdk.metadata.get_cost.call_args.kwargs["symbols"]) == 250
    assert not sdk.batch.submit_job.called
    assert any("degraded" in warning.value for warning in app.warning)
    app.button(key=f"db-submit-{plan['id']}").click().run()
    assert not app.exception
    assert sdk.batch.submit_job.call_count == 1
    restart()
    reopened = open_page()
    completed = finish(reopened, state)
    assert completed["status"] == "completed"
    assert sdk.batch.submit_job.call_count == 1
    pipeline = state["research"].pipeline
    saved = pipeline.list_datasets(DataQuery(provider="databento-xnas.itch"))
    assert len(saved) == 250
    assert all(item.row_count == 1 for item in saved)
    ids = {item.dataset_id for item in saved}
    assert pipeline.read(DataQuery(provider="databento-xnas.itch", symbol="TICK000"))["open"].item() == 100.25
    assert all(outcome.status == "saved" for outcome in pipeline.databento_report(plan["id"]).outcomes)
    for _ in range(3):
        pipeline.advance_databento(plan["id"])
        reopened.run()
    assert {item.dataset_id for item in pipeline.list_datasets(DataQuery(provider="databento-xnas.itch"))} == ids
    assert sdk.batch.submit_job.call_count == 1
    assert not sdk.timeseries.get_range.called
    assert not sdk.batch.download.called
    sdk.batch.list_files.assert_called_once()
    assert "should-not-be-saved" not in str(pipeline.databento_plan(plan["id"]))
    assert any("Download complete" in item.value for item in reopened.success)


def test_real_report_isolates_unresolved_and_invalid_tickers_and_shows_warnings(integrated_dashboard):
    open_page, sdk, state, restart = integrated_dashboard
    app = open_page()
    estimate(app, ["AAPL", "BAD", "MISSING"])
    pipeline = state["research"].pipeline
    plan = pipeline.databento_plans()[0]
    assert plan["groups"][0]["symbols"] == ["AAPL", "BAD"]
    assert any("resolves for only part" in warning.value for warning in app.warning)
    assert any("degraded" in warning.value for warning in app.warning)
    app.button(key=f"db-submit-{plan['id']}").click().run()
    completed = finish(app, state)
    assert completed["status"] == "completed_with_errors"
    report = pipeline.databento_report(plan["id"])
    outcomes = {outcome.ticker: outcome for outcome in report.outcomes}
    assert outcomes["AAPL"].status == "saved"
    assert outcomes["BAD"].error_type == "InvalidOHLCVError"
    assert outcomes["MISSING"].error_type == "SymbolResolutionError"
    assert [item.request.symbol for item in pipeline.list_datasets()] == ["AAPL"]
    assert sdk.batch.submit_job.call_count == 1
    assert sdk.batch.submit_job.call_args.kwargs["symbols"] == ["AAPL", "BAD"]
    restart()
    reopened = open_page()
    assert any("finished with errors" in warning.value for warning in reopened.warning)
    assert sdk.batch.submit_job.call_count == 1
