"""Offline native-batch cost planning, resumability, and isolated publication."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest, InvalidDataRequestError
from data_pipeline.providers import BaseDataProvider, ProviderRegistry
from data_pipeline.schemas import validate_ohlcv

START = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
END = START + timedelta(minutes=3)


def bars(symbol, start=START, count=3, bad=False):
    frame = validate_ohlcv(pl.DataFrame({
        "timestamp": [start + timedelta(minutes=i) for i in range(count)],
        "symbol": [symbol] * count, "open": [100.0] * count, "high": [102.0] * count,
        "low": [99.0] * count, "close": [101.0] * count, "volume": [10.0] * count,
    }))
    return frame.with_columns(pl.lit(-1.0).alias("high")) if bad else frame


class FakeProvider(BaseDataProvider):
    name = "databento-xnas.itch"
    dataset = "XNAS.ITCH"
    supported_timeframes = frozenset({"1m", "1h"})

    def __init__(self):
        self.estimates, self.submissions, self.downloads = [], [], []
        self.jobs = {}
        self.invalid, self.bad, self.empty = set(), set(), set()
        self.submit_error = None
        self.download_error = None
        self.cost = 0.25

    def fetch(self, request):
        raise AssertionError("Sequential single-ticker fetching must not be used")

    def list_schemas(self):
        return ["ohlcv-1m", "ohlcv-1h"]

    def get_dataset_range(self):
        return {"schema": {schema: {"start": "2020-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}
                           for schema in self.list_schemas()}}

    def get_dataset_conditions(self, start, end):
        return [{"date": "2025-01-02", "condition": "degraded"}]

    def preflight_symbols(self, symbols, start, end):
        return {"valid": [s for s in symbols if s not in self.invalid],
                "errors": {s: "not found" for s in symbols if s in self.invalid}, "warnings": []}

    def estimate_cost(self, symbols, start, end, timeframe):
        self.estimates.append((symbols, start, end, timeframe))
        return self.cost

    def submit_job(self, symbols, start, end, timeframe):
        self.submissions.append((symbols, start, end, timeframe))
        identifier = f"job-{len(self.submissions)}"
        job = dict(id=identifier, state="done", symbols=symbols, start=start.isoformat(),
                   end=end.isoformat(), dataset=self.dataset, schema=f"ohlcv-{timeframe}")
        self.jobs[identifier] = job
        if self.submit_error:
            raise self.submit_error
        return job

    def job_status(self, job_id):
        return self.jobs[job_id]

    def download_job(self, job_id, directory):
        self.downloads.append(job_id)
        if self.download_error:
            raise self.download_error
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (job_id + ".dbn")
        path.write_text(job_id)
        return (path,)

    def iter_frames(self, paths, timeframe):
        for path in paths:
            job = self.jobs[Path(path).read_text()]
            start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
            for symbol in job["symbols"]:
                if symbol not in self.empty:
                    yield bars(symbol, start, int((end - start).total_seconds() / 60), symbol in self.bad)


@pytest.fixture
def setup(tmp_path):
    provider = FakeProvider()
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({provider.name: provider}))
    return pipeline, provider


def estimate(pipeline, tickers=("AAPL", "MSFT"), **kwargs):
    return pipeline.estimate_databento(tickers, start=START, end=END, dataset="XNAS.ITCH", **kwargs)


def complete(pipeline, plan, limit=40):
    for _ in range(limit):
        plan = pipeline.advance_databento(plan["id"], submit=True)
        if plan["status"] in {"completed", "completed_with_errors"}:
            return plan
    raise AssertionError(plan)


def test_hundreds_of_symbols_share_one_costed_job_and_resume_after_restart(setup):
    pipeline, provider = setup
    symbols = [f"T{i}" for i in range(500)]
    plan = estimate(pipeline, symbols)
    assert len(plan["groups"]) == 1
    assert plan["groups"][0]["symbols"] == symbols
    assert plan["estimated_cost"] == 0.25
    assert "degraded" in plan["warnings"][0]
    assert provider.submissions == []
    assert estimate(pipeline, symbols)["id"] == plan["id"]
    assert len(provider.estimates) == 1
    assert pipeline.advance_databento(plan["id"])["authorized"] is False
    plan = pipeline.advance_databento(plan["id"], submit=True)
    assert len(provider.submissions) == 1
    reopened = DataPipeline(pipeline.store.data_dir, providers=pipeline.providers)
    assert reopened.databento_plans()[0]["id"] == plan["id"]
    plan = complete(reopened, plan, limit=510)
    assert len(provider.submissions) == 1
    assert len(plan["outcomes"]) == 500
    assert all(row["status"] == "saved" for row in plan["outcomes"])
    assert len(reopened.list_datasets()) == 500
    assert reopened.databento_report(plan["id"]).to_frame().height == 500
    assert all(row["quality"]["status"] == "unknown" for row in plan["outcomes"])


def test_invalid_empty_and_malformed_tickers_do_not_prevent_other_publications(setup):
    pipeline, provider = setup
    provider.invalid.add("MISSING")
    provider.bad.add("BAD")
    provider.empty.add("EMPTY")
    plan = complete(pipeline, estimate(pipeline, ["AAPL", "MISSING", "BAD", "EMPTY", "MSFT"]))
    assert plan["status"] == "completed_with_errors"
    assert {row["ticker"]: row["status"] for row in plan["outcomes"]} == {
        "AAPL": "saved", "MSFT": "saved", "MISSING": "failed", "BAD": "failed", "EMPTY": "empty"}
    assert provider.estimates[0][0] == ["AAPL", "BAD", "EMPTY", "MSFT"]
    assert len(pipeline.list_datasets()) == 2


def test_fetch_missing_estimates_actual_gaps_and_preserves_existing_rows(setup):
    pipeline, provider = setup
    prior = pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m", provider=provider.name),
                                 bars("AAPL", START + timedelta(minutes=1), 1)).raw
    plan = estimate(pipeline)
    assert {(item[1], item[2]) for item in provider.estimates} == {
        (START, START + timedelta(minutes=1)), (START + timedelta(minutes=2), END), (START, END)}
    plan = complete(pipeline, plan)
    outcome = next(row for row in plan["outcomes"] if row["ticker"] == "AAPL")
    assert outcome["status"] == "merged"
    assert outcome["replaced_dataset_ids"] == [prior.dataset_id]
    assert outcome["fetched_row_count"] == 2
    assert pipeline.read_dataset(outcome["dataset_ids"][0]).equals(bars("AAPL"))


def test_local_revision_change_invalidates_estimate_before_any_submission(setup):
    pipeline, provider = setup
    plan = estimate(pipeline)
    pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m", provider=provider.name), bars("AAPL"))
    with pytest.raises(InvalidDataRequestError, match="coverage changed"):
        pipeline.advance_databento(plan["id"], submit=True)
    assert pipeline.databento_plan(plan["id"])["status"] == "stale"
    assert not provider.submissions


def test_cost_cap_prevents_submission_and_does_not_claim_to_cap_billing(setup):
    pipeline, provider = setup
    plan = estimate(pipeline, max_estimated_cost=0.10)
    with pytest.raises(InvalidDataRequestError, match="exceeds the maximum"):
        pipeline.advance_databento(plan["id"], submit=True)
    assert provider.submissions == []
    assert not pipeline.databento_plan(plan["id"])["authorized"]


@pytest.mark.parametrize("error", [RuntimeError("timeout"), KeyboardInterrupt()])
def test_uncertain_submission_is_never_retried_and_can_attach_existing_job(setup, error):
    pipeline, provider = setup
    provider.submit_error = error
    plan = estimate(pipeline)
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            pipeline.advance_databento(plan["id"], submit=True)
    else:
        pipeline.advance_databento(plan["id"], submit=True)
    provider.submit_error = None
    for _ in range(3):
        plan = pipeline.advance_databento(plan["id"], submit=True)
    assert plan["status"] == "needs_reconciliation"
    assert len(provider.submissions) == 1
    plan = pipeline.reconcile_databento_job(plan["id"], plan["groups"][0]["id"], "job-1")
    complete(pipeline, plan)
    assert len(provider.submissions) == 1


def test_failed_download_retry_reuses_job_without_repurchase(setup):
    pipeline, provider = setup
    plan = estimate(pipeline)
    pipeline.advance_databento(plan["id"], submit=True)
    pipeline.advance_databento(plan["id"])
    provider.download_error = RuntimeError("connection reset")
    plan = pipeline.advance_databento(plan["id"])
    assert plan["groups"][0]["status"] == "ready"
    provider.download_error = None
    complete(pipeline, plan)
    assert provider.downloads == ["job-1", "job-1"]
    assert len(provider.submissions) == 1


def test_download_progress_is_persisted_and_resumes_existing_job_after_restart(setup, monkeypatch):
    pipeline, provider = setup
    plan = estimate(pipeline)
    pipeline.advance_databento(plan["id"], submit=True)
    pipeline.advance_databento(plan["id"])
    original = provider.download_job
    partial = dict(downloaded_bytes=50, total_bytes=100, completed_files=0, total_files=1)
    full = dict(downloaded_bytes=100, total_bytes=100, completed_files=1, total_files=1)
    seen = []

    def interrupted(job_id, directory, *, progress):
        progress(partial)
        raise RuntimeError("connection reset")

    monkeypatch.setattr(provider, "download_job", interrupted)
    failed = pipeline.advance_databento(plan["id"], progress=seen.append)
    assert seen == [partial]
    assert failed["groups"][0]["status"] == "ready"
    reopened = DataPipeline(pipeline.store.data_dir, providers=pipeline.providers)
    assert reopened.databento_plan(plan["id"])["groups"][0]["download_progress"] == partial

    def resumed(job_id, directory, *, progress):
        progress(partial)
        paths = original(job_id, directory)
        progress(full)
        return paths

    monkeypatch.setattr(provider, "download_job", resumed)
    resumed_plan = reopened.advance_databento(plan["id"], progress=seen.append)
    assert seen == [partial, partial, full]
    assert resumed_plan["groups"][0]["status"] == "downloaded"
    assert reopened.databento_plan(plan["id"])["groups"][0]["download_progress"] == full
    assert len(provider.submissions) == 1
    assert complete(reopened, resumed_plan)["status"] == "completed"


def test_progress_journal_writes_are_throttled_and_final_counts_are_saved(setup, monkeypatch):
    from data_pipeline import databento_batch

    pipeline, provider = setup
    plan = estimate(pipeline)
    pipeline.advance_databento(plan["id"], submit=True)
    pipeline.advance_databento(plan["id"])
    original_download, original_save = provider.download_job, databento_batch._save
    writes, updates = [], []

    def record_save(*args):
        writes.append(1)
        original_save(*args)

    def download(job_id, directory, *, progress):
        for size in range(21):
            progress(dict(downloaded_bytes=size, total_bytes=20, completed_files=int(size == 20), total_files=1))
        return original_download(job_id, directory)

    monkeypatch.setattr(provider, "download_job", download)
    monkeypatch.setattr(databento_batch, "_save", record_save)
    monkeypatch.setattr(databento_batch.time, "monotonic", lambda: 1.0)
    pipeline.advance_databento(plan["id"], progress=updates.append)
    assert len(updates) == 21
    assert len(writes) == 2  # Initial progress and the completed download stage.
    assert pipeline.databento_plan(plan["id"])["groups"][0]["download_progress"] == updates[-1]


def test_download_cooldown_survives_restart_and_resumes_same_job(setup, monkeypatch):
    from data_pipeline import databento_batch
    from data_pipeline.exceptions import ProviderError

    pipeline, provider = setup
    plan = estimate(pipeline)
    pipeline.advance_databento(plan["id"], submit=True)
    pipeline.advance_databento(plan["id"])
    error = ProviderError("Databento download is rate limited; retry this saved job later.")
    error.retry_after = 120
    provider.download_error = error
    plan = pipeline.advance_databento(plan["id"])
    retry_at = datetime.fromisoformat(plan["groups"][0]["next_download_attempt_at"])
    reopened = DataPipeline(pipeline.store.data_dir, providers=pipeline.providers)
    provider.download_error = None

    class Clock(datetime):
        current = retry_at - timedelta(seconds=1)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    monkeypatch.setattr(databento_batch, "datetime", Clock)
    for _ in range(3):
        assert reopened.advance_databento(plan["id"])["groups"][0]["status"] == "ready"
    assert provider.downloads == ["job-1"]
    Clock.current = retry_at
    plan = reopened.advance_databento(plan["id"])
    assert plan["groups"][0]["status"] == "downloaded"
    assert "next_download_attempt_at" not in plan["groups"][0]
    assert provider.downloads == ["job-1", "job-1"]
    assert len(provider.submissions) == 1


def test_schema_and_history_validation_happen_before_cost_and_submit(setup):
    pipeline, provider = setup
    with pytest.raises(InvalidDataRequestError, match="native"):
        estimate(pipeline, timeframe="30m")
    with pytest.raises(InvalidDataRequestError, match="outside"):
        pipeline.estimate_databento(["AAPL"], dataset="XNAS.ITCH",
            start=datetime(2019, 1, 1, tzinfo=UTC), end=END)
    assert not provider.submissions and not provider.estimates


def test_already_covered_tickers_skip_without_vendor_metadata_calls(setup):
    pipeline, provider = setup
    pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m", provider=provider.name), bars("AAPL"))
    provider.list_schemas = lambda: pytest.fail("No metadata required for a covered request")
    plan = estimate(pipeline, ["AAPL"])
    assert plan["status"] == "completed"
    assert plan["outcomes"][0]["status"] == "skipped"
    assert not provider.estimates


def test_bulk_symbol_limit_splits_native_jobs_and_bad_symbols_are_independent(setup):
    pipeline, provider = setup
    plan = estimate(pipeline, [f"T{i}" for i in range(2001)] + ["ALL_SYMBOLS", "bad,one", "A B"])
    assert [len(group["symbols"]) for group in plan["groups"]] == [2000, 1]
    assert len([item for item in plan["outcomes"] if item["status"] == "failed"]) == 3
    assert not provider.submissions


def test_journal_path_cannot_escape_data_root(setup):
    pipeline, _ = setup
    with pytest.raises(InvalidDataRequestError):
        pipeline.advance_databento("../outside", submit=True)


def test_concurrent_estimates_share_one_durable_plan(setup):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    pipeline, provider = setup
    barrier = Barrier(2)
    original = provider.estimate_cost

    def estimate_together(*args):
        barrier.wait(timeout=5)
        return original(*args)

    provider.estimate_cost = estimate_together
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(estimate, pipeline)
        second = executor.submit(estimate, pipeline)
        assert first.result()["id"] == second.result()["id"]
    assert len(pipeline.databento_plans()) == 1
    assert not provider.submissions


def test_restart_after_publication_commit_recovers_exact_saved_ticker(setup, monkeypatch):
    pipeline, provider = setup
    plan = estimate(pipeline, ["AAPL"])
    for _ in range(4):
        pipeline.advance_databento(plan["id"], submit=True)
    original = pipeline.store.merge_raw

    def interrupt_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline.store, "merge_raw", interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        pipeline.advance_databento(plan["id"])
    assert len(pipeline.list_datasets()) == 1
    assert pipeline.databento_plan(plan["id"])["ticker_plans"]["AAPL"]["status"] == "publishing"
    monkeypatch.setattr(pipeline.store, "merge_raw", original)
    # A fresh estimate after partial publication must resume already paid work.
    assert estimate(pipeline, ["AAPL"])["id"] == plan["id"]
    plan = complete(pipeline, plan)
    assert plan["outcomes"][0]["status"] == "saved"
    assert len(provider.submissions) == 1
    assert len(pipeline.list_datasets()) == 1


def test_local_changes_before_later_job_prevent_another_charge(setup):
    pipeline, provider = setup
    original = pipeline.ingest_frame(DataRequest("AAPL", START, END, timeframe="1m", provider=provider.name),
                                    bars("AAPL", START + timedelta(minutes=1), 1)).raw
    plan = estimate(pipeline, ["AAPL"])
    for _ in range(4):
        plan = pipeline.advance_databento(plan["id"], submit=True)
    assert len(provider.submissions) == 1
    # A new local revision arrives while the first remote range is being handled.
    pipeline.store.refresh_raw(original.request, bars("AAPL"), supersedes=[original.dataset_id])
    plan = complete(pipeline, plan)
    assert len(provider.submissions) == 1
    assert plan["status"] == "completed_with_errors"
    assert "coverage changed" in plan["outcomes"][0]["error_message"]


def test_many_decoder_chunks_append_to_one_lossless_staging_file_per_ticker(setup):
    pipeline, provider = setup
    original = provider.iter_frames

    def small_chunks(*args):
        for frame in original(*args):
            yield from frame.iter_slices(1)

    provider.iter_frames = small_chunks
    plan = complete(pipeline, estimate(pipeline))
    parts = plan["groups"][0]["parts"]
    assert all(len(paths) == 1 and paths[0].endswith(".arrow") for paths in parts.values())
    for outcome in plan["outcomes"]:
        assert pipeline.read_dataset(outcome["dataset_ids"][0]).equals(bars(outcome["ticker"]))
