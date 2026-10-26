"""Batch analysis preserves script semantics, bounded work, and saved identities."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from threading import Event, Lock

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest, FetchQuality, OmittedBar
from research.analysis import BaseChartLogic, LogicContext, LogicRegistry, LogicResult, LogicRunner, LogicSpec, analysis
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore
from research.analysis.scripting.decorator import script_class


START = datetime(2024, 1, 2, tzinfo=UTC)


def bars(symbol="AAPL", size=10):
    values = [10. + index for index in range(size)]
    return pl.DataFrame({"timestamp": [START + timedelta(hours=index) for index in range(size)],
                         "symbol": [symbol] * size, "open": values, "close": values,
                         "high": [value + 1 for value in values], "low": [value - 1 for value in values],
                         "volume": [100.] * size}).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


@analysis("counts")
def counts(ctx, period: int = 3):
    data = ctx.selected(ctx.features(sma=ctx.sma(period)))
    return (ctx.result().text("An ordinary script used unchanged in batch analysis.")
            .table("Summary", pl.DataFrame({"outcome": ["win", "loss", "inconclusive", "total"],
                                              "count": [data.height, 0, 0, data.height]}))
            .table("Values", data.select("source_index", "timestamp", "sma"))
            .signals("Findings", data.head(1)))


class Broken(BaseChartLogic):
    name = "broken"

    def calculate(self, context, config, progress=None):
        raise ValueError("Deliberate analysis failure")


@pytest.fixture
def batch(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    registry = LogicRegistry()
    registry.register_class(script_class(counts))
    store = BatchStore(tmp_path / "results")
    runner = BatchRunner(pipeline, registry, store)
    return pipeline, registry, store, runner


def save(pipeline, symbol="AAPL", size=10):
    frame = bars(symbol, size)
    request = DataRequest(symbol, START, START + timedelta(hours=size), timeframe="1h")
    return pipeline.store.write_raw(request, frame)


def specification(identifier="counts", period=3):
    return LogicSpec(identifier, "counts", "1", {"period": period})


def request(tickers=("AAPL",), **kwargs):
    return BatchRequest(tickers, timeframe="1h", **kwargs)


def test_frozen_revisions_and_lazy_saved_history(batch):
    pipeline, registry, store, runner = batch
    original = save(pipeline)
    first = runner.create(request(), [specification()])
    changed = bars().with_columns((pl.col("close") + .5).alias("close"))
    pipeline.store.replace_raw(original.dataset_id, changed, confirm=original.dataset_id)
    finished = runner.run(first["run_id"])
    saved = store.load_ticker(first["run_id"], "AAPL")
    assert finished["sources"]["AAPL"][0]["dataset_id"] == original.dataset_id
    assert finished["sources"]["AAPL"][0]["checksum_sha256"] == original.checksum_sha256
    assert saved["results"]["counts"]["tables"][1]["rows"][2]["sma"] == 11.
    second = runner.execute(request(), [specification()])
    reopened = BatchStore(store.directory)
    assert [item["run_id"] for item in reopened.list_runs()] == [second["run_id"], first["run_id"]]
    assert reopened.load_ticker(second["run_id"], "AAPL")["results"]["counts"]["tables"][1]["rows"][2]["sma"] == 11.5
    assert first["run_id"] != second["run_id"]
    assert len(finished["engine_sha256"]) == 64
    assert finished["implementations"]["counts"]["implementation_fingerprint"]


@pytest.mark.parametrize("timeframe,end,as_of,expected_count", [
    ("1h", START + timedelta(hours=8, minutes=30), START + timedelta(days=1), 4),
    ("1h", START + timedelta(hours=9), START + timedelta(hours=7, minutes=30), 3),
])
def test_support_and_completed_prefix_match_single_ticker(batch, timeframe, end, as_of, expected_count):
    pipeline, registry, store, runner = batch
    item = save(pipeline)
    start = START + timedelta(hours=4)
    manifest = runner.execute(request(start=start, end=end, as_of=as_of), [specification()])
    details = store.load_ticker(manifest["run_id"], "AAPL")["results"]["counts"]
    frame = pipeline.read_dataset(item.dataset_id)
    selected = frame.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
    support = frame.filter(pl.col("timestamp") < start)
    context = LogicContext(selected, timeframe, "AAPL", min(end, as_of), support_frame=support)
    original = LogicRunner(registry).apply([specification()], context).results["counts"]
    assert details["tables"] == original.tables
    assert details["tables"][0]["rows"][0]["count"] == expected_count
    assert details["tables"][1]["rows"][0]["sma"] == 13.
    assert details["data"]["metadata"]["initialization_bars"] == 4


def test_calendar_end_is_label_boundary_not_completion_cutoff(batch):
    pipeline, registry, store, runner = batch
    frame = bars(size=2).with_columns(pl.Series("timestamp", [START, START + timedelta(days=1)],
                                               dtype=pl.Datetime("ms", "UTC")))
    pipeline.store.write_raw(DataRequest("AAPL", START, START + timedelta(days=2), timeframe="1d"), frame)
    manifest = runner.execute(BatchRequest(("AAPL",), timeframe="1d", end=START + timedelta(days=1),
                                          as_of=START + timedelta(days=5)), [specification()])
    details = store.load_ticker(manifest["run_id"], "AAPL")
    # The selected Jan 2 daily label completes after the Jan 3 00:00 UTC end
    # filter, but is completed at as_of, just as in Market data.
    assert details["results"]["counts"]["tables"][0]["rows"][0]["count"] == 1


def test_errors_missing_data_and_multiple_instances_are_independent(batch):
    pipeline, registry, store, runner = batch
    save(pipeline)
    registry.register_class(Broken)
    runner = BatchRunner(pipeline, registry, store)
    specs = [specification("fast", 2), specification("slow", 5), LogicSpec("bad", "broken", "1", {})]
    manifest = runner.execute(request(("AAPL", "MISSING")), specs)
    assert manifest["status"] == "completed"
    assert manifest["counts"]["partial"] == manifest["counts"]["no_data"] == 1
    entry = manifest["tickers"]["AAPL"]
    assert entry["analysis_status"] == {"fast": "completed", "slow": "completed", "bad": "failed"}
    assert set(entry["summaries"]) == {"fast", "slow"}
    assert entry["summaries"]["fast"]["values"] == {"win": 10, "loss": 0, "inconclusive": 0, "total": 10}
    assert store.load_ticker(manifest["run_id"], "AAPL")["errors"] == {"bad": "Deliberate analysis failure"}
    assert store.load_ticker(manifest["run_id"], "MISSING")["status"] == "no_data"


def test_all_analysis_failure_is_saved_and_other_ticker_continues(batch):
    pipeline, registry, store, _ = batch
    save(pipeline, "AAPL")
    save(pipeline, "MSFT")
    registry.register_class(Broken)
    finished = BatchRunner(pipeline, registry, store).execute(request(("AAPL", "MSFT")),
                                                             [LogicSpec("bad", "broken", "1", {})])
    assert finished["counts"]["failed"] == 2
    assert all("Deliberate analysis failure" in item["error"] for item in finished["tickers"].values())


def test_date_range_without_selected_bars_is_no_data(batch):
    pipeline, _, store, runner = batch
    save(pipeline)
    finished = runner.execute(request(start=START + timedelta(days=1)), [specification()])
    assert finished["counts"]["no_data"] == 1


def test_no_chart_payload_is_saved_but_original_text_and_tables_survive(batch):
    pipeline, _, store, runner = batch
    save(pipeline)
    finished = runner.execute(request(), [specification()])
    result = store.load_ticker(finished["run_id"], "AAPL")["results"]["counts"]
    assert result["overlays"] == []
    assert result["data"]["analysis"] == {"text": ["An ordinary script used unchanged in batch analysis."]}
    assert len(result["tables"]) == 3
    assert "tables" not in json.dumps(finished["tickers"])
    assert not any(key in result["data"] for key in ("trends", "panel"))


def test_cancel_before_start_loads_no_tickers(batch, monkeypatch):
    pipeline, _, store, runner = batch
    save(pipeline)
    manifest = runner.create(request(), [specification()])
    cancel = Event()
    cancel.set()
    monkeypatch.setattr(pipeline, "read_dataset", lambda *args: pytest.fail("Cancelled run loaded market data"))
    finished = runner.run(manifest["run_id"], cancel=cancel)
    assert finished["status"] == "cancelled"
    assert finished["counts"]["cancelled"] == 1


def test_mid_run_cancellation_finishes_only_bounded_inflight_work(batch):
    pipeline, registry, store, _ = batch
    for ticker in ("A", "B", "C", "D"):
        save(pipeline, ticker)
    started, release, cancel = Event(), Event(), Event()
    lock = Lock()
    calls = []

    class Blocking(BaseChartLogic):
        name = "blocking"

        def calculate(self, context, config, progress=None):
            with lock:
                calls.append(context.frame["symbol"][0])
                if len(calls) == 2:
                    started.set()
            assert release.wait(10)
            return LogicResult(tables=[{"name": "Summary", "rows": [{"count": 1}]}])

    registry.register_class(Blocking)
    runner = BatchRunner(pipeline, registry, store)
    prepared = runner.create(request(("A", "B", "C", "D"), workers=2), [LogicSpec("block", "blocking", "1", {})])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.run, prepared["run_id"], cancel=cancel)
        try:
            assert started.wait(10)
            cancel.set()
        finally:
            release.set()
        finished = future.result(timeout=10)
    assert len(calls) == 2
    assert finished["counts"]["completed"] == finished["counts"]["cancelled"] == 2
    assert finished["status"] == "cancelled"
    for ticker in calls:
        assert store.load_ticker(prepared["run_id"], ticker)["status"] == "completed"


def test_incremental_progress_can_reopen_completed_ticker_before_next_starts(batch):
    pipeline, _, store, runner = batch
    save(pipeline, "A")
    save(pipeline, "B")
    observed = []

    def progress(manifest):
        if manifest["tickers"]["A"]["status"] == "completed" and manifest["tickers"]["B"]["status"] == "pending":
            observed.append(store.load_ticker(manifest["run_id"], "A")["ticker"])
            store.save_aggregation(manifest["run_id"], {"counts": {"chosen": "win"}})

    finished = runner.execute(request(("A", "B")), [specification()], progress=progress)
    assert observed == ["A"]
    assert finished["aggregation"] == {"counts": {"chosen": "win"}}


def test_registry_snapshot_and_duplicate_execution_guard(batch):
    pipeline, registry, store, runner = batch
    save(pipeline)
    manifest = runner.create(request(), [specification()])
    registry._classes.clear()
    registry._definitions.clear()
    assert runner.run(manifest["run_id"])["counts"]["completed"] == 1
    with pytest.raises(ValueError, match="newly prepared"):
        runner.run(manifest["run_id"])
    with pytest.raises(ValueError, match="already been started"):
        store.claim_run(manifest["run_id"])


def test_engine_changes_after_preparation_do_not_mislabel_saved_provenance(batch, monkeypatch):
    pipeline, _, store, runner = batch
    save(pipeline)
    manifest = runner.create(request(), [specification()])
    monkeypatch.setattr("research.analysis.batch.runner._engine_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="engine files changed"):
        runner.run(manifest["run_id"])
    assert store.load_manifest(manifest["run_id"])["status"] == "pending"


def test_exact_revision_read_failure_is_isolated(batch, monkeypatch):
    pipeline, _, store, runner = batch
    a = save(pipeline, "A")
    save(pipeline, "B")
    original = pipeline.read_dataset

    def read(dataset_id):
        if dataset_id == a.dataset_id:
            raise ValueError("Checksum mismatch")
        return original(dataset_id)

    monkeypatch.setattr(pipeline, "read_dataset", read)
    manifest = runner.execute(request(("A", "B")), [specification()])
    assert manifest["counts"]["failed"] == manifest["counts"]["completed"] == 1
    assert store.load_ticker(manifest["run_id"], "A")["error"] == "Checksum mismatch"


def test_processed_analysis_does_not_require_backtest_causality_contract(batch):
    pipeline, _, store, runner = batch
    parent = save(pipeline)
    processors_json = json.dumps([{"name": "example", "version": "1", "config": {}}], sort_keys=True, separators=(",", ":"))
    fingerprint = sha256(processors_json.encode()).hexdigest()
    processed = pipeline.store.write_processed(parent.dataset_id, bars(), pipeline_id=fingerprint,
                                               processors_json=processors_json)
    manifest = runner.execute(request(layer="processed", pipeline_id=fingerprint), [specification()])
    assert manifest["counts"]["completed"] == 1
    assert manifest["sources"]["AAPL"][0]["dataset_id"] == processed.dataset_id


@pytest.mark.parametrize("missing_index", [0, 4, 9])
def test_screening_records_omission_warnings_in_saved_results_and_summary(batch, missing_index):
    pipeline, _, store, runner = batch
    frame = bars()
    missing = frame["timestamp"][missing_index]
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(missing, "invalid_bar: open must be greater than zero"),))
    pipeline.store.write_raw(DataRequest("AAPL", START, START + timedelta(hours=10), timeframe="1h"),
                             frame.filter(pl.col("timestamp") != missing), quality=quality)
    manifest = runner.execute(request(start=START, end=START + timedelta(hours=10)), [specification()])
    entry = manifest["tickers"]["AAPL"]
    assert entry["status"] == "completed"
    assert entry["data_validation"]["status"] == "not_checked"
    assert any("incomplete history" in message for message in entry["data_warnings"])
    saved = store.load_ticker(manifest["run_id"], "AAPL")
    result = saved["results"]["counts"]
    assert result["data"]["data_integrity"]["unresolved_omitted_bar_timestamps"] == [missing.isoformat()]
    assert entry["data_warnings"][0] in result["messages"]
    assert saved["data_integrity"] == result["data"]["data_integrity"]


def test_repaired_omission_does_not_warn_in_screening_results(batch):
    pipeline, _, store, runner = batch
    frame = bars()
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(frame["timestamp"][4], "historical omission, repaired"),))
    pipeline.store.write_raw(DataRequest("AAPL", START, START + timedelta(hours=10), timeframe="1h"),
                             frame, quality=quality)
    manifest = runner.execute(request(), [specification()])
    assert manifest["tickers"]["AAPL"]["data_warnings"] == []
    saved = store.load_ticker(manifest["run_id"], "AAPL")
    assert saved["data_integrity"]["unresolved_omitted_bar_count"] == 0


def test_generic_processed_screening_keeps_raw_parent_warning(batch):
    pipeline, _, store, runner = batch
    frame = bars()
    missing = frame["timestamp"][4]
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(missing, "invalid_bar: open is missing (null)"),))
    parent = pipeline.store.write_raw(DataRequest("AAPL", START, START + timedelta(hours=10), timeframe="1h"),
                                      frame.filter(pl.col("timestamp") != missing), quality=quality)
    processors_json = json.dumps([{"name": "example", "version": "1", "config": {}}],
                                 sort_keys=True, separators=(",", ":"))
    fingerprint = sha256(processors_json.encode()).hexdigest()
    pipeline.store.write_processed(parent.dataset_id, frame, pipeline_id=fingerprint,
                                    processors_json=processors_json)
    manifest = runner.execute(request(layer="processed", pipeline_id=fingerprint), [specification()])
    assert manifest["counts"]["completed"] == 1
    entry = manifest["tickers"]["AAPL"]
    assert "unresolved omitted raw-input" in entry["data_warnings"][0]
    result = store.load_ticker(manifest["run_id"], "AAPL")["results"]["counts"]
    assert result["data"]["data_integrity"]["processed_omission_sources"]


def test_screening_warns_about_missing_initialization_bars(batch):
    pipeline, _, store, runner = batch
    frame = bars()
    missing = frame["timestamp"][2]
    quality = FetchQuality(status="reported", skip_invalid_bars=True,
                           omitted_bars=(OmittedBar(missing, "invalid_bar: open is missing (null)"),))
    pipeline.store.write_raw(DataRequest("AAPL", START, START + timedelta(hours=10), timeframe="1h"),
                             frame.filter(pl.col("timestamp") != missing), quality=quality)
    manifest = runner.execute(request(start=START + timedelta(hours=5)), [specification()])
    assert "initialization history" in manifest["tickers"]["AAPL"]["data_warnings"][0]
    integrity = store.load_ticker(manifest["run_id"], "AAPL")["data_integrity"]
    assert integrity["unresolved_omitted_bar_count"] == 0
    assert integrity["initialization_omissions"]["unresolved_omitted_bar_count"] == 1


@pytest.mark.parametrize("provider,strict,expected", [
    ("yahoo", None, "completed"), ("yahoo", True, "failed"),
    ("databento-xnas.itch", None, "failed"), ("databento-xnas.itch", False, "completed"),
])
def test_optional_strict_gate_blocks_missing_daily_bar_before_analysis(batch, provider, strict, expected):
    pipeline, _, store, runner = batch
    end = START + timedelta(days=3)
    frame = bars(size=3).with_columns(pl.Series("timestamp", [START + timedelta(days=i) for i in range(3)],
                                               dtype=pl.Datetime("ms", "UTC")))
    pipeline.store.write_raw(DataRequest("AAPL", START, end, timeframe="1d", provider=provider),
                             frame.filter(pl.col("timestamp") != START + timedelta(days=1)))
    manifest = runner.execute(BatchRequest(("AAPL",), provider=provider, timeframe="1d", start=START,
                                          end=end, strict_validation=strict), [specification()])
    detail = store.load_ticker(manifest["run_id"], "AAPL")
    assert detail["status"] == expected
    if expected == "failed":
        assert "Missing 1 required" in detail["error"]
        assert detail["results"] == {}
        assert manifest["tickers"]["AAPL"]["data_validation"]["status"] == "failed"
    else:
        assert detail["results"]["counts"]["tables"][0]["rows"][0]["count"] == 2


@pytest.mark.parametrize("initial_gap", [False, True])
def test_strict_gate_validates_frozen_revision_even_after_replacement(batch, initial_gap):
    pipeline, _, store, runner = batch
    end = START + timedelta(days=3)
    frame = bars(size=3).with_columns(pl.Series("timestamp", [START + timedelta(days=i) for i in range(3)],
                                               dtype=pl.Datetime("ms", "UTC")))
    incomplete = frame.filter(pl.col("timestamp") != START + timedelta(days=1))
    original = pipeline.store.write_raw(DataRequest("AAPL", START, end, timeframe="1d"),
                                        incomplete if initial_gap else frame)
    prepared = runner.create(BatchRequest(("AAPL",), start=START, end=end, strict_validation=True),
                             [specification()])
    pipeline.store.replace_raw(original.dataset_id, frame if initial_gap else incomplete,
                               confirm=original.dataset_id)
    manifest = runner.run(prepared["run_id"])
    detail = store.load_ticker(manifest["run_id"], "AAPL")
    assert detail["status"] == ("failed" if initial_gap else "completed")
    if not initial_gap:
        assert detail["data_integrity"]["validation"]["source_dataset_ids"] == [original.dataset_id]
        assert manifest["tickers"]["AAPL"]["data_validation"]["status"] == "passed"


@pytest.mark.parametrize("settings", [
    {"tickers": []}, {"tickers": "AAPL"}, {"tickers": ["aapl", "AAPL"]},
    {"workers": 0}, {"workers": True}, {"workers": 33}, {"layer": "processed"},
    {"pipeline_id": "processed-id"}, {"start": datetime(2024, 1, 1)},
    {"as_of": datetime(2024, 1, 1)}, {"start": START, "end": START},
    {"as_of": datetime.now(UTC) + timedelta(days=1)},
])
def test_invalid_requests_are_rejected(settings):
    with pytest.raises((TypeError, ValueError)):
        BatchRequest(**{"tickers": ["AAPL"], **settings})


def test_configuration_validated_before_creating_saved_run(batch):
    pipeline, _, store, runner = batch
    with pytest.raises(ValueError):
        runner.create(request(), [specification(period="not a period")])
    assert store.list_runs() == []


def test_storage_integrity_and_path_safety(batch):
    pipeline, _, store, runner = batch
    save(pipeline)
    manifest = runner.execute(request(), [specification()])
    path = store.directory / manifest["run_id"] / manifest["tickers"]["AAPL"]["detail_file"]
    path.write_bytes(path.read_bytes() + b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        store.load_ticker(manifest["run_id"], "AAPL")
    with pytest.raises(ValueError, match="Invalid batch run ID"):
        store.load_manifest("../escape")
    with pytest.raises(ValueError):
        store.save_aggregation(manifest["run_id"], {"value": float("nan")})
    assert store.load_manifest(manifest["run_id"])["aggregation"] == {}


def test_results_cannot_overlap_market_data(batch):
    pipeline, registry, _, _ = batch
    with pytest.raises(ValueError, match="non-overlapping"):
        BatchRunner(pipeline, registry, BatchStore(pipeline.store.data_dir / "results"))


def test_delete_run_only_removes_selected_analysis(tmp_path):
    store = BatchStore(tmp_path / "analysis")
    first = store.create({})
    second = store.create({})
    store.save_aggregation(first["run_id"], {"instances": {}})
    store.delete_run(first["run_id"])
    assert not (store.directory / first["run_id"]).exists()
    assert [item["run_id"] for item in store.list_runs()] == [second["run_id"]]
    with pytest.raises(ValueError):
        store.delete_run("../analysis")
    link_id = "a" * 32
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.directory / link_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        store.delete_run(link_id)
    assert outside.exists()
