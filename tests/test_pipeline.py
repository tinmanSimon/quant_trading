"""Exercise the package and CLI through public entry points without a vendor."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import subprocess
import sys

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.cli import main
from data_pipeline.exceptions import (
    DataAlreadyExistsError, InvalidDataRequestError, OverlappingDataError,
    ProcessingError, ProviderError, RequestDataMismatchError, SchemaValidationError,
)
from data_pipeline.processing import ScalePrices
from data_pipeline.providers import BaseDataProvider, ProviderRegistry
from data_pipeline.schemas import validate_ohlcv


class FakeProvider(BaseDataProvider):
    def __init__(self, frame):
        self.frame = frame
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        return self.frame.clone()


@pytest.fixture
def request_data():
    return DataRequest("AAPL", datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                       datetime(2024, 1, 2, 17, 30, tzinfo=UTC))


@pytest.fixture
def setup_pipeline(isolated_data_dir, sample_ohlcv_frame):
    provider = FakeProvider(sample_ohlcv_frame)
    registry = ProviderRegistry({"yahoo": provider, "other": provider})
    return DataPipeline(isolated_data_dir, providers=registry), provider


def test_ingest_raw_only_and_reopen(setup_pipeline, request_data, sample_ohlcv_frame):
    pipeline, provider = setup_pipeline
    result = pipeline.ingest(request_data)
    assert result.processed is None
    assert provider.requests == [request_data]
    restarted = DataPipeline(pipeline.store.data_dir)
    assert_frame_equal(restarted.read_dataset(result.raw.dataset_id), sample_ohlcv_frame)
    assert len(restarted.list_coverage()) == 1
    assert pipeline.process(result.raw.dataset_id, processors=[]) == result.raw
    assert len(pipeline.list_datasets()) == 1


def test_provider_switch_without_changing_storage(setup_pipeline, request_data):
    pipeline, provider = setup_pipeline
    a = pipeline.ingest(request_data)
    b = pipeline.ingest(replace(request_data, provider="other"))
    assert a.raw.dataset_id != b.raw.dataset_id
    assert_frame_equal(pipeline.read(DataQuery(provider="other")),
                       pipeline.read(DataQuery(provider="yahoo")))
    with pytest.raises(RequestDataMismatchError, match="Narrow query"):
        pipeline.read(DataQuery())


@pytest.mark.parametrize("symbol", ["aapl", "AaPl"])
def test_yahoo_ticker_case_cannot_bypass_duplicate_protection(
    setup_pipeline, request_data, sample_ohlcv_frame, symbol
):
    pipeline, provider = setup_pipeline
    first = pipeline.ingest(request_data)
    incoming = replace(request_data, symbol=symbol)
    # A provider echoes the request identity, as the Yahoo adapter does.
    provider.frame = sample_ohlcv_frame.with_columns(pl.lit(incoming.symbol).alias("symbol"))
    with pytest.raises(OverlappingDataError):
        pipeline.ingest(incoming)
    assert pipeline.list_datasets() == [first.raw]
    for provider_name in (None, "yahoo"):
        query = DataQuery(provider=provider_name, symbol=symbol, timeframe="1h")
        assert_frame_equal(pipeline.read(query), sample_ohlcv_frame)


def test_other_providers_keep_case_sensitive_instrument_identifiers(request_data):
    assert replace(request_data, symbol="aBc", provider="other").symbol == "aBc"
    assert DataQuery(provider="other", symbol="aBc").symbol == "aBc"


def test_fetch_failure_does_not_save(setup_pipeline, request_data, monkeypatch):
    pipeline, provider = setup_pipeline

    def fail(request):
        raise ProviderError("vendor unavailable")

    monkeypatch.setattr(provider, "fetch", fail)
    with pytest.raises(ProviderError):
        pipeline.ingest(request_data)
    assert not list(pipeline.store.data_dir.rglob("*.parquet"))
    assert pipeline.list_datasets() == []


def test_ingest_process_query_and_reprocess(setup_pipeline, request_data, sample_ohlcv_frame):
    pipeline, _ = setup_pipeline
    result = pipeline.ingest(request_data, processors=[ScalePrices(2)])
    assert result.processed.parent_ids == (result.raw.dataset_id,)
    assert_frame_equal(pipeline.read_dataset(result.raw.dataset_id), sample_ohlcv_frame)
    filtered = pipeline.read(DataQuery(layer="processed", start=request_data.start + timedelta(hours=1),
                                       end=request_data.end - timedelta(hours=1)), columns=["timestamp", "close"])
    assert filtered.height == 1
    assert filtered["close"][0] == sample_ohlcv_frame["close"][1] * 2
    with pytest.raises(DataAlreadyExistsError):
        pipeline.process(result.raw.dataset_id, processors=[ScalePrices(2)])
    other = pipeline.process(result.raw.dataset_id, processors=[ScalePrices(3)])
    assert other.pipeline_id != result.processed.pipeline_id
    with pytest.raises(RequestDataMismatchError):
        pipeline.read(DataQuery(layer="processed"))
    assert pipeline.read(DataQuery(layer="processed", pipeline_id=other.pipeline_id)).height == 3


def test_processing_failure_leaves_recoverable_raw_id(setup_pipeline, request_data):
    pipeline, _ = setup_pipeline

    class Failing:
        name, version, config = "failing", "1", {}

        def transform(self, frame, input_contract):
            raise RuntimeError("processor broke")

    with pytest.raises(ProcessingError, match="Raw data was saved as") as caught:
        pipeline.ingest(request_data, processors=[Failing()])
    raw = pipeline.list_datasets(DataQuery())[0]
    assert raw.dataset_id in str(caught.value)
    assert not pipeline.list_datasets(DataQuery(layer="processed"))
    assert pipeline.process(raw.dataset_id, processors=[ScalePrices(2)]).parent_ids == (raw.dataset_id,)


def test_public_ingest_overlap_aborts(setup_pipeline, request_data):
    pipeline, _ = setup_pipeline
    result = pipeline.ingest(request_data)
    with pytest.raises(OverlappingDataError):
        pipeline.ingest(request_data)
    assert pipeline.list_datasets() == [result.raw]


def test_cli_full_workflow(setup_pipeline, request_data, monkeypatch, capsys, tmp_path):
    pipeline, provider = setup_pipeline
    monkeypatch.setattr("data_pipeline.api.default_providers", pipeline.providers)
    prefix = ["--data-dir", str(pipeline.store.data_dir)]
    ingest = ["ingest", "--symbol", "AAPL", "--timeframe", "1h", "--start",
              request_data.start.isoformat(), "--end", request_data.end.isoformat()]
    assert main(prefix + ingest) == 0
    raw_id = json.loads(capsys.readouterr().out)["raw_id"]
    assert main(prefix + ["list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["dataset_id"] == raw_id
    assert main(prefix + ["inspect", raw_id]) == 0
    assert json.loads(capsys.readouterr().out)["row_count"] == 3
    assert main(prefix + ["read", "--dataset-id", raw_id]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3

    config = tmp_path / "processors.json"
    config.write_text('[{"name":"scale_prices","version":"1","config":{"factor":2}}]')
    assert main(prefix + ["process", raw_id, "--processors", str(config)]) == 0
    processed = json.loads(capsys.readouterr().out)
    assert processed["parent_ids"] == [raw_id]
    assert main(prefix + ingest) == 1
    assert "overlaps" in capsys.readouterr().err
    assert main(prefix + ["replace", raw_id, "--confirm", "wrong-id"]) == 1
    assert "confirm" in capsys.readouterr().err
    assert len(provider.requests) == 2  # Initial fetch and duplicate ingest only.
    assert main(prefix + ["replace", raw_id, "--confirm", raw_id]) == 0
    replacement = json.loads(capsys.readouterr().out)
    assert replacement["supersedes"] == [raw_id]
    assert main(prefix + ["audit"]) == 0
    assert len(json.loads(capsys.readouterr().out)["verified"]) == 3
    assert main(prefix + ["recover"]) == 0
    assert json.loads(capsys.readouterr().out) == {"quarantined": []}


def test_module_cli_works_from_outside_project(tmp_path):
    result = subprocess.run([sys.executable, "-m", "data_pipeline", "--data-dir", str(tmp_path / "empty"),
                             "list"], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_cli_dates_are_utc_and_naive_times_rejected(tmp_path, capsys):
    from data_pipeline.cli import _datetime

    assert _datetime("2024-01-02") == datetime(2024, 1, 2, tzinfo=UTC)
    with pytest.raises(SystemExit) as caught:
        main(["ingest", "--symbol", "AAPL", "--start", "2024-01-02T12:00", "--end", "2024-01-03"])
    assert caught.value.code == 2


@pytest.mark.parametrize("values", [{"start": datetime(2024, 1, 1)}, {"layer": "typo"},
                                   {"start": datetime(2024, 2, 1, tzinfo=UTC),
                                    "end": datetime(2024, 1, 1, tzinfo=UTC)}])
def test_invalid_query_bounds(values):
    with pytest.raises(InvalidDataRequestError):
        DataQuery(**values)


def test_timeframe_alias_has_one_storage_identity(request_data):
    assert replace(request_data, timeframe="60m") == request_data
    assert DataQuery(timeframe="60m") == DataQuery(timeframe="1h")


def test_storage_rejects_forged_processor_fingerprint(setup_pipeline, request_data):
    pipeline, _ = setup_pipeline
    raw = pipeline.ingest(request_data).raw
    with pytest.raises(RequestDataMismatchError, match="fingerprint"):
        pipeline.store.write_processed(raw.dataset_id, pipeline.read_dataset(raw.dataset_id),
                                       pipeline_id="wrong", processors_json='[{"name":"x","version":"1","config":{}}]')
    assert pipeline.list_datasets(DataQuery(layer="processed")) == []


def test_representative_large_batch_compression_and_range_query(isolated_data_dir):
    from time import perf_counter

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=100_000)
    frame = pl.DataFrame({"timestamp": pl.datetime_range(start, end, interval="1m", closed="left",
                                                         time_unit="ms", eager=True)}).with_columns(
        pl.lit("AAPL").alias("symbol"), pl.lit(100.0).alias("open"), pl.lit(101.0).alias("high"),
        pl.lit(99.0).alias("low"), pl.lit(100.5).alias("close"), pl.lit(1000.0).alias("volume"),
    )
    pipeline = DataPipeline(isolated_data_dir)
    begun = perf_counter()
    stored = pipeline.ingest_frame(DataRequest("AAPL", start, end, timeframe="1m"), frame).raw
    query = DataQuery(provider="yahoo", symbol="AAPL", timeframe="1m", start=start, end=start + timedelta(hours=1))
    assert pipeline.read(query, columns=["timestamp", "close"]).height == 60
    assert (isolated_data_dir / stored.relative_path).stat().st_size < frame.estimated_size()
    # Generous local regression ceiling; this fixture contains 100,000 bars.
    assert perf_counter() - begun < 30


def test_validator_does_not_truncate_submillisecond_bar_keys(sample_ohlcv_frame):
    frame = sample_ohlcv_frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                                           + pl.duration(microseconds=1))
    with pytest.raises(SchemaValidationError, match="precision"):
        validate_ohlcv(frame)


def test_validator_does_not_round_large_integer_values(sample_ohlcv_frame):
    frame = sample_ohlcv_frame.with_columns(pl.lit(2**53 + 1, dtype=pl.Int64).alias("volume"))
    with pytest.raises(SchemaValidationError, match="losslessly"):
        validate_ohlcv(frame)
