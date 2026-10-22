"""Offline tests for the official-SDK adapter, with no account or paid calls."""

from datetime import UTC, date, datetime, timedelta
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pandas as pd
import polars as pl
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import (
    DuplicateBarError, EmptyDataError, InvalidDataRequestError, InvalidOHLCVError,
    ProviderError, SchemaValidationError, UnknownProviderError, UnsupportedRequestError,
)
from data_pipeline.providers import DatabentoProvider, ProviderRegistry, get_provider
from data_pipeline.providers.databento import provider_name
from data_pipeline.schemas.ohlcv import OHLCV_SCHEMA, validate_ohlcv

START = datetime(2024, 11, 1, 13, 30, tzinfo=UTC)
END = START + timedelta(minutes=3)


def vendor_frame(*, symbols=("AAPL",), timestamps=None, **overrides):
    count = len(symbols)
    frame = pd.DataFrame({
        "ts_event": pd.DatetimeIndex(timestamps or [START + timedelta(minutes=n) for n in range(count)]),
        "rtype": [33] * count, "publisher_id": [2] * count,
        "instrument_id": list(range(count)), "symbol": list(symbols),
        "open": [100.25] * count, "high": [102.75] * count,
        "low": [99.0] * count, "close": [101.5] * count,
        "volume": pd.Series([1000] * count, dtype="uint64"),
    })
    for column, values in overrides.items():
        frame[column] = values
    return frame.set_index("ts_event")


class Store:
    dataset = "XNAS.ITCH"
    schema = "ohlcv-1m"

    def __init__(self, frames):
        self.frames = frames
        self.to_df = Mock(side_effect=lambda **kwargs: iter(self.frames))


def client(*, store=None):
    return SimpleNamespace(
        key="offline-test-key",
        timeseries=SimpleNamespace(get_range=Mock(return_value=store)),
        metadata=SimpleNamespace(
            list_datasets=Mock(return_value=["XNAS.ITCH", "EQUS.MINI"]),
            list_schemas=Mock(return_value=["trades", "ohlcv-1m", "ohlcv-1h"]),
            get_dataset_range=Mock(return_value={"start": START.isoformat(), "end": END.isoformat(),
                                               "schema": {"ohlcv-1m": {"start": START.isoformat(), "end": END.isoformat()}}}),
            get_dataset_condition=Mock(return_value=[{"date": "2024-11-01", "condition": "degraded"}]),
            get_cost=Mock(return_value=1.25),
        ),
        symbology=SimpleNamespace(resolve=Mock()),
        batch=SimpleNamespace(
            submit_job=Mock(return_value={"id": "XNAS-123", "state": "queued", "api_key": "secret-label"}),
            get_job_details=Mock(return_value={"id": "XNAS-123", "state": "done", "progress": 100}),
            list_files=Mock(), download=Mock(),
        ),
    )


def request(**overrides):
    fields = dict(symbol="AAPL", provider="databento-xnas.itch", timeframe="1m", start=START, end=END)
    fields.update(overrides)
    return DataRequest(**fields)


def test_lazy_registration_and_dataset_identity(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    sdk = Mock(side_effect=AssertionError("Do not import the SDK during discovery"))
    monkeypatch.setattr("data_pipeline.providers.databento.importlib.import_module", sdk)
    assert get_provider("databento").name == "databento"
    assert get_provider("DATABENTO-XNAS.ITCH").dataset == "XNAS.ITCH"
    assert get_provider("databento-equs.mini").name == "databento-equs.mini"
    assert provider_name(" xnas.itch ") == "databento-xnas.itch"
    assert not sdk.called
    with pytest.raises(ProviderError, match="DATABENTO_API_KEY"):
        DatabentoProvider().list_datasets()
    assert not sdk.called


def test_registry_does_not_leak_dynamic_feed_into_other_registries():
    with pytest.raises(UnknownProviderError):
        ProviderRegistry().get("databento-xnas.itch")
    injected = client()
    providers = ProviderRegistry({"databento": DatabentoProvider(client=injected)})
    configured = providers.get("databento-xnas.itch")
    assert configured._client is injected
    assert providers.names() == ("databento",)
    assert configured.list_datasets() == ("XNAS.ITCH", "EQUS.MINI")


def test_fetch_normalizes_sdk_prices_and_half_open_utc_bounds():
    store = Store([vendor_frame(symbols=("AAPL",) * 3, timestamps=[END, START, START - timedelta(minutes=1)])])
    api = client(store=store)
    result = DatabentoProvider("XNAS.ITCH", client=api).fetch_result(request())
    assert result.frame.schema == OHLCV_SCHEMA
    assert result.frame["timestamp"].to_list() == [START]
    assert result.frame["open"].to_list() == [100.25]  # Already SDK-scaled, never divide twice.
    assert result.frame["volume"].to_list() == [1000.0]
    assert result.quality.status == "unknown"
    assert result.quality.skip_missing_ohlc is False
    store.to_df.assert_called_once_with(price_type="float", pretty_ts=True, map_symbols=True,
                                       tz="UTC", count=100_000)
    api.timeseries.get_range.assert_called_once_with(
        dataset="XNAS.ITCH", symbols=["AAPL"], schema="ohlcv-1m", start=START,
        end=END, stype_in="raw_symbol", stype_out="instrument_id")


def test_fetch_all_chunks_and_preserve_date_based_sdk_mapping():
    frames = [vendor_frame(timestamps=[START + timedelta(minutes=i)]) for i in range(3)]
    provider = DatabentoProvider("XNAS.ITCH", client=client(store=Store(frames)), chunk_size=1)
    assert provider.fetch(request()).height == 3


@pytest.mark.parametrize("overrides", [{"provider": "databento-equs.mini"},
                                      {"timeframe": "30m"}, {"symbol": "AAPL,MSFT"}])
def test_invalid_request_cannot_trigger_paid_call(overrides):
    api = client()
    with pytest.raises((UnsupportedRequestError, InvalidDataRequestError)):
        DatabentoProvider("XNAS.ITCH", client=api).fetch(request(**overrides))
    assert not api.timeseries.get_range.called


def test_generic_provider_requires_dataset():
    with pytest.raises(UnsupportedRequestError, match="dataset"):
        DatabentoProvider(client=client()).fetch(request(provider="databento"))


@pytest.mark.parametrize("frames,error", [
    ([], EmptyDataError),
    ([vendor_frame(symbols=("MSFT",))], SchemaValidationError),
    ([vendor_frame(high=[90.0])], InvalidOHLCVError),
    ([vendor_frame(), vendor_frame()], DuplicateBarError),
    ([vendor_frame(symbol=[None])], SchemaValidationError),
    ([vendor_frame(volume=pd.Series([2**53 + 1], dtype="uint64"))], SchemaValidationError),
    ([vendor_frame(timestamps=[pd.Timestamp(START) + pd.Timedelta(nanoseconds=1)])], SchemaValidationError),
])
def test_fetch_rejects_bad_or_missing_data(frames, error):
    with pytest.raises(error):
        DatabentoProvider("XNAS.ITCH", client=client(store=Store(frames))).fetch(request())


def test_bounded_batch_decode_keeps_ticker_validation_independent(tmp_path):
    store = Store([vendor_frame(symbols=("AAPL", "BAD"), high=[102.75, 90.0])])
    loader = Mock(return_value=store)
    provider = DatabentoProvider("XNAS.ITCH", store_loader=loader, chunk_size=500)
    frame, = provider.iter_frames([tmp_path / "metadata.json", tmp_path / "chunk.dbn.zst"], "1m")
    assert frame.height == 2
    assert validate_ohlcv(frame.filter(pl.col("symbol") == "AAPL")).height == 1
    with pytest.raises(InvalidOHLCVError):
        validate_ohlcv(frame.filter(pl.col("symbol") == "BAD"))
    loader.assert_called_once_with(tmp_path / "chunk.dbn.zst")
    assert store.to_df.call_args.kwargs["count"] == 500


@pytest.mark.parametrize("field,value", [("dataset", "EQUS.MINI"), ("schema", "ohlcv-1h")])
def test_decoder_rejects_foreign_feed_or_schema(field, value):
    store = Store([vendor_frame()])
    setattr(store, field, value)
    with pytest.raises(SchemaValidationError, match="dataset/schema"):
        DatabentoProvider("XNAS.ITCH", client=client(store=store)).fetch(request())


def test_decoder_treats_truncation_warning_as_failure():
    store = Store([])
    def truncated(**kwargs):
        warnings.warn("DBN file is truncated or contains an incomplete record")
        return iter([])
    store.to_df.side_effect = truncated
    with pytest.raises(ProviderError, match="decode complete"):
        DatabentoProvider("XNAS.ITCH", client=client(store=store)).fetch(request())


def test_metadata_boundaries_and_500_symbol_preflight():
    symbols = [f"T{i}" for i in range(500)]
    api = client()
    api.symbology.resolve.return_value = {
        "result": {symbol: [{"s": str(i), "d0": "2024-11-01", "d1": "2024-11-02"}]
                   for i, symbol in enumerate(symbols[:-1])},
        "partial": [symbols[0]], "not_found": [symbols[-1]],
    }
    provider = DatabentoProvider("XNAS.ITCH", client=api)
    result = provider.preflight_symbols(symbols, START, END)
    assert len(result["valid"]) == 499
    assert list(result["errors"]) == ["T499"]
    assert "T0" in result["warnings"][0]
    api.symbology.resolve.assert_called_once()
    assert api.symbology.resolve.call_args.kwargs["end_date"].isoformat() == "2024-11-02"
    midnight = datetime(2024, 11, 2, tzinfo=UTC)
    assert provider.get_dataset_conditions(START, midnight)[0]["condition"] == "degraded"
    assert api.metadata.get_dataset_condition.call_args.kwargs["end_date"].isoformat() == "2024-11-01"
    assert "ohlcv-1m" in provider.list_schemas()
    assert "schema" in provider.get_dataset_range()


def test_cost_and_job_use_same_native_request_without_secrets():
    api = client()
    provider = DatabentoProvider("XNAS.ITCH", client=api)
    assert provider.estimate_cost(["AAPL", "MSFT"], START, END, "1m") == 1.25
    job = provider.submit_job(["AAPL", "MSFT"], START, END, "1m")
    assert job == {"id": "XNAS-123", "state": "queued"}
    estimate = api.metadata.get_cost.call_args.kwargs
    submitted = api.batch.submit_job.call_args.kwargs
    assert all(submitted[k] == v for k, v in estimate.items())
    assert submitted["encoding"] == "dbn"
    assert submitted["split_duration"] == "day"
    assert provider.job_status(job["id"])["state"] == "done"


@pytest.mark.parametrize("cost", [float("inf"), float("nan"), -1, "invalid", True])
def test_invalid_cost_cannot_be_used(cost):
    api = client()
    api.metadata.get_cost.return_value = cost
    with pytest.raises(ProviderError, match="cost estimate"):
        DatabentoProvider("XNAS.ITCH", client=api).estimate_cost(["AAPL"], START, END, "1m")


def test_no_blind_submit_retry_and_no_secret_exception():
    api = client()
    api.batch.submit_job.side_effect = RuntimeError("secret-key Authorization=secret")
    with pytest.raises(ProviderError) as error:
        DatabentoProvider("XNAS.ITCH", client=api).submit_job(["AAPL"], START, END, "1m")
    assert "secret" not in str(error.value)
    assert error.value.__cause__ is None
    api.batch.submit_job.assert_called_once()


@pytest.mark.parametrize("symbols", [[], ["ALL_SYMBOLS"], [" AAPL"], ["AAPL,MSFT"],
                                    [f"T{i}" for i in range(2001)]])
def test_symbol_request_limits(symbols):
    api = client()
    with pytest.raises(InvalidDataRequestError):
        DatabentoProvider("XNAS.ITCH", client=api).estimate_cost(symbols, START, END, "1m")
    assert not api.metadata.get_cost.called


class DownloadResponse:
    status_code = 200
    headers = {}

    def __init__(self, contents):
        self.contents = contents

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.contents


def mock_download(monkeypatch, api, contents, *, checksum=None):
    api.batch.list_files.return_value = [{
        "filename": "chunk.dbn.zst", "size": len(contents),
        "hash": "sha256:" + (checksum or hashlib.sha256(contents).hexdigest()),
        "urls": {"https": "https://downloads.databento.com/XNAS-123/chunk.dbn.zst"},
    }]
    get = Mock(side_effect=lambda *args, **kwargs: DownloadResponse(contents))
    session = Mock()
    session.get = get
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("data_pipeline.providers.databento_download.requests.Session", lambda: session)
    return get


def test_download_verifies_files_and_reuses_one_manifest(tmp_path, monkeypatch):
    contents = b"example DBN test payload"
    api = client()
    get = mock_download(monkeypatch, api, contents)
    paths = DatabentoProvider("XNAS.ITCH", client=api).download_job("XNAS-123", tmp_path)
    assert paths == (tmp_path / "XNAS-123" / "chunk.dbn.zst",)
    assert paths[0].read_bytes() == contents
    api.batch.list_files.assert_called_once_with(job_id="XNAS-123")
    get.assert_called_once()
    assert not api.batch.download.called


def test_download_rejects_bad_checksums(tmp_path, monkeypatch):
    api = client()
    mock_download(monkeypatch, api, b"bad", checksum="0" * 64)
    with pytest.raises(ProviderError, match="checksum"):
        DatabentoProvider("XNAS.ITCH", client=api).download_job("XNAS-123", tmp_path)
    assert not (tmp_path / "XNAS-123" / "chunk.dbn.zst").exists()


@pytest.mark.parametrize("corrupt", [b"bad", b"oversized-invalid-file"])
def test_retry_repairs_job_cache_without_resubmitting(tmp_path, monkeypatch, corrupt):
    contents = b"DBN"
    path = tmp_path / "XNAS-123" / "chunk.dbn.zst"
    path.parent.mkdir()
    path.write_bytes(corrupt)
    api = client()
    get = mock_download(monkeypatch, api, contents)
    provider = DatabentoProvider("XNAS.ITCH", client=api)
    assert provider.download_job("XNAS-123", tmp_path) == (path,)
    assert path.read_bytes() == contents
    get.assert_called_once()
    assert not api.batch.submit_job.called


def test_download_rejects_manifest_path_escape_before_download(tmp_path):
    api = client()
    api.batch.list_files.return_value = [{"filename": "../../escape.dbn"}]
    with pytest.raises(ProviderError, match="filename"):
        DatabentoProvider("XNAS.ITCH", client=api).download_job("XNAS-123", tmp_path)
    assert not api.batch.download.called


def test_native_feed_namespaces_store_separately(tmp_path):
    providers = ProviderRegistry({"databento": DatabentoProvider(client=client(store=Store([vendor_frame()])))})
    pipeline = DataPipeline(tmp_path / "data", providers=providers)
    first = pipeline.ingest(request()).raw
    other_store = Store([vendor_frame()])
    other_store.dataset = "EQUS.MINI"
    providers.register("databento-equs.mini", DatabentoProvider("EQUS.MINI", client=client(store=other_store)))
    second = pipeline.ingest(request(provider="databento-equs.mini")).raw
    assert first.dataset_id != second.dataset_id
    assert pipeline.read(DataQuery(provider="databento-xnas.itch")).height == 1
    assert pipeline.read(DataQuery(provider="databento-equs.mini")).height == 1


def real_dbn_bytes():
    """Construct an actual SDK DBN stream locally, with an ID reused on another date."""
    import databento_dbn as dbn

    later = START + timedelta(days=3)
    metadata = dbn.Metadata(
        dataset="XNAS.ITCH", start=pd.Timestamp(START).value,
        end=pd.Timestamp(later + timedelta(minutes=1)).value,
        stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
        schema=dbn.Schema.OHLCV_1M, symbols=["AAPL", "MSFT"],
        mappings=[
            SimpleNamespace(raw_symbol="AAPL", intervals=[SimpleNamespace(
                start_date=date(2024, 11, 1), end_date=date(2024, 11, 2), symbol="123")]),
            SimpleNamespace(raw_symbol="MSFT", intervals=[SimpleNamespace(
                start_date=date(2024, 11, 4), end_date=date(2024, 11, 5), symbol="123")]),
        ],
    )
    records = [dbn.OHLCVMsg(
        rtype=dbn.RType.OHLCV_1M, publisher_id=2, instrument_id=123,
        ts_event=pd.Timestamp(timestamp).value,
        open=100_250_000_000, high=102_750_000_000, low=99_000_000_000,
        close=101_500_000_000, volume=1000,
    ) for timestamp in (START, later)]
    return metadata.encode() + b"".join(bytes(record) for record in records)


@pytest.mark.parametrize("compressed", [False, True])
def test_actual_sdk_file_roundtrip_prices_timestamps_and_date_mapping(tmp_path, compressed):
    import zstandard

    data = real_dbn_bytes()
    path = tmp_path / ("bars.dbn.zst" if compressed else "bars.dbn")
    path.write_bytes(zstandard.ZstdCompressor().compress(data) if compressed else data)
    provider = DatabentoProvider("XNAS.ITCH", chunk_size=1)
    chunks = list(provider.iter_frames([path], "1m"))
    assert len(chunks) == 2
    assert all(chunk.height == 1 for chunk in chunks)
    bars = validate_ohlcv(pl.concat(chunks))
    assert bars["symbol"].to_list() == ["AAPL", "MSFT"]
    assert bars["timestamp"].to_list() == [START, START + timedelta(days=3)]
    assert bars["open"].to_list() == [100.25, 100.25]
    assert bars["high"].to_list() == [102.75, 102.75]
    assert bars["volume"].to_list() == [1000.0, 1000.0]


def test_actual_sdk_truncated_record_is_not_silently_imported(tmp_path):
    path = tmp_path / "truncated.dbn"
    path.write_bytes(real_dbn_bytes()[:-1])
    with pytest.raises(ProviderError, match="decode complete"):
        list(DatabentoProvider("XNAS.ITCH").iter_frames([path], "1m"))
