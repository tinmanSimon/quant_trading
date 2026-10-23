"""Adjustment provenance survives aggregation, storage and legacy catalog reloads."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import DataIntegrityError, ProcessingError, RequestDataMismatchError
from data_pipeline.processing import BaseProcessor, DataContract, Pipeline
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.intraday import ResampleIntraday
from data_pipeline.processing.processors.regular_hours import ResampleRegularHours30m
from data_pipeline.processing.processors.resampling import ResampleOHLCV
from data_pipeline.schemas import OHLCV_SCHEMA
from data_pipeline.storage.models import StoredDataset
from data_pipeline.storage.store import LocalDataStore


METADATA = '{"mode":"split_and_cash_dividend","reference_date":"2024-01-03","snapshot_sha256":"abc"}'


class RecordedAdjustment(BaseProcessor):
    """Isolate provenance propagation from the event calculation itself."""
    name, version, config = "record_adjustment_for_test", "1", {}

    def output_contract(self, input_contract):
        return replace(input_contract, source_basis="unadjusted", adjustment_json=METADATA)

    def _validate_input(self, frame, input_contract):
        pass

    def _transform(self, frame, input_contract, output_contract):
        return frame

    def _validate_output(self, original, output, input_contract, output_contract):
        pass


def test_contract_metadata_is_canonical_and_immutable():
    original = DataContract("1m", source_basis="unadjusted", adjustment_json=' { "z": 1, "a": {"b": 2} } ')
    assert original.adjustment_json == '{"a":{"b":2},"z":1}'
    parsed = json.loads(original.adjustment_json)
    parsed["a"]["b"] = 3
    assert json.loads(original.adjustment_json)["a"]["b"] == 2
    assert replace(original, timeframe="1d", timestamp_convention=None).adjustment_json == original.adjustment_json


@pytest.mark.parametrize("metadata", [None, {}, "{}", "[]", "null", "1", "bad", '{"a":NaN}',
                                      '{"a":Infinity}', '{"a":1,"a":2}', '{"a":{"b":1,"b":2}}'])
def test_invalid_adjustment_metadata_is_rejected(metadata):
    with pytest.raises(ProcessingError, match="[Aa]djustment metadata"):
        DataContract("1m", adjustment_json=metadata)


def test_unknown_request_basis_is_not_assumed_unadjusted():
    request = DataRequest("AAPL", datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC), timeframe="1m")
    assert request.price_adjustment == "unadjusted"
    assert DataContract.from_request(request) == DataContract("1m")
    assert DataContract.from_request(request).source_basis == "unknown"
    with pytest.raises(ProcessingError, match="Source basis"):
        DataContract("1m", source_basis="adjusted")


@pytest.mark.parametrize("kind", ["intraday", "regular30", "regular_daily", "hourly_daily"])
def test_resampler_contracts_preserve_adjustment_metadata(kind):
    day = date(2024, 1, 2)
    daily = ResampleRegularHoursDaily.for_dates(day, day)
    processors = {
        "intraday": ResampleIntraday(daily.sessions),
        "regular30": ResampleRegularHours30m(daily.sessions),
        "regular_daily": daily,
        "hourly_daily": ResampleOHLCV(daily.sessions),
    }
    contract = DataContract("1h" if kind == "hourly_daily" else "1m",
                            source_basis="unadjusted", adjustment_json=METADATA)
    output = processors[kind].output_contract(contract)
    assert output.source_basis == contract.source_basis
    assert output.adjustment_json == contract.adjustment_json
    assert output.timestamp_convention == ("session_date" if "daily" in kind else "bar_start")


def test_processed_daily_reload_and_query_retain_adjustment_contract(tmp_path):
    daily = ResampleRegularHoursDaily.for_dates(date(2024, 1, 2), date(2024, 1, 2))
    session, = daily.sessions
    frame = pl.DataFrame([
        {"timestamp": session.open + timedelta(minutes=i), "symbol": "AAPL", "open": 10.,
         "high": 12., "low": 9., "close": 11., "volume": 20.}
        for i in range(390)
    ], schema=OHLCV_SCHEMA)
    api = DataPipeline(data_dir=tmp_path)
    request = DataRequest("AAPL", session.open, session.close, timeframe="1m")
    raw = api.ingest_frame(request, frame).raw
    processed = api.process(raw.dataset_id, processors=[RecordedAdjustment(), daily])
    expected = DataContract("1d", source_basis="unadjusted", adjustment_json=METADATA)
    assert processed.contract == expected
    restarted = LocalDataStore(tmp_path)
    assert restarted.get_metadata(processed.dataset_id).contract == expected
    query = DataQuery(layer="processed", symbol="AAPL", timeframe="1d")
    assert restarted.list_datasets(query)[0].contract == expected
    assert restarted.read(query)["volume"].to_list() == [7800.]
    assert restarted.get_metadata(raw.dataset_id).contract == DataContract("1m")


def test_old_metadata_without_adjustment_fields_still_loads(tmp_path, sample_ohlcv_frame):
    request = DataRequest("AAPL", sample_ohlcv_frame["timestamp"][0],
                          sample_ohlcv_frame["timestamp"][-1] + timedelta(hours=1))
    item = LocalDataStore(tmp_path).write_raw(request, sample_ohlcv_frame)
    legacy = json.loads(item.to_json())
    del legacy["source_basis"]
    del legacy["adjustment_json"]
    restored = StoredDataset.from_json(json.dumps(legacy), active=True)
    assert restored.contract == DataContract("1h")
    legacy["adjustment_json"] = '{"factor":NaN}'
    with pytest.raises(DataIntegrityError, match="adjustment metadata"):
        StoredDataset.from_json(json.dumps(legacy), active=True)


def test_query_refuses_to_mix_adjustment_reference_dates(tmp_path, sample_ohlcv_frame):
    store = LocalDataStore(tmp_path)
    pipeline = Pipeline([RecordedAdjustment()])
    for index in range(2):
        frame = sample_ohlcv_frame.with_columns(pl.col("timestamp") + timedelta(days=index))
        request = DataRequest("AAPL", frame["timestamp"][0], frame["timestamp"][-1] + timedelta(hours=1))
        raw = store.write_raw(request, frame)
        metadata = json.loads(METADATA)
        metadata["reference_date"] = f"2024-01-0{index + 3}"
        store.write_processed(raw.dataset_id, frame, pipeline_id=pipeline.fingerprint,
                              processors_json=pipeline.canonical_json,
                              output_contract=DataContract("1h", source_basis="unadjusted",
                                                           adjustment_json=json.dumps(metadata)))
    with pytest.raises(RequestDataMismatchError, match="incompatible price adjustment"):
        store.read(DataQuery(layer="processed"))
