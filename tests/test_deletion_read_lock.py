"""Readers materialize while locked; deletion never races their file access."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

import polars as pl
import pytest

from data_pipeline import DataQuery, DataRequest, LocalDataStore
from data_pipeline.exceptions import StorageBusyError


@pytest.mark.parametrize("method", ["read_dataset", "scan", "audit"])
def test_delete_waits_for_read_verification_and_materialization(tmp_path, sample_ohlcv_frame, monkeypatch, method):
    reader = LocalDataStore(tmp_path / "data")
    request = DataRequest(symbol="AAPL", start=sample_ohlcv_frame["timestamp"][0],
                          end=sample_ohlcv_frame["timestamp"][-1] + timedelta(hours=1))
    stored = reader.write_raw(request, sample_ohlcv_frame)
    query = DataQuery(provider="yahoo", symbol="AAPL", timeframe="1h", start=request.start, end=request.end)
    plan = reader.plan_delete(query)
    competitor = LocalDataStore(reader.data_dir, lock_timeout=0.03)
    entered, release = Event(), Event()
    verify = reader._verified_path

    def blocking_verify(item):
        entered.set()
        assert release.wait(5), "Reader test was not released"
        return verify(item)

    monkeypatch.setattr(reader, "_verified_path", blocking_verify)
    operation = {"read_dataset": lambda: reader.read_dataset(stored.dataset_id),
                 "scan": lambda: reader.scan(query), "audit": reader.audit}[method]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        try:
            assert entered.wait(5)
            with pytest.raises(StorageBusyError):
                competitor.delete(plan, confirm=plan.operation_id)
            assert (reader.data_dir / stored.relative_path).is_file()
        finally:
            release.set()
        result = future.result(timeout=5)
    report = competitor.delete(plan, confirm=plan.operation_id)
    assert report.status == "completed"
    assert not (reader.data_dir / stored.relative_path).exists()
    if method == "audit":
        assert result == [stored.dataset_id]
    else:
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        assert result.equals(sample_ohlcv_frame)
