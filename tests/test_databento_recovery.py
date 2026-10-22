"""Cached Databento recovery never purchases or synthesizes replacement bars."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest, InvalidDataRequestError
from data_pipeline import databento_batch
from data_pipeline.exceptions import InvalidOHLCVError
from data_pipeline.providers import ProviderRegistry
from data_pipeline.schemas import validate_ohlcv

from test_databento_batch import FakeProvider, complete


def stamp(value):
    return datetime.fromisoformat(value).astimezone(UTC)


def history(symbol, timestamps, *, invalid=()):
    values = [stamp(value) if isinstance(value, str) else value for value in timestamps]
    frame = pl.DataFrame({
        "timestamp": pl.Series(values, dtype=pl.Datetime("ms", "UTC")),
        "symbol": [symbol] * len(values),
        "open": [100.0] * len(values), "high": [102.0] * len(values),
        "low": [99.0] * len(values), "close": [101.0] * len(values),
        "volume": [10.0] * len(values),
    })
    invalid_stamps = pl.Series([values[index] for index in invalid], dtype=frame.schema["timestamp"])
    return frame.with_columns(
        pl.when(pl.col("timestamp").is_in(invalid_stamps.implode()))
        .then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col(column)).alias(column)
        for column in ("open", "high", "low", "close")
    )


@pytest.fixture
def cached_failure(tmp_path, monkeypatch):
    provider = FakeProvider()
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({provider.name: provider}))
    timestamps = [
        "2025-01-02T14:30:00+00:00",  # Regular session, valid.
        "2025-01-02T22:00:00+00:00",  # Extended session, undefined prices.
        "2025-01-02T22:01:00+00:00",  # Extended session, valid and retained.
    ]
    source = {
        "AAPL": history("AAPL", timestamps, invalid=(1,)),
        "MSFT": history("MSFT", timestamps),
    }

    def frames(paths, timeframe):
        yield from source.values()

    monkeypatch.setattr(provider, "iter_frames", frames)
    plan = pipeline.estimate_databento(
        list(source), start=stamp(timestamps[0]),
        end=stamp(timestamps[-1]) + timedelta(minutes=1), dataset="XNAS.ITCH",
    )
    plan = complete(pipeline, plan)
    assert plan["status"] == "completed_with_errors"
    assert {item["ticker"]: item["status"] for item in plan["outcomes"]} == {
        "AAPL": "failed", "MSFT": "saved",
    }

    def forbid_provider(*args, **kwargs):
        raise AssertionError("Cached recovery must not construct or access a provider")

    monkeypatch.setattr(databento_batch, "_provider", forbid_provider)
    return pipeline, provider, plan, source


def outcomes(plan):
    assert len({item["ticker"] for item in plan["outcomes"]}) == len(plan["outcomes"])
    return {item["ticker"]: item for item in plan["outcomes"]}


def test_offline_recovery_preserves_successful_tickers_and_valid_extended_bars(cached_failure):
    pipeline, provider, plan, source = cached_failure
    successful = outcomes(plan)["MSFT"]
    original = pipeline.read_dataset(successful["dataset_ids"][0])
    submissions, downloads = list(provider.submissions), list(provider.downloads)

    recovered = pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)

    assert recovered["status"] == "completed"
    result = outcomes(recovered)
    assert result["MSFT"] == successful
    assert pipeline.read_dataset(successful["dataset_ids"][0]).equals(original)
    assert result["AAPL"]["status"] == "saved"
    assert result["AAPL"]["row_count"] == result["AAPL"]["fetched_row_count"] == 2
    frame = pipeline.read_dataset(result["AAPL"]["dataset_ids"][0])
    expected = source["AAPL"].filter(pl.col("open").is_not_null())
    assert frame.equals(expected)
    assert provider.submissions == submissions
    assert provider.downloads == downloads
    quality = result["AAPL"]["quality"]
    assert quality["status"] == "reported"
    assert quality["skip_missing_ohlc"] is True
    assert len(quality["omitted_bars"]) == 1
    assert stamp(quality["omitted_bars"][0]["timestamp"]) == stamp("2025-01-02T22:00:00+00:00")
    # The decoded original remains intact and can be independently inspected.
    parts = plan["groups"][0]["parts"]["AAPL"]
    assert databento_batch._read_parts(pipeline, parts).equals(source["AAPL"])


def test_recovery_is_strict_until_omissions_are_explicitly_enabled(cached_failure):
    pipeline, _, plan, _ = cached_failure

    strict = pipeline.recover_databento(plan["id"])

    assert strict["status"] == "completed_with_errors"
    assert outcomes(strict)["AAPL"]["status"] == "failed"
    assert len(pipeline.list_datasets()) == 1
    recovered = pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)
    assert recovered["status"] == "completed"


def test_recovery_is_idempotent_and_survives_pipeline_restart(cached_failure):
    pipeline, provider, plan, _ = cached_failure
    recovered = pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)
    identifiers = {item.dataset_id for item in pipeline.list_datasets()}
    reopened = DataPipeline(pipeline.store.data_dir, providers=pipeline.providers)

    repeated = reopened.recover_databento(plan["id"], drop_invalid_outside_regular=True)

    assert repeated["status"] == "completed"
    assert outcomes(repeated) == outcomes(recovered)
    assert {item.dataset_id for item in reopened.list_datasets()} == identifiers
    assert len(provider.submissions) == 1


def test_recovery_recognizes_commit_before_interrupted_journal_save(cached_failure, monkeypatch):
    pipeline, _, plan, _ = cached_failure
    merge = pipeline.store.merge_raw

    def interrupted_merge(*args, **kwargs):
        merge(*args, **kwargs)
        raise KeyboardInterrupt("Simulated process interruption after commit")

    monkeypatch.setattr(pipeline.store, "merge_raw", interrupted_merge)
    with pytest.raises(KeyboardInterrupt):
        pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)
    assert pipeline.databento_plan(plan["id"])["ticker_plans"]["AAPL"]["status"] == "publishing"
    identifiers = {item.dataset_id for item in pipeline.list_datasets()}
    assert len(identifiers) == 2
    monkeypatch.setattr(pipeline.store, "merge_raw", merge)

    recovered = pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)

    assert recovered["status"] == "completed"
    assert outcomes(recovered)["AAPL"]["status"] == "saved"
    assert {item.dataset_id for item in pipeline.list_datasets()} == identifiers


def test_recovery_does_not_replace_newer_local_data(cached_failure):
    pipeline, provider, plan, source = cached_failure
    item = plan["ticker_plans"]["AAPL"]["request"]
    request = DataRequest("AAPL", stamp(item["start"]), stamp(item["end"]),
                          timeframe="1m", provider=provider.name)
    valid = source["AAPL"].filter(pl.col("open").is_not_null())
    added = pipeline.ingest_frame(request, valid).raw

    with pytest.raises(InvalidDataRequestError, match="local coverage changed"):
        pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)

    assert outcomes(pipeline.databento_plan(plan["id"]))["AAPL"]["status"] == "failed"
    assert len(pipeline.list_datasets()) == 2
    assert pipeline.read_dataset(added.dataset_id).equals(valid)
    assert added.dataset_id in {item.dataset_id for item in pipeline.list_datasets()}


def test_missing_cached_part_fails_locally_without_affecting_successful_data(cached_failure):
    pipeline, provider, plan, _ = cached_failure
    path = pipeline.store.data_dir / "metadata/databento" / plan["groups"][0]["parts"]["AAPL"][0]
    path.unlink()

    with pytest.raises(InvalidDataRequestError, match="cache is missing"):
        pipeline.recover_databento(plan["id"], drop_invalid_outside_regular=True)

    recovered = pipeline.databento_plan(plan["id"])
    assert outcomes(recovered)["AAPL"]["status"] == "failed"
    assert outcomes(recovered)["MSFT"] == outcomes(plan)["MSFT"]
    assert len(pipeline.list_datasets()) == 1
    assert len(provider.submissions) == 1


@pytest.mark.parametrize("outside", [
    "2025-07-03T17:00:00+00:00",  # Early close at 13:00 New York.
    "2025-07-04T15:00:00+00:00",  # Exchange holiday.
    "2025-03-07T14:29:00+00:00",  # EST: just before 09:30.
    "2025-03-10T13:29:00+00:00",  # EDT: just before 09:30.
])
def test_cleaner_uses_calendar_holidays_early_closes_and_dst(outside):
    frame = history("AAPL", ["2025-03-06T15:00:00+00:00", outside], invalid=(1,)).sort("timestamp")

    result = databento_batch._recover_frame(frame, "1m")

    assert validate_ohlcv(result.frame).height == 1
    assert result.quality.skip_missing_ohlc is True
    assert [item.timestamp for item in result.quality.omitted_bars] == [stamp(outside)]
    assert frame.height == 2


@pytest.mark.parametrize("regular,timeframe", [
    ("2025-03-07T14:30:00+00:00", "1m"),  # EST regular open.
    ("2025-03-10T13:30:00+00:00", "1m"),  # EDT regular open.
    ("2025-07-03T16:59:00+00:00", "1m"),  # Last minute before early close.
    ("2025-01-02T14:00:00+00:00", "1h"),  # 09:00-10:00 overlaps 09:30 open.
])
def test_cleaner_never_discards_invalid_bars_overlapping_regular_sessions(regular, timeframe):
    frame = history("AAPL", [regular], invalid=(0,))

    with pytest.raises(InvalidOHLCVError):
        validate_ohlcv(databento_batch._recover_frame(frame, timeframe).frame)


@pytest.mark.parametrize("column,value", [("volume", -1.0), ("volume", None), ("high", -1.0)])
def test_cleaner_does_not_hide_other_corruption_outside_regular_hours(column, value):
    frame = history("AAPL", ["2025-01-02T22:00:00+00:00"])
    frame = frame.with_columns(pl.lit(value, dtype=pl.Float64).alias(column))

    with pytest.raises(InvalidOHLCVError):
        validate_ohlcv(databento_batch._recover_frame(frame, "1m").frame)

