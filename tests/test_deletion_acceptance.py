"""Independent acceptance checks for permanent, layer-specific range deletion.

Every artifact in these tests is generated under pytest's temporary directory.
Assertions use the public pipeline interface and inspect the actual surviving
Parquet files so hidden historical copies cannot satisfy a deletion request.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
import math
import shutil

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import (
    ConfirmationRequiredError,
    DatasetNotFoundError,
    RequestDataMismatchError,
    StorageError,
)
from data_pipeline.processing import ScalePrices
from data_pipeline.quality import FetchQuality, OmittedBar
from data_pipeline.schemas import OHLCV_SCHEMA


START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _frame(*, symbol="AAPL", count=7, start=START, step=timedelta(hours=1)):
    # The low-order Float64 bits and large exactly representable volumes matter:
    # deleting rows must never round or recompute any retained values.
    opens = [math.nextafter(100.0 + index, math.inf) for index in range(count)]
    return pl.DataFrame(
        {
            "timestamp": [start + step * index for index in range(count)],
            "symbol": [symbol] * count,
            "open": opens,
            "high": [value + 2 for value in opens],
            "low": [value - 2 for value in opens],
            "close": [math.nextafter(value, math.inf) for value in opens],
            "volume": [float(2**52 + index) for index in range(count)],
        },
        schema=OHLCV_SCHEMA,
    )


def _request(frame, *, provider="yahoo", timeframe="1h"):
    return DataRequest(
        symbol=frame["symbol"][0],
        start=frame["timestamp"][0],
        end=frame["timestamp"][-1] + timedelta(hours=1),
        provider=provider,
        timeframe=timeframe,
    )


def _query(start=START, end=START + timedelta(hours=7), **updates):
    return DataQuery(
        **dict(
            {"provider": "yahoo", "symbol": "AAPL", "timeframe": "1h",
             "start": start, "end": end},
            **updates,
        )
    )


def _delete(pipeline, query):
    plan = pipeline.plan_delete(query)
    return pipeline.delete(plan, confirm=plan.operation_id)


def _all_metadata(pipeline):
    return [
        item
        for layer in ("raw", "processed")
        for item in pipeline.list_datasets(DataQuery(layer=layer, include_history=True))
    ]


def _parquet_checksums(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*.parquet")}


def _assert_only_cataloged_files_remain(pipeline):
    expected = {item.relative_path: item.checksum_sha256 for item in _all_metadata(pipeline)}
    assert _parquet_checksums(pipeline.store.data_dir) == expected


def _assert_exact_values(actual, expected):
    assert_frame_equal(actual, expected, check_exact=True)


def test_delete_raw_preserves_processed_file_identity_values_and_provenance(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    result = pipeline.ingest_frame(_request(frame), frame, processors=[ScalePrices(2)])
    derived = result.processed
    derived_path = pipeline.store.data_dir / derived.relative_path
    derived_bytes = derived_path.read_bytes()
    derived_frame = pipeline.read_dataset(derived.dataset_id)

    report = _delete(pipeline, _query())

    assert report.removed_rows == frame.height
    assert set(report.deleted_ids) == {result.raw.dataset_id}
    assert not (pipeline.store.data_dir / result.raw.relative_path).exists()
    assert not pipeline.list_datasets(DataQuery(layer="raw", include_history=True))
    assert pipeline.get_metadata(derived.dataset_id) == derived
    assert derived_path.read_bytes() == derived_bytes
    assert derived.parent_ids == (result.raw.dataset_id,)
    _assert_exact_values(pipeline.read_dataset(derived.dataset_id), derived_frame)
    _assert_exact_values(pipeline.read(DataQuery(layer="processed", pipeline_id=derived.pipeline_id)),
                         derived_frame)
    with pytest.raises(DatasetNotFoundError):
        pipeline.read_dataset(result.raw.dataset_id)
    assert pipeline.store.audit() == [derived.dataset_id]
    assert pipeline.store.recover() == []
    _assert_only_cataloged_files_remain(pipeline)


def test_partial_processed_deletion_leaves_raw_and_other_pipeline_byte_identical(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    result = pipeline.ingest_frame(_request(frame), frame, processors=[ScalePrices(2)])
    other = pipeline.process(result.raw.dataset_id, processors=[ScalePrices(3)])
    untouched = (result.raw, other)
    before = {item.dataset_id: (pipeline.store.data_dir / item.relative_path).read_bytes()
              for item in untouched}
    before_frame = pipeline.read_dataset(result.processed.dataset_id)
    query = _query(START + timedelta(hours=2), START + timedelta(hours=5),
                   layer="processed", pipeline_id=result.processed.pipeline_id)

    report = _delete(pipeline, query)

    assert report.removed_rows == 3
    assert set(report.deleted_ids) == {result.processed.dataset_id}
    assert len(report.replacement_ids) == 2
    assert not (pipeline.store.data_dir / result.processed.relative_path).exists()
    for item in untouched:
        assert pipeline.get_metadata(item.dataset_id) == item
        assert (pipeline.store.data_dir / item.relative_path).read_bytes() == before[item.dataset_id]
    expected = pl.concat([before_frame.head(2), before_frame.tail(2)])
    _assert_exact_values(pipeline.read(DataQuery(layer="processed", pipeline_id=result.processed.pipeline_id)),
                         expected)
    replacements = [pipeline.get_metadata(key) for key in report.replacement_ids]
    assert all(item.active and item.parent_ids == (result.raw.dataset_id,) for item in replacements)
    assert all(item.processors_json == result.processed.processors_json for item in replacements)
    assert len(pipeline.store.audit()) == 4
    _assert_only_cataloged_files_remain(pipeline)


def test_full_processed_deletion_allows_reprocessing_without_touching_raw(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    result = pipeline.ingest_frame(_request(frame), frame, processors=[ScalePrices(2)])
    raw_bytes = (pipeline.store.data_dir / result.raw.relative_path).read_bytes()
    processed_before = pipeline.read_dataset(result.processed.dataset_id)

    report = _delete(pipeline, _query(layer="processed", pipeline_id=result.processed.pipeline_id))

    assert report.deleted_ids == (result.processed.dataset_id,)
    assert not report.replacement_ids
    assert (pipeline.store.data_dir / result.raw.relative_path).read_bytes() == raw_bytes
    assert pipeline.get_metadata(result.raw.dataset_id) == result.raw
    rebuilt = pipeline.process(result.raw.dataset_id, processors=[ScalePrices(2)])
    assert rebuilt.dataset_id != result.processed.dataset_id
    _assert_exact_values(pipeline.read_dataset(rebuilt.dataset_id), processed_before)
    _assert_only_cataloged_files_remain(pipeline)


def test_partial_raw_deletion_preserves_active_and_historical_values_and_status(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    original = _frame()
    old = pipeline.ingest_frame(_request(original), original).raw
    updated = original.with_columns((pl.col("volume") + 10).alias("volume"))
    current = pipeline.store.replace_raw(old.dataset_id, updated, confirm=old.dataset_id)
    old_path = pipeline.store.data_dir / old.relative_path
    current_path = pipeline.store.data_dir / current.relative_path
    query = _query(START + timedelta(hours=2), START + timedelta(hours=5))

    report = _delete(pipeline, query)

    assert report.removed_rows == 6  # Both actual copies of each selected bar are gone.
    assert set(report.deleted_ids) == {old.dataset_id, current.dataset_id}
    assert len(report.replacement_ids) == 4
    assert not old_path.exists()
    assert not current_path.exists()
    items = pipeline.list_datasets(DataQuery(include_history=True))
    assert len(items) == 4
    assert sum(item.active for item in items) == 2
    expected_current = pl.concat([updated.head(2), updated.tail(2)])
    _assert_exact_values(pipeline.read(DataQuery()), expected_current)
    expected_history = pl.concat([original.head(2), original.tail(2)])
    inactive_frames = [pipeline.read_dataset(item.dataset_id) for item in items if not item.active]
    _assert_exact_values(pl.concat(inactive_frames).sort("timestamp"), expected_history)
    for item in items:
        assert item.request.end <= query.start or item.request.start >= query.end
        assert (item.first_timestamp, item.last_timestamp) == (
            pipeline.read_dataset(item.dataset_id)["timestamp"].min(),
            pipeline.read_dataset(item.dataset_id)["timestamp"].max(),
        )
    for dataset_id in (old.dataset_id, current.dataset_id):
        with pytest.raises(DatasetNotFoundError):
            pipeline.get_metadata(dataset_id)
    assert set(pipeline.store.audit()) == set(report.replacement_ids)
    _assert_only_cataloged_files_remain(pipeline)


@pytest.mark.parametrize("offsets, retained", [
    ((timedelta(hours=2), timedelta(hours=4)), [0, 1, 4, 5, 6]),
    ((timedelta(hours=2, microseconds=1), timedelta(hours=4)), [0, 1, 2, 4, 5, 6]),
    ((timedelta(hours=2), timedelta(hours=4, microseconds=1)), [0, 1, 5, 6]),
    ((timedelta(hours=2, microseconds=1), timedelta(hours=3)), list(range(7))),
    ((timedelta(0), timedelta(microseconds=1)), [1, 2, 3, 4, 5, 6]),
])
def test_intraday_deletion_uses_exact_half_open_microsecond_bounds(tmp_path, offsets, retained):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    pipeline.ingest_frame(_request(frame), frame)

    report = _delete(pipeline, _query(START + offsets[0], START + offsets[1]))

    assert report.removed_rows == frame.height - len(retained)
    _assert_exact_values(pipeline.read(DataQuery()), frame[retained])
    _assert_only_cataloged_files_remain(pipeline)


def test_daily_deletion_uses_stored_session_labels_not_exchange_opening_times(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    first = datetime(2024, 1, 2, tzinfo=UTC)
    frame = _frame(count=4, start=first, step=timedelta(days=1))
    pipeline.ingest_frame(_request(frame, timeframe="1d"), frame)
    eastern = timezone(timedelta(hours=-5))
    # Midnight UTC labels for Jan 3 and Jan 4 expressed as local evening times.
    start = datetime(2024, 1, 2, 19, tzinfo=eastern)
    end = datetime(2024, 1, 4, 19, tzinfo=eastern)

    report = _delete(pipeline, _query(start, end, timeframe="1d"))

    assert report.removed_rows == 2
    _assert_exact_values(pipeline.read(DataQuery(timeframe="1d")), frame[[0, 3]])
    _assert_only_cataloged_files_remain(pipeline)


def test_deleted_middle_gap_can_be_ingested_again_without_overlap(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    pipeline.ingest_frame(_request(frame), frame)
    query = _query(START + timedelta(hours=2), START + timedelta(hours=5))
    _delete(pipeline, query)

    replacement = frame.slice(2, 3)
    restored = pipeline.ingest_frame(
        DataRequest("AAPL", query.start, query.end, timeframe="1h"), replacement,
    ).raw

    assert restored.row_count == 3
    _assert_exact_values(pipeline.read(DataQuery()), frame)
    assert len(pipeline.list_datasets(DataQuery())) == 3
    _assert_only_cataloged_files_remain(pipeline)


def test_other_symbols_providers_and_timeframes_remain_byte_identical(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    target = pipeline.ingest_frame(_request(frame), frame).raw
    others = [
        pipeline.ingest_frame(_request(_frame(symbol="MSFT")), _frame(symbol="MSFT")).raw,
        pipeline.ingest_frame(_request(frame, provider="other"), frame).raw,
        pipeline.ingest_frame(_request(frame, timeframe="30m"), frame).raw,
    ]
    before = {item.dataset_id: (pipeline.store.data_dir / item.relative_path).read_bytes()
              for item in others}

    report = _delete(pipeline, _query(symbol="aapl", timeframe="60m"))

    assert report.deleted_ids == (target.dataset_id,)
    for item in others:
        assert pipeline.get_metadata(item.dataset_id) == item
        assert (pipeline.store.data_dir / item.relative_path).read_bytes() == before[item.dataset_id]
    _assert_only_cataloged_files_remain(pipeline)


@pytest.mark.parametrize("field", ["symbol", "provider", "timeframe", "start", "end"])
def test_delete_query_requires_complete_identity_and_explicit_bounds(tmp_path, field):
    pipeline = DataPipeline(tmp_path / "data")
    with pytest.raises(RequestDataMismatchError):
        pipeline.plan_delete(replace(_query(), **{field: None}))


def test_processed_deletion_requires_a_specific_pipeline(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    with pytest.raises(RequestDataMismatchError):
        pipeline.plan_delete(_query(layer="processed"))


def test_confirmation_is_required_without_mutating_files_or_metadata(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    original = pipeline.ingest_frame(_request(frame), frame).raw
    plan = pipeline.plan_delete(_query())
    before = _parquet_checksums(pipeline.store.data_dir)

    with pytest.raises(ConfirmationRequiredError):
        pipeline.delete(plan, confirm="wrong-operation-id")

    assert pipeline.list_datasets() == [original]
    assert _parquet_checksums(pipeline.store.data_dir) == before
    _assert_exact_values(pipeline.read_dataset(original.dataset_id), frame)


def test_plan_is_stale_when_a_matching_revision_changes(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    old = pipeline.ingest_frame(_request(frame), frame).raw
    plan = pipeline.plan_delete(_query())
    current = pipeline.store.replace_raw(old.dataset_id, frame, confirm=old.dataset_id)
    before = _parquet_checksums(pipeline.store.data_dir)

    with pytest.raises(StorageError):
        pipeline.delete(plan, confirm=plan.operation_id)

    assert pipeline.list_datasets() == [current]
    assert len(pipeline.list_datasets(DataQuery(include_history=True))) == 2
    assert _parquet_checksums(pipeline.store.data_dir) == before


def test_plan_cannot_be_replayed_into_an_identical_copy_of_another_data_root(tmp_path):
    pipeline = DataPipeline(tmp_path / "source")
    frame = _frame()
    original = pipeline.ingest_frame(_request(frame), frame).raw
    plan = pipeline.plan_delete(_query())
    copied_root = tmp_path / "copy"
    shutil.copytree(pipeline.store.data_dir, copied_root)
    other = DataPipeline(copied_root)
    before = _parquet_checksums(copied_root)

    with pytest.raises(StorageError):
        other.delete(plan, confirm=plan.operation_id)

    assert other.list_datasets() == [original]
    assert _parquet_checksums(copied_root) == before
    _assert_exact_values(pipeline.read_dataset(original.dataset_id), frame)


def test_no_matching_rows_preserves_original_file_even_when_range_bounds_overlap(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()[[0, 6]]
    original = pipeline.ingest_frame(_request(frame), frame).raw
    before = _parquet_checksums(pipeline.store.data_dir)
    plan = pipeline.plan_delete(_query(START + timedelta(hours=2), START + timedelta(hours=5)))
    assert plan.removed_rows == 0

    report = pipeline.delete(plan, confirm=plan.operation_id)

    assert report.removed_rows == 0
    assert not report.deleted_ids
    assert not report.replacement_ids
    assert pipeline.get_metadata(original.dataset_id) == original
    assert _parquet_checksums(pipeline.store.data_dir) == before
    _assert_only_cataloged_files_remain(pipeline)


def test_confirmed_operation_is_idempotent_without_deleting_new_downloads(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    pipeline.ingest_frame(_request(frame), frame)
    plan = pipeline.plan_delete(_query())
    original_report = pipeline.delete(plan, confirm=plan.operation_id)
    new = pipeline.ingest_frame(_request(frame), frame).raw
    before = _parquet_checksums(pipeline.store.data_dir)

    replayed = pipeline.delete(plan, confirm=plan.operation_id)

    assert replayed == original_report
    assert pipeline.get_metadata(new.dataset_id) == new
    assert _parquet_checksums(pipeline.store.data_dir) == before
    _assert_exact_values(pipeline.read_dataset(new.dataset_id), frame)


def test_pending_original_cleanup_blocks_a_new_deletion_from_claiming_completion(tmp_path, monkeypatch):
    from data_pipeline.storage import deletion

    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    original = pipeline.ingest_frame(_request(frame), frame).raw
    plan = pipeline.plan_delete(_query(START + timedelta(hours=2), START + timedelta(hours=5)))
    earlier_broad_plan = pipeline.plan_delete(_query())

    def unavailable(path):
        raise PermissionError("simulated unlink failure")

    with monkeypatch.context() as patch:
        patch.setattr(deletion, "_remove_file", unavailable)
        with pytest.raises(deletion.DeletionCleanupError):
            pipeline.delete(plan, confirm=plan.operation_id)
    # Catalog survivors now exist, but the former original still physically
    # contains every row. A new broader operation cannot truthfully claim that
    # deleting only the survivors physically erased that wider interval.
    assert (pipeline.store.data_dir / original.relative_path).exists()
    assert pipeline.deletion_status(plan.operation_id).status == "pending"
    with pytest.raises(StorageError):
        pipeline.plan_delete(_query())
    with pytest.raises(StorageError):
        pipeline.delete(earlier_broad_plan, confirm=earlier_broad_plan.operation_id)

    pipeline.store.recover()
    assert pipeline.deletion_status(plan.operation_id).status == "completed"
    report = _delete(pipeline, _query())
    assert report.removed_rows == 4
    assert not list(pipeline.store.data_dir.rglob("*.parquet"))


def test_publication_collision_never_unlinks_a_file_the_operation_did_not_create(tmp_path, monkeypatch):
    from data_pipeline.storage import deletion

    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    original = pipeline.ingest_frame(_request(frame), frame).raw
    original_bytes = (pipeline.store.data_dir / original.relative_path).read_bytes()
    plan = pipeline.plan_delete(_query(START + timedelta(hours=2), START + timedelta(hours=5)))
    foreign_paths = []
    foreign_bytes = b"unrelated file appeared immediately before exclusive publication"

    def collide(staged, final):
        final.write_bytes(foreign_bytes)
        foreign_paths.append(final)
        raise FileExistsError("simulated concurrent publication collision")

    monkeypatch.setattr(deletion, "_publish_survivor", collide)
    with pytest.raises(StorageError):
        pipeline.delete(plan, confirm=plan.operation_id)

    assert len(foreign_paths) == 1
    assert foreign_paths[0].read_bytes() == foreign_bytes
    assert pipeline.get_metadata(original.dataset_id) == original
    assert (pipeline.store.data_dir / original.relative_path).read_bytes() == original_bytes
    _assert_exact_values(pipeline.read_dataset(original.dataset_id), frame)
    assert not list((pipeline.store.data_dir / "staging").rglob("*.parquet"))


def test_scan_snapshot_survives_physical_deletion_without_retaining_disk_files(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    stored = pipeline.ingest_frame(_request(frame), frame).raw
    snapshot = pipeline.scan(_query(), columns=["timestamp", "close"])

    _delete(pipeline, _query())

    assert not (pipeline.store.data_dir / stored.relative_path).exists()
    assert not list(pipeline.store.data_dir.rglob("*.parquet"))
    _assert_exact_values(snapshot.collect(), frame.select("timestamp", "close"))
    assert pipeline.scan(_query()).collect().is_empty()


def test_quality_and_request_provenance_survive_partial_deletion(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    frame = _frame()
    quality = FetchQuality(
        status="reported", skip_missing_ohlc=True, provider_version="offline-fixture",
        omitted_bars=(OmittedBar(START - timedelta(hours=1), "all_ohlc_missing"),),
    )
    original = pipeline.store.write_raw(_request(frame), frame, quality=quality)
    query = _query(START + timedelta(hours=2), START + timedelta(hours=5))

    report = _delete(pipeline, query)

    replacements = [pipeline.get_metadata(key) for key in report.replacement_ids]
    assert len(replacements) == 2
    assert all(item.quality == quality for item in replacements)
    assert all(item.request.provider == original.request.provider for item in replacements)
    assert all(item.request.timeframe == original.request.timeframe for item in replacements)
    assert all(item.request.end <= query.start or item.request.start >= query.end for item in replacements)
    restarted = DataPipeline(pipeline.store.data_dir)
    assert restarted.deletion_status(report.operation_id) == report
    assert set(restarted.store.audit()) == set(report.replacement_ids)
    _assert_only_cataloged_files_remain(restarted)
