"""Optional late histories keep warm-up and interior coverage requirements."""

from datetime import UTC, datetime

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest
from data_pipeline.processing import ScalePrices
from research import datasets
from research.datasets import prepare_snapshot
from research.errors import PreflightError, ResearchError
from research.instruments import Instrument, expected_bars


def day(number):
    return datetime(2024, 1, number, tzinfo=UTC)


def ingest(pipeline, symbol="AAPL", days=(3, 4, 5, 8, 9, 10, 11)):
    size = len(days)
    frame = pl.DataFrame({
        "timestamp": [day(number) for number in days], "symbol": [symbol] * size,
        "open": [100.] * size, "high": [102.] * size, "low": [99.] * size,
        "close": [101.] * size, "volume": [1000.] * size,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    return pipeline.ingest_frame(
        DataRequest(symbol, day(min(days)), day(12), timeframe="1d"), frame,
    ).raw


def snapshot(pipeline, tickers=("AAPL",), **kwargs):
    return prepare_snapshot(pipeline, tickers=tickers, start=day(5), end=day(12),
                            timeframe="1d", lookback=2, **kwargs)


@pytest.fixture
def pipeline(tmp_path):
    return DataPipeline(tmp_path / "data")


def test_missing_initial_bars_remain_strict_by_default(pipeline):
    ingest(pipeline, days=(8, 9, 10, 11))
    with pytest.raises(PreflightError) as caught:
        snapshot(pipeline)
    issue = next(issue for issue in caught.value.report.issues if issue.code == "missing_bars")
    assert issue.timestamps == (day(3), day(4), day(5))


def test_late_ticker_joins_only_after_completed_actual_warmup(pipeline):
    ingest(pipeline)
    ingest(pipeline, "NEW", days=(8, 9, 10, 11))
    selected = snapshot(pipeline, ("AAPL", "NEW"), skip_initial_missing_bars=True)
    assert selected.report.ok
    assert selected.effective_starts == {"AAPL": day(5), "NEW": day(10)}
    assert selected.frames["NEW"]["timestamp"].to_list() == [day(8), day(9), day(10), day(11)]
    assert [bar.timestamp for bar in selected.bars["NEW"]] == [day(8), day(9), day(10), day(11)]
    assert selected.manifest["effective_starts"]["NEW"] == day(10).isoformat()
    assert selected.manifest["requested_tickers"] == ["AAPL", "NEW"]
    assert any("NEW: trading starts" in note for note in selected.report.notes)


@pytest.mark.parametrize("days,missing", [
    ((8, 10, 11), (9,)),  # Interior hole after a leading gap.
    ((8, 9, 10), (11,)),  # A trailing omission remains an error.
    ((3, 5, 8, 9, 10, 11), (4,)),  # Missing warm-up within available history.
])
def test_option_does_not_hide_interior_trailing_or_warmup_holes(pipeline, days, missing):
    ingest(pipeline, days=days)
    with pytest.raises(PreflightError) as caught:
        snapshot(pipeline, skip_initial_missing_bars=True)
    issue = next(issue for issue in caught.value.report.issues if issue.code == "missing_bars")
    assert issue.timestamps == tuple(day(number) for number in missing)


def test_short_history_does_not_hide_a_trailing_gap(pipeline):
    ingest(pipeline, days=(10,))
    with pytest.raises(PreflightError) as caught:
        snapshot(pipeline, skip_initial_missing_bars=True)
    issue = next(issue for issue in caught.value.report.issues if issue.code == "missing_bars")
    assert issue.timestamps == (day(11),)


def test_no_data_and_insufficient_history_are_explicitly_skipped(pipeline):
    ingest(pipeline)
    ingest(pipeline, "NEW", days=(10, 11))
    selected = snapshot(pipeline, ("AAPL", "NEW", "ABSENT"), skip_initial_missing_bars=True)
    assert list(selected.frames) == ["AAPL"]
    assert list(selected.bars) == ["AAPL"]
    assert selected.manifest["tickers"] == ["AAPL"]
    assert list(selected.manifest["sources"]) == ["AAPL"]
    assert set(selected.manifest["skipped_tickers"]) == {"NEW", "ABSENT"}
    assert selected.manifest["requested_tickers"] == ["AAPL", "NEW", "ABSENT"]
    assert "warm-up" in selected.manifest["skipped_tickers"]["NEW"]
    assert "No local data" in selected.manifest["skipped_tickers"]["ABSENT"]


def test_all_skipped_tickers_raise_a_clear_error(pipeline):
    ingest(pipeline, days=(10, 11))
    with pytest.raises(ResearchError, match="No tickers have enough local history"):
        snapshot(pipeline, skip_initial_missing_bars=True)


def test_off_grid_bar_still_aborts(pipeline):
    ingest(pipeline, days=(6, 8, 9, 10, 11))
    with pytest.raises(PreflightError) as caught:
        snapshot(pipeline, skip_initial_missing_bars=True)
    issue = next(issue for issue in caught.value.report.issues if issue.code == "unexpected_bars")
    assert issue.timestamps == (day(6),)


def test_complete_history_keeps_requested_start_and_calendar(pipeline):
    ingest(pipeline)
    selected = snapshot(pipeline)
    assert selected.effective_starts == {"AAPL": day(5)}
    assert selected.bars["AAPL"] == expected_bars(
        Instrument("AAPL"), start=day(5), end=day(12), timeframe="1d", lookback=2,
    )
    assert "skip_initial_missing_bars" not in selected.manifest


def test_late_processed_data_keeps_exact_parent_lineage(pipeline):
    raw = ingest(pipeline, days=(8, 9, 10, 11))
    processed = pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    selected = snapshot(pipeline, layer="processed", pipeline_id=processed.pipeline_id,
                        skip_initial_missing_bars=True)
    assert selected.effective_starts["AAPL"] == day(10)
    assert selected.manifest["parent_sources"]["AAPL"][0]["dataset_id"] == raw.dataset_id
    assert selected.manifest["sources"]["AAPL"][0]["dataset_id"] == processed.dataset_id
    (pipeline.store.data_dir / raw.relative_path).write_bytes(b"corrupted raw source")
    with pytest.raises(PreflightError, match="Checksum mismatch"):
        snapshot(pipeline, layer="processed", pipeline_id=processed.pipeline_id,
                 skip_initial_missing_bars=True)


def test_option_requires_a_boolean(pipeline):
    with pytest.raises(ResearchError, match="must be a boolean"):
        snapshot(pipeline, skip_initial_missing_bars="yes")


def test_calendar_schedule_is_built_once_per_exchange_and_snapshot(pipeline, monkeypatch):
    ingest(pipeline, "NEW", days=(8, 9, 10, 11))
    for symbol in ("AAPL", "MSFT", "LONDON"):
        ingest(pipeline, symbol)
    calls = []

    def tracked(instrument, **kwargs):
        calls.append(instrument.calendar)
        return expected_bars(instrument, **kwargs)

    monkeypatch.setattr(datasets, "expected_bars", tracked)
    options = dict(skip_initial_missing_bars=True, instruments={"LONDON": Instrument("LONDON", "XLON")})
    selected = snapshot(pipeline, ("NEW", "AAPL", "MSFT", "LONDON"), **options)
    assert calls == ["XNYS", "XLON"]
    assert len(selected.bars["NEW"]) == 4
    assert len(selected.bars["AAPL"]) == 7  # Trimming NEW cannot alter the cached schedule.
    assert selected.bars["AAPL"] is selected.bars["MSFT"]
    snapshot(pipeline, ("NEW", "AAPL", "MSFT", "LONDON"), **options)
    assert calls == ["XNYS", "XLON", "XNYS", "XLON"]
