"""Independent acceptance checks for strict calendar-based data readiness."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest, FetchQuality, OmittedBar
from research import datasets
from research.datasets import prepare_snapshot
from research.errors import PreflightError, ResearchError
from research.instruments import ExpectedBar, Instrument, expected_bars


def stamp(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def frame_for(bars):
    return pl.DataFrame({
        "timestamp": [bar.timestamp for bar in bars], "symbol": ["AAPL"] * len(bars),
        "open": [100.] * len(bars), "high": [102.] * len(bars),
        "low": [99.] * len(bars), "close": [101.] * len(bars),
        "volume": [1000.] * len(bars),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


@pytest.mark.parametrize("day,open_hour,close_hour", [
    ((2025, 7, 3), 13, 17), ((2025, 11, 28), 14, 18), ((2025, 12, 24), 14, 18),
])
def test_every_early_close_includes_final_half_hour(day, open_hour, close_hour):
    start = stamp(*day)
    bars = expected_bars(Instrument("AAPL"), start=start, end=start + timedelta(days=1), timeframe="1h")
    assert len(bars) == 4
    assert bars[0].timestamp == start.replace(hour=open_hour, minute=30)
    assert bars[-1].close == start.replace(hour=close_hour)
    assert bars[-1].close - bars[-1].open == timedelta(minutes=30)


def test_hourly_warmup_across_dst_uses_last_previous_session_bar():
    bars = expected_bars(Instrument("AAPL"), start=stamp(2025, 3, 10),
                         end=stamp(2025, 3, 11), timeframe="1h", lookback=1)
    assert bars[0].timestamp == stamp(2025, 3, 7, 20, 30)
    assert bars[0].close == stamp(2025, 3, 7, 21)
    assert bars[1].open == stamp(2025, 3, 10, 13, 30)
    assert bars[-1].close == stamp(2025, 3, 10, 20)


def test_daily_labels_and_actual_session_times_remain_distinct():
    bars = expected_bars(Instrument("AAPL"), start=stamp(2025, 12, 24),
                         end=stamp(2025, 12, 25), timeframe="1d", lookback=1)
    assert bars[-1].timestamp == stamp(2025, 12, 24)
    assert bars[-1].open == stamp(2025, 12, 24, 14, 30)
    assert bars[-1].close == stamp(2025, 12, 24, 18)


def test_missing_warmup_aborts_even_with_complete_requested_sessions(tmp_path):
    start, end = stamp(2025, 7, 3), stamp(2025, 7, 4)
    bars = expected_bars(Instrument("AAPL"), start=start, end=end, timeframe="1h", lookback=2)
    pipeline = DataPipeline(tmp_path)
    pipeline.ingest_frame(DataRequest("AAPL", bars[0].timestamp, end), frame_for(bars[1:]))
    with pytest.raises(PreflightError) as error:
        prepare_snapshot(pipeline, tickers=["AAPL"], start=start, end=end, timeframe="1h", lookback=2)
    missing = next(issue for issue in error.value.report.issues if issue.code == "missing_bars")
    assert missing.timestamps == (bars[0].timestamp,)


def test_unknown_quality_is_explicit_but_complete_actual_data_can_pass(tmp_path):
    start, end = stamp(2025, 7, 3), stamp(2025, 7, 4)
    bars = expected_bars(Instrument("AAPL"), start=start, end=end, timeframe="1h", lookback=2)
    pipeline = DataPipeline(tmp_path)
    pipeline.ingest_frame(DataRequest("AAPL", bars[0].timestamp, end), frame_for(bars))
    snapshot = prepare_snapshot(pipeline, tickers=["AAPL"], start=start, end=end, timeframe="1h", lookback=2)
    assert snapshot.report.ok
    assert any("unknown" in note for note in snapshot.report.notes)
    assert snapshot.frames["AAPL"].height == len(bars)
    assert snapshot.manifest["sources"]["AAPL"][0]["quality"]["status"] == "unknown"
    assert snapshot.manifest["layer"] == "raw"
    assert "pipeline_id" not in snapshot.manifest


def test_interval_ending_inside_final_bar_fails_before_simulation(tmp_path):
    start = stamp(2025, 7, 3)
    end = stamp(2025, 7, 3, 16, 45)
    bars = expected_bars(Instrument("AAPL"), start=start, end=end, timeframe="1h", lookback=1)
    pipeline = DataPipeline(tmp_path)
    pipeline.ingest_frame(DataRequest("AAPL", bars[0].timestamp, end), frame_for(bars))
    with pytest.raises(PreflightError) as error:
        prepare_snapshot(pipeline, tickers=["AAPL"], start=start, end=end, timeframe="1h", lookback=1)
    incomplete = next(issue for issue in error.value.report.issues if issue.code == "unfinished_bars")
    assert incomplete.timestamps == (stamp(2025, 7, 3, 16, 30),)


def test_recorded_omission_outside_trading_and_warmup_window_does_not_block(tmp_path):
    source_start, start, end = stamp(2025, 7, 1), stamp(2025, 7, 3), stamp(2025, 7, 4)
    bars = expected_bars(Instrument("AAPL"), start=source_start, end=end, timeframe="1h")
    quality = FetchQuality("reported", True, (OmittedBar(bars[0].timestamp, "all_ohlc_missing"),))
    pipeline = DataPipeline(tmp_path)
    pipeline.store.write_raw(DataRequest("AAPL", source_start, end), frame_for(bars[1:]), quality=quality)
    snapshot = prepare_snapshot(pipeline, tickers=["AAPL"], start=start, end=end, timeframe="1h", lookback=1)
    assert snapshot.report.ok
    assert snapshot.frames["AAPL"].height == 5


def test_unverified_processed_history_is_rejected_before_data_selection(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    with pytest.raises(ResearchError, match="causality"):
        prepare_snapshot(pipeline, tickers=["AAPL"], start=stamp(2025, 7, 3),
                         end=stamp(2025, 7, 4), timeframe="1h", lookback=1,
                         layer="processed")
    assert not pipeline.store.catalog_path.exists()


def test_overnight_session_open_outside_requested_bounds_aborts_in_preflight(tmp_path, monkeypatch):
    start, end = stamp(2025, 1, 2), stamp(2025, 1, 3)
    bar = ExpectedBar(start, start - timedelta(hours=1), start + timedelta(hours=22))
    monkeypatch.setattr(datasets, "expected_bars", lambda *args, **kwargs: (bar,))
    pipeline = DataPipeline(tmp_path)
    pipeline.ingest_frame(DataRequest("AAPL", start, end, timeframe="1d"), frame_for([bar]))
    with pytest.raises(PreflightError) as error:
        prepare_snapshot(pipeline, tickers=["AAPL"], start=start, end=end, timeframe="1d", lookback=0)
    issue = next(issue for issue in error.value.report.issues if issue.code == "out_of_range_opens")
    assert issue.timestamps == (start,)
