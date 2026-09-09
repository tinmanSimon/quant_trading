"""Smoke tests for the test environment and package import boundary."""

from __future__ import annotations

from pathlib import Path

from data_pipeline.base_fetcher import BaseDataProvider, OHLCV_SCHEMA
from data_pipeline.yahoo_fetcher import YFinanceProvider


def test_current_provider_implementation_imports() -> None:
    """Existing providers must load without making a network request."""
    assert issubclass(YFinanceProvider, BaseDataProvider)
    assert set(OHLCV_SCHEMA) == {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }


def test_isolated_data_directory_stays_inside_pytest_temp_area(
    isolated_data_dir: Path, tmp_path: Path
) -> None:
    """Future storage tests must not write to the repository's data directory."""
    marker = isolated_data_dir / "test-only.txt"
    marker.write_text("safe", encoding="utf-8")

    assert isolated_data_dir == tmp_path / "data"
    assert marker.read_text(encoding="utf-8") == "safe"
    assert isolated_data_dir != Path.cwd() / "data"


def test_sample_ohlcv_fixture_is_stable(
    sample_ohlcv_records: list[dict[str, object]],
) -> None:
    """The shared fixture provides deterministic input for later pipeline tests."""
    assert len(sample_ohlcv_records) == 3
    assert sample_ohlcv_records[0]["symbol"] == "AAPL"
    assert sample_ohlcv_records[0]["timestamp"] == "2024-01-02T14:30:00Z"
