"""Shared, deterministic fixtures for data-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_data_dir(tmp_path: Path) -> Path:
    """Return a per-test data directory that is never the project's real data/ root."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_ohlcv_records() -> list[dict[str, object]]:
    """Load a small, stable OHLCV fixture without contacting a vendor."""
    fixture_path = Path(__file__).parent / "fixtures" / "ohlcv_1h.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))
