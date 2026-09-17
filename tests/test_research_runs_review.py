"""Persisted simulations retain exact ledgers and detect damaged artifacts."""

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json

import polars as pl
import pytest

from research import Research
from research.backtesting import ExecutionSettings, run_backtest
from research.errors import ResearchError
from research.runs import comparison_identity, list_runs, load_run, save_run
from research.strategies import Momentum


def result():
    base = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    frame = pl.DataFrame({
        "timestamp": [base + timedelta(hours=i) for i in range(4)],
        "symbol": ["AAPL"] * 4,
        "open": [10.0, 11.0, 12.0, 13.0],
        "high": [10.0, 11.0, 12.0, 13.0],
        "low": [10.0, 11.0, 12.0, 13.0],
        "close": [10.0, 11.0, 12.0, 13.0],
        "volume": [100.0] * 4,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    return run_backtest(frame, Momentum(1), start=base + timedelta(hours=2), end=base + timedelta(hours=4), settings=ExecutionSettings(initial_cash=100, commission_fixed=0.07))


def manifest():
    return {
        "tickers": ["AAPL"], "provider": "yahoo", "timeframe": "1h", "layer": "raw", "pipeline_id": None,
        "start": "2025-01-02T16:30:00+00:00", "end": "2025-01-02T18:30:00+00:00",
        "instruments": {"AAPL": {"calendar": "XNYS", "currency": "USD"}},
        "calendar_version": "test", "price_basis": "vendor_unadjusted", "return_basis": "price_return_no_dividend_credit",
        "account_mode": "independent_long_only_usd_accounts", "execution_settings": result().settings.to_dict(),
        "engine_sha256": "a" * 64,
        "sources": {"AAPL": [{"dataset_id": "b" * 32, "checksum_sha256": "c" * 64}]},
    }


def test_run_roundtrip_preserves_exact_ledger_and_manifest(tmp_path):
    original = result()
    saved = save_run(tmp_path, [original], manifest())
    restored = load_run(tmp_path, saved.run_id)
    actual = restored.results[0]
    assert original.equity.equals(actual.equity)
    assert original.trades.equals(actual.trades)
    assert original.orders.equals(actual.orders)
    assert original.settings == actual.settings
    assert original.metrics == actual.metrics
    assert original.metadata == actual.metadata
    assert original.strategy_spec == actual.strategy_spec
    assert saved.comparison.equals(restored.comparison)
    assert comparison_identity(saved.manifest) == comparison_identity(restored.manifest)
    assert list_runs(tmp_path)[0]["run_id"] == saved.run_id


@pytest.mark.parametrize("name", ["equity", "trades", "orders"])
def test_corrupted_parquet_is_not_loaded(tmp_path, name):
    saved = save_run(tmp_path, [result()], manifest())
    artifact = saved.path / saved.manifest["results"][0]["files"][name]["path"]
    with artifact.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ResearchError, match="checksum"):
        load_run(tmp_path, saved.run_id)


def test_corrupted_manifest_is_not_loaded(tmp_path):
    saved = save_run(tmp_path, [result()], manifest())
    path = saved.path / "manifest.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ResearchError, match="checksum"):
        load_run(tmp_path, saved.run_id)


def test_artifact_traversal_rejected_even_with_updated_manifest_hash(tmp_path):
    saved = save_run(tmp_path, [result()], manifest())
    doc = saved.manifest
    doc["results"][0]["files"]["equity"]["path"] = "../outside.parquet"
    path = saved.path / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    (saved.path / "manifest.sha256").write_text(sha256(path.read_bytes()).hexdigest(), encoding="ascii")
    with pytest.raises(ResearchError, match="escapes"):
        load_run(tmp_path, saved.run_id)


@pytest.mark.parametrize("run_id", ["../data", "invalid", "/tmp/outside", "A" * 32])
def test_unsafe_run_identifier_rejected(tmp_path, run_id):
    with pytest.raises(ResearchError, match="run ID"):
        load_run(tmp_path, run_id)


def test_write_failure_leaves_no_visible_run(tmp_path, monkeypatch):
    original = pl.DataFrame.write_parquet
    calls = 0

    def fail_second(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_second)
    with pytest.raises(OSError, match="disk full"):
        save_run(tmp_path, [result()], manifest())
    assert list(tmp_path.iterdir()) == []


def test_comparison_rejects_different_data_or_execution(tmp_path):
    root = tmp_path / "runs"
    original = save_run(root, [result()], manifest())
    same = save_run(root, [result()], manifest())
    different = manifest()
    different["sources"]["AAPL"][0]["checksum_sha256"] = "d" * 64
    changed = save_run(root, [result()], different)
    research = Research(tmp_path / "data", root)
    comparison = research.compare_runs([original.run_id, same.run_id])
    assert comparison.height == 2
    with pytest.raises(ResearchError, match="different data"):
        research.compare_runs([original.run_id, changed.run_id])


def test_symlink_run_directory_rejected(tmp_path):
    actual = tmp_path / "actual"
    saved = save_run(actual, [result()], manifest())
    other = tmp_path / "other"
    other.mkdir()
    (other / saved.run_id).symlink_to(saved.path, target_is_directory=True)
    with pytest.raises(ResearchError, match="escapes"):
        load_run(other, saved.run_id)
