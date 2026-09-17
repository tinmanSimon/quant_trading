"""Versioned, checksummed research artifacts, separate from market data."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import polars as pl

from .backtesting import BacktestResult, ExecutionSettings, compare_results
from .errors import ResearchError


@dataclass(frozen=True)
class ResearchRun:
    run_id: str
    created_at: str
    results: tuple[BacktestResult, ...]
    comparison: pl.DataFrame
    manifest: dict
    path: Path


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_path(root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ResearchError("Invalid run ID.")
    path = root.resolve() / run_id
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ResearchError("Run path escapes the results directory.")
    return path


def _sync(path: Path):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def save_run(root: Path, results: list[BacktestResult], manifest: dict) -> ResearchRun:
    if not results:
        raise ResearchError("Cannot save an empty backtest run.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    final = _run_path(root, run_id)
    staging = root / (".staging-" + run_id)
    created_at = datetime.now(UTC).isoformat()
    document = json.loads(_json(manifest))
    document.update({"format_version": 1, "run_id": run_id, "created_at": created_at})
    entries = []
    staging.mkdir()
    try:
        for index, result in enumerate(results):
            entry = {"symbol": result.symbol, "strategy_spec": result.strategy_spec,
                     "settings": result.settings.to_dict(), "metrics": result.metrics,
                     "metadata": result.metadata, "files": {}}
            for name in ("equity", "trades", "orders"):
                path = staging / f"{index}-{name}.parquet"
                frame = getattr(result, name)
                frame.write_parquet(path)
                if not pl.read_parquet(path).equals(frame):
                    raise ResearchError("Research artifact changed during Parquet round trip.")
                _sync(path)
                entry["files"][name] = {"path": path.name, "sha256": _checksum(path)}
            entries.append(entry)
        document["results"] = entries
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(_json(document), encoding="utf-8")
        _sync(manifest_path)
        checksum_path = staging / "manifest.sha256"
        checksum_path.write_text(_checksum(manifest_path), encoding="ascii")
        _sync(checksum_path)
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        staging.rename(final)
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        # Only our uncommitted, uniquely named directory can be removed.
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ResearchRun(run_id, created_at, tuple(results), compare_results(results), document, final)


def _manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    try:
        if manifest_path.is_symlink() or (path / "manifest.sha256").is_symlink():
            raise ResearchError("Manifest must not be a symbolic link.")
        if _checksum(manifest_path) != (path / "manifest.sha256").read_text(encoding="ascii"):
            raise ResearchError(f"Research manifest checksum failed: {path.name}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["format_version"] != 1 or manifest["run_id"] != path.name:
            raise ResearchError("Unsupported or mismatched research manifest.")
        return manifest
    except (OSError, KeyError, ValueError) as exc:
        raise ResearchError(f"Cannot read research run {path.name}: {exc}") from exc


def load_run(root: Path, run_id: str) -> ResearchRun:
    path = _run_path(root, run_id)
    document = _manifest(path)
    results = []
    for entry in document["results"]:
        frames = {}
        for name in ("equity", "trades", "orders"):
            item = entry["files"][name]
            artifact = path / item["path"]
            if artifact.is_symlink() or artifact.resolve().parent != path.resolve():
                raise ResearchError("Artifact path escapes its run directory.")
            if not artifact.is_file():
                raise ResearchError(f"Missing research artifact: {artifact.name}.")
            if _checksum(artifact) != item["sha256"]:
                raise ResearchError(f"Research artifact checksum failed: {artifact.name}.")
            frames[name] = pl.read_parquet(artifact)
        settings = ExecutionSettings(**{key: Decimal(value) for key, value in entry["settings"].items()})
        results.append(BacktestResult(symbol=entry["symbol"], strategy_spec=entry["strategy_spec"],
                                      settings=settings, metrics=entry["metrics"],
                                      metadata=entry["metadata"], **frames))
    return ResearchRun(run_id, document["created_at"], tuple(results), compare_results(results), document, path)


def list_runs(root: Path) -> list[dict]:
    if not root.exists():
        return []
    summaries = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", path.name):
            continue
        document = _manifest(_run_path(root, path.name))
        summaries.append({key: document[key] for key in
                          ("run_id", "created_at", "tickers", "timeframe", "start", "end")}
                         | {"strategy_names": [item["strategy_spec"]["name"] for item in document["results"]]})
    return sorted(summaries, key=lambda item: item["created_at"], reverse=True)


def comparison_identity(manifest: dict) -> dict:
    """Compare only identical market revisions, intervals and account rules."""
    keys = ("tickers", "provider", "timeframe", "layer", "pipeline_id", "start", "end",
            "instruments", "calendar_version", "price_basis", "return_basis", "account_mode",
            "execution_settings", "engine_sha256")
    result = {key: manifest[key] for key in keys}
    result["lookback"] = manifest.get("lookback")
    result["packages"] = manifest.get("packages")
    result["sources"] = {symbol: sorted((item["dataset_id"], item["checksum_sha256"]) for item in items)
                         for symbol, items in manifest["sources"].items()}
    return result
