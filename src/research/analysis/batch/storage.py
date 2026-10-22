"""Atomic JSON manifests and separately loaded, checksummed ticker results."""

from datetime import UTC, datetime
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

from ..core.base import json_copy


def _read_json(path, *, compressed=False):
    if path.is_symlink():
        raise ValueError("Batch result files must not be symbolic links.")
    payload = path.read_bytes()
    if compressed:
        payload = gzip.decompress(payload)
    value = json.loads(payload)
    return json_copy(value)


def _write_json(path, value, *, compressed=False):
    # Validate before touching the saved file; os.replace publishes one complete
    # version to polling readers, even if a run is interrupted during a write.
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if compressed:
        payload = gzip.compress(payload, mtime=0)
    if path.is_symlink():
        raise ValueError("Batch result files must not be symbolic links.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256(payload).hexdigest()


class BatchStore:
    """A result directory independent of market-data files and workspace snapshots.

    Manifests contain only settings, provenance, status, and compact metrics.
    Full result tables are opened only by ``load_ticker``. Aggregation settings
    live separately so editing them cannot overwrite a worker's progress.
    """

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()

    def _path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid batch run ID.")
        path = self.directory / run_id
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError("Batch run path escapes the results directory.")
        return path

    def create(self, manifest):
        document = json_copy(manifest)
        run_id = uuid4().hex
        now = datetime.now(UTC).isoformat()
        document.update(version=1, run_id=run_id, created_at=now, updated_at=now)
        path = self._path(run_id)
        path.mkdir(parents=True)
        self.save_manifest(document)
        return self.load_manifest(run_id)

    def save_manifest(self, manifest):
        document = json_copy(manifest)
        document.pop("aggregation", None)
        document["updated_at"] = datetime.now(UTC).isoformat()
        _write_json(self._path(document["run_id"]) / "manifest.json", document)
        return document

    def claim_run(self, run_id):
        """Claim execution once, including across threads and dashboard processes."""
        try:
            with (self._path(run_id) / "execution.claim").open("x") as handle:
                handle.write(datetime.now(UTC).isoformat())
        except FileExistsError:
            raise ValueError("This batch has already been started.") from None

    def load_manifest(self, run_id):
        path = self._path(run_id)
        document = _read_json(path / "manifest.json")
        if document.get("version") != 1 or document.get("run_id") != run_id:
            raise ValueError("Unsupported or mismatched batch manifest.")
        settings = path / "aggregation.json"
        document["aggregation"] = _read_json(settings) if settings.exists() else {}
        return document

    def list_runs(self):
        if not self.directory.exists():
            return []
        manifests = []
        for path in self.directory.iterdir():
            if not re.fullmatch(r"[a-f0-9]{32}", path.name):
                continue
            try:
                manifests.append(self.load_manifest(path.name))
            except (OSError, ValueError, TypeError):
                # An interrupted first write or a damaged run does not prevent
                # opening the other independently saved runs.
                continue
        return sorted(manifests, key=lambda item: item["created_at"], reverse=True)

    def save_ticker(self, run_id, ticker, payload):
        if payload.get("ticker") != ticker:
            raise ValueError("Ticker result identity does not match its destination.")
        name = "ticker-" + sha256(ticker.encode()).hexdigest() + ".json.gz"
        checksum = _write_json(self._path(run_id) / name, payload, compressed=True)
        return {"detail_file": name, "detail_sha256": checksum}

    def load_ticker(self, run_id, ticker):
        manifest = self.load_manifest(run_id)
        entry = manifest["tickers"][ticker]
        name = "ticker-" + sha256(ticker.encode()).hexdigest() + ".json.gz"
        if entry.get("detail_file") != name:
            raise ValueError("This ticker has no saved analysis result.")
        path = self._path(run_id) / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != entry.get("detail_sha256"):
            raise ValueError("Batch ticker result checksum failed.")
        payload = _read_json(path, compressed=True)
        if payload.get("ticker") != ticker:
            raise ValueError("Batch ticker result identity does not match its manifest.")
        return payload

    def save_aggregation(self, run_id, settings):
        self.load_manifest(run_id)
        if not isinstance(settings, dict):
            raise ValueError("Aggregation settings must be an object.")
        _write_json(self._path(run_id) / "aggregation.json", settings)

    def delete_run(self, run_id):
        """Remove only the selected run, including details and summary settings."""
        shutil.rmtree(self._path(run_id))

    def copy_run(self, source, run_id):
        """Publish a finished temporary run atomically into saved history."""
        source.load_manifest(run_id)
        destination = self._path(run_id)
        if destination.exists():
            raise ValueError("This analysis is already saved.")
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.directory, prefix=".saving-") as temporary:
            staged = Path(temporary) / run_id
            shutil.copytree(source._path(run_id), staged)
            os.rename(staged, destination)
