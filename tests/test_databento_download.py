"""Offline transport tests: parallel transfers, byte resume, and safe job reuse."""

from collections import Counter
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from data_pipeline.exceptions import ProviderError
from data_pipeline.providers import DatabentoProvider


JOB = "XNAS-existing-job"
KEY = "offline-secret-key"


class Response:
    def __init__(self, content=b"", *, status=200, headers=None, chunks=None):
        self.status_code = status
        self.headers = headers or {}
        self.content = content
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"unsafe error {KEY}", response=self)

    def iter_content(self, chunk_size):
        yield from self.chunks() if self.chunks is not None else [self.content]


def manifest_file(name, content):
    return {
        "filename": name, "size": len(content),
        "hash": "sha256:" + hashlib.sha256(content).hexdigest(),
        "urls": {"https": f"https://downloads.databento.com/{JOB}/{name}"},
    }


def provider_for(monkeypatch, contents, handler=None, *, workers=4, manifest=None):
    records = manifest if manifest is not None else [manifest_file(k, v) for k, v in contents.items()]
    sdk = SimpleNamespace(key=KEY, batch=SimpleNamespace(
        list_files=Mock(return_value=records), submit_job=Mock(), download=Mock()))
    calls, sessions = [], []
    lock = threading.Lock()

    class Session:
        def __init__(self):
            self.auth = None
            self.closed = False
            with lock:
                sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def get(self, url, **kwargs):
            with lock:
                calls.append((url, dict(kwargs), threading.get_ident(), self))
            if handler is not None:
                return handler(url, **kwargs)
            return Response(contents[url.rsplit("/", 1)[-1]])

        def close(self):
            self.closed = True

    monkeypatch.setattr("data_pipeline.providers.databento_download.requests.Session", Session)
    # Retry behavior is tested independently of wall-clock backoff.
    class ImmediateEvent(threading.Event):
        def wait(self, timeout=None):
            return self.is_set()

    monkeypatch.setattr("data_pipeline.providers.databento_download.Event", ImmediateEvent)
    return DatabentoProvider("XNAS.ITCH", client=sdk, download_workers=workers), sdk, calls, sessions


def test_parallel_limit_single_manifest_connection_reuse_and_main_thread_progress(tmp_path, monkeypatch):
    contents = {f"day-{index}.dbn.zst": f"data-{index}".encode() for index in range(12)}
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = maximum = started = 0

    def handler(url, **kwargs):
        name = url.rsplit("/", 1)[-1]

        def chunks():
            nonlocal active, maximum, started
            with lock:
                active += 1
                maximum = max(maximum, active)
                started += 1
                first_batch = started <= 4
            try:
                if first_batch:
                    barrier.wait(timeout=5)
                yield contents[name][:2]
                yield contents[name][2:]
            finally:
                with lock:
                    active -= 1
        return Response(chunks=chunks)

    provider, sdk, calls, sessions = provider_for(monkeypatch, contents, handler)
    progress = []
    caller = threading.get_ident()
    paths = provider.download_job(JOB, tmp_path, progress=lambda value: progress.append((threading.get_ident(), dict(value))))
    assert [path.name for path in paths] == list(contents)
    assert all(path.read_bytes() == contents[path.name] for path in paths)
    assert maximum == 4
    assert len(sessions) == 4
    assert len(calls) == len(contents)
    assert max(Counter(call[2] for call in calls).values()) > 1
    assert all(session.closed for session in sessions)
    sdk.batch.list_files.assert_called_once_with(job_id=JOB)
    assert not sdk.batch.download.called
    assert not sdk.batch.submit_job.called
    assert len(progress) >= 2
    assert all(thread == caller for thread, _ in progress)
    assert progress[-1][1] == {"downloaded_bytes": sum(map(len, contents.values())),
                               "total_bytes": sum(map(len, contents.values())),
                               "completed_files": len(contents), "total_files": len(contents)}
    for _, value in progress:
        assert 0 <= value["downloaded_bytes"] <= value["total_bytes"]
        assert 0 <= value["completed_files"] <= value["total_files"]
    for _, options, _, session in calls:
        auth = options.get("auth", session.auth)
        assert auth == (KEY, "") or (auth.username == KEY and auth.password == "")
        assert options["stream"] is True
        assert options["allow_redirects"] is False


def test_valid_cached_files_skip_network_but_count_toward_progress(tmp_path, monkeypatch):
    contents = {"first.dbn": b"first", "second.dbn": b"second"}
    directory = tmp_path / JOB
    directory.mkdir()
    (directory / "first.dbn").write_bytes(contents["first.dbn"])
    provider, sdk, calls, sessions = provider_for(monkeypatch, contents)
    progress = []
    provider.download_job(JOB, tmp_path, progress=lambda value: progress.append(dict(value)))
    assert [call[0].rsplit("/", 1)[-1] for call in calls] == ["second.dbn"]
    assert progress[-1]["completed_files"] == 2
    assert progress[-1]["downloaded_bytes"] == 11
    sdk.batch.list_files.assert_called_once()
    assert not sdk.batch.submit_job.called


def test_interrupted_download_reuses_partial_bytes_on_next_run(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123456789"}
    failed = True
    first = True

    def handler(url, **kwargs):
        nonlocal first
        offset = int(kwargs.get("headers", {}).get("Range", "bytes=0-")[6:].split("-")[0])
        if failed:
            if first:
                first = False

                def chunks():
                    yield b"123"
                    raise requests.ConnectionError(f"connection lost with {KEY}")
                return Response(chunks=chunks)
            raise requests.ConnectionError(f"connection lost with {KEY}")
        return Response(contents["bars.dbn"][offset:], status=206,
                        headers={"Content-Range": f"bytes {offset}-8/9"})

    provider, sdk, calls, sessions = provider_for(monkeypatch, contents, handler, workers=1)
    with pytest.raises(ProviderError) as error:
        provider.download_job(JOB, tmp_path)
    assert KEY not in str(error.value)
    assert error.value.__cause__ is None
    assert (tmp_path / JOB / "bars.dbn").read_bytes() == b"123"
    assert 1 < len(calls) <= 3
    failed = False
    paths = provider.download_job(JOB, tmp_path)
    assert calls[-1][1]["headers"]["Range"] == "bytes=3-8"
    assert paths[0].read_bytes() == contents["bars.dbn"]
    assert not sdk.batch.submit_job.called
    assert not sdk.batch.download.called


def test_server_ignoring_range_restarts_without_appending_duplicate_bytes(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123456789"}
    directory = tmp_path / JOB
    directory.mkdir()
    (directory / "bars.dbn").write_bytes(b"123")
    provider, sdk, calls, _ = provider_for(monkeypatch, contents)
    paths = provider.download_job(JOB, tmp_path)
    assert calls[0][1]["headers"]["Range"] == "bytes=3-8"
    assert paths[0].read_bytes() == contents["bars.dbn"]
    assert not sdk.batch.submit_job.called


@pytest.mark.parametrize("content_range", ["bytes 0-8/9", "bytes 3-8/10", "bytes 3-4/9", "garbage", None])
def test_bad_content_range_cannot_corrupt_partial_cache(tmp_path, monkeypatch, content_range):
    contents = {"bars.dbn": b"123456789"}
    directory = tmp_path / JOB
    directory.mkdir()
    path = directory / "bars.dbn"
    path.write_bytes(b"123")
    headers = {"Content-Range": content_range} if content_range is not None else {}
    provider, sdk, calls, _ = provider_for(
        monkeypatch, contents, lambda *args, **kwargs: Response(b"456789", status=206, headers=headers))
    with pytest.raises(ProviderError):
        provider.download_job(JOB, tmp_path)
    assert path.read_bytes() == b"123"
    assert not sdk.batch.submit_job.called


@pytest.mark.parametrize("url", [
    "http://downloads.databento.com/bars.dbn", "https://databento.com.evil.test/bars.dbn",
    "https://downloads.databento.com@evil.test/bars.dbn", "file:///tmp/bars.dbn",
])
def test_untrusted_manifest_url_is_rejected_before_any_network_request(tmp_path, monkeypatch, url):
    contents = {"bars.dbn": b"123"}
    manifest = [manifest_file("bars.dbn", b"123")]
    manifest[0]["urls"]["https"] = url
    provider, sdk, calls, _ = provider_for(monkeypatch, contents, manifest=manifest)
    with pytest.raises(ProviderError):
        provider.download_job(JOB, tmp_path)
    assert not calls
    assert not sdk.batch.submit_job.called


@pytest.mark.parametrize("name", ["../escape.dbn", "/tmp/escape.dbn", "bad\\name.dbn"])
def test_unsafe_filename_rejects_entire_manifest_before_download(tmp_path, monkeypatch, name):
    contents = {"good.dbn": b"good"}
    manifest = [manifest_file("good.dbn", b"good"), manifest_file(name, b"bad")]
    provider, sdk, calls, _ = provider_for(monkeypatch, contents, manifest=manifest)
    with pytest.raises(ProviderError):
        provider.download_job(JOB, tmp_path)
    assert not calls


def test_redirect_is_not_followed_with_api_credentials(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123"}
    provider, sdk, calls, _ = provider_for(monkeypatch, contents, lambda *args, **kwargs: Response(
        status=302, headers={"Location": f"https://untrusted.test/?token={KEY}"}))
    with pytest.raises(ProviderError) as error:
        provider.download_job(JOB, tmp_path)
    assert KEY not in str(error.value)
    assert all(call[1]["allow_redirects"] is False for call in calls)
    assert not sdk.batch.submit_job.called


def test_rate_limit_retry_obeys_retry_after_and_uses_same_paid_job(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123"}
    attempts = []

    def handler(*args, **kwargs):
        attempts.append(1)
        return Response(status=429, headers={"Retry-After": "2"}) if len(attempts) == 1 else Response(b"123")

    provider, sdk, calls, _ = provider_for(monkeypatch, contents, handler, workers=1)
    waits = []
    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return self.is_set()

    monkeypatch.setattr("data_pipeline.providers.databento_download.Event", RecordingEvent)
    provider.download_job(JOB, tmp_path)
    assert len(calls) == 2
    assert 2 in waits
    sdk.batch.list_files.assert_called_once_with(job_id=JOB)
    assert not sdk.batch.submit_job.called


def test_long_rate_limit_delay_surfaces_without_retrying_early(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123"}
    provider, sdk, calls, _ = provider_for(monkeypatch, contents, lambda *args, **kwargs: Response(
        status=429, headers={"Retry-After": "120"}))
    waits = []
    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return self.is_set()

    monkeypatch.setattr("data_pipeline.providers.databento_download.Event", RecordingEvent)
    with pytest.raises(ProviderError):
        provider.download_job(JOB, tmp_path)
    assert len(calls) == 1
    assert not waits
    assert not sdk.batch.submit_job.called


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_download_worker_limit_validated_without_io(workers):
    with pytest.raises(ValueError, match="download_workers"):
        DatabentoProvider(download_workers=workers)


def test_valid_databento_redirect_reuses_streaming_auth_and_resumes(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123456789"}
    directory = tmp_path / JOB
    directory.mkdir()
    (directory / "bars.dbn").write_bytes(b"123")
    target = "https://download2.databento.com/bars.dbn"

    def handler(url, **kwargs):
        if url != target:
            return Response(status=302, headers={"Location": target})
        return Response(b"456789", status=206, headers={"Content-Range": "bytes 3-8/9"})

    provider, sdk, calls, _ = provider_for(monkeypatch, contents, handler, workers=1)
    paths = provider.download_job(JOB, tmp_path)
    assert paths[0].read_bytes() == b"123456789"
    assert len(calls) == 2
    assert calls[1][0] == target
    assert all(call[1]["headers"]["Range"] == "bytes=3-8" for call in calls)
    assert all(call[1]["auth"] == (KEY, "") for call in calls)
    assert not sdk.batch.submit_job.called


def test_symlinked_job_cache_cannot_write_outside_download_directory(tmp_path, monkeypatch):
    contents = {"bars.dbn": b"123"}
    destination = tmp_path / "downloads"
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / JOB).symlink_to(outside, target_is_directory=True)
    provider, sdk, calls, _ = provider_for(monkeypatch, contents)
    with pytest.raises(ProviderError):
        provider.download_job(JOB, destination)
    assert not calls
    assert not sdk.batch.list_files.called
    assert not list(outside.iterdir())


def test_cancelled_progress_is_not_called_again_and_keeps_job_files(tmp_path, monkeypatch):
    class Cancelled(BaseException):
        pass

    provider, sdk, _, _ = provider_for(monkeypatch, {"bars.dbn": b"123"}, workers=1)
    notifications = []

    def cancel(value):
        notifications.append(dict(value))
        if value["downloaded_bytes"]:
            raise Cancelled

    with pytest.raises(Cancelled):
        provider.download_job(JOB, tmp_path, progress=cancel)
    assert len(notifications) == 2
    assert (tmp_path / JOB / "bars.dbn").read_bytes() == b"123"
    provider.download_job(JOB, tmp_path)
    assert not sdk.batch.submit_job.called
