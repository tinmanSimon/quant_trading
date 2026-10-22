"""Bounded, resumable transfers from a single Databento batch manifest."""

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import math
from pathlib import Path
from queue import Empty, Queue
import re
from threading import Event, Lock
from urllib.parse import urljoin, urlsplit

import requests

from ..exceptions import ProviderError

_CHUNK_SIZE = 256 * 1024
_RETRIES = 2


@dataclass(frozen=True)
class _File:
    path: Path
    size: int
    checksum: str
    url: str


def _download_url(value):
    try:
        url = urlsplit(value)
        if (not isinstance(value, str) or url.scheme != "https"
                or not url.hostname or not (url.hostname == "databento.com"
                                            or url.hostname.endswith(".databento.com"))
                or url.port not in (None, 443) or url.username or url.password):
            raise ValueError
    except (TypeError, ValueError, AttributeError):
        raise ProviderError("Databento returned an invalid download URL.") from None
    return value


def _manifest_files(manifest, directory):
    if not isinstance(manifest, list) or not manifest:
        raise ProviderError("Databento returned no files for this batch job.")
    result, names = [], set()
    for item in manifest:
        name = item.get("filename") if isinstance(item, Mapping) else None
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or Path(name).name != name or "\\" in name or name in names
                or not (directory / name).resolve().is_relative_to(directory)):
            raise ProviderError("Databento returned an invalid or duplicate job filename.")
        size, checksum = item.get("size"), item.get("hash")
        if type(size) is not int or size < 0:
            raise ProviderError("Databento returned an invalid download file size.")
        if not isinstance(checksum, str) or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", checksum):
            raise ProviderError("Databento file manifest lacks a SHA-256 checksum.")
        urls = item.get("urls")
        url = _download_url(urls.get("https") if isinstance(urls, Mapping) else None)
        result.append(_File(directory / name, size, checksum[7:].lower(), url))
        names.add(name)
    return result


def _matches(item):
    with item.path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == item.checksum


def _retry_delay(value, attempt):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value).astimezone(UTC) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = 2 ** attempt
    # A long vendor cooldown must be respected, not shortened to retry early.
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 2 ** attempt
    if seconds > 60:
        error = ProviderError("Databento download is rate limited; retry this saved job later.")
        error.retry_after = seconds
        raise error
    return seconds


def _transfer(session, item, key, update, stop):
    for attempt in range(_RETRIES + 1):
        if stop.is_set():
            return
        offset = item.path.stat().st_size if item.path.exists() else 0
        if item.path.exists() and offset >= item.size:
            if offset == item.size and _matches(item):
                update(item.size, True)
                return
            # Only discard an invalid file belonging to this saved job.
            item.path.unlink()
            offset = 0
        update(offset, False)
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-{item.size - 1}"
        retry_delay = 2 ** attempt
        try:
            url = item.url
            # Follow only validated HTTPS Databento redirects, so credentials
            # are never attached to a foreign host or insecure endpoint.
            for redirect in range(4):
                response = session.get(url, headers=headers, auth=(key, ""),
                                       stream=True, timeout=(10, 60), allow_redirects=False)
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                try:
                    url = _download_url(urljoin(url, response.headers.get("Location", "")))
                finally:
                    response.close()
            else:
                raise ProviderError("Too many redirects during Databento download.")
            with response:
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                    raise requests.ConnectionError("Temporary download failure")
                if response.status_code == 206:
                    expected = f"bytes {offset}-{item.size - 1}/{item.size}"
                    if response.headers.get("Content-Range") != expected:
                        raise ProviderError("Databento download has an invalid Content-Range.")
                elif response.status_code == 200:
                    # A server may ignore Range. Replace, never append, its
                    # complete response to avoid duplicating the partial file.
                    offset = 0
                    update(0, False)
                else:
                    detail = {401: "authentication failed", 403: "access denied"}.get(
                        response.status_code, f"HTTP {response.status_code}")
                    raise ProviderError(f"Databento download failed: {detail}.")
                with item.path.open("ab" if offset else "wb") as output:
                    for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                        if stop.is_set():
                            return
                        if not chunk:
                            continue
                        if offset + len(chunk) > item.size:
                            raise ProviderError("Databento download exceeds its expected file size.")
                        output.write(chunk)
                        offset += len(chunk)
                        update(offset, False)
            if offset != item.size:
                raise requests.ConnectionError("Incomplete download")
            if not _matches(item):
                item.path.unlink()
                update(0, False)
                raise ProviderError("Databento download failed SHA-256 checksum validation.")
            update(item.size, True)
            return
        except requests.RequestException:
            if attempt == _RETRIES:
                error = ProviderError("Databento download interrupted; retry to resume this saved job.")
                error.retry_after = retry_delay
                raise error from None
            if stop.wait(retry_delay):
                return


def download_files(manifest, directory, *, key, workers=4, progress=None):
    """Reuse a manifest and cached files; invoke progress only on this thread.

    Each worker reuses its own HTTP session. Only four workers are used by
    default, regardless of ticker count; file buffers and queued work are bounded.
    """
    directory = Path(directory).resolve()
    files = _manifest_files(manifest, directory)
    directory.mkdir(parents=True, exist_ok=True)
    pending = Queue()
    for index, item in enumerate(files):
        pending.put((index, item))
    stop, lock = Event(), Lock()
    sizes, completed = [0] * len(files), [False] * len(files)
    total = sum(item.size for item in files)

    def update(index, size, done):
        with lock:
            sizes[index], completed[index] = size, done

    def worker():
        try:
            with requests.Session() as session:
                while not stop.is_set():
                    try:
                        index, item = pending.get_nowait()
                    except Empty:
                        return
                    _transfer(session, item, key, lambda size, done: update(index, size, done), stop)
        except BaseException:
            stop.set()
            raise

    last, callback_failed = None, False

    def report():
        nonlocal last, callback_failed
        if callback_failed:
            return
        with lock:
            state = {"downloaded_bytes": sum(sizes), "total_bytes": total,
                     "completed_files": sum(completed), "total_files": len(files)}
        if progress is not None and state != last:
            last = dict(state)
            try:
                progress(state)
            except BaseException:
                callback_failed = True
                raise

    report()
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as executor:
            futures = {executor.submit(worker) for _ in range(min(workers, len(files)))}
            try:
                while futures:
                    done, futures = wait(futures, timeout=0.15, return_when=FIRST_COMPLETED)
                    report()
                    for future in done:
                        future.result()
            finally:
                stop.set()
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("Could not download Databento files; retry this saved job to resume.") from None
    finally:
        report()
    return tuple(item.path for item in files)
