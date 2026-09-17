"""Serialize catalog and file publication on a local POSIX filesystem."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import time

from ..exceptions import StorageBusyError
from .layout import contained_path


@contextmanager
def store_lock(root: Path, timeout: float = 10):
    root.mkdir(parents=True, exist_ok=True)
    path = contained_path(root, "metadata/store.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise StorageBusyError(f"Store is busy: {root}") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
