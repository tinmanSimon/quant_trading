"""Content-addressed snapshots, never overwritten; checksum verified on read."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from .models import ActionSnapshot


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, snapshot: ActionSnapshot) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f'{snapshot.checksum}.json'
        payload = json.dumps({'checksum': snapshot.checksum, 'snapshot': snapshot.to_dict()}, indent=2, allow_nan=False)
        fd, temp = tempfile.mkstemp(dir=self.root, prefix='.snapshot-')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(payload + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp, target)
            except FileExistsError:
                if self.load(target).checksum != snapshot.checksum:
                    raise ValueError('Existing snapshot differs from its content address.')
        finally:
            os.unlink(temp)
        return target

    @staticmethod
    def load(path: str | Path) -> ActionSnapshot:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value, dict) or set(value) != {'checksum', 'snapshot'}:
            raise ValueError('Stored snapshot requires checksum and snapshot fields.')
        snapshot = ActionSnapshot.from_dict(value['snapshot'])
        if value['checksum'] != snapshot.checksum:
            raise ValueError('Corporate-action snapshot checksum mismatch.')
        return snapshot
