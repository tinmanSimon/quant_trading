"""Import explicitly verified, normalized evidence without credentials or I/O."""
import json
from collections.abc import Mapping
from ..models import ActionSnapshot


def import_snapshot(payload: str | bytes | Mapping) -> ActionSnapshot:
    value = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if isinstance(value, Mapping) and 'snapshot' in value:
        if set(value) != {'checksum', 'snapshot'}:
            raise ValueError('Snapshot envelope requires exactly checksum and snapshot.')
        snapshot = ActionSnapshot.from_dict(value['snapshot'])
        if value['checksum'] != snapshot.checksum:
            raise ValueError('Corporate-action snapshot checksum mismatch.')
        return snapshot
    return ActionSnapshot.from_dict(value)
