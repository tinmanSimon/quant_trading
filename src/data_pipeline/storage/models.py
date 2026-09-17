"""Serializable metadata for immutable Parquet batches and their lineage."""

from dataclasses import asdict, dataclass
from datetime import datetime
import json

from ..models import DataRequest
from ..processing.contracts import DataContract


@dataclass(frozen=True)
class StoredDataset:
    dataset_id: str
    layer: str
    request: DataRequest
    first_timestamp: datetime
    last_timestamp: datetime
    row_count: int
    relative_path: str
    checksum_sha256: str
    schema_version: int
    created_at: datetime
    pipeline_id: str = ""
    processors_json: str = "[]"
    parent_ids: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    active: bool = True

    @property
    def timestamp_convention(self) -> str:
        return "bar_start" if self.request.timeframe.endswith(("m", "h")) else "session_date"

    @property
    def contract(self) -> DataContract:
        return DataContract(timeframe=self.request.timeframe, dataset=self.request.dataset,
                            schema_version=self.schema_version)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=lambda value: value.isoformat(), sort_keys=True)

    @classmethod
    def from_json(cls, value: str, *, active: bool) -> "StoredDataset":
        fields = json.loads(value)
        request = fields["request"]
        for name in ("start", "end"):
            request[name] = datetime.fromisoformat(request[name])
        fields["request"] = DataRequest(**request)
        for name in ("first_timestamp", "last_timestamp", "created_at"):
            fields[name] = datetime.fromisoformat(fields[name])
        for name in ("parent_ids", "supersedes"):
            fields[name] = tuple(fields[name])
        fields["active"] = active
        return cls(**fields)


RawDataset = StoredDataset
