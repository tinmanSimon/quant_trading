"""Immutable, provider-neutral descriptions of the data at each processing step."""

from dataclasses import dataclass
import json
import re

import polars as pl

from ..exceptions import ProcessingError
from ..models import DataRequest
from ..schemas import OHLCV_SCHEMA, validate_ohlcv


@dataclass(frozen=True, slots=True)
class DataContract:
    """Semantic output contract, not a storage location or a vendor request.

    Only OHLCV schema version 1 is implemented. Instrument identity, provider,
    and source IDs remain owned by orchestration/storage. ``source_basis``
    describes the original price/volume basis; ``adjustment_json`` records
    subsequent adjustments and survives timeframe changes.
    """

    timeframe: str
    dataset: str = "ohlcv"
    schema_version: int = 1
    timestamp_convention: str | None = None
    source_basis: str = "unknown"
    adjustment_json: str = ""

    def __post_init__(self):
        if self.dataset != "ohlcv" or type(self.schema_version) is not int or self.schema_version != 1:
            raise ProcessingError("Only the OHLCV schema version 1 contract is supported.")
        if not isinstance(self.timeframe, str):
            raise ProcessingError("Contract timeframe must be a string.")
        timeframe = self.timeframe.strip().lower()
        if not re.fullmatch(r"[1-9][0-9]*(m|h|d|wk|mo)", timeframe):
            raise ProcessingError("Invalid contract timeframe.")
        if timeframe.endswith("m") and int(timeframe[:-1]) % 60 == 0:
            timeframe = f"{int(timeframe[:-1]) // 60}h"
        convention = "bar_start" if timeframe.endswith(("m", "h")) else "session_date"
        if self.timestamp_convention not in (None, convention):
            raise ProcessingError("Timestamp convention is incompatible with contract timeframe.")
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "timestamp_convention", convention)
        if self.source_basis not in ("unknown", "unadjusted", "vendor_adjusted"):
            raise ProcessingError("Source basis must be unknown, unadjusted or vendor_adjusted.")
        if not isinstance(self.adjustment_json, str):
            raise ProcessingError("Adjustment metadata must be a JSON object string.")
        if self.adjustment_json:
            try:
                metadata = json.loads(self.adjustment_json, object_pairs_hook=_unique_object)
                if not isinstance(metadata, dict) or not metadata:
                    raise ValueError("Adjustment metadata must be a nonempty object.")
                canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ProcessingError(f"Invalid adjustment metadata: {error}") from error
            object.__setattr__(self, "adjustment_json", canonical)

    @property
    def schema(self) -> dict:
        return dict(OHLCV_SCHEMA)

    @classmethod
    def from_request(cls, request: DataRequest) -> "DataContract":
        return cls(timeframe=request.timeframe, dataset=request.dataset)

    def validate(self, frame: pl.DataFrame) -> pl.DataFrame:
        frame = validate_ohlcv(frame)
        if self.timestamp_convention == "session_date":
            if frame.select((pl.col("timestamp") != pl.col("timestamp").dt.truncate("1d")).any()).item():
                raise ProcessingError("Session-date output must use midnight UTC timestamps.")
        return frame


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate adjustment metadata key: {key}")
        value[key] = item
    return value


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Data and its output contract, returned by a processor or a pipeline."""

    frame: pl.DataFrame
    contract: DataContract
