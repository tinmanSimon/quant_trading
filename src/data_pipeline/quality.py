"""Immutable fetch provenance, persisted with every dataset revision.

``reported`` means ingestion recorded its policy and explicit omissions. It
does not certify complete market coverage: a vendor can omit rows entirely.
Older datasets and providers without reporting have ``unknown`` provenance.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import polars as pl


@dataclass(frozen=True)
class OmittedBar:
    timestamp: datetime
    reason: str

    def __post_init__(self):
        if not isinstance(self.timestamp, datetime) or self.timestamp.utcoffset() is None:
            raise ValueError("Omitted bar timestamps must be timezone-aware datetimes.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("An omitted bar requires a nonempty reason.")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))


@dataclass(frozen=True)
class FetchQuality:
    status: Literal["unknown", "reported"] = "unknown"
    skip_missing_ohlc: bool | None = None
    omitted_bars: tuple[OmittedBar, ...] = ()
    provider_version: str | None = None

    def __post_init__(self):
        if self.status not in ("unknown", "reported"):
            raise ValueError("Fetch quality status must be unknown or reported.")
        if self.skip_missing_ohlc is not None and not isinstance(self.skip_missing_ohlc, bool):
            raise ValueError("skip_missing_ohlc must be a boolean or None.")
        if self.provider_version is not None and not isinstance(self.provider_version, str):
            raise ValueError("provider_version must be a string or None.")
        omissions = tuple(self.omitted_bars)
        if any(not isinstance(item, OmittedBar) for item in omissions):
            raise ValueError("omitted_bars must contain OmittedBar records.")
        object.__setattr__(self, "omitted_bars", tuple(sorted(
            set(omissions), key=lambda item: (item.timestamp, item.reason),
        )))

    @classmethod
    def from_dict(cls, value: dict | None) -> "FetchQuality":
        if value is None:
            return cls()
        fields = dict(value)
        fields["omitted_bars"] = tuple(
            OmittedBar(datetime.fromisoformat(item["timestamp"]), item["reason"])
            for item in fields.get("omitted_bars", ())
        )
        return cls(**fields)


@dataclass(frozen=True)
class FetchResult:
    frame: pl.DataFrame
    quality: FetchQuality = field(default_factory=FetchQuality)

    def __post_init__(self):
        if not isinstance(self.frame, pl.DataFrame) or not isinstance(self.quality, FetchQuality):
            raise TypeError("FetchResult requires a Polars DataFrame and FetchQuality.")


def merge_quality(reports: list[FetchQuality]) -> FetchQuality:
    """Retain known source omissions; original policies remain in parent metadata.

    A merged report is unknown if any source is unknown. Its skip flag is true
    if any source enabled skipping, false if all disabled it, otherwise unknown.
    Present rows can repair recorded historical omissions, so coverage checks
    must compare omissions against the actual selected input timestamps.
    """
    if not reports:
        return FetchQuality()
    policies = {report.skip_missing_ohlc for report in reports}
    versions = {report.provider_version for report in reports}
    return FetchQuality(
        status="reported" if all(report.status == "reported" for report in reports) else "unknown",
        skip_missing_ohlc=True if True in policies else False if policies == {False} else None,
        omitted_bars=tuple(bar for report in reports for bar in report.omitted_bars),
        provider_version=next(iter(versions)) if len(versions) == 1 else None,
    )
