"""Pure range planning for refreshes and missing-range acquisition of raw data.

Coverage comes from actual stored first/last timestamps, never the original
vendor request. The final bar contributes one nominal timeframe: UTC elapsed
minutes/hours/days/weeks, or calendar months for month bars. Daily and longer
bars use date labels. This is range-based coverage, not an exchange-calendar or
missing-bar audit; internal gaps and weekends are deliberately not repaired.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import re
from typing import Iterable

from .models import DataRequest
from .storage.models import StoredDataset


@dataclass(frozen=True, slots=True)
class FetchPlan:
    """The original request and its nonmutating acquisition/replacement plan."""

    request: DataRequest
    fetch_request: DataRequest | None
    existing_dataset_ids: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MissingFetchPlan:
    """Missing vendor ranges and the local batches to combine with their data."""

    request: DataRequest
    fetch_requests: tuple[DataRequest, ...]
    merge_request: DataRequest
    existing_dataset_ids: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()


def _identity(request: DataRequest) -> tuple[str, ...]:
    return (request.dataset, request.provider, request.symbol, request.timeframe,
            request.price_adjustment)


def _utc(stamp: datetime) -> datetime:
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Stored coverage timestamps must be timezone-aware.")
    return stamp.astimezone(UTC)


def _terminal_end(stamp: datetime, timeframe: str) -> datetime:
    count, unit = re.fullmatch(r"([1-9][0-9]*)(m|h|d|wk|mo)", timeframe).groups()
    count = int(count)
    if unit == "mo":
        month = stamp.year * 12 + stamp.month - 1 + count
        year, month = divmod(month, 12)
        month += 1
        return stamp.replace(year=year, month=month,
                             day=min(stamp.day, monthrange(year, month)[1]))
    return stamp + timedelta(**{{"m": "minutes", "h": "hours", "d": "days", "wk": "weeks"}[unit]: count})


def _coverage_intervals(request: DataRequest, datasets: Iterable[StoredDataset]):
    identity = _identity(request)
    intervals = []
    for item in datasets:
        if not item.active or item.layer != "raw" or _identity(item.request) != identity:
            continue
        first, last = _utc(item.first_timestamp), _utc(item.last_timestamp)
        if first > last or item.row_count <= 0:
            raise ValueError(f"Stored dataset {item.dataset_id!r} has invalid actual coverage.")
        intervals.append((first, _terminal_end(last, request.timeframe), item.dataset_id))
    intervals.sort()

    return intervals


def plan_fetch(request: DataRequest, datasets: Iterable[StoredDataset]) -> FetchPlan:
    """Skip a covered request, or refresh its strict-overlap closure atomically.

    Adjacent actual spans form continuous coverage for deciding whether to
    skip. A refresh only replaces spans that overlap the requested/expanded
    window; touching its endpoint alone does not make a dataset a target.
    IDs are ordered by actual coverage start, end, and ID for reproducibility.
    """
    intervals = _coverage_intervals(request, datasets)

    covered_until, has_gap, covering = request.start, False, []
    for first, end, identifier in intervals:
        if end <= request.start or first >= request.end:
            continue
        covering.append(identifier)
        if first > covered_until:
            has_gap = True
        covered_until = max(covered_until, end)
    if not has_gap and covered_until >= request.end:
        return FetchPlan(request, None, tuple(covering))

    # Strictly overlapping spans form connected components. Their envelopes
    # give the same fixed point as repeatedly expanding around overlaps, while
    # a single ordered pass also handles long chains efficiently.
    components = []
    for first, end, identifier in intervals:
        if not components or first >= components[-1][1]:
            components.append([first, end, [identifier]])
        else:
            components[-1][1] = max(components[-1][1], end)
            components[-1][2].append(identifier)
    start, end, targets = request.start, request.end, []
    for first, terminal, identifiers in components:
        if first < request.end and terminal > request.start:
            start, end = min(start, first), max(end, terminal)
            targets.extend(identifiers)
    if not targets:
        return FetchPlan(request, request)
    return FetchPlan(request, replace(request, start=start, end=end), supersedes=tuple(targets))



def plan_missing_fetch(request: DataRequest, datasets: Iterable[StoredDataset]) -> MissingFetchPlan:
    """Fetch only uncovered original ranges, then combine adjacent stored spans.

    Fetch requests are ordered, disjoint, and bounded by the original request.
    Unlike refresh planning, local merge targets include adjacency as well as
    overlap, including the whole connected chain on either side. Their full
    spans expand only ``merge_request``; existing bars are never refetched.
    Fully covered requests are skipped without compacting existing batches.
    """
    intervals = _coverage_intervals(request, datasets)
    cursor, missing, covering = request.start, [], []
    for first, end, identifier in intervals:
        if end <= request.start or first >= request.end:
            continue
        covering.append(identifier)
        if first > cursor:
            missing.append(replace(request, start=cursor, end=first))
        cursor = max(cursor, min(end, request.end))
    if cursor < request.end:
        missing.append(replace(request, start=cursor))
    if not missing:
        return MissingFetchPlan(request, (), request, tuple(covering))

    components = []
    for first, end, identifier in intervals:
        if not components or first > components[-1][1]:
            components.append([first, end, [identifier]])
        else:
            components[-1][1] = max(components[-1][1], end)
            components[-1][2].append(identifier)
    start, end, targets = request.start, request.end, []
    for first, terminal, identifiers in components:
        if first <= request.end and terminal >= request.start:
            start, end = min(start, first), max(end, terminal)
            targets.extend(identifiers)
    merge_request = replace(request, start=start, end=end) if targets else request
    return MissingFetchPlan(request, tuple(missing), merge_request, supersedes=tuple(targets))
