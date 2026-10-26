"""Known ingestion gaps carried alongside bars, never inferred from weekends."""

from bisect import bisect_left
from dataclasses import replace
from datetime import UTC, datetime

from research.analysis.flat import analyze_summary
from research.analysis.flat.models import FlatAnalysisResult
from research.analysis.flat.validation import prepare


def _timestamp(value):
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(stamp, datetime) or stamp.utcoffset() is None:
        raise ValueError("Recorded omission timestamps must be timezone-aware datetimes.")
    return stamp.astimezone(UTC)


def omission_metadata(sources, *, pipeline=None):
    """Keep raw gaps distinct from unresolved omissions in processed lineage.

    A resampled output timestamp does not prove that an omitted input was
    repaired. Check the exact raw parents when available; never infer repairs
    from the processed frame. Mapping remaining gaps through arbitrary processor
    recipes requires separate validation.
    """
    raw, processed, parents = set(), [], {}
    for source in sources:
        stamps = {_timestamp(bar["timestamp"]) for bar in
                  (source.get("quality") or {}).get("omitted_bars", [])}
        if source.get("layer", "raw") != "processed":
            raw.update(stamps)
            continue
        if pipeline is not None and stamps:
            present = set()
            for identifier in source.get("parent_ids", []):
                if identifier not in parents:
                    item = pipeline.get_metadata(identifier)
                    if item.layer != "raw":
                        raise ValueError("Processed omission checks require immutable raw parents.")
                    frame = pipeline.read_dataset(identifier)
                    parents[identifier] = (item, set(frame["timestamp"].to_list()))
                item, times = parents[identifier]
                stamps.update(bar.timestamp for bar in item.quality.omitted_bars)
                present.update(times)
            stamps.difference_update(present)
        if stamps:
            processed.append({"dataset_id": source.get("dataset_id", "unknown"),
                              "omitted_bar_timestamps": [stamp.isoformat() for stamp in sorted(stamps)]})
    return {"omitted_bar_timestamps": [stamp.isoformat() for stamp in sorted(raw)],
            "processed_omission_sources": processed}


def omission_report(context, *, start=None, end=None):
    """Describe known compromises; absence of records does not prove coverage.

    Explicit bounds include omitted leading/trailing bars in a selected request.
    Without them, report the completed frame's first-through-last interval.
    Processed gaps stay visible even when their output mapping is not known.
    """
    frame = context.completed_frame
    times = frame["timestamp"].to_list()
    date_labels = bool(times) and not isinstance(times[0], datetime)

    def label(value):
        return value.date() if date_labels and isinstance(value, datetime) else value

    present = set(times)
    lower = label(start) if start is not None else times[0] if times else None
    upper = label(end) if end is not None else times[-1] if times else None
    unresolved = []
    for value in context.metadata.get("omitted_bar_timestamps", []):
        stamp = _timestamp(value)
        key = label(stamp)
        if key in present or lower is not None and key < lower:
            continue
        if upper is not None and (key >= upper if end is not None else key > upper):
            continue
        unresolved.append(stamp.isoformat())
    unresolved = sorted(set(unresolved))
    processed = context.metadata.get("processed_omission_sources", [])
    messages = []
    if unresolved:
        preview = ", ".join(unresolved[:5])
        messages.append(f"Data quality warning: {len(unresolved):,} recorded omitted bars remain absent "
                        f"in the selected data (UTC): {preview}. Screening results use incomplete history; "
                        "strict research requires separate coverage validation.")
    if processed:
        count = len({stamp for source in processed for stamp in source["omitted_bar_timestamps"]})
        messages.append(f"Data quality warning: processed data has {count:,} unresolved omitted raw-input "
                        "bars. Their effect on output bars has not been verified; output timestamps do "
                        "not establish repaired inputs.")
    return {"unresolved_omitted_bar_count": len(unresolved),
            "unresolved_omitted_bar_timestamps": unresolved,
            "processed_omission_sources": processed, "messages": messages}


def analyze_flat_context(context, config, progress=None):
    """Detect independently between known gaps, keeping original row indices."""
    if context.metadata.get("processed_omission_sources"):
        raise ValueError("Cannot validate flat periods for processed data with unresolved omitted raw-input "
                         "bars. Repair the source bars or use raw screening data; a processed bar timestamp "
                         "does not establish complete underlying data.")
    frame = context.completed_frame
    times = frame["timestamp"].to_list()
    present = set(times)
    boundaries = set()
    for value in context.metadata.get("omitted_bar_timestamps", []):
        stamp = _timestamp(value)
        if times and not isinstance(times[0], datetime):
            stamp = stamp.date()
        if stamp in present:
            continue  # Another revision may have repaired this historical omission.
        index = bisect_left(times, stamp)
        if 0 < index < len(times):
            boundaries.add(index)
    if not boundaries:
        return analyze_summary(frame, config=config, progress=progress)
    # Segment-local validation cannot detect different symbols/timeframes on
    # opposite sides of a gap. Preserve the engine's whole-series contract.
    prepare(frame, "timestamp", "high", "low", None)
    periods = []
    offsets = [0, *sorted(boundaries), len(times)]
    for start, end in zip(offsets, offsets[1:]):
        def report(done, total):
            if progress:
                progress(start + done, len(times))
        part = analyze_summary(frame.slice(start, end - start), config=config, progress=report)
        periods.extend(replace(period, start_index=period.start_index + start,
                               end_index=period.end_index + start,
                               period_id=f"F{period.start_index + start}-{period.end_index + start}")
                       for period in part.periods)
    periods.sort(key=lambda period: period.end_index, reverse=True)
    return FlatAnalysisResult(tuple(periods), config, len(times), times[0], times[-1], "summary",
                              "ok" if periods else "insufficient_data" if len(times) < config.min_duration
                              else "no_flat_period")
