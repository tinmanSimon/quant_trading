"""Stable public analysis entry points; no data providers or UI dependencies."""

from dataclasses import replace
from datetime import date, datetime
from typing import Callable, Iterable, Iterator

import numpy as np
import polars as pl

from .models import AnalysisResult, CandidateDiagnostic, Trend, TrendConfig
from .scoring import Scorer, SharedIntervals, volatility
from .segmentation import LinearCosts, PenaltyProvenance, PrefixSegmentations, candidate_provenance, penalty_grid
from .tails import TailInfo, diagnose
from .validation import prepare, window_start


def analyze_current(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", price_col: str = "close",
                    as_of: datetime | date | None = None, config: TrendConfig | None = None) -> AnalysisResult:
    """Describe nested current trends using only bars at or before ``as_of``.

    Timestamps must be unique and increasing. The caller supplies completed
    bars; this package cannot infer an exchange calendar or bar completion.
    Prices are never filled, fetched, modified, or extrapolated. Ordinals in
    the result refer to the original input, including its block-window offset.
    The active window matches historical analysis of the same input prefix.
    """
    config = TrendConfig() if config is None else config
    if not isinstance(config, TrendConfig):
        raise ValueError("config must be a TrendConfig")
    times, values, offset = prepare(frame, timestamp_col, price_col, as_of, config)
    return _describe(times, values, offset, config)


def _describe(times: list, values: np.ndarray, offset: int, config: TrendConfig, *,
              costs: LinearCosts | None = None, provenance: PenaltyProvenance | None = None,
              shared: SharedIntervals | None = None) -> AnalysisResult:
    n = len(values)
    result_args = dict(analysis_start=times[0] if times else None,
                       analysis_end=times[-1] if times else None, source_offset=offset, config=config)
    if n < config.min_segment_length:
        return AnalysisResult((), (), "insufficient_data", **result_args)
    costs = LinearCosts(values) if costs is None else costs
    provenance = candidate_provenance(costs, volatility(values), config) if provenance is None else provenance
    starts = list(provenance)
    scorer = Scorer(values, costs, config, shared=shared, offset=offset)
    # Diagnose distinct structures, then allow strong tails to contribute new
    # candidates until stable. Each promotion is at most half its parent's
    # duration, and each distinct start is diagnosed once.
    tails: dict[int, TailInfo] = {}
    while True:
        # Preserve every qualified interval. Nearby intervals are classified
        # only by dashboard display selection, never removed from the archive.
        retained = [start for start in starts if not scorer.rejection(start)]
        promoted = set()
        for start in retained:
            if start not in tails:
                tails[start] = diagnose(start, scorer)
            if tails[start].promoted_start is not None:
                promoted.add(tails[start].promoted_start)
        new_starts = promoted - set(starts)
        if not new_starts:
            break
        starts = sorted(set(starts) | new_starts)
    trends = []
    for start in retained:
        score = scorer.score(start)
        tail = tails.get(start)
        if tail is None:
            tail = diagnose(start, scorer)
        # Independent fine-resolution countertrends remain visible even if
        # their whole-parent fitting improvement falls below the tail gate.
        local = [item for item in retained if item > start
                 and scorer.score(item).direction != score.direction
                 and (n - item) <= (n - start) * config.max_tail_fraction]
        if local:
            local_start = min(local)
            status = ("Broader direction materially challenged"
                      if tail.retracement >= config.retracement_threshold
                      or (n - local_start) / (n - start) >= config.countertrend_fraction
                      else "Contains a local countertrend")
            tail = replace(tail, status=status, start=local_start)
        parent = max((item for item in retained if item < start), default=None)
        def identifier(local_start: int) -> str:
            direction = "up" if scorer.score(local_start).direction > 0 else "down"
            return f"T{local_start + offset}-{n - 1 + offset}-{direction}"

        parent_id = identifier(parent) if parent is not None else None
        selecting = provenance.get(start, ())
        trends.append(Trend(
            trend_id=identifier(start), start_index=start + offset, end_index=n - 1 + offset,
            start_time=times[start], end_time=times[-1], duration=n - start,
            direction=score.direction, clarity=score.clarity, slope=score.slope,
            intercept=score.intercept, rate_per_bar=float(np.expm1(score.slope)),
            boundary_robustness=scorer.boundary(start), magnitude=score.magnitude,
            scores=score.scores, tail_status=tail.status,
            tail_start_time=times[tail.start] if tail.start is not None else None,
            retracement=tail.retracement, onset_truncated=start == 0,
            tail_fit_improvement=tail.fit_improvement, tail_slope_change=tail.slope_change,
            tail_level_change=tail.level_change, parent_id=parent_id, nearby_start_indices=(start + offset,),
            lambda_values=tuple(value for _, value in selecting),
            lambda_levels=tuple(level for level, _ in selecting),
            origin="segmentation" if selecting else "tail",
        ))
    diagnostics = []
    for start in starts:
        score = scorer.score(start)
        reason = scorer.rejection(start)
        selecting = provenance.get(start, ())
        diagnostics.append(CandidateDiagnostic(
            start_index=start + offset, end_index=n - 1 + offset, duration=n - start,
            direction=score.direction, clarity=score.clarity, magnitude=score.magnitude,
            boundary_robustness=scorer.boundary(start), scores=score.scores,
            accepted=start in retained, rejection_reason=reason,
            lambda_values=tuple(value for _, value in selecting),
            lambda_levels=tuple(level for level, _ in selecting),
            origin="segmentation" if selecting else "tail",
        ))
    return AnalysisResult(tuple(trends), tuple(diagnostics), "ok" if trends else "no_clear_trend", **result_args)


def analyze_history(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", price_col: str = "close",
                    as_of: datetime | date | None = None, config: TrendConfig | None = None,
                    progress: Callable[[int, int], None] | None = None) -> AnalysisResult:
    """Analyze every eligible historical endpoint and retain all its records.

    Windows hold up to twice ``max_history`` bars and advance by ``max_history``
    after reaching capacity. Rebuild the overlapping context once per shift,
    then extend segmentation for each new endpoint. Only the active window's
    prefix through that endpoint determines its analysis and volatility.
    Earlier records are unchanged when future observations are appended.
    ``progress`` counts published endpoints, not overlap rebuilding work.
    """
    config = TrendConfig() if config is None else config
    if not isinstance(config, TrendConfig):
        raise ValueError("config must be a TrendConfig")
    times, values, _ = prepare(frame, timestamp_col, price_col, as_of, config, truncate=False)
    n = len(values)
    total = max(0, n - config.min_segment_length + 1)
    if progress is not None:
        progress(0, total)
    result_args = dict(analysis_start=times[0] if times else None,
                       analysis_end=times[-1] if times else None, source_offset=0,
                       config=config, analysis_mode="history")
    if not total:
        return AnalysisResult((), (), "insufficient_data", **result_args)
    capacity = 2 * config.max_history
    penalties = np.asarray(penalty_grid(config))
    partitions = PrefixSegmentations(values[:capacity], config, penalties=penalties)
    shared = SharedIntervals(values, capacity)
    offset = 0
    trends, diagnostics = [], []
    for completed, end in enumerate(range(config.min_segment_length, n + 1), 1):
        next_offset = window_start(end, config.max_history)
        if next_offset != offset:
            offset = next_offset
            # Release the old DP before allocating the next window's table.
            del partitions
            partitions = PrefixSegmentations(values[offset:offset + capacity], config, penalties=penalties)
            shared.discard_before(offset)
        # On a shift, this first query also rebuilds the H overlapping bars.
        # They supply context only: previous records are never emitted again.
        provenance = partitions.candidate_provenance(end - offset)
        snapshot = _describe(times[offset:end], values[offset:end], offset, config,
                             costs=partitions.costs, provenance=provenance, shared=shared)
        trends.extend(snapshot.trends)
        diagnostics.extend(snapshot.diagnostics)
        if progress is not None:
            progress(completed, total)
    return AnalysisResult(tuple(trends), tuple(diagnostics), "ok" if trends else "no_clear_trend", **result_args)


def analyze(frame: pl.DataFrame, *, timestamp_col: str = "timestamp", price_col: str = "close",
            as_of: datetime | date | None = None, config: TrendConfig | None = None,
            progress: Callable[[int, int], None] | None = None) -> AnalysisResult:
    """Default analysis: causal records for every eligible historical endpoint."""
    return analyze_history(frame, timestamp_col=timestamp_col, price_col=price_col,
                           as_of=as_of, config=config, progress=progress)


def replay(frame: pl.DataFrame, *, as_of_values: Iterable[datetime | date] | None = None,
           step: int = 1, timestamp_col: str = "timestamp", price_col: str = "close",
           config: TrendConfig | None = None) -> Iterator[AnalysisResult]:
    """Yield independent historical-prefix snapshots, with no future state.

    Result-local IDs intentionally do not imply historical identity or a first
    recognition date. Callers can provide a sparse sequence of review dates.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    config = config or TrendConfig()
    if as_of_values is None:
        if timestamp_col not in frame.columns:
            raise ValueError(f"Dataframe requires {timestamp_col!r}")
        as_of_values = frame[timestamp_col].to_list()[config.min_segment_length - 1::step]
    for cutoff in as_of_values:
        yield analyze_current(frame, timestamp_col=timestamp_col, price_col=price_col, as_of=cutoff, config=config)
