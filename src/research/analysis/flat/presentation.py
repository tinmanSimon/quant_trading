"""Plain tables, chart descriptors, and navigation for a flat-period summary."""

from html import escape
import math

from research.analysis.core.base import LogicResult


_PERIOD_COLUMNS = [
    "period_id", "start_index", "end_index", "start", "end", "bars", "lower_price",
    "upper_price", "center_price", "range_pct", "ends_at_current_bar", "_event_id",
]


def summary_result(analysis):
    """Serialize the selected periods once for both dashboard entry points.

    This is calculation output, not a display filter: batch analysis retains
    these same tables without invoking the chart's presentation method.
    """
    rows, overlays, events = [], [], []
    boundary_x, boundary_y = [], []
    last = analysis.source_count - 1
    for period in analysis.periods:
        start, end = period.start_index, period.end_index
        current = end == last
        start_time, end_time = period.start_time.isoformat(), period.end_time.isoformat()
        event_id = f"period:{period.period_id}"
        rows.append({
            "period_id": period.period_id, "start_index": start, "end_index": end,
            "start": start_time, "end": end_time, "bars": period.duration,
            "lower_price": period.lower_price, "upper_price": period.upper_price,
            "center_price": period.center_price, "range_pct": period.range_pct,
            "ends_at_current_bar": current, "_event_id": event_id,
        })
        hover = [
            f"<b>Flat period {escape(period.period_id)}</b>",
            f"Start: {escape(start_time)}", f"End: {escape(end_time)}",
            f"Duration: {period.duration:,} bars",
            f"Band width: {period.range_pct:.6g}%",
            f"Lowest low: {period.lower_price:.6g}", f"Highest high: {period.upper_price:.6g}",
            "Ends at current bar" if current else "Historical period",
        ]
        for suffix, label, price in (("", "center", period.center_price),
                                     ("-lower", "low", period.lower_price),
                                     ("-upper", "high", period.upper_price)):
            overlays.append({
                "id": period.period_id + suffix, "type": "log_line",
                "name": f"Flat {period.period_id} {label}",
                "start_index": start, "end_index": end,
                "slope": 0.0, "intercept": math.log(price),
                "line": {"color": "#0d9488" if current else "#6366f1",
                         "width": 1 if suffix else 2, "dash": "dot" if suffix else "solid"},
                "hovertemplate": "<br>".join(hover) + f"<br>Band {label}: %{{y:.6g}}<extra></extra>",
            })
        events.append({
            "id": event_id, "group": "Flat periods",
            "label": f"{period.period_id} · {period.duration} bars · {period.range_pct:.4g}% range",
            "source_index": start, "timestamp": start_time,
            "end_index": end, "end_timestamp": end_time,
        })
    # The summary is newest-first; chart arrays must be in source-bar order.
    for period in reversed(analysis.periods):
        boundary_x.extend((period.start_index, period.end_index))
        boundary_y.extend((period.center_price, period.center_price))
    if boundary_x:
        overlays.append({
            "id": "flat-boundaries", "type": "markers", "name": "Flat period boundaries",
            "x": boundary_x, "y": boundary_y,
            "marker": {"color": "#0d9488", "size": 7, "symbol": "diamond"},
            "hovertemplate": "Flat period boundary<br>Band center: %{y:.6g}<extra></extra>",
        })
    covered = sum(period.duration for period in analysis.periods)
    current_period = next((period for period in analysis.periods if period.end_index == last), None)
    summary = {
        "period_count": len(analysis.periods), "source_bars": analysis.source_count,
        "covered_bars": covered,
        "coverage_percent": 100 * covered / analysis.source_count if analysis.source_count else 0.0,
        "current_flat": current_period is not None,
        "current_duration": current_period.duration if current_period is not None else 0,
    }
    if analysis.status == "insufficient_data":
        message = f"Insufficient completed bars: need at least {analysis.config.min_duration:,}."
    elif not analysis.periods:
        message = "No flat periods match the maximum band width and duration limits."
    else:
        message = (f"{len(analysis.periods):,} nonoverlapping flat periods, newest first; "
                   f"{covered:,} of {analysis.source_count:,} completed bars covered.")
        if current_period is None:
            message += " No flat period ends at the current bar."
    return LogicResult(
        overlays=overlays,
        tables=[
            {"name": "Flat periods", "columns": _PERIOD_COLUMNS, "rows": rows, "default_page": 1},
            {"name": "Flat summary", "columns": list(summary), "rows": [summary]},
        ],
        messages=[message,
                  "Every candle's high and low must fit inside the maximum band width. "
                  "Width is the full price range divided by its midpoint. Each period uses only "
                  "its endpoint's data. The latest-first summary may select "
                  "different historical periods when the review endpoint changes."],
        data={
            "analysis": {
                "text": [], "events": events,
                "layers": [{key: item[key] for key in ("id", "name", "type")} for item in overlays],
            },
            "metadata": {
                "algorithm_version": analysis.algorithm_version,
                "analysis_mode": analysis.analysis_mode,
                "source_count": analysis.source_count,
                "analysis_start": analysis.analysis_start.isoformat() if analysis.analysis_start else None,
                "analysis_end": analysis.analysis_end.isoformat() if analysis.analysis_end else None,
                "config": analysis.config.to_dict(),
            },
        },
    )
