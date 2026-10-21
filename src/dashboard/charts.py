"""Plot original bars or explicitly labelled display summaries without gap filling.

Candles use consecutive bar positions to compress periods without data.
Intraday timestamps are explicitly
converted to the selected zone before serializing those labels; UTC is retained
in hover text. Session-date labels never undergo timezone conversion.
"""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from html import escape
from zoneinfo import ZoneInfo

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .chart_overlays import overlay_panels, overlay_traces, window_overlays


CHART_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
}


def display_timestamps(
    timestamps: Iterable[datetime], *, timeframe: str, timezone: str,
) -> list[str]:
    """Return explicit display labels while leaving stored timestamps untouched."""
    if timeframe.endswith(("d", "wk", "mo")):
        return [stamp.date().isoformat() for stamp in timestamps]
    zone = ZoneInfo(timezone)
    labels = []
    instants = {}
    previous = None
    for stamp in timestamps:
        label = stamp.astimezone(zone).replace(tzinfo=None).isoformat()
        instant = stamp.astimezone(UTC)
        if ((label in instants and instants[label] != instant)
                or (previous is not None and instant > previous[0] and label < previous[1])):
            raise ValueError("This display timezone repeats or reverses clock times across a daylight-saving transition. "
                             "Select UTC to show these bars at distinct instants.")
        instants[label] = instant
        labels.append(label)
        previous = (instant, label)
    return labels


def bar_table(frame: pl.DataFrame, *, timeframe: str, timezone: str) -> pl.DataFrame:
    """Add human-readable time columns; keep all original OHLCV values."""
    stamps = frame["timestamp"].to_list()
    return frame.with_columns(
        pl.Series("timestamp_utc", [stamp.isoformat() for stamp in stamps]),
        pl.Series("display_time", display_timestamps(stamps, timeframe=timeframe, timezone=timezone)),
    ).select("display_time", "timestamp_utc", pl.exclude("timestamp", "display_time", "timestamp_utc"))


def price_chart(
    frame: pl.DataFrame, *, timeframe: str, timezone: str = "America/New_York",
    title: str = "", omitted_timestamps: Iterable[datetime] = (),
) -> go.Figure:
    """Keep every bar available; open the view on the latest 50 or fewer."""
    stamps = frame["timestamp"].to_list()
    if not stamps:
        raise ValueError("A price chart requires at least one bar.")
    if any(current <= previous for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("Price chart bars must have unique, increasing timestamps.")
    labels = display_timestamps(stamps, timeframe=timeframe, timezone=timezone)
    positions = list(range(len(stamps)))
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.76, 0.24],
    )
    hover = [f"{label}<br>Stored timestamp: {stamp.isoformat()}" for label, stamp in zip(labels, stamps)]
    figure.add_trace(
        go.Candlestick(
            x=positions, open=frame["open"].to_list(), high=frame["high"].to_list(),
            low=frame["low"].to_list(), close=frame["close"].to_list(),
            text=hover, name="OHLC", increasing_line_color="#18a999",
            decreasing_line_color="#e76f51",
            customdata=frame["volume"].to_list(),
            hovertemplate=("%{text}<br>Open: %{open}<br>High: %{high}<br>Low: %{low}"
                           "<br>Close: %{close}<br>Volume: %{customdata}<extra></extra>"),
        ), row=1, col=1,
    )
    figure.add_trace(
        go.Bar(
            x=positions, y=frame["volume"].to_list(), name="Volume", text=hover,
            textposition="none", width=0.65,
            hovertemplate="%{text}<br>Volume: %{y}<extra></extra>",
            marker_color=["#18a999" if close >= opened else "#e76f51"
                          for close, opened in zip(frame["close"], frame["open"])],
        ), row=2, col=1,
    )
    gaps = defaultdict(list)
    present = set(stamps)
    for stamp in sorted(set(omitted_timestamps) - present):
        # Group omissions at the boundary between existing bars, including
        # the outer edges. Never create an extra candle slot for a missing bar.
        boundary = bisect_left(stamps, stamp) - 0.5
        label = display_timestamps([stamp], timeframe=timeframe, timezone=timezone)[0]
        gaps[boundary].append(f"{label} (UTC: {stamp.isoformat()})")
    for boundary, missing in gaps.items():
        figure.add_shape(
            type="line", x0=boundary, x1=boundary, y0=0, y1=1, xref="x", yref="paper",
            line={"color": "#e9a23b", "width": 1, "dash": "dot"},
        )
        figure.add_annotation(
            x=boundary, y=1, xref="x", yref="paper", text=f"Missing: {len(missing)}",
            hovertext="<br>".join(missing),
            showarrow=False, yanchor="bottom", font={"color": "#b87913", "size": 10},
        )
    date_labels = timeframe.endswith(("d", "wk", "mo"))
    figure.update_layout(
        title=title, height=650, margin={"l": 85, "r": 25, "t": 70, "b": 20},
        dragmode="zoom", hovermode="closest", showlegend=False,
        xaxis_rangeslider_visible=False,
        meta={"bar_labels": labels},
    )
    first_visible = max(0, len(stamps) - 50)
    ticks = positions[first_visible::max(1, (min(len(positions), 50) + 7) // 8)]
    figure.update_xaxes(type="linear", fixedrange=False, range=[first_visible - 0.5, len(stamps) - 0.5],
                        minallowed=-0.5, maxallowed=len(stamps) - 0.5,
                        tickmode="array", tickvals=ticks, ticktext=[labels[index] for index in ticks])
    figure.update_xaxes(
        title_text=("Session date" if date_labels else f"Bar start ({timezone})") + " · gaps compressed",
        rangeslider={"visible": True, "thickness": 0.09, "range": [-0.5, len(stamps) - 0.5]}, row=2, col=1,
    )
    # Leave room for complete numbers and separate titles; let long labels grow
    # the margin instead of clipping leading digits. Tick precision stays adaptive.
    figure.update_yaxes(automargin=True, title_standoff=12, tickfont_size=12, ticks="outside")
    figure.update_yaxes(title_text="Price", fixedrange=True, nticks=6,
                        exponentformat="none", separatethousands=True, row=1, col=1)
    figure.update_yaxes(title_text="Volume", fixedrange=True, rangemode="tozero", nticks=4, row=2, col=1)
    return figure


def line_chart(
    frame: pl.DataFrame, *, columns: list[str], title: str, timeframe: str,
    timezone: str = "UTC", y_title: str = "",
) -> go.Figure:
    """Research result curves without changing or resampling their values."""
    figure = go.Figure()
    labels = display_timestamps(frame["timestamp"].to_list(), timeframe=timeframe, timezone=timezone)
    for column in columns:
        figure.add_trace(go.Scatter(x=labels, y=frame[column].to_list(), mode="lines", name=column))
    figure.update_layout(
        title=title, height=360, dragmode="zoom", hovermode="x unified",
        xaxis={"type": "date", "rangeslider": {"visible": True, "thickness": 0.1}},
        yaxis_title=y_title, margin={"l": 20, "r": 20, "t": 45, "b": 20},
    )
    return figure


def _configure_indicator_panels(figure, panels):
    """Save structural layout separately so display changes can survive old viewports."""
    rows = len(panels) + 2
    title = figure.layout.xaxis2.title.to_plotly_json()
    structure = {"height": 650 + 190 * len(panels), "axes": {}, "panels": [], "annotations": []}
    if panels:
        heights = [360, 110] + [172] * len(panels)
        total = sum(heights) + 18 * (rows - 1)
        top = 1.0
        domains = []
        for height in heights:
            domains.append([max(0, top - height / total), top])
            top -= (height + 18) / total
    else:
        domains = [list(figure.layout.yaxis.domain), list(figure.layout.yaxis2.domain)]
    for index, domain in enumerate(domains):
        suffix = str(index + 1) if index else ""
        bottom = index == rows - 1
        x = {"domain": [0, 1], "anchor": "y" + suffix, "matches": "x" if index else None,
             "showticklabels": bottom, "title": title if bottom else {"text": ""},
             "rangeslider": {"visible": False}}
        y = {"domain": domain, "anchor": "x" + suffix}
        if index >= 2:
            panel = panels[index - 2]
            x.update(type="linear", fixedrange=False, automargin=True,
                     minallowed=figure.layout.xaxis.minallowed, maxallowed=figure.layout.xaxis.maxallowed)
            y.update(automargin=True, fixedrange=True, nticks=4, ticks="outside", autorange=False,
                     range=panel.get("range", [0, 1]), zeroline=True)
            structure["panels"].append({**panel, "xaxis": "xaxis" + suffix, "yaxis": "yaxis" + suffix})
            structure["annotations"].append({"name": "indicator-panel:" + panel["id"],
                "xref": "paper", "yref": "paper", "x": 0, "y": domain[1],
                "text": escape(panel["title"]), "showarrow": False,
                "xanchor": "left", "yanchor": "bottom", "font": {"size": 12}})
        structure["axes"]["xaxis" + suffix] = x
        structure["axes"]["yaxis" + suffix] = y
    figure.update_layout(**structure["axes"], height=structure["height"], barmode="overlay")
    figure.update_xaxes(range=figure.layout.xaxis.range, autorange=False)
    for annotation in structure["annotations"]:
        figure.add_annotation(**annotation)
    figure.layout.meta = {**figure.layout.meta, "panel_layout": structure}


def window_price_chart(window, *, timeframe, timezone="America/New_York", title="",
                       gap_markers=(), trend_overlays=(), overlays=()):
    """Build only the bounded display window, using global source-bar positions."""
    figure = price_chart(window.frame, timeframe=timeframe, timezone=timezone, title=title)
    labels = display_timestamps(window.starts, timeframe=timeframe, timezone=timezone)
    end_labels = display_timestamps(window.ends, timeframe=timeframe, timezone=timezone)
    hover = []
    for index, (start, end, count) in enumerate(zip(labels, end_labels, window.source_counts)):
        text = (f"{start}<br>Stored timestamp: {window.starts[index].isoformat()}" if count == 1
                else f"Display summary: {start} through {end}<br>Source bars: {count}"
                     f"<br>First/last source timestamps (UTC): {window.starts[index].isoformat()}"
                     f" through {window.ends[index].isoformat()}")
        text += f"<br>Recorded omissions within summary: {window.omitted_counts[index]}" if count > 1 else ""
        hover.append(text)
    figure.data[0].x = window.positions
    figure.data[0].text = hover
    figure.data[1].x = window.positions
    figure.data[1].text = hover
    figure.data[1].width = [width * 0.65 for width in window.widths]
    figure.update_xaxes(range=window.view_range, minallowed=-0.5, maxallowed=window.total_count - 0.5,
                        tickvals=window.positions[::max(1, len(labels) // 8)],
                        ticktext=labels[::max(1, len(labels) // 8)], rangeslider_visible=False)
    figure.update_layout(meta={"bar_labels": labels}, uirevision="window", margin_b=90)
    figure.update_xaxes(automargin=True, title_standoff=16)
    # Marker payload is bounded too. Complete provenance remains available in
    # the on-demand omission table, even if a window has thousands of omissions.
    for position, count, examples in gap_markers:
        figure.add_shape(type="line", x0=position, x1=position, y0=0, y1=1, xref="x", yref="paper",
                         line={"color": "#e9a23b", "width": 1, "dash": "dot"})
        figure.add_annotation(x=position, y=1, xref="x", yref="paper", text=f"Missing: {count}",
                              hovertext="<br>".join(examples), showarrow=False, yanchor="bottom",
                              font={"color": "#b87913", "size": 10})
    primitives = [{"type": "log_line", "id": f"legacy-trend:{index}:{item['trend_id']}", **item}
                  for index, item in enumerate(trend_overlays)] + list(overlays)
    _configure_indicator_panels(figure, overlay_panels(primitives))
    figure.add_traces(overlay_traces(window_overlays(primitives, window.loaded_range), window.loaded_range))
    return figure
