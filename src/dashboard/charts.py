"""Plot original bars without resampling, filling, or changing numeric values.

Candles use consecutive bar positions to compress periods without data.
Intraday timestamps are explicitly
converted to the selected zone before serializing those labels; UTC is retained
in hover text. Session-date labels never undergo timezone conversion.
"""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            x=positions, y=frame["volume"].to_list(), name="Volume", text=hover, width=0.65,
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
