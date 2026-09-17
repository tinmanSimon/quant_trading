"""Plot original bars without resampling, filling, or changing numeric values.

Plotly date axes use wall-clock labels. Intraday timestamps are explicitly
converted to the selected zone before serializing those labels; UTC is retained
in hover text. Session-date labels never undergo timezone conversion.
"""

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
    """Candlesticks and volume using every supplied bar, including partial bars."""
    stamps = frame["timestamp"].to_list()
    labels = display_timestamps(stamps, timeframe=timeframe, timezone=timezone)
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.76, 0.24],
    )
    hover = [f"Stored timestamp: {stamp.isoformat()}" for stamp in stamps]
    figure.add_trace(
        go.Candlestick(
            x=labels, open=frame["open"].to_list(), high=frame["high"].to_list(),
            low=frame["low"].to_list(), close=frame["close"].to_list(),
            text=hover, name="OHLC", increasing_line_color="#18a999",
            decreasing_line_color="#e76f51",
        ), row=1, col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels, y=frame["volume"].to_list(), name="Volume", text=hover,
            marker_color=["#18a999" if close >= opened else "#e76f51"
                          for close, opened in zip(frame["close"], frame["open"])],
        ), row=2, col=1,
    )
    for label in display_timestamps(omitted_timestamps, timeframe=timeframe, timezone=timezone):
        # Shapes mark an absent bar; no fabricated OHLC/volume values are added.
        figure.add_shape(
            type="line", x0=label, x1=label, y0=0, y1=1, xref="x", yref="paper",
            line={"color": "#e9a23b", "width": 1, "dash": "dot"},
        )
        figure.add_annotation(
            x=label, y=1, xref="x", yref="paper", text="Missing bar",
            showarrow=False, yanchor="bottom", font={"color": "#b87913", "size": 10},
        )
    date_labels = timeframe.endswith(("d", "wk", "mo"))
    figure.update_layout(
        title=title, height=650, margin={"l": 20, "r": 20, "t": 70, "b": 20},
        dragmode="zoom", hovermode="x unified", showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    figure.update_xaxes(type="date", fixedrange=False)
    figure.update_xaxes(
        title_text="Session date" if date_labels else f"Bar start ({timezone})",
        rangeslider={"visible": True, "thickness": 0.09}, row=2, col=1,
    )
    figure.update_yaxes(title_text="Price", fixedrange=False, row=1, col=1)
    figure.update_yaxes(title_text="Volume", fixedrange=False, rangemode="tozero", row=2, col=1)
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
