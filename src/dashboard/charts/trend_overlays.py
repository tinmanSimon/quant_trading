"""Bounded Plotly traces for canonical research trend descriptors."""
import math
import plotly.graph_objects as go
from research.analysis.trend.presentation import lambda_color, trend_descriptors

MAX_TREND_POINTS = 768

def trend_traces(descriptors, loaded_range):
    """Sample the unchanged log-linear fit only inside the loaded chart window."""
    traces = []
    for descriptor in descriptors:
        first = max(descriptor["start_index"], loaded_range[0])
        last = min(descriptor["end_index"], loaded_range[1])
        if last < first:
            continue
        count = min(MAX_TREND_POINTS, max(2, math.ceil(last - first) + 1))
        positions = [first + (last - first) * index / (count - 1) for index in range(count)]
        prices = [math.exp(descriptor["intercept"]
                           + descriptor["slope"] * (position - descriptor["start_index"]))
                  for position in positions]
        traces.append(go.Scatter(
            x=positions, y=prices, mode="lines", name=descriptor.get("name", descriptor["trend_id"]),
            line=descriptor["line"], hovertemplate=descriptor["hovertemplate"],
            xaxis="x", yaxis="y", showlegend=False,
            meta={"trend_overlay": True, "trend_id": descriptor["trend_id"],
                  "lambda_values": descriptor["lambda_values"],
                  "lambda_levels": descriptor["lambda_levels"],
                  "primary_lambda": descriptor["primary_lambda"], "origin": descriptor["origin"]},
        ))
    return traces
