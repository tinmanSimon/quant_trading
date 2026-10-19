"""Bounded chart traces for immutable trend fits in source-bar coordinates.

Descriptors survive window changes: the browser can render a new configuration
against its current window even when the server has an older viewport snapshot.
"""

from collections.abc import Mapping
from hashlib import blake2b
from html import escape
import math

import plotly.graph_objects as go


MAX_TREND_POINTS = 768


def lambda_color(value):
    """Stable numeric-penalty color, independent of direction and result order.

    Hash the exact float rather than its grid position or rounded label, so the
    same numerical lambda keeps its color across endpoints and configurations.
    """
    digest = blake2b(float(value).hex().encode("ascii"), digest_size=8).digest()
    hue = int.from_bytes(digest[:6], "big") / (2**48) * 360
    saturation = 60 + digest[6] / 255 * 20
    lightness = 38 + digest[7] / 255 * 10
    return f"hsl({hue:.6f},{saturation:.3f}%,{lightness:.3f}%)"


def _field(trend, name, default=None):
    return trend.get(name, default) if isinstance(trend, Mapping) else getattr(trend, name, default)


def _stamp(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def trend_descriptors(trends):
    """Serialize fits once, including constant hover content rather than arrays."""
    descriptors = []
    for index, trend in enumerate(trends):
        start, end = _field(trend, "start_index"), _field(trend, "end_index")
        slope, intercept = _field(trend, "slope"), _field(trend, "intercept")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in (start, end, slope, intercept))
                or start < 0 or end < start):
            raise ValueError("Trend overlays require finite fits and ordered source-bar indices.")
        trend_id = str(_field(trend, "trend_id", f"T{index + 1}"))
        direction = "Up" if _field(trend, "direction") == 1 else "Down"
        lambda_values = list(_field(trend, "lambda_values", ()))
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in lambda_values):
            raise ValueError("Trend overlay lambda values must be finite positive numbers.")
        primary_lambda = min(lambda_values, default=None)
        origin = str(_field(trend, "origin", "segmentation"))
        if primary_lambda is not None:
            label = f"λ={primary_lambda:.8g}"
            provenance = ["Segmentation λ values: " + ", ".join(f"{value:.8g}" for value in lambda_values),
                          f"Color λ: {primary_lambda:.8g} (smallest segmentation λ)"]
            line = {"color": lambda_color(primary_lambda), "width": 3, "dash": "solid"}
        else:
            label = "tail" if origin == "tail" else "λ unavailable"
            provenance = [("Tail-discovered · no segmentation λ" if origin == "tail"
                           else "No segmentation λ recorded")]
            line = {"color": "#64748b", "width": 3, "dash": "dash"}
        hover = [f"<b>{escape(trend_id)} · {direction}</b>",
                 *provenance,
                 f"Start: {escape(_stamp(_field(trend, 'start_time')))}",
                 f"End: {escape(_stamp(_field(trend, 'end_time')))}",
                 f"Duration: {_field(trend, 'duration')} bars",
                 f"Clarity: {_field(trend, 'clarity', 0):.3f}",
                 f"Slope (log-price per bar): {slope:+.10g}",
                 f"Fitted rate/bar: {_field(trend, 'rate_per_bar', 0):+.4%}",
                 f"Boundary robustness: {_field(trend, 'boundary_robustness', 0):.1%}",
                 f"Normalized magnitude: {_field(trend, 'magnitude', 0):.3f}",
                 f"Tail: {escape(str(_field(trend, 'tail_status', 'not assessed')).replace('_', ' '))}"]
        tail_start = _field(trend, "tail_start_time")
        if tail_start is not None:
            hover.append(f"Tail start: {escape(_stamp(tail_start))}")
        retracement = _field(trend, "retracement")
        if retracement is not None:
            hover.append(f"Retracement: {retracement:.1%}")
        if _field(trend, "onset_truncated", False):
            hover.append("Start touches available history")
        scores = _field(trend, "scores", ())
        if scores:
            values = (scores.items() if isinstance(scores, Mapping)
                      else zip(("OLS score", "Robust score", "Ordering score", "Efficiency score"), scores))
            hover.extend(f"{escape(str(name))}: {value:+.3f}" for name, value in values)
        descriptors.append({"trend_id": trend_id, "start_index": start, "end_index": end,
                            "slope": slope, "intercept": intercept,
                            "name": f"{trend_id} · {label}", "origin": origin,
                            "lambda_values": lambda_values,
                            "lambda_levels": list(_field(trend, "lambda_levels", ())),
                            "primary_lambda": primary_lambda,
                            "line": line,
                            "hovertemplate": "<br>".join(hover)
                            + "<br>Fitted price: %{y:.6g}<extra></extra>"})
    return descriptors


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
