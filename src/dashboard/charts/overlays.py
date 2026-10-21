"""Generic, bounded chart primitives in original source-bar coordinates.

Logic modules return plain descriptors; this adapter alone knows about Plotly.
Numeric arrays are clipped and reduced for transport, while cached calculations
remain unchanged on the server. Null values explicitly separate line segments.
Styles and hover templates are constant per primitive. Per-point marker style
arrays (including nested marker-line arrays) are deliberately rejected; use
separate primitives for independently styled groups. This keeps every array on
the wire governed by the coordinate sampling budget.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
import math

import plotly.graph_objects as go
from research.analysis.base import validate_panel

from .trend_overlays import MAX_TREND_POINTS, trend_descriptors


MAX_OVERLAY_POINTS = 2048
PRIMITIVE_TYPES = frozenset({"line", "band", "markers", "bar", "log_line"})


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _has_style_array(value):
    if isinstance(value, (list, tuple)):
        return True
    if isinstance(value, Mapping):
        return any(_has_style_array(item) for item in value.values())
    return False


def validate_overlay(descriptor):
    """Validate a plugin output before its calculation is published."""
    if not isinstance(descriptor, Mapping) or descriptor.get("type") not in PRIMITIVE_TYPES:
        raise ValueError("Chart output requires a supported primitive type.")
    if not isinstance(descriptor.get("id"), str) or not descriptor["id"].strip():
        raise ValueError("Chart primitives require a nonempty ID.")
    kind = descriptor["type"]
    if descriptor.get("panel") is not None:
        validate_panel(descriptor["panel"])
    if kind == "log_line":
        values = [descriptor.get(key) for key in ("start_index", "end_index", "slope", "intercept")]
        if (not all(_finite(value) for value in values)
                or values[0] < 0 or values[1] < values[0]):
            raise ValueError("Log-line primitives require finite fits and ordered source-bar indices.")
        try:
            prices = [math.exp(values[3]), math.exp(values[3] + values[2] * (values[1] - values[0]))]
        except OverflowError as exc:
            raise ValueError("Log-line prices must remain finite.") from exc
        if not all(math.isfinite(price) for price in prices):
            raise ValueError("Log-line prices must remain finite.")
    else:
        positions = descriptor.get("x")
        if not isinstance(positions, (list, tuple)) or not all(_finite(value) for value in positions):
            raise ValueError("Chart primitive x values must be finite source-bar coordinates.")
        if any(right < left for left, right in zip(positions, positions[1:])):
            raise ValueError("Chart primitive x values must be increasing.")
        keys = ("lower", "upper") if kind == "band" else ("y",)
        for key in keys:
            values = descriptor.get(key)
            if (not isinstance(values, (list, tuple)) or len(values) != len(positions)
                    or any(value is not None and not _finite(value) for value in values)):
                raise ValueError(f"Chart primitive {key} values must match x and be finite numbers or null.")
        if kind == "band":
            for lower, upper in zip(descriptor["lower"], descriptor["upper"]):
                if (lower is None) != (upper is None) or (
                        lower is not None and lower > upper):
                    raise ValueError("Band boundaries must share gaps and have lower <= upper.")
    if _has_style_array(descriptor.get("marker")):
        raise ValueError("Chart marker styles must use scalar values; use separate primitives for varying styles.")
    if descriptor.get("hovertemplate") is not None and not isinstance(descriptor["hovertemplate"], str):
        raise ValueError("Chart hover templates must be one string per primitive.")
    # Use Plotly's validators once here, rather than letting malformed plugin
    # styles break the shared browser chart after a successful calculation.
    if kind == "bar":
        go.Bar(name=descriptor.get("name"), hovertemplate=descriptor.get("hovertemplate"),
               marker=descriptor.get("marker"))
    else:
        go.Scatter(name=descriptor.get("name"), hovertemplate=descriptor.get("hovertemplate"),
                   line=descriptor.get("line"), marker=descriptor.get("marker"),
                   fillcolor=descriptor.get("fillcolor"))
    return descriptor


def overlay_panels(descriptors):
    """Return consistent panel declarations in first-use order."""
    panels = {}
    for descriptor in descriptors:
        panel = descriptor.get("panel")
        if panel is None:
            continue
        validate_panel(panel)
        if panel["id"] in panels and panels[panel["id"]] != panel:
            raise ValueError("Primitives in the same panel must use the same panel declaration.")
        panels[panel["id"]] = panel
    return list(panels.values())


def legacy_overlays(trends):
    """Adapt existing fit results without losing hover or lambda provenance."""
    return [{"type": "log_line", "id": f"legacy-trend:{index}:{item['trend_id']}", **item}
            for index, item in enumerate(trend_descriptors(trends))]


def _selected_indices(positions, arrays, limit):
    """Bound samples, retaining per-bucket extrema and explicit breaks."""
    count = len(positions)
    if count <= limit:
        return list(range(count))
    # Each bucket contributes its ends, all value extrema, and a gap marker.
    # Include the first gap and suppress any connection across skipped gaps.
    bucket_count = max(1, (limit - 2) // (3 + 2 * len(arrays)))
    chosen = {0, count - 1}
    for bucket in range(bucket_count):
        first = bucket * count // bucket_count
        end = (bucket + 1) * count // bucket_count
        chosen.update((first, end - 1))
        for values in arrays:
            valid = [index for index in range(first, end) if values[index] is not None]
            if valid:
                chosen.add(min(valid, key=values.__getitem__))
                chosen.add(max(valid, key=values.__getitem__))
        gap = next((index for index in range(first, end) if arrays[0][index] is None), None)
        if gap is not None:
            chosen.add(gap)
    return sorted(chosen)


def window_overlays(descriptors, loaded_range):
    """Return bounded wire descriptors; compact analytic fits need no slicing."""
    descriptors = list(descriptors)
    overlay_panels(descriptors)
    output, seen = [], set()
    for source in descriptors:
        validate_overlay(source)
        if source["id"] in seen:
            raise ValueError("Chart primitive IDs must be unique.")
        seen.add(source["id"])
        descriptor = dict(source)
        if descriptor["type"] == "log_line":
            output.append(descriptor)
            continue
        positions = source["x"]
        keys = ("lower", "upper") if source["type"] == "band" else ("y",)
        first = bisect_left(positions, loaded_range[0])
        end = bisect_right(positions, loaded_range[1])
        # Keep adjacent points so lines crossing a fractional boundary can
        # interpolate there. Markers must only include actual in-window points.
        if source["type"] not in ("markers", "bar") and positions:
            first, end = max(0, first - 1), min(len(positions), end + 1)
        clipped_x = list(positions[first:end])
        arrays = [list(source[key][first:end]) for key in keys]
        chosen = _selected_indices(clipped_x, arrays, MAX_OVERLAY_POINTS)
        descriptor["x"] = [clipped_x[index] for index in chosen]
        for key, values in zip(keys, arrays):
            descriptor[key] = [values[index] for index in chosen]
        # Never join disconnected sessions just because intermediate break
        # points were discarded by the bounded display reduction.
        if source["type"] not in ("markers", "bar"):
            previous = None
            for offset, index in enumerate(chosen):
                if previous is not None and any(value is None for value in arrays[0][previous + 1:index]):
                    for key in keys:
                        descriptor[key][offset] = None
                previous = index
        descriptor["sampled"] = source.get("sampled", False) or len(chosen) < len(clipped_x)
        descriptor["source_range"] = source.get("source_range") or ([positions[0], positions[-1]] if positions else None)
        output.append(descriptor)
    return output


def _clip_points(descriptor, limits, keys):
    """Clip polylines at fractional edges, preserving null separators."""
    positions = descriptor["x"]
    output_x, output_y = [], [[] for _ in keys]
    if not positions:
        return output_x, output_y
    first = bisect_left(positions, limits[0])
    end = bisect_right(positions, limits[1])
    arrays = [descriptor[key] for key in keys]

    def edge(left, right, position):
        if left < 0 or right >= len(positions) or positions[left] == positions[right]:
            return
        values = []
        for array in arrays:
            a, b = array[left], array[right]
            values.append(None if a is None or b is None else
                          a + (b - a) * (position - positions[left]) / (positions[right] - positions[left]))
        output_x.append(position)
        for target, value in zip(output_y, values):
            target.append(value)

    if first < len(positions) and positions[first] > limits[0]:
        edge(first - 1, first, limits[0])
    for index in range(first, end):
        output_x.append(positions[index])
        for target, array in zip(output_y, arrays):
            target.append(array[index])
    if end and positions[end - 1] < limits[1]:
        edge(end - 1, end, limits[1])
    return output_x, output_y


def overlay_traces(descriptors, loaded_range):
    """Render primitives on the price axes or their declared indicator panel."""
    descriptors = list(descriptors)
    panel_axes = {panel["id"]: index + 3 for index, panel in enumerate(overlay_panels(descriptors))}
    traces = []
    for descriptor in descriptors:
        kind = descriptor["type"]
        meta = {"chart_overlay": True, "overlay_id": descriptor["id"], "primitive": kind}
        if descriptor.get("trend_id"):
            meta.update({key: descriptor.get(key) for key in (
                "trend_id", "lambda_values", "lambda_levels", "primary_lambda", "origin")})
            meta["trend_overlay"] = True
        panel = descriptor.get("panel")
        suffix = str(panel_axes[panel["id"]]) if panel else ""
        if panel:
            meta["panel_id"] = panel["id"]
        common = dict(name=descriptor.get("name", descriptor["id"]), xaxis="x" + suffix, yaxis="y" + suffix,
                      showlegend=False, meta=meta, hovertemplate=descriptor.get("hovertemplate"),
                      line=descriptor.get("line"), connectgaps=False)
        if kind == "log_line":
            first = max(descriptor["start_index"], loaded_range[0])
            last = min(descriptor["end_index"], loaded_range[1])
            if last < first:
                continue
            size = min(MAX_TREND_POINTS, max(2, math.ceil(last - first) + 1))
            x = [first + (last - first) * index / (size - 1) for index in range(size)]
            y = [math.exp(descriptor["intercept"] + descriptor["slope"]
                          * (position - descriptor["start_index"])) for position in x]
            traces.append(go.Scatter(x=x, y=y, mode="lines", **common))
        elif kind in ("markers", "bar"):
            indices = [index for index, position in enumerate(descriptor["x"])
                       if loaded_range[0] <= position <= loaded_range[1]]
            if indices:
                coordinates = dict(x=[descriptor["x"][index] for index in indices],
                                   y=[descriptor["y"][index] for index in indices],
                                   marker=descriptor.get("marker"))
                if kind == "bar":
                    traces.append(go.Bar(**coordinates, width=0.65,
                                         **{key: value for key, value in common.items()
                                            if key not in ("line", "connectgaps")}))
                else:
                    traces.append(go.Scatter(**coordinates, mode="markers", **common))
        else:
            keys = ("lower", "upper") if kind == "band" else ("y",)
            x, values = _clip_points(descriptor, loaded_range, keys)
            if not x or not any(value is not None for value in values[0]):
                continue
            if kind == "band":
                traces.append(go.Scatter(x=x, y=values[0], mode="lines", **common))
                traces.append(go.Scatter(x=x, y=values[1], mode="lines", fill="tonexty",
                                         fillcolor=descriptor.get("fillcolor", "rgba(96,165,250,0.14)"),
                                         **{**common, "meta": {**meta, "band_upper": True}}))
            else:
                traces.append(go.Scatter(x=x, y=values[0], mode="lines", **common))
    # Plotly omits unused axes. Keep an empty panel alive during warmup and
    # while newer indicator samples are being fetched for the current viewport.
    present = {trace.meta.get("panel_id") for trace in traces}
    for panel_id, number in panel_axes.items():
        if panel_id not in present:
            traces.append(go.Scatter(x=[loaded_range[0]], y=[None], mode="lines",
                xaxis=f"x{number}", yaxis=f"y{number}", showlegend=False, hoverinfo="skip",
                meta={"chart_overlay": True, "panel_id": panel_id, "panel_placeholder": True}))
    return traces
