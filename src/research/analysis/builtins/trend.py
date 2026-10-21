"""Historical trend discovery expressed through the chart-logic contract."""

from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime

import polars as pl

from research.analysis.base import BaseChartLogic, LogicResult
from research.analysis.presentation import trend_descriptors
from research.analysis.trend.selection import (
    filter_trends_by_lambda,
    filter_trends_by_slope,
    select_trends,
)
from research.analysis.trend import ALGORITHM_VERSION, Trend, TrendConfig, analyze


_DEFAULTS = TrendConfig().to_dict()
_REDUNDANCY_FIELDS = (
    "redundancy_percentage", "redundancy_boundary_rule", "redundancy_min_bars",
)
_DISPLAY_FIELDS = (*_REDUNDANCY_FIELDS, "show_redundant", "show_all_lambdas",
                   "selected_lambdas", "include_without_lambda", "slope_mode",
                   "slope_min", "slope_max")
_TITLES = {
    "max_history": "Window advance and overlap (bars)",
    "min_segment_length": "Minimum segment length (bars)",
    "penalty_multiplier": "Starting λ",
    "penalty_grid_step": "Penalty grid step",
    "penalty_grid_levels": "Number of λ levels",
    "weight_ols": "Regression weight",
    "weight_robust": "Robust slope weight",
    "weight_ordering": "Chronological ordering weight",
    "weight_efficiency": "Directional efficiency weight",
    "clarity_threshold": "Minimum clarity",
    "magnitude_threshold": "Minimum normalized slope",
    "boundary_fraction": "Boundary perturbation fraction",
    "boundary_score_threshold": "Boundary evidence threshold",
    "boundary_acceptance": "Required boundary robustness",
    "min_tail_length": "Minimum tail length (bars)",
    "max_tail_fraction": "Maximum tail fraction",
    "tail_fit_threshold": "Tail fit-improvement threshold",
    "tail_slope_threshold": "Tail slope-change threshold",
    "tail_level_threshold": "Tail level-change threshold",
    "slowing_ratio": "Slowing ratio",
    "retracement_threshold": "Material retracement threshold",
    "countertrend_fraction": "Material countertrend duration fraction",
    "shock_threshold": "Young counter-move threshold",
    "redundancy_percentage": "Redundancy tolerance (%)",
    "redundancy_boundary_rule": "Boundary matching rule",
    "redundancy_min_bars": "Minimum redundancy resolution (bars)",
    "dedup_ratio": "Legacy deduplication ratio",
}


def _properties():
    properties = {
        name: {"type": "integer" if type(value) is int else
               "number" if type(value) is float else "string",
               "title": _TITLES[name], "default": value}
        for name, value in _DEFAULTS.items()
    }
    for name in ("min_segment_length", "min_tail_length"):
        properties[name].update(minimum=8, maximum=2000)
    properties["max_history"].update(minimum=8, maximum=100_000)
    properties["max_history"]["description"] = (
        "Analysis windows hold up to twice this many bars and advance by this many bars. "
        "The overlapping half is rebuilt once per shift; each endpoint uses only its "
        "window's data through that bar. This does not limit total historical coverage."
    )
    properties["penalty_multiplier"]["exclusiveMinimum"] = 0
    properties["penalty_grid_step"].update(minimum=0.25, maximum=4)
    properties["penalty_grid_levels"].update(minimum=1, maximum=256)
    for name in ("weight_ols", "weight_robust", "weight_ordering", "weight_efficiency",
                 "clarity_threshold", "boundary_fraction", "boundary_score_threshold",
                 "boundary_acceptance", "tail_fit_threshold", "slowing_ratio",
                 "retracement_threshold", "countertrend_fraction"):
        properties[name].update(minimum=0, maximum=1)
    properties["max_tail_fraction"].update(exclusiveMinimum=0, maximum=0.5)
    for name in ("magnitude_threshold", "tail_slope_threshold", "tail_level_threshold",
                 "shock_threshold"):
        properties[name]["exclusiveMinimum"] = 0
    properties["redundancy_percentage"].update(minimum=0, maximum=100)
    properties["redundancy_boundary_rule"]["enum"] = ["and", "or"]
    properties["redundancy_min_bars"]["minimum"] = 0
    properties["dedup_ratio"]["minimum"] = 1
    properties["dedup_ratio"]["description"] = "Retained for saved-setting compatibility."
    properties["penalty_grid_step"]["description"] = (
        "Each next λ is multiplied by 2 raised to this value. Every endpoint uses the same grid."
    )
    properties["weight_ols"]["description"] = "The four evidence weights must sum to one."
    properties.update({
        "show_redundant": {"type": "boolean", "default": False, "title": "Show redundant trends"},
        "show_all_lambdas": {"type": "boolean", "default": True, "title": "Show all λ values"},
        "selected_lambdas": {
            "type": "array", "items": {"type": "number", "exclusiveMinimum": 0},
            "uniqueItems": True, "default": [], "title": "Selected λ values",
            "description": "Used when Show all λ values is off. Empty selection hides λ-derived fits.",
        },
        "include_without_lambda": {
            "type": "boolean", "default": False, "title": "Include trends without λ",
            "description": "Include tail-derived intervals when filtering by λ.",
        },
        "slope_mode": {
            "type": "string", "enum": ["all", "positive", "negative", "range"],
            "default": "all", "title": "Slope filter",
        },
        "slope_min": {
            "type": "number", "default": -0.01, "title": "Minimum slope",
            "description": "Inclusive log-price slope per source bar; used by the range filter.",
        },
        "slope_max": {
            "type": "number", "default": 0.01, "title": "Maximum slope",
            "description": "Inclusive log-price slope per source bar; used by the range filter.",
        },
    })
    return properties


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _lambda_label(row):
    return "; ".join(
        f"λ={value:.6g} (level {level})"
        for value, level in zip(row["lambda_values"], row["lambda_levels"])
    ) or ("Tail-derived; no λ" if row["origin"] == "tail" else "No λ recorded")


def _archived_trends(result):
    """Restore the immutable engine records needed for display-only selection."""
    trends = []
    for stored in result.data.get("trends", []):
        values = deepcopy(stored)
        for name in ("start_time", "end_time", "tail_start_time"):
            value = values.get(name)
            if isinstance(value, str):
                values[name] = (date.fromisoformat(value) if len(value) == 10
                                else datetime.fromisoformat(value))
        for name in ("scores", "nearby_start_indices", "lambda_values", "lambda_levels"):
            values[name] = tuple(values.get(name, ()))
        trends.append(Trend(**values))
    return tuple(trends)


class TrendLogic(BaseChartLogic):
    name = "trend"
    version = f"1.0.0-engine.{ALGORITHM_VERSION}"
    result_version = "2"
    label = "Trend analysis"
    description = (
        "Analyze every eligible historical endpoint using original, completed bars. "
        "Clarity describes directional evidence, not probability or future continuation."
    )
    required_columns = ("timestamp", "close")
    input_policy = "completed"
    config_schema = {
        "type": "object",
        "properties": {name: field for name, field in _properties().items()
                       if name not in _DISPLAY_FIELDS},
        "additionalProperties": False,
        "x-ui-groups": [
            {"title": "Interval discovery", "expanded": True,
             "fields": ["max_history", "min_segment_length", "penalty_multiplier",
                        "penalty_grid_step", "penalty_grid_levels"]},
            {"title": "Acceptance", "expanded": True,
             "fields": ["clarity_threshold", "magnitude_threshold"]},
            {"title": "Evidence weights", "expanded": False,
             "fields": ["weight_ols", "weight_robust", "weight_ordering", "weight_efficiency"]},
            {"title": "Boundary robustness", "expanded": False,
             "fields": ["boundary_fraction", "boundary_score_threshold", "boundary_acceptance"]},
            {"title": "Tail diagnosis", "expanded": False,
             "fields": ["min_tail_length", "max_tail_fraction", "tail_fit_threshold",
                        "tail_slope_threshold", "tail_level_threshold", "slowing_ratio",
                        "retracement_threshold", "countertrend_fraction", "shock_threshold"]},
            {"title": "Legacy compatibility", "expanded": False, "fields": ["dedup_ratio"]},
        ],
    }

    display_schema = {
        "type": "object",
        "properties": {name: field for name, field in _properties().items()
                       if name in _DISPLAY_FIELDS},
        "additionalProperties": False,
        "x-ui-groups": [
            {"title": "Trend visibility", "expanded": True,
             "fields": ["show_all_lambdas", "selected_lambdas", "include_without_lambda",
                        "slope_mode", "slope_min", "slope_max"]},
            {"title": "Redundancy", "expanded": False,
             "fields": [*_REDUNDANCY_FIELDS, "show_redundant"]},
        ],
    }

    def validate(self, config):
        values = {**{name: field["default"] for name, field in self.config_schema["properties"].items()},
                  **config}
        TrendConfig.from_dict(values)
        return values

    def display_schema_for(self, result):
        schema = deepcopy(self.display_schema)
        trends = _archived_trends(result)
        lambdas = sorted({value for trend in trends for value in trend.lambda_values})
        selected = schema["properties"]["selected_lambdas"]
        selected["x-ui-options"] = lambdas
        if lambdas:
            selected["items"]["enum"] = lambdas
        selected["description"] = (
            "Choose from λ values found in the accepted results. Used when Show all λ values "
            "is off; an empty selection hides λ-derived fits."
        )
        if trends:
            schema["properties"]["slope_min"]["default"] = min(trend.slope for trend in trends)
            schema["properties"]["slope_max"]["default"] = max(trend.slope for trend in trends)
        return schema

    def validate_display(self, config, result):
        schema = self.display_schema_for(result)
        values = {**{name: field["default"] for name, field in schema["properties"].items()},
                  **config}
        select_trends((), **{name: values[name] for name in _REDUNDANCY_FIELDS})
        filter_trends_by_slope((), values["slope_mode"],
                              minimum=values["slope_min"], maximum=values["slope_max"])
        return values

    def calculate(self, context, config, progress=None):
        values = self.validate(config)
        settings = TrendConfig.from_dict(values)
        frame = context.completed_frame
        result = analyze(frame, as_of=context.cutoff, config=settings, progress=progress)
        output = self.from_analysis(result, values)
        excluded = context.frame.height - frame.height
        if excluded:
            output.messages.append(f"{excluded:,} possibly incomplete source bar(s) were excluded from trend analysis.")
        return output

    def from_analysis(self, result, config):
        """Adapt a verified saved engine result without rerunning discovery."""
        values = self.validate(config)
        if any(result.config.to_dict()[name] != value for name, value in values.items()):
            raise ValueError("Saved trend analysis settings do not match this logic configuration.")
        overlays = [
            {"type": "log_line", "id": descriptor["trend_id"], **descriptor}
            for descriptor in trend_descriptors(result.trends)
        ]
        accepted = []
        for row in result.to_frame().to_dicts():
            row["lambda_label"] = _lambda_label(row)
            row["rate_per_bar"] *= 100
            row["on_chart"] = True
            row["status"] = "On chart"
            row["redundant_to"] = None
            accepted.append(_json_value(row))
        diagnostics = result.diagnostics_frame()
        uncertain = []
        if not diagnostics.is_empty():
            for row in diagnostics.filter(~pl.col("accepted")).to_dicts():
                row["lambda_label"] = _lambda_label(row)
                row["status"] = "Uncertain"
                uncertain.append(_json_value(row))
        messages = [
            "Each historical interval uses only bars through its own endpoint. "
            "Start is an estimated onset, not a first-recognition date. "
            "Clarity describes directional evidence, not probability.",
            "Line color identifies the smallest numeric λ that discovered the interval. "
            "Tail-derived intervals have no λ and use a gray dashed line. "
            "The table's rate_per_bar is fitted percent per bar.",
        ]
        if result.analysis_end is not None:
            messages.append(
                f"Analysis range: {result.analysis_start.isoformat()} to {result.analysis_end.isoformat()}. "
                f"Algorithm version: {result.algorithm_version}."
            )
        if result.status == "insufficient_data":
            messages.append("Insufficient completed bars for trend analysis with these settings.")
        elif not result.trends:
            messages.append("No clear trend found for this interval.")
        return LogicResult(overlays=overlays, tables=[
            {"name": "Accepted trends", "rows": accepted},
            {"name": "Uncertain intervals", "rows": uncertain},
        ], messages=messages, data={"trends": [_json_value(asdict(trend)) for trend in result.trends]})

    def present(self, result, config):
        """Select stored fits without discovering, fitting, or deleting any trend."""
        values = self.validate_display(config, result)
        trends = _archived_trends(result)
        selection = select_trends(trends, **{name: values[name] for name in _REDUNDANCY_FIELDS})
        eligible = trends if values["show_redundant"] else selection.chart_trends
        visible = filter_trends_by_lambda(
            eligible, None if values["show_all_lambdas"] else values["selected_lambdas"],
            include_without_lambda=values["include_without_lambda"],
        )
        visible = filter_trends_by_slope(
            visible, values["slope_mode"], minimum=values["slope_min"], maximum=values["slope_max"],
        )
        visible_ids = {trend.trend_id for trend in visible}
        output = deepcopy(result)
        output.overlays = [overlay for overlay in output.overlays if overlay["id"] in visible_ids]
        uncertain_count = 0
        for table in output.tables:
            if table["name"] == "Accepted trends":
                for row in table["rows"]:
                    trend_id = row["trend_id"]
                    row["on_chart"] = trend_id in visible_ids
                    row["status"] = ("Redundant" if trend_id in selection.redundant_to else
                                     "On chart" if trend_id in visible_ids else "Filtered out")
                    row["redundant_to"] = selection.redundant_to.get(trend_id)
            elif table["name"] == "Uncertain intervals":
                uncertain_count = len(table["rows"])
        output.messages.insert(
            0, f"{len(visible):,} trend lines match the display filters; {len(selection.redundant_to):,} redundant intervals; "
            f"{uncertain_count:,} uncertain candidates. All accepted intervals remain in the table.",
        )
        if trends and not visible:
            output.messages.append(
                "No trends match the current display filters. Enable Show all λ values or choose "
                "λ values from the suggestions, and check the slope and redundancy filters."
            )
        return output
