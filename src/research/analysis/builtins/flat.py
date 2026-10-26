"""Shared flat-period calculation for Market Data and batch Analysis."""

from copy import deepcopy

from research.analysis.core.base import BaseChartLogic
from research.analysis.flat import ALGORITHM_VERSION, FlatConfig, analyze
from research.analysis.flat.presentation import summary_result


class FlatLogic(BaseChartLogic):
    name = "flat"
    version = f"1.0.0-engine.{ALGORITHM_VERSION}"
    result_version = "2"
    label = "Flat analysis"
    description = (
        "Find nonoverlapping flat periods, newest first, using completed bars. "
        "Extend sideways while every candle's high and low stays inside a narrow horizontal band."
    )
    required_columns = ("timestamp", "high", "low")
    input_policy = "completed"
    config_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "min_duration": {
                "type": "integer", "minimum": 2, "default": 20,
                "title": "Minimum duration (bars)",
                "description": "Observed bars, including both endpoints. Gaps do not add bars.",
            },
            "max_duration": {
                "type": "integer", "minimum": 3, "default": 200,
                "title": "Maximum duration (bars)",
                "description": "Inclusive maximum period length. Must be greater than the minimum duration.",
            },
            "max_range_pct": {
                "type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 200, "default": 2.0,
                "title": "Maximum band width (%)",
                "description": (
                    "Total highest-high to lowest-low range as a percentage of their midpoint. "
                    "A 2% band around 100 spans 99–101. Every candle, including its wicks, must fit."
                ),
            },
        },
    }
    display_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "show_fit": {"type": "boolean", "default": True, "title": "Show flat bands"},
            "show_boundaries": {"type": "boolean", "default": True, "title": "Show period boundaries"},
        },
    }

    def validate(self, config):
        return FlatConfig.from_dict(config).to_dict()

    def calculate(self, context, config, progress=None):
        analysis = analyze(context.completed_frame, config=FlatConfig.from_dict(config), progress=progress)
        return summary_result(analysis)

    def present(self, result, config):
        values = {"show_fit": True, "show_boundaries": True, **config}
        output = deepcopy(result)
        output.overlays = [item for item in output.overlays if
                           values["show_fit" if item["type"] == "log_line" else "show_boundaries"]]
        if result.overlays and not output.overlays:
            output.messages.append("Flat periods are hidden on the chart; the summary table is unchanged.")
        return output
