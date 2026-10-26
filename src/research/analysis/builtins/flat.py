"""Shared flat-period calculation for Market Data and batch Analysis."""

from copy import deepcopy

from research.analysis.core.base import BaseChartLogic
from research.analysis.flat import ALGORITHM_VERSION, FlatConfig, analyze
from research.analysis.flat.presentation import summary_result


class FlatLogic(BaseChartLogic):
    name = "flat"
    version = f"1.0.0-engine.{ALGORITHM_VERSION}"
    result_version = "3"
    label = "Flat analysis"
    description = (
        "Find nonoverlapping flat periods, newest first, using completed bars. "
        "Keep every candle's high and low inside a fixed or volatility-adjusted horizontal band, "
        "with an optional maximum-bar versus median-bar filter."
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
            "band_width_mode": {
                "type": "string", "enum": ["fixed", "automatic"], "default": "fixed",
                "title": "Band width mode",
                "description": "Use a fixed percentage or calculate the allowed width from the bars within each candidate flat period.",
            },
            "max_range_pct": {
                "type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 200, "default": 2.0,
                "title": "Fixed maximum band width (%)",
                "x-ui-enabled-when": {"band_width_mode": "fixed"},
                "description": (
                    "Total highest-high to lowest-low range as a percentage of their midpoint. "
                    "A 2% band around 100 spans 99–101. Every candle, including its wicks, must fit."
                ),
            },
            "auto_percentile": {
                "type": "number", "minimum": 0, "maximum": 100, "default": 90.0,
                "title": "Automatic bar-length percentile",
                "x-ui-enabled-when": {"band_width_mode": "automatic"},
                "description": (
                    "Percentile of full high–low bar lengths within each candidate period, including both endpoints. "
                    "Each length is measured as a percentage of its own midpoint, using linear interpolation. "
                    "The width is recalculated as the candidate changes; every candle's wicks must still fit."
                ),
            },
            "auto_multiplier": {
                "type": "number", "exclusiveMinimum": 0, "default": 2.0,
                "title": "Automatic band width multiplier",
                "x-ui-enabled-when": {"band_width_mode": "automatic"},
                "description": "Maximum total band width = selected bar-length percentile × this multiplier.",
            },
            "long_bar_filter": {
                "type": "boolean", "default": False,
                "title": "Filter unusually long bars",
                "description": (
                    "Require the largest bar length within each candidate period to be no greater than "
                    "its median bar length × the multiplier below. Applies in either band width mode."
                ),
            },
            "long_bar_multiplier": {
                "type": "number", "minimum": 1, "default": 2.0,
                "title": "Maximum bar / median bar multiplier",
                "x-ui-enabled-when": {"long_bar_filter": True},
                "description": (
                    "Compare full high–low lengths as percentages of each bar's midpoint. Small bars remain "
                    "allowed. A zero median permits only zero-length bars; this filter does not test price direction."
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
