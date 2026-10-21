"""Technical indicator adapters: calculation archives, controls, and chart panels.

The independent ``market_indicators`` package owns all numerical calculations.
Tables are the single full-resolution archive; presentation builds disposable
chart descriptors so values are not duplicated in persisted calculations.
"""

from abc import abstractmethod
from copy import deepcopy

import polars as pl

import market_indicators
from chart_logics.base import BaseChartLogic, LogicResult
from dashboard.trend_input import completed_prefix
from data_pipeline.processing.configuration import validated_config


def _schema(properties):
    return {"type": "object", "additionalProperties": False, "properties": properties}


def _period(default, title="Period", minimum=2):
    return {"type": "integer", "minimum": minimum, "maximum": 100_000,
            "default": default, "title": title}


def _price():
    return {"type": "string", "enum": ["close", "open", "high", "low"],
            "default": "close", "title": "Price source"}


def _color(default, title):
    return {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$", "default": default,
            "title": title, "description": "Six-digit hex color, for example #2563eb."}


def _display(series, references=None, histogram=False):
    properties = {"line_width": {"type": "number", "minimum": 0.5, "maximum": 8,
                                  "default": 1.5, "title": "Line width"}}
    for key, label, color in series:
        properties[f"show_{key}"] = {"type": "boolean", "default": True,
                                      "title": f"Show {label}"}
        properties[f"{key}_color"] = _color(color, f"{label} color")
    if histogram:
        properties.update({
            "show_histogram": {"type": "boolean", "default": True, "title": "Show histogram"},
            "positive_color": _color("#16a34a", "Positive histogram color"),
            "negative_color": _color("#dc2626", "Negative histogram color"),
        })
    if references is not None:
        properties.update({
            "show_reference_levels": {"type": "boolean", "default": True,
                                       "title": "Show reference levels"},
            "reference_levels": {"type": "array", "items": {"type": "number"},
                                 "uniqueItems": True, "maxItems": 20,
                                 "default": references, "title": "Reference levels"},
            "reference_color": _color("#94a3b8", "Reference level color"),
        })
    return _schema(properties)


def _input_identity(frame, timeframe):
    """Reject mixed instruments/timeframes before dropping nonnumeric columns."""
    identity = {}
    for key in ("symbol", "ticker", "provider", "timeframe", "interval"):
        if key not in frame.columns or frame.is_empty():
            continue
        values = frame[key]
        if values.null_count() or values.n_unique() != 1:
            raise ValueError(f"Indicator input requires a single non-null {key}.")
        identity[key] = values[0]
        if key in ("timeframe", "interval") and identity[key] != timeframe:
            raise ValueError("Indicator input timeframe must match the chart timeframe.")
    return identity


def _ordered_timestamps(frame):
    if "timestamp" not in frame.columns:
        raise ValueError("Indicator input requires timestamp.")
    stamps = frame["timestamp"]
    if stamps.null_count() or not stamps.is_sorted() or stamps.n_unique() != frame.height:
        raise ValueError("Indicator timestamps must be non-null, increasing, and unique.")


def _calculation_input(context, columns):
    """Prepend only earlier completed bars, retaining selected-bar coordinates."""
    _ordered_timestamps(context.frame)
    selected_identity = _input_identity(context.frame, context.timeframe)
    selected = context.completed_frame
    missing = set(columns) - set(selected.columns)
    if missing:
        raise ValueError(f"Missing required data columns: {', '.join(sorted(missing))}.")
    selected = selected.select(columns)
    support = context.support_frame
    if selected.is_empty() or support is None or support.is_empty():
        return selected, 0
    if "timestamp" not in support.columns:
        raise ValueError("Indicator initialization history requires timestamp.")
    support = support.filter(pl.col("timestamp") < selected["timestamp"][0])
    _ordered_timestamps(support)
    support_identity = _input_identity(support, context.timeframe)
    for key in set(selected_identity) & set(support_identity):
        if selected_identity[key] != support_identity[key]:
            raise ValueError(f"Indicator initialization history has a different {key}.")
    selected_symbol = selected_identity.get("symbol", selected_identity.get("ticker"))
    support_symbol = support_identity.get("symbol", support_identity.get("ticker"))
    if selected_symbol is not None and support_symbol is not None and selected_symbol != support_symbol:
        raise ValueError("Indicator initialization history has a different symbol.")
    support = completed_prefix(support, timeframe=context.timeframe, cutoff=context.cutoff,
                               session_timezone=context.session_timezone)
    if support.is_empty():
        return selected, 0
    missing = set(columns) - set(support.columns)
    if missing:
        raise ValueError("Indicator initialization history is missing required columns: "
                         f"{', '.join(sorted(missing))}.")
    support = support.select(columns)
    return pl.concat([support, selected], how="vertical_relaxed"), support.height


class _IndicatorLogic(BaseChartLogic):
    """Shared adapter, kept abstract so discovery only exposes real indicators."""

    # Calculation identity includes the independently versioned backend contract.
    version = f"1-{market_indicators.ALGORITHM_VERSION}"
    result_version = "1"
    input_policy = "completed"
    required_columns = ("timestamp",)
    function = ""
    series = ()
    panel_range = None

    @abstractmethod
    def title(self, config):
        """Display the active calculation settings in the panel title."""

    def validate(self, config):
        values = validated_config(config, self.config_schema)
        for name, schema in self.config_schema["properties"].items():
            if schema.get("type") == "integer" and type(values[name]) is not int:
                raise ValueError(f"{name} must be an integer.")
        if "fast" in values and values["fast"] >= values["slow"]:
            raise ValueError("The fast MACD period must be less than the slow period.")
        return values

    def validate_display(self, config, result):
        values = validated_config(config, self.display_schema)
        if self.panel_range is not None and "reference_levels" in values:
            lower, upper = self.panel_range
            if any(value < lower or value > upper for value in values["reference_levels"]):
                raise ValueError(f"Reference levels must be between {lower} and {upper}.")
        return values

    def calculate(self, context, config, progress=None):
        values = self.validate(config)
        columns = list(self.required_columns)
        if "price_col" in values and values["price_col"] not in columns:
            columns.append(values["price_col"])
        frame, support_count = _calculation_input(context, columns)
        calculated = getattr(market_indicators, self.function)(frame, **values)
        selected = calculated.frame.slice(support_count).with_columns(
            (pl.col("source_index") - support_count).alias("source_index"))
        rows = selected.to_dicts()
        for row in rows:
            row["timestamp"] = row["timestamp"].isoformat()
        panel = {"id": "indicator", "title": self.title(values)}
        if self.panel_range is not None:
            panel["range"] = list(self.panel_range)
        metadata = deepcopy(calculated.metadata)
        metadata.update({"initialization_bars": support_count, "selected_bars": selected.height})
        messages = []
        if selected.is_empty():
            messages.append(f"No completed bars are available for {self.label}.")
        else:
            if support_count:
                messages.append(f"Initialized with {support_count:,} earlier completed bars; "
                                "the table and chart contain only selected bars.")
            else:
                messages.append("Initialized from the first selected completed bar; "
                                "values remain empty until the indicator is ready.")
            output_columns = [name for name in selected.columns if name not in ("timestamp", "source_index")]
            if all(selected[name].null_count() == selected.height for name in output_columns):
                messages.append("Insufficient completed history for these settings; "
                                "all indicator values are still initializing.")
        if progress is not None:
            progress(1, 1)
        return LogicResult(tables=[{"name": panel["title"], "rows": rows}], messages=messages,
                           data={"panel": panel, "metadata": metadata})

    def present(self, result, config):
        values = self.validate_display(config, result)
        # The registry supplies a disposable copy of the complete archive.
        rows = result.tables[0]["rows"] if result.tables else []
        result.overlays = []
        if not rows:
            return result
        panel = result.data["panel"]
        xs = [row["source_index"] for row in rows]
        for key, label, _color_value in self.series:
            if not values[f"show_{key}"]:
                continue
            result.overlays.append({
                "id": key, "type": "line", "name": label, "panel": deepcopy(panel),
                "x": xs, "y": [row[key] for row in rows],
                "line": {"color": values[f"{key}_color"], "width": values["line_width"]},
                "hovertemplate": f"{label}: %{{y:.6g}}<extra></extra>",
            })
        if values.get("show_histogram"):
            for positive, key in ((True, "positive"), (False, "negative")):
                result.overlays.append({
                    "id": f"histogram_{key}", "type": "bar", "name": "MACD histogram",
                    "panel": deepcopy(panel), "x": xs,
                    "y": [row["histogram"] if row["histogram"] is not None
                          and (row["histogram"] >= 0) == positive else None for row in rows],
                    "marker": {"color": values[f"{key}_color"]},
                    "hovertemplate": "Histogram: %{y:.6g}<extra></extra>",
                })
        # Hiding every data series also hides the panel rather than leaving
        # reference lines as an otherwise empty indicator.
        if result.overlays and values.get("show_reference_levels"):
            for index, level in enumerate(values["reference_levels"]):
                result.overlays.append({
                    "id": f"reference_{index}", "type": "line", "name": f"Reference {level:g}",
                    "panel": deepcopy(panel), "x": [xs[0], xs[-1]], "y": [level, level],
                    "line": {"color": values["reference_color"], "width": 1, "dash": "dash"},
                    "hovertemplate": f"Reference: {level:g}<extra></extra>",
                })
        if not result.overlays:
            result.messages.append("All indicator series are hidden. Enable a series to show this panel.")
        return result


class MovingAverageLogic(_IndicatorLogic):
    name = "moving_average"
    label = "Moving average"
    description = "Simple, exponential, or weighted average in a synchronized indicator panel."
    function = "moving_average"
    series = (("value", "Moving average", "#2563eb"),)
    config_schema = _schema({
        "kind": {"type": "string", "enum": ["sma", "ema", "wma"], "default": "sma",
                 "title": "Average type"},
        "period": _period(20), "price_col": _price(),
    })
    display_schema = _display(series)

    def title(self, config):
        return f"{config['kind'].upper()} ({config['period']}, {config['price_col']})"


class RSILogic(_IndicatorLogic):
    name = "rsi"
    label = "RSI"
    description = "Relative strength index with customizable reference levels."
    function = "rsi"
    series = (("value", "RSI", "#8b5cf6"),)
    panel_range = (0, 100)
    config_schema = _schema({"period": _period(14), "price_col": _price()})
    display_schema = _display(series, [30, 70])

    def title(self, config):
        return f"RSI ({config['period']}, {config['price_col']})"


class MACDLogic(_IndicatorLogic):
    name = "macd"
    label = "MACD"
    description = "Moving average convergence/divergence, signal, and histogram."
    function = "macd"
    series = (("macd", "MACD", "#2563eb"), ("signal", "Signal", "#f97316"))
    config_schema = _schema({"fast": _period(12, "Fast period"), "slow": _period(26, "Slow period"),
                             "signal": _period(9, "Signal period", minimum=1), "price_col": _price()})
    display_schema = _display(series, [0], histogram=True)

    def title(self, config):
        return f"MACD ({config['fast']}, {config['slow']}, {config['signal']}, {config['price_col']})"


class ATRLogic(_IndicatorLogic):
    name = "atr"
    label = "ATR / normalized ATR"
    description = "Average true range in price units or as a percentage of closing price."
    function = "atr"
    required_columns = ("timestamp", "high", "low", "close")
    series = (("value", "ATR", "#f59e0b"),)
    config_schema = _schema({"period": _period(14, minimum=1), "normalized": {
        "type": "boolean", "default": False, "title": "Normalize to percent"}})
    display_schema = _display(series)

    def title(self, config):
        kind = "Normalized ATR (%)" if config["normalized"] else "ATR"
        return f"{kind} ({config['period']})"


class BollingerBandsLogic(_IndicatorLogic):
    name = "bollinger_bands"
    label = "Bollinger Bands"
    description = "Simple moving average with independently configurable upper and lower deviations."
    function = "bollinger_bands"
    series = (("middle", "Middle", "#2563eb"), ("upper", "Upper band", "#16a34a"),
              ("lower", "Lower band", "#dc2626"))
    config_schema = _schema({
        "period": _period(20), "price_col": _price(),
        "upper_deviation": {"type": "number", "minimum": 0, "maximum": 100_000,
                            "default": 2., "title": "Upper standard deviations"},
        "lower_deviation": {"type": "number", "minimum": 0, "maximum": 100_000,
                            "default": 2., "title": "Lower standard deviations"},
    })
    display_schema = _display(series)

    def title(self, config):
        return (f"Bollinger Bands ({config['period']}, +{config['upper_deviation']:g}/"
                f"−{config['lower_deviation']:g}σ, {config['price_col']})")


class ADXLogic(_IndicatorLogic):
    name = "adx"
    label = "ADX / directional indicators"
    description = "Trend strength and positive/negative directional indicators."
    function = "adx"
    required_columns = ("timestamp", "high", "low", "close")
    series = (("adx", "ADX", "#8b5cf6"), ("plus_di", "+DI", "#16a34a"),
              ("minus_di", "−DI", "#dc2626"))
    panel_range = (0, 100)
    config_schema = _schema({"period": _period(14)})
    display_schema = _display(series, [25])

    def title(self, config):
        return f"ADX / DI ({config['period']})"


class StochasticLogic(_IndicatorLogic):
    name = "stochastic"
    label = "Stochastic oscillator"
    description = "Smoothed %K and %D with customizable lookback and reference levels."
    function = "stochastic"
    required_columns = ("timestamp", "high", "low", "close")
    series = (("k", "%K", "#2563eb"), ("d", "%D", "#f97316"))
    panel_range = (0, 100)
    config_schema = _schema({
        "period": _period(14, "Lookback period", minimum=1),
        "k_smoothing": _period(3, "%K smoothing period", minimum=1),
        "d_period": _period(3, "%D period", minimum=1),
    })
    display_schema = _display(series, [20, 80])

    def title(self, config):
        return f"Stochastic ({config['period']}, {config['k_smoothing']}, {config['d_period']})"


class OBVLogic(_IndicatorLogic):
    name = "obv"
    label = "On-balance volume"
    description = "Cumulative signed volume initialized from supplied history."
    function = "obv"
    required_columns = ("timestamp", "volume")
    series = (("value", "OBV", "#0891b2"),)
    config_schema = _schema({"price_col": _price()})
    display_schema = _display(series)

    def title(self, config):
        return f"OBV ({config['price_col']})"
