"""The extension contract: source bars in, declarative chart results out."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import json
import math

import polars as pl


def json_copy(value):
    """Snapshot JSON data, rejecting executable objects and nonfinite numbers."""
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True)
class LogicContext:
    frame: pl.DataFrame
    timeframe: str
    identity: object
    cutoff: datetime
    session_timezone: str = "America/New_York"
    support_frame: pl.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def completed_frame(self):
        from dashboard.trend_input import completed_prefix
        return completed_prefix(self.frame, timeframe=self.timeframe, cutoff=self.cutoff,
                                session_timezone=self.session_timezone)


@dataclass(frozen=True)
class LogicSpec:
    instance_id: str
    name: str
    version: str
    config: dict

    def to_dict(self):
        return json_copy({"instance_id": self.instance_id, "name": self.name,
                          "version": self.version, "config": self.config})

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"instance_id", "name", "version", "config"}:
            raise ValueError("A chart logic requires instance_id, name, version, and config.")
        if any(not isinstance(value[key], str) or not value[key].strip()
               for key in ("instance_id", "name", "version")) or not isinstance(value["config"], dict):
            raise ValueError("Invalid chart logic identity or configuration.")
        return cls(**json_copy(value))


@dataclass
class LogicResult:
    overlays: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return json_copy({"overlays": self.overlays, "tables": self.tables, "messages": self.messages, "data": self.data})

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) not in
                ({"overlays", "tables", "messages"}, {"overlays", "tables", "messages", "data"})):
            raise ValueError("Invalid chart result.")
        result = cls(**json_copy(value))
        result.validate()
        return result

    def validate(self):
        """Reject malformed plugin output before persistence or chart rendering."""
        if not all(isinstance(value, list) for value in (self.overlays, self.tables, self.messages)):
            raise ValueError("Chart result collections must be lists.")
        if not isinstance(self.data, dict):
            raise ValueError("Chart result data must be a JSON object.")
        if not all(isinstance(message, str) for message in self.messages):
            raise ValueError("Chart messages must be strings.")
        for table in self.tables:
            if (not isinstance(table, dict) or not isinstance(table.get("name"), str)
                    or not isinstance(table.get("rows"), list)
                    or not all(isinstance(row, dict) for row in table["rows"])):
                raise ValueError("Result tables require a name and a list of row objects.")
        ids, panels = set(), {}
        for overlay in self.overlays:
            if not isinstance(overlay, dict):
                raise ValueError("An overlay must be an object.")
            identifier = overlay.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Overlay IDs must be nonempty and unique within a logic.")
            ids.add(identifier)
            panel = overlay.get("panel")
            if panel is not None:
                validate_panel(panel)
                if panel["id"] in panels and panels[panel["id"]] != panel:
                    raise ValueError("Primitives in the same panel must use the same panel declaration.")
                panels[panel["id"]] = panel
            kind = overlay.get("type")
            if kind == "log_line":
                values = [overlay.get(key) for key in ("start_index", "end_index", "slope", "intercept")]
                if (not all(_finite(value) for value in values)
                        or values[0] < 0 or values[1] < values[0]):
                    raise ValueError("Log-line overlays require finite fits and ordered source indices.")
                try:
                    if not all(math.isfinite(math.exp(value)) for value in
                               (values[3], values[3] + values[2] * (values[1] - values[0]))):
                        raise ValueError("Log-line fit produces nonfinite prices.")
                except OverflowError as error:
                    raise ValueError("Log-line fit produces nonfinite prices.") from error
            elif kind in ("line", "band", "markers", "bar"):
                xs = overlay.get("x")
                if not isinstance(xs, list) or not all(_finite(value) for value in xs):
                    raise ValueError("Overlay x coordinates must be finite source-bar indices.")
                if any(right < left for left, right in zip(xs, xs[1:])):
                    raise ValueError("Overlay source-bar indices must be ordered.")
                keys = ("lower", "upper") if kind == "band" else ("y",)
                for key in keys:
                    ys = overlay.get(key)
                    if (not isinstance(ys, list) or len(ys) != len(xs)
                            or not all(value is None or _finite(value) for value in ys)):
                        raise ValueError("Overlay values must match x and contain finite numbers or null.")
                if kind == "band" and any(lo is not None and hi is not None and lo > hi
                                          for lo, hi in zip(overlay["lower"], overlay["upper"])):
                    raise ValueError("Band lower values must not exceed upper values.")
            else:
                raise ValueError(f"Unsupported chart primitive {kind!r}.")
        self.to_dict()
        return self


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_panel(panel):
    """Validate an optional independent-axis panel shared by several primitives."""
    if (not isinstance(panel, dict) or set(panel) - {"id", "title", "range", "scope"}
            or any(not isinstance(panel.get(key), str) or not panel[key].strip()
                   for key in ("id", "title"))):
        raise ValueError("Indicator panels require nonempty id and title strings.")
    if panel.get("scope", "instance") not in ("instance", "logic"):
        raise ValueError("Indicator panel scope must be instance or logic.")
    limits = panel.get("range")
    if "range" in panel and (not isinstance(limits, list) or len(limits) != 2
                            or not all(_finite(value) for value in limits)
                            or limits[0] >= limits[1]):
        raise ValueError("Indicator panel ranges must contain two ordered finite values.")
    return panel


class BaseChartLogic(ABC):
    """Subclass in plugins/; no central list or renderer-specific code is needed.

    Calculation settings and live display settings use separate JSON Schemas.
    ``calculate`` returns the complete archive; ``present`` creates a display
    view without calling calculation code. Display values never enter cache keys.
    Bump ``result_version`` if the stored archive representation changes.
    Settings use JSON Schema. Calculations receive immutable input semantics:
    they must not mutate shared data or use chart viewport/display grouping.
    Bump ``version`` whenever calculation behavior or parameter semantics change.
    ``input_policy`` declares whether cache identity follows all original bars
    or only the completed prefix. Time-dependent plugins can opt into exact
    cutoff identity with ``uses_cutoff = True``.
    """

    name = ""
    version = "1"
    label = ""
    description = ""
    config_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    display_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    result_version = "1"
    required_columns = ("timestamp",)
    supported_timeframes = ()
    input_policy = "original"
    uses_cutoff = False

    def validate(self, config):
        return deepcopy(config)

    def display_schema_for(self, result):
        """Describe live controls, optionally using choices in the complete result."""
        return deepcopy(self.display_schema)

    def validate_display(self, config, result):
        return deepcopy(config)

    def present(self, result, config):
        """Filter/style a complete result without reading source data or calculating."""
        return deepcopy(result)

    @abstractmethod
    def calculate(self, context: LogicContext, config: dict, progress=None) -> LogicResult:
        """Compute outputs once when Apply is clicked; progress accepts done,total."""
