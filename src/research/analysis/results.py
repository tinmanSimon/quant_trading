"""Plain-data findings for research scripts and optional dashboard presentation.

Builders accept ordinary Polars frames. Chart coordinates are always validated
against the context's selected completed bars; earlier initialization rows must
be trimmed explicitly with ``ctx.selected`` before adding chart findings.
"""

from datetime import date, datetime
from hashlib import sha256
from html import escape
import math
from numbers import Integral

import polars as pl

from .base import LogicResult, json_copy, validate_panel


def _json(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _frame(value):
    if not isinstance(value, pl.DataFrame):
        raise TypeError("Analysis outputs must use a Polars DataFrame.")
    return value


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Result names must be nonempty strings.")
    return value


def _id(kind, name):
    return kind + "-" + sha256(name.encode()).hexdigest()[:16]


class AnalysisResult:
    """Build a result with text/tables and optional chart layers and events.

    Methods return ``self`` and can be chained. Return the builder from an
    ``@analysis`` function, or call ``to_logic_result`` for the execution API.
    No method imports the dashboard or a plotting library.
    """

    def __init__(self, context):
        self._bars = context.bars
        self._stamps = dict(self._bars.select("source_index", "timestamp").iter_rows())
        self._indices = {stamp: index for index, stamp in self._stamps.items()}
        self._context = context
        self._text = []
        self._tables = []
        self._overlays = []
        self._events = []
        self._groups = set()

    def text(self, markdown):
        """Add Markdown; the dashboard renders it without enabling raw HTML."""
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("Analysis text must be a nonempty string.")
        self._text.append(markdown)
        return self

    def table(self, name, frame):
        """Add arbitrary columns, including an empty table as a valid finding."""
        name = _name(name)
        if any(table["name"] == name for table in self._tables):
            raise ValueError(f"A result table named {name!r} already exists.")
        frame = _frame(frame)
        self._tables.append(json_copy({"name": name, "columns": frame.columns,
                                       "rows": _json(frame.to_dicts())}))
        return self

    def _position(self, value):
        if isinstance(value, Integral) and not isinstance(value, bool):
            index = int(value)
            if index in self._stamps:
                return index
        elif isinstance(value, (datetime, date)):
            if value in self._indices:
                return self._indices[value]
        raise ValueError("Chart findings must reference a selected completed bar; use ctx.selected(frame) first.")

    def _located(self, frame, *, start=None):
        frame = _frame(frame)
        if start is not None:
            if start not in frame.columns:
                raise ValueError(f"Finding position column {start!r} is missing.")
        elif not {"source_index", "timestamp"}.intersection(frame.columns):
            raise ValueError("Chart findings need a source_index or timestamp column.")
        indices = []
        for row in frame.iter_rows(named=True):
            index = self._position(row[start] if start else row.get("source_index", row.get("timestamp")))
            if start is None and "timestamp" in row and row["timestamp"] != self._stamps[index]:
                raise ValueError("Finding timestamps and source indices do not identify the same bar.")
            indices.append(index)
        return indices

    @staticmethod
    def _values(frame, column):
        if column not in frame.columns:
            raise ValueError(f"Chart value column {column!r} is missing.")
        values = frame[column].to_list()
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
               for value in values):
            raise ValueError("Chart values must be finite numbers or null warmup values.")
        return values

    def _layer(self, kind, name, frame, *, panel=None, **values):
        name = _name(name)
        identifier = _id(kind, name)
        if any(item["id"] == identifier for item in self._overlays):
            raise ValueError(f"A {kind} layer named {name!r} already exists.")
        indices = self._located(frame)
        order = sorted(range(len(indices)), key=indices.__getitem__)
        descriptor = {"id": identifier, "name": name, "type": kind,
                      "x": [indices[index] for index in order]}
        for key in ("y", "lower", "upper"):
            if key in values:
                column = self._values(frame, values.pop(key))
                descriptor[key] = [column[index] for index in order]
        if panel is not None:
            descriptor["panel"] = json_copy(validate_panel(panel))
        descriptor.update(values)
        self._overlays.append(json_copy(descriptor))
        return self

    def markers(self, name, frame, *, price="close", color="#16a34a", symbol="circle",
                size=10, panel=None, hovertemplate=None):
        """Draw points on price, or an explicitly declared independent panel."""
        return self._layer("markers", name, frame, y=price, panel=panel,
                           marker={"color": color, "symbol": symbol, "size": size},
                           hovertemplate=hovertemplate or escape(name) + ": %{y:.6g}<extra></extra>")

    def line(self, name, frame, *, value="close", color="#2563eb", width=2,
             dash="solid", panel=None, hovertemplate=None):
        """Draw a line; null values preserve breaks and indicator warmup."""
        return self._layer("line", name, frame, y=value, panel=panel,
                           line={"color": color, "width": width, "dash": dash},
                           hovertemplate=hovertemplate or escape(name) + ": %{y:.6g}<extra></extra>")

    def band(self, name, frame, *, lower, upper, color="rgba(37,99,235,0.16)",
             line_color="#2563eb", panel=None):
        lows, highs = self._values(frame, lower), self._values(frame, upper)
        if any((lo is None) != (hi is None) or (lo is not None and lo > hi)
               for lo, hi in zip(lows, highs)):
            raise ValueError("Band boundaries must share null gaps and have lower <= upper.")
        return self._layer("band", name, frame, lower=lower, upper=upper, panel=panel,
                           fillcolor=color, line={"color": line_color, "width": 1})

    def _add_events(self, name, frame, *, label=None, start=None, end=None):
        name = _name(name)
        if name in self._groups:
            raise ValueError(f"An event group named {name!r} already exists.")
        indices = self._located(frame, start=start)
        identifiers, records = [], []
        for ordinal, (index, row) in enumerate(zip(indices, frame.iter_rows(named=True))):
            identifier = f"{_id('event', name)}:{index}:{ordinal}"
            text = row[label] if label in row else label
            item = {"id": identifier, "group": name, "label": str(text if text is not None else name),
                    "source_index": index, "timestamp": self._stamps[index].isoformat()}
            end_column = end or ("end_index" if "end_index" in row else
                                 "end_timestamp" if "end_timestamp" in row else None)
            if end_column:
                if end_column not in row:
                    raise ValueError(f"Event end column {end_column!r} is missing.")
                end_index = self._position(row[end_column])
                if end_index < index:
                    raise ValueError("Event intervals must end at or after their starting bar.")
                if "end_timestamp" in row and row["end_timestamp"] != self._stamps[end_index]:
                    raise ValueError("Event end timestamps and indices must identify the same bar.")
                item.update(end_index=end_index, end_timestamp=self._stamps[end_index].isoformat())
            identifiers.append(identifier)
            records.append(item)
        self._groups.add(name)
        self._events.extend(records)
        return identifiers

    def events(self, name, frame, *, label=None, start=None, end=None):
        """Register points or intervals for Previous/Next and event selection.

        ``label`` may be a column name or constant text. ``start`` and ``end``
        optionally name columns containing original indices or timestamps.
        """
        self._add_events(name, frame, label=label, start=start, end=end)
        return self

    def signals(self, name, frame, *, price="close", color="#16a34a", symbol="circle", size=10,
                label=None):
        """Create a table, optional-chart markers, and navigation events together."""
        if "_event_id" in frame.columns:
            raise ValueError("_event_id is reserved for linking result tables to navigation events.")
        identifiers = self._add_events(name, frame, label=label)
        self.table(name, frame.with_columns(pl.Series("_event_id", identifiers, dtype=pl.String)))
        self.markers(name, frame, price=price, color=color, symbol=symbol, size=size)
        return self

    def to_logic_result(self):
        if not self._text and not self._tables:
            raise ValueError("An analysis must return text or at least one table, including an empty table.")
        result = LogicResult(overlays=json_copy(self._overlays), tables=json_copy(self._tables),
                             data={"analysis": {"text": list(self._text), "events": json_copy(self._events),
                                                "layers": [{key: item[key] for key in ("id", "name", "type")}
                                                           for item in self._overlays]},
                                   "metadata": json_copy(_json(self._context.metadata))})
        return result.validate()
