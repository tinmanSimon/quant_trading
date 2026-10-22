"""Conservative, table-based summaries shared by batch research and the UI.

Inference deliberately has no knowledge of an analysis's implementation or its
prose. Counts can be added; other quantities need an explicit aggregation choice.
Tables that cannot be inferred are still available through explicit MetricSpecs.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re


REDUCERS = ("single", "first", "last", "sum", "mean", "min", "max", "count")
AGGREGATIONS = ("none", "sum", "mean", "min", "max")


@dataclass(frozen=True)
class MetricSpec:
    """Select one numeric metric, or derive a ratio from two selected metrics.

    ``reducer`` combines a ticker's selected table cells. ``aggregation`` then
    combines that metric across tickers. ``single`` requires exactly one row,
    preventing repeated categories from silently turning into totals. ``count``
    counts non-null cells, including strings. ``first`` and ``last`` preserve
    endpoint nulls and use table order. A ratio references *base* metric names
    and is aggregated as the ratio of sums over matching contributors.
    """

    name: str
    table: str = ""
    column: str = ""
    reducer: str = "single"
    aggregation: str = "none"
    category_column: str | None = None
    category_value: str | int | float | bool | None = None
    numerator: str | None = None
    denominator: str | None = None
    scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A metric requires a nonempty name.")
        if self.reducer not in REDUCERS or self.aggregation not in AGGREGATIONS:
            raise ValueError("Unsupported metric reduction or aggregation.")
        if not _number(self.scale):
            raise ValueError("A metric scale must be finite.")
        if self.is_ratio:
            if not all(isinstance(value, str) and value.strip()
                       for value in (self.numerator, self.denominator)):
                raise ValueError("A ratio requires numerator and denominator metric names.")
            if self.table or self.column or self.category_column is not None:
                raise ValueError("A ratio references metrics, not table columns.")
        elif not all(isinstance(value, str) and value.strip()
                     for value in (self.table, self.column)):
            raise ValueError("A source metric requires a table and column.")
        if self.category_column is not None and (
                not isinstance(self.category_column, str) or not self.category_column.strip()):
            raise ValueError("A category column must be a nonempty string.")
        if self.category_value is not None and not isinstance(self.category_value, (str, int, float, bool)):
            raise ValueError("A category value must be a scalar.")
        if isinstance(self.category_value, float) and not math.isfinite(self.category_value):
            raise ValueError("A category value must be finite.")
        if self.category_column is None and self.category_value is not None:
            raise ValueError("A category value requires a category column.")

    @property
    def is_ratio(self):
        return self.numerator is not None or self.denominator is not None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("A metric specification must be an object.")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("Invalid metric specification fields.") from error


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _words(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _count_name(name):
    normalized = _words(name)
    return normalized in {
        "count", "n", "total", "win", "wins", "loss", "losses", "inconclusive",
        "signals", "entries", "exits", "observations", "samples", "trades",
    } or normalized.endswith(("_count", "_counts", "_total")) or normalized.startswith("number_of_")


def _coordinate(name):
    normalized = _words(name)
    return normalized in {"id", "index", "timestamp", "time", "date", "bar", "bars"} or (
        normalized.endswith(("_id", "_index", "_timestamp", "_time", "_date")))


def _price(name):
    normalized = _words(name)
    return "price" in normalized.split("_") or normalized in {"open", "high", "low", "close", "volume"}


def _columns(table):
    return list(dict.fromkeys([*table.get("columns", []),
                              *(key for row in table.get("rows", []) for key in row)]))


def infer_metrics(tables):
    """Recognize unique category/count tables and single-row numeric summaries.

    Unknown detail tables, coordinates, prices, boolean flags, text and malformed
    numeric columns do not become automatic totals. Single-row non-count metrics
    remain available for ranking, with overall aggregation disabled by default.
    """
    names = Counter(table.get("name") for table in tables)
    candidates = []
    for table in tables:
        name, rows = table.get("name"), table.get("rows", [])
        if not isinstance(name, str) or names[name] != 1 or not rows:
            continue
        columns = _columns(table)
        category_columns = [column for column in columns if _words(column) in {
            "outcome", "category", "metric", "name", "status", "result",
        }]
        count_columns = [column for column in columns if _words(column) in {"count", "n", "total"}]
        if len(columns) == 2 and len(category_columns) == len(count_columns) == 1:
            category_column, count_column = category_columns[0], count_columns[0]
            categories = [row.get(category_column) for row in rows]
            values = [row.get(count_column) for row in rows]
            if (all(isinstance(value, str) and value.strip() for value in categories)
                    and len(set(categories)) == len(categories)
                    and all(value is None or (_number(value) and value >= 0 and int(value) == value)
                            for value in values)):
                candidates.extend(MetricSpec(category, name, count_column, aggregation="sum",
                                             category_column=category_column, category_value=category)
                                  for category in categories)
            continue
        summary = any(word in _words(name).split("_") for word in
                      ("summary", "summaries", "metrics", "statistics", "stats", "overview", "totals"))
        if len(rows) != 1 or (not summary and any(_coordinate(column) for column in columns)):
            continue
        for column in columns:
            if _coordinate(column) or _price(column) or not _number(rows[0].get(column)):
                continue
            candidates.append(MetricSpec(column, name, column,
                                         aggregation="sum" if _count_name(column) else "none"))

    return _unique_names(candidates)


def _unique_names(candidates):
    # A table name disambiguates same-named metrics from different tables.
    counts = Counter(spec.name for spec in candidates)
    output, used = [], set()
    for spec in candidates:
        name = spec.name if counts[spec.name] == 1 else f"{spec.table}: {spec.name}"
        unique, suffix = name, 2
        while unique in used:
            unique, suffix = f"{name} ({suffix})", suffix + 1
        used.add(unique)
        output.append(MetricSpec.from_dict({**spec.to_dict(), "name": unique}))
    return output


def merge_summaries(per_ticker_summaries):
    """Union saved inferred summaries by source, never by display name alone.

    Each input is ``{ticker: {"metrics": [MetricSpec dict], "values": {name: value}}}``.
    Return canonical specifications and ticker values with the canonical names.
    A ticker may lack tables or categories that others returned. This also handles
    inference's display-name disambiguation changing when a table is absent.
    Ratios belong to the later user configuration and are not inferred summaries.
    """
    sources, ticker_sources = {}, {}
    for ticker, summary in per_ticker_summaries.items():
        selected = {}
        for spec in _specs(summary.get("metrics", [])):
            if spec.is_ratio:
                raise ValueError("Merge inferred source metrics before adding derived ratios.")
            key = (spec.table, spec.column, spec.reducer, spec.category_column,
                   type(spec.category_value), spec.category_value)
            value = summary.get("values", {}).get(spec.name)
            if value is not None and not _number(value):
                raise ValueError(f"Metric {spec.name!r}: summary values must be finite numbers or null.")
            if key in selected and selected[key] != value:
                raise ValueError(f"Ticker {ticker!r} supplies conflicting values for one metric source.")
            if key in sources and (sources[key].aggregation != spec.aggregation or sources[key].scale != spec.scale):
                raise ValueError(f"Metric {spec.name!r} has conflicting aggregation settings across tickers.")
            sources.setdefault(key, spec)
            selected[key] = value
        ticker_sources[ticker] = selected

    candidates = [MetricSpec.from_dict({**spec.to_dict(), "name": (
        str(spec.category_value) if spec.category_column is not None else spec.column)})
                  for spec in sources.values()]
    specs = _unique_names(candidates)
    source_names = dict(zip(sources, (spec.name for spec in specs)))
    values = {ticker: {source_names[key]: selected.get(key) for key in sources}
              for ticker, selected in ticker_sources.items()}
    return specs, values


def _specs(specs):
    specs = [spec if isinstance(spec, MetricSpec) else MetricSpec.from_dict(spec) for spec in specs]
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("Metric names must be unique.")
    for spec in specs:
        if spec.is_ratio and any(name not in by_name or by_name[name].is_ratio
                                 for name in (spec.numerator, spec.denominator)):
            raise ValueError(f"Ratio {spec.name!r} must reference existing source metrics.")
    return specs


def _reduce(values, operation):
    if operation == "single":
        if len(values) != 1:
            raise ValueError("The single reducer requires exactly one selected row.")
        return values[0]
    if operation in ("first", "last"):
        return values[0 if operation == "first" else -1] if values else None
    if operation == "count":
        return sum(value is not None for value in values)
    values = [value for value in values if value is not None]
    if not values:
        return None
    if operation == "sum":
        return sum(values)
    if operation == "mean":
        return sum(values) / len(values)
    if operation == "min":
        return min(values)
    if operation == "max":
        return max(values)
    return None


def _ratio(numerator, denominator, scale):
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator * scale
    if not _number(value):
        raise ValueError("Ratio calculation produced a nonfinite result.")
    return value


def extract_metrics(tables, specs):
    """Extract ticker values; missing data stays None, invalid data raises.

    A missing table/category/column contributes no observation. An existing empty
    column explicitly reduced with count yields zero; other empty reductions are
    unknown. Numeric strings are not silently converted to numbers.
    """
    specs = _specs(specs)
    values = {}
    for spec in specs:
        if spec.is_ratio:
            continue
        matches = [table for table in tables if table.get("name") == spec.table]
        if len(matches) > 1:
            raise ValueError(f"Metric {spec.name!r}: table name {spec.table!r} is ambiguous.")
        if not matches or spec.column not in _columns(matches[0]):
            values[spec.name] = None
            continue
        table = matches[0]
        rows = table.get("rows", [])
        if spec.category_column is not None:
            rows = [row for row in rows if spec.category_column in row
                    and type(row[spec.category_column]) is type(spec.category_value)
                    and row[spec.category_column] == spec.category_value]
            if not rows:
                values[spec.name] = None
                continue
        selected = [row.get(spec.column) for row in rows]
        if spec.reducer != "count" and any(value is not None and not _number(value) for value in selected):
            raise ValueError(f"Metric {spec.name!r}: selected values must be finite numbers or null.")
        if not selected and spec.reducer != "count":
            values[spec.name] = None
            continue
        try:
            value = _reduce(selected, spec.reducer)
        except ValueError as error:
            raise ValueError(f"Metric {spec.name!r}: {error}") from error
        if value is not None and not _number(value):
            raise ValueError(f"Metric {spec.name!r}: reduction produced a nonfinite result.")
        values[spec.name] = value
    for spec in specs:
        if spec.is_ratio:
            values[spec.name] = _ratio(values[spec.numerator], values[spec.denominator], spec.scale)
    return {spec.name: values[spec.name] for spec in specs}


def aggregate_metrics(per_ticker_values, specs):
    """Aggregate supplied successful ticker values and report contributors.

    Callers omit failed runs. Missing metrics are never changed into zero. Ratios
    use only tickers supplying both inputs, so their numerator and denominator
    always describe the same population, including a ticker with zero outcomes.
    """
    specs = _specs(specs)
    output = {}
    for spec in specs:
        if spec.is_ratio:
            paired = [(row.get(spec.numerator), row.get(spec.denominator))
                      for row in per_ticker_values.values()]
            if any(value is not None and not _number(value) for pair in paired for value in pair):
                raise ValueError(f"Metric {spec.name!r}: ratio inputs must be finite numbers or null.")
            paired = [(numerator, denominator) for numerator, denominator in paired
                      if numerator is not None and denominator is not None]
            value = _ratio(sum(pair[0] for pair in paired), sum(pair[1] for pair in paired), spec.scale)
            operation, contributors = "ratio", len(paired)
        else:
            values = [row.get(spec.name) for row in per_ticker_values.values()]
            if any(value is not None and not _number(value) for value in values):
                raise ValueError(f"Metric {spec.name!r}: values must be finite numbers or null.")
            values = [value for value in values if value is not None]
            value = _reduce(values, spec.aggregation)
            operation, contributors = spec.aggregation, len(values)
        if value is not None and not _number(value):
            raise ValueError(f"Metric {spec.name!r}: aggregation produced a nonfinite result.")
        output[spec.name] = {"value": value, "contributors": contributors, "aggregation": operation}
    return output


def rank_metrics(per_ticker_values, metric, *, descending=True):
    """Return ticker names ranked stably, with missing metrics always last."""
    known, missing = [], []
    for ticker, row in per_ticker_values.items():
        value = row.get(metric)
        if value is None:
            missing.append(ticker)
        elif not _number(value):
            raise ValueError(f"Metric {metric!r}: ranking values must be finite numbers or null.")
        else:
            known.append((ticker, value))
    return [ticker for ticker, _ in sorted(known, key=lambda pair: pair[1], reverse=descending)] + missing
