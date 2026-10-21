"""Completed, aligned research inputs and reusable calculations for scripts.

The selected range has zero-based source indices. Earlier initialization bars
have negative indices. Indicator helpers calculate on the complete available
history; evaluate conditions there, then call ``selected`` to trim findings.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime
import hashlib
import json
from numbers import Integral

import polars as pl

from . import indicators
from .input import bar_available_at, completed_prefix


_IDENTITY_COLUMNS = ("symbol", "ticker", "instrument", "provider", "timeframe", "interval")


def _validate_frame(frame):
    if not isinstance(frame, pl.DataFrame) or "timestamp" not in frame.columns:
        raise ValueError("Analysis input requires a Polars DataFrame with timestamp")
    stamps = frame["timestamp"]
    if stamps.dtype.base_type() not in (pl.Date, pl.Datetime):
        raise ValueError("Analysis timestamps must have Date or Datetime dtype")
    if stamps.null_count() or not stamps.is_sorted() or stamps.n_unique() != frame.height:
        raise ValueError("Analysis timestamps must be non-null, increasing, and unique")
    for name in _IDENTITY_COLUMNS:
        if name in frame.columns and frame.height:
            values = frame[name]
            if values.null_count() or values.n_unique() != 1:
                raise ValueError(f"Analysis input requires a single non-null {name}")


def _fingerprint(frame):
    digest = hashlib.sha256(str(frame.schema).encode())
    digest.update(frame.hash_rows(seed=0).to_numpy().tobytes())
    return digest.hexdigest()


class Feature:
    """An aligned calculation with provenance and one or several value columns.

    ``ctx.features(momentum=ctx.macd())`` creates momentum_macd,
    momentum_signal, and momentum_histogram. Use ``ctx.macd().select('macd')``
    to expose just one output under the supplied alias.
    """

    def __init__(self, frame: pl.DataFrame, metadata=None):
        self._frame = frame.clone()
        self._metadata = deepcopy(metadata or {})

    @property
    def frame(self):
        return self._frame.clone()

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def columns(self):
        return tuple(name for name in self._frame.columns if name not in ("timestamp", "source_index"))

    def select(self, *columns):
        if not columns or any(name not in self.columns for name in columns):
            raise ValueError(f"Select one or more feature outputs from {self.columns}")
        if len(set(columns)) != len(columns):
            raise ValueError("Feature output selections must be unique")
        return Feature(self._frame.select("source_index", "timestamp", *columns), self._metadata)

    def to_frame(self):
        return self.frame


class AnalysisContext:
    """Analysis inputs with explicit initialization and historical boundaries.

    Construct from the common ``LogicContext`` using ``from_logic``. Public
    frames and metadata are copies, so scripts cannot mutate another analysis's
    input or cached calculations. A shared cache can be passed for one run.
    """

    @classmethod
    def from_logic(cls, context, shared_cache=None):
        _validate_frame(context.frame)
        selected = completed_prefix(context.frame, timeframe=context.timeframe,
                                    cutoff=context.cutoff, session_timezone=context.session_timezone)
        if "source_index" in selected.columns:
            selected = selected.drop("source_index")
        selected = selected.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int64).alias("source_index"))
        support = context.support_frame
        if support is not None:
            _validate_frame(support)
            if support.schema["timestamp"] != selected.schema["timestamp"]:
                raise ValueError("Initialization and selected timestamps must have the same dtype and timezone")
            # The original range start still bounds history if every selected
            # bar is incomplete. An empty source supplies no usable boundary.
            if context.frame.height:
                support = support.filter(pl.col("timestamp") < context.frame["timestamp"][0])
            else:
                support = support.head(0)
            support = completed_prefix(support, timeframe=context.timeframe,
                                       cutoff=context.cutoff, session_timezone=context.session_timezone)
            for name in _IDENTITY_COLUMNS:
                if name in support.columns and name in context.frame.columns and support.height and context.frame.height:
                    if support[name][0] != context.frame[name][0]:
                        raise ValueError(f"Initialization history and selected bars have different {name}")
            if support.height:
                required = {name for name in ("open", "high", "low", "close", "volume") if name in selected.columns}
                if required - set(support.columns):
                    raise ValueError("Initialization history is missing selected OHLCV columns")
                if "source_index" in support.columns:
                    support = support.drop("source_index")
                support = support.with_columns(pl.int_range(-support.height, 0, dtype=pl.Int64).alias("source_index"))
                history = pl.concat([support, selected], how="diagonal_relaxed")
            else:
                history = selected
        else:
            history = selected
        return cls._create(history, context, shared_cache if shared_cache is not None else {})

    @classmethod
    def _create(cls, history, original, cache, *, as_of=None):
        self = cls()
        self._history = history.clone()
        self._bars = history.filter(pl.col("source_index") >= 0)
        self._original = original
        self._cache = cache
        self._fingerprint = _fingerprint(history)
        self._as_of = as_of
        self._calculations = []
        self.timeframe = original.timeframe
        self.identity = deepcopy(original.identity)
        self.cutoff = (bar_available_at(as_of, timeframe=original.timeframe,
                                       session_timezone=original.session_timezone)
                       if as_of is not None else original.cutoff)
        self.session_timezone = original.session_timezone
        return self

    @property
    def bars(self):
        return self._bars.clone()

    @property
    def history(self):
        return self._history.clone()

    @property
    def metadata(self):
        metadata = deepcopy(self._original.metadata)
        metadata.update({
            "timeframe": self.timeframe,
            "identity": str(self.identity),
            "cutoff": self.cutoff.isoformat(),
            "source_cutoff": self._original.cutoff.isoformat(),
            "initialization_bars": self._history.height - self._bars.height,
            "selected_bars": self._bars.height,
            "available_rows": self._history.height,
            "history_start": self._history["timestamp"][0].isoformat() if self._history.height else None,
            "history_end": self._history["timestamp"][-1].isoformat() if self._history.height else None,
            "as_of": self._as_of.isoformat() if self._as_of is not None else None,
            "calculations": deepcopy(self._calculations),
        })
        return metadata

    def _aligned(self, frame, *, complete=False):
        if not isinstance(frame, pl.DataFrame) or not {"timestamp", "source_index"} <= set(frame.columns):
            raise ValueError("Aligned frames require timestamp and source_index columns")
        indices = frame["source_index"]
        if not indices.dtype.is_integer() or indices.null_count():
            raise ValueError("source_index must contain non-null integer bar identities")
        if frame.schema["timestamp"] != self._history.schema["timestamp"]:
            raise ValueError("Aligned timestamps must preserve the history dtype and timezone")
        if complete:
            if not frame.select("source_index", "timestamp").equals(self._history.select("source_index", "timestamp")):
                raise ValueError("Feature rows must exactly align with the available history")
        else:
            invalid = frame.select("source_index", "timestamp").join(
                self._history.select("source_index", "timestamp"), on=["source_index", "timestamp"], how="anti")
            if invalid.height:
                raise ValueError("Result rows contain unknown or mismatched bar identities")
        return frame

    def selected(self, frame):
        """Trim an aligned frame to selected bars; preserve columns and row order.

        Duplicate findings on one bar are permitted. Timestamp/index pairs must
        belong to this context, preventing accidental joins to another series.
        """
        return self._aligned(frame).filter(pl.col("source_index") >= 0)

    def features(self, **named_features):
        """Attach named, aligned outputs to a copy of the working history."""
        frame = self._history.clone()
        columns = []
        used = set(frame.columns)
        for alias, feature in named_features.items():
            if not isinstance(alias, str) or not alias:
                raise ValueError("Feature aliases must be nonempty strings")
            if isinstance(feature, (Feature, indicators.IndicatorResult)):
                values = feature.frame
            elif isinstance(feature, pl.DataFrame):
                values = feature
            else:
                raise ValueError("Features must be aligned Feature, IndicatorResult, or DataFrame objects")
            self._aligned(values, complete=True)
            outputs = [name for name in values.columns if name not in ("timestamp", "source_index")]
            if not outputs:
                raise ValueError(f"Feature {alias!r} has no output columns")
            for name in outputs:
                output = alias if len(outputs) == 1 else f"{alias}_{name}"
                if output in used:
                    raise ValueError(f"Feature output {output!r} conflicts with an existing column")
                used.add(output)
                columns.append(values[name].alias(output))
        return frame.with_columns(columns)

    def _indicator(self, name, parameters):
        key = ("indicator", indicators.ALGORITHM_VERSION, self._fingerprint, name,
               json.dumps(parameters, sort_keys=True, allow_nan=False))
        if key not in self._cache:
            result = getattr(indicators, name)(self._history, **parameters)
            frame = result.frame.with_columns(self._history["source_index"])
            self._cache[key] = Feature(frame, result.metadata)
        cached = self._cache[key]
        if cached.metadata not in self._calculations:
            self._calculations.append(cached.metadata)
        return Feature(cached.frame, cached.metadata)

    def moving_average(self, period=20, *, kind="sma", price_col="close"):
        return self._indicator("moving_average", {"period": period, "kind": kind, "price_col": price_col})

    def sma(self, period=20, *, price_col="close"):
        return self.moving_average(period, kind="sma", price_col=price_col)

    def ema(self, period=20, *, price_col="close"):
        return self.moving_average(period, kind="ema", price_col=price_col)

    def wma(self, period=20, *, price_col="close"):
        return self.moving_average(period, kind="wma", price_col=price_col)

    def rsi(self, period=14, *, price_col="close"):
        return self._indicator("rsi", {"period": period, "price_col": price_col})

    def macd(self, fast=12, slow=26, signal=9, *, price_col="close"):
        return self._indicator("macd", {"fast": fast, "slow": slow, "signal": signal, "price_col": price_col})

    def atr(self, period=14, *, normalized=False):
        return self._indicator("atr", {"period": period, "normalized": normalized})

    def bollinger_bands(self, period=20, *, upper_deviation=2., lower_deviation=2., price_col="close"):
        return self._indicator("bollinger_bands", {"period": period, "upper_deviation": upper_deviation,
                                                   "lower_deviation": lower_deviation, "price_col": price_col})

    def adx(self, period=14):
        return self._indicator("adx", {"period": period})

    def stochastic(self, period=14, *, k_smoothing=3, d_period=3):
        return self._indicator("stochastic", {"period": period, "k_smoothing": k_smoothing, "d_period": d_period})

    def obv(self, *, price_col="close"):
        return self._indicator("obv", {"price_col": price_col})

    def at(self, index_or_timestamp):
        """Return a restricted context through an original bar identity.

        Integer arguments are source_index values, not positions in history.
        Timestamp arguments must identify an available bar exactly. Calculations
        on the returned context cannot see later observations.
        """
        if isinstance(index_or_timestamp, Integral) and not isinstance(index_or_timestamp, bool):
            match = self._history.filter(pl.col("source_index") == int(index_or_timestamp))
        elif isinstance(index_or_timestamp, (datetime, date)):
            try:
                match = self._history.filter(pl.col("timestamp") == pl.lit(index_or_timestamp))
            except pl.exceptions.PolarsError as error:
                raise ValueError("As-of timestamp must match the history timezone") from error
        else:
            raise ValueError("As-of requires a source_index or timestamp")
        if match.height != 1:
            raise ValueError("As-of target must identify one available bar")
        history = self._history.filter(pl.col("source_index") <= match["source_index"][0])
        return self._create(history, self._original, self._cache, as_of=match["timestamp"][0])

    def trends(self, config=None, *, price_col="close", include_history=False, progress=None, **settings):
        """Return causal historical trend records, with observation endpoints.

        A trend is observable at its end_index/end_time, never at its fitted
        starting point. Earlier initialization endpoints are omitted by default;
        fitted intervals may still start at negative initialization indices.
        """
        from . import trend

        if not isinstance(include_history, bool):
            raise ValueError("include_history must be a boolean")
        if config is not None and settings:
            raise ValueError("Supply a TrendConfig or keyword settings, not both")
        if config is None:
            config = trend.TrendConfig(**settings)
        elif isinstance(config, dict):
            config = trend.TrendConfig.from_dict(config)
        if not isinstance(config, trend.TrendConfig):
            raise ValueError("config must be a TrendConfig or settings dictionary")
        key = ("trend", trend.ALGORITHM_VERSION, self._fingerprint, price_col,
               json.dumps(config.to_dict(), sort_keys=True))
        if key not in self._cache:
            self._cache[key] = trend.analyze_history(self._history, price_col=price_col,
                                                     config=config, progress=progress)
        result = self._cache[key]
        provenance = {"indicator": "trend", "parameters": config.to_dict(), "price_col": price_col,
                      "algorithm_version": trend.ALGORITHM_VERSION, "input_rows": self._history.height,
                      "observation_endpoint": "end_index/end_time"}
        if provenance not in self._calculations:
            self._calculations.append(provenance)
        offset = int(self._history["source_index"][0]) if self._history.height else 0
        identifiers = {item.trend_id: f"T{item.start_index + offset}-{item.end_index + offset}-{'up' if item.direction > 0 else 'down'}"
                       for item in result.trends}
        trends = tuple(replace(item, trend_id=identifiers[item.trend_id],
                               start_index=item.start_index + offset, end_index=item.end_index + offset,
                               parent_id=identifiers.get(item.parent_id),
                               nearby_start_indices=tuple(index + offset for index in item.nearby_start_indices))
                       for item in result.trends if include_history or item.end_index + offset >= 0)
        diagnostics = tuple(replace(item, start_index=item.start_index + offset, end_index=item.end_index + offset)
                            for item in result.diagnostics if include_history or item.end_index + offset >= 0)
        return replace(result, trends=trends, diagnostics=diagnostics, source_offset=result.source_offset + offset,
                       status="no_clear_trend" if result.status == "ok" and not trends else result.status)

    def result(self):
        from .results import AnalysisResult
        return AnalysisResult(self)
