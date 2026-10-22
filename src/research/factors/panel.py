"""Causal per-symbol factors and cross-sectional ranks at each timestamp."""

from collections.abc import Mapping

import polars as pl

from .models import FactorBacktestSpec, FactorSpec
from .registry import FactorRegistry


def build_factor_panel(frames: Mapping[str, pl.DataFrame], spec: FactorBacktestSpec, registry=None) -> pl.DataFrame:
    if not isinstance(spec, FactorBacktestSpec):
        raise ValueError("spec must be a FactorBacktestSpec.")
    registry = registry if registry is not None else FactorRegistry()
    auxiliary = None
    if spec.portfolio.name == "inverse_volatility":
        column = spec.portfolio.config.get("volatility_factor", "volatility")
        if column not in {factor.name for factor in spec.factors}:
            if not isinstance(column, str) or not column.strip() or column in {"timestamp", "symbol", "score"} or column.startswith("__"):
                raise ValueError("volatility_factor must be a nonreserved factor column name.")
            auxiliary = (column, FactorSpec("volatility", spec.portfolio.config.get("volatility_lookback", 20)))
    # Resolve every component even for an empty input universe.
    for factor in spec.factors:
        registry.load(factor)
    parts = []
    for symbol, frame in sorted(frames.items()):
        if not isinstance(symbol, str) or not symbol or not isinstance(frame, pl.DataFrame):
            raise ValueError("frames must map nonempty symbol names to Polars DataFrames.")
        if not {"timestamp", "open", "high", "close"}.issubset(frame.columns):
            raise ValueError(f"Factor data for {symbol} requires timestamp, open, high, and close.")
        ordered = frame.sort("timestamp")
        if ordered["timestamp"].null_count() or ordered["timestamp"].n_unique() != ordered.height:
            raise ValueError(f"Factor data for {symbol} has null or duplicated timestamps.")
        part = ordered.select("timestamp").with_columns(pl.lit(symbol).alias("symbol"))
        for factor in spec.factors:
            values = registry.load(factor).calculate(ordered.clone())
            if not isinstance(values, pl.Series) or len(values) != ordered.height:
                raise ValueError(f"Factor {factor.name} must return one value per input bar.")
            values = values.cast(pl.Float64, strict=True)
            values = values.set(~values.is_finite().fill_null(False), None)
            part = part.with_columns(values.alias(factor.name))
        if auxiliary is not None:
            column, auxiliary_spec = auxiliary
            # Sizing-only feature: it never enters score ranks or completeness.
            values = registry.load(auxiliary_spec).calculate(ordered.clone()).cast(pl.Float64)
            part = part.with_columns(values.set(~values.is_finite().fill_null(False), None).alias(column))
        parts.append(part)
    if not parts:
        return pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "symbol": pl.String,
                                    **{factor.name: pl.Float64 for factor in spec.factors},
                                    **({auxiliary[0]: pl.Float64} if auxiliary else {}), "score": pl.Float64})
    return score_factor_panel(pl.concat(parts, how="vertical_relaxed"), spec)


def score_factor_panel(panel: pl.DataFrame, spec: FactorBacktestSpec) -> pl.DataFrame:
    """Rank already calculated factors within the supplied eligible universe.

    The caller filters history/warm-up eligibility before calling this helper.
    Existing scores are replaced; factor values and sizing-only features remain
    unchanged. Rows with incomplete/nonfinite factors retain a null score.
    """
    if not isinstance(spec, FactorBacktestSpec):
        raise ValueError("spec must be a FactorBacktestSpec.")
    required = {"timestamp", "symbol", *(factor.name for factor in spec.factors)}
    if not isinstance(panel, pl.DataFrame) or not required.issubset(panel.columns):
        raise ValueError("Factor panel requires timestamp, symbol, and every configured factor column.")
    if panel["timestamp"].null_count() or panel["symbol"].null_count() or panel.select(
        pl.struct("timestamp", "symbol").is_duplicated().any()
    ).item():
        raise ValueError("Factor panel requires unique, nonnull timestamp/symbol pairs.")
    if "score" in panel.columns:
        panel = panel.drop("score")
    complete = panel.filter(pl.all_horizontal(
        pl.col(factor.name).is_not_null() & pl.col(factor.name).is_finite() for factor in spec.factors))
    weight_sum = sum(factor.weight for factor in spec.factors)
    score = sum(
        (pl.col(factor.name) * factor.direction).rank(method="average").over("timestamp")
        / pl.len().over("timestamp") * (factor.weight / weight_sum)
        for factor in spec.factors
    )
    scored = complete.select("timestamp", "symbol", score.alias("score"))
    return panel.join(scored, on=["timestamp", "symbol"], how="left").sort(["timestamp", "symbol"])
