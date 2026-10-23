"""Session-open aligned intraday bars with explicit missing-minute policies."""

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import ClassVar
import warnings

import polars as pl

from ...exceptions import ProcessingError
from ...schemas import CANONICAL_OHLCV_COLUMNS
from ..contracts import DataContract
from .base import BaseProcessor
from .resampling import ResampleOHLCV, TradingSession


class ResamplingWarning(UserWarning):
    """Input bars excluded or missing minutes reported during resampling."""


@dataclass(frozen=True, slots=True)
class ResampleIntraday(BaseProcessor):
    """Aggregate 1m bars within an explicit, immutable session schedule.

    The dashboard can generate the schedule from an exchange calendar, including
    DST and early closes. Its exact UTC boundaries are saved in the recipe.
    Output intervals start at the session open; the final interval ends at the
    session close even when shorter than the target interval. Bars outside the
    listed sessions are excluded and reported. Missing minutes are never filled:
    an incomplete interval raises, is dropped, or aggregates its available minutes
    with a warning. Wholly empty intervals are never fabricated. Include every
    intended session in the schedule to detect wholly absent sessions, too.
    """

    sessions: tuple[TradingSession, ...]
    target_timeframe: str = "30m"
    incomplete: str = "raise"

    name: ClassVar[str] = "resample_intraday"
    version: ClassVar[str] = "1"
    label: ClassVar[str] = "Resample intraday (1m to 30m / 1h)"
    description: ClassVar[str] = (
        "Build 30-minute or hourly bars from 1-minute bars, anchored at each session's open. "
        "Generate exchange regular hours to use 09:30 US-equity opens, DST and early closes. "
        "Bars outside the listed sessions are excluded and reported. Missing minutes are never filled."
    )
    causal: ClassVar[bool] = True
    config_schema: ClassVar[dict] = {"properties": {
        "sessions": {"title": "Trading sessions", "format": "trading_sessions", "minItems": 1,
                     "description": "Generate every intended trading session, including sessions with missing input."},
        "target_timeframe": {"title": "Output timeframe", "enum": ["30m", "1h"]},
        "incomplete": {"title": "Incomplete intervals", "enum": ["raise", "drop", "aggregate_available"],
                       "description": "raise: fail on gaps; drop: omit affected intervals; aggregate_available: use available minutes and warn. Empty intervals stay absent; gaps are not verified as trading halts."},
    }}

    def __post_init__(self):
        if self.target_timeframe not in ("30m", "1h") or self.incomplete not in ("raise", "drop", "aggregate_available"):
            raise ProcessingError("Intraday resampling supports 30m or 1h output and incomplete='raise', 'drop' or 'aggregate_available'.")
        # Share the existing immutable schedule validation, without changing the
        # hourly-to-daily processor's behavior or introducing calendar imports.
        validated = ResampleOHLCV(self.sessions)
        object.__setattr__(self, "sessions", validated.sessions)
        for session in self.sessions:
            if any(stamp.second or stamp.microsecond for stamp in (session.open, session.close)):
                raise ProcessingError("Intraday session boundaries must fall on whole minutes.")

    @property
    def config(self) -> dict:
        return {"sessions": [session.to_config() for session in self.sessions],
                "target_timeframe": self.target_timeframe, "incomplete": self.incomplete}

    def output_contract(self, input_contract: DataContract) -> DataContract:
        if input_contract.timeframe != "1m":
            raise ProcessingError("Intraday resampling requires a 1m input contract.")
        return replace(input_contract, timeframe=self.target_timeframe, timestamp_convention=None)

    def _intervals(self):
        duration = timedelta(minutes=30 if self.target_timeframe == "30m" else 60)
        for session in self.sessions:
            start = session.open
            while start < session.close:
                end = min(start + duration, session.close)
                yield start, end
                start = end

    def _schedule(self):
        intervals = tuple(self._intervals())
        return pl.DataFrame({
            "_start": [start for start, _ in intervals],
            "_end": [end for _, end in intervals],
            "expected_minutes": [int((end - start) / timedelta(minutes=1)) for start, end in intervals],
        }).with_columns(pl.col("_start", "_end").cast(pl.Datetime("ms", "UTC")))

    def _assigned(self, frame):
        schedule = self._schedule()
        assigned = frame.sort("timestamp").join_asof(
            schedule, left_on="timestamp", right_on="_start", strategy="backward"
        ).filter(pl.col("_start").is_not_null() & (pl.col("timestamp") < pl.col("_end")))
        counts = assigned.group_by("symbol", "_start").len(name="observed_minutes")
        checked = frame.select("symbol").unique().join(schedule, how="cross").join(
            counts, on=["symbol", "_start"], how="left"
        ).with_columns(pl.col("observed_minutes").fill_null(0).cast(pl.Int64)).with_columns(
            (pl.col("expected_minutes") - pl.col("observed_minutes")).alias("missing_minutes")
        )
        return assigned, checked

    def quality_report(self, frame: pl.DataFrame) -> pl.DataFrame:
        """List every incomplete output interval, including wholly absent ones."""
        frame = DataContract("1m").validate(frame)
        self._validate_input(frame, DataContract("1m"))
        _, checked = self._assigned(frame)
        return checked.filter(pl.col("missing_minutes") != 0).rename(
            {"_start": "timestamp", "_end": "end"}
        ).sort("symbol", "timestamp")

    def validate_backtest_calendar(self, bars, *, input_contract, output_contract):
        if self.output_contract(input_contract) != output_contract:
            raise ProcessingError("Intraday output contract does not match the execution calendar.")
        intervals = dict(self._intervals())
        for bar in bars:
            if bar.open != bar.timestamp or intervals.get(bar.timestamp) != bar.close:
                raise ProcessingError(
                    f"Intraday interval {bar.timestamp.isoformat()} does not match the execution calendar's "
                    "opening and completion times."
                )

    def _validate_input(self, frame, input_contract):
        if frame.select((pl.col("timestamp").dt.epoch("ms") % 60_000 != 0).any()).item():
            raise ProcessingError("Input one-minute bars are off the whole-minute grid.")

    def _retained_intervals(self, checked):
        condition = (pl.col("observed_minutes") > 0 if self.incomplete == "aggregate_available"
                     else pl.col("missing_minutes") == 0)
        return checked.filter(condition).select("symbol", "_start")

    def _transform(self, frame, input_contract, output_contract):
        assigned, checked = self._assigned(frame)
        incomplete = checked.filter(pl.col("missing_minutes") != 0)
        if incomplete.height and self.incomplete == "raise":
            raise ProcessingError(
                f"{incomplete.height} incomplete intraday intervals with missing 1m bars. "
                f"First intervals: {incomplete.select('symbol', '_start', 'missing_minutes').head(5).to_dicts()}"
            )
        if incomplete.height and self.incomplete == "aggregate_available":
            partial = incomplete.filter(pl.col("observed_minutes") > 0)
            empty = incomplete.filter(pl.col("observed_minutes") == 0)
            warnings.warn(
                f"Aggregated available minutes in {partial.height} incomplete {self.target_timeframe} intervals; "
                f"omitted {empty.height} empty intervals ({incomplete['missing_minutes'].sum()} missing 1m bars). "
                "No prices or volume were filled. Gap causes are unverified. "
                "Open/close use the first/last available minute within each interval. "
                f"First intervals: {incomplete.sort('symbol', '_start').select('symbol', '_start', 'observed_minutes', 'missing_minutes').head(5).to_dicts()}",
                ResamplingWarning, stacklevel=2,
            )
        elif incomplete.height:
            warnings.warn(
                f"Dropped {incomplete.height} incomplete {self.target_timeframe} intervals "
                f"({incomplete['missing_minutes'].sum()} missing 1m bars). "
                f"First intervals: {incomplete.select('symbol', '_start', 'missing_minutes').head(5).to_dicts()}",
                ResamplingWarning, stacklevel=2,
            )
        excluded = frame.height - assigned.height
        if excluded:
            warnings.warn(f"Excluded {excluded} one-minute bars outside the configured trading sessions.",
                          ResamplingWarning, stacklevel=2)
        retained = self._retained_intervals(checked)
        groups = assigned.join(retained, on=["symbol", "_start"], how="semi").sort("symbol", "timestamp")
        if groups.is_empty():
            if self.incomplete == "aggregate_available":
                raise ProcessingError("No observed intraday intervals remain after resampling.")
            raise ProcessingError("No complete intraday intervals remain after resampling.")
        return groups.group_by("symbol", "_start").agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(),
        ).rename({"_start": "timestamp"}).select(CANONICAL_OHLCV_COLUMNS).sort("symbol", "timestamp")

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        # Verify the output uses exactly the intervals retained by the policy. Shared
        # OHLCV validation separately enforces finite prices and valid ranges.
        assigned, checked = self._assigned(input_frame)
        retained = self._retained_intervals(checked).select(
            "symbol", pl.col("_start").alias("timestamp")
        ).sort("symbol", "timestamp")
        if not output_frame.select("symbol", "timestamp").equals(retained):
            raise ProcessingError("Intraday output does not match the expected intervals for the missing-minute policy.")
        # Independently use window reductions over the source rows to check the
        # resulting OHLCV, rather than trusting the aggregation's schema alone.
        values = assigned.sort("symbol", "timestamp").with_columns(
            pl.col("open").first().over("symbol", "_start").alias("open"),
            pl.col("high").max().over("symbol", "_start").alias("high"),
            pl.col("low").min().over("symbol", "_start").alias("low"),
            pl.col("close").last().over("symbol", "_start").alias("close"),
            pl.col("volume").sum().over("symbol", "_start").alias("volume"),
        ).unique(["symbol", "_start"]).drop("timestamp").rename({"_start": "timestamp"})
        expected = values.join(retained, on=["symbol", "timestamp"], how="semi").select(
            CANONICAL_OHLCV_COLUMNS
        ).sort("symbol", "timestamp")
        for column in ("open", "high", "low", "close", "volume"):
            if not output_frame[column].is_close(expected[column], rel_tol=1e-12, abs_tol=1e-9).all():
                raise ProcessingError(f"Intraday {column} aggregation is incorrect.")
