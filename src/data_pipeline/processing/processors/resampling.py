"""Hourly-to-daily OHLCV using explicit, immutable session schedules.

No exchange calendar is guessed. Each session is one continuous [open, close)
window with hourly bar starts anchored at open; the final bar may be shorter.
Supply actual UTC offsets for DST and explicit early-close windows. Exchanges
with intraday breaks or differently anchored bars need another implementation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import math
from typing import ClassVar

import polars as pl

from ...exceptions import ProcessingError
from ...schemas import CANONICAL_OHLCV_COLUMNS
from .base import BaseProcessor
from ..contracts import DataContract


@dataclass(frozen=True, slots=True)
class TradingSession:
    """Explicit session date label and actual aware opening/closing instants."""

    label: date
    open: datetime
    close: datetime

    def __post_init__(self):
        if type(self.label) is not date:
            raise ProcessingError("Session label must be a date, not a datetime.")
        for field in ("open", "close"):
            value = getattr(self, field)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ProcessingError(f"Session {field} must be a timezone-aware datetime.")
            if value.microsecond % 1000:
                raise ProcessingError("Session times must have millisecond precision.")
            object.__setattr__(self, field, value.astimezone(UTC))
        if self.open >= self.close:
            raise ProcessingError("Session open must be before close.")

    @property
    def expected_bars(self) -> int:
        return math.ceil((self.close - self.open) / timedelta(hours=1))

    def to_config(self) -> dict:
        return {"label": self.label.isoformat(), "open": self.open.isoformat(), "close": self.close.isoformat()}

    @classmethod
    def from_config(cls, spec: Mapping) -> "TradingSession":
        if not isinstance(spec, Mapping) or set(spec) != {"label", "open", "close"}:
            raise ProcessingError("Session config requires exactly label, open and close.")
        try:
            return cls(date.fromisoformat(spec["label"]), datetime.fromisoformat(spec["open"]),
                       datetime.fromisoformat(spec["close"]))
        except (TypeError, ValueError) as error:
            raise ProcessingError(f"Invalid session configuration: {error}") from error


@dataclass(frozen=True, slots=True)
class ResampleOHLCV(BaseProcessor):
    """Aggregate 1h input into 1d session labels; incomplete='raise' or 'drop'.

    The schedule is part of config/fingerprint and must list every intended
    session, including sessions entirely missing from the input. 'drop' omits
    incomplete groups but never discards out-of-session or off-grid input.
    All-empty output remains an error. No filling or partial daily bars.
    """

    sessions: tuple[TradingSession, ...]
    incomplete: str = "raise"
    target_timeframe: str = "1d"
    name: ClassVar[str] = "resample_ohlcv"
    version: ClassVar[str] = "1"

    def __post_init__(self):
        if self.incomplete not in ("raise", "drop") or self.target_timeframe != "1d":
            raise ProcessingError("Resampling supports target_timeframe='1d' and incomplete='raise' or 'drop'.")
        if not isinstance(self.sessions, (list, tuple)) or not self.sessions:
            raise ProcessingError("Supply a nonempty ordered list/tuple of TradingSession objects.")
        if any(not isinstance(session, TradingSession) for session in self.sessions):
            raise ProcessingError("Every session must be a TradingSession.")
        sessions = tuple(sorted(self.sessions, key=lambda session: session.open))
        if len({session.label for session in sessions}) != len(sessions):
            raise ProcessingError("Session labels must be unique.")
        for previous, current in zip(sessions, sessions[1:]):
            if previous.close > current.open or previous.label >= current.label:
                raise ProcessingError("Sessions must not overlap and labels must increase chronologically.")
        object.__setattr__(self, "sessions", sessions)

    @property
    def config(self) -> dict:
        return {"sessions": [session.to_config() for session in self.sessions],
                "incomplete": self.incomplete, "target_timeframe": self.target_timeframe}

    def output_contract(self, input_contract: DataContract) -> DataContract:
        if input_contract.timeframe != "1h":
            raise ProcessingError("Hourly-to-daily resampling requires a 1h input contract.")
        return DataContract("1d")

    def _validate_input(self, frame, input_contract: DataContract):
        self._groups(frame)

    def _groups(self, frame: pl.DataFrame) -> pl.DataFrame:
        schedule = pl.DataFrame({
            "_open": [session.open for session in self.sessions],
            "_close": [session.close for session in self.sessions],
            "_label": [datetime.combine(session.label, datetime.min.time(), UTC) for session in self.sessions],
            "_expected": [session.expected_bars for session in self.sessions],
        }).with_columns(pl.col("_open", "_close", "_label").cast(pl.Datetime("ms", "UTC")))
        assigned = frame.sort("timestamp").join_asof(
            schedule, left_on="timestamp", right_on="_open", strategy="backward"
        )
        if assigned.filter(pl.col("_open").is_null() | (pl.col("timestamp") >= pl.col("_close"))).height:
            raise ProcessingError("Input contains bars outside the explicit session schedule.")
        if assigned.filter((pl.col("timestamp") - pl.col("_open")).dt.total_milliseconds() % 3_600_000 != 0).height:
            raise ProcessingError("Input hourly bars are off the session-open anchored grid.")
        counts = assigned.group_by("symbol", "_label").len()
        # Cross join includes wholly absent sessions for each observed symbol.
        expected = frame.select("symbol").unique().join(schedule, how="cross")
        checked = expected.join(counts, on=["symbol", "_label"], how="left").with_columns(pl.col("len").fill_null(0))
        incomplete = checked.filter(pl.col("len") != pl.col("_expected"))
        if incomplete.height and self.incomplete == "raise":
            missing = incomplete.select("symbol", "_label").head(5).to_dicts()
            raise ProcessingError(f"Incomplete hourly session groups: {missing}")
        complete = checked.filter(pl.col("len") == pl.col("_expected")).select("symbol", "_label")
        return assigned.join(complete, on=["symbol", "_label"], how="semi").sort("symbol", "timestamp")

    def _transform(self, frame, input_contract, output_contract):
        return self._groups(frame).group_by("symbol", "_label").agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(),
        ).rename({"_label": "timestamp"}).select(CANONICAL_OHLCV_COLUMNS).sort("symbol", "timestamp")

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        # Separate scalar aggregation checks the vectorized computation. Tests
        # additionally use hand-calculated output, independent of this helper.
        groups = self._groups(input_frame).partition_by("symbol", "_label", as_dict=True)
        if output_frame.height != len(groups):
            raise ValueError("Resampling output does not match the complete session groups.")
        for row in output_frame.iter_rows(named=True):
            group = groups.get((row["symbol"], row["timestamp"]))
            if group is None:
                raise ValueError("Resampling introduced an unexpected session key.")
            expected = {"open": group["open"][0], "high": max(group["high"]),
                        "low": min(group["low"]), "close": group["close"][-1]}
            if any(row[column] != value for column, value in expected.items()):
                raise ValueError("Resampling OHLC aggregation is incorrect.")
            if not math.isclose(row["volume"], math.fsum(group["volume"]), rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("Resampling volume aggregation is incorrect.")


def resampler_from_config(config: Mapping) -> ResampleOHLCV:
    values = dict(config)
    if "sessions" not in values or not isinstance(values["sessions"], list):
        raise ProcessingError("Resampling config requires a sessions list.")
    values["sessions"] = tuple(TradingSession.from_config(session) for session in values["sessions"])
    return ResampleOHLCV(**values)
