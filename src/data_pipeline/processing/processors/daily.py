"""One-minute to daily OHLCV, restricted to XNYS regular trading sessions."""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
import math
from typing import ClassVar
import warnings

import polars as pl

from ...exceptions import ProcessingError
from ...schemas import CANONICAL_OHLCV_COLUMNS
from ..contracts import DataContract
from .base import BaseProcessor
from .intraday import ResamplingWarning
from .regular_hours import ResampleRegularHours30m, _regular_sessions
from .resampling import TradingSession


@dataclass(frozen=True, slots=True)
class ResampleRegularHoursDaily(BaseProcessor):
    """Aggregate regular-session minutes into midnight-UTC session-date labels.

    The exact exchange schedule is pinned in the recipe, including DST and
    early closes. Supply every intended session to detect wholly missing days.
    Out-of-session bars are excluded and reported. Missing minutes raise by
    default; ``drop`` omits the entire day, while ``aggregate_available`` uses
    observed minutes with a warning. Empty days are never fabricated.
    """

    sessions: tuple[TradingSession, ...]
    incomplete: str = "raise"

    name: ClassVar[str] = "resample_regular_hours_1d"
    version: ClassVar[str] = "1"
    label: ClassVar[str] = "Regular hours: 1m to 1d"
    description: ClassVar[str] = (
        "Combine 1-minute bars into daily US-equity bars during XNYS regular hours only. "
        "Normally 09:30–16:00 New York, with DST, holidays and early closes. "
        "Pre-market and after-hours bars are excluded; missing minutes are never filled."
    )
    causal: ClassVar[bool] = True
    config_schema: ClassVar[dict] = {"properties": {
        "sessions": {"title": "Regular trading sessions", "format": "trading_sessions", "minItems": 1,
                     "regular_only": True, "calendar": "XNYS",
                     "description": "Include every intended exchange session. Exact UTC boundaries are saved in the recipe."},
        "incomplete": {"title": "Incomplete sessions", "enum": ["raise", "drop", "aggregate_available"],
                       "description": "raise: fail on gaps; drop: omit affected days; aggregate_available: use available regular-session minutes and warn. Empty days stay absent; gaps are not verified as trading halts."},
    }}

    def __post_init__(self):
        # Reuse the existing schedule/policy validation without resampling into
        # intermediate 30m bars, which would hide missing underlying minutes.
        validated = ResampleRegularHours30m(self.sessions, incomplete=self.incomplete)
        object.__setattr__(self, "sessions", validated.sessions)

    @property
    def config(self) -> dict:
        return {"sessions": [session.to_config() for session in self.sessions], "incomplete": self.incomplete}

    @classmethod
    def for_dates(cls, first: date, last: date, *, incomplete="raise"):
        """Construct regular sessions for an inclusive local-date range."""
        sessions = _regular_sessions(first, last)
        if not sessions:
            raise ProcessingError("No XNYS regular sessions fall within the selected dates.")
        return cls(sessions, incomplete=incomplete)

    def output_contract(self, input_contract: DataContract) -> DataContract:
        if input_contract.timeframe != "1m":
            raise ProcessingError("Regular-hours daily resampling requires a 1m input contract.")
        return replace(input_contract, timeframe="1d", timestamp_convention=None)

    def validate_backtest_calendar(self, bars, *, input_contract, output_contract):
        if self.output_contract(input_contract) != output_contract:
            raise ProcessingError("Daily output contract does not match the execution calendar.")
        sessions = {datetime.combine(session.label, datetime.min.time(), UTC): session
                    for session in self.sessions}
        for bar in bars:
            session = sessions.get(bar.timestamp)
            if session is None or (session.open, session.close) != (bar.open, bar.close):
                raise ProcessingError(
                    f"Daily session {bar.timestamp.date()} does not match the execution calendar's "
                    "opening and completion times."
                )

    def _validate_input(self, frame, input_contract):
        if frame.select((pl.col("timestamp").dt.epoch("ms") % 60_000 != 0).any()).item():
            raise ProcessingError("Input one-minute bars are off the whole-minute grid.")

    def _assigned(self, frame):
        schedule = pl.DataFrame({
            "_open": [session.open for session in self.sessions],
            "_close": [session.close for session in self.sessions],
            "_label": [datetime.combine(session.label, datetime.min.time(), UTC) for session in self.sessions],
            "expected_minutes": [int((session.close - session.open) / timedelta(minutes=1))
                                 for session in self.sessions],
        }).with_columns(pl.col("_open", "_close", "_label").cast(pl.Datetime("ms", "UTC")))
        assigned = frame.sort("timestamp").join_asof(
            schedule, left_on="timestamp", right_on="_open", strategy="backward"
        ).filter(pl.col("_open").is_not_null() & (pl.col("timestamp") < pl.col("_close")))
        counts = assigned.group_by("symbol", "_label").len(name="observed_minutes")
        # Include wholly absent sessions for each input symbol in the report.
        checked = frame.select("symbol").unique().join(schedule, how="cross").join(
            counts, on=["symbol", "_label"], how="left"
        ).with_columns(pl.col("observed_minutes").fill_null(0).cast(pl.Int64)).with_columns(
            (pl.col("expected_minutes") - pl.col("observed_minutes")).alias("missing_minutes")
        )
        return assigned, checked

    def quality_report(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Report incomplete symbol/sessions, including wholly absent days."""
        frame = DataContract("1m").validate(frame)
        self._validate_input(frame, DataContract("1m"))
        _, checked = self._assigned(frame)
        return checked.filter(pl.col("missing_minutes") != 0).rename(
            {"_label": "timestamp", "_open": "start", "_close": "end"}
        ).sort("symbol", "timestamp")

    def _retained_sessions(self, checked):
        condition = (pl.col("observed_minutes") > 0 if self.incomplete == "aggregate_available"
                     else pl.col("missing_minutes") == 0)
        return checked.filter(condition).select("symbol", "_label")

    def _transform(self, frame, input_contract, output_contract):
        assigned, checked = self._assigned(frame)
        excluded = frame.height - assigned.height
        if excluded:
            warnings.warn(f"Excluded {excluded} one-minute bars outside XNYS regular trading sessions.",
                          ResamplingWarning, stacklevel=2)
        incomplete = checked.filter(pl.col("missing_minutes") != 0).sort("symbol", "_label")
        details = incomplete.select("symbol", "_label", "observed_minutes", "missing_minutes").head(5).to_dicts()
        if incomplete.height and self.incomplete == "raise":
            raise ProcessingError(
                f"{incomplete.height} incomplete daily sessions with missing 1m bars. First sessions: {details}"
            )
        if incomplete.height and self.incomplete == "aggregate_available":
            partial = incomplete.filter(pl.col("observed_minutes") > 0).height
            empty = incomplete.height - partial
            warnings.warn(
                f"Aggregated available minutes in {partial} incomplete daily sessions; "
                f"omitted {empty} empty sessions ({incomplete['missing_minutes'].sum()} missing 1m bars). "
                "No prices or volume were filled. Gap causes are unverified. "
                "Open/close use the first/last available regular-session minute. "
                f"First sessions: {details}", ResamplingWarning, stacklevel=2,
            )
        elif incomplete.height:
            warnings.warn(
                f"Dropped {incomplete.height} incomplete daily sessions "
                f"({incomplete['missing_minutes'].sum()} missing 1m bars). First sessions: {details}",
                ResamplingWarning, stacklevel=2,
            )
        groups = assigned.join(self._retained_sessions(checked), on=["symbol", "_label"], how="semi").sort(
            "symbol", "timestamp"
        )
        if groups.is_empty():
            raise ProcessingError("No daily bars remain after regular-hours resampling.")
        return groups.group_by("symbol", "_label").agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(),
        ).rename({"_label": "timestamp"}).select(CANONICAL_OHLCV_COLUMNS).sort("symbol", "timestamp")

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        assigned, checked = self._assigned(input_frame)
        retained = self._retained_sessions(checked)
        expected_keys = retained.rename({"_label": "timestamp"}).sort("symbol", "timestamp")
        if not output_frame.select("symbol", "timestamp").equals(expected_keys):
            raise ProcessingError("Daily output does not match the expected sessions for the missing-minute policy.")
        groups = assigned.join(retained, on=["symbol", "_label"], how="semi").sort(
            "symbol", "timestamp"
        ).partition_by("symbol", "_label", as_dict=True)
        # Check aggregation independently with scalar reductions over each day.
        for row in output_frame.iter_rows(named=True):
            group = groups[row["symbol"], row["timestamp"]]
            expected = {"open": group["open"][0], "high": max(group["high"]),
                        "low": min(group["low"]), "close": group["close"][-1],
                        "volume": math.fsum(group["volume"])}
            for column, value in expected.items():
                if not math.isclose(row[column], value, rel_tol=1e-12, abs_tol=1e-9):
                    raise ProcessingError(f"Daily {column} aggregation is incorrect.")
