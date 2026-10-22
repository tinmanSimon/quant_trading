"""US-equity 30-minute bars restricted to the exchange's regular sessions."""

from dataclasses import dataclass, field
from datetime import UTC, date, timedelta
from functools import lru_cache
from typing import ClassVar

from ...exceptions import ProcessingError
from .intraday import ResampleIntraday
from .resampling import TradingSession


@lru_cache(maxsize=16)
def _regular_sessions(first: date, last: date) -> tuple[TradingSession, ...]:
    import exchange_calendars as xcals

    if type(first) is not date or type(last) is not date or first > last:
        raise ProcessingError("Supply ordered first and last session dates (inclusive).")
    # Padding permits a single date or a holiday-only selection; slice back to
    # the requested dates so neighboring sessions cannot enter the recipe.
    calendar = xcals.get_calendar("XNYS", start=str(first - timedelta(days=7)),
                                 end=str(last + timedelta(days=7)))
    return tuple(TradingSession(label.date(), row["open"].to_pydatetime().astimezone(UTC),
                                row["close"].to_pydatetime().astimezone(UTC))
                 for label, row in calendar.schedule.loc[str(first):str(last)].iterrows())


@dataclass(frozen=True, slots=True)
class ResampleRegularHours30m(ResampleIntraday):
    """Aggregate only XNYS regular hours, using pinned calendar boundaries.

    Normal sessions yield 13 bars starting at 09:30, 10:00, ..., 15:30 New York.
    Holidays, DST, and early closes follow the exchange calendar. Extended-hours
    input is excluded before aggregation. Missing regular-session minutes raise
    by default; ``incomplete='drop'`` omits their entire interval with a warning.
    ``incomplete='aggregate_available'`` aggregates observed minutes and warns;
    wholly empty intervals remain absent, and gaps are not classified as halts.
    """

    target_timeframe: str = field(default="30m", init=False)

    name: ClassVar[str] = "resample_regular_hours_30m"
    version: ClassVar[str] = "1"
    label: ClassVar[str] = "Regular hours: 1m to 30m"
    description: ClassVar[str] = (
        "Combine 1-minute bars into 30-minute US-equity bars during XNYS regular hours only. "
        "Normally 09:30–16:00 New York, with DST, holidays and early closes. "
        "Pre-market and after-hours bars are excluded; missing minutes are never filled."
    )
    config_schema: ClassVar[dict] = {"properties": {
        "sessions": {"title": "Regular trading sessions", "format": "trading_sessions", "minItems": 1,
                     "regular_only": True, "calendar": "XNYS",
                     "description": "Use exchange regular sessions only. Exact UTC boundaries are saved in the recipe."},
        "incomplete": {"title": "Incomplete intervals", "enum": ["raise", "drop", "aggregate_available"],
                       "description": "raise: fail on gaps; drop: omit affected 30-minute bars; aggregate_available: use available regular-session minutes and warn. Empty intervals stay absent; gaps are not verified as trading halts."},
    }}

    def __post_init__(self):
        ResampleIntraday.__post_init__(self)
        expected = {session.label: session for session in _regular_sessions(
            self.sessions[0].label, self.sessions[-1].label)}
        for session in self.sessions:
            if expected.get(session.label) != session:
                raise ProcessingError(
                    f"Session {session.label} must match XNYS regular hours exactly, including early closes. "
                    "Custom hours, weekends and exchange holidays are not allowed."
                )

    @property
    def config(self) -> dict:
        # The timeframe is fixed, so it is not a user-configurable constructor
        # field. Preserve the complete session schedule for saved-run replay.
        return {"sessions": [session.to_config() for session in self.sessions], "incomplete": self.incomplete}

    @classmethod
    def for_dates(cls, first: date, last: date, *, incomplete="raise"):
        """Construct regular sessions for an inclusive local-date range."""
        sessions = _regular_sessions(first, last)
        if not sessions:
            raise ProcessingError("No XNYS regular sessions fall within the selected dates.")
        return cls(sessions, incomplete=incomplete)
