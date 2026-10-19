"""Calendar discovery and explicit continuous-session schedule generation."""
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals


REGULAR_HOURS = "Exchange regular hours"
CUSTOM_HOURS = "Custom session hours"


@dataclass(frozen=True)
class CalendarOption:
    name: str
    timezone: str
    aliases: tuple[str, ...] = ()

    @property
    def label(self):
        aliases = f" ({', '.join(self.aliases)})" if self.aliases else ""
        return f"{self.name} · {self.timezone}{aliases}"


def _calendar(name, first, last):
    # Padding permits single-day and holiday-only selections. The returned
    # schedule is always sliced back to the exact requested session dates.
    return xcals.get_calendar(name, start=str(first - timedelta(days=7)),
                              end=str(last + timedelta(days=7)))


def _has_breaks(schedule):
    return any(schedule[column].notna().any() for column in ("break_start", "break_end"))


@lru_cache(maxsize=16)
def _calendar_options(first, last, names):
    options, excluded = [], []
    aliases = xcals.names_to_aliases()
    for name in names:
        try:
            calendar = _calendar(name, first, last)
            selected = calendar.schedule.loc[str(first):str(last)]
            # Keep closed-day selections available so generation can explain
            # that no sessions exist. Check the surrounding schedule for breaks.
            if _has_breaks(selected if not selected.empty else calendar.schedule):
                excluded.append((name, "Scheduled intraday breaks are not supported by the current resampler."))
                continue
            options.append(CalendarOption(name, str(calendar.tz), tuple(sorted(aliases.get(name, [])))))
        except Exception as error:
            excluded.append((name, f"Calendar unavailable for these dates: {error}"))
    return tuple(sorted(options, key=lambda option: (option.name != "XNYS", option.name))), tuple(excluded)


def calendar_options(first: date, last: date):
    """Offer only installed calendars with continuous sessions in this window.

    Cache metadata, not the large schedules; registration of a new calendar
    changes the names key. Exact schedule validation is repeated at generation.
    """
    return _calendar_options(min(first, last), max(first, last),
                             tuple(xcals.get_calendar_names(include_aliases=False)))


def _local_instant(day: date, clock: time, timezone: ZoneInfo) -> datetime:
    naive = datetime.combine(day, clock)
    if clock.tzinfo is not None:
        raise ValueError("Enter local clock times without offsets; select their timezone separately.")
    candidates = set()
    for fold in (0, 1):
        instant = naive.replace(tzinfo=timezone, fold=fold).astimezone(UTC)
        if instant.astimezone(timezone).replace(tzinfo=None) == naive:
            candidates.add(instant)
    if not candidates:
        raise ValueError(f"{naive.isoformat()} does not exist in {timezone.key} due to a clock change. Choose another time.")
    if len(candidates) != 1:
        raise ValueError(f"{naive.isoformat()} is ambiguous in {timezone.key} due to a clock change. "
                         "Choose another time or enter explicit offset-aware times in the schedule JSON.")
    return candidates.pop()


def generate_schedule(name: str, first: date, last: date, *, mode=REGULAR_HOURS,
                      timezone: str | None = None, opening: time = time(4), closing: time = time(20),
                      opening_day: str = "Session date") -> list[dict]:
    if first > last:
        raise ValueError("The last date must be on or after the first date.")
    if mode not in (REGULAR_HOURS, CUSTOM_HOURS):
        raise ValueError("Choose exchange regular hours or custom session hours.")
    calendar = _calendar(name, first, last)
    schedule = calendar.schedule.loc[str(first):str(last)]
    if schedule.empty:
        raise ValueError("No trading sessions fall within the selected dates.")
    if _has_breaks(schedule):
        raise ValueError("Calendars with scheduled intraday breaks are not supported by the current resampler.")
    if opening_day not in ("Session date", "Previous calendar day"):
        raise ValueError("Choose a supported opening day.")
    zone = ZoneInfo(timezone or str(calendar.tz)) if mode == CUSTOM_HOURS else None
    sessions = []
    previous_close = None
    for label, row in schedule.iterrows():
        day = label.date()
        if mode == REGULAR_HOURS:
            start = row["open"].to_pydatetime().astimezone(UTC)
            end = row["close"].to_pydatetime().astimezone(UTC)
        else:
            start_day = day - timedelta(days=opening_day == "Previous calendar day")
            end_day = start_day + timedelta(days=closing <= opening)
            start = _local_instant(start_day, opening, zone)
            end = _local_instant(end_day, closing, zone)
        if start >= end:
            raise ValueError(f"Session {day} must close after it opens.")
        if previous_close is not None and start < previous_close:
            raise ValueError("Generated sessions overlap. Adjust the custom hours or opening day.")
        sessions.append({"label": day.isoformat(), "open": start.isoformat(), "close": end.isoformat()})
        previous_close = end
    return sessions
