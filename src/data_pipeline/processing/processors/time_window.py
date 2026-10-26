"""Filter intraday bars by a recurring local-clock window without altering bars."""
from dataclasses import dataclass
from datetime import time, timedelta
from typing import ClassVar
from zoneinfo import ZoneInfo, available_timezones

import polars as pl

from ...exceptions import ProcessingError
from ..contracts import DataContract
from .base import BaseProcessor


_TIMEZONES = tuple(sorted(available_timezones() - {"localtime"}))


@dataclass(frozen=True, slots=True)
class TimeWindowFilter(BaseProcessor):
    """Keep bar starts in [start_time, end_time) on every date in timezone.

    A start later than the end crosses midnight; equal times keep the full day.
    DST follows the timezone of each source instant. Retained OHLCV values and
    UTC timestamps are unchanged. This does not split boundary-crossing bars,
    infer exchange holidays/early closes, or make daily bars intraday-aware.
    """

    start_time: time = time(9)
    end_time: time = time(16)
    timezone: str = "America/New_York"

    name: ClassVar[str] = "filter_time_window"
    version: ClassVar[str] = "1"
    label: ClassVar[str] = "Filter by time window"
    description: ClassVar[str] = (
        "Keep intraday bars whose start timestamps fall within a recurring local-time window. "
        "Start is inclusive; end is exclusive. Retained bars are unchanged, not split. "
        "Apply before daily resampling."
    )
    causal: ClassVar[bool] = True
    price_basis: ClassVar[str] = "preserve"
    config_schema: ClassVar[dict] = {"properties": {
        "start_time": {"title": "Start time (inclusive)", "description": "Local clock time, e.g. 09:00. No UTC offset."},
        "end_time": {"title": "End time (exclusive)", "description": (
            "Local clock time, e.g. 16:00. An end earlier than the start crosses midnight; "
            "equal times keep the full day."
        )},
        "timezone": {"title": "Window timezone", "enum": list(_TIMEZONES), "description": (
            "Times are interpreted in this timezone on every date, including daylight-saving changes. "
            "UTC timestamps in retained bars are not changed."
        )},
    }}

    def __post_init__(self):
        for field in ("start_time", "end_time"):
            value = getattr(self, field)
            if type(value) is not time or value.tzinfo is not None:
                raise ProcessingError(f"{field} must be a local datetime.time without an offset; configure timezone separately.")
        if not isinstance(self.timezone, str) or self.timezone not in _TIMEZONES:
            raise ProcessingError("Choose a valid named timezone, such as America/New_York or UTC.")
        ZoneInfo(self.timezone)

    def output_contract(self, input_contract: DataContract) -> DataContract:
        if input_contract.timestamp_convention != "bar_start":
            raise ProcessingError("Time-window filtering requires intraday bars; apply it before daily or longer resampling.")
        return input_contract

    def _validate_input(self, frame, input_contract):
        # The shared wrapper validates OHLCV; output_contract checks intraday semantics.
        pass

    def validate_backtest_calendar(self, bars, *, input_contract, output_contract):
        """Reject selected output intervals whose required inputs are filtered.

        This also runs when a later processor aggregates into longer bars. A
        complete raw parent cannot prove that those intermediate inputs survived
        this filter, even if the final aggregation has an output for every day.
        """
        BaseProcessor.validate_backtest_calendar(self, bars, input_contract=input_contract,
                                                output_contract=output_contract)
        self.output_contract(input_contract)
        timeframe = input_contract.timeframe
        duration = timedelta(minutes=int(timeframe[:-1]) * (60 if timeframe.endswith("h") else 1))
        zone = ZoneInfo(self.timezone)
        missing, examples = 0, []
        for bar in bars:
            stamp = bar.open
            while stamp < bar.close:
                clock = stamp.astimezone(zone).time()
                included = (self.start_time <= clock < self.end_time if self.start_time < self.end_time
                            else clock >= self.start_time or clock < self.end_time)
                if not included:
                    missing += 1
                    if len(examples) < 5:
                        examples.append(stamp.isoformat())
                stamp += duration
        if missing:
            raise ProcessingError(
                f"Time-window filter removes {missing} required {timeframe} source bars from selected "
                f"trading/warm-up intervals (UTC): {', '.join(examples)}. "
                "Later aggregation cannot establish complete source coverage after those exclusions."
            )

    def _window_mask(self):
        clock = pl.col("timestamp").dt.convert_time_zone(self.timezone).dt.time()
        after_start = clock >= pl.lit(self.start_time)
        before_end = clock < pl.lit(self.end_time)
        return after_start & before_end if self.start_time < self.end_time else after_start | before_end

    def _transform(self, frame, input_contract, output_contract):
        output = frame.filter(self._window_mask())
        if output.is_empty():
            raise ProcessingError(
                "No bars remain inside the configured time window. Check the window, timezone, and selected input data."
            )
        return output

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        if not output_frame.equals(input_frame.filter(self._window_mask())):
            raise ProcessingError("Time-window output must contain exactly the matching input bars with unchanged values.")
