"""Independent expected schedule boundaries, calendar compatibility, and DST cases."""
from datetime import date, time

import pytest

from dashboard.session_schedules import CUSTOM_HOURS, calendar_options, generate_schedule


def test_calendar_choices_are_discovered_deduplicated_and_compatible():
    options, excluded = calendar_options(date(2024, 10, 11), date(2024, 10, 11))
    lookup = {option.name: option for option in options}
    assert {"XNYS", "XLON", "XTSE", "XASX", "24/5", "24/7"} <= lookup.keys()
    assert options[0].name == "XNYS"
    assert lookup["XNYS"].timezone == "America/New_York"
    assert "XNAS" in lookup["XNYS"].aliases
    assert "XNAS" not in lookup
    assert not {"XHKG", "XSHG", "XTKS"} & lookup.keys()
    assert all("intraday breaks" in dict(excluded)[name] for name in ("XHKG", "XSHG", "XTKS"))


def test_exchange_mode_uses_london_hours():
    assert generate_schedule("XLON", date(2024, 10, 11), date(2024, 10, 11)) == [{
        "label": "2024-10-11", "open": "2024-10-11T07:00:00+00:00", "close": "2024-10-11T15:30:00+00:00",
    }]


def test_custom_hours_convert_dst_per_date_and_exclude_weekends():
    assert generate_schedule("XNYS", date(2024, 11, 1), date(2024, 11, 4), mode=CUSTOM_HOURS,
                             opening=time(4), closing=time(20), timezone="America/New_York") == [
        {"label": "2024-11-01", "open": "2024-11-01T08:00:00+00:00", "close": "2024-11-02T00:00:00+00:00"},
        {"label": "2024-11-04", "open": "2024-11-04T09:00:00+00:00", "close": "2024-11-05T01:00:00+00:00"},
    ]


def test_custom_hours_preserve_holidays_but_explicitly_override_early_closes():
    regular = generate_schedule("XNYS", date(2024, 11, 28), date(2024, 11, 29))
    assert regular == [{"label": "2024-11-29", "open": "2024-11-29T14:30:00+00:00", "close": "2024-11-29T18:00:00+00:00"}]
    custom = generate_schedule("XNYS", date(2024, 11, 28), date(2024, 11, 29), mode=CUSTOM_HOURS)
    assert custom == [{"label": "2024-11-29", "open": "2024-11-29T09:00:00+00:00", "close": "2024-11-30T01:00:00+00:00"}]


@pytest.mark.parametrize("opening_day,expected_open,expected_close", [
    ("Session date", "2024-11-04T23:00:00+00:00", "2024-11-05T22:00:00+00:00"),
    ("Previous calendar day", "2024-11-03T23:00:00+00:00", "2024-11-04T22:00:00+00:00"),
])
def test_overnight_hours_keep_the_calendar_session_label(opening_day, expected_open, expected_close):
    assert generate_schedule("XNYS", date(2024, 11, 4), date(2024, 11, 4), mode=CUSTOM_HOURS,
                             opening=time(18), closing=time(17), opening_day=opening_day) == [{
        "label": "2024-11-04", "open": expected_open, "close": expected_close,
    }]


def test_full_day_window_respects_dst_change():
    assert generate_schedule("24/7", date(2024, 11, 2), date(2024, 11, 2), mode=CUSTOM_HOURS,
                             timezone="America/New_York", opening=time(4), closing=time(4)) == [{
        "label": "2024-11-02", "open": "2024-11-02T08:00:00+00:00", "close": "2024-11-03T09:00:00+00:00",
    }]


@pytest.mark.parametrize("day,opening,message", [
    (date(2024, 3, 10), time(2, 30), "does not exist"),
    (date(2024, 11, 3), time(1, 30), "is ambiguous"),
])
def test_custom_hours_reject_ambiguous_or_nonexistent_local_times(day, opening, message):
    with pytest.raises(ValueError, match=message):
        generate_schedule("24/7", day, day, mode=CUSTOM_HOURS,
                          timezone="America/New_York", opening=opening, closing=time(20))


@pytest.mark.parametrize("mode", ["Exchange regular hours", CUSTOM_HOURS])
def test_generation_cannot_bypass_intraday_break_restriction(mode):
    with pytest.raises(ValueError, match="intraday breaks"):
        generate_schedule("XHKG", date(2024, 10, 11), date(2024, 10, 15), mode=mode)


def test_custom_timezone_is_independent_of_calendar_dates():
    assert generate_schedule("XNYS", date(2024, 10, 11), date(2024, 10, 11), mode=CUSTOM_HOURS,
                             timezone="UTC", opening=time(9), closing=time(17)) == [{
        "label": "2024-10-11", "open": "2024-10-11T09:00:00+00:00", "close": "2024-10-11T17:00:00+00:00",
    }]


def test_empty_and_reversed_date_windows_fail_clearly():
    with pytest.raises(ValueError, match="No trading sessions"):
        generate_schedule("XNYS", date(2024, 1, 1), date(2024, 1, 1), mode=CUSTOM_HOURS)
    with pytest.raises(ValueError, match="last date"):
        generate_schedule("XNYS", date(2024, 1, 3), date(2024, 1, 2))
