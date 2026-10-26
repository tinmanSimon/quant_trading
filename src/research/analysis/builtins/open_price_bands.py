"""Percentage levels from selected-bar or verified exchange-session opens."""

from bisect import bisect_right
from copy import deepcopy
from datetime import UTC, datetime, time, timedelta
import math
import re
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from research.analysis.core.base import BaseChartLogic, LogicResult


class OpenPriceBandsLogic(BaseChartLogic):
    name = "open_price_bands"
    version = "1.0.0"
    result_version = "2"
    label = "Open price bands"
    description = "Plot percentage levels above and below a chosen opening price."
    required_columns = ("timestamp", "open")
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reference": {
                "type": "string", "enum": ["session", "first_bar", "each_bar"],
                "default": "each_bar", "title": "Open reference",
                "description": "Each bar's own open (default). Session uses one fixed open per "
                "trading day; first_bar uses the first selected bar's open for the entire selection.",
            },
            "upper_percent": {
                "type": "number", "minimum": 0, "default": 5.0, "title": "Upper offset (%)",
            },
            "lower_percent": {
                "type": "number", "minimum": 0, "maximum": 100, "default": 5.0,
                "title": "Lower offset (%)",
            },
            "session_hours": {
                "type": "string", "enum": ["regular", "extended"], "default": "regular",
                "title": "Session hours",
                "description": "For session reference. Regular follows the exchange calendar. "
                "Extended uses 04:00–20:00 New York time, ending at 17:00 on early-close days. "
                "Daily bars use the provider's daily open and cannot separate extended hours.",
            },
            "calendar": {
                "type": "string", "enum": ["XNYS"], "default": "XNYS", "title": "Session calendar",
                "description": "US equities trading calendar, including holidays and early closes.",
            },
        },
    }

    display_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "show_upper": {"type": "boolean", "default": True, "title": "Show upper level"},
            "show_lower": {"type": "boolean", "default": True, "title": "Show lower level"},
            "show_open": {"type": "boolean", "default": True, "title": "Show open reference"},
        },
    }

    def validate(self, config):
        values = {**{name: field["default"] for name, field in self.config_schema["properties"].items()},
                  **config}
        for name in ("upper_percent", "lower_percent"):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if values["lower_percent"] > 100:
            raise ValueError("Lower offset cannot exceed 100%.")
        if values["reference"] not in ("session", "first_bar", "each_bar"):
            raise ValueError("Choose session, first_bar, or each_bar as the open reference.")
        if values["session_hours"] not in ("regular", "extended") or values["calendar"] != "XNYS":
            raise ValueError("Choose supported session hours and the XNYS calendar.")
        return values

    def calculate(self, context, config, progress=None):
        values = self.validate(config)
        frame = context.frame
        if frame.is_empty():
            return LogicResult(messages=["No selected bars are available for open price bands."])
        stamps = frame["timestamp"].to_list()
        opens = frame["open"].to_list()
        if any(value is None or not math.isfinite(value) or value <= 0 for value in opens):
            raise ValueError("Open price bands require finite positive opening prices.")
        reference = values["reference"]
        if reference == "first_bar":
            groups = [("selected", list(range(frame.height)), [float(opens[0])] * frame.height)]
            messages = ["Reference: the first selected bar's opening price."]
        elif reference == "each_bar":
            groups = [("each_bar", list(range(frame.height)), [float(value) for value in opens])]
            messages = ["Each bar uses its own opening price."]
        else:
            groups, messages = _session_groups(context, values, stamps, opens)
        levels = [
            ("upper", values["upper_percent"], "#16a34a", "dash"),
            ("lower", -values["lower_percent"], "#dc2626", "dash"),
            ("open", 0, "#64748b", "dot"),
        ]
        rows = []
        series = {(kind, mode): {"x": [], "y": []}
                  for kind, *_ in levels for mode in ("line", "markers")}
        for group_id, indices, reference_opens in groups:
            if not indices:
                continue
            plot_indices = indices
            plot_opens = reference_opens
            if reference != "each_bar" and len(indices) > 2:
                plot_indices = [indices[0], indices[-1]]
                plot_opens = [reference_opens[0], reference_opens[-1]]
            mode = "markers" if len(indices) == 1 else "line"
            for kind, percent, color, dash in levels:
                target = series[kind, mode]
                if mode == "line" and target["x"]:
                    # Null separates sessions in a single trace. Keeping only
                    # segment endpoints avoids multiplying bars by levels.
                    target["x"].append(plot_indices[0])
                    target["y"].append(None)
                target["x"].extend(plot_indices)
                target["y"].extend(value * (1 + percent / 100) for value in plot_opens)
            for index, opening in zip(indices, reference_opens):
                rows.append({
                    "source_index": index, "timestamp": stamps[index].isoformat(),
                    "reference": group_id, "open_reference": opening,
                    "upper": opening * (1 + values["upper_percent"] / 100),
                    "lower": opening * (1 - values["lower_percent"] / 100),
                })
        overlays = []
        for kind, percent, color, dash in levels:
            label = "Open" if kind == "open" else f"Open {percent:+g}%"
            for mode in ("line", "markers"):
                target = series[kind, mode]
                if not target["x"]:
                    continue
                overlay = {
                    "type": mode, "id": f"{mode}:{kind}", "name": label, **target,
                    "hovertemplate": f"{label}<br>Price: %{{y:.6g}}<extra></extra>",
                }
                if mode == "markers":
                    overlay["marker"] = {"color": color, "size": 6}
                else:
                    overlay["line"] = {"color": color, "width": 1.5, "dash": dash}
                overlays.append(overlay)
        if progress is not None:
            progress(1, 1)
        return LogicResult(overlays=overlays, tables=[{"name": "Open price levels", "rows": rows}],
                           messages=messages)

    def present(self, result, config):
        values = {**{name: field["default"] for name, field in self.display_schema["properties"].items()},
                  **config}
        output = deepcopy(result)
        output.overlays = [
            overlay for overlay in output.overlays
            if values[f"show_{overlay['id'].rsplit(':', 1)[-1]}"]
        ]
        if result.overlays and not output.overlays:
            output.messages.append(
                "All open-price levels are hidden. Enable an upper, lower, or open level to show it."
            )
        return output


def _session_groups(context, values, stamps, opens):
    if context.timeframe != "1d" and not re.fullmatch(r"[1-9][0-9]*(m|h)", context.timeframe):
        raise ValueError("Session open requires intraday or 1d bars; use first_bar or each_bar for this timeframe.")
    if context.timeframe != "1d" and any(
        stamp.tzinfo is None or stamp.utcoffset() is None for stamp in stamps
    ):
        raise ValueError("Session open requires timezone-aware intraday timestamps.")
    zone = ZoneInfo("America/New_York")
    # Daily vendor timestamps are calendar labels, not New York instants.
    days = [stamp.date() if context.timeframe == "1d" else stamp.astimezone(zone).date()
            for stamp in stamps]
    calendar = xcals.get_calendar(
        values["calendar"], start=str(min(days) - timedelta(days=7)),
        end=str(max(days) + timedelta(days=7)),
    )
    schedule = calendar.schedule.loc[str(min(days)):str(max(days))]
    session_days = {label.date() for label in schedule.index}
    messages, groups = [], []
    if context.timeframe == "1d":
        excluded = 0
        for index, day in enumerate(days):
            if day in session_days:
                groups.append((day.isoformat(), [index], [float(opens[index])]))
            else:
                excluded += 1
        if excluded:
            messages.append(f"{excluded:,} daily bars are outside the chosen exchange's trading sessions.")
        messages.append("Daily session levels use the provider's daily opening price.")
        if values["session_hours"] == "extended":
            messages.append(
                "Daily bars cannot distinguish regular and extended opens; "
                "the provider's daily open is used. Choose intraday bars for an extended-session reference."
            )
        return groups, messages

    selected_by_time = dict(zip(stamps, opens))
    support = context.support_frame
    support_by_time = {}
    if support is not None and not support.is_empty():
        support_by_time = dict(support.select("timestamp", "open").iter_rows())
    # Selected bars win over supporting data if both contain the same timestamp.
    supporting_opens = {**support_by_time, **selected_by_time}
    sessions = []
    for label, row in schedule.iterrows():
        day = label.date()
        regular_start = row["open"].to_pydatetime().astimezone(UTC)
        regular_end = row["close"].to_pydatetime().astimezone(UTC)
        if values["session_hours"] == "regular":
            start, end = regular_start, regular_end
        else:
            # The fixed 04–20 US-equity convention is explicit in the control,
            # with 17:00 late close on exchange-calendar early-close days:
            # https://www.nyse.com/trade/hours-calendars
            late_close = 17 if regular_end.astimezone(zone).time() < time(16) else 20
            start = datetime.combine(day, time(4), zone).astimezone(UTC)
            end = datetime.combine(day, time(late_close), zone).astimezone(UTC)
        sessions.append((day, start, end))
    starts = [entry[1] for entry in sessions]
    grouped = {}
    outside = 0
    for index, stamp in enumerate(stamps):
        session_index = bisect_right(starts, stamp) - 1
        if session_index < 0 or stamp >= sessions[session_index][2]:
            outside += 1
            continue
        grouped.setdefault(session_index, []).append(index)
    missing = []
    for session_index, indices in grouped.items():
        day, opening_stamp, _ = sessions[session_index]
        opening = supporting_opens.get(opening_stamp)
        if opening is None or not math.isfinite(opening) or opening <= 0:
            missing.append(day.isoformat())
            continue
        groups.append((day.isoformat(), indices, [float(opening)] * len(indices)))
    if missing:
        listed = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        messages.append(
            f"Session open unavailable for {len(missing):,} session(s): {listed}. "
            "Select data including the opening bar or provide supporting bars. "
            "The first visible bar was not substituted."
        )
    if outside:
        messages.append(f"{outside:,} selected bars are outside the chosen session hours.")
    messages.append(
        f"Reference: verified {values['session_hours']} session open in America/New_York. "
        "Levels reset each trading session."
    )
    return groups, messages
