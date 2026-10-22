"""Full-resolution finding navigation and malformed-command rejection."""

from types import SimpleNamespace

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from research.analysis.core.base import LogicResult, LogicSpec
from research.analysis.core.runner import AppliedRun
from dashboard.charts.component import navigation_request, window_payload
from dashboard.market_data.navigation import analysis_events, event_command
from test_dashboard_trend_overlays import series


def run_for(source):
    event = {"id": "entry:12", "group": "Entries", "label": "MA crossing", "source_index": 12,
             "timestamp": source.timestamps[12].isoformat()}
    return AppliedRun("source", [LogicSpec("a", "private/example", "1", {}),
                                  LogicSpec("b", "private/example", "1", {})],
                      {key: LogicResult(data={"analysis": {"events": [event]}}) for key in ("a", "b")},
                      {}, {"a": "result-a", "b": "result-b"})


def test_events_are_namespaced_validated_and_identified_independently_of_chart_sampling():
    source = series(100)
    run = run_for(source)
    events, identity = analysis_events(run, source)
    assert [event["id"] for event in events] == ["a:entry:12", "b:entry:12"]
    assert events[0]["source_index"] == 12
    run.keys["a"] = "new-parameters"
    assert analysis_events(run, source)[1] != identity
    run.results["a"].data["analysis"]["events"] = [dict(events[0], source_index=13)]
    assert len(analysis_events(run, source)[0]) == 1
    assert analysis_events(None, source) == ([], None)


@pytest.mark.parametrize("change", [
    {"chart_id": "other-data"}, {"source_index": -1}, {"source_index": 100},
    {"source_index": True}, {"source_index": 12.0}, {"end_index": 11},
    {"sequence": 0}, {"sequence": 2**53}, {"sequence": True},
    {"result_id": ""}, {"label": None},
])
def test_invalid_navigation_commands_are_rejected(change):
    command = {"chart_id": "test", "sequence": 1, "result_id": "result", "event_id": "a:entry:12",
               "label": "Entry", "source_index": 12, "end_index": 12} | change
    assert navigation_request(command, "test", 100) is None


def test_command_preserves_event_identity_in_chart_payload():
    source = series(100)
    events, identity = analysis_events(run_for(source), source)
    command = event_command(events[0], chart_id="test", result_id=identity, sequence=2)
    payload = window_payload(source, chart_id="test", timeframe="1h", timezone="UTC", navigation_command=command)
    assert payload["navigation_command"] == command
    assert payload["navigation_result_id"] == identity


def navigation_app():
    from datetime import UTC, datetime, timedelta
    import polars as pl
    import streamlit as st
    from research.analysis.core.base import LogicResult, LogicSpec
    from research.analysis.core.runner import AppliedRun
    from dashboard.charts.series import ChartSeries
    from dashboard.market_data.navigation import render_analysis_navigation

    times = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(20)]
    frame = pl.DataFrame({"timestamp": times, "open": [10.] * 20, "high": [11.] * 20,
                          "low": [9.] * 20, "close": [10.] * 20, "volume": [5.] * 20})
    events = [{"id": str(i), "source_index": i, "timestamp": times[i].isoformat(), "group": "Entries", "label": f"Entry {i}"}
              for i in (2, 7, 12)]
    run = AppliedRun("source", [LogicSpec("one", "example", "1", {})],
                      {"one": LogicResult(data={"analysis": {"events": events}})}, {}, {"one": "result"})
    command = render_analysis_navigation(run, ChartSeries(frame, timeframe="1d"), "chart")
    st.json(command)
    st.button("Unrelated rerun")


def test_previous_next_search_and_unrelated_reruns_only_issue_new_commands_for_new_selection():
    app = AppTest.from_function(navigation_app).run()
    assert not app.exception
    assert app.json[0].value == "null"
    app.button[1].click().run()  # Next selects the first event.
    assert not app.exception
    import json
    first = json.loads(app.json[0].value)
    assert first["source_index"] == 2
    app.button[1].click().run()
    second = json.loads(app.json[0].value)
    assert second["source_index"] == 7 and second["sequence"] > first["sequence"]
    app.button[0].click().run()
    assert json.loads(app.json[0].value)["source_index"] == 2
    app.selectbox[1].set_value("one:12").run()
    third = json.loads(app.json[0].value)
    assert third["source_index"] == 12
    app.button[2].click().run()
    assert json.loads(app.json[0].value) == third
