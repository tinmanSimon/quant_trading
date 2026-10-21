"""Navigate immutable analysis findings without running calculations again."""

from hashlib import sha256
import json
import time

import streamlit as st


def analysis_events(run, series):
    """Return valid namespaced events and a source/result identity.

    Event coordinates come from full results, never sampled chart descriptors.
    Invalid or stale archive positions cannot select a bar in another dataset.
    """
    if run is None:
        return [], None
    stamps = series.timestamps
    events = []
    for spec in run.specs:
        result = run.results.get(spec.instance_id)
        if result is None:
            continue
        for event in result.data.get("analysis", {}).get("events", []):
            first, last = event.get("source_index"), event.get("end_index", event.get("source_index"))
            if (type(first) is not int or type(last) is not int or not 0 <= first <= last < len(stamps)
                    or event.get("timestamp") != stamps[first].isoformat()
                    or ("end_timestamp" in event and event["end_timestamp"] != stamps[last].isoformat())
                    or not isinstance(event.get("id"), str)):
                continue
            events.append({**event, "id": f"{spec.instance_id}:{event['id']}",
                           "instance_id": spec.instance_id,
                           "group_id": f"{spec.instance_id}:{event.get('group', '')}",
                           "group_label": f"{spec.name} · {spec.instance_id[:8]} · {event.get('group', 'Findings')}",
                           "end_index": last})
    events.sort(key=lambda event: (event["source_index"], event["id"]))
    identity = sha256(json.dumps({"source": run.source_key, "keys": run.keys,
                                  "events": events}, sort_keys=True).encode()).hexdigest()
    return events, identity


def event_command(event, *, chart_id, result_id, sequence):
    """Small, serializable command; the browser chooses the display-only window."""
    return {"chart_id": chart_id, "result_id": result_id, "sequence": sequence,
            "event_id": event["id"], "label": event["label"],
            "source_index": event["source_index"], "end_index": event.get("end_index", event["source_index"])}


def request_analysis_event(instance_id, event_id, *, source_key=None, result_key=None):
    """Called by a result table selection; consumed on the next chart render."""
    st.session_state["analysis-navigation-request"] = {"id": f"{instance_id}:{event_id}",
                                                        "instance_id": instance_id,
                                                        "source_key": source_key, "result_key": result_key}


def render_analysis_navigation(run, series, chart_id, memory=None, *, event_data=None):
    """Render Previous/Next and searchable selectors, returning the last command.

    Repeated payloads deliberately retain the same sequence; the browser only
    executes each command once, so unrelated reruns never reset a manual pan.
    """
    events, identity = analysis_events(run, series) if event_data is None else event_data
    state_key = "analysis-navigation-state:" + chart_id
    state = st.session_state.get(state_key, {})
    if state.get("result_id") != identity:
        state = {"result_id": identity, "selected": None, "command": None}
    if not events:
        st.session_state[state_key] = state
        return None
    by_id = {event["id"]: event for event in events}
    queued = st.session_state.pop("analysis-navigation-request", None)
    requested = None
    if isinstance(queued, dict) and queued.get("id") in by_id:
        if ((queued.get("source_key") is None or queued["source_key"] == run.source_key)
                and (queued.get("result_key") is None or queued["result_key"] == run.keys.get(queued.get("instance_id")))):
            requested = queued["id"]
    group_labels = dict.fromkeys(event["group_id"] for event in events)
    labels = {event["group_id"]: event["group_label"] for event in events}
    prefix = "analysis-navigation:" + chart_id + ":" + identity[:12]
    st.caption("Navigate analysis findings")
    group_key = prefix + ":group"
    if requested:
        st.session_state[group_key] = by_id[requested]["group_id"]
    group = st.selectbox("Analysis event group", options=["", *group_labels], key=group_key,
                         format_func=lambda value: labels.get(value, "All findings"))
    visible = [event for event in events if not group or event["group_id"] == group]
    ids = [event["id"] for event in visible]
    selected = requested or (state["selected"] if state["selected"] in ids else None)
    previous, chooser, following = st.columns([1, 6, 1])
    event_key = prefix + ":event"
    if requested:
        st.session_state[event_key] = requested
    elif st.session_state.get(event_key) not in ids:
        st.session_state[event_key] = None

    selected = st.session_state.get(event_key) or selected

    def choose(target):
        st.session_state[event_key] = target

    position = ids.index(selected) if selected in ids else -1
    with previous:
        st.button("Previous", key=prefix + ":previous", disabled=not ids or selected == ids[0],
                  on_click=choose, args=(ids[position - 1] if position > 0 else ids[-1],))
    with chooser:
        choice = st.selectbox("Analysis event", options=ids, index=None, key=event_key,
                              placeholder="Search findings by label or timestamp",
                              format_func=lambda value: f"{by_id[value]['timestamp']} · {by_id[value]['label']}"
                                                        f" · {by_id[value]['group_label']}")
    with following:
        st.button("Next", key=prefix + ":next", disabled=not ids or selected == ids[-1],
                  on_click=choose, args=(ids[min(len(ids) - 1, position + 1)],))
    target = requested or (choice if choice != state.get("widget_choice") and choice in by_id else None)
    state["widget_choice"] = choice
    if target:
        event = by_id[target]
        sequence = max(time.time_ns() // 1000, st.session_state.get("analysis-navigation-sequence", 0) + 1)
        st.session_state["analysis-navigation-sequence"] = sequence
        state["selected"] = target
        state["command"] = event_command(event, chart_id=chart_id, result_id=identity, sequence=sequence)
    if state.get("selected") in by_id:
        active = by_id[state["selected"]]
        st.caption(f"Selected finding: {active['timestamp']} · {active['label']}")
    st.session_state[state_key] = state
    return state.get("command")
