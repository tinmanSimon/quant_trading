"""Fetch market bars and inspect provider outcomes."""

import streamlit as st

from dashboard.components.source_controls import _date_fields, _provider_choice, _tickers
from research import Research
from research.timeframes import INTRADAY_DURATIONS, fetch_timeframes

def _fetch_page(research: Research):
    st.title("Fetch market data")
    st.caption("Each ticker is attempted independently; a failure does not stop the remaining downloads.")
    provider = _provider_choice(research.pipeline.providers.names(), key="fetch-provider", result_key="fetch-report")
    if provider is None:
        return
    adapter = research.pipeline.providers.get(provider)
    intervals = fetch_timeframes(adapter.supported_timeframes)
    if not intervals:
        st.warning("This provider declares no fetch intervals supported by the dashboard.")
        return
    timeframe = st.selectbox("Bar size", intervals, key="fetch-timeframe")
    with st.form("fetch-data"):
        tickers = st.text_area("Tickers (commas, spaces, or new lines)", "AAPL, MSFT")
        start, end = _date_fields("fetch", intraday=timeframe in INTRADAY_DURATIONS)
        fetch_options = {}
        if provider == "massive":
            fetch_options["massive_request_interval_seconds"] = st.number_input(
                "Minimum seconds between API requests", min_value=0.0, value=0.0, step=1.0,
                key="fetch-massive-request-interval",
            )
            st.caption("Massive only: its free plan allows 5 requests per minute. About 13 seconds "
                       "helps stay within that limit when no other downloads share your API key. "
                       "0 disables deliberate pacing.")
        overlap_choice = st.selectbox(
            "Existing data",
            ["Fail on overlap", "Skip covered requests; refresh overlapping ranges",
             "Fetch missing ranges; merge locally"],
            key="fetch-overlap-policy",
        )
        st.caption("Both optional modes skip fully covered requests without contacting the provider. "
                   "Refresh downloads the combined range again. Fetch missing ranges downloads only "
                   "uncovered portions and merges them locally, preserving existing bars. "
                   "Historical revisions are retained; processed data derived from replaced inputs "
                   "becomes inactive and needs rebuilding.")
        skip = None
        if provider == "yahoo":
            skip = st.checkbox("Skip bars where every OHLC price is missing", value=False)
            st.caption("Skipped timestamps and reasons are recorded. Other invalid values still fail validation.")
        submitted = st.form_submit_button("Fetch and save", type="primary")
    if submitted:
        if not _tickers(tickers, provider) or start >= end:
            st.error("Provide at least one ticker and an end after the start.")
            return
        progress_area = st.container()
        ticker_progress = {}

        def show_progress(event):
            index = event["index"]
            if index not in ticker_progress:
                ticker_progress[index] = progress_area.empty()
            lines = [f'**{event["ticker"]} ({index}/{event["total"]}) · {event["status"]}**']
            if event["fetch_ranges"]:
                lines.append("Vendor request ranges (UTC; end exclusive):")
                lines.extend(f'- {item["start"]} → {item["end"]} · {item["status"]}'
                             for item in event["fetch_ranges"])
            elif event["status"] == "skipped":
                lines.append("Already covered; no vendor request.")
            ticker_progress[index].markdown("\n\n".join(lines))

        overlap_policies = {
            "Fail on overlap": "error",
            "Skip covered requests; refresh overlapping ranges": "refresh_overlap",
            "Fetch missing ranges; merge locally": "fetch_missing",
        }
        with st.spinner("Fetching and verifying each ticker…"):
            report = research.pipeline.fetch_many(_tickers(tickers, provider), provider=provider, timeframe=timeframe,
                                                  start=start, end=end, skip_missing_ohlc=skip,
                                                  overlap_policy=overlap_policies[overlap_choice],
                                                  progress=show_progress, **fetch_options)
        st.session_state["fetch-report"] = report
    report = st.session_state.get("fetch-report")
    if report is not None:
        st.subheader("Most recent fetch report")
        st.dataframe(report.to_frame(), hide_index=True)
        st.caption("row_count is the saved dataset size; fetched_row_count counts newly downloaded bars. "
                   "fetch_ranges shows each planned vendor request and its status. fetch_start/end spans "
                   "the plan and can include stored data that was not downloaded.")
        counts = {status: sum(item.status == status for item in report.outcomes)
                  for status in ("saved", "refreshed", "merged", "skipped", "empty", "failed")}
        completed = [f"{counts[status]} ticker(s) {status}" for status in ("saved", "refreshed", "merged")
                     if counts[status]]
        if completed:
            st.success("; ".join(completed) + ".")
        if counts["skipped"]:
            st.info(f'{counts["skipped"]} ticker(s) already covered; no vendor request was made for them.')
        if counts["empty"]:
            st.info(f'{counts["empty"]} ticker(s) returned no bars; no datasets were changed for those requests.')
        if counts["failed"]:
            st.warning(f'{counts["failed"]} ticker(s) failed. Review their errors above; the remaining tickers were attempted.')
        for outcome in report.outcomes:
            if getattr(outcome, "warning_message", None):
                st.warning(f"{outcome.ticker}: {outcome.warning_message}")


