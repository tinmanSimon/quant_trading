"""Review costs and resume native Databento historical batch jobs."""

from hashlib import sha256
import json
from pathlib import Path

import streamlit as st

from dashboard.components.source_controls import _date_fields, _tickers


_POLICIES = {
    "Fetch missing ranges; merge locally": "fetch_missing",
    "Fail on overlap": "error",
    "Skip covered requests; refresh overlapping ranges": "refresh_overlap",
}
_FINISHED = {"completed", "completed_with_errors", "stale"}
_STAGES = {
    "planned": "Not submitted", "submitting": "Submitting job", "submitted": "Waiting for Databento",
    "ready": "Download files", "downloaded": "Import bars", "decoded": "Save tickers",
    "failed": "Failed", "needs_reconciliation": "Reconnect job",
}


def _signature(config):
    return sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def databento_fetch_page(research, adapter):
    """No vendor I/O on page load; only estimates and authorized jobs contact it."""
    pipeline = research.pipeline
    root = str(Path(pipeline.store.data_dir).resolve())
    if st.session_state.get("db-fetch-root") != root:
        for key in list(st.session_state):
            if key.startswith("db-"):
                del st.session_state[key]
        st.session_state["db-fetch-root"] = root

    st.caption("Set DATABENTO_API_KEY in the environment before starting the dashboard. "
               "Estimates use free metadata requests; fetching data uses your Databento credits or balance.")
    if st.button("Load available datasets", key="db-load-datasets"):
        try:
            st.session_state["db-datasets"] = sorted(adapter.list_datasets())
        except Exception as error:
            st.error(str(error))
    datasets = st.session_state.get("db-datasets", [])
    dataset = st.selectbox("Databento dataset", datasets, index=None,
                           placeholder="Load datasets, then choose a feed", key="db-dataset")
    st.caption("Choose the feed whose venue coverage and history you need. Different feeds are saved "
               "separately. Available datasets may require additional account entitlements.")
    timeframe = st.selectbox("Bar size", ["1m", "1h"], key="db-timeframe")
    if timeframe == "1m":
        st.caption("For 09:30-aligned 30-minute bars: download 1-minute data, then open Process data "
                   "and add Resample intraday (1m to 30m / 1h). Raw downloads retain all trading sessions.")
    else:
        st.caption("Native hourly bars keep the vendor's timestamps; they are not shifted to 09:30.")
    ticker_text = st.text_area("Tickers (commas, spaces, or new lines)", "AAPL, MSFT", key="db-tickers")
    tickers = _tickers(ticker_text, "databento")
    start, end = _date_fields("db-fetch", intraday=True)
    policy = st.selectbox("Existing data", list(_POLICIES), key="db-overlap-policy")
    st.caption("Fetch missing ranges reuses local coverage. Refresh downloads overlapping ranges again. "
               "Range coverage does not certify that every expected bar exists.")
    maximum = st.number_input("Maximum estimated cost (USD; 0 means no limit)",
                              min_value=0.0, value=10.0, step=1.0, key="db-cost-limit")
    st.caption("This checks the estimate before submitting; it is not a vendor-enforced billing cap. "
               "Your remaining credit balance is shown in Databento's billing portal.")
    config = {"tickers": tickers, "dataset": dataset, "timeframe": timeframe,
              "start": start, "end": end, "overlap_policy": _POLICIES[policy],
              "max_estimated_cost": maximum or None}
    current_signature = _signature(config)
    st.session_state["db-current-signature"] = current_signature
    if st.button("Estimate download", type="primary", key="db-estimate", disabled=not dataset):
        if not tickers or start >= end:
            st.error("Provide at least one ticker and an end after the start.")
        else:
            try:
                with st.spinner("Checking history, local coverage and estimated cost…"):
                    plan = pipeline.estimate_databento(**config)
                st.session_state["db-estimate-signature"] = current_signature
                st.session_state["db-estimate-id"] = plan["id"]
                st.session_state["db-plan-selection"] = plan["id"]
            except Exception as error:
                st.session_state.pop("db-estimate-id", None)
                st.error(str(error))

    plans = pipeline.databento_plans()
    if not plans:
        st.info("Estimate a download to create a saved plan. Submitted jobs can be resumed here after restarting.")
        return
    by_id = {plan["id"]: plan for plan in plans}
    if st.session_state.get("db-plan-selection") not in by_id:
        st.session_state["db-plan-selection"] = next(iter(by_id))
    plan_id = st.selectbox("Saved Databento downloads", list(by_id), key="db-plan-selection",
                          format_func=lambda key: _plan_label(by_id[key]))
    _download_panel(pipeline, plan_id)


def _plan_label(plan):
    return (f"{plan['dataset']} · {len(plan['tickers'])} tickers · {plan['timeframe']} · "
            f"{plan['status']} · {plan['id'][:10]}")


def _bytes_label(size):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{size:,} B"
        size /= 1024


def _show_download_progress(placeholder, value):
    downloaded, total = value["downloaded_bytes"], value["total_bytes"]
    completed, files = value["completed_files"], value["total_files"]
    fraction = downloaded / total if total else completed / files if files else 0.0
    placeholder.progress(min(1.0, max(0.0, fraction)), text=(
        f"Files: {completed:,}/{files:,} verified · "
        f"{_bytes_label(downloaded)} / {_bytes_label(total)}"))


def _show_stage(placeholder, plan):
    messages = []
    for index, group in enumerate(plan.get("groups", []), 1):
        state = group["status"]
        if state == "submitted":
            remote = group.get("remote_status", "submitted")
            message = f"Waiting for Databento: **{remote}**. File transfer has not started."
            percent = group.get("progress")
            if isinstance(percent, (int, float)) and 0 <= percent <= 100:
                message += f" Server preparation: {percent:g}%."
        elif state == "ready":
            message = "Files are ready to download."
        elif state == "downloaded":
            message = "Files downloaded and verified. Next: import bars."
        elif state == "decoded":
            count = len(plan.get("outcomes", []))
            message = f"Bars imported. Ticker results: **{count:,}/{len(plan['tickers']):,}**."
        elif state == "failed":
            message = f"Failed: {group.get('error', 'See the error report below.')}"
        else:
            message = _STAGES.get(state, state)
        messages.append(f"Request {index}: {message}" if len(plan["groups"]) > 1 else message)
    placeholder.markdown("\n\n".join(messages))


def _saved_transfer_progress(plan):
    # Keep verified totals visible after transfer and on later fragment reruns.
    values = [group["download_progress"] for group in plan.get("groups", [])
              if group.get("download_progress")]
    return ({key: sum(value[key] for value in values) for key in
             ("downloaded_bytes", "total_bytes", "completed_files", "total_files")}
            if values else None)


@st.fragment(run_every="3s")
def _download_panel(pipeline, plan_id):
    plan = pipeline.databento_plan(plan_id)
    st.subheader("Download plan")
    st.write(f"**{plan['dataset']} · {plan['timeframe']} · {len(plan['tickers']):,} tickers**")
    st.caption(f"UTC range: {plan['start']} → {plan['end']} (end exclusive)")
    st.metric("Estimated data cost", f"${plan['estimated_cost']:.4f}")
    st.caption("The estimate is the data charge before any remaining account credits are applied.")
    for warning in plan.get("warnings", []):
        st.warning(warning)

    stage = st.empty()
    _show_stage(stage, plan)
    transfer = st.empty()
    saved_progress = _saved_transfer_progress(plan)
    if saved_progress:
        _show_download_progress(transfer, saved_progress)

    def report_progress(value):
        stage.markdown("**Downloading and verifying files…**")
        _show_download_progress(transfer, value)

    if plan["status"] == "estimated":
        matches = (st.session_state.get("db-estimate-id") == plan_id
                   and st.session_state.get("db-estimate-signature") == st.session_state.get("db-current-signature"))
        cap = plan.get("max_estimated_cost")
        over_budget = cap is not None and plan["estimated_cost"] > cap
        if not matches:
            st.info("Estimate the current inputs before submitting. This saved estimate is not linked to the current inputs.")
        if over_budget:
            st.warning("This estimate exceeds your maximum. Change the request or budget and estimate again.")
        if st.button("Fetch and save", type="primary", disabled=not matches or over_budget, key=f"db-submit-{plan_id}"):
            try:
                plan = pipeline.advance_databento(plan_id, submit=True, progress=report_progress)
            except Exception as error:
                st.error(str(error))
                plan = pipeline.databento_plan(plan_id)
    elif plan["status"] not in _FINISHED:
        automatic = st.checkbox("Automatically check and continue this download", value=True,
                                 key=f"db-auto-{plan_id}")
        manual = st.button("Check progress / resume", key=f"db-resume-{plan_id}")
        if manual or (automatic and plan["status"] in {"running", "needs_reconciliation"}):
            try:
                active = next((group for group in plan.get("groups", [])
                               if group["status"] not in {"decoded", "failed"}), None)
                if active and active["status"] == "downloaded":
                    stage.markdown("**Importing downloaded bars…**")
                plan = pipeline.advance_databento(plan_id, progress=report_progress)
            except Exception as error:
                st.error(str(error))
                plan = pipeline.databento_plan(plan_id)

    _show_stage(stage, plan)
    saved_progress = _saved_transfer_progress(plan)
    if saved_progress:
        _show_download_progress(transfer, saved_progress)
    st.write(f"Status: **{plan['status'].replace('_', ' ')}**")
    groups = plan.get("groups", [])
    if groups:
        st.caption("Databento state tracks server preparation; local stage tracks downloading and saving.")
        for group in groups:
            if group.get("next_download_attempt_at"):
                st.info(f"Databento requested a pause. This job can resume after "
                        f"{group['next_download_attempt_at']} (UTC).")
        st.dataframe([{
            "group": group["id"], "job_id": group.get("job_id"),
            "local_stage": ("Finished" if group["status"] == "decoded" and plan["status"] in _FINISHED
                            else _STAGES.get(group["status"], group["status"])),
            "databento_state": group.get("remote_status", "unknown"),
            "tickers": len(group["symbols"]), "start": group["start"], "end": group["end"],
            "estimated_cost": group["estimated_cost"], "error": group.get("error"),
        } for group in groups], hide_index=True)
    with st.expander("Tickers in this plan"):
        st.write(", ".join(plan["tickers"]))
    report = pipeline.databento_report(plan_id)
    if report.outcomes:
        st.dataframe(report.to_frame(), hide_index=True)
        for outcome in report.outcomes:
            if outcome.warning_message:
                st.warning(f"{outcome.ticker}: {outcome.warning_message}")
    if plan["status"] == "completed":
        st.success("Download complete. Saved datasets are available in Market data, Process data and Backtest.")
    elif plan["status"] == "completed_with_errors":
        st.warning("Download finished with errors. Successful tickers were saved; review the report above.")
    elif plan["status"] == "stale":
        st.warning("Local data changed after this estimate. Create a new estimate before downloading.")

    uncertain = [group for group in groups if group["status"] == "needs_reconciliation"]
    if uncertain:
        st.warning("A job submission could not be confirmed. Check Databento's Download center for the job ID; "
                   "the app will not submit that request again automatically.")
        with st.expander("Reconnect an existing Databento job"):
            group_id = st.selectbox("Request group", [group["id"] for group in uncertain], key=f"db-group-{plan_id}")
            job_id = st.text_input("Existing job ID", key=f"db-job-{plan_id}")
            if st.button("Reconnect job", disabled=not job_id.strip(), key=f"db-reconnect-{plan_id}"):
                try:
                    pipeline.reconcile_databento_job(plan_id, group_id, job_id.strip())
                    st.rerun(scope="fragment")
                except Exception as error:
                    st.error(str(error))
