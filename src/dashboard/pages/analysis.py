"""Run unchanged research analyses across many local tickers."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from data_pipeline import DataQuery
from dashboard.batch_analysis.state import BatchJob, analysis_memory, job_key
from dashboard.components.analysis_selection import select_analyses
from dashboard.components.source_controls import _date_fields, _provider_choice, _tickers
from dashboard.pages.processing import pipeline_label
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore
from research.analysis.scripting.discovery import load_registry


def _source_selection(research, memory):
    layer = memory.widget(st.radio, "Layer", options=["raw", "processed"],
                          horizontal=True, key="batch-layer")
    items = research.pipeline.list_datasets(DataQuery(layer=layer))
    if not items:
        st.info("No matching active local datasets. Fetch or process data before starting a new run.")
        return None
    provider = _provider_choice((item.request.provider for item in items), key="batch-provider", memory=memory)
    items = [item for item in items if item.request.provider == provider]
    timeframe = memory.widget(st.selectbox, "Timeframe", options=sorted({item.request.timeframe for item in items}),
                              key="batch-timeframe")
    items = [item for item in items if item.request.timeframe == timeframe]
    pipeline_id = None
    if layer == "processed":
        pipeline_id = memory.widget(st.selectbox, "Processing pipeline",
                                    options=sorted({item.pipeline_id for item in items}), key="batch-pipeline",
                                    format_func=lambda value: pipeline_label(research, value, items))
        items = [item for item in items if item.pipeline_id == pipeline_id]
    symbols = sorted({item.request.symbol for item in items})
    all_symbols = memory.widget(st.checkbox, "Select all available tickers", key="batch-all-tickers")
    selected = memory.widget(st.multiselect, "Tickers", options=symbols, key="batch-tickers",
                             disabled=all_symbols)
    pasted = memory.widget(st.text_area, "Additional tickers", key="batch-extra-tickers",
                           placeholder="AAPL, MSFT, NVDA", help="Separate symbols with commas, spaces, or newlines.")
    tickers = list(dict.fromkeys((symbols if all_symbols else selected) + _tickers(pasted, provider)))
    st.caption(f"{len(tickers):,} tickers selected. Missing local data is reported per ticker.")
    first = min(item.first_timestamp for item in items).date()
    last = (max(item.last_timestamp for item in items) + timedelta(days=1)).date()
    start, end = _date_fields("batch", first=first, last=last, memory=memory)
    workers = memory.widget(st.number_input, "Concurrent tickers", min_value=1, max_value=8,
                            value=1, step=1, key="batch-workers",
                            help="Each worker loads one ticker. Start with one; more workers use more memory.")
    if start >= end:
        st.error("The end must be later than the start.")
        return None
    return dict(tickers=tickers, layer=layer, provider=provider, timeframe=timeframe,
                pipeline_id=pipeline_id, start=start, end=end, workers=workers)


def _saved_results(store, memory, job):
    from dashboard.batch_analysis.results import render_run_results

    if job and not job.running and not job.notified:
        job.notified = True
        # End polling and re-enable Run after a background worker completes.
        st.rerun()
    if job and job.error:
        st.error(f"The batch worker stopped: {job.error}. Completed ticker results remain available in this session.")
    if job and job.running:
        st.caption("The batch is running in the background. You can inspect completed results below.")
        if st.button("Stop after current tickers", key="batch-stop", disabled=job.cancel.is_set()):
            job.cancel.set()
        if job.cancel.is_set():
            st.info("Stopping: running tickers will finish; further tickers will not start.")
    if job and not job.saved:
        st.subheader("Current analysis")
        st.caption("Temporary results. Save this analysis to keep it after this session or a new run.")
        if st.button("Save analysis", key="batch-save", disabled=job.running):
            try:
                store.copy_run(job.runner.store, job.run_id)
            except (OSError, ValueError) as error:
                st.error(f"Could not save this analysis: {error}")
            else:
                job.saved = True
                if job.temporary is not None:
                    job.temporary.cleanup()
                memory.values["selected_run"] = job.run_id
                memory.values["widgets"]["batch-saved-run"] = job.run_id
                st.session_state.pop("batch-saved-run", None)
                memory.save()
                st.rerun()
        render_run_results(job.runner.store, job.runner.store.load_manifest(job.run_id), memory,
                           active=job.running)
    st.subheader("Saved results")
    try:
        runs = store.list_runs()
    except (OSError, ValueError) as error:
        st.error(f"Could not read saved batch runs: {error}")
        return
    if not runs:
        st.info("No saved analysis runs yet.")
        return
    lookup = {run["run_id"]: run for run in runs}
    saved_id = memory.values.get("selected_run")
    if saved_id in lookup and "batch-saved-run" not in memory.values["widgets"]:
        memory.values["widgets"]["batch-saved-run"] = saved_id
    run_id = memory.widget(st.selectbox, "Saved analysis run", options=list(lookup), key="batch-saved-run",
                           format_func=lambda value: (
                               f"{lookup[value]['created_at']} · {lookup[value]['status']} · "
                               f"{lookup[value]['counts']['requested']} tickers · {value[:8]}"))
    memory.values["selected_run"] = run_id
    memory.save()
    if st.button("Delete analysis", key="batch-delete"):
        try:
            store.delete_run(run_id)
        except (OSError, ValueError) as error:
            st.error(f"Could not delete this analysis: {error}")
        else:
            if job and job.run_id == run_id:
                st.session_state.pop(job_key(memory), None)
            memory.values.pop("selected_run", None)
            memory.values["widgets"].pop("batch-saved-run", None)
            st.session_state.pop("batch-saved-run", None)
            memory.save()
            st.rerun()
    render_run_results(store, lookup[run_id], memory,
                       active=bool(job and job.running and job.run_id == run_id))


def _analysis_page(research):
    memory = analysis_memory(research)
    store = BatchStore(Path(research.runs_dir) / "analysis")
    job = st.session_state.get(job_key(memory))
    st.title("Analysis")
    st.caption("Run research analyses across local tickers, compare their summaries, and reopen saved runs. "
               "This page displays text and tables.")
    try:
        with st.expander("New analysis run", expanded=True):
            source = _source_selection(research, memory)
            if source is not None:
                reload = st.button("Reload analyses", key="batch-reload")
                registry = load_registry(reload=reload)
                st.subheader("Analyses")
                specs = select_analyses(registry, memory)
                running = job is not None and job.running
                if st.button("Run analysis", type="primary", key="batch-run",
                             disabled=running or not source["tickers"] or not specs):
                    temporary = None
                    try:
                        temporary = TemporaryDirectory(prefix="quant-analysis-")
                        runner = BatchRunner(research.pipeline, registry, BatchStore(temporary.name))
                        manifest = runner.create(BatchRequest(**source), specs)
                    except Exception as error:
                        if temporary is not None:
                            temporary.cleanup()
                        st.error(f"Could not prepare this analysis run: {error}")
                    else:
                        if job and job.temporary is not None:
                            job.temporary.cleanup()
                        job = BatchJob(runner, manifest["run_id"], temporary=temporary)
                        st.session_state[job_key(memory)] = job
                        job.start()
                        memory.save()
                        st.rerun()
        st.fragment(run_every=1 if job and job.running else None)(_saved_results)(store, memory, job)
    finally:
        memory.save()
        if memory.warning:
            st.warning(memory.warning.replace("Market data", "Analysis"))
