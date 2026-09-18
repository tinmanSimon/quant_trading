"""Preview and physically delete an exact local series without crossing layers."""

from datetime import UTC, datetime, time, timedelta

import streamlit as st

from data_pipeline import DataQuery
from data_pipeline.exceptions import DatasetNotFoundError


def _bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _clear_preview():
    st.session_state.pop("delete-plan", None)


def _show_report(report):
    st.caption(f"Deletion operation: {report.operation_id}")
    if report.status == "completed":
        st.success(
            f"Physical deletion completed: {report.removed_rows:,} stored rows removed. "
            f"Original files removed: {_bytes(report.removed_bytes)}; "
            f"surviving files written: {_bytes(report.written_bytes)}."
        )
        st.caption("Rows are counted across all matching revisions. Other data layers were left unchanged.")
    elif getattr(report, "phase", "cleanup") == "preparation":
        st.warning("Deletion stopped during preparation, before any selected rows were deleted. "
                   "Temporary files need cleanup; the original datasets remain unchanged.")
    else:
        st.warning("Deletion was recorded, but physical cleanup is still pending. "
                   "Do not treat the selected data as physically removed yet.")
    with st.expander(f"Deletion details · {report.operation_id}"):
        st.json(report.to_dict())


def _refresh_operation(pipeline, report):
    try:
        current = pipeline.deletion_status(report.operation_id)
    except DatasetNotFoundError:
        # Recovery discards an uncommitted preparation, not a completed data
        # deletion. Its journal disappears rather than becoming 'completed'.
        if getattr(report, "phase", "cleanup") != "preparation":
            raise
        if getattr(st.session_state.get("delete-report"), "operation_id", None) == report.operation_id:
            st.session_state.pop("delete-report", None)
        _clear_preview()
        st.session_state["delete-recovery-notice"] = (
            f"Preparation cleanup completed for {report.operation_id}. "
            "No selected rows were deleted by that operation. Create a fresh preview to delete them."
        )
        return None
    st.session_state["delete-report"] = current
    return current


def _operations(pipeline) -> bool:
    """Keep interrupted cleanup discoverable even when no datasets remain."""
    pending = pipeline.list_deletions(pending_only=True)
    last = st.session_state.get("delete-report")
    if last is not None and all(report.operation_id != last.operation_id for report in pending):
        # An operation might have completed in another process since this page ran.
        last = _refresh_operation(pipeline, last)
        if last is not None:
            _show_report(last)
    for report in pending:
        _show_report(report)
        if st.button("Retry physical cleanup", key=f"delete-recover-{report.operation_id}"):
            try:
                with st.spinner("Finishing physical cleanup…"):
                    pipeline.recover()
                _refresh_operation(pipeline, report)
                st.rerun()
            except Exception as error:
                st.error(f"Physical cleanup did not finish: {type(error).__name__}: {error}")
        if st.button("Refresh cleanup status", key=f"delete-refresh-{report.operation_id}"):
            _refresh_operation(pipeline, report)
            st.rerun()
    if st.session_state.get("delete-recovery-notice"):
        st.info(st.session_state["delete-recovery-notice"])
    return bool(pending)


def deletion_page(research):
    """Render the two-step deletion workflow against the public pipeline API."""
    pipeline = research.pipeline
    root = str(pipeline.store.data_dir.resolve())
    if st.session_state.get("delete-root") != root:
        for key in ("delete-plan", "delete-query", "delete-report", "delete-recovery-notice"):
            st.session_state.pop(key, None)
        st.session_state["delete-root"] = root

    st.title("Delete local data")
    st.caption("Permanently remove selected bars from one local data series, including historical revisions. "
               "Raw and processed data are independent: deleting one does not delete or invalidate the other.")
    pending = _operations(pipeline)
    if pending:
        st.info("Finish the pending physical cleanup before starting another deletion.")

    layer = st.radio("Delete from layer", ["raw", "processed"], horizontal=True, key="delete-layer")
    items = pipeline.list_datasets(DataQuery(layer=layer, include_history=True))
    if not items:
        _clear_preview()
        st.info(f"No local {layer} datasets are available to delete.")
        return
    provider = st.selectbox("Provider", sorted({item.request.provider for item in items}), key="delete-provider")
    items = [item for item in items if item.request.provider == provider]
    symbol = st.selectbox("Ticker to delete", sorted({item.request.symbol for item in items}), key="delete-symbol")
    items = [item for item in items if item.request.symbol == symbol]
    timeframe = st.selectbox("Timeframe", sorted({item.request.timeframe for item in items}), key="delete-timeframe")
    items = [item for item in items if item.request.timeframe == timeframe]
    pipeline_id = None
    if layer == "processed":
        pipeline_id = st.selectbox("Processor pipeline", sorted({item.pipeline_id for item in items}),
                                   key="delete-pipeline")
        items = [item for item in items if item.pipeline_id == pipeline_id]

    window = f"delete-window-{root}-{layer}-{provider}-{symbol}-{timeframe}-{pipeline_id}"
    first = min(item.first_timestamp for item in items).date()
    last = max(item.last_timestamp for item in items).date() + timedelta(days=1)
    left, right = st.columns(2)
    start_date = left.date_input("Deletion start date (UTC, inclusive)", first, key=f"{window}-start-date")
    start_time = left.time_input("Deletion start time (UTC)", time.min, key=f"{window}-start-time")
    end_date = right.date_input("Deletion end date (UTC, exclusive)", last, key=f"{window}-end-date")
    end_time = right.time_input("Deletion end time (UTC)", time.min, key=f"{window}-end-time")
    start = datetime.combine(start_date, start_time, tzinfo=UTC)
    end = datetime.combine(end_date, end_time, tzinfo=UTC)
    st.caption("The interval uses stored bar timestamps: start is inclusive and end is exclusive. "
               "Daily and longer bars use midnight-UTC session-date labels, not exchange opening times.")
    identity = (root, layer, provider, symbol, timeframe, pipeline_id, start, end)
    if st.session_state.get("delete-query") != identity:
        _clear_preview()
        st.session_state["delete-query"] = identity
    if start >= end:
        st.error("The deletion end must be later than the start.")
        return
    query = DataQuery(layer=layer, provider=provider, symbol=symbol, timeframe=timeframe,
                      pipeline_id=pipeline_id, start=start, end=end, include_history=True)
    if st.button("Preview deletion", key="delete-preview", disabled=pending):
        _clear_preview()
        st.session_state.pop("delete-recovery-notice", None)
        try:
            with st.spinner("Verifying affected files and counting selected rows…"):
                st.session_state["delete-plan"] = pipeline.plan_delete(query)
        except Exception as error:
            st.error(f"Could not prepare deletion: {type(error).__name__}: {error}")

    plan = st.session_state.get("delete-plan")
    if plan is None:
        return
    if not plan.removed_rows:
        st.info("No stored bars match this interval. Nothing will be deleted.")
        return
    st.subheader("Deletion preview")
    st.write(f"{layer} · {provider} · {symbol} · {timeframe}")
    if pipeline_id:
        st.caption(f"Processor pipeline: {pipeline_id}")
    st.write(f"{start.isoformat()} ≤ timestamp < {end.isoformat()}")
    st.dataframe(plan.to_frame(), hide_index=True)
    st.write(f"Rows to remove: {plan.removed_rows:,} · Rows to retain: {plan.retained_rows:,}")
    st.caption(f"Affected original files: {_bytes(plan.source_bytes)}. "
               f"Estimated temporary space for surviving data: {_bytes(plan.estimated_temporary_bytes)}. "
               "The estimate is not the final compressed file size or an exact free-space guarantee.")
    st.warning("This physically deletes matching data from all revisions in this selected series. "
               "There is no undo or retained backup. Rows outside the interval remain. "
               "Saved backtest results remain available, but deleted source revisions cannot be reread.")
    confirmed = st.checkbox("I understand this permanently deletes the rows shown above.",
                            key=f"delete-confirm-{plan.operation_id}")
    execute = st.button("Delete permanently", type="primary", key=f"delete-execute-{plan.operation_id}",
                        disabled=not confirmed or pending)
    if execute and confirmed and not pending:
        # Consume this preview on every attempt: a failed/stale plan must never
        # remain armed after an error or after a concurrent storage change.
        _clear_preview()
        try:
            with st.spinner("Preserving remaining rows and physically removing original files…"):
                report = pipeline.delete(plan, confirm=plan.operation_id)
            st.session_state["delete-report"] = report
            st.rerun()
        except Exception as error:
            operation_id = getattr(error, "operation_id", None)
            if operation_id is not None:
                st.warning(f"Physical deletion is incomplete. Operation {operation_id} requires cleanup.")
                st.error(f"{type(error).__name__}: {error}")
                # Avoid hiding the original failure if a status lookup also fails.
                try:
                    report = pipeline.deletion_status(operation_id)
                    st.session_state["delete-report"] = report
                    _show_report(report)
                except Exception as status_error:
                    st.error(f"Could not read cleanup status: {status_error}")
                st.info("Refresh this page to retry pending physical cleanup.")
            else:
                st.error(f"Deletion did not complete: {type(error).__name__}: {error}. "
                         "Review the current selection and create a fresh preview before retrying.")
