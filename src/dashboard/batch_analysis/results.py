"""Summaries, rankings and lazy text details from persisted batch results."""

from copy import deepcopy
from dataclasses import replace
import json

import polars as pl
import streamlit as st

from research.analysis.batch.aggregation import (
    AGGREGATIONS, REDUCERS, MetricSpec, aggregate_metrics, extract_metrics, merge_summaries, rank_metrics,
)


def inferred_specs(manifest, instance_id):
    """Union metric sources across all completed tickers, preserving missing data."""
    specs, _ = merge_summaries({ticker: status.get("summaries", {}).get(instance_id, {})
                               for ticker, status in manifest["tickers"].items()})
    return specs


def _selection_identity(spec):
    value = spec.to_dict()
    value.pop("aggregation")
    return value


def _extract_independent(tables, specs):
    """An invalid custom metric must not discard unrelated valid totals."""
    values, errors = {}, []
    base = {spec.name: spec for spec in specs if not spec.is_ratio}
    for spec in base.values():
        try:
            values.update(extract_metrics(tables, [spec]))
        except (ValueError, TypeError) as error:
            values[spec.name] = None
            errors.append(str(error))
    for spec in specs:
        if not spec.is_ratio:
            continue
        if values.get(spec.numerator) is None or values.get(spec.denominator) is None:
            values[spec.name] = None
            continue
        try:
            dependencies = [base[name] for name in dict.fromkeys((spec.numerator, spec.denominator))]
            values[spec.name] = extract_metrics(tables, [*dependencies, spec])[spec.name]
        except (ValueError, TypeError) as error:
            values[spec.name] = None
            errors.append(str(error))
    return values, errors


def metric_values(store, manifest, instance_id, specs, *, cache=None):
    """Reuse compact inferred values; stream details only for custom selections."""
    values, errors = {}, {}
    inferred, compact = merge_summaries({ticker: status.get("summaries", {}).get(instance_id, {})
                                         for ticker, status in manifest["tickers"].items()})
    known = {spec.name: spec for spec in inferred}
    if all(spec.name in known and _selection_identity(spec) == _selection_identity(known[spec.name])
           for spec in specs):
        return {ticker: {spec.name: row.get(spec.name) for spec in specs}
                for ticker, row in compact.items()}, errors
    cache = cache if cache is not None else {}
    configuration = json.dumps([_selection_identity(spec) for spec in specs], sort_keys=True)
    for ticker, status in manifest["tickers"].items():
        if not status.get("detail_file"):
            values[ticker] = {spec.name: None for spec in specs}
            continue
        key = (str(store.directory), manifest["run_id"], ticker, status.get("detail_sha256"),
               instance_id, configuration)
        if key in cache:
            values[ticker], problem = cache[key]
            if problem:
                errors[ticker] = problem
            continue
        try:
            detail = store.load_ticker(manifest["run_id"], ticker)
            result = detail.get("results", {}).get(instance_id)
            if result is not None:
                values[ticker], issues = _extract_independent(result.get("tables", []), specs)
                if issues:
                    errors[ticker] = "; ".join(issues)
            else:
                values[ticker] = {spec.name: None for spec in specs}
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors[ticker] = str(error)
            values[ticker] = {spec.name: None for spec in specs}
        cache[key] = values[ticker], errors.get(ticker)
        while len(cache) > 4000:
            cache.pop(next(iter(cache)))
    return values, errors


def _cached_values(store, manifest, instance_id, specs, memory):
    cache = st.session_state.setdefault("_batch_ticker_metric_cache", {})
    return metric_values(store, manifest, instance_id, specs, cache=cache)


def _load_detail(store, manifest, ticker):
    # At most two selected reports stay in memory; summary extraction streams
    # through its own reader and stores only the small numeric result.
    cache = st.session_state.setdefault("_batch_detail_cache", {})
    key = (str(store.directory), manifest["run_id"], ticker,
           manifest["tickers"][ticker].get("detail_sha256"))
    if key not in cache:
        cache[key] = store.load_ticker(manifest["run_id"], ticker)
        while len(cache) > 2:
            cache.pop(next(iter(cache)))
    return cache[key]


def _custom_metric(store, manifest, instance_id, specs, prefix):
    tickers = [name for name, value in manifest["tickers"].items()
               if value.get("detail_file") and value.get("analysis_status", {}).get(instance_id) == "completed"]
    if not tickers:
        st.caption("Complete a ticker before selecting a result table.")
        return None
    sample = st.selectbox("Example ticker for table selection", tickers, key=prefix + "-sample")
    try:
        detail = _load_detail(store, manifest, sample)
    except (OSError, ValueError) as error:
        st.error(f"Could not read the example result: {error}")
        return None
    tables = detail.get("results", {}).get(instance_id, {}).get("tables", [])
    if not tables:
        st.info("This ticker's analysis has no result tables to summarize.")
        return None
    by_name = {table["name"]: table for table in tables}
    chosen = st.selectbox("Source table", list(by_name), key=prefix + "-table")
    table = by_name[chosen]
    columns = list(dict.fromkeys([*table.get("columns", []),
                                 *(name for row in table.get("rows", []) for name in row)]))
    if not columns:
        st.caption("This table does not declare columns.")
        return None
    column = st.selectbox("Value column", columns, key=prefix + "-column")
    category = st.selectbox("Filter category column", [None, *columns], key=prefix + "-category",
                            format_func=lambda value: "All rows" if value is None else value)
    category_value = None
    if category is not None:
        categories = list(dict.fromkeys(row.get(category) for row in table.get("rows", [])
                                       if isinstance(row.get(category), (str, int, float, bool))))
        if not categories:
            st.info("This column has no category values.")
            return None
        category_value = st.selectbox("Category value", categories, key=prefix + "-category-value")
    with st.form(prefix + "-add-metric-form"):
        name = st.text_input("Metric name", value=column, key=prefix + "-metric-name")
        reducer = st.selectbox("Calculation within each ticker", REDUCERS, key=prefix + "-reducer",
                               help="single requires one row; count counts nonempty values.")
        aggregate = st.selectbox("Overall calculation", AGGREGATIONS, key=prefix + "-aggregate")
        if st.form_submit_button("Add summary metric"):
            try:
                if name in {spec.name for spec in specs}:
                    raise ValueError("Choose a unique metric name.")
                return MetricSpec(name, chosen, column, reducer=reducer, aggregation=aggregate,
                                  category_column=category, category_value=category_value)
            except (ValueError, TypeError) as error:
                st.error(str(error))
    return None


def _summary_settings(store, manifest, instance_id, specs, memory, prefix):
    previous = [spec.to_dict() for spec in specs]
    edited = []
    removed = set()
    reset = False
    with st.expander("Summary settings"):
        st.caption("Counts are added automatically. Choose how other metrics should combine. "
                   "These controls only read saved results; they do not rerun analyses.")
        for index, spec in enumerate(specs):
            if st.button(f"Remove {spec.name}", key=f"{prefix}-remove-metric-{index}"):
                removed.add(spec.name)
                continue
            if spec.is_ratio:
                st.caption(f"{spec.name}: {spec.numerator} / {spec.denominator} × {spec.scale:g}; "
                           "overall uses matching tickers' summed numerator and denominator.")
                edited.append(spec)
            else:
                aggregation = memory.widget(
                    st.selectbox, f"Overall {spec.name}", options=AGGREGATIONS,
                    index=AGGREGATIONS.index(spec.aggregation), key=f"{prefix}-aggregation-{index}-{spec.name}",
                )
                edited.append(replace(spec, aggregation=aggregation))
        if st.checkbox("Add a metric from a result table", key=prefix + "-custom"):
            added = _custom_metric(store, manifest, instance_id, edited, prefix)
            if added is not None:
                edited.append(added)
        base = [spec.name for spec in edited if not spec.is_ratio]
        if len(base) >= 2 and st.checkbox("Add a ratio or percentage", key=prefix + "-ratio"):
            with st.form(prefix + "-ratio-form"):
                name = st.text_input("Ratio name", key=prefix + "-ratio-name")
                numerator = st.selectbox("Numerator metric", base, key=prefix + "-numerator")
                denominator = st.selectbox("Denominator metric", base, index=1, key=prefix + "-denominator")
                percent = st.checkbox("Display as percentage", value=True, key=prefix + "-percent")
                if st.form_submit_button("Add ratio metric"):
                    try:
                        if name in {spec.name for spec in edited}:
                            raise ValueError("Choose a unique metric name.")
                        edited.append(MetricSpec(name, numerator=numerator, denominator=denominator,
                                                 scale=100.0 if percent else 1.0))
                    except ValueError as error:
                        st.error(str(error))
        edited = [spec for spec in edited if not spec.is_ratio or
                  not {spec.numerator, spec.denominator}.intersection(removed)]
        if st.button("Reset to inferred metrics", key=prefix + "-reset-metrics"):
            edited = inferred_specs(manifest, instance_id)
            reset = True
            for key in list(memory.values["widgets"]):
                if key.startswith(prefix + "-aggregation-"):
                    memory.values["widgets"].pop(key)
    if previous != [spec.to_dict() for spec in edited]:
        settings = deepcopy(manifest.get("aggregation", {}))
        settings.setdefault("instances", {})[instance_id] = [spec.to_dict() for spec in edited]
        store.save_aggregation(manifest["run_id"], settings)
    if reset:
        memory.save()
        st.rerun()
    return edited


def _details(store, manifest, instance_id, spec, tickers, prefix):
    if not tickers:
        return
    st.subheader("Individual results")
    ticker = st.selectbox("Ticker details", tickers, key=prefix + "-detail-ticker")
    status = manifest["tickers"][ticker]
    st.caption(f"{ticker}: {status['status']}")
    if status.get("error"):
        st.error(status["error"])
    if not status.get("detail_file"):
        st.info("No completed result is saved for this ticker.")
        return
    try:
        detail = _load_detail(store, manifest, ticker)
    except (OSError, ValueError) as error:
        st.error(f"Could not load this ticker's results: {error}")
        return
    if instance_id in detail.get("errors", {}):
        st.error(detail["errors"][instance_id])
    result = detail.get("results", {}).get(instance_id)
    if result is None:
        return
    for message in result.get("messages", []):
        st.caption(message)
    for text in result.get("data", {}).get("analysis", {}).get("text", []):
        st.markdown(text)
    for index, table in enumerate(result.get("tables", [])):
        st.markdown(f"**{table['name']}**")
        rows = table.get("rows", [])
        if not rows:
            st.caption("No rows.")
            continue
        if len(rows) > 500:
            page = st.number_input(f"{table['name']} page", min_value=1, max_value=(len(rows) + 499) // 500,
                                   value=1, step=1, key=f"{prefix}-{ticker}-table-{index}-page")
            st.caption(f"{len(rows):,} rows; showing up to 500 per page.")
            rows = rows[(page - 1) * 500:page * 500]
        try:
            st.dataframe(pl.DataFrame(rows, infer_schema_length=None, strict=False), hide_index=True,
                         width="stretch", column_config={"_event_id": None})
        except (pl.exceptions.PolarsError, TypeError, ValueError):
            st.json(rows)
    with st.expander("Applied analysis settings"):
        st.json(spec)


def render_run_results(store, manifest, memory, *, active=False):
    counts = manifest["counts"]
    completed = sum(counts.get(key, 0) for key in ("completed", "partial", "failed", "no_data", "cancelled"))
    label = manifest['status'].capitalize()
    if manifest["status"] in ("pending", "running") and not active:
        label = "Saved partial progress"
        st.warning("This session has no active worker for this run. If its original process stopped, "
                   "unfinished tickers were not completed. Saved results remain available; start a new run to recalculate.")
    st.progress(completed / max(counts["requested"], 1),
                text=f"{label}: {completed:,} of {counts['requested']:,} tickers finished")
    st.caption(" · ".join(f"{key.replace('_', ' ')}: {counts.get(key, 0):,}" for key in
                           ("completed", "partial", "failed", "no_data", "cancelled")))
    with st.expander("Run settings"):
        st.json(manifest["request"])
    specifications = {spec["instance_id"]: spec for spec in manifest["specs"]}
    if not specifications:
        return
    prefix = "batch-result-" + manifest["run_id"]
    instance_id = memory.widget(st.selectbox, "Analysis results", options=list(specifications),
                                format_func=lambda value: (
                                    f"{specifications[value]['name']} · {list(specifications).index(value) + 1}"),
                                key=prefix + "-instance")
    prefix += "-" + instance_id
    saved = manifest.get("aggregation", {}).get("instances", {}).get(instance_id)
    summary_error = False
    try:
        inferred = inferred_specs(manifest, instance_id)
        specs = [MetricSpec.from_dict(item) for item in saved] if saved is not None else inferred
        specs = _summary_settings(store, manifest, instance_id, specs, memory, prefix)
        values, errors = _cached_values(store, manifest, instance_id, specs, memory)
        overall = aggregate_metrics(values, specs)
    except (ValueError, TypeError, OSError) as error:
        st.error(f"Could not summarize these results: {error}")
        summary_error = True
        specs, overall, errors = [], {}, {}
        values = {ticker: {} for ticker in manifest["tickers"]}
        if st.button("Clear invalid summary settings", key=prefix + "-clear-invalid"):
            settings = deepcopy(manifest.get("aggregation", {}))
            settings.get("instances", {}).pop(instance_id, None)
            store.save_aggregation(manifest["run_id"], settings)
            st.rerun()
    st.subheader("Overall summary")
    if specs:
        st.dataframe(pl.DataFrame([{"metric": name, **value} for name, value in overall.items()],
                                  infer_schema_length=None), hide_index=True, width="stretch")
        st.caption("Contributors count tickers with a usable value. Missing or failed results are not zero. "
                   "An overall calculation of none leaves the overall value empty.")
    elif not summary_error:
        st.info("No summary metrics were inferred. Use Summary settings to select a table and calculation. "
                "Original text and tables remain available below.")
    if errors:
        with st.expander(f"Summary extraction issues ({len(errors)})"):
            for ticker, error in errors.items():
                st.warning(f"{ticker}: {error}")
    st.subheader("Ticker summaries")
    search = memory.widget(st.text_input, "Filter tickers", key=prefix + "-search")
    analysis_statuses = {ticker: item.get("analysis_status", {}).get(instance_id, item["status"])
                         for ticker, item in manifest["tickers"].items()}
    statuses = sorted(set(analysis_statuses.values()))
    selected_status = memory.widget(st.selectbox, "Analysis status", options=[None, *statuses],
                                    format_func=lambda value: "All statuses" if value is None else value,
                                    key=prefix + "-status")
    ranking = memory.widget(st.selectbox, "Rank by", options=[None, *[spec.name for spec in specs]],
                            format_func=lambda value: "Ticker" if value is None else value, key=prefix + "-rank")
    descending = memory.widget(st.checkbox, "Highest first", value=True, key=prefix + "-descending")
    tickers = (rank_metrics(values, ranking, descending=descending) if ranking else sorted(values))
    tickers = [ticker for ticker in tickers if search.upper() in ticker.upper()
               and (selected_status is None or analysis_statuses[ticker] == selected_status)]
    display_names, used = {}, {"ticker", "status", "analysis_status"}
    for spec in specs:
        name = spec.name if spec.name not in used else "metric: " + spec.name
        while name in used:
            name = "metric: " + name
        display_names[spec.name] = name
        used.add(name)
    rows = [{"ticker": ticker, "status": manifest["tickers"][ticker]["status"],
             "analysis_status": analysis_statuses[ticker],
             **{display_names[name]: value for name, value in values[ticker].items()}}
            for ticker in tickers]
    if rows:
        st.dataframe(pl.DataFrame(rows, infer_schema_length=None), hide_index=True, width="stretch")
    else:
        st.info("No tickers match the current filters.")
    _details(store, manifest, instance_id, specifications[instance_id], tickers, prefix)
    memory.save()
