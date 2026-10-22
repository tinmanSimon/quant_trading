"""Local processing workspace, reusable recipes, and exact-output navigation."""
from collections import defaultdict
from datetime import time, timedelta
import json
from pathlib import Path
from uuid import uuid4
import warnings

import streamlit as st

from data_pipeline import DataQuery
from data_pipeline.exceptions import DataAlreadyExistsError
from data_pipeline.processing import discover_processors
from data_pipeline.processing.processors.intraday import ResamplingWarning
from dashboard.components.schema_controls import processor_controls
from dashboard.state.workspace import market_memory


def _scope(research):
    return str(Path(research.pipeline.store.data_dir).resolve())


def processing_memory(research):
    return market_memory("processing:" + _scope(research))


def backtest_memory(research):
    return market_memory("backtest:" + _scope(research))


def pipeline_label(research, pipeline_id, items=None):
    recipe = processing_memory(research).values.get("recipes", {}).get(pipeline_id, {})
    label = recipe.get("name")
    if not label:
        items = items if items is not None else research.pipeline.list_datasets(DataQuery(layer="processed"))
        item = next((item for item in items if item.pipeline_id == pipeline_id), None)
        try:
            label = " → ".join(step["name"].replace("_", " ") for step in json.loads(item.processors_json)) if item else "Pipeline"
        except (ValueError, TypeError, KeyError):
            label = "Pipeline"
    return f"{label} · {pipeline_id[:12]}"


def _add_step(memory, identity):
    name, version = identity
    memory.values.setdefault("steps", []).append({"id": uuid4().hex, "name": name, "version": version, "config": {}})
    memory.save()


def _change_step(memory, index, action):
    steps = memory.values["steps"]
    if action == "remove":
        step = steps.pop(index)
        prefix = "process-step-" + step["id"]
        memory.values["widgets"] = {key: value for key, value in memory.values["widgets"].items() if not key.startswith(prefix)}
    else:
        destination = index + action
        steps[index], steps[destination] = steps[destination], steps[index]
    memory.save()


def _load_recipe(memory, pipeline_id):
    recipe = memory.values["recipes"][pipeline_id]
    memory.values["steps"] = [dict(step, id=uuid4().hex) for step in recipe["specs"]]
    memory.values["widgets"]["process-name"] = recipe["name"]
    memory.save()


def _save_recipe(memory, pipeline, name):
    memory.values.setdefault("recipes", {})[pipeline.fingerprint] = {
        "name": name.strip() or " → ".join(item["name"] for item in json.loads(pipeline.canonical_json)),
        "specs": json.loads(pipeline.canonical_json),
    }
    memory.save()


def _open_output(research, dataset_id, page):
    item = research.pipeline.store.get_metadata(dataset_id)
    if page == "Market data":
        memory = market_memory(_scope(research))
        identity = (item.request.provider, item.request.timeframe, item.pipeline_id)
        window = f"browse-{item.request.symbol}-{identity}-One exact revision-{dataset_id}"
        memory.values["widgets"].update({
            "browse-layer": "processed", "browse-history": False, "browse-search": "",
            "browse-provider": item.request.provider, "browse-symbol": item.request.symbol,
            "browse-series": identity, "browse-read": "One exact revision", "browse-revision": dataset_id,
            window + "-start": item.first_timestamp.date(),
            window + "-end": (item.last_timestamp + timedelta(days=1)).date(),
        })
    else:
        memory = backtest_memory(research)
        memory.values["widgets"].update({
            "backtest-layer": "processed", "backtest-provider": item.request.provider,
            "backtest-pipeline": item.pipeline_id, "backtest-timeframe": item.request.timeframe,
            "backtest-tickers": [item.request.symbol], "backtest-extra": "",
            "backtest-start": item.first_timestamp.date(),
            "backtest-end": (item.last_timestamp + timedelta(days=1)).date(),
            "backtest-start-time": time.min, "backtest-end-time": time.min,
        })
        st.session_state.pop("backtest-run", None)
    memory.save()
    st.session_state["dashboard-page"] = page


def processing_page(research):
    memory = processing_memory(research)
    try:
        _render_processing(research, memory)
    finally:
        memory.save()
        if memory.warning:
            st.warning(memory.warning)


def _render_processing(research, memory):
    st.title("Process data")
    st.caption("Build an ordered pipeline from installed processors. Preview the full input, then save new processed datasets.")
    registry = discover_processors()
    for issue in registry.issues:
        st.warning(issue)
    definitions = {(definition.name, definition.version): definition for definition in registry.definitions()}
    st.subheader("Raw input")
    raw = research.pipeline.list_datasets(DataQuery(layer="raw"))
    selected = []
    if raw:
        provider = memory.widget(st.selectbox, "Provider", options=sorted({item.request.provider for item in raw}), key="process-provider")
        raw = [item for item in raw if item.request.provider == provider]
        timeframe = memory.widget(st.selectbox, "Source bar size", options=sorted({item.request.timeframe for item in raw}), key="process-timeframe")
        raw = [item for item in raw if item.request.timeframe == timeframe]
        symbols = sorted({item.request.symbol for item in raw})
        tickers = memory.widget(st.multiselect, "Tickers to process", options=symbols, default=symbols[:1], key="process-tickers")
        raw = [item for item in raw if item.request.symbol in tickers]
        # Keep an independent batch selection for each ticker set.
        batch_key = "process-batches-" + repr((provider, timeframe, tuple(tickers)))
        labels = {item.dataset_id: f"{item.request.symbol} · {item.first_timestamp.date()} to {item.last_timestamp.date()} · {item.row_count:,} bars · {item.dataset_id[:8]}" for item in raw}
        ids = memory.widget(st.multiselect, "Raw batches", options=list(labels), default=list(labels),
                            format_func=labels.__getitem__, key=batch_key)
        selected = [item for item in raw if item.dataset_id in ids]
        st.caption("Selected batches are combined per ticker in timestamp order. Each ticker is processed separately.")
    else:
        st.info("No active raw datasets. Open Fetch to download input data; you can still prepare a pipeline here.")
    bounds = (min(item.first_timestamp for item in selected).date(), max(item.last_timestamp for item in selected).date()) if selected else None

    st.subheader("Processing pipeline")
    recipes = memory.values.get("recipes", {})
    if recipes:
        choice = memory.widget(st.selectbox, "Saved pipeline", options=list(recipes), key="process-recipe",
                               format_func=lambda key: f"{recipes[key]['name']} · {key[:12]}")
        st.button("Load saved pipeline", on_click=_load_recipe, args=(memory, choice))
    name = memory.widget(st.text_input, "Pipeline name", value="", key="process-name", placeholder="My processing pipeline")
    if definitions:
        identity = memory.widget(st.selectbox, "Processor to add", options=list(definitions), key="process-add",
                                 format_func=lambda key: f"{definitions[key].label} (v{key[1]})")
        st.button("Add processor", on_click=_add_step, args=(memory, identity))
    else:
        st.info("No usable processors were discovered in the processors package.")
    st.caption("New processor files are discovered automatically. Restart the dashboard after editing already imported Python code.")
    steps = memory.values.setdefault("steps", [])
    specs = []
    valid = bool(steps)
    for index, step in enumerate(steps):
        identity = step["name"], step["version"]
        definition = definitions.get(identity)
        prefix = "process-step-" + step["id"]
        with st.container(border=True):
            st.markdown(f"**{index + 1}. {definition.label if definition else step['name']} (v{step['version']})**")
            left, middle, right = st.columns(3)
            left.button("Move up", key=prefix + "-up", disabled=index == 0,
                        on_click=_change_step, args=(memory, index, -1))
            middle.button("Move down", key=prefix + "-down", disabled=index == len(steps) - 1,
                          on_click=_change_step, args=(memory, index, 1))
            right.button("Remove", key=prefix + "-remove", on_click=_change_step, args=(memory, index, "remove"))
            if definition is None:
                st.error("This processor version is unavailable. Restore its module or remove this step.")
                valid = False
                continue
            st.caption(definition.description)
            if not definition.causal:
                st.info("This processor is available for charts; backtesting requires an explicit causal declaration from its author.")
            try:
                config = processor_controls(definition, memory, prefix, step.get("config"), bounds)
                step["config"] = config
                canonical = registry.configure(step["name"], step["version"], config)
                specs.append({"name": step["name"], "version": step["version"], "config": canonical})
            except Exception as error:
                st.error(f"Step {index + 1}: {error}")
                valid = False
    if not steps:
        st.info("Add at least one processor to create processed data.")
    pipeline = registry.load(specs) if valid else None
    if st.button("Save pipeline recipe", disabled=pipeline is None):
        _save_recipe(memory, pipeline, name)
        st.success("Pipeline recipe saved.")
    if pipeline:
        with st.expander("Exact pipeline configuration"):
            st.caption(f"Pipeline ID: {pipeline.fingerprint}")
            st.json(json.loads(pipeline.canonical_json))
    groups = defaultdict(list)
    for item in selected:
        groups[item.request.symbol].append(item.dataset_id)
    signature = (pipeline.fingerprint if pipeline else None, tuple(sorted(item.to_json() for item in selected)))
    preview_key = "process-preview:" + _scope(research)
    left, right = st.columns(2)
    preview = left.button("Preview processing", disabled=not pipeline or not groups)
    save = right.button("Process and save", type="primary", disabled=not pipeline or not groups)
    if preview:
        results = []
        with st.spinner("Processing all selected input bars for preview…"):
            for symbol, ids in groups.items():
                try:
                    parents, source = research.pipeline.store.read_raw_inputs(ids)
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always", ResamplingWarning)
                        output = pipeline.run(source, parents[0].contract)
                    results.append({"symbol": symbol, "input_rows": source.height, "output_rows": output.frame.height,
                                    "input_timeframe": parents[0].contract.timeframe,
                                    "output_timeframe": output.contract.timeframe,
                                    "warnings": list(dict.fromkeys(str(item.message) for item in captured)),
                                    "before": source.head(100), "after": output.frame.head(100)})
                except Exception as error:
                    results.append({"symbol": symbol, "error": str(error)})
        st.session_state[preview_key] = (signature, results)
    cached = st.session_state.get(preview_key)
    if cached and cached[0] == signature:
        st.subheader("Preview")
        st.caption("The full selected input was processed. Tables show the first 100 rows; preview does not save data.")
        for result in cached[1]:
            st.markdown(f"**{result['symbol']}**")
            if "error" in result:
                st.error(result["error"])
                continue
            st.write(f"{result['input_rows']:,} {result['input_timeframe']} bars → {result['output_rows']:,} {result['output_timeframe']} bars")
            for warning in result.get("warnings", []):
                st.warning(warning)
            before, after = st.columns(2)
            before.caption("Raw input")
            before.dataframe(result["before"], hide_index=True)
            after.caption("Processed output")
            after.dataframe(result["after"], hide_index=True)
    if save:
        report = []
        with st.spinner("Processing and saving each ticker…"):
            for symbol, ids in groups.items():
                try:
                    status = "Saved"
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always", ResamplingWarning)
                        try:
                            output = research.pipeline.process(ids, processors=pipeline)
                        except DataAlreadyExistsError:
                            matches = [item for item in research.pipeline.list_datasets(DataQuery(layer="processed", pipeline_id=pipeline.fingerprint))
                                       if set(item.parent_ids) == set(ids)]
                            if len(matches) != 1:
                                raise
                            output = matches[0]
                            research.pipeline.read_dataset(output.dataset_id)
                            status = "Already processed"
                    report.append({"symbol": symbol, "status": status, "dataset_id": output.dataset_id, "rows": output.row_count,
                                   "warnings": list(dict.fromkeys(str(item.message) for item in captured))})
                except Exception as error:
                    report.append({"symbol": symbol, "status": "Failed", "error": str(error)})
        memory.values["report"] = report
        _save_recipe(memory, pipeline, name)
    report = memory.values.get("report", [])
    if report:
        st.subheader("Last processing results")
        for row in report:
            if row["status"] == "Failed":
                st.error(f"{row['symbol']}: {row['error']}")
            else:
                st.success(f"{row['symbol']}: {row['status']} · {row['rows']:,} rows · {row['dataset_id'][:12]}")
            for warning in row.get("warnings", []):
                st.warning(f"{row['symbol']}: {warning}")
        active = {item.dataset_id for item in research.pipeline.list_datasets(DataQuery(layer="processed"))}
        outputs = [row for row in report if row.get("dataset_id") in active]
        if outputs:
            lookup = {row["dataset_id"]: row["symbol"] + " · " + row["dataset_id"][:12] for row in outputs}
            chosen = st.selectbox("Saved output", list(lookup), format_func=lookup.__getitem__, key="process-output")
            left, right = st.columns(2)
            left.button("Open in Market data", on_click=_open_output, args=(research, chosen, "Market data"))
            right.button("Use in Backtest", on_click=_open_output, args=(research, chosen, "Backtest"))
