"""Process a requested ticker set with pinned actions and durable batch reports."""

from datetime import UTC, datetime
import json
from pathlib import Path
import warnings
from uuid import uuid4

from .exceptions import DataAlreadyExistsError, ProcessingError
from .models import DataQuery
from .processing import Pipeline


def process_many(app, inputs, *, processors, preview=False):
    """Include every requested ticker, even those without selected raw batches.

    ``inputs`` maps ticker to raw revision IDs. Preview returns before/after
    frames in memory. Saved reports contain only JSON metadata and event rows.
    Both paths pin exactly the snapshot included in the processor recipe.
    """
    pipeline = processors if isinstance(processors, Pipeline) else Pipeline(processors)
    snapshots = []
    for spec in pipeline.identities:
        if spec["name"] == "adjust_corporate_actions":
            from .corporate_actions.models import ActionSnapshot
            snapshots.append(ActionSnapshot.from_dict(spec["config"]["snapshot"]))
    if len(snapshots) > 1:
        raise ProcessingError("Only one corporate-action adjustment step is allowed per pipeline.")
    other_events = []
    for snapshot in snapshots:
        from .corporate_actions.reporting import other_events_report
        from .corporate_actions.storage import SnapshotStore
        SnapshotStore(Path(app.store.data_dir) / "corporate_actions").save(snapshot)
        other_events.extend(other_events_report(snapshot, list(inputs)))
    results = []
    for symbol, raw_ids in inputs.items():
        raw_ids = [raw_ids] if isinstance(raw_ids, str) else list(raw_ids)
        row = {"symbol": symbol, "status": "Failed", "warnings": []}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                if not raw_ids:
                    raise ProcessingError("No raw batches selected for this ticker.")
                parents, source = app.store.read_raw_inputs(raw_ids)
                if any(parent.request.symbol != symbol for parent in parents):
                    raise ProcessingError("Selected raw batches do not match the requested ticker.")
                if snapshots and parents[0].request.provider == "yahoo":
                    raise ProcessingError(
                        "Yahoo raw prices may already contain split adjustments. "
                        "Use a source with verified unadjusted OHLCV for corporate-action processing."
                    )
                row.update(input_rows=source.height, input_timeframe=parents[0].contract.timeframe)
                if preview:
                    output = pipeline.run(source, parents[0].contract)
                    row.update(status="Preview", output_rows=output.frame.height,
                               output_timeframe=output.contract.timeframe,
                               adjustment_json=output.contract.adjustment_json,
                               before=source.head(100), after=output.frame.head(100))
                else:
                    status = "Saved"
                    try:
                        output = app.process(raw_ids, processors=pipeline)
                    except DataAlreadyExistsError:
                        matches = [item for item in app.list_datasets(DataQuery(
                            layer="processed", pipeline_id=pipeline.fingerprint))
                            if set(item.parent_ids) == set(raw_ids)]
                        if len(matches) != 1:
                            raise
                        output = matches[0]
                        app.read_dataset(output.dataset_id)
                        status = "Already processed"
                    row.update(status=status, dataset_id=output.dataset_id, rows=output.row_count,
                               output_timeframe=output.contract.timeframe,
                               adjustment_json=output.contract.adjustment_json)
            except Exception as error:
                row["error"] = str(error)
            row["warnings"] = list(dict.fromkeys(str(item.message) for item in captured))
        results.append(row)
    report = {"created_at": datetime.now(UTC).isoformat(), "pipeline_id": pipeline.fingerprint,
              "processors": pipeline.identities, "requested_tickers": list(inputs),
              "snapshot_checksums": [snapshot.checksum for snapshot in snapshots],
              "other_events": other_events, "results": results}
    if not preview:
        directory = Path(app.store.data_dir) / "processing_reports"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid4().hex}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        report["report_path"] = str(path)
    return report
