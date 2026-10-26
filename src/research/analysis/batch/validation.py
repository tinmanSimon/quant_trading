"""Optional exchange-calendar coverage checks for frozen analysis inputs.

The selected range uses [start, end) bar timestamps. Explicit bounds require
the whole scheduled range, including leading/trailing bars. With omitted
bounds only the supplied span can be certified. All supplied initialization
history is checked too. No rows are filled or silently removed. This validates
structure and coverage, not the economic truth of a vendor's prices.
"""

from datetime import UTC, datetime, timedelta
import json

import exchange_calendars as xcals
import polars as pl

from data_pipeline import validate_ohlcv
from data_pipeline.processing import DataContract, discover_processors
from data_pipeline.storage.models import StoredDataset
from research.errors import ResearchError
from research.instruments import Instrument, expected_bars
from research.processed import _validate_parent_coverage
from research.timeframes import bar_duration


def _preview(stamps, omissions=None):
    omissions = omissions or {}
    return "; ".join(
        stamp.isoformat() + (f" ({omissions[stamp]})" if stamp in omissions else "")
        for stamp in sorted(stamps)[:10]
    )


def _frozen_sources(pipeline, sources):
    items = []
    for source in sources:
        frozen = StoredDataset.from_json(json.dumps(source), active=source.get("active", True))
        current = pipeline.get_metadata(frozen.dataset_id)
        frozen_fields = json.loads(frozen.to_json())
        current_fields = json.loads(current.to_json())
        # Superseded revisions remain valid immutable inputs for a frozen run.
        frozen_fields.pop("active")
        current_fields.pop("active")
        if frozen_fields != current_fields:
            raise ResearchError(f"Frozen source metadata changed for revision {frozen.dataset_id}.")
        items.append(frozen)
    if not items or len({item.dataset_id for item in items}) != len(items):
        raise ResearchError("Strict validation requires distinct frozen source revisions.")
    return items


def _continuous_source_calendar(calendar, bars):
    """The source-coverage mapping supports continuous execution sessions."""
    schedule = xcals.get_calendar(
        calendar, start=str(bars[0].open.date() - timedelta(days=7)),
        end=str(bars[-1].close.date() + timedelta(days=2)),
    ).schedule
    selected = schedule.loc[str(bars[0].open.date()):str(bars[-1].close.date())]
    if "break_start" in selected and selected["break_start"].notna().any():
        raise ResearchError(
            "Cannot verify intraday raw-source coverage for processed bars on a calendar with a lunch break."
        )


def _validate_processed(pipeline, item, bars, registry, parent_cache, calendar):
    """Check static transformation semantics without imposing a trading policy."""
    configured = registry.load(item.processors_json)
    if not configured.identities or configured.fingerprint != item.pipeline_id:
        raise ResearchError("Processed pipeline fingerprint does not match its registered configuration.")
    if not item.parent_ids or len(set(item.parent_ids)) != len(item.parent_ids):
        raise ResearchError("Processed revision requires distinct, exact raw parent revisions.")
    parents = []
    for parent_id in item.parent_ids:
        if parent_id not in parent_cache:
            parent = pipeline.get_metadata(parent_id)
            if parent.layer != "raw":
                raise ResearchError("Processed lineage must point to raw source revisions.")
            # Historical parents are read by immutable ID, even after refetch.
            frame = pipeline.read_dataset(parent_id)
            parent_cache[parent_id] = (parent, frame["timestamp"])
        parents.append(parent_cache[parent_id][0])
    identities = {(parent.request.dataset, parent.request.provider,
                   parent.request.symbol, parent.contract) for parent in parents}
    if len(identities) != 1:
        raise ResearchError("Processed parents have incompatible instruments or source contracts.")
    dataset, provider, symbol, contract = next(iter(identities))
    if (dataset, provider, symbol) != (item.request.dataset, item.request.provider, item.request.symbol):
        raise ResearchError("Processed output instrument does not match its pinned raw lineage.")
    selected = tuple(bar for bar in bars if item.first_timestamp <= bar.timestamp <= item.last_timestamp)
    if selected and contract.timestamp_convention == "bar_start":
        _continuous_source_calendar(calendar, selected)
    for spec in configured.identities:
        processor = registry.create(spec["name"], spec["version"], spec["config"])
        output_contract = getattr(processor, "output_contract", None)
        calendar_validator = getattr(processor, "validate_backtest_calendar", None)
        if not callable(output_contract) or not callable(calendar_validator):
            raise ResearchError("Processed analysis requires verifiable output contracts and calendar mappings.")
        next_contract = output_contract(contract)
        if not isinstance(next_contract, DataContract):
            raise ResearchError("Processor output contract is invalid.")
        # These hooks verify bar timing/coverage, independently of whether the
        # processor is causal or appropriate for a particular return model.
        calendar_validator(selected, input_contract=contract, output_contract=next_contract)
        contract = next_contract
    if contract != item.contract:
        raise ResearchError("Processed output contract does not match its registered pipeline.")
    _validate_parent_coverage(parents, selected, item.contract, parent_cache)


def validate_batch_data(pipeline, request, sources, context):
    """Validate frozen selected/support data and return a JSON-safe audit report.

    Missing bars, unexpected session timestamps, unfinished requested bars,
    unverifiable transformations, and incomplete raw parents raise ResearchError
    (or a strict schema/storage error). The caller chooses whether to invoke
    this optional gate; provider-specific defaults belong to request settings.
    """
    timeframe = request.timeframe
    duration = bar_duration(timeframe)  # Unsupported weekly/monthly grids fail closed.
    calendar = getattr(request, "calendar", "XNYS")
    selected = context.frame
    support = context.support_frame
    if selected.is_empty():
        raise ResearchError("Strict validation requires selected bars.")
    combined = validate_ohlcv(pl.concat(
        [support, selected] if support is not None and not support.is_empty() else [selected]
    ).sort("symbol", "timestamp"))
    symbols = combined["symbol"].unique().to_list()
    if len(symbols) != 1 or symbols[0] not in request.tickers:
        raise ResearchError("Strict analysis inputs must contain exactly one requested ticker.")
    symbol = symbols[0]
    items = _frozen_sources(pipeline, sources)
    for item in items:
        if (item.layer, item.request.provider, item.request.symbol, item.request.timeframe) != (
                request.layer, request.provider, symbol, timeframe):
            raise ResearchError("Frozen source does not match the selected layer, provider, ticker, or timeframe.")
        if item.layer == "processed" and item.pipeline_id != request.pipeline_id:
            raise ResearchError("Frozen processed source does not match the selected pipeline.")
    first, last = combined["timestamp"][0], combined["timestamp"][-1]
    selected_start = request.start if request.start is not None else selected["timestamp"][0]
    checked_start = min(first, selected_start)
    checked_end = request.end if request.end is not None else last + (duration or timedelta(days=1))
    if checked_start >= checked_end:
        raise ResearchError("Strict analysis requires a nonempty validation interval.")
    bars = expected_bars(Instrument(symbol, calendar), start=checked_start,
                         end=checked_end, timeframe=timeframe)
    cutoff = request.as_of or context.cutoff or datetime.now(UTC)
    if request.end is not None:
        cutoff = min(cutoff, request.end)
    unfinished = {bar.timestamp for bar in bars if bar.close > cutoff}
    if unfinished:
        raise ResearchError(
            f"Strict validation found {len(unfinished)} unfinished requested/session bars at the "
            f"end/as-of cutoff {cutoff.isoformat()} (UTC): {_preview(unfinished)}."
        )
    actual = set(combined["timestamp"].to_list())
    needed = {bar.timestamp for bar in bars}
    omissions = {bar.timestamp: bar.reason for item in items for bar in item.quality.omitted_bars}
    missing, extra = needed - actual, actual - needed
    problems = []
    if missing:
        problems.append(f"Missing {len(missing)} required trading/initialization bars (UTC): {_preview(missing, omissions)}")
    if extra:
        problems.append(f"Found {len(extra)} unexpected bars outside the regular-session grid (UTC): {_preview(extra)}")
    if problems:
        raise ResearchError("Strict data validation failed. " + ". ".join(problems))
    parent_cache = {}
    if request.layer == "processed":
        registry = discover_processors()
        for item in items:
            _validate_processed(pipeline, item, bars, registry, parent_cache, calendar)
    notes = [
        "Checks scheduled regular-session coverage and OHLCV structure; does not certify vendor prices or economic accuracy.",
        "All supplied initialization history is checked. History before the first supplied initialization bar is not certified.",
        "Exchange-close completion is validated; an analysis may conservatively use fewer completed bars.",
    ]
    if request.start is None or request.end is None:
        notes.append("Omitted request bounds are derived from the available data span; coverage beyond those bounds is not certified.")
    return {
        "status": "passed", "calendar": calendar, "session_scope": "regular",
        "timeframe": timeframe, "ticker": symbol,
        "checked_start": checked_start.isoformat(), "checked_end_exclusive": checked_end.isoformat(),
        "requested_start": request.start.isoformat() if request.start else None,
        "requested_end_exclusive": request.end.isoformat() if request.end else None,
        "start_bound": "requested" if request.start is not None else "available_span",
        "end_bound": "requested" if request.end is not None else "available_span",
        "completion_cutoff": cutoff.isoformat(), "expected_bar_count": len(needed),
        "actual_bar_count": len(actual), "selected_bar_count": selected.height,
        "initialization_bar_count": 0 if support is None else support.height,
        "analysis_completed_selected_bar_count": context.completed_frame.height,
        "source_dataset_ids": [item.dataset_id for item in items],
        "raw_parent_dataset_ids": sorted(parent_cache), "notes": notes,
    }
