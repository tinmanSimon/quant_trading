"""Verify processed provenance and the availability of derived price bars."""


from data_pipeline import DataPipeline
from data_pipeline.processing import DataContract
from data_pipeline.storage.models import StoredDataset

from .errors import ResearchError
from .instruments import ExpectedBar


def validate_processed_source(
    pipeline: DataPipeline,
    item: StoredDataset,
    *,
    pipeline_id: str,
    registry,
    bars: tuple[ExpectedBar, ...],
    parent_cache: dict,
) -> tuple[list[dict], list[StoredDataset], str]:
    """Validate exact registered semantics without rerunning a transformation.

    ``causal`` is the processor author's explicit guarantee that output values
    require no observations beyond the output bar's completion. It defaults to
    false. Each processor also owns validation of its output's availability
    against the execution calendar; transformations that change timeframe need
    an explicit implementation of that calendar validation hook.
    """
    if item.layer != "processed" or not item.active or item.pipeline_id != pipeline_id:
        raise ResearchError("Processed revision does not match the selected active pipeline.")
    configured = registry.load(item.processors_json)
    specs = configured.identities
    if not specs or configured.fingerprint != pipeline_id:
        raise ResearchError("Processed pipeline fingerprint does not match its effective processor configuration.")
    if not item.parent_ids or len(set(item.parent_ids)) != len(item.parent_ids):
        raise ResearchError("Processed revision requires distinct, exact raw parent revisions.")

    parents = []
    for parent_id in item.parent_ids:
        if parent_id not in parent_cache:
            parent = pipeline.get_metadata(parent_id)
            if parent.layer != "raw":
                raise ResearchError("Processed lineage must point to raw source revisions.")
            # Verify the immutable source checksum, even when the parent is
            # historical. Never replace a selected parent with an active query.
            pipeline.read_dataset(parent_id)
            parent_cache[parent_id] = parent
        parents.append(parent_cache[parent_id])
    identities = {(parent.request.dataset, parent.request.provider,
                   parent.request.symbol, parent.contract) for parent in parents}
    if len(identities) != 1:
        raise ResearchError("Processed parents have incompatible instrument or source contracts.")
    dataset, provider, symbol, contract = next(iter(identities))
    if (dataset, provider, symbol) != (item.request.dataset, item.request.provider, item.request.symbol):
        raise ResearchError("Processed output instrument does not match its raw source lineage.")

    price_basis = "vendor_unadjusted"
    selected_bars = tuple(bar for bar in bars
                          if item.first_timestamp <= bar.timestamp <= item.last_timestamp)
    for spec in specs:
        definition = registry.definition(spec["name"], spec["version"])
        if not definition.causal:
            raise ResearchError(
                f"Processor {spec['name']!r} version {spec['version']!r} has no declared causality guarantee for backtesting."
            )
        if definition.price_basis == "transformed":
            price_basis = "processed_transformed_prices"
        elif definition.price_basis != "preserve":
            raise ResearchError(f"Processor {spec['name']!r} has an unsupported execution price basis.")
        processor = registry.create(spec["name"], spec["version"], spec["config"])
        output_contract = getattr(processor, "output_contract", None)
        if not callable(output_contract):
            raise ResearchError(f"Processor {spec['name']!r} must declare its output contract for backtesting.")
        next_contract = output_contract(contract)
        if not isinstance(next_contract, DataContract):
            raise ResearchError("Processor output contract is invalid.")
        calendar_validator = getattr(processor, "validate_backtest_calendar", None)
        if not callable(calendar_validator):
            raise ResearchError(f"Processor {spec['name']!r} must declare its backtest calendar validation.")
        calendar_validator(selected_bars, input_contract=contract, output_contract=next_contract)
        contract = next_contract
    if contract != item.contract:
        raise ResearchError("Processed output timeframe/contract does not match its processor pipeline.")
    return specs, parents, price_basis
