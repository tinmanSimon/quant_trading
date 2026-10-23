"""The retrospective corporate-action exception is explicit and narrowly scoped."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json

import polars as pl
import pytest

from data_pipeline import DataPipeline, DataRequest
from data_pipeline.corporate_actions.models import ActionCoverage, ActionSnapshot, CorporateAction
from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import DataContract, ScalePrices, discover_processors
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.price_adjustment import AdjustCorporateActions
from data_pipeline.processing.registry import ProcessorRegistry
from research.datasets import prepare_snapshot
from research.errors import PreflightError, ResearchError
from research.modes import ADJUSTED_PRICE_BASIS, ADJUSTED_RETURN_BASIS, TRANSFORMED_ADJUSTED_PRICE_BASIS


def day(number, hour=0):
    return datetime(2024, 1, number, hour, tzinfo=UTC)


@pytest.fixture
def history(tmp_path):
    pipeline = DataPipeline(tmp_path / "data")
    evidence = ActionSnapshot(
        "verified-import", date(2024, 1, 31), datetime(2024, 2, 1, tzinfo=UTC),
        (ActionCoverage("AAPL", date(2024, 1, 1), date(2024, 1, 31),
                        source_basis="unadjusted", security_id="AAPL"),),
        (CorporateAction("future-split", "AAPL", "forward_split", day(10, 5),
                         security_id="AAPL", share_ratio=2.),
         CorporateAction("future-dividend", "AAPL", "cash_dividend", day(11, 5),
                         security_id="AAPL", price_factor=.98)),
    )
    adjuster = AdjustCorporateActions(evidence)
    daily = ResampleRegularHoursDaily.for_dates(date(2024, 1, 2), date(2024, 1, 4))
    stamps = [session.open + timedelta(minutes=minute)
              for session in daily.sessions for minute in range(390)]
    source = pl.DataFrame({
        "timestamp": stamps, "symbol": ["AAPL"] * len(stamps),
        "open": [100.] * len(stamps), "high": [102.] * len(stamps),
        "low": [99.] * len(stamps), "close": [101.] * len(stamps),
        "volume": [1000.] * len(stamps),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    raw = pipeline.ingest_frame(DataRequest("AAPL", day(2), day(5), timeframe="1m",
                                           provider="verified-import"), source).raw
    return pipeline, raw, adjuster, daily


def prepare(pipeline, output, **kwargs):
    return prepare_snapshot(pipeline, tickers=["AAPL"], start=day(2), end=day(5),
                            timeframe="1d", lookback=0, layer="processed",
                            pipeline_id=output.pipeline_id, provider="verified-import", **kwargs)


def test_adjusted_mode_accepts_pinned_future_events_and_preserves_lineage(history):
    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[adjuster, daily])
    with pytest.raises(PreflightError, match="Select Adjusted-price research mode"):
        prepare(pipeline, output)
    selected = prepare(pipeline, output, research_mode="adjusted_price")
    assert selected.report.ok
    assert selected.frames["AAPL"]["open"].to_list() == pytest.approx([49.] * 3)
    assert selected.manifest["research_mode"] == "adjusted_price"
    assert selected.manifest["price_basis"] == ADJUSTED_PRICE_BASIS
    assert selected.manifest["return_basis"] == ADJUSTED_RETURN_BASIS
    assert selected.manifest["corporate_action_adjustment"] == json.loads(output.contract.adjustment_json)
    assert selected.manifest["corporate_action_adjustment"]["snapshot_checksum"] == adjuster.snapshot.checksum
    assert selected.manifest["corporate_action_adjustment"]["reference_date"] == "2024-01-31"
    assert selected.manifest["parent_sources"]["AAPL"][0]["dataset_id"] == raw.dataset_id
    assert selected.manifest["sources"]["AAPL"][0]["dataset_id"] == output.dataset_id
    assert selected.report.warnings[0]["code"] == "adjusted_price_research"
    assert not discover_processors().definition(adjuster.name, adjuster.version).causal


@pytest.mark.parametrize("mode", [None, True, "permissive", "adjusted", []])
def test_invalid_research_modes_reject_before_dataset_selection(tmp_path, mode):
    pipeline = DataPipeline(tmp_path / "data")
    with pytest.raises(ResearchError, match="research_mode must be"):
        prepare_snapshot(pipeline, tickers=["AAPL"], start=day(2), end=day(5),
                         timeframe="1d", lookback=0, research_mode=mode)


def test_adjusted_mode_requires_processed_adjusted_data(history):
    pipeline, raw, _, daily = history
    with pytest.raises(ResearchError, match="requires processed data"):
        prepare_snapshot(pipeline, tickers=["AAPL"], start=day(2), end=day(5),
                         timeframe="1d", lookback=0, research_mode="adjusted_price")
    output = pipeline.process(raw.dataset_id, processors=[daily])
    with pytest.raises(PreflightError, match="requires a pipeline containing the built-in"):
        prepare(pipeline, output, research_mode="adjusted_price")
    assert prepare(pipeline, output).manifest["research_mode"] == "strict"


def test_other_price_transformations_are_labelled(history):
    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[adjuster, ScalePrices(2), daily])
    selected = prepare(pipeline, output, research_mode="adjusted_price")
    assert selected.manifest["price_basis"] == TRANSFORMED_ADJUSTED_PRICE_BASIS
    assert selected.manifest["return_basis"] == ADJUSTED_RETURN_BASIS


def test_adjusted_mode_does_not_allow_other_noncausal_processors(history, monkeypatch):
    import data_pipeline.processing as processing

    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[adjuster, ScalePrices(2), daily])
    registry = discover_processors()
    definition = registry.definition
    monkeypatch.setattr(registry, "definition", lambda name, version: (
        replace(definition(name, version), causal=False) if name == "scale_prices"
        else definition(name, version)
    ))
    monkeypatch.setattr(processing, "discover_processors", lambda: registry)
    with pytest.raises(PreflightError, match="scale_prices.*no declared causality"):
        prepare(pipeline, output, research_mode="adjusted_price")


@pytest.mark.parametrize("causal", [False, True])
def test_adjustment_identity_cannot_be_spoofed_by_another_class(history, monkeypatch, causal):
    import data_pipeline.processing as processing

    class Impostor(AdjustCorporateActions):
        pass

    Impostor.causal = causal
    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[Impostor(adjuster.snapshot), daily])
    registry = ProcessorRegistry()
    registry.register_class(Impostor)
    registry.register_class(ResampleRegularHoursDaily)
    monkeypatch.setattr(processing, "discover_processors", lambda: registry)
    message = "requires a pipeline containing the built-in" if causal else "no declared causality"
    with pytest.raises(PreflightError, match=message):
        prepare(pipeline, output, research_mode="adjusted_price")


def test_later_processor_cannot_discard_adjustment_metadata(history, monkeypatch):
    import data_pipeline.processing as processing

    class DiscardAdjustment(ScalePrices):
        name = "discard_adjustment"

        def output_contract(self, input_contract):
            return replace(input_contract, adjustment_json="")

    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[adjuster, DiscardAdjustment(1), daily])
    registry = discover_processors()
    registry.register_class(DiscardAdjustment)
    monkeypatch.setattr(processing, "discover_processors", lambda: registry)
    with pytest.raises(PreflightError, match="preserve the pinned corporate-action adjustment metadata"):
        prepare(pipeline, output, research_mode="adjusted_price")


def test_adjusted_mode_retains_calendar_and_completed_bar_checks(history, monkeypatch):
    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[adjuster, daily])
    with pytest.raises(PreflightError, match="Requested bars are not complete"):
        prepare(pipeline, output, research_mode="adjusted_price", as_of=day(4, 20))

    def reject_calendar(self, bars, *, input_contract, output_contract):
        raise ProcessingError("Calendar mismatch must still block adjusted research.")

    monkeypatch.setattr(ResampleRegularHoursDaily, "validate_backtest_calendar", reject_calendar)
    with pytest.raises(PreflightError, match="Calendar mismatch"):
        prepare(pipeline, output, research_mode="adjusted_price")


def test_adjusted_mode_retains_immutable_parent_checksums(history):
    pipeline, raw, adjuster, daily = history
    output = pipeline.process(raw.dataset_id, processors=[adjuster, daily])
    (pipeline.store.data_dir / raw.relative_path).write_bytes(b"corrupted parent data")
    with pytest.raises(PreflightError, match="Checksum mismatch"):
        prepare(pipeline, output, research_mode="adjusted_price")


def test_adjustment_calendar_hook_cannot_change_bar_timing(history):
    _, _, adjuster, _ = history
    source = DataContract("1m")
    expected = adjuster.output_contract(source)
    adjuster.validate_backtest_calendar((), input_contract=source, output_contract=expected)
    with pytest.raises(ProcessingError, match="preserve bar timing"):
        adjuster.validate_backtest_calendar((), input_contract=source,
                                           output_contract=replace(expected, timeframe="1d", timestamp_convention=None))
