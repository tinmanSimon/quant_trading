"""Processed histories retain strict preflight, provenance and bar availability."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json

import polars as pl
import pytest

from data_pipeline import DataRequest
from data_pipeline.processing import Pipeline, ResampleOHLCV, ScalePrices, TradingSession
from research import PreflightError, Research, ResearchError
from research.datasets import prepare_snapshot
from research.instruments import Instrument, expected_bars
from research.strategies import MovingAverageCross


def day(number, hour=0, minute=0):
    return datetime(2024, 1, number, hour, minute, tzinfo=UTC)


def price_frame(bars, symbol="AAPL"):
    size = len(bars)
    return pl.DataFrame({
        "timestamp": [bar.timestamp for bar in bars], "symbol": [symbol] * size,
        "open": [100. + index for index in range(size)],
        "high": [102. + index for index in range(size)],
        "low": [99. + index for index in range(size)],
        "close": [101. + index for index in range(size)], "volume": [1000.] * size,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


@pytest.fixture
def app(tmp_path):
    return Research(tmp_path / "data", tmp_path / "runs")


def ingest(app, *, symbol="AAPL", timeframe="1d", start=None):
    start = start or day(2)
    bars = expected_bars(Instrument(symbol), start=start, end=day(10), timeframe=timeframe)
    raw = app.pipeline.ingest_frame(DataRequest(symbol, start, day(10), timeframe=timeframe),
                                    price_frame(bars, symbol)).raw
    return raw, bars


def selected(app, pipeline_id, **kwargs):
    return app.backtest(["AAPL"], strategies=[MovingAverageCross(1, 2)],
                        start=day(5), end=day(10), layer="processed", pipeline_id=pipeline_id, **kwargs)


def resample(app, *, offset=timedelta()):
    hourly = expected_bars(Instrument("AAPL"), start=day(2), end=day(10), timeframe="1h")
    source = price_frame(hourly).with_columns(pl.col("timestamp") + offset)
    raw = app.pipeline.ingest_frame(DataRequest("AAPL", day(2), day(10), timeframe="1h"), source).raw
    daily = expected_bars(Instrument("AAPL"), start=day(2), end=day(10), timeframe="1d")
    sessions = tuple(TradingSession(bar.timestamp.date(), bar.open + offset, bar.close + offset)
                     for bar in daily)
    output = app.pipeline.process(raw.dataset_id, processors=[ResampleOHLCV(sessions)])
    return raw, output


def test_scaling_uses_selected_pipeline_and_records_exact_lineage(app):
    raw, _ = ingest(app)
    original = app.pipeline.read_dataset(raw.dataset_id)
    twice = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    three = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(3)])
    run = selected(app, twice.pipeline_id)
    assert run.results[0].equity["price"][-1] == original["close"][-1] * 2
    assert run.results[0].trades["price"][0] == original.filter(pl.col("timestamp") == day(5))["open"][0] * 2
    assert run.results[0].metadata["price_basis"] == "processed_transformed_prices"
    assert run.manifest["pipeline_id"] == twice.pipeline_id
    assert run.manifest["processors"] == Pipeline([ScalePrices(2)]).identities
    assert run.manifest["sources"]["AAPL"][0]["dataset_id"] == twice.dataset_id
    assert run.manifest["parent_sources"]["AAPL"][0]["dataset_id"] == raw.dataset_id
    assert run.manifest["parent_sources"]["AAPL"][0]["checksum_sha256"] == raw.checksum_sha256
    assert app.load_run(run.run_id).manifest == run.manifest
    assert app.pipeline.read_dataset(raw.dataset_id).equals(original)
    same = selected(app, twice.pipeline_id)
    assert app.compare_runs([run.run_id, same.run_id]).height == 2
    different = selected(app, three.pipeline_id)
    with pytest.raises(ResearchError, match="different"):
        app.compare_runs([run.run_id, different.run_id])


def test_processed_selection_never_falls_back_to_raw_or_another_pipeline(app):
    raw, _ = ingest(app)
    app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    with pytest.raises(PreflightError) as error:
        selected(app, Pipeline([ScalePrices(3)]).fingerprint)
    assert [issue.code for issue in error.value.report.issues] == ["no_data"]


def test_different_ticker_pipeline_cannot_be_mixed(app):
    first, _ = ingest(app)
    second, _ = ingest(app, symbol="MSFT")
    output = app.pipeline.process(first.dataset_id, processors=[ScalePrices(2)])
    app.pipeline.process(second.dataset_id, processors=[ScalePrices(3)])
    report = app.preflight(["AAPL", "MSFT"], strategies=[MovingAverageCross(1, 2)],
                           start=day(5), end=day(10), layer="processed", pipeline_id=output.pipeline_id)
    assert not report.ok
    assert [(issue.ticker, issue.code) for issue in report.issues] == [("MSFT", "no_data")]
    assert app.list_runs() == []


def test_hourly_resampling_uses_completed_sessions_before_next_open(app):
    raw, output = resample(app)
    run = selected(app, output.pipeline_id)
    result = run.results[0]
    assert result.trades["decision_timestamp"][0] == day(4, 21)
    assert result.trades["timestamp"][0] == day(5, 14, 30)
    assert result.equity["valuation_time"][-1] == day(9, 21)
    assert run.manifest["price_basis"] == "vendor_unadjusted"
    assert run.manifest["parent_sources"]["AAPL"][0]["request"]["timeframe"] == "1h"
    assert run.manifest["sources"]["AAPL"][0]["request"]["timeframe"] == "1d"
    with pytest.raises(PreflightError) as error:
        selected(app, output.pipeline_id, as_of=day(9, 20))
    assert any(issue.code == "unfinished_bars" for issue in error.value.report.issues)


def test_resampling_rejects_sessions_that_complete_after_calendar_close(app):
    _, output = resample(app, offset=timedelta(hours=1))
    with pytest.raises(PreflightError, match="completion times"):
        selected(app, output.pipeline_id)
    assert app.list_runs() == []


def test_processed_histories_still_require_every_warmup_bar(app):
    raw, _ = ingest(app, start=day(5))
    output = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    with pytest.raises(PreflightError) as error:
        selected(app, output.pipeline_id)
    missing = next(issue for issue in error.value.report.issues if issue.code == "missing_bars")
    assert missing.timestamps == (day(3), day(4))


@pytest.mark.parametrize("mode", ["unavailable", "noncausal"])
def test_unavailable_or_unclassified_processors_cannot_enter_backtests(app, monkeypatch, mode):
    import data_pipeline.processing as processing
    from data_pipeline.processing.registry import ProcessorRegistry

    raw, _ = ingest(app)
    output = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    registry = processing.discover_processors()
    if mode == "unavailable":
        registry = ProcessorRegistry()
    else:
        definition = registry.definition
        monkeypatch.setattr(registry, "definition", lambda *args: replace(definition(*args), causal=False))
    monkeypatch.setattr(processing, "discover_processors", lambda: registry)
    with pytest.raises(PreflightError) as error:
        selected(app, output.pipeline_id)
    expected = "Unknown processor" if mode == "unavailable" else "causality"
    assert expected in str(error.value)
    assert app.list_runs() == []


def test_changed_effective_specs_do_not_match_stored_fingerprint(app, monkeypatch):
    raw, _ = ingest(app)
    output = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    changed = replace(output, processors_json=Pipeline([ScalePrices(3)]).canonical_json)
    monkeypatch.setattr(app.pipeline, "list_datasets", lambda query: [changed])
    with pytest.raises(PreflightError, match="fingerprint"):
        selected(app, output.pipeline_id)


def test_processed_output_contract_must_match_the_loaded_pipeline(app):
    from data_pipeline.processing import DataContract

    raw, _ = ingest(app, timeframe="1h")
    spec = Pipeline([ScalePrices(2)])
    daily = expected_bars(Instrument("AAPL"), start=day(2), end=day(10), timeframe="1d")
    output = app.pipeline.store.write_processed(
        raw.dataset_id, price_frame(daily), pipeline_id=spec.fingerprint,
        processors_json=spec.canonical_json, output_contract=DataContract("1d"),
    )
    with pytest.raises(PreflightError, match="output timeframe/contract"):
        selected(app, output.pipeline_id)


def test_corrupted_parent_source_is_not_hidden_by_valid_processed_output(app):
    raw, _ = ingest(app)
    output = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    (app.pipeline.store.data_dir / raw.relative_path).write_bytes(b"corrupted raw data")
    with pytest.raises(PreflightError, match="Checksum mismatch"):
        selected(app, output.pipeline_id)


def test_source_instrument_lineage_must_match_processed_revision(app, monkeypatch):
    raw, _ = ingest(app)
    output = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    metadata = app.pipeline.get_metadata
    monkeypatch.setattr(app.pipeline, "get_metadata", lambda key: (
        replace(raw, request=replace(raw.request, symbol="MSFT")) if key == raw.dataset_id else metadata(key)
    ))
    with pytest.raises(PreflightError, match="source lineage"):
        selected(app, output.pipeline_id)


def test_selection_requires_one_explicit_pipeline_and_raw_rejects_pipeline_id(app):
    kwargs = dict(tickers=["AAPL"], start=day(5), end=day(10), timeframe="1d", lookback=2)
    with pytest.raises(ResearchError, match="explicit pipeline_id"):
        prepare_snapshot(app.pipeline, **kwargs, layer="processed")
    with pytest.raises(ResearchError, match="must not specify"):
        prepare_snapshot(app.pipeline, **kwargs, pipeline_id="abc")
    assert not app.pipeline.store.catalog_path.exists()


def test_cli_selects_processed_layer(app, tmp_path, capsys):
    from research.cli import main

    raw, _ = ingest(app)
    output = app.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    specs = tmp_path / "strategies.json"
    specs.write_text(json.dumps([{"name": "moving_average_cross", "version": "1", "config": {"fast": 1, "slow": 2}}]))
    status = main(["--data-dir", str(app.pipeline.store.data_dir), "--runs-dir", str(app.runs_dir),
                   "backtest", "--tickers", "AAPL", "--start", "2024-01-05", "--end", "2024-01-10",
                   "--strategies", str(specs), "--layer", "processed", "--pipeline-id", output.pipeline_id])
    assert status == 0, capsys.readouterr().out
    run = app.load_run(app.list_runs()[0]["run_id"])
    assert run.manifest["pipeline_id"] == output.pipeline_id


def test_new_processor_owns_calendar_validation_without_research_changes(app, monkeypatch):
    import data_pipeline.processing as processing
    from data_pipeline.exceptions import ProcessingError

    class CustomScale(ScalePrices):
        name = "custom_scale"

        def validate_backtest_calendar(self, bars, *, input_contract, output_contract):
            raise ProcessingError("Custom processor calendar rule rejected these bars.")

    registry = processing.discover_processors()
    registry.register_class(CustomScale)
    monkeypatch.setattr(processing, "discover_processors", lambda: registry)
    raw, _ = ingest(app)
    output = app.pipeline.process(raw.dataset_id, processors=[CustomScale(2)])
    with pytest.raises(PreflightError, match="Custom processor calendar rule"):
        selected(app, output.pipeline_id)


def test_new_timeframe_changing_processor_needs_explicit_calendar_validation(app, monkeypatch):
    import data_pipeline.processing as processing
    from data_pipeline.processing import BaseProcessor

    class UnverifiedDaily(ResampleOHLCV):
        name = "unverified_daily"
        validate_backtest_calendar = BaseProcessor.validate_backtest_calendar

    registry = processing.discover_processors()
    registry.register_class(UnverifiedDaily)
    monkeypatch.setattr(processing, "discover_processors", lambda: registry)
    raw, _ = ingest(app, timeframe="1h")
    daily = expected_bars(Instrument("AAPL"), start=day(2), end=day(10), timeframe="1d")
    sessions = tuple(TradingSession(bar.timestamp.date(), bar.open, bar.close) for bar in daily)
    output = app.pipeline.process(raw.dataset_id, processors=[UnverifiedDaily(sessions)])
    with pytest.raises(PreflightError, match="calendar"):
        selected(app, output.pipeline_id)
