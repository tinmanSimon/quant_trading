"""Recorded invalid-bar exclusions cannot create flat periods across a gap."""

from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json

import polars as pl
import pytest

from data_pipeline import DataRequest, FetchQuality, OmittedBar
from research import Research
from research.analysis import LogicContext, LogicSpec, load_registry
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore
from research.analysis.builtins.flat import FlatLogic
from research.analysis.builtins.flat_bar_statistics import FlatBarStatisticsLogic
from research.analysis.core.omissions import analyze_flat_context, omission_metadata, omission_report
from research.analysis.flat import FlatConfig


START = datetime(2024, 1, 2, tzinfo=UTC)


def bars():
    return pl.DataFrame({
        "timestamp": [START + timedelta(days=i) for i in (0, 1, 2, 4, 5, 6)],
        "symbol": ["TEST"] * 6, "open": [100.] * 6, "high": [101.] * 6,
        "low": [99.] * 6, "close": [100.] * 6, "volume": [10.] * 6,
    })


@pytest.mark.parametrize("mode", ["fixed", "automatic"])
def test_flat_periods_stop_at_known_gap_but_not_at_repaired_omission(mode):
    frame = bars()
    config = FlatConfig(min_duration=2, max_duration=20, max_range_pct=3., band_width_mode=mode)
    context = LogicContext(frame, "1d", "test", START + timedelta(days=10), metadata={
        "omitted_bar_timestamps": [(START + timedelta(days=3)).isoformat()],
    })
    progress = []
    result = analyze_flat_context(context, config, lambda done, total: progress.append((done, total)))
    assert [(p.start_index, p.end_index, p.period_id) for p in result.periods] == [(3, 5, "F3-5"), (0, 2, "F0-2")]
    assert result.source_count == 6
    assert all(0 <= done <= total == 6 for done, total in progress)
    assert not analyze_flat_context(context, replace(config, min_duration=4)).periods
    assert len(analyze_flat_context(replace(context, metadata={}), config).periods) == 1
    # A source omission that is present in selected bars has been repaired.
    repaired = replace(context, metadata={"omitted_bar_timestamps": [START.isoformat()]})
    assert len(analyze_flat_context(repaired, config).periods) == 1


def test_batch_flat_statistics_and_flat_analysis_receive_saved_omissions(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    request = DataRequest("TEST", START, START + timedelta(days=7), timeframe="1d")
    research.pipeline.store.write_raw(request, bars(), quality=FetchQuality(
        status="reported", skip_invalid_bars=True,
        omitted_bars=(OmittedBar(START + timedelta(days=3), "invalid_bar: open must be greater than zero"),)))
    specs = [LogicSpec("flat", FlatLogic.name, FlatLogic.version,
                       {"min_duration": 2, "max_duration": 20, "max_range_pct": 3.}),
             LogicSpec("stats", FlatBarStatisticsLogic.name, FlatBarStatisticsLogic.version,
                       {"min_duration": 2, "max_duration": 20, "max_range_pct": 3.})]
    store = BatchStore(tmp_path / "results")
    manifest = BatchRunner(research.pipeline, load_registry(), store).execute(
        BatchRequest(("TEST",), provider="yahoo", timeframe="1d", as_of=START + timedelta(days=10)), specs)
    assert manifest["counts"]["completed"] == 1
    results = store.load_ticker(manifest["run_id"], "TEST")["results"]
    summary = results["stats"]["tables"][0]["rows"][0]
    assert summary["flat_period_count"] == 2
    assert summary["flat_bar_count"] == 6
    assert summary["return_count"] == 4
    assert summary["return_volatility_pct"] == 0
    assert any("recorded missing bars" in message.lower() for message in results["flat"]["messages"])


def test_omission_report_includes_requested_edges_and_ignores_repaired_bars():
    context = LogicContext(bars(), "1d", "test", START + timedelta(days=10), metadata={
        "omitted_bar_timestamps": [
            (START - timedelta(days=1)).isoformat(), START.isoformat(),
            (START + timedelta(days=3)).isoformat(), (START + timedelta(days=7)).isoformat(),
            (START + timedelta(days=8)).isoformat(),
        ],
    })
    report = omission_report(context, start=START - timedelta(days=1), end=START + timedelta(days=8))
    assert report["unresolved_omitted_bar_count"] == 3
    assert START.isoformat() not in report["unresolved_omitted_bar_timestamps"]
    assert "incomplete history" in report["messages"][0]
    assert omission_report(context)["unresolved_omitted_bar_count"] == 1


def test_metadata_normalizes_equivalent_utc_offsets():
    metadata = omission_metadata([{"quality": {"omitted_bars": [
        {"timestamp": "2024-01-05T00:00:00+00:00"},
        {"timestamp": "2024-01-04T19:00:00-05:00"},
    ]}}])
    assert metadata["omitted_bar_timestamps"] == ["2024-01-05T00:00:00+00:00"]


@pytest.mark.parametrize("column", ["symbol", "ticker", "instrument", "timeframe"])
def test_gap_segmentation_cannot_hide_mixed_series(column):
    frame = bars().with_columns(pl.Series(column, ["AAA"] * 3 + ["BBB"] * 3))
    context = LogicContext(frame, "1d", "mixed", START + timedelta(days=10), metadata={
        "omitted_bar_timestamps": [(START + timedelta(days=3)).isoformat()],
    })
    with pytest.raises(ValueError, match=f"Mixed {column}"):
        analyze_flat_context(context, FlatConfig(min_duration=2, max_duration=20))


@pytest.mark.parametrize("invalid", ["2024-01-05T00:00:00", "2024-01-05"])
def test_naive_omission_timestamps_fail_explicitly(invalid):
    context = LogicContext(bars(), "1d", "test", START + timedelta(days=10),
                           metadata={"omitted_bar_timestamps": [invalid]})
    with pytest.raises(ValueError, match="timezone-aware"):
        analyze_flat_context(context, FlatConfig())


@pytest.mark.parametrize("repaired", [False, True])
def test_processed_output_label_does_not_repair_missing_raw_minute(tmp_path, repaired):
    from data_pipeline.processing.processors.intraday import ResampleIntraday, ResamplingWarning
    from data_pipeline.processing.processors.resampling import TradingSession

    research = Research(tmp_path / "data", tmp_path / "runs")
    opened = START + timedelta(hours=14, minutes=30)
    omitted = opened + timedelta(minutes=30)
    minutes = [index for index in range(90) if repaired or index != 30]
    frame = pl.DataFrame({
        "timestamp": [opened + timedelta(minutes=index) for index in minutes],
        "symbol": ["TEST"] * len(minutes), "open": [100.] * len(minutes),
        "high": [101.] * len(minutes), "low": [99.] * len(minutes),
        "close": [100.] * len(minutes), "volume": [10.] * len(minutes),
    })
    request = DataRequest("TEST", opened, opened + timedelta(minutes=90), timeframe="1m")
    raw = research.pipeline.store.write_raw(request, frame, quality=FetchQuality(
        status="reported", skip_invalid_bars=True,
        omitted_bars=(OmittedBar(omitted, "invalid_bar: invalid open"),)))
    processor = ResampleIntraday((TradingSession(START.date(), opened, opened + timedelta(minutes=90)),),
                                 incomplete="aggregate_available")
    if repaired:
        processed = research.pipeline.process(raw.dataset_id, processors=[processor])
    else:
        with pytest.warns(ResamplingWarning, match="Aggregated available"):
            processed = research.pipeline.process(raw.dataset_id, processors=[processor])
    output = research.pipeline.read_dataset(processed.dataset_id)
    assert omitted in output["timestamp"].to_list()
    metadata = omission_metadata([json.loads(processed.to_json())], pipeline=research.pipeline)
    assert metadata["omitted_bar_timestamps"] == []
    context = LogicContext(output, "30m", "processed", START + timedelta(days=1), metadata=metadata)
    config = FlatConfig(min_duration=2, max_duration=20, max_range_pct=3.)
    if repaired:
        assert not omission_report(context)["messages"]
        assert len(analyze_flat_context(context, config).periods) == 1
    else:
        assert metadata["processed_omission_sources"][0]["omitted_bar_timestamps"] == [omitted.isoformat()]
        assert "unresolved omitted raw-input" in omission_report(context)["messages"][0]
        with pytest.raises(ValueError, match="unresolved omitted raw-input"):
            analyze_flat_context(context, config)


def test_market_data_does_not_label_processed_omission_repaired():
    from types import SimpleNamespace
    from dashboard.pages.market_data import _quality

    item = SimpleNamespace(dataset_id="processed", layer="processed", quality=FetchQuality(
        status="reported", omitted_bars=(OmittedBar(START, "invalid_bar: invalid open"),)))
    calls = {name: [] for name in ("caption", "info", "warning", "dataframe")}
    fake = SimpleNamespace(**{name: (lambda value, _name=name, **kwargs: calls[_name].append(value))
                              for name in calls})
    marks = _quality([item], START, START + timedelta(days=1), present_timestamps=[START], ui=fake)
    assert marks == []
    assert "unresolved omitted raw-input" in calls["warning"][0]
    assert not calls["info"]
    assert calls["dataframe"][0]["current_state"].to_list() == ["unresolved in raw inputs"]
