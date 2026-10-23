"""Adjusted research uses pinned prices throughout execution and saved reports."""

from datetime import UTC, date, datetime, timedelta
import json

import polars as pl
import pytest

from data_pipeline import DataRequest
from data_pipeline.corporate_actions.models import ActionCoverage, ActionSnapshot, CorporateAction
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.price_adjustment import AdjustCorporateActions
from research import FactorBacktestSpec, Research
from research.backtesting import ExecutionSettings
from research.factors import ComponentSpec, FactorSpec
from research.instruments import Instrument, expected_bars
from research.runs import comparison_identity
from research.strategies import MovingAverageCross
from research.strategies.bracket import BracketStrategy


def day(number, hour=0):
    return datetime(2024, 1, number, hour, tzinfo=UTC)


class EveryBar(BracketStrategy):
    name, lookback, config = "adjusted_research_every_bar", 1, {}
    stop_percentage, win_percentage = .5, 2.

    def entry_signal(self, history):
        return True


@pytest.fixture
def adjusted(tmp_path):
    app = Research(tmp_path / "data", tmp_path / "runs")
    sessions = expected_bars(Instrument("AAA"), start=day(2), end=day(10), timeframe="1d")
    rows = []
    for index, session in enumerate(sessions):
        price = 100. + index * 10
        for minute in range(int((session.close - session.open).total_seconds() / 60)):
            rows.append({"timestamp": session.open + timedelta(minutes=minute), "symbol": "AAA",
                         "open": price, "high": price, "low": price, "close": price, "volume": 100.})
    frame = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    raw = app.pipeline.ingest_frame(DataRequest("AAA", day(2), day(10), timeframe="1m",
                                               provider="verified-import"), frame).raw
    snapshot = ActionSnapshot(
        "verified-test", date(2024, 1, 31), datetime(2024, 2, 1, tzinfo=UTC),
        (ActionCoverage("AAA", date(2024, 1, 1), date(2024, 1, 31),
                        source_basis="unadjusted", security_id="security-AAA"),),
        (CorporateAction("dividend", "AAA", "cash_dividend", day(8, 5),
                         cash_amount=2., reference_close=130., security_id="security-AAA"),
         # This split is deliberately after the backtest end and as_of date.
         CorporateAction("later-split", "AAA", "forward_split", day(11, 5),
                         share_ratio=2., security_id="security-AAA")),
    )
    output = app.pipeline.process(raw.dataset_id, processors=[
        AdjustCorporateActions(snapshot), ResampleRegularHoursDaily.for_dates(date(2024, 1, 2), date(2024, 1, 9)),
    ])
    kwargs = dict(start=day(5), end=day(10), as_of=day(10), layer="processed", provider="verified-import",
                  pipeline_id=output.pipeline_id, research_mode="adjusted_price")
    return app, raw, output, snapshot, frame, kwargs


def assert_adjusted_saved_run(app, run, raw, output, snapshot):
    restored = app.load_run(run.run_id)
    assert restored.manifest == run.manifest
    assert restored.comparison.equals(run.comparison)
    manifest = run.manifest
    assert manifest["research_mode"] == "adjusted_price"
    assert manifest["price_basis"] == "split_and_cash_dividend_adjusted"
    assert manifest["return_basis"] == "adjusted_price_return_no_additional_dividend_credit"
    assert manifest["corporate_action_adjustment"]["reference_date"] == "2024-01-31"
    assert manifest["corporate_action_adjustment"]["snapshot_checksum"] == snapshot.checksum
    assert manifest["sources"]["AAA"][0]["dataset_id"] == output.dataset_id
    assert manifest["parent_sources"]["AAA"][0]["dataset_id"] == raw.dataset_id
    assert manifest["processors"][0]["config"]["snapshot"] == snapshot.to_dict()
    assert "no additional dividend cash credit" in manifest["assumptions"]["dividends"].lower()
    for result, reloaded in zip(run.results, restored.results):
        assert result.metadata == reloaded.metadata
        assert result.metadata["research_mode"] == "adjusted_price"
        assert result.metadata["dividends"] == "reflected_in_adjusted_prices_no_additional_cash_credit"
        assert result.metadata["return_basis"] == manifest["return_basis"]
        assert result.metrics["return_basis"] == manifest["return_basis"]
        assert result.trades.height > 0
        assert all(decision < execution for decision, execution in result.trades.select(
            "decision_timestamp", "timestamp").iter_rows())
    assert run.comparison["return_basis"].unique().to_list() == [manifest["return_basis"]]


@pytest.mark.parametrize("allocation_mode", ["per_entry", "initial_split", "fixed_tenth"])
def test_adjusted_strategy_runs_reach_every_execution_path_and_preserve_units(adjusted, allocation_mode):
    app, raw, output, snapshot, original, kwargs = adjusted
    strategy = EveryBar() if allocation_mode == "fixed_tenth" else MovingAverageCross(1, 2)
    report = app.preflight(["AAA"], strategies=[strategy], **kwargs)
    assert report.ok
    assert report.warnings
    run = app.backtest(["AAA"], strategies=[strategy], allocation_mode=allocation_mode,
                       settings=ExecutionSettings(initial_cash=1000), **kwargs)
    assert_adjusted_saved_run(app, run, raw, output, snapshot)
    # Jan 5's historical 130 quote becomes 64 after the cash dividend and future split.
    assert run.results[0].trades["price"][0] == pytest.approx(64.)
    assert app.pipeline.read_dataset(raw.dataset_id).equals(original)
    stored_prices = app.pipeline.read_dataset(output.dataset_id)
    assert stored_prices.filter(pl.col("timestamp") == day(5))["open"].item() == pytest.approx(64.)
    assert json.loads(output.contract.adjustment_json)["snapshot_checksum"] == snapshot.checksum
    if allocation_mode != "fixed_tenth":
        # The held position has no cash movements on/after the dividend ex-date.
        closes = run.results[0].equity.filter(pl.col("phase") == "close")
        assert closes["cash"].to_list() == [40., 40., 40.]
        assert closes["holdings"].to_list() == [15, 15, 15]


def test_adjusted_factor_preflight_execution_and_saved_assumptions(adjusted):
    app, raw, output, snapshot, _, kwargs = adjusted
    spec = FactorBacktestSpec(
        factors=(FactorSpec("momentum", lookback=1),),
        portfolio=ComponentSpec("equal_weight", config={"top_n": 1}),
        cash_reserve=0., max_position_weight=1.,
    )
    assert app.factor_preflight(["AAA"], spec=spec, **kwargs).ok
    run = app.factor_backtest(["AAA"], spec=spec, **kwargs)
    assert_adjusted_saved_run(app, run, raw, output, snapshot)
    assert run.results[0].trades["price"][0] == pytest.approx(64.)
    assert run.manifest["assumptions"]["research_mode"] == "adjusted_price"


def test_mode_participates_in_comparison_and_legacy_defaults_remain_strict(adjusted):
    app, _, _, _, _, kwargs = adjusted
    run = app.backtest(["AAA"], strategies=[MovingAverageCross(1, 2)], **kwargs)
    manifest = run.manifest
    assert comparison_identity(manifest) != comparison_identity({**manifest, "research_mode": "strict"})
    legacy = {key: value for key, value in manifest.items() if key != "research_mode"}
    assert comparison_identity(legacy) == comparison_identity({**legacy, "research_mode": "strict"})
