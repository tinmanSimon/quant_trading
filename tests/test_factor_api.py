"""Factor API acceptance tests with real local storage and audited saved runs."""

from datetime import UTC, datetime
from hashlib import sha256
import json

import polars as pl
import pytest

from data_pipeline import DataRequest
from research import FactorBacktestSpec, Instrument, PreflightError, Research, ResearchError
from research.backtesting import BacktestResult, ExecutionSettings
from research.factors import ComponentSpec, FactorSpec
from research.runs import load_run, save_run


def day(number):
    return datetime(2024, 1, number, tzinfo=UTC)


def bars(symbol, days=(2, 3, 4, 5, 8, 9)):
    prices = [100.0 + index * (2 if symbol == "AAA" else 1) for index in range(len(days))]
    return pl.DataFrame({
        "timestamp": [day(value) for value in days], "symbol": [symbol] * len(days),
        "open": prices, "high": [price + 2 for price in prices],
        "low": [price - 1 for price in prices], "close": [price + 1 for price in prices],
        "volume": [1000.0] * len(days),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


def store(app, symbol, days=(2, 3, 4, 5, 8, 9)):
    return app.pipeline.store.write_raw(
        DataRequest(symbol=symbol, provider="yahoo", timeframe="1d", start=day(2), end=day(10)),
        bars(symbol, days),
    )


def spec(**kwargs):
    return FactorBacktestSpec(
        factors=(FactorSpec("momentum", lookback=1),),
        portfolio=kwargs.pop("portfolio", ComponentSpec("equal_weight", config={"top_n": 1})),
        cash_reserve=0.0, max_position_weight=1.0, **kwargs,
    )


@pytest.fixture
def app(tmp_path):
    return Research(tmp_path / "data", tmp_path / "runs")


def run(app, symbols=("AAA", "BBB"), **kwargs):
    return app.factor_backtest(symbols, spec=kwargs.pop("spec", spec()), start=day(5), end=day(10), **kwargs)


def test_factor_preflight_does_not_load_strategy_registry_or_create_run(app, monkeypatch):
    import research.api as api

    store(app, "AAA")
    monkeypatch.setattr(api, "load_registry", lambda: pytest.fail("Factor flow loaded private strategies."))
    report = app.factor_preflight(["AAA"], spec=spec(), start=day(5), end=day(10))
    assert report.ok
    assert app.list_runs() == []


def test_factor_preflight_reports_missing_history_and_unfinished_bars(app):
    store(app, "AAA")
    report = app.factor_preflight(["AAA", "BBB"], spec=spec(), start=day(5), end=day(10))
    assert [(issue.ticker, issue.code) for issue in report.issues] == [("BBB", "no_data")]
    unfinished = app.factor_preflight(["AAA"], spec=spec(), start=day(5), end=day(10), as_of=day(8))
    assert any(issue.code == "unfinished_bars" for issue in unfinished.issues)


def test_mixed_calendars_rejected_before_any_snapshot_read(tmp_path, monkeypatch):
    import research.api as api

    app = Research(tmp_path / "data", tmp_path / "runs", instruments={"BBB": Instrument("BBB", "XLON")})
    monkeypatch.setattr(api, "prepare_snapshot", lambda *args, **kwargs: pytest.fail("Read data before calendar validation."))
    with pytest.raises(ResearchError, match="common exchange calendar"):
        app.factor_preflight(["AAA", "BBB"], spec=spec(), start=day(5), end=day(10))


def test_factor_configuration_rejected_before_snapshot_read(app, monkeypatch):
    import research.api as api

    monkeypatch.setattr(api, "prepare_snapshot", lambda *args, **kwargs: pytest.fail("Read data with invalid configuration."))
    with pytest.raises((ValueError, ResearchError), match="unknown|Unknown|registered|Unsupported"):
        app.factor_preflight(["AAA"], spec=spec(portfolio=ComponentSpec("not_a_policy")), start=day(5), end=day(10))
    with pytest.raises(TypeError, match="spec"):
        app.factor_preflight(["AAA"], spec="momentum", start=day(5), end=day(10))
    with pytest.raises(ValueError, match="nonreserved"):
        app.factor_preflight(["AAA"], spec=spec(portfolio=ComponentSpec("inverse_volatility", config={
            "volatility_factor": "score"})), start=day(5), end=day(10))


def test_factor_round_trip_preserves_all_audit_tables_and_original_bars(app):
    originals = {symbol: store(app, symbol) for symbol in ("AAA", "BBB")}
    original_frames = {symbol: app.pipeline.read_dataset(item.dataset_id) for symbol, item in originals.items()}
    saved = run(app, spec=spec().to_dict())
    restored = app.load_run(saved.run_id)
    result = saved.results[0]
    assert len(saved.results) == 1
    assert result.symbol == "PORTFOLIO"
    assert result.settings.fractional_shares
    assert result.metadata["account_model"] == "shared_cash_long_only_factor"
    assert saved.manifest["flow"] == "factor"
    assert saved.manifest["factor_spec"] == spec().to_dict()
    assert saved.manifest["lookback"] == 2
    assert saved.manifest["return_basis"] == "price_return_no_dividend_credit"
    assert "point-in-time" in saved.manifest["assumptions"]["universe"]
    for name in ("equity", "trades", "orders", "factor_values", "rebalances", "targets", "positions"):
        original = getattr(result, name)
        assert original.width > 0, name
        assert getattr(restored.results[0], name).equals(original), name
    for symbol, dataset in originals.items():
        assert app.pipeline.read_dataset(dataset.dataset_id).equals(original_frames[symbol])
        assert saved.manifest["sources"][symbol][0]["dataset_id"] == dataset.dataset_id
    assert saved.comparison.equals(restored.comparison)
    assert app.list_runs()[0]["strategy_names"] == ["factor_portfolio"]


def test_missing_ticker_aborts_before_factor_execution_and_publishes_nothing(app, monkeypatch):
    import research.backtesting.factor_engine as engine

    store(app, "AAA")
    monkeypatch.setattr(engine, "run_factor_backtest", lambda *args, **kwargs: pytest.fail("Executed before preflight completed."))
    with pytest.raises(PreflightError):
        run(app)
    assert app.list_runs() == []


def test_skip_problematic_tickers_persists_errors_and_selection_bias_warning(app):
    store(app, "AAA")
    store(app, "BBB", (2, 3, 4, 5, 9))
    report = app.factor_preflight(["AAA", "BBB"], spec=spec(), start=day(5), end=day(10), skip_problematic_tickers=True)
    assert not report.ok
    assert any(item["code"] == "retrospective_ticker_exclusion" for item in report.warnings)
    saved = run(app, skip_problematic_tickers=True)
    assert saved.manifest["tickers"] == ["AAA"]
    assert saved.manifest["requested_tickers"] == ["AAA", "BBB"]
    assert saved.manifest["ticker_errors"][0]["ticker"] == "BBB"
    assert saved.manifest["ticker_errors"][0]["code"] == "missing_bars"
    assert any(item["code"] == "retrospective_ticker_exclusion" for item in saved.manifest["warnings"])
    assert "selection bias" in saved.manifest["assumptions"]["selection"]


def test_later_listing_waits_for_factor_warmup(app):
    store(app, "AAA")
    store(app, "BBB", (5, 8, 9))
    saved = run(app, skip_initial_missing_bars=True)
    assert saved.manifest["effective_starts"]["BBB"] == day(9).isoformat()
    assert saved.manifest["tickers"] == ["AAA", "BBB"]
    assert any("BBB: trading starts" in note for note in saved.manifest["notes"])


def test_entire_universe_can_join_after_initial_missing_history(app):
    store(app, "BBB", (5, 8, 9))
    saved = run(app, ("BBB",), skip_initial_missing_bars=True)
    assert saved.manifest["start"] == day(5).isoformat()
    assert saved.manifest["effective_starts"] == {"BBB": day(9).isoformat()}
    assert saved.results[0].trades.height > 0
    assert saved.results[0].trades["timestamp"].min() >= day(9)
    assert all(stamp >= day(8) for stamp in saved.results[0].trades["decision_timestamp"])


def test_entire_universe_with_insufficient_warmup_does_not_publish_run(app):
    store(app, "BBB", (8, 9))
    with pytest.raises(ResearchError, match="enough local history"):
        run(app, ("BBB",), skip_initial_missing_bars=True)
    assert app.list_runs() == []


def test_valid_data_with_no_eligible_factor_values_stays_in_cash(app):
    store(app, "AAA")  # All closes exceed their opens, so no red-bar observations.
    configured = FactorBacktestSpec(
        factors=(FactorSpec("red_bar_high_percentage", lookback=1),),
        portfolio=ComponentSpec("equal_weight", config={"top_n": 1}),
        cash_reserve=0.0, max_position_weight=1.0,
    )
    result = run(app, ("AAA",), spec=configured).results[0]
    assert result.trades.is_empty()
    assert result.metrics["final_cash"] == result.metrics["initial_cash"]
    assert result.metrics["open_position_count"] == 0
    assert result.factor_values["score"].null_count() == result.factor_values.height


@pytest.mark.parametrize("schedule", ["daily", "weekly", "monthly"])
def test_calendar_schedule_uses_real_early_close_for_decisions(app, schedule):
    stamps = [datetime(2024, month, date, tzinfo=UTC) for month, date in (
        (11, 25), (11, 26), (11, 27), (11, 29), (12, 2), (12, 3), (12, 4))]
    prices = [100.0 + index for index in range(len(stamps))]
    frame = pl.DataFrame({
        "timestamp": stamps, "symbol": ["AAA"] * len(stamps),
        "open": prices, "high": [price + 1 for price in prices],
        "low": prices, "close": [price + 1 for price in prices],
        "volume": [1000.0] * len(stamps),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    app.pipeline.store.write_raw(DataRequest(
        symbol="AAA", provider="yahoo", timeframe="1d", start=stamps[0],
        end=datetime(2024, 12, 5, tzinfo=UTC)), frame)
    saved = app.factor_backtest(
        ["AAA"], spec=spec(schedule=ComponentSpec(schedule)),
        start=datetime(2024, 11, 29, tzinfo=UTC), end=datetime(2024, 12, 5, tzinfo=UTC))
    early_close = datetime(2024, 11, 29, 18, tzinfo=UTC)
    after_holiday = datetime(2024, 12, 2, 14, 30, tzinfo=UTC)
    rebalanced = saved.results[0].rebalances.filter(pl.col("decision_timestamp") == early_close)
    assert rebalanced.height == 1
    assert rebalanced["execution_timestamp"][0] == after_holiday
    assert rebalanced["status"][0] == "executed"
    assert all(decision <= execution for decision, execution in saved.results[0].trades.select(
        "decision_timestamp", "timestamp").iter_rows())


def test_explicit_execution_settings_override_fractional_default(app):
    store(app, "AAA")
    settings = ExecutionSettings(initial_cash=500, fractional_shares=False, commission_fixed=1)
    saved = run(app, ("AAA",), settings=settings)
    assert saved.results[0].settings == settings
    assert not saved.manifest["execution_settings"]["fractional_shares"]


def test_factor_comparison_allows_portfolio_policy_changes_but_not_execution_changes(app):
    for symbol in ("AAA", "BBB"):
        store(app, symbol)
    first = run(app)
    second = run(app, spec=spec(portfolio=ComponentSpec("score_weight", config={"top_n": 1})))
    assert app.compare_runs([first.run_id, second.run_id]).height == 2
    for field, value in (("sale_proceeds", "next_open"), ("funding", "score_priority")):
        changed = run(app, spec=spec(**{field: value}))
        with pytest.raises(ResearchError, match="different"):
            app.compare_runs([first.run_id, changed.run_id])


def test_snapshot_revisions_are_pinned_before_factor_execution(app, monkeypatch):
    import research.backtesting.factor_engine as engine

    source = store(app, "AAA")
    original = app.pipeline.read_dataset(source.dataset_id)
    execute = engine.run_factor_backtest
    observed = []

    def replace_source(frames, *args, **kwargs):
        app.pipeline.store.replace_raw(source.dataset_id,
            original.with_columns(*(pl.col(name) * 2 for name in ("open", "high", "low", "close"))),
            confirm=source.dataset_id)
        observed.append(frames["AAA"].clone())
        return execute(frames, *args, **kwargs)

    monkeypatch.setattr(engine, "run_factor_backtest", replace_source)
    saved = run(app, ("AAA",))
    assert observed[0]["close"].to_list() == original.filter(pl.col("timestamp") >= day(3))["close"].to_list()
    assert saved.manifest["sources"]["AAA"][0]["dataset_id"] == source.dataset_id


def audit_result():
    return BacktestResult(
        symbol="PORTFOLIO", strategy_spec={"name": "factor_portfolio", "version": "1", "config": {}},
        settings=ExecutionSettings(), metrics={}, equity=pl.DataFrame({"equity": [10000.0]}),
        trades=pl.DataFrame({"rebalance_id": ["rebalance-1"]}),
        orders=pl.DataFrame({"rebalance_id": ["rebalance-1"]}),
        factor_values=pl.DataFrame({"symbol": ["AAA"], "score": [1.0]}),
        rebalances=pl.DataFrame({"rebalance_id": ["rebalance-1"]}),
        targets=pl.DataFrame({"rebalance_id": ["rebalance-1"], "target_weight": [0.5]}),
        positions=pl.DataFrame({"rebalance_id": ["rebalance-1"], "quantity_exact": ["50"]}),
    )


@pytest.mark.parametrize("table", ["factor_values", "rebalances", "targets", "positions"])
def test_factor_audit_artifacts_are_checksummed(tmp_path, table):
    saved = save_run(tmp_path, [audit_result()], {})
    artifact = saved.path / saved.manifest["results"][0]["files"][table]["path"]
    with artifact.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ResearchError, match="checksum"):
        load_run(tmp_path, saved.run_id)


def test_legacy_results_load_without_factor_audits(tmp_path):
    original = audit_result()
    for name in ("factor_values", "rebalances", "targets", "positions"):
        setattr(original, name, pl.DataFrame())
    saved = save_run(tmp_path, [original], {})
    assert set(saved.manifest["results"][0]["files"]) == {"equity", "trades", "orders"}
    restored = load_run(tmp_path, saved.run_id)
    for name in ("factor_values", "rebalances", "targets", "positions"):
        assert getattr(restored.results[0], name).width == 0


def test_audit_path_cannot_escape_run_directory(tmp_path):
    saved = save_run(tmp_path, [audit_result()], {})
    manifest = saved.manifest
    manifest["results"][0]["files"]["targets"]["path"] = "../outside.parquet"
    path = saved.path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (saved.path / "manifest.sha256").write_text(sha256(path.read_bytes()).hexdigest(), encoding="ascii")
    with pytest.raises(ResearchError, match="escapes"):
        load_run(tmp_path, saved.run_id)
