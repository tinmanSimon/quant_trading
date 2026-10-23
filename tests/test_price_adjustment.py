"""Economic invariants and reproducibility of split/cash-dividend adjustment."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from zoneinfo import ZoneInfo

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataRequest
from data_pipeline.corporate_actions.models import ActionCoverage, ActionSnapshot, CorporateAction
from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import DataContract, Pipeline, discover_processors, load_pipeline
from data_pipeline.processing._config import canonical_json
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.price_adjustment import AdjustCorporateActions, PriceAdjustmentWarning
from data_pipeline.schemas import OHLCV_SCHEMA
from research.errors import ResearchError
from research.processed import validate_processed_source


NY = ZoneInfo("America/New_York")


def boundary(day):
    return datetime.combine(date.fromisoformat(day), datetime.min.time(), NY).astimezone(UTC)


def event(kind="forward_split", day="2020-08-31", symbol="TSLA", **kwargs):
    return CorporateAction(f"{symbol}-{kind}-{day}", symbol, kind, boundary(day),
                           security_id=f"security-{symbol}", **kwargs)


def snapshot(events=(), symbols=("TSLA",), **kwargs):
    return ActionSnapshot("verified-import", date(2020, 9, 30), datetime(2020, 10, 1, tzinfo=UTC),
        tuple(ActionCoverage(symbol, date(2020, 8, 1), date(2020, 9, 30),
                             source_basis="unadjusted", security_id=f"security-{symbol}") for symbol in symbols),
        tuple(events), **kwargs)


def frame(stamps=None, symbol="TSLA"):
    if stamps is None:
        stamps = [boundary("2020-08-28") + timedelta(hours=10),
                  boundary("2020-08-31") + timedelta(hours=4),
                  boundary("2020-09-02") + timedelta(hours=10)]
    return pl.DataFrame([{"timestamp": stamp, "symbol": symbol, "open": 100., "high": 110.,
                         "low": 90., "close": 105., "volume": 20.} for stamp in stamps], schema=OHLCV_SCHEMA)


def apply(events, source=None):
    return AdjustCorporateActions(snapshot(events)).transform(frame() if source is None else source, DataContract("1m"))


@pytest.mark.parametrize("kind,ratio", [("forward_split", 5.), ("reverse_split", .1), ("stock_dividend", 1.1)])
def test_split_changes_all_prices_and_inverse_share_volume(kind, ratio):
    source = frame()
    original = source.clone()
    result = apply([event(kind, share_ratio=ratio)], source)
    for column in ("open", "high", "low", "close"):
        assert result.frame[column].to_list() == pytest.approx([source[column][0] / ratio, *source[column][1:]])
    assert result.frame["volume"].to_list() == pytest.approx([20. * ratio, 20., 20.])
    assert result.frame["close"][0] * result.frame["volume"][0] == pytest.approx(105. * 20.)
    assert_frame_equal(source, original)


@pytest.mark.parametrize("kind", ["cash_dividend", "special_cash_dividend"])
def test_regular_and_special_cash_dividends_lower_prior_prices_only(kind):
    result = apply([event(kind, cash_amount=2., reference_close=100.)]).frame
    assert result["open"].to_list() == pytest.approx([98., 100., 100.])
    assert result["high"].to_list() == pytest.approx([107.8, 110., 110.])
    assert result["low"].to_list() == pytest.approx([88.2, 90., 90.])
    assert result["close"].to_list() == pytest.approx([102.9, 105., 105.])
    assert result["volume"].to_list() == [20., 20., 20.]


def test_multiple_events_after_a_batch_end_still_affect_that_batch():
    events = [event(share_ratio=5.), event("cash_dividend", "2020-09-01", price_factor=.98),
              event("reverse_split", "2020-09-15", share_ratio=.1)]
    processor = AdjustCorporateActions(snapshot(events))
    source = frame()
    together = processor.transform(source, DataContract("1m")).frame
    separately = pl.concat([processor.transform(source.slice(i, 1), DataContract("1m")).frame
                            for i in range(source.height)])
    assert_frame_equal(together, separately)
    assert together["open"].to_list() == pytest.approx([196., 980., 1000.])
    assert together["volume"].to_list() == pytest.approx([10., 2., 2.])


@pytest.mark.parametrize("day", ["2020-03-09", "2020-11-02"])
def test_exchange_local_boundaries_include_premarket_and_dst(day):
    stamp = boundary(day)
    source = frame([stamp - timedelta(minutes=1), stamp, stamp + timedelta(hours=4)])
    coverage = ActionCoverage("TSLA", date(2020, 1, 1), date(2020, 12, 31),
                              source_basis="unadjusted", security_id="security-TSLA")
    evidence = replace(snapshot([event(day=day, share_ratio=2.)]), reference_date=date(2020, 12, 31), coverage=(coverage,))
    result = AdjustCorporateActions(evidence).transform(source, DataContract("1m")).frame
    assert result["open"].to_list() == [50., 100., 100.]
    assert result["volume"].to_list() == [40., 20., 20.]


def test_same_day_split_and_dividend_require_explicit_factor():
    split = event(share_ratio=5.)
    dividend = event("cash_dividend", cash_amount=2., reference_close=100.)
    with pytest.raises(ProcessingError, match="shares a date with a split"):
        apply([split, dividend])
    result = apply([split, replace(dividend, price_factor=.98)]).frame
    assert result["open"].to_list() == pytest.approx([19.6, 100., 100.])
    assert result["volume"].to_list() == [100., 20., 20.]


def test_two_cash_distributions_need_verified_sequential_factors():
    ordinary = event("cash_dividend", cash_amount=2., reference_close=100.)
    special = event("special_cash_dividend", cash_amount=5., reference_close=100.)
    with pytest.raises(ProcessingError, match="Multiple cash dividends"):
        apply([ordinary, special])
    result = apply([replace(ordinary, price_factor=.98), replace(special, price_factor=93 / 98)]).frame
    assert result["open"].to_list() == pytest.approx([93., 100., 100.])


@pytest.mark.parametrize("changes,match", [
    ({"share_ratio": None}, "share_ratio"), ({"share_ratio": .5}, "greater than one"),
    ({"price_factor": .9}, "conflicts"),
])
def test_invalid_split_terms_fail(changes, match):
    with pytest.raises(ProcessingError, match=match):
        apply([replace(event(share_ratio=5.), **changes)])


@pytest.mark.parametrize("kwargs,match", [
    ({"cash_amount": 2.}, "reference_close"),
    ({"cash_amount": 100., "reference_close": 100.}, "between zero and one"),
    ({"cash_amount": 101., "reference_close": 100.}, "between zero and one"),
    ({"price_factor": 1.1}, "between zero and one"),
    ({"price_factor": .98, "currency": "EUR"}, "USD"),
])
def test_invalid_dividend_terms_fail(kwargs, match):
    with pytest.raises(ProcessingError, match=match):
        apply([event("cash_dividend", **kwargs)])


def test_report_only_events_warn_and_are_not_adjusted():
    unsupported = event("spin_off")
    with pytest.warns(PriceAdjustmentWarning, match="other corporate actions reported but not adjusted"):
        result = apply([unsupported])
    assert_frame_equal(result.frame, frame())
    assert json.loads(result.contract.adjustment_json)["unsupported_events"] == "reported_not_adjusted"


def test_cancelled_and_out_of_range_events_do_not_apply():
    events = [event(share_ratio=5., status="cancelled"),
              event("cash_dividend", "2020-10-01", price_factor=.95),
              event("forward_split", "2020-08-03", share_ratio=2.)]
    assert_frame_equal(apply(events).frame, frame())


@pytest.mark.parametrize("changes,match", [
    ({"status": "pending"}, "unresolved"),
    ({"security_id": "another-security"}, "security_id does not match"),
])
def test_unresolved_or_wrong_identity_event_fails(changes, match):
    with pytest.raises(ProcessingError, match=match):
        apply([replace(event(share_ratio=5.), **changes)])


def test_duplicate_terms_with_different_event_ids_fail():
    split = event(share_ratio=5.)
    with pytest.raises(ProcessingError, match="duplicate supported event terms"):
        apply([split, replace(split, event_id="duplicated-vendor-record")])


@pytest.mark.parametrize("changes,match", [
    ({"status": "partial"}, "complete event coverage"),
    ({"start": date(2020, 8, 29)}, "complete event coverage"),
    ({"end": date(2020, 9, 20)}, "complete event coverage"),
    ({"source_basis": "unknown"}, "source price and volume basis"),
    ({"source_basis": "split_adjusted"}, "source price and volume basis"),
    ({"security_id": ""}, "security_id mapping"),
])
def test_coverage_and_source_basis_fail_closed(changes, match):
    evidence = snapshot()
    evidence = replace(evidence, coverage=(replace(evidence.coverage[0], **changes),))
    with pytest.raises(ProcessingError, match=match):
        AdjustCorporateActions(evidence).transform(frame(), DataContract("1m"))


def test_source_contract_can_supply_known_unadjusted_basis():
    evidence = snapshot()
    evidence = replace(evidence, coverage=(replace(evidence.coverage[0], source_basis="unknown"),))
    assert_frame_equal(AdjustCorporateActions(evidence).transform(frame(), DataContract("1m", source_basis="unadjusted")).frame, frame())


def test_wrong_timeframe_double_adjustment_and_reference_date_rejected():
    processor = AdjustCorporateActions(snapshot())
    for contract, match in [(DataContract("1h"), "1m input"),
                            (DataContract("1m", source_basis="vendor_adjusted"), "Vendor-adjusted"),
                            (processor.output_contract(DataContract("1m")), "double adjustment")]:
        with pytest.raises(ProcessingError, match=match):
            processor.transform(frame(), contract)
    with pytest.raises(ProcessingError, match="after the adjustment reference date"):
        processor.transform(frame([boundary("2020-10-01")]), DataContract("1m"))
    with pytest.raises(ProcessingError, match="whole-minute grid"):
        processor.transform(frame([boundary("2020-08-28") + timedelta(seconds=1)]), DataContract("1m"))


def test_multisymbol_factors_and_failures_remain_isolated():
    split = event(share_ratio=5.)
    evidence = snapshot([split], symbols=("TSLA", "AAPL"))
    source = pl.concat([frame(symbol="AAPL"), frame()])
    processor = AdjustCorporateActions(evidence)
    result = processor.transform(source, DataContract("1m")).frame
    assert_frame_equal(result.filter(pl.col("symbol") == "AAPL"), frame(symbol="AAPL"))
    assert result.filter(pl.col("symbol") == "TSLA")["open"][0] == 20.
    broken = replace(evidence, events=(replace(split, status="pending"),))
    assert_frame_equal(AdjustCorporateActions(broken).transform(frame(symbol="AAPL"), DataContract("1m")).frame, frame(symbol="AAPL"))
    with pytest.raises(ProcessingError, match="TSLA.*unresolved"):
        AdjustCorporateActions(broken).transform(frame(), DataContract("1m"))


@pytest.mark.parametrize("column", ["timestamp", "open", "high", "low", "close", "volume"])
def test_output_semantics_reject_corruption(column):
    class Broken(AdjustCorporateActions):
        def _transform(self, frame, input_contract, output_contract):
            output = super()._transform(frame, input_contract, output_contract)
            delta = timedelta(seconds=1) if column == "timestamp" else (-.01 if column == "low" else .01)
            return output.with_columns((pl.col(column) + delta).alias(column))

    with pytest.raises(ProcessingError, match="bar keys" if column == "timestamp" else f"Adjusted {column}"):
        Broken(snapshot([event(share_ratio=5.)])).transform(frame(), DataContract("1m"))


def test_discovery_snapshot_recipe_roundtrip_and_revision_identity():
    processor = AdjustCorporateActions(snapshot([event(share_ratio=5.)]))
    registry = discover_processors()
    assert not registry.issues
    definition = registry.definition(processor.name, processor.version)
    assert not definition.causal and definition.price_basis == "transformed"
    assert definition.schema["properties"]["snapshot"]["format"] == "corporate_action_snapshot"
    configured = Pipeline([processor])
    replay = load_pipeline(configured.canonical_json)
    assert replay.fingerprint == configured.fingerprint
    assert_frame_equal(replay.run(frame(), DataContract("1m")).frame,
                       configured.run(frame(), DataContract("1m")).frame)
    changed = replace(processor, snapshot=replace(processor.snapshot, events=(event(share_ratio=4.),)))
    assert Pipeline([changed]).fingerprint != configured.fingerprint
    assert json.loads(processor.output_contract(DataContract("1m")).adjustment_json)["snapshot_checksum"] == processor.snapshot.checksum


def test_legacy_snapshot_config_preserves_saved_pipeline_fingerprint():
    evidence = snapshot([event(share_ratio=5.)]).to_dict()
    assert "other_events_status" not in evidence["coverage"][0]
    legacy_config = {"snapshot": evidence}
    legacy_spec = [{"name": "adjust_corporate_actions", "version": "1", "config": legacy_config}]
    saved_json = canonical_json(legacy_spec)
    saved_fingerprint = hashlib.sha256(saved_json.encode("utf-8")).hexdigest()
    registry = discover_processors()
    assert registry.configure("adjust_corporate_actions", "1", legacy_config) == legacy_config
    replay = registry.load(saved_json)
    assert replay.canonical_json == saved_json
    assert replay.fingerprint == saved_fingerprint
    assert_frame_equal(replay.run(frame(), DataContract("1m")).frame,
                       AdjustCorporateActions(ActionSnapshot.from_dict(evidence)).transform(frame(), DataContract("1m")).frame)


def test_explicit_other_event_coverage_changes_pipeline_fingerprint():
    legacy = snapshot([event(share_ratio=5.)])
    scoped = replace(legacy, coverage=(replace(legacy.coverage[0], other_events_status="unknown"),))
    configured = Pipeline([AdjustCorporateActions(scoped)])
    assert configured.identities[0]["config"]["snapshot"]["coverage"][0]["other_events_status"] == "unknown"
    assert configured.fingerprint != Pipeline([AdjustCorporateActions(legacy)]).fingerprint
    assert load_pipeline(configured.canonical_json).fingerprint == configured.fingerprint


def test_saved_output_preserves_lineage_and_backtest_gate(tmp_path):
    pipeline = DataPipeline(tmp_path)
    source = frame()
    raw = pipeline.ingest_frame(DataRequest("TSLA", boundary("2020-08-28"), boundary("2020-09-03"),
                                           timeframe="1m", provider="verified-import"), source).raw
    processor = AdjustCorporateActions(snapshot([event(share_ratio=5.)]))
    configured = Pipeline([processor])
    output = pipeline.process([raw.dataset_id], processors=configured)
    assert output.contract.adjustment_json == processor.output_contract(raw.contract).adjustment_json
    assert_frame_equal(pipeline.read_dataset(raw.dataset_id), source)
    with pytest.raises(ResearchError, match="no declared causality guarantee"):
        validate_processed_source(pipeline, output, pipeline_id=configured.fingerprint,
                                  registry=discover_processors(), bars=(), parent_cache={})


def test_daily_resampling_preserves_adjustment_basis():
    resampler = ResampleRegularHoursDaily.for_dates(date(2020, 8, 28), date(2020, 8, 28))
    session, = resampler.sessions
    source = frame([session.open + timedelta(minutes=i) for i in range(390)])
    adjuster = AdjustCorporateActions(snapshot([event(share_ratio=5.)]))
    result = Pipeline([adjuster, resampler]).run(source, DataContract("1m"))
    assert result.contract.timeframe == "1d"
    assert result.contract.adjustment_json == adjuster.output_contract(DataContract("1m")).adjustment_json
    assert result.frame["open"][0] == 20.
    assert result.frame["volume"][0] == 39000.
