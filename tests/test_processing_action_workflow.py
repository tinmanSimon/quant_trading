"""Adjustment and event reporting across the entire requested ticker batch."""

from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.corporate_actions.models import ActionCoverage, ActionSnapshot, CorporateAction
from data_pipeline.exceptions import ProcessingError, RequestDataMismatchError
from data_pipeline.processing import Pipeline
from data_pipeline.processing.processors.daily import ResampleRegularHoursDaily
from data_pipeline.processing.processors.price_adjustment.processor import AdjustCorporateActions
from data_pipeline.schemas import OHLCV_SCHEMA


def midnight(day):
    return datetime.combine(day, datetime.min.time(), UTC)


@pytest.fixture
def workflow(tmp_path):
    app = DataPipeline(tmp_path / "data")
    days = [date(2024, 1, day) for day in (2, 3, 4)]
    daily = ResampleRegularHoursDaily.for_dates(days[0], days[-1])
    snapshot = ActionSnapshot("verified test import", days[-1], midnight(date(2024, 1, 5)),
        tuple(ActionCoverage(symbol, days[0], days[-1], source_basis="unadjusted", security_id=symbol)
              for symbol in ("AAA", "BBB", "CCC")), (
            CorporateAction("split", "AAA", "forward_split", datetime(2024, 1, 3, tzinfo=ZoneInfo("America/New_York")),
                            security_id="AAA", share_ratio=2),
            CorporateAction("cash", "AAA", "cash_dividend", datetime(2024, 1, 4, tzinfo=ZoneInfo("America/New_York")),
                            security_id="AAA", cash_amount=1, reference_close=50),
            CorporateAction("other", "AAA", "spin_off", datetime(2024, 1, 3, tzinfo=ZoneInfo("America/New_York")),
                            security_id="AAA", details="One child share per five parent shares"),
            CorporateAction("pending", "BBB", "cash_dividend", datetime(2024, 1, 4, tzinfo=ZoneInfo("America/New_York")),
                            security_id="BBB", status="pending"),
        ))
    ids, sources = {}, {}
    for symbol in ("AAA", "BBB", "CCC"):
        rows = []
        for session, price in zip(daily.sessions, (100., 50., 49.), strict=True):
            for minute in range(390):
                rows.append({"timestamp": session.open + timedelta(minutes=minute), "symbol": symbol,
                             "open": price, "high": price, "low": price, "close": price, "volume": 10.})
        source = pl.DataFrame(rows, schema=OHLCV_SCHEMA)
        raw = app.ingest_frame(DataRequest(symbol, midnight(days[0]), midnight(days[-1] + timedelta(days=1)),
                              timeframe="1m", provider="databento-xnas.itch"), source).raw
        ids[symbol] = raw.dataset_id
        sources[symbol] = source
    return app, ids, sources, snapshot, daily


def test_batch_report_includes_failed_empty_and_uncovered_tickers(workflow):
    app, ids, sources, snapshot, daily = workflow
    recipe = Pipeline([AdjustCorporateActions(snapshot), daily])
    inputs = {"AAA": [ids["AAA"]], "BBB": [ids["BBB"]], "CCC": [], "DDD": []}
    preview = app.process_many(inputs, processors=recipe, preview=True)
    assert not app.list_datasets(DataQuery(layer="processed"))
    assert {row["ticker"] for row in preview["other_events"]} == set(inputs)
    assert [row["status"] for row in preview["results"]] == ["Preview", "Failed", "Failed", "Failed"]
    actual = preview["results"][0]["after"]
    assert actual["close"].to_list() == pytest.approx([49.] * 3)
    assert actual["volume"].to_list() == [7800., 3900., 3900.]
    assert any(row["ticker"] == "DDD" and row["coverage"] == "unknown" for row in preview["other_events"])
    saved = app.process_many(inputs, processors=recipe)
    assert saved["other_events"] == preview["other_events"]
    stored_report = json.loads(Path(saved["report_path"]).read_text())
    assert stored_report["requested_tickers"] == list(inputs)
    assert stored_report["snapshot_checksums"] == [snapshot.checksum]
    output, = app.list_datasets(DataQuery(layer="processed"))
    assert_frame_equal(app.read_dataset(output.dataset_id), actual)
    metadata = json.loads(output.contract.adjustment_json)
    assert metadata["snapshot_checksum"] == snapshot.checksum
    for symbol, raw_id in ids.items():
        assert_frame_equal(app.read_dataset(raw_id), sources[symbol])
    repeated = app.process_many(inputs, processors=recipe)
    assert repeated["results"][0]["status"] == "Already processed"
    assert app.process_many({"AAA": ids["AAA"]}, processors=recipe)["results"][0]["status"] == "Already processed"


def test_cli_multi_ticker_report(workflow, tmp_path, capsys):
    from data_pipeline.cli import main
    app, ids, _, snapshot, daily = workflow
    path = tmp_path / "recipe.json"
    path.write_text(Pipeline([AdjustCorporateActions(snapshot), daily]).canonical_json)
    assert main(["--data-dir", str(app.store.data_dir), "process-many", ids["AAA"], ids["BBB"],
                 "--tickers", "ccc", "--processors", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert set(report["requested_tickers"]) == {"AAA", "BBB", "CCC"}
    assert len(report["other_events"]) == 3
    assert main(["--data-dir", str(app.store.data_dir), "process-many", ids["AAA"],
                 "--processors", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "Already processed"


def test_yahoo_inputs_cannot_override_vendor_adjustment_with_snapshot(workflow):
    app, _, sources, snapshot, _ = workflow
    source = sources["AAA"]
    raw = app.ingest_frame(DataRequest("AAA", datetime(2024, 1, 2, tzinfo=UTC),
                          datetime(2024, 1, 5, tzinfo=UTC), timeframe="1m"), source).raw
    with pytest.raises(ProcessingError, match="Yahoo raw prices"):
        app.process(raw.dataset_id, processors=[AdjustCorporateActions(snapshot)])


@pytest.mark.parametrize("wrapped", [False, True])
def test_dashboard_reports_all_tickers_and_restores_report(workflow, monkeypatch, tmp_path, wrapped):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as dashboard
    from research import Research
    app, ids, _, snapshot, _ = workflow
    research = Research(app.store.data_dir, tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    script = "from dashboard.app import main\nmain()"
    ui = AppTest.from_string(script, default_timeout=25).run()
    ui.sidebar.radio[0].set_value("Process data").run()
    ui.multiselect(key="process-tickers").set_value(["AAA", "BBB", "CCC"]).run()
    next(field for field in ui.multiselect if field.label == "Raw batches").set_value([ids["AAA"], ids["BBB"]]).run()
    ui.selectbox(key="process-add").set_value(("adjust_corporate_actions", "1")).run()
    next(button for button in ui.button if button.label == "Add processor").click().run()
    payload = {"checksum": snapshot.checksum, "snapshot": snapshot.to_dict()} if wrapped else snapshot.to_dict()
    if wrapped:
        invalid = dict(payload, checksum="corrupt")
        next(field for field in ui.text_area if field.label == "Corporate-action snapshot (JSON)").set_value(
            json.dumps(invalid)).run()
        assert any("checksum mismatch" in item.value for item in ui.error)
        assert next(button for button in ui.button if button.label == "Preview processing").disabled
    next(field for field in ui.text_area if field.label == "Corporate-action snapshot (JSON)").set_value(
        json.dumps(payload)).run()
    assert not ui.exception and not ui.error
    next(button for button in ui.button if button.label == "Preview processing").click().run()
    assert not ui.exception
    tables = [table.value for table in ui.dataframe if "ticker" in table.value.columns]
    assert set(tables[0]["ticker"]) == {"AAA", "BBB", "CCC"}
    assert any("pending" in str(item.value).lower() for item in ui.error)
    next(button for button in ui.button if button.label == "Process and save").click().run()
    assert not ui.exception
    saved, = research.pipeline.list_datasets(DataQuery(layer="processed"))
    assert saved.request.timeframe == "1m"
    restored = AppTest.from_string(script, default_timeout=25).run()
    restored.sidebar.radio[0].set_value("Process data").run()
    assert not restored.exception
    tables = [table.value for table in restored.dataframe if "ticker" in table.value.columns]
    assert set(tables[0]["ticker"]) == {"AAA", "BBB", "CCC"}


def test_dashboard_yahoo_fetch_uses_selected_bars_and_preserves_unknown_coverage(workflow, monkeypatch, tmp_path):
    from dataclasses import replace
    from streamlit.testing.v1 import AppTest
    import dashboard.app as dashboard
    from research import Research
    from data_pipeline.corporate_actions.providers import YahooActionProvider, DatabentoActionProvider

    app, ids, sources, original, _ = workflow
    snapshot = replace(original, source="yahoo-finance", coverage=tuple(
        replace(item, other_events_status="unknown") for item in original.coverage))
    calls = []

    def fetch(self, tickers, start, reference_date, *, source_basis, reference_bars, coverage_starts):
        calls.append((tickers, start, reference_date, source_basis))
        assert coverage_starts == {"AAA": date(2024, 1, 2), "BBB": date(2024, 1, 2)}
        assert_frame_equal(reference_bars("AAA"), sources["AAA"])
        assert_frame_equal(reference_bars("BBB"), sources["BBB"])
        with pytest.raises(RequestDataMismatchError, match="input|IDs|empty|least"):
            reference_bars("CCC")
        return snapshot

    monkeypatch.setattr(YahooActionProvider, "fetch", fetch)
    monkeypatch.setattr(DatabentoActionProvider, "fetch", lambda *a, **kw: pytest.fail("Unexpected paid provider request"))
    research = Research(app.store.data_dir, tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    ui = AppTest.from_string("from dashboard.app import main\nmain()", default_timeout=25).run()
    ui.sidebar.radio[0].set_value("Process data").run()
    ui.multiselect(key="process-tickers").set_value(["AAA", "BBB", "CCC"]).run()
    next(field for field in ui.multiselect if field.label == "Raw batches").set_value([ids["AAA"], ids["BBB"]]).run()
    ui.selectbox(key="process-add").set_value(("adjust_corporate_actions", "1")).run()
    next(button for button in ui.button if button.label == "Add processor").click().run()
    choice = next(field for field in ui.selectbox if field.label == "Corporate-action provider")
    assert choice.value == "Yahoo Finance (free)"
    assert calls == []
    next(button for button in ui.button if button.label == "Fetch corporate actions from Yahoo").click().run()
    assert not ui.exception and not ui.error
    assert calls == [(["AAA", "BBB", "CCC"], date(2024, 1, 2), date(2024, 1, 4), "unadjusted")]
    next(button for button in ui.button if button.label == "Preview processing").click().run()
    assert not ui.exception
    table = next(table.value for table in ui.dataframe if "ticker" in table.value.columns)
    assert set(table["ticker"]) == {"AAA", "BBB", "CCC"}
    assert set(table["coverage"]) == {"unknown"}
    assert "no_other_events" not in set(table["event_type"])
    assert len(calls) == 1  # Preview replays the snapshot without network calls.
    next(field for field in ui.selectbox if field.label == "Corporate-action provider").set_value(
        "Databento Reference (subscription)").run()
    text = next(field.value for field in ui.text_area if field.label == "Corporate-action snapshot (JSON)")
    assert json.loads(text)["source"] == "yahoo-finance"
    assert any(button.label == "Fetch corporate actions from Databento" for button in ui.button)
    assert len(calls) == 1
