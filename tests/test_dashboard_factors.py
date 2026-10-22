"""Factor workspace wiring, persistence, and rebalance audit presentation."""

from datetime import date

from streamlit.testing.v1 import AppTest

from data_pipeline.processing import ScalePrices
from research import Research, ResearchError
from test_dashboard_processed_backtest import _source


def _open(monkeypatch, research):
    import dashboard.app as dashboard
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Factor backtest").run()
    assert not app.exception and not app.error
    return app


def _small_factors(app):
    app.date_input(key="factor-start").set_value(date(2024, 1, 5))
    app.date_input(key="factor-end").set_value(date(2024, 1, 10))
    app.number_input(key="factor-momentum-lookback").set_value(2)
    app.number_input(key="factor-top-n").set_value(1)


def test_factor_page_empty_store_and_existing_routes(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    app = _open(monkeypatch, research)
    assert any("No local raw datasets" in info.value for info in app.info)
    assert not any(button.key == "factor-submit" for button in app.button)
    app.sidebar.radio[0].set_value("Backtest").run()
    assert any(title.value == "Backtest strategies" for title in app.title)
    assert not app.exception and not app.error


def test_policy_and_factor_settings_survive_navigation_and_restart(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    app.multiselect(key="factor-names").set_value(["momentum", "volatility", "red_bar_high_percentage"]).run()
    app.selectbox(key="factor-schedule").set_value("every_n_bars").run()
    numbers = {
        "factor-momentum-lookback": 7, "factor-momentum-weight": 2.0,
        "factor-volatility-lookback": 12, "factor-red_bar_high_percentage-lookback": 30,
        "factor-top-n": 8, "factor-schedule-bars": 3, "factor-reserve": 5.0,
        "factor-cap": 15.0, "factor-drift": 2.0, "factor-turnover": 75.0,
        "factor-initial-capital": 4321.0, "factor-commission-fixed": 1.0,
        "factor-commission-bps": 2.0, "factor-slippage-bps": 3.0,
    }
    for key, value in numbers.items():
        app.number_input(key=key).set_value(value)
    app.selectbox(key="factor-policy").set_value("score_weight")
    app.selectbox(key="factor-proceeds").set_value("next_open")
    app.selectbox(key="factor-funding").set_value("score_priority")
    app.selectbox(key="factor-momentum-direction").set_value(-1)
    app.checkbox(key="factor-skip-problematic").check()
    app.checkbox(key="factor-allow-late").check()
    app.checkbox(key="factor-fractional").uncheck().run()
    app.sidebar.radio[0].set_value("Backtest").run()
    app.sidebar.radio[0].set_value("Factor backtest").run()
    for session in (app, _open(monkeypatch, research)):
        assert not session.exception and not session.error
        for key, value in numbers.items():
            assert session.number_input(key=key).value == value
        assert session.selectbox(key="factor-policy").value == "score_weight"
        assert session.selectbox(key="factor-proceeds").value == "next_open"
        assert session.selectbox(key="factor-funding").value == "score_priority"
        assert session.selectbox(key="factor-momentum-direction").value == -1
        assert session.checkbox(key="factor-skip-problematic").value
        assert session.checkbox(key="factor-allow-late").value
        assert not session.checkbox(key="factor-fractional").value


def test_factor_controls_are_scoped_to_data_and_runs_store(tmp_path, monkeypatch):
    first = Research(tmp_path / "data", tmp_path / "runs-one")
    _source(first)
    app = _open(monkeypatch, first)
    app.number_input(key="factor-top-n").set_value(7).run()
    second = Research(tmp_path / "data", tmp_path / "runs-two")
    other = _open(monkeypatch, second)
    assert other.number_input(key="factor-top-n").value == 20
    restored = _open(monkeypatch, first)
    assert restored.number_input(key="factor-top-n").value == 7


def test_factor_dashboard_routes_processed_source_and_execution_options(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    raw = _source(research)
    processed = research.pipeline.process(raw.dataset_id, processors=[ScalePrices(2)])
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        raise ResearchError("intentional test stop")

    monkeypatch.setattr(research, "factor_backtest", record)
    app = _open(monkeypatch, research)
    app.radio(key="factor-layer").set_value("processed").run()
    _small_factors(app)
    app.number_input(key="factor-reserve").set_value(5.0)
    app.number_input(key="factor-cap").set_value(20.0)
    app.selectbox(key="factor-policy").set_value("inverse_volatility")
    app.selectbox(key="factor-proceeds").set_value("next_open")
    app.selectbox(key="factor-funding").set_value("score_priority")
    app.checkbox(key="factor-allow-late").check()
    app.checkbox(key="factor-skip-problematic").check()
    app.button(key="factor-submit").click().run()
    assert not app.exception
    assert len(calls) == 1
    request = calls[0]
    assert request["tickers"] == ["AAPL"]
    assert request["layer"] == "processed" and request["pipeline_id"] == processed.pipeline_id
    assert request["calendar"] == "XNYS" and request["timeframe"] == "1d"
    assert request["skip_initial_missing_bars"] and request["skip_problematic_tickers"]
    assert request["settings"].fractional_shares
    assert request["settings"].minimum_trade_budget == 5
    assert request["spec"].cash_reserve == .05
    assert request["spec"].max_position_weight == .2
    assert request["spec"].portfolio.name == "inverse_volatility"
    assert request["spec"].sale_proceeds == "next_open"
    assert request["spec"].funding == "score_priority"
    assert research.list_runs() == []


def test_factor_run_audit_and_saved_run_restore(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    _source(research, "MSFT", offset=50)
    app = _open(monkeypatch, research)
    _small_factors(app)
    app.button(key="factor-submit").click().run(timeout=30)
    assert not app.exception and not app.error
    run = app.session_state["factor-run"]
    assert run.manifest["flow"] == "factor"
    assert run.results[0].rebalances.height > 0
    assert any(field.label == "Inspect rebalance" for field in app.selectbox)
    assert len(app.get("plotly_chart")) == 2
    app.sidebar.radio[0].set_value("Saved runs").run()
    assert not app.exception and not app.error
    assert any(title.value.startswith("Factor run ") for title in app.subheader)
    assert any(field.label == "Inspect rebalance" for field in app.selectbox)
    restored = _open(monkeypatch, research)
    assert restored.session_state["factor-run"].run_id == run.run_id
    assert restored.date_input(key="factor-start").value == date(2024, 1, 5)
    assert any("Latest completed run" in item.value for item in restored.caption)
    # Editing controls keeps the prior result explicitly identified as a completed run.
    restored.number_input(key="factor-top-n").set_value(2).run()
    assert restored.session_state["factor-run"].run_id == run.run_id


def test_factor_preflight_errors_and_skipped_ticker_report(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    _small_factors(app)
    app.text_input(key="factor-extra").set_value("ABSENT")
    app.button(key="factor-submit").click().run(timeout=30)
    assert not app.exception
    assert any("required data did not pass preflight" in error.value for error in app.error)
    assert research.list_runs() == []
    app.checkbox(key="factor-skip-problematic").check()
    app.button(key="factor-submit").click().run(timeout=30)
    assert not app.exception and not app.error
    assert any("Skipped ticker errors" in warning.value for warning in app.warning)
    assert app.session_state["factor-run"].manifest["ticker_errors"][0]["ticker"] == "ABSENT"


def test_no_factors_cannot_submit(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    app = _open(monkeypatch, research)
    app.multiselect(key="factor-names").set_value([]).run()
    assert not app.exception and not app.error
    assert app.button(key="factor-submit").disabled
    assert research.list_runs() == []


def test_inverse_volatility_changes_sizing_without_adding_a_scoring_factor(tmp_path, monkeypatch):
    research = Research(tmp_path / "data", tmp_path / "runs")
    _source(research)
    _source(research, "MSFT", offset=50)
    app = _open(monkeypatch, research)
    _small_factors(app)
    app.number_input(key="factor-top-n").set_value(2)
    app.button(key="factor-submit").click().run(timeout=30)
    assert not app.exception and not app.error
    before = app.session_state["factor-run"]
    app.selectbox(key="factor-policy").set_value("inverse_volatility").run()
    app.number_input(key="factor-sizing-volatility-lookback").set_value(2)
    app.button(key="factor-submit").click().run(timeout=30)
    assert not app.exception and not app.error
    after = app.session_state["factor-run"]
    assert [factor["name"] for factor in after.manifest["factor_spec"]["factors"]] == ["momentum"]
    assert after.manifest["factor_spec"]["portfolio"]["config"]["volatility_lookback"] == 2
    columns = ["timestamp", "symbol", "score"]
    assert before.results[0].factor_values.select(columns).equals(after.results[0].factor_values.select(columns))
    assert not before.results[0].targets["target_weight"].equals(after.results[0].targets["target_weight"])
