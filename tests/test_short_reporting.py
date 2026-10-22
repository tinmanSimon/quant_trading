"""Optional shorts retain direction and account semantics in saved reports and UI."""

from datetime import UTC, datetime

import pytest

from research.backtesting import ExecutionSettings
from test_dashboard_processed_backtest import _open
from test_initial_split import EveryBar, research


class OptionalShorts(EveryBar):
    name = "test_optional_shorts_reporting"

    def __init__(self, enable_shorts=False):
        self.enable_shorts = enable_shorts
        self.config = {"enable_shorts": enable_shorts}

    def entry_signal(self, history):
        return False

    def short_entry_signal(self, history):
        return self.enable_shorts


@pytest.mark.parametrize("mode", ["per_entry", "fixed_tenth", "initial_split"])
def test_saved_short_reports_preserve_direction_and_account_model(research, mode):
    run = research.backtest(
        ["A", "B"], strategies=[OptionalShorts(True)], allocation_mode=mode,
        start=datetime(2024, 1, 3, tzinfo=UTC), end=datetime(2024, 1, 6, tzinfo=UTC),
        settings=ExecutionSettings(initial_cash=1000, fractional_shares=True),
    )
    restored = research.load_run(run.run_id)
    for result in restored.results:
        assert "long_short" in result.metadata["account_model"]
        assert set(result.trades["direction"]) == {"short"}
        for name in ("completed_trades", "open_positions"):
            table = getattr(result, name)
            assert "direction" in table.columns
            assert set(table["direction"]) <= {"short"}
    if mode == "initial_split":
        summary = restored.manifest["allocation_summary"][0]
        assert summary["losses"] == sum(result.metrics["losses"] for result in restored.results)
        assert summary["total_pnl"] == pytest.approx(sum(result.metrics["total_pnl"] for result in restored.results))
    else:
        assert restored.manifest["account_mode"] == "shared_long_short_usd_portfolios"


def test_short_setting_persists_and_shared_report_explains_collateral(research, monkeypatch):
    research.strategy_registry.register(
        OptionalShorts.name, "1", lambda config: OptionalShorts(**config),
        default_config={"enable_shorts": False},
        parameters={"enable_shorts": {"type": "boolean", "label": "Enable shorts"}},
    )
    app = _open(monkeypatch, research)
    app.multiselect(key="backtest-strategies").set_value([(OptionalShorts.name, "1")]).run()
    option = next(field for field in app.checkbox if field.label == "Enable shorts")
    assert not option.value
    option.check().run()
    app = _open(monkeypatch, research)
    assert next(field for field in app.checkbox if field.label == "Enable shorts").value
    app.multiselect(key="backtest-tickers").set_value(["A", "B"])
    app.date_input(key="backtest-start").set_value(datetime(2024, 1, 3).date())
    app.date_input(key="backtest-end").set_value(datetime(2024, 1, 6).date())
    next(button for button in app.button if button.label == "Validate all data and run").click().run()
    assert not app.exception and not app.error
    assert any("one shared cash account" in caption.value for caption in app.caption)
    assert any("sale proceeds stay locked until buyback" in caption.value for caption in app.caption)
    assert any("closed at the modeled stop price" in caption.value for caption in app.caption)
    assert any("direction" in table.value.columns for table in app.dataframe)
