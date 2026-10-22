"""Fractional orders preserve budgets, exact quantities, and saved ledgers."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
import json

import polars as pl
import pytest

from research.backtesting import ExecutionSettings, run_backtest
from research.backtesting.execution import SHARE_QUANTUM, fee, rebalance
from research.runs import load_run, save_run
from research.strategies import Strategy


class BuyThenSell(Strategy):
    name = "fractional_test"
    lookback = 1

    @property
    def config(self):
        return {}

    def target_weight(self, history):
        return 1.0 if history.height == 1 else 0.0


def simulate(settings):
    base = datetime(2025, 1, 2, tzinfo=UTC)
    prices = [1000.0, 1000.0, 1200.0]
    frame = pl.DataFrame({
        "timestamp": [base + timedelta(days=i) for i in range(3)],
        "symbol": ["AAPL"] * 3,
        **{name: prices for name in ("open", "high", "low", "close")},
        "volume": [100.0] * 3,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    return run_backtest(frame, BuyThenSell(), start=base + timedelta(days=1),
                        end=base + timedelta(days=3), settings=settings)


def test_five_dollar_budget_can_buy_a_fraction_of_an_expensive_stock():
    settings = ExecutionSettings(fractional_shares=True)
    fill = rebalance(Decimal("5"), 0, Decimal("1000"), 1.0, settings)
    assert fill.quantity == Decimal("0.00500000")
    assert fill.cash == 0
    assert fill.holdings == fill.quantity
    assert fill.quantity.as_tuple().exponent == -8


def test_budget_below_five_is_rejected_without_charging_fees():
    settings = ExecutionSettings(fractional_shares=True, commission_fixed=1)
    fill = rebalance(Decimal("4.99"), 0, Decimal("1000"), 1.0, settings)
    assert fill.reason == "below_minimum_trade_budget"
    assert fill.quantity == fill.fees == 0
    assert fill.cash == Decimal("4.99")


def test_minimum_applies_to_purchase_allocation_not_total_cash():
    settings = ExecutionSettings(fractional_shares=True)
    fill = rebalance(Decimal("100"), 0, Decimal("1000"), 0.04, settings)
    assert fill.reason == "below_minimum_trade_budget"
    assert fill.cash == 100


def test_fractional_fees_and_slippage_stay_inside_allocation():
    settings = ExecutionSettings(fractional_shares=True, commission_fixed=Decimal("0.37"),
                                 commission_bps=7, slippage_bps=13)
    fill = rebalance(Decimal("100"), 0, Decimal("1234.56"), 0.5, settings)
    spent = Decimal("100") - fill.cash
    assert Decimal("49.999") < spent <= Decimal("50")
    assert fill.quantity * fill.price + fill.fees == spent
    assert fill.quantity.as_tuple().exponent == -8
    next_quantity = fill.quantity + SHARE_QUANTUM
    assert next_quantity * fill.price + fee(next_quantity * fill.price, settings) > 50


def test_fractional_sale_liquidates_all_shares_even_below_five_dollars():
    settings = ExecutionSettings(fractional_shares=True)
    fill = rebalance(Decimal("0"), Decimal("0.00500001"), Decimal("200"), 0.0, settings)
    assert fill.quantity == Decimal("0.00500001")
    assert fill.holdings == 0
    assert fill.cash == Decimal("1.00000200")


def test_fractional_sale_that_cannot_cover_fees_keeps_position():
    settings = ExecutionSettings(fractional_shares=True, commission_fixed=2)
    fill = rebalance(Decimal("0"), Decimal("0.00500001"), Decimal("200"), 0.0, settings)
    assert fill.reason == "sale_proceeds_cannot_cover_fees"
    assert fill.holdings == Decimal("0.00500001")
    assert fill.cash == 0


def test_fractional_backtest_tracks_exact_quantities_and_final_sale():
    result = simulate(ExecutionSettings(initial_cash=5, fractional_shares=True))
    buy, sell = result.trades.to_dicts()
    assert buy["quantity"] == sell["quantity"] == 0.005
    assert Decimal(buy["quantity_exact"]) == Decimal(sell["quantity_exact"]) == Decimal("0.005")
    assert Decimal(buy["holdings_exact"]) == Decimal("0.005")
    assert Decimal(sell["holdings_exact"]) == 0
    assert result.equity.schema["holdings"] == pl.Float64
    assert "requested_quantity_exact" in result.orders.columns
    assert result.metrics["final_equity"] == 6
    assert result.metrics["open_shares"] == 0
    assert Decimal(result.metadata["share_quantity_precision"]) == SHARE_QUANTUM
    with localcontext() as context:
        context.prec = 800
        for row in result.equity.to_dicts():
            if row["price"] is not None:
                assert Decimal(row["equity_exact"]) == (
                    Decimal(row["cash_exact"]) + Decimal(row["holdings_exact"]) * Decimal(str(row["price"]))
                )


def test_fractional_settings_and_exact_ledgers_roundtrip(tmp_path):
    settings = ExecutionSettings(initial_cash=5, fractional_shares=True, commission_fixed=Decimal("0.01"))
    result = simulate(settings)
    saved = save_run(tmp_path, [result], {})
    settings_json = saved.manifest["results"][0]["settings"]
    assert settings_json["fractional_shares"] is True
    assert settings_json["minimum_trade_budget"] == "5"
    loaded = load_run(tmp_path, saved.run_id).results[0]
    assert loaded.settings == settings
    assert loaded.equity.equals(result.equity)
    assert loaded.trades.equals(result.trades)
    assert loaded.orders.equals(result.orders)


def test_settings_from_before_fractional_support_load_with_defaults(tmp_path):
    result = simulate(ExecutionSettings(initial_cash=10000))
    saved = save_run(tmp_path, [result], {})
    document = saved.manifest
    settings = document["results"][0]["settings"]
    settings.pop("fractional_shares")
    settings.pop("minimum_trade_budget")
    manifest = saved.path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    (saved.path / "manifest.sha256").write_text(sha256(manifest.read_bytes()).hexdigest(), encoding="ascii")
    loaded = load_run(tmp_path, saved.run_id).results[0]
    assert loaded.settings.fractional_shares is False
    assert loaded.settings.minimum_trade_budget == Decimal("5")
    assert loaded.trades.schema["quantity"] == pl.Int64
    assert "quantity_exact" not in loaded.trades.columns


@pytest.mark.parametrize("settings", [{"fractional_shares": "true"}, {"fractional_shares": 1},
                                      {"minimum_trade_budget": -1}, {"minimum_trade_budget": True}])
def test_invalid_fractional_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ExecutionSettings(**settings)
