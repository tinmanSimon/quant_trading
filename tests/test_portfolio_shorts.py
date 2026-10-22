"""Short bracket execution, collateral, and mixed-direction capital accounting."""

from datetime import timedelta
from decimal import Decimal, localcontext

import polars as pl
import pytest

from research.backtesting import ExecutionSettings
from research.backtesting.initial_split import run_initial_split_backtests, summarize_initial_split
from research.backtesting.portfolio import run_portfolio_backtest
from research.runs import load_run, save_run
from research.strategies.bracket import BracketStrategy
from test_portfolio_backtest import BASE, bars


class DirectionSignals(BracketStrategy):
    name, lookback = "test_direction_signals", 1
    stop_percentage, win_percentage = 0.8, 1.25

    def __init__(self, shorts=None, longs=None):
        self.shorts = {"A": (0,)} if shorts is None else shorts
        self.longs = {} if longs is None else longs

    @property
    def config(self):
        return {"shorts": self.shorts, "longs": self.longs,
                "stop_percentage": self.stop_percentage, "win_percentage": self.win_percentage}

    def entry_signal(self, history):
        return history.height - 1 in self.longs.get(history["symbol"][-1], ())

    def short_entry_signal(self, history):
        return history.height - 1 in self.shorts.get(history["symbol"][-1], ())


class PreparedDirectionSignals(DirectionSignals):
    def prepare_entry_signals(self, history):
        return pl.Series([index in self.longs.get(history["symbol"][0], ())
                          for index in range(history.height)])

    def prepare_short_entry_signals(self, history):
        return pl.Series([index in self.shorts.get(history["symbol"][0], ())
                          for index in range(history.height)])


def run(frames, strategy=None, *, settings=None, **kwargs):
    return run_portfolio_backtest(
        frames, DirectionSignals() if strategy is None else strategy,
        start=BASE + timedelta(hours=1),
        end=BASE + timedelta(hours=max(frame.height for frame in frames.values())),
        settings=ExecutionSettings(initial_cash=1000) if settings is None else settings,
        **kwargs,
    )


def exact(record, name):
    return Decimal(record[name + "_exact"])


def assert_pnl_reconciles(result):
    with localcontext() as context:
        context.prec = 800
        realized = sum((Decimal(value) for value in result.completed_trades["net_pnl_exact"]), Decimal(0))
        unrealized = sum((Decimal(value) for value in result.open_positions["unrealized_pnl_exact"]), Decimal(0))
        assert Decimal(result.metrics["realized_pnl_exact"]) == realized
        assert Decimal(result.metrics["unrealized_pnl_exact"]) == unrealized
        assert Decimal(result.metrics["total_pnl_exact"]) == realized + unrealized
        assert Decimal(result.metrics["final_equity_exact"]) == result.settings.initial_cash + realized + unrealized


@pytest.mark.parametrize(
    ("high", "low", "outcome", "exit_price", "pnl"),
    [(100, 79, "win", 80, 200), (126, 100, "loss", 125, -250),
     (126, 79, "inconclusive", 125, -250)],
)
def test_short_entry_bar_exits_use_reciprocal_thresholds(high, low, outcome, exit_price, pnl):
    result = run({"A": bars("A", [100, 100], highs=[100, high], lows=[100, low])})
    record = result.completed_trades.row(0, named=True)
    assert record["direction"] == "short"
    assert record["outcome"] == outcome
    assert record["entry_index"] == record["exit_index"] == 1
    assert record["stop_price"] == 125 and record["win_price"] == 80
    assert record["quantity"] == 10
    assert record["entry_price"] == 100 and record["exit_price"] == exit_price
    assert record["net_pnl"] == pnl
    assert result.trades["side"].to_list() == ["sell", "buy"]
    assert result.trades["direction"].to_list() == ["short", "short"]
    assert result.trades["position_action"].to_list() == ["entry", "exit"]
    assert result.open_positions.is_empty()
    assert result.metrics["wins"] == int(outcome == "win")
    assert result.metrics["losses"] == int(outcome == "loss")
    assert result.metrics["inconclusive"] == int(outcome == "inconclusive")
    assert result.equity["cash"][-1] == 1000 + pnl
    assert_pnl_reconciles(result)


@pytest.mark.parametrize(("high", "low"), [(125, 100), (100, 80), (125, 80)])
def test_short_threshold_touches_keep_position_open(high, low):
    result = run({"A": bars("A", [100, 100], highs=[100, high], lows=[100, low])})
    assert result.completed_trades.is_empty()
    assert result.open_positions.height == 1
    assert result.trades["side"].to_list() == ["sell"]
    assert result.metrics["final_equity"] == 1000
    assert_pnl_reconciles(result)


def test_short_signal_uses_next_open_and_original_source_indices():
    result = run({"A": bars("A", [90, 100, 200], lows=[90, 100, 150])},
                 DirectionSignals(shorts={"A": (1,)}))
    entry = result.trades.row(0, named=True)
    record = result.completed_trades.row(0, named=True)
    assert entry["timestamp"] == BASE + timedelta(hours=2)
    assert entry["decision_timestamp"] == BASE + timedelta(hours=1)
    assert (record["signal_index"], record["entry_index"], record["exit_index"]) == (1, 2, 2)
    assert record["entry_open"] == record["entry_price"] == 200
    assert record["stop_price"] == 250 and record["win_price"] == 160
    assert record["net_pnl"] == 200


def test_short_liability_and_locked_cash_are_in_open_equity():
    result = run({"A": bars("A", [100, 100], lows=[100, 90], closes=[100, 90])})
    position = result.open_positions.row(0, named=True)
    equity = result.equity.row(-1, named=True)
    assert position["direction"] == "short"
    assert position["market_value"] == -900
    assert position["unrealized_pnl"] == 100
    assert exact(position, "reserved_margin") == 1000
    assert exact(position, "short_proceeds") == 1000
    assert equity["cash"] == 0
    assert exact(equity, "reserved_margin") == exact(equity, "short_proceeds") == 1000
    assert equity["equity"] == 1100
    assert_pnl_reconciles(result)


def test_short_sale_proceeds_cannot_fund_additional_long_or_short_entries():
    result = run(
        {symbol: bars(symbol, [100] * 3) for symbol in ("A", "B", "C")},
        DirectionSignals(shorts={"A": (0,), "C": (1,)}, longs={"B": (1,)}),
        allocation_mode="fixed_tenth", entry_percentage=50,
    )
    entries = result.trades.filter(pl.col("position_action") == "entry")
    assert entries["symbol"].to_list() == ["A", "B"]
    assert entries["direction"].to_list() == ["short", "long"]
    assert entries["quantity"].to_list() == [5, 5]
    assert entries["cash"].to_list() == [500, 0]
    rejected = result.orders.filter(pl.col("status") == "rejected")
    assert rejected["symbol"].to_list() == ["C"]
    assert rejected["reason"].to_list() == ["insufficient_cash_for_fixed_allocation"]
    assert result.metrics["final_equity"] == 1000
    assert_pnl_reconciles(result)


def test_short_cover_cash_is_available_only_at_a_later_open():
    result = run(
        {"A": bars("A", [100] * 4, lows=[100, 100, 79, 100]),
         "B": bars("B", [100] * 4)},
        DirectionSignals(shorts={"A": (0,)}, longs={"B": (1, 2)}),
        allocation_mode="fixed_tenth", entry_percentage=100,
    )
    entries = result.trades.filter(pl.col("position_action") == "entry")
    assert entries["symbol"].to_list() == ["A", "B"]
    assert entries["timestamp"].to_list() == [BASE + timedelta(hours=1), BASE + timedelta(hours=3)]
    assert result.orders.filter(pl.col("status") == "rejected")["symbol"].to_list() == ["B"]
    assert result.equity["cash"][-1] == 200
    assert result.metrics["final_equity"] == 1200
    assert_pnl_reconciles(result)


def test_opposite_signal_cannot_overlap_an_existing_position():
    result = run(
        {"A": bars("A", [100] * 4, highs=[100, 100, 126, 126])},
        DirectionSignals(shorts={"A": (0, 1)}, longs={"A": (1, 2)}),
    )
    assert result.completed_trades["direction"].to_list() == ["short", "long"]
    assert result.completed_trades["entry_index"].to_list() == [1, 3]
    assert result.completed_trades["exit_index"].to_list() == [2, 3]
    assert result.trades["side"].to_list() == ["sell", "buy", "buy", "sell"]
    assert_pnl_reconciles(result)


def test_short_adverse_slippage_fees_and_raw_open_collateral_reconcile():
    result = run(
        {"A": bars("A", [100, 100], lows=[100, 79])},
        settings=ExecutionSettings(initial_cash=1000, commission_fixed=1, commission_bps=100, slippage_bps=100),
    )
    entry = result.trades.row(0, named=True)
    record = result.completed_trades.row(0, named=True)
    assert record["quantity"] == 9
    assert exact(record, "entry_price") == 99
    assert exact(record, "exit_price") == Decimal("80.8")
    assert exact(record, "reserved_margin") == 900
    assert exact(record, "short_proceeds") == 891
    assert exact(entry, "fees") == Decimal("9.91")
    assert exact(entry, "cash") == Decimal("90.09")
    assert exact(record, "exit_fees") == Decimal("8.272")
    assert exact(record, "net_pnl") == Decimal("145.618")
    assert Decimal(result.metrics["final_equity_exact"]) == Decimal("1145.618")
    assert_pnl_reconciles(result)


@pytest.mark.parametrize("allocation_mode", ["per_entry", "fixed_tenth", "initial_split"])
@pytest.mark.parametrize("budget", [5, 4.99])
def test_fractional_short_entries_honor_minimum_budget_in_every_allocation_mode(allocation_mode, budget):
    frames = {"A": bars("A", [1000, 1000], lows=[1000, 799])}
    settings = ExecutionSettings(initial_cash=budget, fractional_shares=True)
    if allocation_mode == "initial_split":
        result, = run_initial_split_backtests(
            frames, [DirectionSignals()], start=BASE + timedelta(hours=1),
            end=BASE + timedelta(hours=2), settings=settings,
        )
    else:
        result = run(frames, settings=settings, allocation_mode=allocation_mode, entry_percentage=100)
    if budget == 5:
        record = result.completed_trades.row(0, named=True)
        assert exact(record, "quantity") == Decimal("0.005")
        assert exact(record, "net_pnl") == 1
        assert result.metrics["final_equity"] == 6
    else:
        assert result.trades.is_empty()
        assert result.orders["reason"].to_list() == ["below_minimum_trade_budget"]
    assert_pnl_reconciles(result)


def test_fractional_short_quantity_is_rounded_down_within_raw_open_budget():
    result = run(
        {"A": bars("A", [123.456789] * 2, lows=[123.456789, 110], closes=[123.456789, 110])},
        settings=ExecutionSettings(initial_cash=100, fractional_shares=True, commission_fixed=Decimal("0.037"),
                                   commission_bps=7, slippage_bps=13),
    )
    entry = result.trades.row(0, named=True)
    position = result.open_positions.row(0, named=True)
    with localcontext() as context:
        context.prec = 800
        quantity = exact(entry, "quantity")
        reserved = quantity * exact(position, "entry_open")
        assert quantity.as_tuple().exponent == -8
        assert reserved == exact(position, "reserved_margin")
        assert exact(position, "short_proceeds") == quantity * exact(entry, "price")
        assert Decimal("99.99") < reserved + exact(entry, "fees") <= 100
        assert exact(entry, "cash") == 100 - reserved - exact(entry, "fees")
    assert_pnl_reconciles(result)


@pytest.mark.parametrize(("high", "low", "outcome", "multiplier"),
                         [(101.23, 97, "win", 1.03), (104, 101.23, "loss", .98)])
def test_nonterminating_reciprocal_thresholds_keep_exact_fractional_ledger(high, low, outcome, multiplier):
    strategy = DirectionSignals(shorts={"A": (0,), "B": (0,)})
    strategy.stop_percentage, strategy.win_percentage = .98, 1.03
    result = run(
        {"A": bars("A", [101.23] * 2, highs=[101.23, high], lows=[101.23, low]),
         "B": bars("B", [101.23] * 2, lows=[101.23, 100], closes=[101.23, 100])},
        strategy,
        settings=ExecutionSettings(initial_cash=100, fractional_shares=True, commission_fixed=Decimal("0.037"),
                                   commission_bps=7, slippage_bps=13),
    )
    record = result.completed_trades.row(0, named=True)
    assert record["outcome"] == outcome
    threshold = "win_price" if outcome == "win" else "stop_price"
    assert record[threshold] == pytest.approx(101.23 / multiplier)
    with localcontext() as context:
        context.prec = 800
        assert exact(record, "exit_price") == exact(record, threshold) * Decimal("1.0013")
        gross = (exact(record, "entry_price") - exact(record, "exit_price")) * exact(record, "quantity")
        assert exact(record, "gross_pnl") == gross
        assert exact(record, "net_pnl") == gross - exact(record, "entry_fees") - exact(record, "exit_fees")
    assert_pnl_reconciles(result)


def test_short_stop_is_covered_even_when_the_loss_exceeds_account_cash():
    strategy = DirectionSignals(shorts={"A": (0, 2)})
    strategy.stop_percentage = 0.1
    result = run({"A": bars("A", [100] * 4, highs=[100, 100, 1001, 100])}, strategy)
    assert result.trades["side"].to_list() == ["sell", "buy"]
    assert result.completed_trades["exit_index"].to_list() == [2]
    assert result.completed_trades["net_pnl"].to_list() == [-9000]
    assert result.open_positions.is_empty()
    assert result.equity["cash"][-1] == result.metrics["final_equity"] == -8000
    assert_pnl_reconciles(result)


def test_prepared_short_signals_match_completed_history_signals():
    frames = {"A": bars("A", [100] * 4, highs=[100, 100, 126, 126])}
    kwargs = dict(shorts={"A": (0, 1)}, longs={"A": (2,)})
    fallback = run(frames, DirectionSignals(**kwargs))
    prepared = run(frames, PreparedDirectionSignals(**kwargs))
    assert prepared.metrics == fallback.metrics
    for name in ("trades", "orders", "equity", "completed_trades", "open_positions"):
        assert getattr(prepared, name).equals(getattr(fallback, name))


def test_parallel_initial_split_short_results_match_serial_and_combined_equity():
    frames = {"A": bars("A", [100, 100], lows=[100, 79]),
              "B": bars("B", [100, 100], lows=[100, 90], closes=[100, 90])}
    kwargs = dict(start=BASE + timedelta(hours=1), end=BASE + timedelta(hours=2),
                  settings=ExecutionSettings(initial_cash=1000, fractional_shares=True))
    strategy = DirectionSignals(shorts={"A": (0,), "B": (0,)})
    serial = run_initial_split_backtests(frames, [strategy], **kwargs)
    parallel = run_initial_split_backtests(frames, [strategy], max_workers=2, **kwargs)
    for left, right in zip(serial, parallel):
        assert left.metrics == right.metrics
        for name in ("trades", "orders", "equity", "completed_trades", "open_positions"):
            assert getattr(left, name).equals(getattr(right, name))
        assert_pnl_reconciles(right)
    summary = summarize_initial_split(parallel)[0]
    assert summary["realized_pnl"] == 100
    assert summary["unrealized_pnl"] == 50
    assert summary["final_equity"] == 1150


def test_short_details_and_locked_balances_survive_saved_run_roundtrip(tmp_path):
    original = run(
        {"A": bars("A", [100, 100], lows=[100, 79]),
         "B": bars("B", [100, 100], lows=[100, 90], closes=[100, 90])},
        DirectionSignals(shorts={"A": (0,), "B": (0,)}),
        settings=ExecutionSettings(initial_cash=1000, fractional_shares=True),
    )
    saved = save_run(tmp_path, [original], {})
    restored = load_run(tmp_path, saved.run_id).results[0]
    for name in ("trades", "orders", "equity", "completed_trades", "open_positions"):
        assert getattr(restored, name).equals(getattr(original, name))
    assert restored.metrics == original.metrics
    assert_pnl_reconciles(restored)
