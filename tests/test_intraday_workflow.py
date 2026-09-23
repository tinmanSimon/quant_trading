"""Exact interval schedules and offline Yahoo -> storage -> backtest acceptance."""

from copy import deepcopy
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from unittest.mock import Mock

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from yfinance.base import TickerBase
from yfinance.data import YfData

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import InvalidDataRequestError
from data_pipeline.providers import YFinanceProvider
from research import PreflightError, Research, ResearchError
from research.backtesting import ExecutionSettings
from research.datasets import prepare_snapshot
from research.instruments import Instrument, expected_bars
from research.strategies import Momentum, Strategy
from research.timeframes import BACKTEST_TIMEFRAMES, fetch_timeframes


INTERVALS = [("1m", 1), ("2m", 2), ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("90m", 90)]


def stamp(day, hour=0, minute=0):
    return datetime(2025, 7, day, hour, minute, tzinfo=UTC)


def source_frame(timeframe, symbol="AAPL"):
    # Explicit fixture sessions, independent of the production calendar builder:
    # July 1/2 regular sessions and July 3 early close, all in EDT.
    labels = []
    if timeframe == "1d":
        labels = [stamp(day) for day in (1, 2, 3)]
    else:
        minutes = dict(INTERVALS)[timeframe]
        for day, length in ((1, 390), (2, 390), (3, 210)):
            labels.extend(stamp(day, 13, 30) + timedelta(minutes=i) for i in range(0, length, minutes))
    n = len(labels)
    prices = [100.1234567890123 + i * 0.01 for i in range(n)]
    return pl.DataFrame({
        "timestamp": labels, "symbol": [symbol] * n,
        "open": prices, "high": [value + 2 for value in prices],
        "low": [value - 1 for value in prices], "close": [value + 1 for value in prices],
        "volume": [float(1000 + i) for i in range(n)],
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))


@pytest.mark.parametrize("timeframe,minutes", INTERVALS)
@pytest.mark.parametrize("day,length", [(2, 390), (3, 210)])
def test_session_grid_preserves_exact_interval_and_partial_last_bar(timeframe, minutes, day, length):
    bars = expected_bars(Instrument("AAPL"), start=stamp(day), end=stamp(day + 1), timeframe=timeframe)
    labels = [stamp(day, 13, 30) + timedelta(minutes=i) for i in range(0, length, minutes)]
    assert [bar.timestamp for bar in bars] == labels
    assert [bar.open for bar in bars] == labels
    assert bars[-1].close == stamp(day, 13, 30) + timedelta(minutes=length)
    assert all(bar.close - bar.open == timedelta(minutes=minutes) for bar in bars[:-1])
    assert timedelta(0) < bars[-1].close - bars[-1].open <= timedelta(minutes=minutes)


@pytest.mark.parametrize("timeframe,minutes", INTERVALS)
def test_warmup_across_weekend_and_dst_uses_completed_previous_session(timeframe, minutes):
    start = datetime(2025, 3, 10, tzinfo=UTC)
    bars = expected_bars(Instrument("AAPL"), start=start, end=start + timedelta(days=1),
                         timeframe=timeframe, lookback=2)
    friday = datetime(2025, 3, 7, 14, 30, tzinfo=UTC)
    labels = [friday + timedelta(minutes=i) for i in range(0, 390, minutes)]
    assert [bar.timestamp for bar in bars[:2]] == labels[-2:]
    assert bars[1].close == datetime(2025, 3, 7, 21, tzinfo=UTC)
    assert bars[2].open == datetime(2025, 3, 10, 13, 30, tzinfo=UTC)


def test_holiday_is_not_expected_and_warmup_uses_early_close_session():
    with pytest.raises(ResearchError, match="no scheduled"):
        expected_bars(Instrument("AAPL"), start=stamp(4), end=stamp(5), timeframe="15m")
    bars = expected_bars(Instrument("AAPL"), start=stamp(7), end=stamp(8), timeframe="15m", lookback=1)
    assert bars[0].timestamp == stamp(3, 16, 45)
    assert bars[0].close == stamp(3, 17)


def test_sub_bar_start_does_not_reanchor_schedule():
    bars = expected_bars(Instrument("AAPL"), start=stamp(3, 13, 31), end=stamp(3, 14),
                         timeframe="15m", lookback=1)
    assert [bar.timestamp for bar in bars] == [stamp(3, 13, 30), stamp(3, 13, 45)]


@pytest.mark.parametrize("timeframe", ["1wk", "1mo", "3m", "2h", "0m"])
def test_unsupported_timeframes_are_explicit(timeframe):
    with pytest.raises((ResearchError, InvalidDataRequestError)):
        expected_bars(Instrument("AAPL"), start=stamp(3), end=stamp(4), timeframe=timeframe)


def test_alias_has_the_same_grid_and_lunch_breaks_remain_unsupported():
    kwargs = dict(start=stamp(3), end=stamp(4), lookback=1)
    assert expected_bars(Instrument("AAPL"), timeframe="60m", **kwargs) == expected_bars(
        Instrument("AAPL"), timeframe="1h", **kwargs)
    with pytest.raises(ResearchError, match="lunch break"):
        expected_bars(Instrument("TEST", calendar="XHKG"), timeframe="15m", **kwargs)


def test_warmup_window_scales_with_required_sessions_not_minute_bar_count(monkeypatch):
    import research.instruments as module
    original = module.xcals.get_calendar
    windows = []

    def calendar(name, *, start, end):
        windows.append((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days)
        return original(name, start=start, end=end)

    monkeypatch.setattr(module.xcals, "get_calendar", calendar)
    bars = expected_bars(Instrument("AAPL"), start=stamp(3), end=stamp(4), timeframe="1m", lookback=1000)
    assert len(bars) == 1210
    assert max(windows) < 30
    windows.clear()
    daily = expected_bars(Instrument("AAPL"), start=stamp(3), end=stamp(4), timeframe="1d", lookback=100)
    assert len(daily) == 101
    assert len(windows) > 1


@pytest.fixture
def yahoo_http(monkeypatch):
    """Run the actual pinned yfinance parser with synthetic Yahoo HTTP payloads."""
    calls = []

    def respond(self, url, params=None, **kwargs):
        symbol = url.rsplit("/", 1)[-1]
        interval = params["interval"]
        calls.append((symbol, interval))
        if symbol == "BAD":
            payload = {"chart": {"error": {"code": "Not Found", "description": "fixture unavailable"}, "result": None}}
        else:
            frame = source_frame(interval, symbol)
            labels = frame["timestamp"].to_list()
            # Yahoo daily timestamps identify the exchange session; canonical
            # storage converts these to midnight-UTC date labels.
            if interval == "1d":
                labels = [value.replace(hour=13, minute=30) for value in labels]
            quote = {name: frame[name].to_list() for name in ("open", "high", "low", "close", "volume")}
            payload = {"chart": {"error": None, "result": [{
                "meta": {"instrumentType": "EQUITY", "exchangeTimezoneName": "America/New_York",
                         "currency": "USD", "symbol": symbol, "dataGranularity": interval,
                         "tradingPeriods": [[{"start": int(stamp(day, 13, 30).timestamp()),
                                              "end": int(stamp(day, close).timestamp()),
                                              "timezone": "EDT", "gmtoffset": -14400}]
                                            for day, close in ((1, 20), (2, 20), (3, 17))]},
                "timestamp": [int(value.timestamp()) for value in labels],
                "indicators": {"quote": [quote]},
            }]}}
        response = Mock(text="{}")
        response.json.side_effect = lambda: deepcopy(payload)
        return response

    monkeypatch.setattr(YfData, "cache_get", respond)
    monkeypatch.setattr(YfData, "get", respond)
    monkeypatch.setattr(TickerBase, "_get_ticker_tz", lambda self, timeout: "America/New_York")
    return calls


def test_yahoo_fetch_store_reopen_backtest_every_workflow_interval(tmp_path, yahoo_http):
    pipeline = DataPipeline(tmp_path / "data")
    supported = fetch_timeframes(YFinanceProvider.supported_timeframes)
    assert set(supported) == {"1m", "2m", "5m", "15m", "1h", "90m", "1d"}
    assert "30m" in BACKTEST_TIMEFRAMES and "30m" not in supported
    for interval in supported:
        report = pipeline.fetch_many(["AAPL", "BAD", "MSFT"], start=stamp(1), end=stamp(4), timeframe=interval)
        assert [item.status for item in report.outcomes] == ["saved", "failed", "saved"], report.outcomes
        assert report.outcomes[0].quality["status"] == "reported"
    assert len(pipeline.list_datasets()) == 2 * len(supported)
    research = Research(tmp_path / "data", tmp_path / "runs")
    for interval in supported:
        query = DataQuery(provider="yahoo", symbol="AAPL", timeframe=interval)
        before = research.pipeline.read(query)
        assert_frame_equal(before, source_frame(interval))
        run = research.backtest(["AAPL", "MSFT"], strategies=[Momentum(1)], start=stamp(3), end=stamp(4),
                                timeframe=interval, settings=ExecutionSettings(commission_bps=1, slippage_bps=2))
        assert run.manifest["timeframe"] == interval
        for result in run.results:
            assert all(Decimal(value) >= 0 for value in result.equity["cash_exact"])
            assert result.trades.height > 0
            assert all(decision <= filled for decision, filled in result.trades.select(
                "decision_timestamp", "timestamp").iter_rows())
            assert all(decision < filled for decision, filled in result.trades.select(
                "decision_bar_timestamp", "bar_timestamp").iter_rows())
            source = run.manifest["sources"][result.symbol]
            assert all(item["request"]["timeframe"] == interval for item in source)
        restored = research.load_run(run.run_id)
        assert_frame_equal(restored.results[0].equity, run.results[0].equity)
        assert_frame_equal(research.pipeline.read(query), before)
    assert len(yahoo_http) == 3 * len(supported)
    # An alias cannot create a second copy under a different storage identity.
    duplicate = pipeline.fetch_many(["AAPL"], start=stamp(1), end=stamp(4), timeframe="60m")
    assert duplicate.outcomes[0].error_type == "OverlappingDataError"
    assert yahoo_http[-1] == ("AAPL", "1h")
    unsupported = pipeline.fetch_many(["AAPL"], start=stamp(1), end=stamp(4), timeframe="30m")
    assert unsupported.outcomes[0].error_type == "UnsupportedRequestError"


@pytest.mark.parametrize("interval", ["1m", "15m", "1h", "30m"])
@pytest.mark.parametrize("defect", ["warmup", "trading", "extra", "unfinished"])
def test_bad_intraday_coverage_aborts_before_strategy_execution(tmp_path, interval, defect):
    frame = source_frame(interval)
    start, end = stamp(3), stamp(4)
    if defect == "warmup":
        frame = frame.filter(pl.col("timestamp") != frame.filter(pl.col("timestamp") < start)["timestamp"][-1])
    elif defect == "trading":
        frame = frame.filter(pl.col("timestamp") != stamp(3, 13, 30))
    elif defect == "extra":
        off_grid = frame.filter(pl.col("timestamp") == stamp(3, 13, 30)).with_columns(
            pl.col("timestamp") + timedelta(seconds=1))
        frame = pl.concat([frame, off_grid]).sort("timestamp")
    else:
        end = stamp(3, 13, 30) + timedelta(seconds=30)
    research = Research(tmp_path / "data", tmp_path / "runs")
    research.pipeline.ingest_frame(DataRequest("AAPL", stamp(1), stamp(4), timeframe=interval), frame)

    class Spy(Strategy):
        name, lookback, config, calls = "spy", 2, {}, []

        def target_weight(self, history):
            self.calls.append(history.height)
            return 1.0

    with pytest.raises(PreflightError) as error:
        research.backtest(["AAPL", "MSFT"], strategies=[Spy()], start=start, end=end, timeframe=interval)
    assert any(issue.ticker == "AAPL" for issue in error.value.report.issues)
    assert any(issue.ticker == "MSFT" and issue.code == "no_data" for issue in error.value.report.issues)
    assert Spy.calls == [] and research.list_runs() == []


def test_interval_selection_never_substitutes_other_local_bars(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    research.pipeline.ingest_frame(DataRequest("AAPL", stamp(1), stamp(4), timeframe="1h"), source_frame("1h"))
    with pytest.raises(PreflightError) as error:
        research.backtest(["AAPL"], strategies=[Momentum(1)], start=stamp(3), end=stamp(4), timeframe="15m")
    assert error.value.report.issues[0].code == "no_data"


def test_thirty_minute_local_raw_data_can_be_backtested_without_yahoo_fetch(tmp_path):
    research = Research(tmp_path / "data", tmp_path / "runs")
    research.pipeline.ingest_frame(DataRequest("AAPL", stamp(1), stamp(4), timeframe="30m", provider="custom"),
                                   source_frame("30m"))
    run = research.backtest(["AAPL"], strategies=[Momentum(1)], start=stamp(3), end=stamp(4),
                            timeframe="30m", provider="custom")
    assert run.manifest["timeframe"] == "30m"
    assert run.results[0].equity.height == 8


@pytest.mark.parametrize("interval", ["1m", "15m", "90m"])
def test_as_of_cutoff_inside_bar_blocks_snapshot(tmp_path, interval):
    pipeline = DataPipeline(tmp_path / "data")
    pipeline.ingest_frame(DataRequest("AAPL", stamp(1), stamp(4), timeframe=interval), source_frame(interval))
    with pytest.raises(PreflightError) as error:
        prepare_snapshot(pipeline, tickers=["AAPL"], start=stamp(3), end=stamp(4),
                         timeframe=interval, lookback=2, as_of=stamp(3, 13, 30) + timedelta(seconds=30))
    unfinished = next(issue for issue in error.value.report.issues if issue.code == "unfinished_bars")
    assert unfinished.timestamps[0] == stamp(3, 13, 30)


@pytest.mark.parametrize("interval", ["1m", "15m"])
def test_strategy_only_sees_bars_completed_before_each_execution(tmp_path, interval):
    research = Research(tmp_path / "data", tmp_path / "runs")
    frame = source_frame(interval)
    research.pipeline.ingest_frame(DataRequest("AAPL", stamp(1), stamp(4), timeframe=interval), frame)
    duration = timedelta(minutes=dict(INTERVALS)[interval])

    class Spy(Strategy):
        name, lookback, config, calls = "spy", 2, {}, []

        def target_weight(self, history):
            execution_time = stamp(3, 13, 30) + len(self.calls) * duration
            assert history["timestamp"][-1] < execution_time
            self.calls.append(history["timestamp"][-1])
            # Deliberately mutate the supplied view. Engine inputs/storage must
            # still retain their original numeric values.
            history.replace_column(history.get_column_index("open"), pl.Series("open", [1.] * history.height))
            return 1.

    result = research.backtest(["AAPL"], strategies=[Spy()], start=stamp(3), end=stamp(4), timeframe=interval)
    assert len(Spy.calls) == 210 // dict(INTERVALS)[interval]
    assert result.results[0].trades["price"][0] == frame.filter(pl.col("timestamp") == stamp(3, 13, 30))["open"][0]
    assert_frame_equal(research.pipeline.read(DataQuery(provider="yahoo", symbol="AAPL", timeframe=interval)), frame)


@pytest.mark.parametrize("interval", ["1m", "15m", "1h", "1d"])
def test_dashboard_fetch_to_local_backtest_preserves_selected_interval(tmp_path, monkeypatch, yahoo_http, interval):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as dashboard

    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    view = AppTest.from_string("from dashboard.app import main\nmain()").run()
    view.sidebar.radio[0].set_value("Fetch").run()
    view.selectbox(key="fetch-provider").set_value("yahoo").run()
    assert set(view.selectbox(key="fetch-timeframe").options) == set(fetch_timeframes(YFinanceProvider.supported_timeframes))
    view.selectbox(key="fetch-timeframe").set_value(interval).run()
    view.text_area[0].set_value("AAPL")
    view.date_input(key="fetch-start").set_value(stamp(1).date())
    view.date_input(key="fetch-end").set_value(stamp(4).date())
    next(button for button in view.button if button.label == "Fetch and save").click().run(timeout=30)
    assert not view.exception
    assert view.session_state["fetch-report"].ok
    assert yahoo_http == [("AAPL", interval)]
    stored = research.pipeline.list_datasets()[0]
    assert stored.request.timeframe == interval
    before = research.pipeline.read_dataset(stored.dataset_id)

    view.sidebar.radio[0].set_value("Backtest").run()
    assert set(view.selectbox(key="backtest-timeframe").options) == set(BACKTEST_TIMEFRAMES)
    view.selectbox(key="backtest-timeframe").set_value(interval).run()
    view.date_input(key="backtest-start").set_value(stamp(3).date())
    view.date_input(key="backtest-end").set_value(stamp(4 if interval == "1d" else 3).date())
    if interval != "1d":
        view.time_input(key="backtest-start-time").set_value(time(13, 30))
        view.time_input(key="backtest-end-time").set_value(time(17))
    next(item for item in view.number_input if item.label == "Fast moving average (bars)").set_value(1)
    next(item for item in view.number_input if item.label == "Slow moving average (bars)").set_value(2)
    next(button for button in view.button if button.label == "Validate all data and run").click().run(timeout=30)
    assert not view.exception
    assert not view.error
    run = view.session_state["backtest-run"]
    assert run.manifest["timeframe"] == interval
    assert run.manifest["sources"]["AAPL"][0]["dataset_id"] == stored.dataset_id
    if interval != "1d":
        assert run.manifest["start"] == stamp(3, 13, 30).isoformat()
        assert run.manifest["end"] == stamp(3, 17).isoformat()
    assert yahoo_http == [("AAPL", interval)]  # Backtesting did not fetch again.
    assert_frame_equal(research.pipeline.read_dataset(stored.dataset_id), before)
    assert len(view.get("plotly_chart")) == 2
