"""Chart construction must faithfully represent stored bars and their labels."""

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dashboard.charts import CHART_CONFIG, bar_table, display_timestamps, price_chart


def test_chart_preserves_every_numeric_value_and_does_not_mutate(sample_ohlcv_frame):
    source = sample_ohlcv_frame.clone()
    chart = price_chart(source, timeframe="1h", timezone="America/New_York")
    for name in ("open", "high", "low", "close"):
        assert list(getattr(chart.data[0], name)) == source[name].to_list()
    assert list(chart.data[1].y) == source["volume"].to_list()
    assert list(chart.data[0].customdata) == source["volume"].to_list()
    assert "Volume: %{customdata}" in chart.data[0].hovertemplate
    assert len(chart.data[0].x) == source.height
    assert list(chart.data[0].x) == list(chart.data[1].x) == list(range(source.height))
    assert chart.layout.xaxis.type == "linear"
    assert_frame_equal(source, sample_ohlcv_frame)
    assert CHART_CONFIG["scrollZoom"] is True
    assert chart.layout.xaxis2.rangeslider.visible is True


def test_intraday_labels_apply_daylight_saving_and_retain_utc_hover():
    timestamps = [datetime(2025, 7, 3, 16, 30, tzinfo=UTC), datetime(2025, 12, 24, 17, 30, tzinfo=UTC)]
    labels = display_timestamps(timestamps, timeframe="1h", timezone="America/New_York")
    assert labels == ["2025-07-03T12:30:00", "2025-12-24T12:30:00"]
    frame = pl.DataFrame({"timestamp": timestamps, "symbol": ["AAPL", "AAPL"],
                          "open": [1., 2.], "high": [2., 3.], "low": [1., 2.],
                          "close": [2., 3.], "volume": [10., 20.]})
    figure = price_chart(frame, timeframe="1h", timezone="America/New_York")
    assert "2025-12-24T17:30:00+00:00" in figure.data[0].text[1]


def test_session_dates_do_not_shift_to_previous_day():
    stamps = [datetime(2025, 12, 24, tzinfo=UTC)]
    for timeframe in ("1d", "1wk", "1mo"):
        assert display_timestamps(stamps, timeframe=timeframe, timezone="America/New_York") == ["2025-12-24"]
        assert display_timestamps(stamps, timeframe=timeframe, timezone="Asia/Shanghai") == ["2025-12-24"]


def test_ambiguous_dst_labels_require_utc_instead_of_overplotting_bars():
    stamps = [datetime(2025, 11, 2, 5, 30, tzinfo=UTC), datetime(2025, 11, 2, 6, 30, tzinfo=UTC)]
    with pytest.raises(ValueError, match="Select UTC"):
        display_timestamps(stamps, timeframe="1h", timezone="America/New_York")
    assert display_timestamps(stamps, timeframe="1h", timezone="UTC") == [
        "2025-11-02T05:30:00", "2025-11-02T06:30:00"]
    with pytest.raises(ValueError, match="Select UTC"):
        display_timestamps([datetime(2025, 11, 2, 5, 45, tzinfo=UTC), stamps[1]],
                           timeframe="1h", timezone="America/New_York")
    assert display_timestamps([stamps[0], stamps[0]], timeframe="1h", timezone="America/New_York") == [
        "2025-11-02T01:30:00", "2025-11-02T01:30:00"]


def test_missing_bar_markers_do_not_add_or_fill_rows(sample_ohlcv_frame):
    missing = datetime(2026, 1, 5, 17, 30, tzinfo=UTC)
    figure = price_chart(sample_ohlcv_frame, timeframe="1h", omitted_timestamps=[missing])
    assert len(figure.data[0].x) == sample_ohlcv_frame.height
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == sample_ohlcv_frame.height - 0.5
    assert "2026-01-05T12:30:00" in figure.layout.annotations[0].hovertext
    assert "2026-01-05T17:30:00+00:00" in figure.layout.annotations[0].hovertext


@pytest.mark.parametrize("timeframe", ["1m", "15m", "1h", "1d"])
def test_gaps_compress_without_changing_bars_or_real_labels(sample_ohlcv_frame, timeframe):
    stamps = [datetime(2025, 1, day, 15, tzinfo=UTC) for day in (3, 6, 21)]
    frame = sample_ohlcv_frame.with_columns(pl.Series("timestamp", stamps))
    original = frame.clone()
    chart = price_chart(frame, timeframe=timeframe, timezone="America/New_York")
    assert list(chart.data[0].x) == list(chart.data[1].x) == [0, 1, 2]
    assert list(chart.layout.xaxis.range) == [-0.5, 2.5]
    assert chart.layout.xaxis.minallowed == -0.5 and chart.layout.xaxis.maxallowed == 2.5
    assert chart.layout.meta["bar_labels"] == display_timestamps(stamps, timeframe=timeframe, timezone="America/New_York")
    for index, stamp in enumerate(stamps):
        assert stamp.isoformat() in chart.data[0].text[index]
    for name in ("open", "high", "low", "close"):
        assert list(getattr(chart.data[0], name)) == frame[name].to_list()
    assert list(chart.data[1].y) == frame["volume"].to_list()
    assert_frame_equal(frame, original)


def test_omissions_share_boundaries_without_expanding_axis(sample_ohlcv_frame):
    stamps = sample_ohlcv_frame["timestamp"].to_list()
    middle = [stamps[0] + timedelta(minutes=10), stamps[0] + timedelta(minutes=20)]
    missing = [stamps[0] - timedelta(days=1), *middle, stamps[-1] + timedelta(days=1),
               middle[0], stamps[1]]  # Deduplicate omissions; ignore restored bars.
    chart = price_chart(sample_ohlcv_frame, timeframe="1h", omitted_timestamps=missing)
    assert [shape.x0 for shape in chart.layout.shapes] == [-0.5, 0.5, 2.5]
    assert chart.layout.annotations[1].text == "Missing: 2"
    assert all(stamp.isoformat() in chart.layout.annotations[1].hovertext for stamp in middle)
    assert list(chart.layout.xaxis.range) == [-0.5, 2.5]
    assert len(chart.data[0].x) == sample_ohlcv_frame.height


def test_price_chart_requires_nonempty_chronological_unique_bars(sample_ohlcv_frame):
    for frame in (sample_ohlcv_frame.head(0), sample_ohlcv_frame.reverse(),
                  pl.concat([sample_ohlcv_frame.head(1)] * 2)):
        with pytest.raises(ValueError):
            price_chart(frame, timeframe="1h")


@pytest.mark.parametrize("count", [1, 49, 50, 51, 2500])
def test_initial_view_shows_latest_fifty_without_discarding_history(sample_ohlcv_frame, count):
    frame = pl.concat([sample_ohlcv_frame.head(1)] * count).with_columns(
        pl.Series("timestamp", [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(count)]))
    chart = price_chart(frame, timeframe="1m")
    expected = [max(0, count - 50) - 0.5, count - 0.5]
    assert list(chart.layout.xaxis.range) == list(chart.layout.xaxis2.range) == expected
    assert list(chart.layout.xaxis2.rangeslider.range) == [-0.5, count - 0.5]
    assert list(chart.data[0].x) == list(range(count))
    assert list(chart.data[1].y) == frame["volume"].to_list()
    assert chart.layout.yaxis.automargin is True


def test_candle_hover_preserves_fractional_volume(sample_ohlcv_frame):
    volumes = [0.0, 100.125, 1000.1234567890123]
    frame = sample_ohlcv_frame.with_columns(pl.Series("volume", volumes))
    chart = price_chart(frame, timeframe="1h")
    assert list(chart.data[0].customdata) == list(chart.data[1].y) == volumes


def test_bar_table_keeps_numeric_precision(sample_ohlcv_frame):
    frame = sample_ohlcv_frame.with_columns(pl.lit(1.0000000000000002).alias("close"))
    before = frame.clone()
    table = bar_table(frame, timeframe="1h", timezone="Asia/Shanghai")
    assert table["close"].to_list() == frame["close"].to_list()
    assert table["timestamp_utc"].to_list() == [stamp.isoformat() for stamp in frame["timestamp"]]
    assert_frame_equal(frame, before)


def test_dashboard_navigates_with_empty_local_store(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as app
    from research import Research

    monkeypatch.setattr(app, "Research", lambda **kwargs: Research(tmp_path / "data", tmp_path / "runs"))
    view = AppTest.from_string("from dashboard.app import main\nmain()").run()
    assert not view.exception
    assert not view.error
    assert "No matching local datasets" in view.info[0].value
    for page in ("Fetch", "Backtest", "Saved runs"):
        view.sidebar.radio[0].set_value(page).run()
        assert not view.exception
        assert not view.error


def test_quality_displays_known_gaps_even_when_provenance_is_unknown(monkeypatch):
    from types import SimpleNamespace
    import dashboard.app as app
    from data_pipeline import FetchQuality, OmittedBar

    missing = datetime(2025, 12, 24, 17, 30, tzinfo=UTC)
    restored = datetime(2025, 12, 24, 16, 30, tzinfo=UTC)
    item = SimpleNamespace(dataset_id="example", quality=FetchQuality(
        status="unknown", omitted_bars=(OmittedBar(missing, "all_ohlc_missing"),
                                        OmittedBar(restored, "all_ohlc_missing"))))
    calls = {name: [] for name in ("caption", "info", "warning", "dataframe")}
    fake = SimpleNamespace(**{name: (lambda value, _name=name, **kwargs: calls[_name].append(value))
                              for name in calls})
    monkeypatch.setattr(app, "st", fake)
    marks = app._quality([item], datetime(2025, 12, 24, tzinfo=UTC),
                         datetime(2025, 12, 25, tzinfo=UTC), present_timestamps=[restored])
    assert marks == [missing]
    assert "unavailable" in calls["caption"][0]
    assert "still absent" in calls["warning"][0]
    assert "historical" in calls["info"][0]
    states = dict(calls["dataframe"][0].select("timestamp_utc", "current_state").iter_rows())
    assert states == {missing: "absent from selected data", restored: "present in selected data"}


def test_equity_chart_uses_valuation_time_instead_of_earlier_bar_label(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import dashboard.app as app

    opened = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    closed = datetime(2025, 1, 2, 15, 30, tzinfo=UTC)
    equity = pl.DataFrame({"timestamp": [opened, opened], "valuation_time": [opened, closed],
                           "equity": [100.0, 110.0]})
    original = equity.clone()
    result = SimpleNamespace(symbol="AAPL", strategy_spec={"name": "example"}, equity=equity,
                             trades=pl.DataFrame(), orders=pl.DataFrame(), metrics={})
    run = SimpleNamespace(run_id="example", results=[result], comparison=pl.DataFrame(), manifest={})
    charts = []
    ignore = lambda *args, **kwargs: None
    fake = SimpleNamespace(subheader=ignore, caption=ignore, dataframe=ignore, write=ignore, json=ignore,
                            selectbox=lambda *args, **kwargs: 0,
                            expander=lambda *args, **kwargs: nullcontext(),
                            plotly_chart=lambda figure, **kwargs: charts.append(figure))
    monkeypatch.setattr(app, "st", fake)
    app._show_result(run)
    assert list(charts[0].data[0].x) == ["2025-01-02T14:30:00", "2025-01-02T15:30:00"]
    assert list(charts[0].data[0].y) == [100.0, 110.0]
    assert_frame_equal(equity, original)


def test_dashboard_browses_verified_data_and_aborts_missing_ticker(tmp_path, monkeypatch, sample_ohlcv_frame):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as app
    from data_pipeline import DataRequest
    from research import Research

    research = Research(tmp_path / "data", tmp_path / "runs")
    research.pipeline.ingest_frame(DataRequest(symbol="AAPL", timeframe="1h",
        start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 3, tzinfo=UTC)), sample_ohlcv_frame)
    monkeypatch.setattr(app, "Research", lambda **kwargs: research)
    view = AppTest.from_string("from dashboard.app import main\nmain()").run()
    assert not view.exception
    assert not view.error
    assert len(view.get("bidi_component")) == 2  # Constant Plotly runtime + bounded chart window.
    view.sidebar.radio[0].set_value("Backtest").run()
    next(item for item in view.text_input if item.label == "Additional required tickers").set_value("MSFT")
    next(item for item in view.button if item.label == "Validate all data and run").click().run(timeout=20)
    assert not view.exception
    assert "Backtest aborted" in view.error[0].value
    assert not research.list_runs()


def test_dashboard_runs_complete_daily_backtest_and_reopens_saved_run(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import dashboard.app as app
    from data_pipeline import DataRequest
    from research import Research

    research = Research(tmp_path / "data", tmp_path / "runs")
    days = (2, 3, 4, 5, 8, 9)
    frame = pl.DataFrame({
        "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in days],
        "symbol": ["AAPL"] * len(days),
        "open": [100.0 + index for index in range(len(days))],
        "high": [101.0 + index for index in range(len(days))],
        "low": [99.0 + index for index in range(len(days))],
        "close": [100.5 + index for index in range(len(days))],
        "volume": [1000.0] * len(days),
    })
    research.pipeline.ingest_frame(DataRequest(
        symbol="AAPL", timeframe="1d", start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 10, tzinfo=UTC)), frame)
    monkeypatch.setattr(app, "Research", lambda **kwargs: research)
    view = AppTest.from_string("from dashboard.app import main\nmain()").run()
    view.sidebar.radio[0].set_value("Backtest").run()
    next(item for item in view.date_input if item.label == "Start date (UTC, inclusive)").set_value(date(2024, 1, 5))
    next(item for item in view.date_input if item.label == "End date (UTC, exclusive)").set_value(date(2024, 1, 10))
    next(item for item in view.number_input if item.label == "Fast moving average (bars)").set_value(1)
    next(item for item in view.number_input if item.label == "Slow moving average (bars)").set_value(2)
    next(item for item in view.button if item.label == "Validate all data and run").click().run(timeout=20)
    assert not view.exception
    assert not view.error
    assert len(view.get("plotly_chart")) == 2
    runs = research.list_runs()
    assert len(runs) == 1
    assert research.load_run(runs[0]["run_id"]).results[0].symbol == "AAPL"

    view.sidebar.radio[0].set_value("Saved runs").run()
    assert not view.exception
    assert not view.error
    assert len(view.get("plotly_chart")) == 2
    assert any(item.value == f"Run {runs[0]['run_id']}" for item in view.subheader)
