"""The dashboard fetches through the pipeline and displays every ticker outcome."""

from calendar import monthrange
from datetime import date, timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest, FetchQuality, FetchResult
from data_pipeline.exceptions import ProviderError
from data_pipeline.providers import YFinanceProvider
from research import Research


def test_dashboard_exclude_invalid_bars_is_opt_in_and_saves_omissions(tmp_path, monkeypatch):
    import pandas as pd
    from data_pipeline.providers import yahoo

    research = Research(tmp_path / "data", tmp_path / "runs")
    frame = pd.DataFrame({"Open": [100., 0., 100.], "High": [101.] * 3,
                          "Low": [99.] * 3, "Close": [100.] * 3, "Volume": [10.] * 3},
                         index=pd.date_range("2024-01-02", periods=3, tz="America/New_York"))
    monkeypatch.setattr(yahoo, "_download_history", lambda **kwargs: frame.copy())
    app = AppTest.from_string(f"""
from research import Research
from dashboard.pages.fetch import _fetch_page
_fetch_page(Research({str(research.pipeline.store.data_dir)!r}, {str(research.runs_dir)!r}))
""").run()
    app.selectbox(key="fetch-provider").set_value("yahoo").run()
    app.selectbox(key="fetch-timeframe").set_value("1d").run()
    app.text_area[0].set_value("TEST")
    app.date_input(key="fetch-start").set_value(date(2024, 1, 2))
    app.date_input(key="fetch-end").set_value(date(2024, 1, 5))
    assert app.checkbox(key="fetch-skip-invalid-bars").value is False
    next(button for button in app.button if button.label == "Fetch and save").click().run()
    assert app.session_state["fetch-report"].outcomes[0].status == "failed"
    app.checkbox(key="fetch-skip-invalid-bars").check()
    with pytest.warns(UserWarning, match="excluded 1 invalid bars"):
        next(button for button in app.button if button.label == "Fetch and save").click().run()
    assert not app.exception and not app.error
    outcome = app.session_state["fetch-report"].outcomes[0]
    assert outcome.status == "saved" and outcome.row_count == 2
    assert outcome.quality["skip_invalid_bars"] is True
    bar, = outcome.quality["omitted_bars"]
    assert bar["timestamp"] == "2024-01-03T00:00:00+00:00"
    assert "open=0.0" in bar["reason"]
    assert any("1 invalid-bar exclusions" in item.value for item in app.warning)
    assert any(item.label == "Excluded invalid bars" for item in app.expander)


@pytest.mark.parametrize("skip", [False, True])
def test_dashboard_batch_fetch_uses_pipeline_and_keeps_later_successes(
    tmp_path, monkeypatch, sample_ohlcv_frame, skip,
):
    import dashboard.app as dashboard

    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    calls = []
    requested_starts = []
    policies = []
    fetch_many = research.pipeline.fetch_many

    def tracked_fetch(tickers, **kwargs):
        policies.append(kwargs["overlap_policy"])
        return fetch_many(tickers, **kwargs)

    monkeypatch.setattr(research.pipeline, "fetch_many", tracked_fetch)

    def fetch_result(provider, request):
        calls.append((request.symbol, provider.skip_missing_ohlc))
        requested_starts.append(request.start.date())
        if request.symbol == "BAD":
            raise ProviderError("synthetic Yahoo failure")
        frame = sample_ohlcv_frame.with_columns(pl.lit(request.symbol).alias("symbol"))
        return FetchResult(frame, FetchQuality(status="reported", skip_missing_ohlc=provider.skip_missing_ohlc))

    monkeypatch.setattr(YFinanceProvider, "fetch_result", fetch_result)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Fetch").run()
    app.selectbox(key="fetch-provider").set_value("yahoo").run()
    app.text_area[0].set_value("aapl, BAD, MSFT")
    app.selectbox(key="fetch-timeframe").set_value("1h")
    day = sample_ohlcv_frame["timestamp"][0].date()
    today = date.today()
    earliest = today.replace(year=today.year - 100,
                             day=min(today.day, monthrange(today.year - 100, today.month)[1]))
    assert app.date_input(key="fetch-start").min == earliest
    assert app.date_input(key="fetch-end").min == earliest
    requested_start = earliest if skip else day
    app.date_input(key="fetch-start").set_value(requested_start)
    app.date_input(key="fetch-end").set_value(day + timedelta(days=1))
    app.checkbox[0].set_value(skip)
    assert app.selectbox(key="fetch-overlap-policy").value == "Fail on overlap"
    next(button for button in app.button if button.label == "Fetch and save").click().run()
    assert not app.exception
    assert calls == [("AAPL", skip), ("BAD", skip), ("MSFT", skip)]
    assert requested_starts == [requested_start] * 3
    assert policies == ["error"]
    report = app.session_state["fetch-report"]
    assert [item.status for item in report.outcomes] == ["saved", "failed", "saved"]
    assert any("1 ticker(s) failed" in warning.value for warning in app.warning)
    assert len(research.pipeline.list_datasets()) == 2
    for outcome in (report.outcomes[0], report.outcomes[2]):
        expected = sample_ohlcv_frame.with_columns(pl.lit(outcome.ticker).alias("symbol"))
        assert_frame_equal(research.pipeline.read_dataset(outcome.dataset_ids[0]), expected)
        assert outcome.quality["skip_missing_ohlc"] is skip
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("case", ["default_overlap", "covered", "partial", "empty"])
def test_dashboard_existing_data_policy_uses_real_pipeline(
    tmp_path, monkeypatch, sample_ohlcv_frame, case,
):
    import dashboard.app as dashboard

    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    stamps = sample_ohlcv_frame["timestamp"].to_list()
    existing = sample_ohlcv_frame if case in {"covered", "default_overlap"} else sample_ohlcv_frame.tail(2)
    original = research.pipeline.ingest_frame(
        DataRequest("AAPL", existing["timestamp"][0], stamps[-1] + timedelta(hours=1), timeframe="1h"),
        existing,
    ).raw
    requests = []
    policies = []
    fetch_many = research.pipeline.fetch_many

    def tracked_fetch(tickers, **kwargs):
        policies.append(kwargs["overlap_policy"])
        return fetch_many(tickers, **kwargs)

    def fetch_result(provider, request):
        requests.append(request)
        frame = sample_ohlcv_frame.head(0) if case == "empty" else sample_ohlcv_frame
        return FetchResult(frame, FetchQuality(status="reported"))

    monkeypatch.setattr(research.pipeline, "fetch_many", tracked_fetch)
    monkeypatch.setattr(YFinanceProvider, "fetch_result", fetch_result)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Fetch").run()
    app.selectbox(key="fetch-provider").set_value("yahoo").run()
    app.selectbox(key="fetch-timeframe").set_value("1h").run()
    app.text_area[0].set_value("AAPL")
    start, end = stamps[0], stamps[-1]
    if case == "default_overlap":
        end += timedelta(hours=1)
    app.date_input(key="fetch-start").set_value(start.date())
    app.date_input(key="fetch-end").set_value(end.date())
    app.time_input(key="fetch-start-time").set_value(start.time())
    app.time_input(key="fetch-end-time").set_value(end.time())
    if case != "default_overlap":
        app.selectbox(key="fetch-overlap-policy").set_value(
            "Skip covered requests; refresh overlapping ranges"
        )
    next(button for button in app.button if button.label == "Fetch and save").click().run()
    assert not app.exception
    assert policies == ["error" if case == "default_overlap" else "refresh_overlap"]
    report = app.session_state["fetch-report"]
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    expected_status = {"default_overlap": "failed", "covered": "skipped",
                       "partial": "refreshed", "empty": "empty"}[case]
    assert outcome.status == expected_status
    assert len(research.pipeline.list_datasets()) == 1
    # Historical revision and its bytes remain readable for every outcome.
    assert_frame_equal(research.pipeline.read_dataset(original.dataset_id), existing)
    if case == "covered":
        assert requests == []
        assert outcome.row_count == 0
        assert outcome.dataset_ids == (original.dataset_id,)
        assert any("already covered; no vendor request" in item.value for item in app.info)
        assert not app.success
    else:
        assert len(requests) == 1
        assert requests[0].start == start
        if case == "partial":
            assert requests[0].end > stamps[-1]
            assert outcome.replaced_dataset_ids == (original.dataset_id,)
            assert not research.pipeline.get_metadata(original.dataset_id).active
            assert_frame_equal(research.pipeline.read_dataset(outcome.dataset_ids[0]), sample_ohlcv_frame)
            assert any("1 ticker(s) refreshed" in item.value for item in app.success)
        else:
            assert research.pipeline.get_metadata(original.dataset_id).active
            assert not app.success
            if case == "empty":
                assert any("returned no bars; no datasets were changed" in item.value for item in app.info)
            else:
                assert any("1 ticker(s) failed" in item.value for item in app.warning)


@pytest.mark.parametrize("case", ["tail", "adjacent", "covered", "multiple", "mixed_empty", "failed"])
def test_dashboard_fetches_only_missing_ranges_and_merges_existing_bars(
    tmp_path, monkeypatch, sample_ohlcv_frame, case,
):
    import dashboard.app as dashboard

    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    origin = sample_ohlcv_frame["timestamp"][0]
    stamps = [origin + timedelta(hours=index) for index in range(8)]
    frame = pl.concat([sample_ohlcv_frame.head(1)] * 7).with_columns(
        pl.Series("timestamp", stamps[:7], dtype=pl.Datetime("ms", "UTC")),
        pl.Series("volume", [1000.0 + index for index in range(7)]),
    )
    if case in {"multiple", "mixed_empty", "failed"}:
        old_frames = [frame.slice(0, 2), frame.slice(3, 2), frame.slice(6, 1)]
        start, end = stamps[0], stamps[7]
        expected_ranges = [(stamps[2], stamps[3]), (stamps[5], stamps[6])]
    elif case == "adjacent":
        old_frames = [frame.slice(0, 2)]
        start, end = stamps[2], stamps[5]
        expected_ranges = [(stamps[2], stamps[5])]
    else:
        old_frames = [frame.slice(0, 3)]
        start, end = stamps[1], stamps[3 if case == "covered" else 5]
        expected_ranges = [] if case == "covered" else [(stamps[3], stamps[5])]
    originals = [research.pipeline.ingest_frame(
        DataRequest("AAPL", old["timestamp"][0], old["timestamp"][-1] + timedelta(hours=1), timeframe="1h"),
        old,
    ).raw for old in old_frames]
    requests, events = [], []
    fetch_many = research.pipeline.fetch_many

    def tracked_fetch(tickers, **kwargs):
        assert kwargs["overlap_policy"] == "fetch_missing"
        callback = kwargs["progress"]

        def capture_progress(event):
            import copy
            events.append(copy.deepcopy(event))
            callback(event)

        kwargs["progress"] = capture_progress
        return fetch_many(tickers, **kwargs)

    def fetch_result(provider, request):
        # Exact planned ranges must reach the UI before any vendor request.
        assert events[-1]["status"] == "fetching"
        assert any(item["start"] == request.start.isoformat()
                   and item["end"] == request.end.isoformat()
                   and item["status"] == "fetching" for item in events[-1]["fetch_ranges"])
        requests.append((request.start, request.end))
        if case == "failed" and len(requests) == 2:
            raise ProviderError("second gap failed")
        result = frame.filter((pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end))
        if case == "mixed_empty" and len(requests) == 1:
            result = result.head(0)
        return FetchResult(result, FetchQuality(status="reported"))

    monkeypatch.setattr(research.pipeline, "fetch_many", tracked_fetch)
    monkeypatch.setattr(YFinanceProvider, "fetch_result", fetch_result)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Fetch").run()
    app.selectbox(key="fetch-provider").set_value("yahoo").run()
    app.selectbox(key="fetch-timeframe").set_value("1h").run()
    app.text_area[0].set_value("AAPL")
    app.selectbox(key="fetch-overlap-policy").set_value("Fetch missing ranges; merge locally")
    app.date_input(key="fetch-start").set_value(start.date())
    app.date_input(key="fetch-end").set_value(end.date())
    app.time_input(key="fetch-start-time").set_value(start.time())
    app.time_input(key="fetch-end-time").set_value(end.time())
    next(button for button in app.button if button.label == "Fetch and save").click().run()
    assert not app.exception
    assert requests == expected_ranges
    outcome = app.session_state["fetch-report"].outcomes[0]
    assert outcome.status == ("skipped" if case == "covered" else "failed" if case == "failed" else "merged")
    for original, old in zip(originals, old_frames):
        assert_frame_equal(research.pipeline.read_dataset(original.dataset_id), old)
    if case in {"covered", "failed"}:
        assert all(research.pipeline.get_metadata(item.dataset_id).active for item in originals)
        assert len(research.pipeline.list_datasets()) == len(originals)
        assert not app.success
        if case == "covered":
            assert outcome.fetched_row_count == 0
            assert any("already covered" in item.value for item in app.info)
        else:
            assert any("1 ticker(s) failed" in item.value for item in app.warning)
    else:
        expected = frame.head(7 if case in {"multiple", "mixed_empty"} else 5)
        if case == "mixed_empty":
            expected = expected.filter(pl.col("timestamp") != stamps[2])
            assert outcome.warning_message
            assert any(outcome.warning_message in item.value for item in app.warning)
        active = research.pipeline.list_datasets()
        assert len(active) == 1
        assert_frame_equal(research.pipeline.read_dataset(active[0].dataset_id), expected)
        assert outcome.row_count == expected.height
        assert outcome.fetched_row_count == expected.height - sum(old.height for old in old_frames)
        assert set(outcome.replaced_dataset_ids) == {item.dataset_id for item in originals}
        assert all(not research.pipeline.get_metadata(item.dataset_id).active for item in originals)
        assert any("1 ticker(s) merged" in item.value for item in app.success)
        assert any("Vendor request ranges" in item.value for item in app.markdown)
