"""The dashboard fetches through the pipeline and displays every ticker outcome."""

from datetime import timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import FetchQuality, FetchResult
from data_pipeline.exceptions import ProviderError
from data_pipeline.providers import YFinanceProvider
from research import Research


@pytest.mark.parametrize("skip", [False, True])
def test_dashboard_batch_fetch_uses_pipeline_and_keeps_later_successes(
    tmp_path, monkeypatch, sample_ohlcv_frame, skip,
):
    import dashboard.app as dashboard

    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    calls = []

    def fetch_result(provider, request):
        calls.append((request.symbol, provider.skip_missing_ohlc))
        if request.symbol == "BAD":
            raise ProviderError("synthetic Yahoo failure")
        frame = sample_ohlcv_frame.with_columns(pl.lit(request.symbol).alias("symbol"))
        return FetchResult(frame, FetchQuality(status="reported", skip_missing_ohlc=provider.skip_missing_ohlc))

    monkeypatch.setattr(YFinanceProvider, "fetch_result", fetch_result)
    app = AppTest.from_string("from dashboard.app import main\nmain()").run()
    app.sidebar.radio[0].set_value("Fetch").run()
    app.text_area[0].set_value("aapl, BAD, MSFT")
    app.selectbox(key="fetch-timeframe").set_value("1h")
    day = sample_ohlcv_frame["timestamp"][0].date()
    app.date_input(key="fetch-start").set_value(day)
    app.date_input(key="fetch-end").set_value(day + timedelta(days=1))
    app.checkbox[0].set_value(skip)
    next(button for button in app.button if button.label == "Fetch and save").click().run()
    assert not app.exception
    assert calls == [("AAPL", skip), ("BAD", skip), ("MSFT", skip)]
    report = app.session_state["fetch-report"]
    assert [item.status for item in report.outcomes] == ["saved", "failed", "saved"]
    assert any("1 ticker(s) failed" in warning.value for warning in app.warning)
    assert len(research.pipeline.list_datasets()) == 2
    for outcome in (report.outcomes[0], report.outcomes[2]):
        expected = sample_ohlcv_frame.with_columns(pl.lit(outcome.ticker).alias("symbol"))
        assert_frame_equal(research.pipeline.read_dataset(outcome.dataset_ids[0]), expected)
        assert outcome.quality["skip_missing_ohlc"] is skip
    assert not (tmp_path / "runs").exists()
