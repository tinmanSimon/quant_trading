"""Massive's historical adapter, exercised entirely with recorded-style JSON."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from urllib.parse import parse_qs, urlsplit

import polars as pl
from polars.testing import assert_frame_equal
import pytest
import requests

from data_pipeline import DataPipeline, DataQuery, DataRequest
from data_pipeline.exceptions import (
    DataPipelineError, DuplicateBarError, EmptyDataError, ProviderError,
    UnsupportedRequestError,
)
from data_pipeline.providers import MassiveProvider, ProviderRegistry, get_provider
from data_pipeline.schemas.ohlcv import OHLCV_SCHEMA


API_KEY = "test-only-massive-secret"
FIRST = datetime(2025, 7, 1, 13, 30, tzinfo=UTC)


def milliseconds(stamp):
    return int(stamp.timestamp() * 1000)


def bar(stamp=FIRST, **overrides):
    return {"t": milliseconds(stamp), "o": 100.125, "h": 102.375,
            "l": 99.0625, "c": 101.25, "v": 1000.125, **overrides}


def payload(results=None, **overrides):
    return {"ticker": "AAPL", "adjusted": False, "status": "OK",
            "results": [bar()] if results is None else results, **overrides}


class Response:
    def __init__(self, data=None, *, status=200, headers=None, raw=None):
        self.status_code = status
        self.headers = headers or {}
        self.text = json.dumps(data) if raw is None else raw
        self.closed = False

    def json(self, **kwargs):
        return json.loads(self.text, **kwargs)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def request_data():
    return DataRequest("AAPL", FIRST, FIRST + timedelta(minutes=30),
                       timeframe="15m", provider="massive")


@pytest.fixture
def http(monkeypatch):
    """Unexpected extra requests fail locally rather than using the network."""
    pending, calls = [], []

    def get(session, url, **kwargs):
        calls.append({"url": url, "headers": {**session.headers, **kwargs.get("headers", {})},
                      **{key: value for key, value in kwargs.items() if key != "headers"}})
        assert pending, f"Unexpected Massive request: {url}"
        item = pending.pop(0)
        if callable(item):
            item = item(calls[-1])
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, Response) else Response(item)

    monkeypatch.setattr(requests.Session, "get", get)
    # Retry tests exercise the policy without sleeping.
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    return pending, calls


def test_registered_provider_is_lazy_and_missing_credentials_fail_only_on_fetch(http, request_data):
    provider = get_provider("massive")
    assert isinstance(provider, MassiveProvider)
    with pytest.raises(ProviderError, match="MASSIVE_API_KEY|API key"):
        provider.fetch(request_data)
    assert http[1] == []


@pytest.mark.parametrize("key_source", ["constructor", "environment"])
def test_native_values_and_unknown_quality_are_preserved(http, monkeypatch, request_data, key_source):
    pending, calls = http
    pending.append(payload([bar(FIRST + timedelta(minutes=15), c=102.0), bar()]))
    if key_source == "environment":
        provider = MassiveProvider(timeout=3.5)
        monkeypatch.setenv("MASSIVE_API_KEY", API_KEY)
    else:
        provider = MassiveProvider(api_key=API_KEY, timeout=3.5)
    fetched = provider.fetch_result(request_data)
    assert fetched.frame.schema == OHLCV_SCHEMA
    assert fetched.frame["timestamp"].to_list() == [FIRST, FIRST + timedelta(minutes=15)]
    assert fetched.frame["symbol"].to_list() == ["AAPL", "AAPL"]
    assert fetched.frame["open"].to_list() == [100.125, 100.125]
    assert fetched.frame["high"].to_list() == [102.375, 102.375]
    assert fetched.frame["low"].to_list() == [99.0625, 99.0625]
    assert fetched.frame["close"].to_list() == [101.25, 102.0]
    assert fetched.frame["volume"].to_list() == [1000.125, 1000.125]
    assert fetched.quality.status == "unknown"
    assert fetched.quality.omitted_bars == ()
    assert fetched.quality.provider_version == "massive-rest-v2"
    assert len(calls) == 1
    call = calls[0]
    assert urlsplit(call["url"]).hostname == "api.massive.com"
    assert "/v2/aggs/ticker/AAPL/range/15/minute/" in call["url"]
    assert call["params"]["adjusted"] == "false"
    assert call["params"]["sort"] == "asc"
    assert call["params"]["limit"] == 50000
    assert call["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert call["timeout"] == 3.5
    assert API_KEY not in call["url"]
    assert API_KEY not in str(call["params"])


def test_constructor_credentials_override_environment(http, monkeypatch, request_data):
    http[0].append(payload())
    monkeypatch.setenv("MASSIVE_API_KEY", "other-secret")
    MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert http[1][0]["headers"]["Authorization"] == f"Bearer {API_KEY}"


@pytest.mark.parametrize("timeframe,multiplier,timespan", [
    ("1m", 1, "minute"), ("2m", 2, "minute"), ("5m", 5, "minute"),
    ("15m", 15, "minute"), ("30m", 30, "minute"), ("1h", 1, "hour"),
    ("60m", 1, "hour"), ("90m", 90, "minute"),
])
def test_timeframes_use_native_intervals_without_relabeling(http, request_data, timeframe, multiplier, timespan):
    native_stamp = FIRST.replace(minute=0)
    request_data = replace(request_data, timeframe=timeframe, start=native_stamp,
                           end=native_stamp + timedelta(hours=2))
    http[0].append(payload([bar(native_stamp)]))
    frame = MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert f"/range/{multiplier}/{timespan}/" in http[1][0]["url"]
    assert frame["timestamp"].to_list() == [native_stamp]


@pytest.mark.parametrize("timeframe", ["13m", "1wk", "1mo"])
def test_unsupported_intervals_do_not_make_requests(http, request_data, timeframe):
    with pytest.raises(UnsupportedRequestError):
        MassiveProvider(api_key=API_KEY).fetch(replace(request_data, timeframe=timeframe))
    assert not http[1]


def test_request_for_different_provider_is_rejected(http, request_data):
    with pytest.raises(UnsupportedRequestError):
        MassiveProvider(api_key=API_KEY).fetch(replace(request_data, provider="yahoo"))
    assert not http[1]


def test_exact_microsecond_bounds_filter_without_rounding_query(http, request_data):
    second = FIRST + timedelta(minutes=15)
    http[0].append(payload([bar(FIRST), bar(second), bar(second + timedelta(milliseconds=1))]))
    request_data = replace(request_data, start=FIRST + timedelta(microseconds=1),
                           end=second + timedelta(microseconds=1))
    result = MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert result["timestamp"].to_list() == [second]


@pytest.mark.parametrize("native,label", [
    (datetime(2025, 3, 7, 5, tzinfo=UTC), datetime(2025, 3, 7, tzinfo=UTC)),
    (datetime(2025, 3, 10, 4, tzinfo=UTC), datetime(2025, 3, 10, tzinfo=UTC)),
])
def test_daily_labels_use_new_york_session_date_across_dst(http, request_data, native, label):
    http[0].append(payload([bar(native)]))
    request_data = replace(request_data, timeframe="1d", start=label, end=label + timedelta(days=1))
    result = MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert result["timestamp"].to_list() == [label]
    assert "/range/1/day/" in http[1][0]["url"]
    # The API request must reach local midnight, rather than truncate at UTC midnight.
    api_from, api_to = map(int, urlsplit(http[1][0]["url"]).path.split("/")[-2:])
    assert api_from <= milliseconds(native) <= api_to


def test_extended_hours_and_real_zero_volume_are_not_silently_dropped(http, request_data):
    premarket = FIRST.replace(hour=8, minute=0)
    http[0].append(payload([bar(premarket, v=0), bar(FIRST)]))
    result = MassiveProvider(api_key=API_KEY).fetch(replace(request_data, start=premarket))
    assert result["timestamp"].to_list() == [premarket, FIRST]
    assert result["volume"].to_list() == [0.0, 1000.125]


@pytest.mark.parametrize("status", ["OK", "DELAYED"])
def test_success_statuses_are_accepted(http, request_data, status):
    http[0].append(payload(status=status))
    assert MassiveProvider(api_key=API_KEY).fetch(request_data).height == 1


@pytest.mark.parametrize("changes", [
    {"status": "ERROR", "error": "bad request"},
    {"status": "NOT_AUTHORIZED"}, {"ticker": "MSFT"}, {"adjusted": True},
    {"results": "not-a-list"},
])
def test_invalid_vendor_envelope_is_rejected(http, request_data, changes):
    http[0].append(payload(**changes))
    with pytest.raises(DataPipelineError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


@pytest.mark.parametrize("column", ["o", "h", "l", "c", "v"])
@pytest.mark.parametrize("missing", [True, False])
def test_missing_numeric_values_are_rejected_never_filled(http, request_data, column, missing):
    record = bar(**{column: None})
    if missing:
        record.pop(column)
    http[0].append(payload([record]))
    with pytest.raises(DataPipelineError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


@pytest.mark.parametrize("changes", [
    {"v": -1}, {"o": float("inf")}, {"c": float("nan")}, {"v": True},
    {"h": 98}, {"l": 103}, {"t": None}, {"t": True},
    {"t": milliseconds(FIRST) + 0.5}, {"t": "1751376600000"},
    {"v": 9007199254740993},
])
def test_invalid_values_and_lossy_integer_conversion_are_rejected(http, request_data, changes):
    http[0].append(payload([bar(**changes)]))
    with pytest.raises(DataPipelineError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


def test_decimal_json_precision_is_not_silently_rounded(http, request_data):
    raw = json.dumps(payload()).replace('"o": 100.125', '"o": 100.1234567890123456789')
    http[0].append(Response(raw=raw))
    with pytest.raises(DataPipelineError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


@pytest.mark.parametrize("results", [[], [bar(FIRST - timedelta(days=1))]])
def test_empty_selected_range_raises(http, request_data, results):
    http[0].append(payload(results))
    with pytest.raises(EmptyDataError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


def next_page():
    start = FIRST.replace(hour=4, minute=0)
    lower = milliseconds(start)
    upper = milliseconds(start + timedelta(days=1)) - 1
    return f"https://api.massive.com/v2/aggs/ticker/AAPL/range/15/minute/{lower}/{upper}?cursor=next"


def test_all_pages_are_collected_sorted_and_identical_overlap_deduplicated(http, request_data):
    second = FIRST + timedelta(minutes=15)
    http[0].extend([payload([bar()], next_url=next_page()),
                    payload([bar(second), bar()])])
    result = MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert result["timestamp"].to_list() == [FIRST, second]
    assert len(http[1]) == 2
    assert urlsplit(http[1][1]["url"]).path == urlsplit(next_page()).path
    assert parse_qs(urlsplit(http[1][1]["url"]).query)["cursor"] == ["next"]
    assert http[1][1]["headers"]["Authorization"] == f"Bearer {API_KEY}"


def test_conflicting_duplicate_values_fail_instead_of_choosing_one(http, request_data):
    http[0].extend([payload(next_url=next_page()), payload([bar(c=102.0)])])
    with pytest.raises(DuplicateBarError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


def test_pagination_can_advance_start_but_cannot_change_semantics_or_leak_query_key(http, request_data):
    second = FIRST + timedelta(minutes=15)
    upper = milliseconds(FIRST.replace(hour=4, minute=0) + timedelta(days=1)) - 1
    next_url = (f"https://api.massive.com/v2/aggs/ticker/AAPL/range/15/minute/"
                f"{milliseconds(second)}/{upper}?cursor=next&apiKey=discard-me&adjusted=true&sort=desc&limit=1")
    http[0].extend([payload(next_url=next_url), payload([bar(second)])])
    result = MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert result["timestamp"].to_list() == [FIRST, second]
    parsed = urlsplit(http[1][1]["url"])
    query = parse_qs(parsed.query)
    assert query["adjusted"] == ["false"]
    assert query["sort"] == ["asc"]
    assert query["limit"] == ["50000"]
    assert "apikey" not in {key.lower() for key in query}
    assert "discard-me" not in http[1][1]["url"]
    assert parsed.path.split("/")[-2] == str(milliseconds(second))


@pytest.mark.parametrize("timeframe,start,end,maximum_days", [
    ("15m", datetime(2025, 2, 20, tzinfo=UTC), datetime(2025, 4, 2, tzinfo=UTC), 14),
    ("1d", datetime(2023, 1, 1, tzinfo=UTC), datetime(2025, 3, 12, tzinfo=UTC), 365),
])
def test_long_requests_use_bounded_contiguous_chunks(http, request_data, timeframe, start, end, maximum_days):
    produced = []

    def page(call):
        lower, upper = map(int, urlsplit(call["url"]).path.split("/")[-2:])
        # Stay on a native minute boundary, or local midnight for daily bars.
        stamp = datetime.fromtimestamp(lower / 1000, UTC)
        if timeframe != "1d":
            stamp += timedelta(hours=13)
        produced.append(stamp)
        return payload([bar(stamp)])

    http[0].extend([page] * 10)
    result = MassiveProvider(api_key=API_KEY).fetch(
        replace(request_data, start=start, end=end, timeframe=timeframe))
    assert len(http[1]) > 1
    ranges = [tuple(map(int, urlsplit(call["url"]).path.split("/")[-2:])) for call in http[1]]
    for lower, upper in ranges:
        # A daylight-saving transition can add an hour to a calendar-day chunk.
        assert upper - lower + 1 <= (maximum_days * 24 + 1) * 3600 * 1000
    assert all(before[1] + 1 == after[0] for before, after in zip(ranges, ranges[1:]))
    assert result.height > 1
    assert all(start <= stamp < end for stamp in result["timestamp"])


@pytest.mark.parametrize("changes", [
    {"queryCount": 50000}, {"resultsCount": 2}, {"queryCount": -1},
    {"queryCount": "1"}, {"resultsCount": True},
])
def test_unexplained_truncation_and_inconsistent_counts_fail(http, request_data, changes):
    http[0].append(payload(**changes))
    with pytest.raises(DataPipelineError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


@pytest.mark.parametrize("url", [
    "https://evil.example/v2/aggs/ticker/AAPL/range/15/minute/0/1?cursor=next",
    "http://api.massive.com/v2/aggs/ticker/AAPL/range/15/minute/0/1?cursor=next",
    "https://api.massive.com@evil.example/v2/aggs/ticker/AAPL/range/15/minute/0/1",
    "https://api.massive.com/v2/aggs/ticker/MSFT/range/15/minute/0/1?cursor=next",
])
def test_untrusted_pagination_is_rejected_before_sending_credentials(http, request_data, url):
    http[0].append(payload(next_url=url))
    with pytest.raises(ProviderError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert len(http[1]) == 1


def test_repeating_page_cursor_is_rejected(http, request_data):
    http[0].extend([payload(next_url=next_page()), payload(next_url=next_page())])
    with pytest.raises(ProviderError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)
    assert len(http[1]) == 2


@pytest.mark.parametrize("failure", [
    Response({}, status=429, headers={"Retry-After": "0"}),
    Response({}, status=503), requests.Timeout("timed out"),
])
def test_transient_errors_retry_then_return_complete_data(http, request_data, failure):
    http[0].extend([failure, payload()])
    assert MassiveProvider(api_key=API_KEY, max_retries=1).fetch(request_data).height == 1
    assert len(http[1]) == 2


def test_retries_are_bounded_and_credentials_are_not_in_the_error(http, request_data):
    http[0].extend([requests.Timeout(f"secret={API_KEY}")] * 3)
    with pytest.raises(ProviderError) as raised:
        MassiveProvider(api_key=API_KEY, max_retries=2).fetch(request_data)
    assert len(http[1]) == 3
    assert API_KEY not in str(raised.value)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_http_errors_fail_without_retry_or_echoing_vendor_secrets(http, request_data, status):
    http[0].append(Response({"error": API_KEY}, status=status))
    with pytest.raises(ProviderError) as raised:
        MassiveProvider(api_key=API_KEY, max_retries=2).fetch(request_data)
    assert len(http[1]) == 1
    assert API_KEY not in str(raised.value)


def test_bad_json_is_a_provider_error(http, request_data):
    http[0].append(Response(raw="<html>unavailable</html>"))
    with pytest.raises(ProviderError):
        MassiveProvider(api_key=API_KEY).fetch(request_data)


def test_partial_download_failure_does_not_write_dataset(http, request_data, tmp_path):
    http[0].extend([payload(next_url=next_page()), Response({}, status=403)])
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "massive": MassiveProvider(api_key=API_KEY),
    }))
    with pytest.raises(ProviderError):
        pipeline.ingest(request_data)
    assert len(http[1]) == 2
    assert pipeline.list_datasets() == []
    assert not list((tmp_path / "data").rglob("*.parquet"))


def test_pipeline_batch_continues_and_reopened_storage_preserves_provider_and_values(http, request_data, tmp_path):
    http[0].extend([payload(), Response({}, status=404), payload(ticker="MSFT")])
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "massive": MassiveProvider(api_key=API_KEY),
    }))
    report = pipeline.fetch_many(["AAPL", "BAD", "MSFT"], provider="massive",
                                 start=request_data.start, end=request_data.end, timeframe="15m")
    assert [outcome.status for outcome in report.outcomes] == ["saved", "failed", "saved"]
    assert report.outcomes[1].error_type == "ProviderError"
    reopened = DataPipeline(tmp_path / "data")
    for outcome in (report.outcomes[0], report.outcomes[2]):
        metadata = reopened.get_metadata(outcome.dataset_ids[0])
        assert metadata.request.provider == "massive"
        assert metadata.quality.status == "unknown"
        assert metadata.quality.provider_version == "massive-rest-v2"
        expected = pl.DataFrame({"timestamp": [FIRST], "symbol": [outcome.ticker],
                                 "open": [100.125], "high": [102.375], "low": [99.0625],
                                 "close": [101.25], "volume": [1000.125]}, schema=OHLCV_SCHEMA)
        assert_frame_equal(reopened.read_dataset(metadata.dataset_id), expected)
        assert_frame_equal(reopened.read(DataQuery(provider="massive", symbol=outcome.ticker)), expected)
    # The same symbol from another provider must remain independently queryable.
    aapl = reopened.read(DataQuery(provider="massive", symbol="AAPL"))
    yahoo = aapl.with_columns(pl.lit(101.5).alias("close"))
    reopened.ingest_frame(replace(request_data, provider="yahoo"), yahoo)
    assert_frame_equal(reopened.read(DataQuery(provider="massive", symbol="AAPL")), aapl)
    assert_frame_equal(reopened.read(DataQuery(provider="yahoo", symbol="AAPL")), yahoo)


def test_native_15_minute_data_can_pass_calendar_preflight(http, request_data, tmp_path):
    from research.datasets import prepare_snapshot

    http[0].append(payload([bar(FIRST + timedelta(minutes=15 * i)) for i in range(3)]))
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "massive": MassiveProvider(api_key=API_KEY),
    }))
    end = FIRST + timedelta(minutes=45)
    pipeline.ingest(replace(request_data, end=end))
    snapshot = prepare_snapshot(pipeline, tickers=["AAPL"], provider="massive", timeframe="15m",
                                start=FIRST + timedelta(minutes=15), end=end, lookback=1)
    assert snapshot.report.ok
    assert snapshot.frames["AAPL"].height == 3
    assert [bar.timestamp for bar in snapshot.bars["AAPL"]] == snapshot.frames["AAPL"]["timestamp"].to_list()
    assert len(http[1]) == 1  # Preflight reads local data and never downloads it again.


def test_native_hour_bars_are_not_mistaken_for_session_aligned_bars(http, request_data, tmp_path):
    from research.datasets import prepare_snapshot
    from research.errors import PreflightError

    native = FIRST.replace(minute=0)
    http[0].append(payload([bar(native + timedelta(hours=i)) for i in range(3)]))
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({
        "massive": MassiveProvider(api_key=API_KEY),
    }))
    pipeline.ingest(replace(request_data, timeframe="1h", start=native, end=native + timedelta(hours=3)))
    with pytest.raises(PreflightError) as raised:
        prepare_snapshot(pipeline, tickers=["AAPL"], provider="massive", timeframe="1h",
                         start=FIRST, end=FIRST + timedelta(hours=2), lookback=0)
    codes = {issue.code for issue in raised.value.report.issues}
    assert {"missing_bars", "unexpected_bars"} <= codes


@pytest.mark.parametrize("missing_index", [None, 0, 2])
def test_massive_daily_backtest_uses_prior_days_and_still_requires_complete_data(http, tmp_path, missing_index):
    from research import Research
    from research.errors import PreflightError

    days = [datetime(2025, 6, 30, tzinfo=UTC),
            *[datetime(2025, 7, day, tzinfo=UTC) for day in (1, 2, 3)]]
    # Native daily labels are midnight Eastern (04:00 UTC in summer).
    records = [bar(day + timedelta(hours=4), c=100.25 + index * 0.5)
               for index, day in enumerate(days) if index != missing_index]
    http[0].append(payload(records))
    research = Research(tmp_path / "data", tmp_path / "runs", providers=ProviderRegistry({
        "massive": MassiveProvider(api_key=API_KEY),
    }))
    end = datetime(2025, 7, 4, tzinfo=UTC)
    stored = research.pipeline.ingest(DataRequest(
        "AAPL", days[0], end, timeframe="1d", provider="massive"))
    original = research.pipeline.read_dataset(stored.raw.dataset_id)
    kwargs = dict(provider="massive", timeframe="1d", start=days[2], end=end,
                  strategies=[{"name": "momentum", "version": "1", "config": {"lookback": 1}}])
    if missing_index is not None:
        with pytest.raises(PreflightError) as raised:
            research.backtest(["AAPL"], **kwargs)
        assert any(issue.code == "missing_bars" and days[missing_index] in issue.timestamps
                   for issue in raised.value.report.issues)
        assert not research.list_runs()
    else:
        run = research.backtest(["AAPL"], **kwargs)
        assert run.manifest["provider"] == "massive"
        assert run.manifest["timeframe"] == "1d"
        trades = run.results[0].trades
        assert trades.height > 0
        assert trades["decision_bar_timestamp"][0] == days[1]
        assert trades["bar_timestamp"][0] == days[2]
        assert trades["timestamp"][0] == days[2] + timedelta(hours=13, minutes=30)
        assert research.load_run(run.run_id).results[0].symbol == "AAPL"
    assert_frame_equal(research.pipeline.read_dataset(stored.raw.dataset_id), original)
    assert len(http[1]) == 1  # Backtesting uses only local data.
