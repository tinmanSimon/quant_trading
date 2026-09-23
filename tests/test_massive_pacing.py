"""Request pacing uses a fake monotonic clock, never network or real waits."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from urllib.parse import urlsplit

import pytest
import requests

from data_pipeline import DataPipeline, DataRequest, InvalidDataRequestError, fetch_many
from data_pipeline.exceptions import ProviderError
from data_pipeline.providers import MassiveProvider, ProviderRegistry
from data_pipeline.providers import massive


FIRST = datetime(2025, 7, 1, 13, 30, tzinfo=UTC)
SECRET = "test-only-key"


class Response:
    def __init__(self, payload, *, status=200, headers=None):
        self.text = json.dumps(payload)
        self.status_code = status
        self.headers = headers or {}

    def json(self, **kwargs):
        return json.loads(self.text, **kwargs)

    def close(self):
        pass


class Transport:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.calls = []
        self.pending = []

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds

    def get(self, url, **kwargs):
        call = {"at": self.now, "url": url, **kwargs}
        self.calls.append(call)
        assert self.pending, f"Unexpected HTTP request: {url}"
        # Model time spent waiting for a response as well as explicit sleeps.
        latency, item = self.pending.pop(0)
        self.now += latency
        if callable(item):
            item = item(call)
        if isinstance(item, BaseException):
            raise item
        return item


def success(call, *, paginate=False):
    path = urlsplit(call["url"]).path.split("/")
    symbol = path[4]
    lower = int(path[-2])
    stamp = datetime.fromtimestamp(lower / 1000, UTC) + timedelta(hours=13)
    payload = {"ticker": symbol, "adjusted": False, "status": "OK", "results": [
        {"t": int(stamp.timestamp() * 1000), "o": 100.0, "h": 101.0,
         "l": 99.0, "c": 100.5, "v": 1000.0},
    ]}
    if paginate:
        payload["next_url"] = call["url"] + "?cursor=second-page"
    return Response(payload)


@pytest.fixture
def transport(monkeypatch):
    result = Transport()
    monkeypatch.setattr(massive.clock, "monotonic", lambda: result.now)
    monkeypatch.setattr(massive.clock, "sleep", result.sleep)
    monkeypatch.setattr(requests.Session, "get", lambda session, url, **kw: result.get(url, **kw))
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    return result


@pytest.fixture
def request_data():
    return DataRequest("AAPL", FIRST, FIRST + timedelta(hours=6), timeframe="1h", provider="massive")


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, False, "13", None])
def test_provider_rejects_invalid_intervals(value):
    with pytest.raises(ValueError, match="request_interval_seconds"):
        MassiveProvider(request_interval_seconds=value)


@pytest.mark.parametrize("interval,latency,expected", [(13, 3, 13), (13, 20, 20), (0, 0, 0), (0.25, 0.1, 0.25)])
def test_pages_use_start_to_start_spacing_and_credit_http_time(transport, request_data, interval, latency, expected):
    transport.pending = [(latency, lambda call: success(call, paginate=True)), (0, success)]
    result = MassiveProvider(api_key=SECRET, request_interval_seconds=interval).fetch(request_data)
    assert result.height == 1  # Identical overlap still deduplicates normally.
    assert [call["at"] for call in transport.calls] == pytest.approx([0, expected])
    assert sum(transport.sleeps) == pytest.approx(max(0, interval - latency))


def test_chunks_and_direct_fetches_share_one_provider_timer(transport, request_data):
    transport.pending = [(0, success)] * 4
    provider = MassiveProvider(api_key=SECRET, request_interval_seconds=13)
    result = provider.fetch(replace(request_data, end=FIRST + timedelta(days=64)))
    assert result.height == 3
    provider.fetch(replace(request_data, symbol="MSFT"))
    assert [call["at"] for call in transport.calls] == [0, 13, 26, 39]


@pytest.mark.parametrize("failure,expected", [
    (Response({}, status=503), 13),
    (requests.Timeout("timeout"), 13),
    (Response({}, status=429, headers={"Retry-After": "20"}), 23),
    (Response({}, status=429, headers={"Retry-After": "0"}), 13),
    (Response({}, status=429), 63),
])
def test_retry_and_pacing_deadlines_use_the_longer_wait_not_the_sum(transport, request_data, failure, expected):
    transport.pending = [(3, failure), (0, success)]
    MassiveProvider(api_key=SECRET, request_interval_seconds=13).fetch(request_data)
    assert [call["at"] for call in transport.calls] == [0, expected]
    assert sum(transport.sleeps) == expected - 3


@pytest.mark.parametrize("retry_after", [None, "", "invalid", "-1"])
def test_invalid_or_missing_429_delay_waits_a_full_minute(transport, request_data, retry_after):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    transport.pending = [(0, Response({}, status=429, headers=headers)), (0, success)]
    MassiveProvider(api_key=SECRET).fetch(request_data)
    assert [call["at"] for call in transport.calls] == [0, 60]


def test_429_retries_still_stop_at_the_configured_limit(transport, request_data):
    transport.pending = [(0, Response({}, status=429))] * 3
    with pytest.raises(ProviderError, match="rate limit"):
        MassiveProvider(api_key=SECRET, max_retries=2, request_interval_seconds=13).fetch(request_data)
    assert [call["at"] for call in transport.calls] == [0, 60, 120]
    assert transport.sleeps == [60, 60]


@pytest.mark.parametrize("retry_after", ["61", "nan", "inf"])
def test_excessive_retry_after_fails_instead_of_retrying_early(transport, request_data, retry_after):
    transport.pending = [(0, Response({}, status=429, headers={"Retry-After": retry_after}))]
    with pytest.raises(ProviderError, match="60 seconds"):
        MassiveProvider(api_key=SECRET, request_interval_seconds=13).fetch(request_data)
    assert len(transport.calls) == 1
    assert transport.sleeps == []


@pytest.mark.parametrize("entrypoint", ["pipeline", "standalone"])
def test_batch_paces_across_failed_tickers_without_mutating_registered_configuration(
    transport, request_data, tmp_path, entrypoint,
):
    original = MassiveProvider(api_key=SECRET, timeout=3.5, max_retries=0, request_interval_seconds=29)
    registry = ProviderRegistry({"massive": original})
    pipeline = DataPipeline(tmp_path / "data", providers=registry)
    transport.pending = [(0, success), (0, Response({}, status=503)), (0, success)]
    kwargs = dict(start=request_data.start, end=request_data.end, timeframe="1h", provider="massive",
                  massive_request_interval_seconds=13)
    if entrypoint == "pipeline":
        report = pipeline.fetch_many(["AAPL", "BAD", "MSFT"], **kwargs)
    else:
        report = fetch_many(pipeline, tickers=["AAPL", "BAD", "MSFT"], **kwargs)
    assert [outcome.status for outcome in report.outcomes] == ["saved", "failed", "saved"]
    assert [call["at"] for call in transport.calls] == [0, 13, 26]
    assert all(call["timeout"] == 3.5 for call in transport.calls)
    assert all(call["headers"]["Authorization"] == f"Bearer {SECRET}" for call in transport.calls)
    assert pipeline.providers is registry and registry.get("massive") is original
    assert original.request_interval_seconds == 29
    assert len(pipeline.list_datasets()) == 2
    # The registered instance has not been fetched and retains an independent clock.
    transport.pending = [(0, success)]
    original.fetch(replace(request_data, symbol="IBM"))
    assert transport.calls[-1]["at"] == 26


@pytest.mark.parametrize("override,expected_interval", [(13, 13), (None, 29)])
def test_factory_is_resolved_once_and_one_configured_provider_serves_the_batch(
    transport, request_data, tmp_path, override, expected_interval,
):
    constructed, fetched = [], []

    class ConfiguredProvider(MassiveProvider):
        def __init__(self):
            super().__init__(api_key=SECRET, request_interval_seconds=29)
            self.custom_setting = "preserved"
            constructed.append(self)

        def fetch_result(self, request):
            assert self.custom_setting == "preserved"
            fetched.append(self)
            return super().fetch_result(request)

    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({"massive": ConfiguredProvider}))
    transport.pending = [(0, success), (0, success)]
    report = pipeline.fetch_many(["AAPL", "MSFT"], start=request_data.start, end=request_data.end,
                                 timeframe="1h", provider="massive", massive_request_interval_seconds=override)
    assert report.ok
    assert len(constructed) == 1
    assert len(fetched) == 2 and fetched[0] is fetched[1]
    assert constructed[0].request_interval_seconds == 29
    assert [call["at"] for call in transport.calls] == [0, expected_interval]


def test_omitted_batch_override_preserves_configured_provider_pacing(transport, request_data, tmp_path):
    provider = MassiveProvider(api_key=SECRET, request_interval_seconds=29)
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({"massive": provider}))
    transport.pending = [(0, success), (0, success)]
    report = pipeline.fetch_many(["AAPL", "MSFT"], start=request_data.start, end=request_data.end,
                                 timeframe="1h", provider="massive")
    assert report.ok
    assert [call["at"] for call in transport.calls] == [0, 29]


def test_exhausted_rate_limit_wait_is_retained_for_the_next_ticker(transport, request_data, tmp_path):
    provider = MassiveProvider(api_key=SECRET, max_retries=1)
    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({"massive": provider}))
    transport.pending = [(0, Response({}, status=429)), (0, Response({}, status=429)), (0, success)]
    report = pipeline.fetch_many(["BAD", "MSFT"], start=request_data.start, end=request_data.end,
                                 timeframe="1h", provider="massive", massive_request_interval_seconds=13)
    assert [outcome.status for outcome in report.outcomes] == ["failed", "saved"]
    assert [call["at"] for call in transport.calls] == [0, 60, 120]
    assert len(pipeline.list_datasets()) == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, False, "13"])
def test_invalid_batch_interval_fails_before_provider_construction_or_download(transport, request_data, tmp_path, value):
    def factory():
        pytest.fail("Invalid settings should not construct a provider")

    pipeline = DataPipeline(tmp_path / "data", providers=ProviderRegistry({"massive": factory}))
    with pytest.raises(InvalidDataRequestError):
        pipeline.fetch_many(["AAPL"], start=request_data.start, end=request_data.end,
                            provider="massive", massive_request_interval_seconds=value)
    assert transport.calls == []
    assert pipeline.list_datasets() == []


@pytest.mark.parametrize("value", [0, 13])
def test_massive_option_cannot_be_silently_ignored_by_other_providers(transport, request_data, tmp_path, value):
    pipeline = DataPipeline(tmp_path / "data")
    with pytest.raises(InvalidDataRequestError, match="Massive-specific"):
        pipeline.fetch_many(["AAPL"], start=request_data.start, end=request_data.end,
                            provider="yahoo", massive_request_interval_seconds=value)
    assert transport.calls == []
