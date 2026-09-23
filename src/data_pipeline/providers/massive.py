"""Native, unadjusted US stock aggregates from Massive's REST API.

Intraday timestamps and extended-hours bars are preserved. Native grids are
anchored at midnight Eastern, not the regular session open. Daily timestamps
are Eastern session-date labels represented at midnight UTC. No resampling,
regular-session filtering, missing-bar filling, or price repair is performed.

https://massive.com/docs/rest/stocks/aggregates/custom-bars
"""
from __future__ import annotations

from copy import copy
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from email.utils import parsedate_to_datetime
import math
from numbers import Real
import os
import time as clock
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import polars as pl
import requests

from ..exceptions import (
    DuplicateBarError, EmptyDataError, InvalidDataRequestError,
    InvalidOHLCVError, ProviderError, SchemaValidationError, UnsupportedRequestError,
)
from ..models import DataRequest
from ..quality import FetchQuality, FetchResult
from ..schemas.ohlcv import OHLCV_SCHEMA, validate_ohlcv
from .base import BaseDataProvider

_ORIGIN = "https://api.massive.com"
_EASTERN = ZoneInfo("America/New_York")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_INTERVALS = {"1m": (1, "minute"), "2m": (2, "minute"), "5m": (5, "minute"),
              "15m": (15, "minute"), "30m": (30, "minute"), "1h": (1, "hour"),
              "90m": (90, "minute"), "1d": (1, "day")}
_FIELDS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}


class MassiveProvider(BaseDataProvider):
    """Fetch native bars; API credentials are needed only at fetch time.

    ``timeout`` bounds each HTTP wait. ``max_retries`` bounds retries per page
    for connection failures, 429 and temporary server errors. Retry delays are
    capped at 60 seconds; a longer Retry-After fails instead of retrying early.
    ``request_interval_seconds`` spaces HTTP request starts on this instance,
    including pages, retries and successive tickers. Zero disables pacing.
    Long downloads use non-overlapping Eastern-date chunks below the vendor's
    base-aggregate limit, following every page before returning any data.
    """
    name = "massive"
    supported_timeframes = frozenset(_INTERVALS)
    supported_datasets = frozenset({"ohlcv"})
    supported_price_adjustments = frozenset({"unadjusted"})

    def __init__(self, *, api_key: str | None = None, timeout: float = 10.0,
                 max_retries: int = 2, request_interval_seconds: float = 0.0) -> None:
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be a nonempty string or None.")
        if (isinstance(timeout, bool) or not isinstance(timeout, Real)
                or not math.isfinite(timeout) or not 0 < timeout <= 60):
            raise ValueError("timeout must be a finite number in (0, 60] seconds.")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer in [0, 5].")
        self._api_key = api_key.strip() if api_key is not None else None
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.request_interval_seconds = _validate_request_interval(request_interval_seconds)
        self._next_request_at: float | None = None

    def with_request_interval(self, seconds: float | None = None) -> MassiveProvider:
        """Copy configuration for one batch, with its own request clock."""
        provider = copy(self)
        provider.request_interval_seconds = _validate_request_interval(
            self.request_interval_seconds if seconds is None else seconds)
        provider._next_request_at = None
        return provider

    def fetch(self, request: DataRequest) -> pl.DataFrame:
        return self.fetch_result(request).frame

    def fetch_result(self, request: DataRequest) -> FetchResult:
        self._validate_request(request)
        key = self._api_key if self._api_key is not None else os.environ.get("MASSIVE_API_KEY", "").strip()
        if not key or any(c.isspace() for c in key) or not key.isascii():
            raise ProviderError("Set MASSIVE_API_KEY or pass a valid api_key to MassiveProvider.")
        headers = {"Authorization": f"Bearer {key}"}
        multiplier, span = _INTERVALS[request.timeframe]
        daily = request.timeframe == "1d"
        # Enclose whole Eastern days so vendor snapping has a stable origin.
        # Daily requests instead use the caller's canonical UTC date labels.
        first = request.start.date() if daily else request.start.astimezone(_EASTERN).date()
        last = request.end.date() if daily else request.end.astimezone(_EASTERN).date()
        stop = last + timedelta(days=1)
        chunk_days = 365 if daily else 30  # <50,000 minute bases, including DST and snapping.
        collected: dict[datetime, dict] = {}
        with requests.Session() as session:
            while first < stop:
                following = min(first + timedelta(days=chunk_days), stop)
                lower = datetime.combine(first, time.min, _EASTERN).astimezone(UTC)
                upper = datetime.combine(following, time.min, _EASTERN).astimezone(UTC)
                path = (f"/v2/aggs/ticker/{quote(request.symbol, safe='')}/range/"
                        f"{multiplier}/{span}/{_milliseconds(lower)}/{_milliseconds(upper) - 1}")
                url = _ORIGIN + path
                params = {"adjusted": "false", "sort": "asc", "limit": 50000}
                visited = set()
                while url:
                    if url in visited:
                        raise ProviderError("Massive returned a repeated pagination URL; download aborted.")
                    visited.add(url)
                    payload = self._get_page(session, url, params, headers)
                    records = _page_records(payload, request.symbol)
                    next_url = payload.get("next_url")
                    if not next_url and payload.get("queryCount", 0) >= 50000:
                        raise ProviderError("Massive hit the base-aggregate limit without pagination; download aborted.")
                    for record in records:
                        row = _normalize_bar(record, request.symbol, daily=daily)
                        label = row["timestamp"]
                        if label in collected and collected[label] != row:
                            raise DuplicateBarError(f"Massive {request.symbol}: conflicting bars at {label.isoformat()}.")
                        collected[label] = row
                    url = _next_page(next_url, path) if next_url is not None else None
                    params = None
                first = following
        if not collected:
            raise EmptyDataError(f"Massive returned no bars for {request.symbol!r} in the requested interval.")
        # Validate all vendor rows before applying precise canonical bounds.
        frame = validate_ohlcv(pl.DataFrame(list(collected.values()), schema=OHLCV_SCHEMA).sort("symbol", "timestamp"))
        frame = frame.filter(
            (pl.col("timestamp").cast(pl.Datetime("us", "UTC")) >= pl.lit(request.start, dtype=pl.Datetime("us", "UTC")))
            & (pl.col("timestamp").cast(pl.Datetime("us", "UTC")) < pl.lit(request.end, dtype=pl.Datetime("us", "UTC"))))
        if frame.is_empty():
            raise EmptyDataError(f"Massive returned no bars for {request.symbol!r} in the requested [start, end) interval.")
        # No synthetic omissions: Massive does not enumerate absent bars or
        # explain whether each gap represents no trades, a halt, or missing data.
        return FetchResult(frame, FetchQuality(status="unknown", skip_missing_ohlc=False,
                                               provider_version="massive-rest-v2"))

    def _validate_request(self, request):
        if not isinstance(request, DataRequest):
            raise InvalidDataRequestError("fetch expects a DataRequest.")
        if request.provider != self.name:
            raise UnsupportedRequestError(f"Massive cannot serve provider {request.provider!r}.")
        if request.dataset not in self.supported_datasets:
            raise UnsupportedRequestError(f"Massive does not support dataset {request.dataset!r}.")
        if request.price_adjustment not in self.supported_price_adjustments:
            raise UnsupportedRequestError("Massive supports only unadjusted prices.")
        if request.timeframe not in self.supported_timeframes:
            raise UnsupportedRequestError(f"Massive does not support timeframe {request.timeframe!r}.")
        if "," in request.symbol or any(c.isspace() for c in request.symbol):
            raise UnsupportedRequestError("Massive requests must contain a single ticker.")

    def _get_page(self, session, url, params, headers):
        for attempt in range(self.max_retries + 1):
            self._wait_for_request()
            try:
                response = session.get(url, params=params, headers=headers,
                                       timeout=self.timeout, allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.max_retries:
                    raise ProviderError("Massive connection failed after bounded retries.") from None
                self._defer_request(2 ** attempt)
                continue
            except requests.RequestException:
                # Vendor bodies, URLs and exception chains can contain secrets.
                raise ProviderError("Massive HTTP request failed.") from None
            try:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json(parse_float=Decimal)
                    except (ValueError, requests.RequestException):
                        raise ProviderError("Massive returned invalid JSON.") from None
                    return payload
                retryable = status in (429, 500, 502, 503, 504)
                # Keep rate-limit cooldowns even after this ticker fails, so
                # the next ticker in the batch does not immediately retry it.
                if status == 429 or (retryable and attempt < self.max_retries):
                    delay = _retry_delay(response.headers.get("Retry-After"), attempt,
                                         rate_limited=status == 429)
                    self._defer_request(delay)
                if not retryable or attempt == self.max_retries:
                    detail = {401: "invalid or missing API key", 403: "subscription or access restriction",
                              429: "rate limit exhausted"}.get(status, "request failed")
                    raise ProviderError(f"Massive HTTP {status}: {detail}.")
            finally:
                response.close()
        raise ProviderError("Massive request exhausted retries.")

    def _wait_for_request(self):
        if self._next_request_at is not None:
            delay = self._next_request_at - clock.monotonic()
            if delay > 0:
                clock.sleep(delay)
        self._next_request_at = clock.monotonic() + self.request_interval_seconds

    def _defer_request(self, delay):
        # Retry waits and pacing overlap; time spent receiving a response counts.
        self._next_request_at = max(self._next_request_at, clock.monotonic() + delay)


def _milliseconds(value: datetime) -> int:
    return (value - _EPOCH) // timedelta(milliseconds=1)


def _validate_request_interval(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError("request_interval_seconds must be a finite nonnegative number.")
    return float(value)


def _retry_delay(value, attempt, *, rate_limited=False):
    fallback = 60.0 if rate_limited else float(2 ** attempt)
    delay = fallback
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
    if not math.isfinite(delay) or delay > 60:
        raise ProviderError("Massive requested a retry delay above 60 seconds; retry this fetch later.")
    return fallback if delay < 0 else delay


def _next_page(value, path):
    if not isinstance(value, str) or not value:
        raise ProviderError("Massive returned an invalid pagination URL.")
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise ProviderError("Massive returned an invalid pagination URL.") from None
    prefix, first, last = path.rsplit("/", 2)
    parts = parsed.path.rsplit("/", 2)
    if (parsed.scheme != "https" or parsed.netloc != "api.massive.com"
            or len(parts) != 3 or parts[0] != prefix or parsed.fragment):
        raise ProviderError("Massive returned an unexpected pagination URL; download aborted.")
    try:
        # Pagination may advance the path's start as well as its cursor.
        lower = _page_bound(parts[1], end=False)
        upper = _page_bound(parts[2], end=True)
        if not int(first) <= lower <= upper <= int(last):
            raise ValueError
    except (ValueError, OverflowError):
        raise ProviderError("Massive pagination escaped the requested date chunk.") from None
    # Keep credentials in headers, and pin request semantics across pages.
    params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
              if k.lower() not in {"apikey", "adjusted", "sort", "limit"}]
    params.extend([("adjusted", "false"), ("sort", "asc"), ("limit", "50000")])
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(params), ""))


def _page_bound(value, *, end):
    if value.lstrip("-").isdigit():
        return int(value)
    date = datetime.strptime(value, "%Y-%m-%d").date()
    if end:
        date += timedelta(days=1)
    stamp = datetime.combine(date, time.min, _EASTERN).astimezone(UTC)
    return _milliseconds(stamp) - int(end)


def _page_records(payload, symbol):
    if not isinstance(payload, dict) or payload.get("status") not in ("OK", "DELAYED"):
        raise ProviderError("Massive returned an unsuccessful or malformed API response.")
    if payload.get("ticker") != symbol:
        raise SchemaValidationError("Massive response ticker does not match the requested symbol.")
    if payload.get("adjusted") is not False:
        raise SchemaValidationError("Massive response must explicitly confirm unadjusted prices.")
    records = payload.get("results", [])
    if not isinstance(records, list):
        raise SchemaValidationError("Massive results must be an array.")
    for name in ("resultsCount", "queryCount"):
        if name in payload and (type(payload[name]) is not int or payload[name] < 0):
            raise SchemaValidationError(f"Massive {name} must be a nonnegative integer.")
    if "resultsCount" in payload and payload["resultsCount"] != len(records):
        raise SchemaValidationError("Massive resultsCount does not match the returned rows.")
    return records


def _normalize_bar(record, symbol, *, daily):
    if not isinstance(record, dict) or type(record.get("t")) is not int:
        raise SchemaValidationError("Massive bar timestamps must be integer Unix milliseconds.")
    try:
        stamp = _EPOCH + timedelta(milliseconds=record["t"])
    except (OverflowError, ValueError):
        raise SchemaValidationError("Massive timestamp is out of range.") from None
    if daily:
        local = stamp.astimezone(_EASTERN)
        if local.time() != time.min:
            raise SchemaValidationError("Massive daily bars must be labeled at midnight Eastern.")
        stamp = datetime.combine(local.date(), time.min, UTC)
    row = {"timestamp": stamp, "symbol": symbol}
    for vendor, column in _FIELDS.items():
        value = record.get(vendor)
        location = f"Massive {symbol} at {stamp.isoformat()}: {column}"
        if value is None:
            raise InvalidOHLCVError(f"{location} must be finite and non-null.")
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise SchemaValidationError(f"{location} must be numeric.")
        try:
            converted = float(value)
        except (ValueError, OverflowError):
            raise SchemaValidationError(f"{location} cannot be represented as Float64.") from None
        if not math.isfinite(converted):
            raise InvalidOHLCVError(f"{location} must be finite and non-null.")
        if (isinstance(value, int) and value != converted
                or isinstance(value, Decimal) and value != Decimal(str(converted))):
            raise SchemaValidationError(f"{location} cannot be represented losslessly as Float64.")
        row[column] = converted
    return row
