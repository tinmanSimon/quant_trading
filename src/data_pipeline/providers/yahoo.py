"""Yahoo OHLCV adapter with source-volume validation before yfinance cleanup.

Daily and longer bars carry *session-date labels*: exchange-local YYYY-MM-DD
is represented at 00:00:00 UTC, regardless of the exchange's UTC offset. This
is a date convention, not the actual session opening instant. Intraday bars
instead retain their actual instant and are converted from the vendor's aware
index to UTC. Naive intraday timestamps are ambiguous and are rejected.

``unadjusted`` means Yahoo's supplied Open/High/Low/Close, with yfinance's
auto-adjust, back-adjust and repair disabled. It does not undo any historical
split treatment already present in Yahoo's source data. Adj Close is unused.

API documentation: https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, time, timedelta
import math
from numbers import Real
import warnings

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)
import polars as pl
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError

from ..exceptions import (
    EmptyDataError,
    InvalidDataRequestError,
    InvalidOHLCVError,
    ProviderError,
    SchemaValidationError,
    UnsupportedRequestError,
)
from ..models import DataRequest
from ..schemas.ohlcv import validate_ohlcv
from .base import BaseDataProvider


_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_DATE_INTERVALS = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})


class YFinanceProvider(BaseDataProvider):
    """Single-symbol Yahoo provider with explicit interval capabilities.

    ``30m`` is intentionally unsupported: yfinance fetches 15-minute data and
    resamples it internally. Other unsupported intervals are also rejected;
    this adapter never substitutes a timeframe, vendor, or adjustment mode.
    Vendor retention/range failures surface as errors without retrying a
    shorter window. ``timeout`` bounds each download response wait (seconds),
    not the total fetch duration; yfinance also performs metadata/cookie calls
    using its own finite timeouts. Downloads run without worker threads and
    this adapter adds no retries. Opt-in ``skip_missing_ohlc`` drops bars whose
    four OHLC prices are all missing, warning with every omitted timestamp.
    Other invalid values still fail validation.
    """

    name = "yahoo"
    supported_timeframes = frozenset(
        {"1m", "2m", "5m", "15m", "60m", "90m", "1h", *_DATE_INTERVALS}
    )
    supported_datasets = frozenset({"ohlcv"})
    supported_price_adjustments = frozenset({"unadjusted"})

    def __init__(self, *, timeout: float = 10.0, skip_missing_ohlc: bool = False) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 60
        ):
            raise ValueError("timeout must be a finite number in (0, 60] seconds.")
        self.timeout = float(timeout)
        if not isinstance(skip_missing_ohlc, bool):
            raise ValueError("skip_missing_ohlc must be a boolean.")
        self.skip_missing_ohlc = skip_missing_ohlc

    def fetch(self, request: DataRequest) -> pl.DataFrame:
        self._validate_request(request)
        date_labels = request.timeframe in _DATE_INTERVALS
        start, end = _download_bounds(request, date_labels=date_labels)
        try:
            downloaded = _download_history(
                symbol=request.symbol,
                start=start,
                end=end,
                interval=request.timeframe,
                timeout=self.timeout,
                skip_missing_ohlc=self.skip_missing_ohlc,
            )
        except (InvalidOHLCVError, SchemaValidationError, EmptyDataError):
            raise
        except YFPricesMissingError as exc:
            raise EmptyDataError(f"Yahoo returned no data for {request.symbol!r}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(
                f"Yahoo download failed for {request.symbol!r} "
                f"at interval {request.timeframe!r}."
            ) from exc

        if downloaded is None or (
            isinstance(downloaded, pd.DataFrame) and downloaded.empty
        ):
            raise EmptyDataError(
                f"Yahoo returned no data for {request.symbol!r} "
                f"in [{request.start.isoformat()}, {request.end.isoformat()}); "
                "the symbol/range may be unavailable or the vendor request failed."
            )
        if not isinstance(downloaded, pd.DataFrame):
            raise ProviderError("Yahoo download did not return a pandas DataFrame.")

        canonical = _normalize(downloaded, request.symbol, date_labels=date_labels)
        if self.skip_missing_ohlc:
            missing_ohlc = pl.all_horizontal(
                [pl.col(column).is_null() | pl.col(column).is_nan()
                 for column in ("open", "high", "low", "close")]
            )
            omitted = canonical.filter(missing_ohlc)
            if not omitted.is_empty():
                timestamps = ", ".join(
                    timestamp.isoformat() for timestamp in omitted["timestamp"].to_list()
                )
                warnings.warn(
                    f"Yahoo {request.symbol}: skipped {omitted.height} bars with all OHLC "
                    f"prices missing (UTC): {timestamps}. Returned data has gaps.",
                    UserWarning,
                    stacklevel=2,
                )
                canonical = canonical.filter(~missing_ohlc)
        # Validate remaining bars before slicing; other invalid values must
        # not be hidden by range filtering. Empty results still raise an error.
        canonical = validate_ohlcv(canonical.sort(["symbol", "timestamp"]))
        canonical = canonical.filter(
            (pl.col("timestamp") >= request.start)
            & (pl.col("timestamp") < request.end)
        )
        if canonical.is_empty():
            raise EmptyDataError(
                f"Yahoo returned no bars in the requested [start, end) interval "
                f"for {request.symbol!r}."
            )
        return canonical

    def _validate_request(self, request: DataRequest) -> None:
        if not isinstance(request, DataRequest):
            raise InvalidDataRequestError("fetch expects a DataRequest.")
        if request.provider != self.name:
            raise UnsupportedRequestError(
                f"Yahoo cannot serve provider {request.provider!r}."
            )
        if request.dataset not in self.supported_datasets:
            raise UnsupportedRequestError(
                f"Yahoo does not support dataset {request.dataset!r}."
            )
        if request.price_adjustment not in self.supported_price_adjustments:
            raise UnsupportedRequestError("Yahoo supports only unadjusted prices.")
        if request.timeframe not in self.supported_timeframes:
            detail = (
                " yfinance resamples 15m bars for 30m requests."
                if request.timeframe == "30m"
                else ""
            )
            raise UnsupportedRequestError(
                f"Yahoo does not support timeframe {request.timeframe!r}." + detail
            )
        if "," in request.symbol or any(c.isspace() for c in request.symbol):
            raise UnsupportedRequestError("Yahoo requests must contain a single ticker.")


def _download_history(*, symbol, start, end, interval, timeout, skip_missing_ohlc):
    # yfinance 1.7.0 fills null volume with zero even with repair=False and
    # keepna=True. Check the same HTTP response before history() can do so.
    # PriceHistory is private to this Ticker; never patch yfinance globals or
    # mutate its shared YfData transport. These private hooks are covered by
    # raw-response tests and must be rechecked when upgrading pinned yfinance.
    history = yf.Ticker(symbol)._lazy_load_price_history()
    history._data = _VolumeCheckedData(history._data, symbol, skip_missing_ohlc)
    # Retain per-call exception handling instead of changing yfinance's global
    # hide_exceptions setting. The pinned release still supports this argument.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="'raise_errors' deprecated", category=DeprecationWarning,
        )
        return history.history(
            start=start, end=end, interval=interval, timeout=timeout,
            auto_adjust=False, back_adjust=False, repair=False, actions=False,
            keepna=True, rounding=False, prepost=False, raise_errors=True,
        )


class _VolumeCheckedData:
    """Per-fetch transport wrapper; inspect source JSON without changing it."""

    def __init__(self, transport, symbol: str, skip_missing_ohlc: bool):
        self._transport = transport
        self._symbol = symbol
        self._skip_missing_ohlc = skip_missing_ohlc

    def __getattr__(self, name):
        return getattr(self._transport, name)

    def get(self, *args, **kwargs):
        return self._check(self._transport.get(*args, **kwargs), kwargs.get("params", {}))

    def cache_get(self, *args, **kwargs):
        return self._check(self._transport.cache_get(*args, **kwargs), kwargs.get("params", {}))

    def _check(self, response, params):
        payload = response.json()
        chart = payload.get("chart") or {}
        if chart.get("error") or not chart.get("result"):
            return response  # Let yfinance report the vendor's error.
        result = chart["result"][0]
        timestamps = result.get("timestamp") or []
        if not timestamps:
            return response
        try:
            quotes = result["indicators"]["quote"][0]
            volumes = quotes["volume"]
            if len(volumes) != len(timestamps):
                raise ValueError("Volume count differs from timestamp count")
            missing = []
            omitted = []
            for index, (timestamp, volume) in enumerate(zip(timestamps, volumes, strict=True)):
                # Yahoo can append a latest quote outside the download window;
                # that quote must not prevent ingestion of historical bars.
                if ("period1" in params and timestamp < params["period1"]) or (
                    "period2" in params and timestamp >= params["period2"]
                ):
                    continue
                if not _missing_number(volume):
                    continue
                # Remove wholly missing bars before yfinance can merge them
                # into another bar. Missing volume alone never triggers a skip.
                if self._skip_missing_ohlc and all(
                    _missing_number(quotes[column][index])
                    for column in ("open", "high", "low", "close")
                ):
                    omitted.append(index)
                    continue
                missing.append(pd.Timestamp(timestamp, unit="s", tz="UTC").isoformat())
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SchemaValidationError(f"Malformed Yahoo source volume data: {exc}") from exc
        if missing:
            raise InvalidOHLCVError(
                f"Yahoo {self._symbol}: missing source volume at UTC timestamps: "
                + ", ".join(missing)
                + ". Refusing yfinance's conversion of missing volume to zero."
            )
        if omitted:
            labels = ", ".join(
                pd.Timestamp(timestamps[index], unit="s", tz="UTC").isoformat()
                for index in omitted
            )
            warnings.warn(
                f"Yahoo {self._symbol}: skipped {len(omitted)} bars with all OHLC "
                f"prices missing (UTC): {labels}. Returned data has gaps.",
                UserWarning, stacklevel=3,
            )
            if len(omitted) == len(timestamps):
                raise EmptyDataError(f"Yahoo returned no usable bars for {self._symbol!r}.")
            payload = deepcopy(payload)
            result = payload["chart"]["result"][0]
            excluded = set(omitted)
            result["timestamp"] = [value for index, value in enumerate(timestamps) if index not in excluded]
            for groups in result["indicators"].values():
                for group in groups:
                    for field, values in group.items():
                        if not isinstance(values, list) or len(values) != len(timestamps):
                            raise SchemaValidationError(f"Malformed Yahoo indicator array: {field}")
                        group[field] = [value for index, value in enumerate(values) if index not in excluded]
            return _SourceResponse(response, payload)
        return response


class _SourceResponse:
    def __init__(self, response, payload):
        self._response = response
        self._payload = payload

    def __getattr__(self, name):
        return getattr(self._response, name)

    def json(self):
        return self._payload


def _missing_number(value) -> bool:
    return value is None or (isinstance(value, Real) and math.isnan(value))


def _download_bounds(
    request: DataRequest, *, date_labels: bool
) -> tuple[str | datetime, str | datetime]:
    if not date_labels:
        return request.start, request.end
    # Date strings are interpreted by yfinance in the exchange timezone. UTC
    # instants here could omit the first session for exchanges east of UTC.
    # Enclose partial days and apply exact UTC label bounds after normalization.
    end_date = request.end.date()
    if request.end.time() != time.min:
        end_date += timedelta(days=1)
    return request.start.date().isoformat(), end_date.isoformat()


def _single_ticker_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels != 2:
            raise SchemaValidationError("Yahoo columns must have two MultiIndex levels.")
        # Accept both (Price, Ticker) and (Ticker, Price), without relying on
        # optional level names or silently flattening several tickers together.
        field_levels = [
            level
            for level in range(2)
            if set(_PRICE_COLUMNS).issubset(frame.columns.get_level_values(level))
        ]
        if len(field_levels) != 1:
            raise SchemaValidationError("Yahoo columns must include Open/High/Low/Close/Volume.")
        field_level = field_levels[0]
        tickers = frame.columns.get_level_values(1 - field_level).unique()
        if (
            len(tickers) != 1
            or not isinstance(tickers[0], str)
            or tickers[0].casefold() != symbol.casefold()
        ):
            raise SchemaValidationError(
                f"Yahoo response must contain only the requested ticker {symbol!r}."
            )
        frame = frame.copy(deep=False)
        frame.columns = frame.columns.get_level_values(field_level)
    if not frame.columns.is_unique:
        raise SchemaValidationError("Yahoo response contains duplicate columns.")
    missing = sorted(set(_PRICE_COLUMNS) - set(frame.columns))
    if missing:
        raise SchemaValidationError(f"Yahoo response is missing columns: {', '.join(missing)}.")
    return frame.loc[:, list(_PRICE_COLUMNS)]


def _normalize(
    downloaded: pd.DataFrame, symbol: str, *, date_labels: bool
) -> pl.DataFrame:
    frame = _single_ticker_columns(downloaded, symbol)
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise SchemaValidationError("Yahoo response must have a DatetimeIndex.")
    if index.hasnans:
        raise InvalidOHLCVError("Yahoo timestamps must be non-null.")
    if date_labels:
        timestamps = index.tz_localize(None).normalize().tz_localize("UTC")
    else:
        if index.tz is None:
            raise SchemaValidationError("Yahoo intraday timestamps must be timezone-aware.")
        timestamps = index.tz_convert("UTC")

    data = pd.DataFrame({"timestamp": timestamps, "symbol": symbol})
    for column in _PRICE_COLUMNS:
        try:
            # pandas otherwise converts datetime/timedelta columns into epoch
            # counts, which could masquerade as plausible prices or volume.
            if is_datetime64_any_dtype(frame[column].dtype) or is_timedelta64_dtype(
                frame[column].dtype
            ):
                raise ValueError("real numeric values required")
            values = pd.to_numeric(frame[column], errors="raise")
            if (
                not is_numeric_dtype(values.dtype)
                or is_bool_dtype(values.dtype)
                or is_complex_dtype(values.dtype)
            ):
                raise ValueError("real numeric values required")
            # Keep integer/nullable dtypes until validate_ohlcv checks that
            # Float64 can represent every value without a precision loss.
            data[column.lower()] = values.array
        except (TypeError, ValueError, OverflowError) as exc:
            raise SchemaValidationError(f"Yahoo {column} must contain real numeric values.") from exc
    # Preserve the index's us/ns precision as well: only validate_ohlcv may
    # cast to the canonical millisecond timestamp after checking for loss.
    return pl.from_pandas(data)
