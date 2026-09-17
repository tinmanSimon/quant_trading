"""Yahoo OHLCV adapter using the installed yfinance download API.

Daily and longer bars carry *session-date labels*: exchange-local YYYY-MM-DD
is represented at 00:00:00 UTC, regardless of the exchange's UTC offset. This
is a date convention, not the actual session opening instant. Intraday bars
instead retain their actual instant and are converted from the vendor's aware
index to UTC. Naive intraday timestamps are ambiguous and are rejected.

``unadjusted`` means Yahoo's supplied Open/High/Low/Close, with yfinance's
auto-adjust, back-adjust and repair disabled. It does not undo any historical
split treatment already present in Yahoo's source data. Adj Close is unused.

API documentation: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
import math
from numbers import Real

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
    this adapter adds no retries.
    """

    name = "yahoo"
    supported_timeframes = frozenset(
        {"1m", "2m", "5m", "15m", "60m", "90m", "1h", *_DATE_INTERVALS}
    )
    supported_datasets = frozenset({"ohlcv"})
    supported_price_adjustments = frozenset({"unadjusted"})

    def __init__(self, *, timeout: float = 10.0) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 60
        ):
            raise ValueError("timeout must be a finite number in (0, 60] seconds.")
        self.timeout = float(timeout)

    def fetch(self, request: DataRequest) -> pl.DataFrame:
        self._validate_request(request)
        date_labels = request.timeframe in _DATE_INTERVALS
        start, end = _download_bounds(request, date_labels=date_labels)
        try:
            downloaded = yf.download(
                tickers=request.symbol,
                start=start,
                end=end,
                interval=request.timeframe,
                auto_adjust=False,
                back_adjust=False,
                repair=False,
                actions=False,
                keepna=True,
                rounding=False,
                prepost=False,
                ignore_tz=date_labels,
                group_by="column",
                multi_level_index=True,
                threads=False,
                progress=False,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ProviderError(
                f"Yahoo download failed for {request.symbol!r} "
                f"at interval {request.timeframe!r}."
            ) from exc

        if downloaded is None or (
            isinstance(downloaded, pd.DataFrame) and downloaded.empty
        ):
            # download() commonly suppresses vendor errors and returns empty.
            raise EmptyDataError(
                f"Yahoo returned no data for {request.symbol!r} "
                f"in [{request.start.isoformat()}, {request.end.isoformat()}); "
                "the symbol/range may be unavailable or the vendor request failed."
            )
        if not isinstance(downloaded, pd.DataFrame):
            raise ProviderError("Yahoo download did not return a pandas DataFrame.")

        canonical = _normalize(downloaded, request.symbol, date_labels=date_labels)
        # Validate before slicing so missing keys/values cannot be hidden by
        # filtering. Reject an empty requested interval after slicing.
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
