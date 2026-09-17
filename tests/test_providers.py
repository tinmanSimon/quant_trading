"""Deterministic provider contract tests; every Yahoo download is mocked."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_frame_equal as assert_pandas_equal
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.base_fetcher import BaseDataProvider as LegacyBase
from data_pipeline.base_fetcher import OHLCV_SCHEMA
from data_pipeline.exceptions import (
    DuplicateBarError,
    EmptyDataError,
    InvalidDataRequestError,
    InvalidOHLCVError,
    ProviderError,
    SchemaValidationError,
    UnsupportedRequestError,
    UnknownProviderError,
)
from data_pipeline.models import DataRequest
from data_pipeline.providers import (
    BaseDataProvider,
    ProviderRegistry,
    YFinanceProvider,
    get_provider,
    register_provider,
)
from data_pipeline.yahoo_fetcher import YFinanceProvider as LegacyYahoo
import data_pipeline.providers as providers
import data_pipeline.yahoo_fetcher as legacy_yahoo


@pytest.fixture
def request_data() -> DataRequest:
    return DataRequest(
        symbol="AAPL",
        start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 16, 30, tzinfo=UTC),
        timeframe="1h",
        provider="yahoo",
    )


def _vendor_frame(index: pd.Index | None = None) -> pd.DataFrame:
    if index is None:
        index = pd.date_range(
            "2024-01-02 09:30", periods=3, freq="h", tz="America/New_York", name="Datetime"
        )
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(len(index))],
            "High": [102 + i for i in range(len(index))],
            "Low": [99 + i for i in range(len(index))],
            "Close": [101 + i for i in range(len(index))],
            "Volume": [1000 + i for i in range(len(index))],
            # Must never be used as Close, even when this field is null.
            "Adj Close": [None] * len(index),
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def download(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mocked = Mock(return_value=_vendor_frame())
    # Patching this historical path must affect the new implementation, too.
    monkeypatch.setattr(legacy_yahoo.yf, "download", mocked)
    return mocked


class RecordingProvider(BaseDataProvider):
    name = "recording"

    def __init__(self) -> None:
        self.requests: list[DataRequest] = []

    def fetch(self, request: DataRequest) -> pl.DataFrame:
        self.requests.append(request)
        return pl.DataFrame(schema=OHLCV_SCHEMA)


def test_legacy_and_new_imports_are_identical() -> None:
    assert LegacyBase is BaseDataProvider
    assert LegacyYahoo is YFinanceProvider
    assert issubclass(LegacyYahoo, LegacyBase)


def test_base_requires_fetch_implementation() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseDataProvider()  # type: ignore[abstract]


def test_legacy_adapter_builds_request_with_utc_defaults() -> None:
    provider = RecordingProvider()
    result = provider.fetch_ohlcv("AAPL", "2024-01-02", "2024-01-03")

    assert isinstance(result, pl.DataFrame)
    assert provider.requests == [
        DataRequest(
            "AAPL",
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
            provider="recording",
            timeframe="1h",
            dataset="ohlcv",
            price_adjustment="unadjusted",
        )
    ]


def test_legacy_adapter_preserves_explicit_offsets() -> None:
    provider = RecordingProvider()
    provider.fetch_ohlcv(
        "AAPL", "2024-01-02T09:30:00-05:00", "2024-01-02T16:30:00Z", "60m"
    )
    assert provider.requests[0].start == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    # The shared request model canonicalizes the equivalent 60m alias to 1h.
    assert provider.requests[0].timeframe == "1h"


@pytest.mark.parametrize("bad_date", ["", "not-a-date", "2024-02-30", None, 123])
def test_legacy_adapter_rejects_invalid_dates(bad_date: object) -> None:
    with pytest.raises(InvalidDataRequestError, match="ISO"):
        RecordingProvider().fetch_ohlcv("AAPL", bad_date, "2024-02-01")  # type: ignore[arg-type]


def test_registry_reuses_registered_instance() -> None:
    provider = RecordingProvider()
    registry = ProviderRegistry()
    registry.register("  Recording  ", provider)
    assert registry.get("RECORDING") is provider
    assert registry.get(" recording ") is provider


def test_registry_defers_factory_and_calls_it_per_lookup() -> None:
    factory = Mock(side_effect=RecordingProvider)
    registry = ProviderRegistry({"test": factory})
    factory.assert_not_called()
    first, second = registry.get("test"), registry.get("test")
    assert isinstance(first, RecordingProvider)
    assert isinstance(second, RecordingProvider)
    assert first is not second
    assert factory.call_count == 2


def test_registry_accepts_provider_class() -> None:
    registry = ProviderRegistry({"test": RecordingProvider})
    assert isinstance(registry.get("test"), RecordingProvider)


def test_registry_unknown_provider_does_not_fall_back(download: Mock) -> None:
    with pytest.raises(UnknownProviderError, match="missing"):
        ProviderRegistry({"yahoo": YFinanceProvider}).get("missing")
    download.assert_not_called()


def test_registry_overwrite_requires_explicit_replace() -> None:
    original, replacement = RecordingProvider(), RecordingProvider()
    registry = ProviderRegistry({"test": original})
    with pytest.raises(ValueError, match="already registered"):
        registry.register("TEST", replacement)
    assert registry.get("test") is original
    registry.register("test", replacement, replace=True)
    assert registry.get("test") is replacement


@pytest.mark.parametrize("name", ["", "  ", None, 1])
def test_registry_rejects_invalid_names(name: object) -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="name"):
        registry.register(name, RecordingProvider)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        registry.get(name)  # type: ignore[arg-type]


def test_registry_rejects_invalid_entry() -> None:
    with pytest.raises(TypeError, match="instance or factory"):
        ProviderRegistry({"bad": object()})  # type: ignore[dict-item]


def test_registry_rejects_invalid_factory_result() -> None:
    registry = ProviderRegistry({"bad": lambda: object()})  # type: ignore[dict-item, return-value]
    with pytest.raises(ProviderError, match="did not return"):
        registry.get("bad")


def test_registry_wraps_factory_failure() -> None:
    failure = RuntimeError("configuration failed")
    registry = ProviderRegistry({"bad": Mock(side_effect=failure)})
    with pytest.raises(ProviderError, match="construct") as error:
        registry.get("bad")
    assert error.value.__cause__ is failure


def test_default_registry_constructs_yahoo_without_download(download: Mock) -> None:
    assert isinstance(get_provider("yahoo"), YFinanceProvider)
    download.assert_not_called()


def test_default_registry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "registry", ProviderRegistry())
    instance = RecordingProvider()
    register_provider("recording", instance)
    assert get_provider("recording") is instance


def test_yahoo_normalizes_sorts_and_slices_half_open_interval(
    download: Mock, request_data: DataRequest
) -> None:
    original = _vendor_frame().iloc[[2, 0, 1]].copy()
    download.return_value = original
    snapshot = original.copy(deep=True)

    result = YFinanceProvider(timeout=3.5).fetch(request_data)

    assert result.schema == OHLCV_SCHEMA
    assert result["timestamp"].to_list() == [
        datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
    ]
    assert result["symbol"].to_list() == ["AAPL", "AAPL"]
    assert result["close"].to_list() == [101.0, 102.0]
    assert result["volume"].to_list() == [1000.0, 1001.0]
    assert_pandas_equal(original, snapshot)
    download.assert_called_once()
    kwargs = download.call_args.kwargs
    assert kwargs["start"] == request_data.start
    assert kwargs["end"] == request_data.end
    assert kwargs["interval"] == "1h"
    assert kwargs["tickers"] == "AAPL"
    assert kwargs["timeout"] == 3.5
    assert kwargs["keepna"] is True
    assert kwargs["multi_level_index"] is True
    for option in (
        "auto_adjust", "back_adjust", "repair", "actions", "rounding", "prepost",
        "ignore_tz", "threads", "progress",
    ):
        assert kwargs[option] is False


def test_slice_excludes_bars_before_start(download: Mock, request_data: DataRequest) -> None:
    request_data = replace(request_data, start=datetime(2024, 1, 2, 15, tzinfo=UTC))
    result = YFinanceProvider().fetch(request_data)
    assert result["timestamp"].to_list() == [datetime(2024, 1, 2, 15, 30, tzinfo=UTC)]


@pytest.mark.parametrize("layout", ["price-first", "ticker-first"])
@pytest.mark.parametrize("named", [True, False])
def test_yahoo_accepts_single_ticker_multiindex_in_either_orientation(
    download: Mock, request_data: DataRequest, layout: str, named: bool
) -> None:
    frame = _vendor_frame()
    columns = [(column, "AAPL") for column in frame.columns]
    names = ["Price", "Ticker"]
    if layout == "ticker-first":
        columns = [(ticker, column) for column, ticker in columns]
        names.reverse()
    frame.columns = pd.MultiIndex.from_tuples(columns, names=names if named else None)
    snapshot = frame.copy(deep=True)
    download.return_value = frame

    result = YFinanceProvider().fetch(request_data)

    assert result.schema == OHLCV_SCHEMA
    assert result["close"].to_list() == [101.0, 102.0]
    assert_pandas_equal(frame, snapshot)


def test_multiindex_matches_canonical_yahoo_ticker_case(
    download: Mock, request_data: DataRequest
) -> None:
    frame = _vendor_frame()
    frame.columns = pd.MultiIndex.from_tuples([(col, "AAPL") for col in frame.columns])
    download.return_value = frame
    result = YFinanceProvider().fetch(replace(request_data, symbol="aapl"))
    assert result["symbol"].unique().to_list() == ["AAPL"]


@pytest.mark.parametrize("wrong_ticker", ["MSFT", "AAPL,MSFT"])
def test_yahoo_rejects_wrong_or_multiple_tickers(
    download: Mock, request_data: DataRequest, wrong_ticker: str
) -> None:
    frame = _vendor_frame()
    if "," in wrong_ticker:
        frame = pd.concat({"AAPL": frame, "MSFT": frame}, axis=1)
    else:
        frame.columns = pd.MultiIndex.from_product([[wrong_ticker], frame.columns])
    download.return_value = frame
    with pytest.raises(SchemaValidationError, match="only the requested ticker"):
        YFinanceProvider().fetch(request_data)


@pytest.mark.parametrize("timezone", [None, "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize("timeframe", ["1d", "5d", "1wk", "1mo", "3mo"])
def test_daily_and_longer_bars_use_utc_midnight_session_date_labels(
    download: Mock, request_data: DataRequest, timezone: str | None, timeframe: str
) -> None:
    index = pd.date_range("2024-01-02", periods=3, freq="D", tz=timezone, name="Date")
    download.return_value = _vendor_frame(index)
    request_data = replace(
        request_data,
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 4, tzinfo=UTC),
        timeframe=timeframe,
    )

    result = YFinanceProvider().fetch(request_data)

    assert result["timestamp"].to_list() == [
        datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    ]
    assert download.call_args.kwargs["start"] == "2024-01-02"
    assert download.call_args.kwargs["end"] == "2024-01-04"
    assert download.call_args.kwargs["ignore_tz"] is True


def test_daily_partial_day_bounds_enclose_dates_then_slice_labels(
    download: Mock, request_data: DataRequest
) -> None:
    download.return_value = _vendor_frame(pd.date_range("2024-01-02", periods=3, freq="D"))
    request_data = replace(
        request_data,
        start=datetime(2024, 1, 2, 12, tzinfo=UTC),
        end=datetime(2024, 1, 3, 12, tzinfo=UTC),
        timeframe="1d",
    )
    result = YFinanceProvider().fetch(request_data)
    assert result["timestamp"].to_list() == [datetime(2024, 1, 3, tzinfo=UTC)]
    assert download.call_args.kwargs["start"] == "2024-01-02"
    assert download.call_args.kwargs["end"] == "2024-01-04"


def test_intraday_converts_real_instants_across_dst(
    download: Mock, request_data: DataRequest
) -> None:
    index = pd.to_datetime(["2024-11-03T05:30:00Z", "2024-11-03T06:30:00Z"]).tz_convert(
        "America/New_York"
    )
    download.return_value = _vendor_frame(index)
    result = YFinanceProvider().fetch(
        replace(
            request_data,
            start=datetime(2024, 11, 3, 5, tzinfo=UTC),
            end=datetime(2024, 11, 3, 7, tzinfo=UTC),
        )
    )
    assert result["timestamp"].to_list() == [
        datetime(2024, 11, 3, 5, 30, tzinfo=UTC),
        datetime(2024, 11, 3, 6, 30, tzinfo=UTC),
    ]


def test_intraday_rejects_naive_index(download: Mock, request_data: DataRequest) -> None:
    frame = _vendor_frame()
    frame.index = frame.index.tz_localize(None)
    download.return_value = frame
    with pytest.raises(SchemaValidationError, match="timezone-aware"):
        YFinanceProvider().fetch(request_data)


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_yahoo_rejects_empty_vendor_result(
    download: Mock, request_data: DataRequest, empty: pd.DataFrame | None
) -> None:
    download.return_value = empty
    with pytest.raises(EmptyDataError, match="no data"):
        YFinanceProvider().fetch(request_data)
    download.assert_called_once()


def test_yahoo_rejects_empty_result_after_slice(download: Mock, request_data: DataRequest) -> None:
    with pytest.raises(EmptyDataError, match="requested"):
        YFinanceProvider().fetch(
            replace(
                request_data,
                start=datetime(2024, 1, 3, tzinfo=UTC),
                end=datetime(2024, 1, 4, tzinfo=UTC),
            )
        )


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Volume"])
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), float("-inf")])
def test_yahoo_rejects_null_and_nonfinite_values(
    download: Mock, request_data: DataRequest, column: str, invalid: float | None
) -> None:
    frame = _vendor_frame().astype({column: "float64"})
    frame.loc[frame.index[0], column] = invalid
    download.return_value = frame
    with pytest.raises(InvalidOHLCVError, match="finite|non-null"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_does_not_hide_null_timestamps_when_slicing(
    download: Mock, request_data: DataRequest
) -> None:
    frame = _vendor_frame()
    frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    download.return_value = frame
    with pytest.raises(InvalidOHLCVError, match="non-null"):
        YFinanceProvider().fetch(request_data)


@pytest.mark.parametrize("column,value", [("Low", 105), ("Open", 0), ("Volume", -1)])
def test_yahoo_validates_market_data_invariants(
    download: Mock, request_data: DataRequest, column: str, value: int
) -> None:
    frame = _vendor_frame()
    frame.loc[frame.index[0], column] = value
    download.return_value = frame
    with pytest.raises(InvalidOHLCVError):
        YFinanceProvider().fetch(request_data)


def test_yahoo_rejects_duplicate_bars(download: Mock, request_data: DataRequest) -> None:
    frame = _vendor_frame()
    download.return_value = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(DuplicateBarError, match="duplicate"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_normalizes_numeric_strings_and_nullable_integers(
    download: Mock, request_data: DataRequest
) -> None:
    frame = _vendor_frame()
    frame["Open"] = frame["Open"].astype(str)
    frame["Volume"] = frame["Volume"].astype("Int64")
    download.return_value = frame
    result = YFinanceProvider().fetch(request_data)
    assert result.schema == OHLCV_SCHEMA
    assert result["open"].to_list() == [100.0, 101.0]


@pytest.mark.parametrize("delta", [pd.Timedelta(microseconds=1), pd.Timedelta(nanoseconds=1)])
def test_yahoo_rejects_timestamp_precision_loss_before_casting(
    download: Mock, request_data: DataRequest, delta: pd.Timedelta
) -> None:
    frame = _vendor_frame()
    frame.index = frame.index + delta
    download.return_value = frame
    with pytest.raises(SchemaValidationError, match="timestamp precision"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_accepts_exact_millisecond_timestamps(
    download: Mock, request_data: DataRequest
) -> None:
    frame = _vendor_frame()
    frame.index = frame.index + pd.Timedelta(milliseconds=1)
    download.return_value = frame
    result = YFinanceProvider().fetch(request_data)
    assert result.schema == OHLCV_SCHEMA
    assert result["timestamp"][0] == datetime(2024, 1, 2, 14, 30, 0, 1000, tzinfo=UTC)


@pytest.mark.parametrize("value", [2**53 + 1, 2**63 - 1, 2**64 - 1, str(2**53 + 1)])
def test_yahoo_rejects_numeric_precision_loss_before_casting(
    download: Mock, request_data: DataRequest, value: int | str
) -> None:
    frame = _vendor_frame()
    frame["Volume"] = value
    download.return_value = frame
    with pytest.raises(SchemaValidationError, match="losslessly"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_accepts_large_exactly_representable_integer(
    download: Mock, request_data: DataRequest
) -> None:
    frame = _vendor_frame()
    frame["Volume"] = 2**53 + 2
    download.return_value = frame
    result = YFinanceProvider().fetch(request_data)
    assert result["volume"].to_list() == [float(2**53 + 2)] * 2


@pytest.mark.parametrize(
    "value", ["invalid", True, 1 + 2j, pd.Timestamp("2024-01-02"), pd.Timedelta(days=1)]
)
def test_yahoo_rejects_invalid_numeric_types(
    download: Mock, request_data: DataRequest, value: object
) -> None:
    frame = _vendor_frame()
    frame["Open"] = value
    download.return_value = frame
    with pytest.raises(SchemaValidationError, match="numeric"):
        YFinanceProvider().fetch(request_data)


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Volume"])
def test_yahoo_rejects_missing_columns(
    download: Mock, request_data: DataRequest, column: str
) -> None:
    download.return_value = _vendor_frame().drop(columns=column)
    with pytest.raises(SchemaValidationError, match="missing columns"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_rejects_duplicate_column_names(download: Mock, request_data: DataRequest) -> None:
    frame = _vendor_frame()
    download.return_value = pd.concat([frame, frame[["Open"]]], axis=1)
    with pytest.raises(SchemaValidationError, match="duplicate columns"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_rejects_unexpected_index(download: Mock, request_data: DataRequest) -> None:
    download.return_value = _vendor_frame(pd.RangeIndex(3))
    with pytest.raises(SchemaValidationError, match="DatetimeIndex"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_rejects_unexpected_response_type(download: Mock, request_data: DataRequest) -> None:
    download.return_value = "bad response"
    with pytest.raises(ProviderError, match="pandas DataFrame"):
        YFinanceProvider().fetch(request_data)


def test_yahoo_rejects_extra_multiindex_levels(download: Mock, request_data: DataRequest) -> None:
    frame = _vendor_frame()
    frame.columns = pd.MultiIndex.from_tuples([(col, "AAPL", "extra") for col in frame.columns])
    download.return_value = frame
    with pytest.raises(SchemaValidationError, match="two MultiIndex levels"):
        YFinanceProvider().fetch(request_data)


@pytest.mark.parametrize("timeframe", ["3m", "30m", "2h", "2d", "2wk", "6mo"])
def test_yahoo_rejects_unsupported_intervals_before_download(
    download: Mock, request_data: DataRequest, timeframe: str
) -> None:
    with pytest.raises(UnsupportedRequestError, match="timeframe"):
        YFinanceProvider().fetch(replace(request_data, timeframe=timeframe))
    download.assert_not_called()


@pytest.mark.parametrize("timeframe", ["1m", "2m", "5m", "15m", "60m", "90m", "1h"])
def test_yahoo_passes_supported_intraday_interval_without_substitution(
    download: Mock, request_data: DataRequest, timeframe: str
) -> None:
    normalized_request = replace(request_data, timeframe=timeframe)
    YFinanceProvider().fetch(normalized_request)
    download.assert_called_once()
    assert download.call_args.kwargs["interval"] == normalized_request.timeframe


@pytest.mark.parametrize("symbol", ["AAPL MSFT", "AAPL,MSFT", "AAPL\tMSFT"])
def test_yahoo_rejects_multiple_ticker_request(
    download: Mock, request_data: DataRequest, symbol: str
) -> None:
    with pytest.raises(UnsupportedRequestError, match="single ticker"):
        YFinanceProvider().fetch(replace(request_data, symbol=symbol))
    download.assert_not_called()


def test_yahoo_rejects_other_provider(download: Mock, request_data: DataRequest) -> None:
    with pytest.raises(UnsupportedRequestError, match="provider"):
        YFinanceProvider().fetch(replace(request_data, provider="other"))
    download.assert_not_called()


@pytest.mark.parametrize("field,value", [("dataset", "trades"), ("price_adjustment", "adjusted")])
def test_yahoo_defends_capabilities_if_request_contract_is_extended(
    download: Mock, request_data: DataRequest, field: str, value: str
) -> None:
    # DataRequest currently rejects these at construction. Exercise the vendor
    # boundary independently, so widening the shared model cannot enable them.
    object.__setattr__(request_data, field, value)
    with pytest.raises(UnsupportedRequestError):
        YFinanceProvider().fetch(request_data)
    download.assert_not_called()


def test_yahoo_rejects_non_request_input(download: Mock) -> None:
    with pytest.raises(InvalidDataRequestError, match="DataRequest"):
        YFinanceProvider().fetch("AAPL")  # type: ignore[arg-type]
    download.assert_not_called()


@pytest.mark.parametrize("timeout", [None, 0, -1, 61, float("inf"), float("nan"), True, "10"])
def test_yahoo_rejects_unbounded_or_invalid_timeout(timeout: object, download: Mock) -> None:
    with pytest.raises(ValueError, match="timeout"):
        YFinanceProvider(timeout=timeout)  # type: ignore[arg-type]
    download.assert_not_called()


@pytest.mark.parametrize("failure", [TimeoutError("slow vendor"), RuntimeError("rate limited")])
def test_yahoo_wraps_download_errors_without_retry(
    download: Mock, request_data: DataRequest, failure: Exception
) -> None:
    download.side_effect = failure
    with pytest.raises(ProviderError, match="AAPL") as error:
        YFinanceProvider().fetch(request_data)
    assert error.value.__cause__ is failure
    download.assert_called_once()


def test_yahoo_legacy_fetch_matches_request_fetch(request_data: DataRequest) -> None:
    provider = LegacyYahoo()
    legacy = provider.fetch_ohlcv(
        "AAPL", "2024-01-02T14:30:00Z", "2024-01-02T16:30:00Z", "1h"
    )
    assert_frame_equal(legacy, provider.fetch(request_data))
