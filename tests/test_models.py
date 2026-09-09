"""Tests for provider-neutral data requests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from data_pipeline import DataRequest, InvalidDataRequestError


def test_data_request_normalizes_identifiers_and_datetimes() -> None:
    request = DataRequest(
        symbol=" AAPL ",
        start=datetime(2024, 1, 2, 22, 30, tzinfo=timezone(timedelta(hours=8))),
        end=datetime(2024, 1, 2, 23, 30, tzinfo=timezone(timedelta(hours=8))),
        timeframe=" 1H ",
        provider=" Yahoo ",
        dataset=" OHLCV ",
    )

    assert request.symbol == "AAPL"
    assert request.provider == "yahoo"
    assert request.dataset == "ohlcv"
    assert request.timeframe == "1h"
    assert request.start == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    assert request.end == datetime(2024, 1, 2, 15, 30, tzinfo=UTC)


def test_data_request_uses_expected_defaults() -> None:
    request = DataRequest(
        symbol="AAPL",
        start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
    )

    assert request.dataset == "ohlcv"
    assert request.provider == "yahoo"
    assert request.timeframe == "1h"


@pytest.mark.parametrize("timeframe", ["", "0h", "hour", "1x"])
def test_data_request_rejects_invalid_timeframes(timeframe: str) -> None:
    with pytest.raises(InvalidDataRequestError, match="timeframe"):
        DataRequest(
            symbol="AAPL",
            start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
            end=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
            timeframe=timeframe,
        )


def test_data_request_rejects_naive_datetime_bounds() -> None:
    with pytest.raises(InvalidDataRequestError, match="timezone-aware"):
        DataRequest(
            symbol="AAPL",
            start=datetime(2024, 1, 2, 14, 30),
            end=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        )


def test_data_request_rejects_an_empty_identifier() -> None:
    with pytest.raises(InvalidDataRequestError, match="symbol must not be empty"):
        DataRequest(
            symbol="  ",
            start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
            end=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
        )


def test_data_request_rejects_unknown_dataset() -> None:
    with pytest.raises(InvalidDataRequestError, match="Unsupported dataset"):
        DataRequest(
            symbol="AAPL",
            start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
            end=datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
            dataset="trades",
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (
            datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        ),
        (
            datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
            datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        ),
    ],
)
def test_data_request_requires_a_nonempty_time_range(
    start: datetime, end: datetime
) -> None:
    with pytest.raises(InvalidDataRequestError, match="start must be earlier"):
        DataRequest(symbol="AAPL", start=start, end=end)
