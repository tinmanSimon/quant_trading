"""Strict analysis coverage is an explicit, persisted per-run policy."""

from datetime import UTC, datetime

import pytest

from research.analysis.batch import BatchRequest


@pytest.mark.parametrize("provider,expected", [
    ("yahoo", False),
    ("massive", False),
    ("databento", True),
    ("databento-xnas.itch", True),
    ("databento-equs.mini", True),
    (" DATABENTO-XNAS.ITCH ", True),
])
def test_new_batch_validation_default_follows_provider(provider, expected):
    request = BatchRequest(("AAA",), provider=provider)
    assert request.strict_validation is expected
    assert request.calendar == "XNYS"
    assert request.to_dict()["strict_validation"] is expected


@pytest.mark.parametrize("provider", ["yahoo", "databento-xnas.itch"])
@pytest.mark.parametrize("strict_validation", [False, True])
def test_validation_override_roundtrips(provider, strict_validation):
    request = BatchRequest(("AAA",), provider=provider, strict_validation=strict_validation,
                           calendar=" XNYS ", session_timezone="America/New_York",
                           start=datetime(2024, 1, 2, tzinfo=UTC),
                           end=datetime(2024, 1, 3, tzinfo=UTC))
    settings = request.to_dict()
    assert settings["strict_validation"] is strict_validation
    assert settings["calendar"] == "XNYS"
    assert BatchRequest.from_dict(settings) == request


@pytest.mark.parametrize("invalid", [0, 1, "true", "false", [], {}])
def test_validation_policy_rejects_non_boolean_values(invalid):
    with pytest.raises(ValueError, match="strict_validation must be a boolean"):
        BatchRequest(("AAA",), strict_validation=invalid)


@pytest.mark.parametrize("invalid", [None, "", "  ", 1, True, []])
def test_calendar_requires_nonempty_string_even_when_strict_disabled(invalid):
    with pytest.raises(ValueError, match="calendar must be a nonempty string"):
        BatchRequest(("AAA",), strict_validation=False, calendar=invalid)


@pytest.mark.parametrize("provider", ["yahoo", "databento-xnas.itch"])
def test_legacy_saved_run_does_not_claim_strict_validation(provider):
    settings = {"tickers": ["AAA"], "provider": provider, "timeframe": "1d"}
    restored = BatchRequest.from_dict(settings)
    assert restored.strict_validation is False
    assert restored.calendar == "XNYS"
    assert "strict_validation" not in settings
    assert "calendar" not in settings


def test_explicit_none_uses_provider_default():
    request = BatchRequest(("AAA",), provider="databento-xnas.itch", strict_validation=None)
    assert request.strict_validation is True
