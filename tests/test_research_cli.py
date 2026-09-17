"""Offline command-line behavior using a substituted research service."""

from datetime import UTC, datetime
from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from research import BatchFetchReport, DataIssue, FetchOutcome, PreflightError, PreflightReport
from research import cli


@pytest.fixture
def app(monkeypatch):
    service = Mock()
    factory = Mock(return_value=service)
    monkeypatch.setattr(cli, "Research", factory)
    return service, factory


def fetch_args(tmp_path):
    return ["--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs"),
            "fetch", "--tickers", "AAPL", "MSFT", "NVDA", "--start", "2025-01-01",
            "--end", "2025-02-01", "--timeframe", "1h"]


def backtest_args(tmp_path):
    specs = [{"name": "momentum", "version": "1", "config": {"lookback": 2}}]
    config = tmp_path / "strategies.json"
    config.write_text(json.dumps(specs), encoding="utf-8")
    return ["--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs"),
            "backtest", "--tickers", "AAPL", "MSFT", "--start", "2025-01-01",
            "--end", "2025-02-01", "--strategies", str(config)]


def test_help_describes_commands_without_initializing_storage(app, capsys):
    _, factory = app
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "fetch" in output and "backtest" in output and "list-runs" in output
    factory.assert_not_called()


@pytest.mark.parametrize("args,fragment", [
    ([], "required"),
    (["fetch"], "--tickers"),
    (["fetch", "--tickers", "AAPL", "--start", "bad-date", "--end", "2025-02-01"], "ISO"),
    (["fetch", "--tickers", "AAPL", "--start", "2025-01-01", "--end", "2025-02-01", "--timeframe", "7h"], "invalid choice"),
])
def test_bad_arguments_exit_cleanly_without_storage(app, capsys, args, fragment):
    _, factory = app
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert fragment in capsys.readouterr().err
    factory.assert_not_called()


def test_fetch_reports_all_tickers_and_failure_exit_code(app, capsys, tmp_path):
    service, factory = app
    service.fetch_many.return_value = BatchFetchReport((
        FetchOutcome("AAPL", "saved", ("aapl-id",), 42),
        FetchOutcome("MSFT", "failed", error_type="ProviderError", error_message="upstream timed out"),
        FetchOutcome("NVDA", "saved", ("nvda-id",), 40),
    ))
    assert cli.main(fetch_args(tmp_path) + ["--skip-missing-ohlc"]) == 1
    factory.assert_called_once_with(str(tmp_path / "data"), str(tmp_path / "runs"))
    service.fetch_many.assert_called_once_with(
        tickers=["AAPL", "MSFT", "NVDA"], start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 2, 1, tzinfo=UTC), timeframe="1h", skip_missing_ohlc=True,
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["AAPL: saved 42 rows; aapl-id", "MSFT: FAILED (ProviderError): upstream timed out",
                     "NVDA: saved 40 rows; nvda-id"]
    service.backtest.assert_not_called()


def test_fetch_all_success_returns_zero_and_defaults_to_strict(app, capsys, tmp_path):
    service, _ = app
    service.fetch_many.return_value = BatchFetchReport(tuple(
        FetchOutcome(ticker, "saved", (ticker.lower(),), 7) for ticker in ["AAPL", "MSFT", "NVDA"]
    ))
    assert cli.main(fetch_args(tmp_path)) == 0
    assert service.fetch_many.call_args.kwargs["skip_missing_ohlc"] is False
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_explicit_offset_dates_are_normalized_to_utc(app, tmp_path):
    service, _ = app
    service.fetch_many.return_value = BatchFetchReport((FetchOutcome("AAPL", "saved", ("id",), 1),))
    assert cli.main(["--data-dir", str(tmp_path / "data"), "fetch", "--tickers", "AAPL",
                     "--start", "2025-01-02T09:30:00-05:00", "--end", "2025-01-02T16:00:00-05:00"]) == 0
    assert service.fetch_many.call_args.kwargs["start"] == datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    assert service.fetch_many.call_args.kwargs["end"] == datetime(2025, 1, 2, 21, tzinfo=UTC)


def test_backtest_preflight_abort_prints_all_problem_tickers(app, capsys, tmp_path):
    service, _ = app
    service.backtest.side_effect = PreflightError(PreflightReport((
        DataIssue("AAPL", "missing_bars", "Missing 1 required bar."),
        DataIssue("MSFT", "no_data", "No local data."),
    )))
    assert cli.main(backtest_args(tmp_path)) == 1
    output = capsys.readouterr().out
    assert "Backtest aborted before execution" in output
    assert "AAPL: Missing 1 required bar." in output
    assert "MSFT: No local data." in output
    assert "Saved run" not in output
    service.fetch_many.assert_not_called()
    assert not (tmp_path / "runs").exists()


def test_backtest_success_passes_specs_settings_and_prints_saved_path(app, capsys, tmp_path):
    service, _ = app
    path = tmp_path / "runs" / "example-run"
    service.backtest.return_value = SimpleNamespace(
        run_id="example-run", path=path,
        comparison=pl.DataFrame({"symbol": ["AAPL", "MSFT"], "return": [0.01, 0.02]}),
    )
    args = backtest_args(tmp_path) + ["--initial-cash", "12500.50", "--commission-fixed", "1.25",
                                    "--commission-bps", "0.1", "--slippage-bps", "2", "--calendar", "XNYS"]
    assert cli.main(args) == 0
    kwargs = service.backtest.call_args.kwargs
    assert kwargs["strategies"] == [{"name": "momentum", "version": "1", "config": {"lookback": 2}}]
    assert kwargs["settings"].initial_cash == Decimal("12500.50")
    assert kwargs["settings"].commission_fixed == Decimal("1.25")
    assert kwargs["settings"].commission_bps == Decimal("0.1")
    assert kwargs["settings"].slippage_bps == Decimal("2")
    assert kwargs["timeframe"] == "1d"
    assert kwargs["calendar"] == "XNYS"
    output = capsys.readouterr().out
    assert "Saved run: example-run" in output
    assert str(path) in output
    service.fetch_many.assert_not_called()


@pytest.mark.parametrize("option", ["--initial-cash", "--commission-fixed", "--commission-bps", "--slippage-bps"])
def test_invalid_decimal_is_a_clean_argument_error(app, capsys, tmp_path, option):
    _, factory = app
    with pytest.raises(SystemExit) as error:
        cli.main(backtest_args(tmp_path) + [option, "not-a-number"])
    assert error.value.code == 2
    assert "error" in capsys.readouterr().err.lower()
    factory.assert_not_called()


def test_negative_cash_fails_without_starting_backtest(app, capsys, tmp_path):
    service, _ = app
    assert cli.main(backtest_args(tmp_path) + ["--initial-cash", "-1"]) == 1
    assert "positive" in capsys.readouterr().out
    service.backtest.assert_not_called()


def test_invalid_strategy_json_is_a_failed_command(app, capsys, tmp_path):
    service, _ = app
    args = backtest_args(tmp_path)
    (tmp_path / "strategies.json").write_text("not json", encoding="utf-8")
    assert cli.main(args) == 1
    assert "JSONDecodeError" in capsys.readouterr().out
    service.backtest.assert_not_called()


def test_storage_initialization_error_is_reported_without_traceback(app, capsys, tmp_path):
    _, factory = app
    factory.side_effect = ValueError("data_dir must be a directory")
    assert cli.main(fetch_args(tmp_path)) == 1
    assert "data_dir must be a directory" in capsys.readouterr().out


def test_list_runs_prints_valid_json(app, capsys, tmp_path):
    service, _ = app
    service.list_runs.return_value = [{"run_id": "one"}, {"run_id": "two"}]
    assert cli.main(["--runs-dir", str(tmp_path / "runs"), "list-runs"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"run_id": "one"}, {"run_id": "two"}]
    service.fetch_many.assert_not_called()
    service.backtest.assert_not_called()
