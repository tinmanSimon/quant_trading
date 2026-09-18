"""Command-line entry points for batch fetching and local-only research."""

import argparse
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from .api import Research
from .backtesting import ExecutionSettings
from .errors import PreflightError


def _date(value):
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO date or datetime.") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _decimal(value):
    try:
        number = Decimal(value)
        if not number.is_finite():
            raise InvalidOperation
        return number
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("Use a finite decimal number.") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch Yahoo fetches and verified local backtests.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--runs-dir", default="runs")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "backtest"):
        sub = commands.add_parser(name)
        sub.add_argument("--tickers", nargs="+", required=True)
        sub.add_argument("--start", type=_date, required=True)
        sub.add_argument("--end", type=_date, required=True)
        sub.add_argument("--timeframe", choices=("1h", "1d"), default="1d")
        if name == "fetch":
            sub.add_argument("--skip-missing-ohlc", action="store_true")
        else:
            sub.add_argument("--strategies", type=Path, required=True,
                             help="JSON file containing a list of versioned strategy specifications.")
            sub.add_argument("--calendar", default="XNYS")
            sub.add_argument("--initial-cash", type=_decimal, default=Decimal("10000"))
            sub.add_argument("--commission-fixed", type=_decimal, default=Decimal("0"))
            sub.add_argument("--commission-bps", type=_decimal, default=Decimal("0"))
            sub.add_argument("--slippage-bps", type=_decimal, default=Decimal("0"))
    commands.add_parser("list-runs")
    args = parser.parse_args(argv)
    try:
        app = Research(args.data_dir, args.runs_dir)
        if args.command == "list-runs":
            print(json.dumps(app.list_runs(), indent=2))
            return 0
        query = dict(tickers=args.tickers, start=args.start, end=args.end, timeframe=args.timeframe)
        if args.command == "fetch":
            report = app.pipeline.fetch_many(**query, skip_missing_ohlc=args.skip_missing_ohlc)
            for outcome in report.outcomes:
                if outcome.status == "failed":
                    print(f"{outcome.ticker}: FAILED ({outcome.error_type}): {outcome.error_message}")
                else:
                    print(f"{outcome.ticker}: saved {outcome.row_count} rows; {', '.join(outcome.dataset_ids)}")
            return 0 if report.ok else 1
        specs = json.loads(args.strategies.read_text(encoding="utf-8"))
        settings = ExecutionSettings(initial_cash=args.initial_cash, commission_fixed=args.commission_fixed,
                                     commission_bps=args.commission_bps, slippage_bps=args.slippage_bps)
        run = app.backtest(**query, strategies=specs, settings=settings, calendar=args.calendar)
        print(run.comparison)
        print(f"Saved run: {run.run_id}\nFiles: {run.path}")
        return 0
    except PreflightError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
