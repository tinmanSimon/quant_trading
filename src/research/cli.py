"""Command-line entry points for batch fetching and local-only research."""

import argparse
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from data_pipeline import DataPipelineError
from data_pipeline.providers import YFinanceProvider

from .api import Research
from .backtesting import ExecutionSettings
from .errors import PreflightError
from .timeframes import BACKTEST_TIMEFRAMES, fetch_timeframes, normalize_timeframe


def _timeframe(value):
    try:
        return normalize_timeframe(value)
    except DataPipelineError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
        choices = fetch_timeframes(YFinanceProvider.supported_timeframes) if name == "fetch" else BACKTEST_TIMEFRAMES
        sub.add_argument("--timeframe", type=_timeframe, choices=choices, default="1d")
        if name == "fetch":
            sub.add_argument("--skip-missing-ohlc", action="store_true")
            sub.add_argument("--skip-invalid-bars", action="store_true",
                             help="Exclude invalid Yahoo bars and record their original values and reasons.")
        else:
            sub.add_argument("--strategies", type=Path, required=True,
                             help="JSON file containing a list of versioned strategy specifications.")
            sub.add_argument("--calendar", default="XNYS")
            sub.add_argument("--layer", choices=("raw", "processed"), default="raw")
            sub.add_argument("--pipeline-id", help="Exact processing pipeline fingerprint; required for processed backtests.")
            sub.add_argument("--research-mode", choices=("strict", "adjusted_price"), default="strict",
                             help="Use adjusted_price for processed split/dividend-adjusted histories; strict requires causal processing (default).")
            sub.add_argument("--initial-cash", type=_decimal, default=Decimal("10000"))
            sub.add_argument("--commission-fixed", type=_decimal, default=Decimal("0"))
            sub.add_argument("--commission-bps", type=_decimal, default=Decimal("0"))
            sub.add_argument("--slippage-bps", type=_decimal, default=Decimal("0"))
            sub.add_argument("--allocation-mode", choices=("per_entry", "initial_split", "fixed_tenth", "holding_slots"), default="per_entry",
                             help="Allocate shared cash, split capital once, use a fixed percentage, or divide current cash by remaining slots (25 by default).")
            sub.add_argument("--holding-slots", type=int, default=25,
                             help="Initial available holdings in holding_slots mode (positive integer; default: 25).")
            sub.add_argument("--entry-percentage", type=_decimal, default=Decimal("10"),
                             help="Initial capital percentage per entry in fixed_tenth mode (default: 10).")
            sub.add_argument("--max-workers", type=int, default=1,
                             help="Worker processes for initial_split mode (1 to 32).")
            sub.add_argument("--fractional-shares", action=argparse.BooleanOptionalAction, default=True,
                             help="Allow fractional purchases with a minimum $5 purchase budget (default: enabled).")
            sub.add_argument("--skip-initial-missing-bars", action="store_true",
                             help="Start late tickers after available warm-up history; interior or trailing gaps still fail.")
    commands.add_parser("list-runs")
    args = parser.parse_args(argv)
    try:
        app = Research(args.data_dir, args.runs_dir)
        if args.command == "list-runs":
            print(json.dumps(app.list_runs(), indent=2))
            return 0
        query = dict(tickers=args.tickers, start=args.start, end=args.end, timeframe=args.timeframe)
        if args.command == "fetch":
            report = app.pipeline.fetch_many(**query, skip_missing_ohlc=args.skip_missing_ohlc,
                                             skip_invalid_bars=args.skip_invalid_bars)
            for outcome in report.outcomes:
                if outcome.status == "failed":
                    print(f"{outcome.ticker}: FAILED ({outcome.error_type}): {outcome.error_message}")
                else:
                    print(f"{outcome.ticker}: saved {outcome.row_count} rows; {', '.join(outcome.dataset_ids)}")
            return 0 if report.ok else 1
        specs = json.loads(args.strategies.read_text(encoding="utf-8"))
        settings = ExecutionSettings(initial_cash=args.initial_cash, commission_fixed=args.commission_fixed,
                                     commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
                                     fractional_shares=args.fractional_shares)
        run = app.backtest(**query, strategies=specs, settings=settings, calendar=args.calendar,
                           layer=args.layer, pipeline_id=args.pipeline_id, research_mode=args.research_mode,
                           allocation_mode=args.allocation_mode, max_workers=args.max_workers, entry_percentage=args.entry_percentage,
                           initial_available_holdings=args.holding_slots,
                           skip_initial_missing_bars=args.skip_initial_missing_bars)
        if args.allocation_mode == "initial_split" and run.manifest.get("allocation_summary"):
            print(json.dumps(run.manifest["allocation_summary"], indent=2))
        print(run.comparison)
        print(f"Saved run: {run.run_id}\nFiles: {run.path}")
        return 0
    except PreflightError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
