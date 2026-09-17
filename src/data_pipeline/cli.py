"""Command-line ingestion, discovery, processing and deliberate replacement."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import polars as pl

from .api import DataPipeline
from .exceptions import DataPipelineError
from .models import DataQuery, DataRequest
from .processing import load_pipeline


def _datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if len(value) == 10:
            return parsed.replace(tzinfo=UTC)
        if parsed.tzinfo is None:
            raise ValueError("Time-of-day bounds require a timezone offset or Z")
        return parsed
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid timestamp {value!r}: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Fetch raw OHLCV and optionally process it")
    ingest.add_argument("--symbol", required=True)
    ingest.add_argument("--provider", default="yahoo")
    ingest.add_argument("--timeframe", default="1d")
    ingest.add_argument("--start", type=_datetime, required=True)
    ingest.add_argument("--end", type=_datetime, required=True)
    ingest.add_argument("--processors", type=Path, help="JSON file containing an ordered processor list")
    for name in ("list", "coverage", "read"):
        command = commands.add_parser(name)
        command.add_argument("--layer", choices=["raw", "processed"], default="raw")
        for field in ("provider", "symbol", "timeframe", "pipeline-id"):
            command.add_argument(f"--{field}")
        command.add_argument("--start", type=_datetime)
        command.add_argument("--end", type=_datetime)
        if name != "read":
            command.add_argument("--include-history", action="store_true")
        else:
            command.add_argument("--dataset-id", help="Read an exact active or historical revision")
            command.add_argument("--columns", nargs="+")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("dataset_id")
    process = commands.add_parser("process")
    process.add_argument("dataset_ids", nargs="+", help="Compatible active raw input IDs")
    process.add_argument("--processors", type=Path, required=True)
    replace = commands.add_parser("replace", help="Refetch and replace a whole raw revision; preserve history")
    replace.add_argument("dataset_id")
    replace.add_argument("--confirm", required=True, help="Repeat the exact dataset ID")
    compact = commands.add_parser("compact", help="Merge raw files; retain superseded revisions")
    compact.add_argument("dataset_ids", nargs="+")
    compact.add_argument("--confirm", nargs="+", required=True, help="Repeat all target IDs")
    commands.add_parser("recover", help="Quarantine abandoned writes without deleting them")
    commands.add_parser("audit", help="Verify checksums and metadata for every stored revision")
    return parser


def _processors(path: Path | None):
    return load_pipeline("[]" if path is None else path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = DataPipeline(args.data_dir)
        if args.command == "ingest":
            request = DataRequest(symbol=args.symbol, start=args.start, end=args.end,
                                  timeframe=args.timeframe, provider=args.provider)
            result = pipeline.ingest(request, processors=_processors(args.processors))
            print(json.dumps({"raw_id": result.raw.dataset_id,
                              "processed_id": result.processed.dataset_id if result.processed else None,
                              "rows": result.raw.row_count}))
        elif args.command in {"list", "coverage", "read"}:
            query = DataQuery(layer=args.layer, provider=args.provider, symbol=args.symbol,
                              timeframe=args.timeframe, start=args.start, end=args.end,
                              pipeline_id=args.pipeline_id,
                              include_history=getattr(args, "include_history", False))
            if args.command == "read":
                if args.dataset_id:
                    if any((args.provider, args.symbol, args.timeframe, args.start, args.end, args.pipeline_id)):
                        raise ValueError("--dataset-id cannot be combined with query filters")
                    frame = pipeline.read_dataset(args.dataset_id)
                    if args.columns:
                        frame = frame.select(args.columns)
                else:
                    frame = pipeline.read(query, columns=args.columns)
                print(frame.write_json())
            else:
                print("[" + ",".join(item.to_json() for item in pipeline.list_datasets(query)) + "]")
        elif args.command == "inspect":
            print(pipeline.get_metadata(args.dataset_id).to_json())
        elif args.command == "process":
            print(pipeline.process(args.dataset_ids, processors=_processors(args.processors)).to_json())
        elif args.command == "replace":
            print(pipeline.refetch(args.dataset_id, confirm=args.confirm).to_json())
        elif args.command == "compact":
            print(pipeline.store.compact_raw(args.dataset_ids, confirm=args.confirm).to_json())
        elif args.command == "recover":
            print(json.dumps({"quarantined": pipeline.store.recover()}))
        elif args.command == "audit":
            print(json.dumps({"verified": pipeline.store.audit()}))
        return 0
    except (DataPipelineError, ValueError, OSError, pl.exceptions.PolarsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
