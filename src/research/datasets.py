"""Strict all-ticker preflight and immutable revision selection."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Mapping

import exchange_calendars as xcals
import polars as pl

from data_pipeline import DataPipeline, DataQuery, DataRequest, validate_ohlcv

from .errors import PreflightError, ResearchError
from .instruments import ExpectedBar, Instrument, expected_bars


@dataclass(frozen=True)
class DataIssue:
    ticker: str
    code: str
    message: str
    timestamps: tuple[datetime, ...] = ()


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[DataIssue, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([
            {"ticker": issue.ticker, "code": issue.code, "message": issue.message}
            for issue in self.issues
        ], schema={"ticker": pl.String, "code": pl.String, "message": pl.String})


@dataclass(frozen=True)
class DatasetSnapshot:
    frames: Mapping[str, pl.DataFrame]
    bars: Mapping[str, tuple[ExpectedBar, ...]]
    manifest: dict
    report: PreflightReport = field(default_factory=PreflightReport)


def normalize_tickers(tickers, *, provider: str, start: datetime, end: datetime,
                      timeframe: str) -> tuple[str, ...]:
    if isinstance(tickers, (str, bytes)):
        raise ResearchError("Supply a list of ticker symbols, not one string.")
    symbols = tuple(DataRequest(symbol=ticker, provider=provider, start=start, end=end,
                                timeframe=timeframe).symbol for ticker in tickers)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ResearchError("Supply at least one ticker; normalized tickers must be distinct.")
    return symbols


def prepare_snapshot(pipeline: DataPipeline, *, tickers, start: datetime, end: datetime,
                     timeframe: str, lookback: int, provider: str = "yahoo", calendar: str = "XNYS",
                     instruments: Mapping[str, Instrument] | None = None,
                     as_of: datetime | None = None, layer: str = "raw",
                     pipeline_id: str | None = None) -> DatasetSnapshot:
    symbols = normalize_tickers(tickers, provider=provider, start=start, end=end, timeframe=timeframe)
    query = DataQuery(provider=provider, start=start, end=end, timeframe=timeframe, layer=layer,
                      pipeline_id=pipeline_id)
    start, end, timeframe, provider = query.start, query.end, query.timeframe, query.provider
    now = datetime.now(UTC)
    if as_of is None:
        as_of = now
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ResearchError("as_of must be a timezone-aware datetime.")
    as_of = as_of.astimezone(UTC)
    if as_of > now:
        raise ResearchError("as_of cannot be in the future.")
    if layer != "raw" or pipeline_id is not None:
        raise ResearchError("Backtests currently require raw data; arbitrary processed histories have no causality guarantee.")
    frames, bar_map, sources, instrument_specs = {}, {}, {}, {}
    issues, notes = [], []
    for symbol in symbols:
        try:
            instrument = (instruments or {}).get(symbol, Instrument(symbol, calendar))
            if instrument.symbol != symbol:
                raise ResearchError("Instrument configuration symbol does not match the requested ticker.")
            bars = expected_bars(instrument, start=start, end=end, timeframe=timeframe, lookback=lookback)
            unfinished = tuple(bar.timestamp for bar in bars if bar.close > min(as_of, end))
            if unfinished:
                issues.append(DataIssue(symbol, "unfinished_bars", "Requested bars are not complete by the interval end/as_of.", unfinished))
            invalid_opens = tuple(bar.timestamp for bar in bars
                                  if start <= bar.timestamp < end and not start <= bar.open < end)
            if invalid_opens:
                issues.append(DataIssue(symbol, "out_of_range_opens", "Requested session opens fall outside the execution interval.", invalid_opens))
            needed = {bar.timestamp for bar in bars}
            selection = DataQuery(layer=layer, provider=provider, symbol=symbol, timeframe=timeframe,
                                  start=bars[0].timestamp, end=end, pipeline_id=pipeline_id)
            # IDs identify immutable files. Read those exact revisions, never
            # re-run an active-data query after preflight or between strategies.
            items = tuple(pipeline.list_datasets(selection))
            if not items:
                issues.append(DataIssue(symbol, "no_data", "No local data for the required trading and warm-up interval."))
                continue
            frame = pl.concat([pipeline.read_dataset(item.dataset_id) for item in items]).sort("symbol", "timestamp")
            frame = frame.filter(
                (pl.col("timestamp").cast(pl.Datetime("us", "UTC")) >= bars[0].timestamp)
                & (pl.col("timestamp").cast(pl.Datetime("us", "UTC")) < end)
            )
            frame = validate_ohlcv(frame)
            if frame["symbol"].unique().to_list() != [symbol]:
                raise ResearchError("Stored data contains the wrong instrument.")
            actual = set(frame["timestamp"].to_list())
            missing, extra = tuple(sorted(needed - actual)), tuple(sorted(actual - needed))
            if missing:
                preview = ", ".join(stamp.isoformat() for stamp in missing[:5])
                issues.append(DataIssue(symbol, "missing_bars", f"Missing {len(missing)} required trading/warm-up bars (UTC): {preview}", missing))
            if extra:
                issues.append(DataIssue(symbol, "unexpected_bars", f"Found {len(extra)} bars outside the expected session grid.", extra))
            for item in items:
                quality = getattr(item, "quality", None)
                if quality is None or quality.status == "unknown":
                    notes.append(f"{symbol}: fetch-quality history is unknown for revision {item.dataset_id}; actual bar coverage was checked.")
            frames[symbol] = frame.clone()
            bar_map[symbol] = bars
            sources[symbol] = [json.loads(item.to_json()) for item in items]
            instrument_specs[symbol] = {"calendar": instrument.calendar, "currency": instrument.currency}
        except Exception as exc:
            issues.append(DataIssue(symbol, type(exc).__name__, str(exc)))
    report = PreflightReport(tuple(issues), tuple(notes))
    if not report.ok:
        raise PreflightError(report)
    return DatasetSnapshot(frames, bar_map, {
        "tickers": list(symbols), "provider": provider, "timeframe": timeframe, "layer": layer,
        "pipeline_id": pipeline_id, "start": start.isoformat(), "end": end.isoformat(),
        "lookback": lookback, "as_of": as_of.isoformat(), "sources": sources,
        "instruments": instrument_specs, "calendar_version": xcals.__version__,
        "price_basis": "vendor_unadjusted", "return_basis": "price_return_no_dividend_credit",
        "account_mode": "independent_long_only_usd_accounts", "notes": list(notes),
    }, report)
