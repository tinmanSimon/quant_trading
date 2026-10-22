"""Strict all-ticker preflight and immutable revision selection."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo
import json
from typing import Mapping

import exchange_calendars as xcals
import polars as pl

from data_pipeline import DataPipeline, DataQuery, DataRequest, validate_ohlcv

from .errors import PreflightError, ResearchError
from .instruments import ExpectedBar, Instrument, expected_bars
from .processed import validate_processed_source


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
    warnings: tuple[dict, ...] = ()

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
    effective_starts: Mapping[str, datetime] = field(default_factory=dict)


def normalize_tickers(tickers, *, provider: str, start: datetime, end: datetime,
                      timeframe: str) -> tuple[str, ...]:
    if isinstance(tickers, (str, bytes)):
        raise ResearchError("Supply a list of ticker symbols, not one string.")
    # Reuse DataRequest post_init validation and symbol normalization
    symbols = tuple(DataRequest(symbol=ticker, provider=provider, start=start, end=end,
                                timeframe=timeframe).symbol for ticker in tickers)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ResearchError("Supply at least one ticker; normalized tickers must be distinct.")
    return symbols


def prepare_snapshot(pipeline: DataPipeline, *, tickers, start: datetime, end: datetime,
                     timeframe: str, lookback: int, provider: str = "yahoo", calendar: str = "XNYS",
                     instruments: Mapping[str, Instrument] | None = None,
                     as_of: datetime | None = None, layer: str = "raw",
                     pipeline_id: str | None = None,
                     skip_initial_missing_bars: bool = False,
                     skip_problematic_tickers: bool = False,
                     trading_period: str = "regular") -> DatasetSnapshot:
    if type(skip_initial_missing_bars) is not bool:
        raise ResearchError("skip_initial_missing_bars must be a boolean.")
    if type(skip_problematic_tickers) is not bool:
        raise ResearchError("skip_problematic_tickers must be a boolean.")
    symbols = normalize_tickers(tickers, provider=provider, start=start, end=end, timeframe=timeframe)
    if layer == "processed" and pipeline_id is None:
        raise ResearchError("Processed backtests require an explicit pipeline_id and registered processors with a causality guarantee.")
    if layer == "raw" and pipeline_id is not None:
        raise ResearchError("Raw backtests must not specify pipeline_id.")
    query = DataQuery(provider=provider, start=start, end=end, timeframe=timeframe,
                      layer=layer, pipeline_id=pipeline_id)
    pipeline_id = query.pipeline_id
    start, end, timeframe, provider = query.start, query.end, query.timeframe, query.provider
    if trading_period not in {"regular", "09:00-15:00"}:
        raise ResearchError("Unsupported trading period.")
    if trading_period != "regular" and (timeframe != "1h" or calendar != "XNYS"):
        raise ResearchError("09:00-15:00 trading requires 1h bars on the XNYS calendar.")
    now = datetime.now(UTC)
    if as_of is None:
        as_of = now
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ResearchError("as_of must be a timezone-aware datetime.")
    as_of = as_of.astimezone(UTC)
    if as_of > now:
        raise ResearchError("as_of cannot be in the future.")
    registry = None
    if layer == "processed":
        from data_pipeline.processing import discover_processors
        registry = discover_processors()
    frames, bar_map, sources, instrument_specs = {}, {}, {}, {}
    schedule_cache = {}
    early_close_cache = {}
    warnings = []
    effective_starts, skipped_tickers = {}, {}
    parent_sources, parent_cache = {}, {}
    processor_specs = None
    price_basis = "vendor_unadjusted"
    issues, notes = [], []
    for symbol in symbols:
        try:
            instrument = (instruments or {}).get(symbol, Instrument(symbol, calendar))
            if instrument.symbol != symbol:
                raise ResearchError("Instrument configuration symbol does not match the requested ticker.")
            # A snapshot has one interval/timeframe/lookback, and instruments
            # on the same exchange share the immutable expected-bar schedule.
            if instrument.calendar not in schedule_cache:
                schedule_cache[instrument.calendar] = expected_bars(
                    instrument, start=start, end=end, timeframe=timeframe, lookback=lookback,
                    **({"trading_period": trading_period} if trading_period != "regular" else {}),
                )
            bars = schedule_cache[instrument.calendar]
            early_closes = ()
            if trading_period != "regular":
                if instrument.calendar not in early_close_cache:
                    daily = expected_bars(instrument, start=bars[0].timestamp.replace(hour=0, minute=0),
                                          end=end, timeframe="1d")
                    early_close_cache[instrument.calendar] = tuple(
                        bar for bar in daily if bar.close.astimezone(ZoneInfo("America/New_York")).time() < time(16)
                    )
                early_closes = early_close_cache[instrument.calendar]
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
                if skip_initial_missing_bars:
                    reason = "No local data for the required trading and warm-up interval."
                    skipped_tickers[symbol] = reason
                    notes.append(f"{symbol}: skipped. {reason}")
                    continue
                issues.append(DataIssue(symbol, "no_data", "No local data for the required trading and warm-up interval."))
                continue
            selected_parents = {}
            if layer == "processed":
                for item in items:
                    specs, parents, basis = validate_processed_source(
                        pipeline, item, pipeline_id=pipeline_id, registry=registry,
                        bars=bars, parent_cache=parent_cache,
                    )
                    if processor_specs is not None and (specs != processor_specs or basis != price_basis):
                        raise ResearchError("Cannot combine different processed pipeline specifications or price bases.")
                    processor_specs, price_basis = specs, basis
                    selected_parents.update({parent.dataset_id: parent for parent in parents})
            frame = pl.concat([pipeline.read_dataset(item.dataset_id) for item in items]).sort("symbol", "timestamp")
            frame = frame.filter(
                (pl.col("timestamp").cast(pl.Datetime("us", "UTC")) >= bars[0].timestamp)
                & (pl.col("timestamp").cast(pl.Datetime("us", "UTC")) < end)
            )
            if trading_period == "09:00-15:00":
                local_time = pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.time()
                frame = frame.filter((local_time >= time(9)) & (local_time < time(16)))
            if frame.is_empty() and skip_initial_missing_bars:
                reason = "No local bars in the required trading and warm-up interval."
                skipped_tickers[symbol] = reason
                notes.append(f"{symbol}: skipped. {reason}")
                continue
            frame = validate_ohlcv(frame)
            if frame["symbol"].unique().to_list() != [symbol]:
                raise ResearchError("Stored data contains the wrong instrument.")
            for session in early_closes:
                local_close = session.close.astimezone(ZoneInfo("America/New_York"))
                cutoff = local_close.replace(hour=16, minute=0).astimezone(UTC)
                outside = (pl.col("timestamp") >= session.close) & (pl.col("timestamp") < cutoff)
                removed = frame.filter(outside).height
                frame = frame.filter(~outside)
                warnings.append({"ticker": symbol, "code": "early_close",
                                 "date": str(local_close.date()),
                                 "message": f"Exchange closed at {local_close:%H:%M} America/New_York; later bars are not required. Excluded {removed} post-close bars."})
            if frame.is_empty():
                issues.append(DataIssue(symbol, "no_data", "No usable bars remain in the selected trading period."))
                continue
            actual = set(frame["timestamp"].to_list())
            prefix_count = 0
            if skip_initial_missing_bars:
                first_available = min(actual)
                prefix_count = sum(bar.timestamp < first_available for bar in bars)
                bars = bars[prefix_count:]
                needed = {bar.timestamp for bar in bars}
                if prefix_count:
                    notes.append(f"{symbol}: skipped {prefix_count} initial missing bars; "
                                 f"available history starts {first_available.isoformat()}.")
            missing, extra = tuple(sorted(needed - actual)), tuple(sorted(actual - needed))
            if missing:
                preview = ", ".join(stamp.isoformat() for stamp in missing[:5])
                issues.append(DataIssue(symbol, "missing_bars", f"Missing {len(missing)} required trading/warm-up bars (UTC): {preview}", missing))
            if extra:
                issues.append(DataIssue(symbol, "unexpected_bars", f"Found {len(extra)} bars outside the expected session grid.", extra))
            # Check the entire remaining grid before skipping short histories:
            # insufficient warm-up must never conceal interior/trailing gaps.
            if skip_initial_missing_bars and len(bars) <= lookback:
                reason = f"Need {lookback} completed warm-up bars and one execution bar; only {len(bars)} available."
                skipped_tickers[symbol] = reason
                notes.append(f"{symbol}: skipped. {reason}")
                continue
            effective_start = max(start, bars[lookback].timestamp) if prefix_count else start
            if effective_start > start:
                notes.append(f"{symbol}: trading starts {effective_start.isoformat()} after {lookback} completed warm-up bars.")
            for item in items:
                quality = getattr(item, "quality", None)
                if quality is None or quality.status == "unknown":
                    notes.append(f"{symbol}: fetch-quality history is unknown for revision {item.dataset_id}; actual bar coverage was checked.")
            frames[symbol] = frame.clone()
            bar_map[symbol] = bars
            sources[symbol] = [json.loads(item.to_json()) for item in items]
            instrument_specs[symbol] = {"calendar": instrument.calendar, "currency": instrument.currency}
            effective_starts[symbol] = effective_start
            if layer == "processed":
                parent_sources[symbol] = [json.loads(parent.to_json()) for _, parent in sorted(selected_parents.items())]
        except Exception as exc:
            issues.append(DataIssue(symbol, type(exc).__name__, str(exc)))
    report = PreflightReport(tuple(issues), tuple(notes), tuple(warnings))
    if not report.ok and not skip_problematic_tickers:
        raise PreflightError(report)
    if skip_problematic_tickers:
        for symbol in dict.fromkeys(issue.ticker for issue in issues):
            skipped_tickers[symbol] = "; ".join(issue.message for issue in issues if issue.ticker == symbol)
            for mapping in (frames, bar_map, sources, instrument_specs, effective_starts, parent_sources):
                mapping.pop(symbol, None)
        if not frames and issues:
            raise PreflightError(report)
    if not frames:
        raise ResearchError("No tickers have enough local history for the requested backtest interval and warm-up.")
    manifest = {
        "trading_period": trading_period, "warnings": warnings,
        "tickers": list(frames), "provider": provider, "timeframe": timeframe, "layer": layer,
        "start": start.isoformat(), "end": end.isoformat(),
        "lookback": lookback, "as_of": as_of.isoformat(), "sources": sources,
        "instruments": instrument_specs, "calendar_version": xcals.__version__,
        "price_basis": price_basis, "return_basis": "price_return_no_dividend_credit",
        "account_mode": "independent_long_only_usd_accounts", "notes": list(notes),
    }
    if layer == "processed":
        manifest.update(pipeline_id=pipeline_id, processors=processor_specs, parent_sources=parent_sources)
    if skip_initial_missing_bars or skip_problematic_tickers:
        manifest.update(skip_initial_missing_bars=skip_initial_missing_bars, requested_tickers=list(symbols),
                        effective_starts={symbol: stamp.isoformat() for symbol, stamp in effective_starts.items()},
                        skipped_tickers=skipped_tickers)
    if skip_problematic_tickers:
        manifest.update(skip_problematic_tickers=True, ticker_errors=[
            {"ticker": issue.ticker, "code": issue.code, "message": issue.message,
             "timestamps": [stamp.isoformat() for stamp in issue.timestamps]}
            for issue in issues
        ])
    return DatasetSnapshot(frames, bar_map, manifest, report, effective_starts)
