"""Small application service joining fetching, preflight and simulations."""

from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import polars as pl

from data_pipeline import DataPipeline

from .backtesting import ExecutionSettings, run_backtest
from .batch_fetch import fetch_many
from .datasets import prepare_snapshot
from .errors import PreflightError, ResearchError
from .runs import ResearchRun, comparison_identity, list_runs, load_run, save_run
from .strategies import StrategyRegistry, load_registry, load_strategy, strategy_spec


class Research:
    def __init__(self, data_dir="data", runs_dir="runs", *, providers=None, instruments=None,
                 strategies: StrategyRegistry | None = None):
        data_root = Path(data_dir).expanduser().resolve()
        self.runs_dir = Path(runs_dir).expanduser().resolve()
        if data_root == self.runs_dir or data_root in self.runs_dir.parents or self.runs_dir in data_root.parents:
            raise ResearchError("Market data and research results must use separate, non-overlapping directories.")
        self.pipeline = DataPipeline(data_root, providers=providers)
        self.instruments = dict(instruments or {})
        self.strategies = strategies

    @property
    def strategy_registry(self):
        """Load local strategy code only when discovery or construction needs it."""
        if self.strategies is None:
            self.strategies = load_registry()
        return self.strategies

    def fetch_many(self, tickers, *, start: datetime, end: datetime, timeframe="1d",
                   provider="yahoo", skip_missing_ohlc=None):
        return fetch_many(self.pipeline, tickers=tickers, start=start, end=end, timeframe=timeframe,
                          provider=provider, skip_missing_ohlc=skip_missing_ohlc)

    def _strategies(self, strategies):
        if isinstance(strategies, (str, bytes, dict)):
            raise ResearchError("Supply a list of strategy objects or versioned specifications.")
        loaded = [load_strategy(item, registry=self.strategy_registry) if isinstance(item, dict) else item
                  for item in strategies]
        if not loaded:
            raise ResearchError("Supply at least one strategy.")
        for item in loaded:
            strategy_spec(item)
        return loaded

    def _snapshot(self, tickers, strategies, **kwargs):
        return prepare_snapshot(self.pipeline, tickers=tickers,
                                lookback=max(item.lookback for item in strategies),
                                instruments=self.instruments, **kwargs)

    def preflight(self, tickers, *, strategies, start, end, timeframe="1d", provider="yahoo",
                  calendar="XNYS", as_of=None):
        workers = self._strategies(strategies)
        try:
            return self._snapshot(tickers, workers, start=start, end=end, timeframe=timeframe,
                                  provider=provider, calendar=calendar, as_of=as_of).report
        except PreflightError as exc:
            return exc.report

    def backtest(self, tickers, *, strategies, start, end, timeframe="1d", provider="yahoo",
                 calendar="XNYS", as_of=None, settings: ExecutionSettings | None = None) -> ResearchRun:
        workers = self._strategies(strategies)
        settings = settings if settings is not None else ExecutionSettings()
        if not isinstance(settings, ExecutionSettings):
            raise TypeError("settings must be ExecutionSettings.")
        # Finish checking EVERY requested instrument before evaluating any strategy.
        snapshot = self._snapshot(tickers, workers, start=start, end=end, timeframe=timeframe,
                                  provider=provider, calendar=calendar, as_of=as_of)
        results = []
        for symbol, frame in snapshot.frames.items():
            times = {bar.timestamp: (bar.open, bar.close) for bar in snapshot.bars[symbol]}
            for worker in workers:
                results.append(run_backtest(frame, worker, start=start, end=end,
                                            settings=settings, bar_times=times))
        manifest = dict(snapshot.manifest)
        manifest["execution_settings"] = settings.to_dict()
        manifest["strategies"] = [strategy_spec(worker) for worker in workers]
        source_root = Path(__file__).parent
        digest = sha256()
        for source in sorted(source_root.rglob("*.py")):
            digest.update(str(source.relative_to(source_root)).encode())
            digest.update(source.read_bytes())
        manifest["engine_sha256"] = digest.hexdigest()
        manifest["packages"] = {name: version(name) for name in ("polars", "pandas", "exchange-calendars")}
        return save_run(self.runs_dir, results, manifest)

    def list_runs(self):
        return list_runs(self.runs_dir)

    def load_run(self, run_id):
        return load_run(self.runs_dir, run_id)

    def compare_runs(self, run_ids) -> pl.DataFrame:
        if isinstance(run_ids, (str, bytes)):
            raise ResearchError("Supply a list of run IDs.")
        runs = [self.load_run(run_id) for run_id in run_ids]
        if not runs:
            raise ResearchError("Select at least one run.")
        identity = comparison_identity(runs[0].manifest)
        if any(comparison_identity(run.manifest) != identity for run in runs[1:]):
            raise ResearchError("Runs use different data revisions, periods, warm-up, calendars, execution rules or engine environments.")
        return pl.concat([run.comparison.with_columns(pl.lit(run.run_id).alias("run_id")) for run in runs])
