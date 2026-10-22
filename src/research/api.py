"""Research preflight, simulations and saved runs."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import polars as pl

from data_pipeline import DataPipeline

from .backtesting import ExecutionSettings, run_backtest, run_portfolio_backtest
from .backtesting.initial_split import run_initial_split_backtests, summarize_initial_split, validate_allocation_options
from .backtesting.portfolio import validate_entry_percentage
from .datasets import prepare_snapshot
from .errors import PreflightError, ResearchError
from .runs import ResearchRun, comparison_identity, list_runs, load_run, save_run
from .strategies import StrategyRegistry, load_registry, load_strategy, strategy_spec
from .strategies.bracket import BracketStrategy


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
                  calendar="XNYS", as_of=None, layer="raw", pipeline_id=None,
                  skip_initial_missing_bars=False, skip_problematic_tickers=False, trading_period="regular"):
        workers = self._strategies(strategies)
        try:
            return self._snapshot(tickers, workers, start=start, end=end, timeframe=timeframe,
                                  provider=provider, calendar=calendar, as_of=as_of,
                                  layer=layer, pipeline_id=pipeline_id,
                                  skip_initial_missing_bars=skip_initial_missing_bars,
                                  skip_problematic_tickers=skip_problematic_tickers, trading_period=trading_period).report
        except PreflightError as exc:
            return exc.report

    def backtest(self, tickers, *, strategies, start, end, timeframe="1d", provider="yahoo",
                 calendar="XNYS", as_of=None, settings: ExecutionSettings | None = None,
                 layer="raw", pipeline_id=None, allocation_mode="per_entry", max_workers=1,
                 skip_initial_missing_bars=False, skip_problematic_tickers=False, trading_period="regular", entry_percentage=10) -> ResearchRun:
        validate_allocation_options(allocation_mode, max_workers)
        entry_percentage = validate_entry_percentage(entry_percentage)
        workers = self._strategies(strategies)
        if allocation_mode == "fixed_tenth" and any(not isinstance(worker, BracketStrategy) for worker in workers):
            raise ValueError("Fixed percentage allocation requires stop/target strategies with entry signals.")
        settings = settings if settings is not None else ExecutionSettings()
        if not isinstance(settings, ExecutionSettings):
            raise TypeError("settings must be ExecutionSettings.")
        # Finish checking EVERY requested instrument before evaluating any strategy.
        snapshot = self._snapshot(tickers, workers, start=start, end=end, timeframe=timeframe,
                                  provider=provider, calendar=calendar, as_of=as_of,
                                  layer=layer, pipeline_id=pipeline_id,
                                  skip_initial_missing_bars=skip_initial_missing_bars,
                                  skip_problematic_tickers=skip_problematic_tickers, trading_period=trading_period)
        results = []
        portfolio_workers = [worker for worker in workers if isinstance(worker, BracketStrategy)]
        independent_workers = [worker for worker in workers if not isinstance(worker, BracketStrategy)]
        times_by_symbol = {
            symbol: {bar.timestamp: (bar.open, bar.close) for bar in snapshot.bars[symbol]}
            for symbol in snapshot.frames
        }
        effective_starts = snapshot.effective_starts
        if allocation_mode == "initial_split":
            results = run_initial_split_backtests(
                snapshot.frames, workers, start=start, end=end, settings=settings,
                bar_times=times_by_symbol, effective_starts=effective_starts, max_workers=max_workers,
            )
        else:
            for symbol, frame in snapshot.frames.items():
                for worker in independent_workers:
                    result = run_backtest(frame, worker, start=effective_starts.get(symbol, start), end=end,
                                          settings=settings, bar_times=times_by_symbol[symbol])
                    results.append(result)
            for worker in portfolio_workers:
                results.append(run_portfolio_backtest(
                    snapshot.frames, worker, start=start, end=end,
                    settings=settings, bar_times=times_by_symbol, effective_starts=effective_starts,
                    allocation_mode=allocation_mode, entry_percentage=entry_percentage,
                ))
        for result in results:
            result.metadata.update(layer=snapshot.manifest["layer"],
                                   price_basis=snapshot.manifest["price_basis"])
            if layer == "processed":
                result.metadata["pipeline_id"] = snapshot.manifest["pipeline_id"]
        manifest = dict(snapshot.manifest)
        manifest["allocation_mode"] = allocation_mode
        if allocation_mode == "fixed_tenth":
            manifest["entry_percentage"] = format(entry_percentage.normalize(), "f")
        manifest["max_workers"] = max_workers if allocation_mode == "initial_split" else 1
        if allocation_mode == "initial_split":
            manifest["account_mode"] = "initial_capital_split_among_independent_ticker_accounts"
            manifest["allocation_summary"] = summarize_initial_split(results)
        elif portfolio_workers:
            manifest["account_mode"] = ("mixed_independent_and_shared_usd_accounts" if independent_workers
                                        else "shared_long_only_usd_portfolios")
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
