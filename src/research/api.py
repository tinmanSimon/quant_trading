"""Research preflight, simulations and saved runs."""

from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import polars as pl

from data_pipeline import DataPipeline

from .backtesting import ExecutionSettings, run_backtest, run_portfolio_backtest
from .backtesting.initial_split import run_initial_split_backtests, summarize_initial_split, validate_allocation_options
from .backtesting.portfolio import DEFAULT_HOLDING_SLOTS, validate_entry_percentage, validate_holding_slots
from .datasets import normalize_tickers, prepare_snapshot
from .errors import PreflightError, ResearchError
from .instruments import Instrument
from .runs import ResearchRun, comparison_identity, list_runs, load_run, save_run
from .strategies import StrategyRegistry, load_registry, load_strategy, strategy_spec
from .strategies.bracket import BracketStrategy


def _engine_environment() -> dict:
    source_root = Path(__file__).parent
    digest = sha256()
    for source in sorted(source_root.rglob("*.py")):
        digest.update(str(source.relative_to(source_root)).encode())
        digest.update(source.read_bytes())
    return {
        "engine_sha256": digest.hexdigest(),
        "packages": {name: version(name) for name in ("polars", "pandas", "exchange-calendars")},
    }


_RETROSPECTIVE_EXCLUSION = {
    "ticker": "PORTFOLIO", "code": "retrospective_ticker_exclusion",
    "message": "Skipping problematic tickers removes them from the entire test using full-period data checks. "
               "A later data problem can therefore change earlier portfolio decisions and introduce selection bias.",
}


def _price_assumptions(manifest: dict) -> dict:
    """Describe the units actually traded, without inventing a cash-action ledger."""
    adjusted = manifest.get("research_mode", "strict") == "adjusted_price"
    return {
        "research_mode": manifest.get("research_mode", "strict"),
        "price_basis": manifest["price_basis"],
        "return_basis": manifest["return_basis"],
        "corporate_actions": (
            "Historical splits and cash dividends are reflected in the processed price series. "
            "No additional corporate-action position accounting; other event types remain unadjusted."
            if adjusted else "No automatic split adjustment or corporate-action position accounting."
        ),
        "dividends": (
            "Reflected in adjusted prices and returns; no additional dividend cash credit."
            if adjusted else "Excluded from cash and returns."
        ),
        **({"execution_units": "Prices, share quantities, whole-share rounding and per-share fees use "
                                  "adjusted units, not historical quoted prices or share counts."}
           if adjusted else {}),
    }


def _set_result_price_metadata(result, manifest: dict, assumptions: dict) -> None:
    result.metadata.update(
        layer=manifest["layer"], price_basis=manifest["price_basis"],
        return_basis=manifest["return_basis"], research_mode=manifest.get("research_mode", "strict"),
        assumptions=assumptions,
    )
    if manifest.get("research_mode", "strict") == "adjusted_price":
        # Execution engines have no dividend ledger. Their default "excluded"
        # label must not misdescribe dividend-adjusted input returns.
        result.metadata["dividends"] = "reflected_in_adjusted_prices_no_additional_cash_credit"
        result.metrics["return_basis"] = manifest["return_basis"]
    if manifest["layer"] == "processed":
        result.metadata["pipeline_id"] = manifest["pipeline_id"]


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
                  skip_initial_missing_bars=False, skip_problematic_tickers=False, trading_period="regular",
                  research_mode="strict"):
        workers = self._strategies(strategies)
        try:
            return self._snapshot(tickers, workers, start=start, end=end, timeframe=timeframe,
                                  provider=provider, calendar=calendar, as_of=as_of,
                                  layer=layer, pipeline_id=pipeline_id, research_mode=research_mode,
                                  skip_initial_missing_bars=skip_initial_missing_bars,
                                  skip_problematic_tickers=skip_problematic_tickers, trading_period=trading_period).report
        except PreflightError as exc:
            return exc.report

    @staticmethod
    def _factor_configuration(spec, factor_registry=None, portfolio_registry=None):
        from .factors import FactorBacktestSpec, FactorRegistry, FactorSpec
        from .portfolio import PortfolioRegistry, validate_schedule

        if isinstance(spec, dict):
            spec = FactorBacktestSpec.from_dict(spec)
        if not isinstance(spec, FactorBacktestSpec):
            raise TypeError("spec must be FactorBacktestSpec or its serialized dictionary.")
        factor_registry = factor_registry if factor_registry is not None else FactorRegistry()
        portfolio_registry = portfolio_registry if portfolio_registry is not None else PortfolioRegistry()
        # Configuration errors must fail before reading historical data or
        # running any factor/policy, including when no ticker passes preflight.
        for factor in spec.factors:
            factor_registry.load(factor)
        portfolio_registry.load(spec.portfolio)
        if spec.portfolio.name == "inverse_volatility":
            column = spec.portfolio.config.get("volatility_factor", "volatility")
            if column not in {factor.name for factor in spec.factors}:
                if (not isinstance(column, str) or not column.strip()
                        or column in {"timestamp", "symbol", "score"} or column.startswith("__")):
                    raise ValueError("volatility_factor must be a nonreserved factor column name.")
                factor_registry.load(FactorSpec(
                    "volatility", lookback=spec.portfolio.config.get("volatility_lookback", 20)))
        validate_schedule(spec.schedule)
        return spec, factor_registry, portfolio_registry

    def _factor_snapshot(self, tickers, spec, **kwargs):
        symbols = normalize_tickers(tickers, provider=kwargs["provider"], start=kwargs["start"],
                                    end=kwargs["end"], timeframe=kwargs["timeframe"])
        calendar = kwargs["calendar"]
        for symbol in symbols:
            instrument = self.instruments.get(symbol, Instrument(symbol, calendar))
            if instrument.calendar != calendar:
                raise ResearchError("Factor portfolios require one common exchange calendar; "
                                    f"{symbol} uses {instrument.calendar}, but this run uses {calendar}.")
        return prepare_snapshot(self.pipeline, tickers=symbols, lookback=spec.lookback,
                                instruments=self.instruments, **kwargs)

    def factor_preflight(self, tickers, *, spec, start, end, timeframe="1d", provider="yahoo",
                         calendar="XNYS", as_of=None, layer="raw", pipeline_id=None,
                         skip_initial_missing_bars=False, skip_problematic_tickers=False,
                         trading_period="regular", factor_registry=None, portfolio_registry=None,
                         research_mode="strict"):
        """Validate a factor experiment against pinned local, completed bars.

        No factors or portfolio decisions run during preflight. The optional
        problematic-ticker filter uses the entire interval, so it is explicitly
        reported as retrospective universe selection.
        """
        spec, _, _ = self._factor_configuration(spec, factor_registry, portfolio_registry)
        try:
            report = self._factor_snapshot(
                tickers, spec, start=start, end=end, timeframe=timeframe, provider=provider,
                calendar=calendar, as_of=as_of, layer=layer, pipeline_id=pipeline_id, research_mode=research_mode,
                skip_initial_missing_bars=skip_initial_missing_bars,
                skip_problematic_tickers=skip_problematic_tickers, trading_period=trading_period,
            ).report
        except PreflightError as exc:
            report = exc.report
        if skip_problematic_tickers:
            report = replace(report, warnings=report.warnings + (dict(_RETROSPECTIVE_EXCLUSION),))
        return report

    def factor_backtest(self, tickers, *, spec, start, end, timeframe="1d", provider="yahoo",
                        calendar="XNYS", as_of=None, settings: ExecutionSettings | None = None,
                        layer="raw", pipeline_id=None, skip_initial_missing_bars=False,
                        skip_problematic_tickers=False, trading_period="regular",
                        factor_registry=None, portfolio_registry=None, research_mode="strict") -> ResearchRun:
        """Run a long-only shared-account factor portfolio and save its audits.

        Scores use completed bars and targets execute at a later bar's open.
        Factor portfolios default to fractional shares with the established
        $5 minimum purchase budget; explicit ExecutionSettings take precedence.
        """
        import exchange_calendars as xcals

        from .backtesting.factor_engine import run_factor_backtest

        spec, factor_registry, portfolio_registry = self._factor_configuration(
            spec, factor_registry, portfolio_registry)
        settings = settings if settings is not None else ExecutionSettings(fractional_shares=True)
        if not isinstance(settings, ExecutionSettings):
            raise TypeError("settings must be ExecutionSettings.")
        snapshot = self._factor_snapshot(
            tickers, spec, start=start, end=end, timeframe=timeframe, provider=provider,
            calendar=calendar, as_of=as_of, layer=layer, pipeline_id=pipeline_id, research_mode=research_mode,
            skip_initial_missing_bars=skip_initial_missing_bars,
            skip_problematic_tickers=skip_problematic_tickers, trading_period=trading_period,
        )
        times_by_symbol = {
            symbol: {bar.timestamp: (bar.open, bar.close) for bar in snapshot.bars[symbol]}
            for symbol in snapshot.frames
        }
        result = run_factor_backtest(
            snapshot.frames, spec, start=start, end=end, settings=settings,
            bar_times=times_by_symbol, effective_starts=snapshot.effective_starts,
            factor_registry=factor_registry, portfolio_registry=portfolio_registry,
            session_timezone=str(xcals.get_calendar(calendar).tz),
        )
        serialized_spec = spec.to_dict()
        assumptions = {
            **_price_assumptions(snapshot.manifest),
            "universe": "User-selected symbols, not a reconstructed point-in-time historical universe.",
            "selection": ("Full-period problematic-ticker exclusions may introduce selection bias."
                          if skip_problematic_tickers else "Strict complete-data preflight; no problematic-ticker exclusions."),
        }
        manifest = dict(snapshot.manifest)
        manifest.update(
            flow="factor", factor_spec=serialized_spec,
            account_mode="shared_cash_long_only_factor", allocation_mode="factor_targets",
            execution_settings=settings.to_dict(), assumptions=assumptions,
            factor_execution={"sale_proceeds": spec.sale_proceeds, "funding": spec.funding,
                              "unfilled_orders": "expire_after_intended_open"},
            strategies=[result.strategy_spec], **_engine_environment(),
        )
        result.metadata["flow"] = "factor"
        _set_result_price_metadata(result, manifest, assumptions)
        if skip_problematic_tickers:
            manifest["warnings"] = [*manifest.get("warnings", []), dict(_RETROSPECTIVE_EXCLUSION)]
        return save_run(self.runs_dir, [result], manifest)

    def backtest(self, tickers, *, strategies, start, end, timeframe="1d", provider="yahoo",
                 calendar="XNYS", as_of=None, settings: ExecutionSettings | None = None,
                 layer="raw", pipeline_id=None, allocation_mode="per_entry", max_workers=1,
                 skip_initial_missing_bars=False, skip_problematic_tickers=False, trading_period="regular",
                 entry_percentage=10, research_mode="strict", initial_available_holdings=DEFAULT_HOLDING_SLOTS) -> ResearchRun:
        validate_allocation_options(allocation_mode, max_workers)
        initial_available_holdings = validate_holding_slots(initial_available_holdings)
        entry_percentage = validate_entry_percentage(entry_percentage)
        workers = self._strategies(strategies)
        if allocation_mode in ("fixed_tenth", "holding_slots") and any(not isinstance(worker, BracketStrategy) for worker in workers):
            raise ValueError("This capital allocation requires stop/target strategies with entry signals.")
        settings = settings if settings is not None else ExecutionSettings()
        if not isinstance(settings, ExecutionSettings):
            raise TypeError("settings must be ExecutionSettings.")
        # Finish checking EVERY requested instrument before evaluating any strategy.
        snapshot = self._snapshot(tickers, workers, start=start, end=end, timeframe=timeframe,
                                  provider=provider, calendar=calendar, as_of=as_of,
                                  layer=layer, pipeline_id=pipeline_id, research_mode=research_mode,
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
                    initial_available_holdings=initial_available_holdings,
                ))
        assumptions = _price_assumptions(snapshot.manifest)
        for result in results:
            _set_result_price_metadata(result, snapshot.manifest, assumptions)
        manifest = dict(snapshot.manifest)
        manifest["assumptions"] = assumptions
        manifest["allocation_mode"] = allocation_mode
        if allocation_mode == "holding_slots":
            manifest["initial_available_holdings"] = initial_available_holdings
        if allocation_mode == "fixed_tenth":
            manifest["entry_percentage"] = format(entry_percentage.normalize(), "f")
        manifest["max_workers"] = max_workers if allocation_mode == "initial_split" else 1
        if allocation_mode == "initial_split":
            manifest["account_mode"] = "initial_capital_split_among_independent_ticker_accounts"
            manifest["allocation_summary"] = summarize_initial_split(results)
        elif portfolio_workers:
            short_enabled = any(result.metadata.get("account_model", "").startswith("shared_cash_long_short")
                                for result in results)
            manifest["account_mode"] = ("mixed_independent_and_shared_usd_accounts" if independent_workers
                                        else "shared_long_short_usd_portfolios" if short_enabled
                                        else "shared_long_only_usd_portfolios")
        manifest["execution_settings"] = settings.to_dict()
        manifest["strategies"] = [strategy_spec(worker) for worker in workers]
        manifest.update(_engine_environment())
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
