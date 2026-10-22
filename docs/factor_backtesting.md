# Factor portfolio backtesting

The **Factor backtest** dashboard page and `Research.factor_backtest(...)` run a
long-only portfolio in one shared cash account. Factors, ranking, portfolio
selection/weighting, rebalance schedules, allocation constraints, and execution
are separate components. Changing a portfolio policy does not require changing
the factors or the cash ledger. Existing strategy and stop/target backtests keep
their own behavior.

This is an additional research workflow. A high factor score is a hypothesis to
test, not evidence of predictive power or a recommendation to buy a stock.

## Run an experiment

Choose local tickers, provider, timeframe, raw or processed layer, and dates in
**Factor backtest**. Configure factors, a portfolio policy, a rebalance schedule,
and execution assumptions. Run preflight before the simulation to review missing
bars and warm-up requirements. Settings persist across navigation and restarts.

Python example using previously downloaded local daily data:

```python
from datetime import UTC, datetime

from research import Research, FactorBacktestSpec
from research.backtesting import ExecutionSettings
from research.factors import ComponentSpec, FactorSpec

research = Research(data_dir="data", runs_dir="runs")
spec = FactorBacktestSpec(
    factors=(
        FactorSpec("momentum", lookback=20, weight=2, direction=1),
        FactorSpec("volatility", lookback=20, weight=1, direction=-1),
    ),
    portfolio=ComponentSpec("equal_weight", config={"top_n": 20}),
    schedule=ComponentSpec("weekly"),
    cash_reserve=0.05,
    max_position_weight=0.10,
    min_weight_change=0.005,
    max_turnover=1.0,
    sale_proceeds="same_open",
    funding="pro_rata",
)
selection = dict(
    spec=spec,
    start=datetime(2024, 1, 2, tzinfo=UTC),
    end=datetime(2025, 1, 1, tzinfo=UTC),
    provider="yahoo",
    timeframe="1d",
)
report = research.factor_preflight(["AAPL", "MSFT", "AMZN"], **selection)
print(report.to_frame())
if report.ok:
    run = research.factor_backtest(
        ["AAPL", "MSFT", "AMZN"],
        settings=ExecutionSettings(initial_cash=10000, fractional_shares=True),
        **selection,
    )
    print(run.comparison)
    print(run.results[0].rebalances)
```

The example's 10% position cap means three available holdings can invest at most
30% of equity. Unused allocation remains cash; the engine does not relax caps to
force full investment. Fetch the preceding warm-up history as well as the test
interval. A 20-return factor needs 21 preceding completed observations.

The factor API defaults to fractional shares and a $5 minimum purchase budget,
including fees. Passing an explicit `ExecutionSettings` respects all its values;
that shared settings class still defaults to whole shares for compatibility with
older strategies. The minimum budget does not prevent closing small holdings.

To test a different portfolio policy against the same factors:

```python
from dataclasses import replace

score_spec = replace(spec, portfolio=ComponentSpec("score_weight", config={"top_n": 20}))
# Pass score_spec to factor_backtest with the same local data and dates.
```

`FactorBacktestSpec.to_dict()` and `.from_dict()` provide JSON-compatible
configuration. The public methods also accept the serialized dictionary directly.
There are no network requests during factor preflight or backtesting.

## Factors and scoring

Built-in factor version `1` supports:

| Name | Calculation |
| --- | --- |
| `momentum` | `close / close.shift(lookback) - 1` |
| `volatility` | Population standard deviation of the last `lookback` one-bar simple returns; not annualized |
| `red_bar_high_percentage` | Mean `(high - open) / high` among red bars (`close < open`) in the trailing `lookback` bars |

The red-bar factor uses a rolling window of bars, not the last N red bars. A
window with no red bars has no factor value. Its input is a fraction: `0.01` means
1%. It does not change the existing standalone high-percentage analysis.

At a decision boundary, eligible symbols with all required factor values receive
cross-sectional percentile ranks, with average ranks for ties. `direction=1`
prefers larger values and `direction=-1` prefers smaller values. Positive factor
weights combine the ranks into a weighted average score. Symbol ordering breaks
selection ties deterministically. A newly listed stock participates only after
its required warm-up observations are available.

Prices from a bar are usable only after that bar closes. In particular, the
bar's high, low, and red/green status never determine a purchase at its own open.
Built-in factors are causal. Custom factor implementations are trusted code and
must obey the same requirement; the engine cannot prove their causality.

## Portfolio policies and targets

Built-in portfolio policy version `1` supports:

| Name | Config | Allocation among selected stocks |
| --- | --- | --- |
| `equal_weight` | `top_n` | Equal weights |
| `score_weight` | `top_n` | Proportional to positive scores |
| `inverse_volatility` | `top_n`, `volatility_factor` (default `volatility`), `volatility_lookback` (default 20) | Proportional to inverse positive volatility |

Inverse-volatility weighting uses the named factor when it is already selected.
Otherwise it computes a sizing-only volatility feature using
`volatility_lookback`, without adding that feature to the score. Its required
history contributes to the experiment's warm-up. Zero or missing volatility is
excluded from this weighting rather than dividing by zero.

A policy returns a **complete target portfolio**, not buy/sell orders. Weights
refer to total portfolio equity. Omitted holdings target zero; a complete empty
portfolio requests liquidation. `TargetPortfolio(no_change=True)` means do
nothing. `frozen_symbols` preserves existing quantities. Existing holdings
without an eligible factor score are also frozen instead of silently liquidated.

Shared constraints apply to the proposal: cash reserve, maximum position weight,
minimum weight-change tolerance, and a turnover budget. `max_turnover` is the sum
of absolute position-weight changes: moving 100% from one stock to another costs
2.0 units, and deploying 100% cash costs 1.0. Position caps and reserve reductions
take priority over discretionary turnover limits; reasons identify these
mandatory reductions. Frozen holdings can prevent a reserve or cap from being
achievable, which is reported rather than hidden.

Schedules are `every_n_bars` with `{"bars": N}`, or `daily`, `weekly`, and
`monthly` without additional config. Calendar schedules decide at the final
available completed bar of the session/week/month, using exchange-local dates;
the resulting trades use a subsequent opening event. Known calendar boundaries,
including holidays and early closes, are separate from future prices.

## Rebalance and execution sequence

Each decision has a `rebalance_id` linking the decision, targets, orders, fills,
and achieved positions:

1. Update eligible factors and scores after the decision bar closes.
2. Check the schedule, run the portfolio policy, and apply constraints.
3. Freeze the target decision; later prices cannot change that decision's ranks.
4. At the intended subsequent opening event, take one account valuation snapshot,
   recheck allocation constraints after any price gaps while keeping selection
   fixed, and compute all desired quantities from that same equity value.
5. Sell position reductions. Only successful fills generate proceeds; account
   for their fees before funding purchases.
6. Fund all increases jointly, maintaining the requested cash reserve and paying
   transaction costs. `pro_rata` scales insufficient purchase budgets together;
   `score_priority` funds higher-ranked requests first with deterministic ties.
7. Execute affordable quantities, keeping fractional precision and recording
   skipped or reduced purchases. Unfilled orders expire after their intended
   opening event. They are reconsidered only through a new portfolio decision.
8. Record actual holdings, remaining cash, and later close valuations separately
   from desired targets.

`sale_proceeds="same_open"` lets successful reductions fund purchases at that
same opening event. This assumes same-event sale-and-buy execution; it is not a
broker settlement model. `"next_open"` withholds these new proceeds until the
next opening event. Neither setting creates cash from failed sales. Slippage is
a deterministic adverse basis-point adjustment, not a historical spread model.
Zero-volume execution bars reject orders. Otherwise the model assumes unlimited
quantity at the slipped opening price; it has no volume-participation limit,
shorting, leverage, or financing.

Rebalancing changes each holding only by the required quantity difference. It
does not liquidate and rebuild the entire portfolio every time. Gaps and costs
can make achieved weights differ from targets. Residual cash remains cash.

## Reports, saved runs, and comparison

A factor run uses the existing `ResearchRun` artifact directory and checksum
validation. Its manifest records `flow="factor"`, the complete `factor_spec`,
execution assumptions, local data revisions, engine source hash, package
versions, and warm-up/coverage policies. The result symbol is `PORTFOLIO`.

The result contains the usual `equity`, `orders`, `trades`, and summary metrics,
plus optional audit tables:

- `factor_values`: per-symbol values and scores at decision events.
- `rebalances`: decision and execution events and their outcomes.
- `targets`: proposed, decision-time constrained, opening-time constrained, and
  achieved allocations with reasons.
- `positions`: achieved holdings and allocations.

All populated audit tables are checksummed Parquet artifacts. Old saved strategy
runs remain readable without these additional files. The dashboard exposes the
audit tables alongside portfolio metrics. Factor correlation/forward-return
research diagnostics are a separate future extension, not part of the initial
portfolio report.

`Research.compare_runs([...])` permits different factor/policy/schedule/constraint
configurations only when the market revisions, warm-up requirements, periods,
coverage policies, engine environment, and execution settings match. Funding and
sale-proceeds assumptions must also match. Different factor lookbacks can require
different warm-up selections and therefore fail the strict comparison check.

## Adding a custom portfolio policy

Policies implement `propose(scores, snapshot) -> TargetPortfolio`. `scores` is a
Polars frame with `symbol`, `score`, and factor columns for eligible stocks.
`snapshot` is an immutable mapping of current holdings to decision-time weights.
Construct one fresh policy per experiment; instance state can retain holding-rule
history across decisions without leaking into a later experiment.

```python
from research.portfolio import PortfolioRegistry, TargetPortfolio

class BestOne:
    def propose(self, scores, snapshot):
        if scores.is_empty():
            return TargetPortfolio(no_change=True)
        best = scores.sort(["score", "symbol"], descending=[True, False])["symbol"][0]
        return TargetPortfolio(weights={best: 1.0}, reasons={best: "highest score"})

policies = PortfolioRegistry()
policies.register("best_one", "1", lambda config: BestOne())
custom_spec = replace(spec, portfolio=ComponentSpec("best_one"))
# research.factor_backtest(..., spec=custom_spec, portfolio_registry=policies)
```

The same engine still applies position caps, cash reserve, funding, and fills.
A factory can declare validated `default_config` and `parameters` when registering
more configurable policies. Custom factors similarly register with
`FactorRegistry.register(name, version, factory)`, where the factory receives a
`FactorSpec` and returns an object whose `calculate(frame)` yields one Polars
Series value per input bar. Specify sufficient `lookback` for the feature.
Increment the registered component version whenever custom behavior changes.
External custom source is not automatically archived or fingerprinted.

## Data assumptions and boundaries

The first release requires one common calendar/timeframe and USD long-only
holdings in shared cash. It does not use the older per-ticker isolated initial
capital split mode. Raw and causal processed inputs retain existing strict
preflight checks, exact source revisions, and completed-bar/as-of rules.

Prices remain in their recorded basis. There is no new split adjustment,
corporate-action share accounting, dividend credit, interest accrual, or
reconstructed historical constituent list. A current symbol list can introduce
survivorship bias. These issues matter before interpreting a long historical
simulation as an investable result.

`skip_initial_missing_bars=True` permits genuinely late histories to join after
warm-up; it does not fill interior gaps. `skip_problematic_tickers=True` is
optional and defaults to false. It removes a problematic ticker from the whole
run based on full-period validation, so later gaps can change earlier portfolio
choices. The preflight and saved manifest explicitly report this retrospective
selection risk as well as each ticker error. Missing bars never fabricate a
liquidation price or sale proceeds.
