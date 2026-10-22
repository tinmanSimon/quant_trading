# Writing research analyses

`research.analysis` owns the reusable calculation and result contract.
The dashboard supplies bars and settings, runs analyses when you click Apply,
and displays their plain-data results. Scripts also run from research code;
they do not need Streamlit, Plotly, a strategy class, or a portfolio simulation.

## Start with the example

The root `secrete_analysis/` package is ignored by Git. A working example lives
in `secrete_analysis/analyses/ma_rsi_example.py`. The generic, tracked template
is in `examples/secrete_analysis/`; copy that directory to the project root on
a fresh checkout. Keep personal modifications in the ignored copy.

In Market data, click **Reload analyses**, add **Example: MA crossover and RSI**,
adjust its settings, and click **Apply**. It reports independent entry/exit
observations, plots both averages on price, plots RSI in a synchronized panel,
and supplies signal tables and navigation events. Optional trend confirmation
uses a trend's observation endpoint, not its fitted starting point.

The private package can contain:

```text
secrete_analysis/
├── __init__.py
├── analyses/
│   ├── __init__.py
│   └── ma_rsi_example.py
└── helpers/
    ├── __init__.py
    └── validation.py
```

Import shared helpers relatively: `from ..helpers.validation import ...`.
Discovery imports trusted local Python in a private namespace and does not
modify `sys.path`. It finds decorated functions defined in modules under
`analyses/`, including nested packages. An absent private directory is normal.
This is code execution with the application's permissions, not a sandbox.

## A script is a function

```python
import polars as pl
from research.analysis import analysis, crossed_above

@analysis(
    "rsi_recovery",
    label="RSI recovery",
    parameters={
        "period": {"minimum": 2, "maximum": 100_000},
        "threshold": {"minimum": 0, "maximum": 100},
    },
)
def run(ctx, period: int = 14, threshold: float = 30):
    data = ctx.features(rsi=ctx.rsi(period))
    data = data.with_columns(recovery=crossed_above("rsi", threshold))
    findings = ctx.selected(data.filter("recovery"))
    result = ctx.result()
    result.text(f"Found {findings.height} RSI recoveries.")
    result.signals("Recoveries", findings, price="close")
    return result
```

The decorator leaves the function callable. Typed arguments and defaults define
settings; `parameters` adds JSON Schema constraints, titles, and descriptions.
Use ordinary Python validation for relationships between settings, as the MA
example does for `fast < slow`. Unknown settings and invalid values are rejected.
Give each analysis a stable, unique name. Add several instances with different
settings to compare ideas.

Functions must be synchronous and take context first, followed by named settings.
`Literal` choices and typed lists/nested settings use the existing schema controls.
Return the builder from `ctx.result()`. Every result requires text or at least
one table; an empty table is valid. Chart outputs are optional.

## Inputs, alignment, and historical accuracy

| API | Meaning |
| --- | --- |
| `ctx.bars` | Selected completed bars with original `source_index` values starting at zero. |
| `ctx.history` | Available earlier initialization bars plus the selected completed bars. Earlier indices are negative. |
| `ctx.features(**features)` | Copies history and attaches named, exactly aligned indicator outputs. |
| `ctx.selected(frame)` | Validates timestamp/index identities and trims initialization rows from findings. |
| `ctx.at(index_or_timestamp)` | A restricted context containing observations through that original bar. |
| `ctx.metadata` | Source/timeframe, selected and initialization counts, cutoff, and calculation provenance. |

All frames are Polars DataFrames. Public frames and metadata are copies. Input
timestamps must be increasing and unique, and an input must represent one series.
Indicators preserve rows and null warmup values. The toolkit rejects mismatched
feature timestamps/indices instead of silently joining the wrong rows.

Evaluate crossovers, rolling windows, and other shifts on history **before**
calling `selected`. This allows the first selected bar to compare with an earlier
bar. `crossed_above(a, b)` means current `a > b` and previous `a <= b`;
`crossed_below` reverses that relation. Operands can be column names, Polars
expressions, or numbers. Missing current or previous values produce false.

Helpers include `sma`, `ema`, `wma`, `moving_average`, `rsi`, `macd`, `atr`,
`bollinger_bands`, `adx`, `stochastic`, and `obv`. Their parameters follow the
underlying `research.analysis.indicators` functions. Periods count input bars.

```python
data = ctx.features(
    fast=ctx.ema(20),
    slow=ctx.sma(60),
    momentum=ctx.macd(),           # momentum_macd, momentum_signal, momentum_histogram
    volatility=ctx.atr(14),
)
macd_only = ctx.features(macd=ctx.macd().select("macd"))
```

Equivalent calculations share a cache within one execution of the analysis set.
Results depend on the supplied initialization history; recursive indicators can
depend on its starting point. The toolkit does not invent missing history.

`ctx.trends(config=TrendConfig(...))` or `ctx.trends(max_history=2000)` returns
historical trend records. Each record is observable at `end_index/end_time`.
Its fit may start earlier, even at a negative initialization index. Do not label
that earlier start as a signal known at that time. Initialization endpoints are
omitted by default; `include_history=True` includes them.

Vectorized expressions are convenient, but loops are ordinary Python too:

```python
findings = []
for row in ctx.bars.iter_rows(named=True):
    as_of = ctx.at(row["source_index"])
    # Calculate a custom condition using as_of.history or as_of helpers.
    # Append original source_index/timestamp and your explanatory columns.
```

Prefer calculating causal indicators once and iterating their aligned rows when
possible; recalculating every prefix can be expensive. A full history frame also
allows arbitrary Python to look ahead. The toolkit's alignment checks cannot
prove that a custom expression avoids future data; use `ctx.at` when evaluating
an idea that must have access only to a historical prefix.

Analyses are cached by completed bars by default. If a script deliberately uses
the exact clock cutoff in addition to observations, declare
`@analysis("name", uses_cutoff=True)` so that time also enters its cache key.

## Text, tables, chart layers, and navigation

| Builder method | Output |
| --- | --- |
| `text(markdown)` | Custom text rendered without enabling raw HTML. |
| `table(name, frame)` | Arbitrary tabular columns, including empty results. |
| `markers(name, frame, price="close")` | Points on price or a declared indicator panel. |
| `line(name, frame, value="close")` | A line with null gaps preserved. |
| `band(name, frame, lower="low", upper="high")` | A shaded region between two columns. |
| `events(name, frame, label=..., start=..., end=...)` | Navigable point or interval findings. |
| `signals(name, frame, price="close", label=...)` | A table, markers, and linked navigation events together. |

Chart and event positions must identify selected completed bars using original
`source_index` and/or `timestamp`. If both are supplied, they must agree. Tables
may contain any research columns. Builders serialize timestamps to ISO strings
and reject nonfinite numeric output. Null chart values are supported.

Chart methods accept colors and styles. A separate pane is optional:

```python
result.line("RSI", ctx.selected(data), value="rsi",
            panel={"id": "rsi", "title": "RSI", "range": [0, 100]})
```

Use the same panel declaration for several lines that should share an axis.
Price is the default when no panel is supplied. Panes follow the main chart.
The dashboard derives layer controls from the result; hiding layers or the
entire analysis never recalculates it. Text-only results have no chart controls.

Navigation above the chart offers a group filter, searchable findings, Previous,
and Next. Navigation preserves the current zoom and centers points or interval
midpoints. Selecting a signal
table row also navigates. Hidden layers retain their findings. Coordinates stay
in original bars even when the chart groups or loads only part of the data.
Navigation and viewport loading update only the chart section, reusing the
verified data snapshot and prepared analysis layers. They do not reread the
dataset or rebuild analysis controls and result tables. Changing page settings,
reloading analyses, or clicking Apply runs the full page with fresh inputs.

Methods are chainable. Call `result.to_logic_result()` for the plain result, then
`to_dict()` for JSON-safe persistence; `run_script` does the first conversion.
Calculated output can contain sensitive findings even though private source is
ignored by Git. Dashboard workspaces and saved result snapshots remain local.

## Run the same analysis from research

```python
from datetime import UTC, datetime
from research.analysis import LogicContext, LogicRunner, LogicSpec, load_registry

registry = load_registry(project_dir="/path/to/quant_trading")
context = LogicContext(
    frame=selected_bars,
    support_frame=earlier_bars,   # optional; same series, before the selected range
    timeframe="1d",
    identity=("verified dataset revision", "AAPL"),
    cutoff=datetime.now(UTC),
)
run = LogicRunner(registry).apply([
    LogicSpec("experiment-1", "ma_rsi_example", "1", {"fast": 20, "slow": 60}),
], context)
if run.errors:
    print(run.errors)
else:
    result = run.results["experiment-1"]
    print(result.data["analysis"]["text"])
    print(result.tables)
```

For a decorated function you already imported, use
`run_script(function, context, fast=20, slow=60)` to validate settings and obtain
the same result. Or call `function(analysis_context, ...)` directly for normal
Python use, then finalize its builder.

## Multi-ticker analysis and saved runs

The dashboard's **Analysis** page runs the same analyses across a selected list
of local tickers. Choose the provider, timeframe, raw or processed series, date
range, and tickers (including pasted lists or all available tickers). Add one or
more analyses, configure their settings, then click **Run analysis**. Different
instances of an analysis retain independent settings and summaries.

The page displays overall metrics, a ranked ticker summary, and original text
and tables for the selected ticker. It does not render charts. Category/count
tables, such as `outcome` / `count`, automatically become summable metrics.
Single-row numeric summaries also supply ranking columns; non-count quantities
have no overall aggregation until one is selected. **Summary settings** can
select table columns and per-ticker reducers, overall sum/mean/min/max, or ratios
of named metrics. Ratios use matching tickers' summed numerators and denominators,
not the average of their individual percentages. Missing values and failures
remain distinct from zero. Free-form prose is shown without guessing numeric
totals. Sorting, filtering, and changing summary rules do not execute analyses.

Each run pins its source revision IDs, analysis settings, and code fingerprints.
Earlier available bars supply indicator initialization history just as in Market
data; the chosen date range bounds the reported analysis. There are no automatic
downloads or backtest portfolio/account rules. Missing data and calculation
failures are reported independently; successful analyses remain available.

**Concurrent tickers** bounds the number of loaded ticker histories. The default
is one. Increasing it can help workloads that release Python's GIL, but uses
more memory and does not guarantee faster Python-only scripts. **Stop after
current tickers** prevents further submissions; active calculations finish and
remain available to inspect or save. Existing scripts may still construct chart descriptors internally;
the batch runner discards them before persistence.

Dashboard runs are temporary by default. After completion (including a stopped run),
**Save analysis** keeps the results; **Delete analysis** removes the selected saved run.
Unsaved results are available only in the current session until the next run.
Historical saved batches live under `<research runs directory>/analysis/<run-id>/`.
Each contains a manifest with settings, source identities, statuses, and compact
metrics, plus separately checksummed compressed JSON for ticker details.
Aggregation preferences are stored separately. **Saved analysis run** reopens a
batch without executing its scripts. An interrupted server does not automatically
resume calculations, but completed saved results remain readable. Market data's
single latest workspace snapshot remains separate from this batch history.

The same batch engine works from research code:

```python
from datetime import UTC, datetime
from data_pipeline import DataPipeline
from research.analysis import LogicSpec, load_registry
from research.analysis.batch import BatchRequest, BatchRunner, BatchStore

store = BatchStore("runs/analysis")
runner = BatchRunner(DataPipeline("data"), load_registry(), store)
manifest = runner.execute(
    BatchRequest(
        tickers=("AAPL", "MSFT"), timeframe="1d",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC), workers=1,
    ),
    [LogicSpec("example", "ma_rsi_example", "1", {"fast": 20, "slow": 60})],
)
saved = store.load_ticker(manifest["run_id"], "AAPL")
```

`create(request, specs)` followed by `run(run_id, cancel=event)` provides the
same execution with explicit preparation and cooperative cancellation. A run
can be executed once; start another run to calculate again. Loading saved results
does not require the original private script to remain installed.

## Reloading and compatibility

Script and helper source fingerprints participate in cache identity, including
helpers imported by relative imports. Edits preserve the stable name/version
and configured instance, but invalidate its previous calculation. Click Apply
after changes. Reload analyses also repeats module loading when a dependency
has been fixed. Import errors and duplicate IDs appear in discovery issues;
successful analyses remain available. A calculation failure is isolated to that
instance. Results store a fingerprint, not private source code.

The public imports remain in `research.analysis`; private scripts do not need to
change imports when implementation files move. Internally the package is grouped
by responsibility:

- `core/`: analysis contracts, registry, input boundaries, and execution for one
  data context.
- `scripting/`: `@analysis`, function adapters, private discovery, `AnalysisContext`,
  result builders, and signal expressions.
- `batch/`: multi-ticker orchestration, metric aggregation, and saved batch runs.
- `indicators/`: shared numerical indicators.
- `trend/`: trend calculations, selection, and declarative trend presentation.
- `builtins/`: discoverable indicator, trend, and open-price-band analyses.

The former loose infrastructure modules have been removed. Import indicators
from `research.analysis.indicators` and trend analysis from
`research.analysis.trend`; their former top-level packages have also been removed.
The research analysis packages do not import the dashboard.
