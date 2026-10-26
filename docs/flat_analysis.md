# Flat analysis

Flat analysis finds periods whose candles all fit inside a narrow horizontal
price band. It uses every bar's high and low, including wicks, from the original
selected completed data. The same built-in analysis, registered as `flat`, runs
on the **Market data** and **Analysis** pages.

## Settings and definition

| Setting | Default | Meaning |
| --- | --- | --- |
| `min_duration` | `20` | Minimum number of observed bars, including both endpoints; at least 2. |
| `max_duration` | `200` | Maximum number of observed bars; must be strictly greater than `min_duration`. |
| `max_range_pct` | `2.0` | Maximum total band width as a percentage of the band's midpoint. |

These defaults are illustrative starting settings, not validated trading
thresholds. Duration settings must be integers, with
`2 <= min_duration < max_duration`. The maximum range must be finite and
strictly between 0% and 200%. A duration counts observations, not elapsed clock
time: overnight gaps do not add bars.

For an interval from index $s$ through index $t$, let $n=t-s+1$. Its bounds and
midpoint are:

$$
L=\min_{i=s}^{t}(\operatorname{low}_i),\qquad
H=\max_{i=s}^{t}(\operatorname{high}_i),\qquad
C=L+\frac{H-L}{2}.
$$

Its total percentage width is:

$$
R=100\frac{H-L}{C}.
$$

The interval qualifies when:

$$
\operatorname{min\_duration}\le n\le\operatorname{max\_duration}
\quad\text{and}\quad
R\le\operatorname{max\_range\_pct}.
$$

Both duration boundaries and the width threshold are inclusive. For example,
low 99 and high 101 give midpoint 100 and total width 2%. That is a band
extending 1% below and above its midpoint, not a 2% offset on each side. The
midpoint comes from the two extreme prices; it is not an average candle price.
Multiplying every high and low by the same positive constant preserves the
percentage width and qualification.

Every high and low must fit within the band. A large rise followed by an equal
fall cannot cancel out through averaging or a near-zero fitted slope. A gradual
upward or downward drift can qualify while its entire range remains within the
width limit. Individual candles need not have equal heights. A single wide
candle can prevent any interval containing it from qualifying.

Flat analysis is separate from the directional trend detector. A rejected
upward or downward trend does not establish that a flat period exists.

## Current periods and historical summaries

At each completed endpoint $t$, the detector finds the **earliest qualifying
start** within the duration limits. With zero-based indices, minimum duration
$m$, and maximum duration $M$, eligible starts satisfy:

$$
\max(0,t-M+1)\le s\le t-m+1.
$$

The result is the longest qualifying interval ending exactly at $t$ within the
configured maximum. If no interval meets both duration and width limits, there
is no current flat period.

The default result is a compact, strictly disjoint historical summary. After
computing endpoint information in a forward scan, selection works backward
from the final completed bar:

1. At the current cursor, take its earliest-start qualifying period.
2. If a period `[start, cursor]` exists, retain it and move the cursor to
   `start - 1`.
3. Otherwise, move the cursor back by one bar.
4. Repeat while at least the minimum duration remains.

Intervals do not share even a boundary bar. Newer endpoints take priority; at
each chosen endpoint, the earliest qualifying start wins. The summary contains
the current period whenever it exists, followed by selected disjoint periods
before it, newest first. It is not an archive of every overlapping or nested
qualifying interval, nor an optimization of the total number of covered bars.

Consecutive retained periods may be adjacent. A continuously narrow history
longer than `max_duration` can produce several adjacent capped periods. Their
boundaries reflect the duration cap and selection rule; they do not establish
that the market changed regime at those bars.

For detailed historical research, `analyze_history` returns the earliest-start
qualifying period at every eligible endpoint where one exists. Those records
can overlap. Applying `select_periods` to them produces the same disjoint
summary. Neither API materializes every qualifying start at every endpoint.
Use `analyze_current` on a prefix or `replay` to inspect what was available at a
particular time.

## Historical accuracy and input boundaries

An interval ending at bar 100 uses only highs and lows through bar 100. A
period spanning bars 60–100 is observable at bar 100; its start does not mean
it was known at bar 60. Appending later observations does not change an earlier
endpoint's period for the same input snapshot and configuration.

**The final summary's membership can change when new bars arrive.** A new
current period may overlap and replace periods selected by an earlier summary.
For a historical decision, analyze the prefix available then or use `replay`;
do not treat today's summary membership as a signal known in the past.

The dashboard passes completed bars from the selected date range and verified
data revision. It does not use grouped display candles, the zoom window, or
silently added initialization bars. There is no automatic download or hidden
lookback limit. `max_duration` explicitly caps each candidate period; historical
analysis still considers the entire selected date range. Changing the selected
start date, source data, or settings can change a recomputed history.

The Python package accepts a single chronological series with unique increasing
timestamps, finite positive highs and lows, and `low <= high` for every bar.
If `open` or `close` columns are supplied, their prices must also be finite,
positive, and inside their own bar's low/high range.
It does not infer bar completion from an exchange calendar; callers must supply
completed bars. `as_of` restricts the series to a historical cutoff. Missing
bars are not filled and do not contribute to duration.

## Dashboard use

On **Market data**, choose **Flat analysis** under **Chart logic**, add it,
configure the duration limits and maximum total band width, then click **Apply**.
The result contains **Flat periods** and **Flat summary** tables. The chart draws
horizontal lower, midpoint, and upper lines bounded by each selected period.
**Show flat bands** and **Show period boundaries** only change presentation;
they do not recalculate periods or change the saved tables. Selecting a period
row navigates to its interval on the chart.

**Flat periods** reports `lower_price`, `upper_price`, `center_price`,
`range_pct`, and whether the period `ends_at_current_bar`, alongside its
boundaries and duration. **Flat summary** reports `period_count`, `source_bars`,
`covered_bars`, `coverage_percent`, `current_flat`, and `current_duration`.
Covered bars count only the disjoint summary periods.

On **Analysis**, select the same analysis to run it across local tickers. The
existing batch runner displays text and tables without charts, with bounded
ticker concurrency, summary metrics, and the usual **Save analysis** and
**Saved analysis run** workflow. Both pages use the same numerical definition.

Engine version `2.0.0` replaces the former log-close slope detector with this
high/low band detector. Restored dashboard selections from flat engines `1.0.0`
and `1.1.0` retain their duration settings, remove the former slope settings,
and receive the new default `max_range_pct=2.0` in their draft configuration.
If the old selection has no maximum duration, the draft receives
`max_duration = max(200, min_duration + 1)`. Unknown configuration fields are
preserved for validation rather than silently discarded.

The interface announces the changed definition. Review the band width and
duration limits, then click **Apply** in Market data or **Run analysis** in
Analysis. Migration does not rewrite applied settings, result snapshots, or
previously saved batch results. Historical saved batches retain their original
calculation and outputs. The chart's saved `show_fit` display key remains
compatible, with its label changed to **Show flat bands**.

## Python API

```python
from research.analysis.flat import (
    FlatConfig,
    analyze,
    analyze_current,
    analyze_history,
    analyze_summary,
    replay,
    select_periods,
)

config = FlatConfig(
    min_duration=20,
    max_duration=200,
    max_range_pct=2.0,
)

# bars is a Polars frame with completed timestamp/high/low observations.
summary = analyze(bars, config=config)  # Same behavior as analyze_summary.
period_table = summary.to_frame()
current = analyze_current(bars, config=config)

# Opt in to one qualifying period per historical endpoint.
history = analyze_history(bars, config=config)
selected = select_periods(history.periods)

# Evaluate independent historical prefixes every 20 eligible endpoints.
for snapshot in replay(bars, config=config, step=20):
    current_at_that_time = snapshot.periods
```

The analysis functions accept `timestamp_col`, `high_col`, `low_col`, `as_of`,
and `config`. `analyze_summary` and `analyze_history` also accept a progress
callback. `replay` accepts explicit `as_of_values` or a positive integer `step`.
There is no `price_col` or slope-bound setting in this definition.

Results contain `periods`, configuration, source count, analysis boundaries,
analysis mode, status, and algorithm version. Status distinguishes `ok`,
`no_flat_period`, and `insufficient_data`. Each period records its original
start/end indices and timestamps, duration, lower and upper prices, midpoint,
and percentage range. `range_pct` is already a percentage: 2.0 means 2%.
`to_frame()` returns `period_id`, `start_index`, `end_index`, `start`, `end`,
`bars`, `lower_price`, `upper_price`, `center_price`, and `range_pct`.

## Cost and storage

For positive prices, extending an interval cannot reduce its high/low percentage
range. The scan can therefore maintain the longest valid suffix ending at each
bar, advancing its left boundary when the duration or width becomes too large.
Two monotonic queues maintain the current maximum high and minimum low. Each
bar enters and leaves each queue at most once.

With $N$ input bars, minimum duration $m$, and maximum duration $M$:

- The complete endpoint scan is $O(N)$ time; it does not test every interval.
- Queue memory is $O(\min(N,M))$. Input and endpoint arrays use $O(N)$ memory.
- Historical output retains at most one period per endpoint, so it is $O(N)$.
- The disjoint summary has at most $\lfloor N/m\rfloor$ periods; its backward
  selection is also $O(N)$.

Calling an analysis API again rebuilds its scan for the supplied input. Repeated
independent prefix calls through `replay` take $O(N^2)$ total time when every
endpoint is requested, compared with one $O(N)$ `analyze_history` scan. The
maximum duration caps period length even though the linear-time algorithm does
not require it for avoiding exhaustive searches.

Runtime still depends on loading, validation, the number of tickers, and
presentation or persistence overhead. Ticker concurrency can improve throughput
but uses more memory. Changing calculation settings requires a new analysis;
display-only changes reuse the existing result.
