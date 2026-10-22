# Quant Trading

An OHLCV ingestion package with interchangeable providers, local raw/processed
Parquet storage, ordered processors, revision history, and a Python/CLI query API.
Yahoo Finance and Massive (formerly Polygon.io) are the included vendor adapters.

The `data_pipeline` package also handles batch fetching and per-ticker outcomes.
The `research` package adds strict local-data preflight,
independent-account backtests, versioned strategies and saved comparisons.
A local Streamlit dashboard provides dataset navigation and interactive charts.

## Research quick start

Create the environment with Python 3.12 or newer (required by the pinned
dependencies), then activate it once per terminal:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e . --no-deps
```

Fetch every ticker independently. Failures identify their ticker, other tickers
are still attempted, and the command exits nonzero if any request failed:

```bash
quant-research fetch --tickers AAPL MSFT --timeframe 1d \
  --start 2024-01-01 --end 2025-01-01 --skip-missing-ohlc
```

Omitting `--skip-missing-ohlc` keeps strict Yahoo validation. Enabling it records
each wholly missing OHLC bar's UTC timestamp and reason alongside its immutable
dataset revision; missing volume is never silently changed to zero. By default,
existing overlapping data raises an error instead of being overwritten. The
optional policies described below can skip covered requests, refresh overlapping
ranges, or fetch only missing portions and merge locally. Fetch requests still
obey Yahoo's availability/retention limits.

Batch fetching can also be used directly without creating a research service:

```python
from datetime import UTC, datetime
from data_pipeline import DataPipeline

pipeline = DataPipeline("data")
report = pipeline.fetch_many(
    ["AAPL", "MSFT"], timeframe="1d",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2025, 1, 1, tzinfo=UTC),
    skip_missing_ohlc=True,
)
print(report.to_frame())
```

`BatchFetchReport`, `FetchOutcome`, and the standalone `fetch_many(pipeline, ...)`
function now live in `data_pipeline`. Omitting the Python `skip_missing_ohlc`
option preserves the pipeline's registered provider configuration; an explicit
boolean uses a Yahoo provider configured just for that batch. Each successful
ticker is saved independently, even if another fails. Call
`pipeline.fetch_many(...)` (or `research.pipeline.fetch_many(...)` when using a
research service), and import reports directly from `data_pipeline`.
Invalid batch options raise `InvalidDataRequestError`.

`overlap_policy="error"` is the default. Use
`pipeline.fetch_many(..., overlap_policy="refresh_overlap")` to skip requests
whose entire range is already covered by matching active raw datasets, without
contacting the vendor. For a partial overlap, the pipeline expands the fetch to
the union of the request and all overlapping datasets, including any additional
datasets reached by that expansion. A successful validated response becomes
one active raw revision, retaining the old revisions as history. For example,
a request for October 1–10 overlapping stored October 6–15 fetches October 1–15
instead. API date bounds remain inclusive-start/exclusive-end.

Use `pipeline.fetch_many(..., overlap_policy="fetch_missing")` to download only
the uncovered portions of the original requested range. Fully covered requests
are skipped without a vendor call. For example, existing October 1–10 plus a
request for October 1–15 downloads only October 11–15 (with API end bounds still
exclusive). Downloaded bars are merged locally with overlapping or adjacent
active datasets into one new revision. Existing bar values remain unchanged;
the old revisions stay readable by ID. Multiple missing ranges are requested
separately, while the combined result is published atomically after all requests
and validations finish. Any failed range leaves the original datasets active.
An empty range is recorded and other ranges continue: if all return no bars,
nothing is published; if some return bars, those bars are saved with a warning
identifying the empty ranges. No prices or bars are invented for those gaps.
After a merge, an empty internal gap is inside the combined first-to-last span
and is considered covered on subsequent requests; empty leading or trailing
ranges remain outside stored coverage.

Coverage starts at the actual first stored timestamp and extends through one
nominal bar interval after the last timestamp (calendar months for monthly bars),
not through the original request bounds. Daily and longer bars use date labels;
coverage does not infer exchange sessions, holidays, or shortened closing bars.
Gaps between these spans remain uncovered, while this option does not detect
missing bars inside a dataset's first-to-last interval. Each new revision must
contain every previously stored timestamp it supersedes. In refresh mode, failed
or empty responses and incomplete replacements leave the old datasets active.
Concurrent conflicts are reported without overwriting another writer's updates.
Processed datasets derived from replaced inputs become inactive and need
rebuilding in either optional mode.

Reports distinguish `saved`, `refreshed`, `merged`, `skipped`, `empty`, and `failed`.
They include requested bounds, returned first/last timestamps, replaced dataset
IDs, and `fetch_ranges`, a JSON list of exact planned vendor ranges and their
statuses (`pending`, `fetching`, `fetched`, `empty`, or `failed`). `fetch_start/end`
is the envelope of these ranges; it does not imply that intervening stored data
was downloaded again or that a pending range was attempted after an earlier
failure. `row_count` is the total size of the saved revision, while
`fetched_row_count` counts newly downloaded bars. `warning_message` identifies
empty ranges when other ranges succeeded. For a skipped request, `dataset_ids`
lists the covering datasets and both row counts are zero.

The dashboard shows each ticker's planned vendor ranges before downloading and
updates their status as requests finish. Python callers can pass
`progress=callback` to receive the same events: `ticker`, one-based `index`,
`total`, `status`, requested bounds, and `fetch_ranges` with ISO start/end bounds,
status, and downloaded `row_count`. Progress reporting does not change storage
validation or publication.

The dashboard and `quant-research fetch` use the pipeline API directly.
As with other research source changes, this refactor changes the code fingerprint
recorded in new backtests. Existing runs remain readable; comparisons still
require matching fingerprints.

Run the three example strategies independently against every ticker:

```bash
quant-research backtest --tickers AAPL MSFT --timeframe 1d \
  --start 2024-03-01 --end 2025-01-01 --strategies strategies.example.json \
  --initial-cash 10000 --commission-bps 5 --slippage-bps 5
quant-research list-runs
```

All dates have an inclusive start and exclusive end. The fetch begins earlier
than the backtest because strategies need preceding warm-up history. These are
example historical daily ranges; choose recent ranges for Yahoo intraday data.
`python -m research` also works in place of `quant-research`.

Launch the dashboard from the repository root:

```bash
python -m streamlit run src/dashboard/app.py
```

### Processing data in the dashboard

Open **Process data**, select a provider, source interval, tickers, and active raw
batches. Add processors in execution order and edit their settings. Steps can be
reordered or removed. **Preview processing** runs the entire selected input and
shows input/output counts plus the first 100 rows without saving datasets.
**Process and save** creates immutable processed revisions separately for each
ticker and reports individual failures. Repeating the same pipeline on the same
raw revisions reuses the verified existing output; raw bars remain unchanged.

Use **Save pipeline recipe** to name and reuse an ordered configuration. Draft
steps, parameter edits, source selections, recipes, and the last save report are
stored per data directory in `.local/market_workspaces/`, including across new
dashboard sessions. Preview tables stay in the current session and disappear
when their inputs or configuration change. Recipes include exact processor
versions and effective configuration; identical configurations share a pipeline
ID even if their display name changes.

To strip bars outside a recurring clock window, add **Filter by time window**
(`filter_time_window`, version `1`). Configure **Start time (inclusive)**,
**End time (exclusive)** as local times such as `09:00` and `16:00`, and choose
**Window timezone** (default `America/New_York`). This keeps bars whose start
timestamps satisfy `09:00 <= local time < 16:00` on each date. Named timezones
handle daylight-saving offsets automatically; retained timestamps remain UTC
and their OHLCV values stay unchanged. The filter supports intraday bars only.
An end earlier than the start crosses midnight; equal times keep the full day.
If no bars match, processing reports the empty result instead of saving an empty
dataset. The fixed window does not infer holidays or early closes.

Place this step **before resampling** when removing unwanted hours. Filtering
uses bar start timestamps and does not split bars: a 09:00–10:00 bar is removed
by a window starting at 09:30, including its regular-session portion. A retained
bar that crosses the window's end stays whole. Accurate partial-bar trimming
requires finer input data. The subsequent resampler still requires bars aligned
to its own session opening and checks session completeness.

Python callers can import the class from its module:

```python
from datetime import time
from data_pipeline.processing.processors.time_window import TimeWindowFilter

window = TimeWindowFilter(start_time=time(9), end_time=time(16),
                          timezone="America/New_York")
processed = pipeline.process(raw_id, processors=[window])
```

For **Resample OHLCV**, choose inclusive session dates and a calendar from the
searchable dropdown, then select **Exchange regular hours** or **Custom session
hours**. The dropdown discovers calendars from the installed `exchange_calendars`
library, groups aliases under their canonical name, and offers calendars with
continuous sessions for the chosen dates. Examples include `XNYS` (also used by
the library for aliases such as `XNAS`), `XLON`, `XTSE`, and `XASX`. Calendars with
scheduled intraday breaks, or unavailable date coverage, are excluded with the
reason shown in **Calendars unavailable for these dates**. This is support for
the library's modeled schedules; consult the provider's bar conventions when
matching data to a calendar.

**Exchange regular hours** uses the calendar's exact openings, closings, holidays,
and early closes. **Custom session hours** uses its trading-date list and
holidays, but replaces opening and closing times with the local times you enter.
Choose a named timezone, initially the calendar's timezone, so daylight-saving
offsets are calculated separately for every date. Custom hours override early
closes as well; edit individual session JSON entries if a date needs different
hours. Select **Previous calendar day** when the opening belongs to the evening
before the session label. A closing clock time at or before the opening clock
time closes on the following day; equal times mean a full day. Ambiguous or
nonexistent local times at daylight-saving transitions are rejected explicitly.

Click **Generate session schedule** to apply those controls. Editing generator
controls alone leaves the current schedule unchanged; the JSON is the actual
processor configuration and remains directly editable. Generated timestamps
are UTC. For example, 04:00–20:00 America/New_York on 2024-10-11 becomes
08:00 UTC that day through 00:00 UTC the following day. Settings persist across
page navigation and new dashboard sessions.

The current resampler converts hourly bars into daily OHLCV. Its incomplete-session
policy can raise or drop; out-of-session and off-grid bars are always rejected.
Custom hours do not split mixed regular/extended-hours bars or repair alignment:
input must already match an hourly grid anchored to the configured opening.
The schedule is part of the saved pipeline identity. Review it when applying a
saved recipe to different dates. These calendar choices apply to processing;
the dashboard's Backtest page still uses XNYS and requires resampled session
boundaries to match its calendar.

Saved outputs have **Open in Market data** and **Use in Backtest** shortcuts.
Market data's processed layer and Backtest's processed source let you choose a
specific named pipeline. Backtest retains source selections and date/time bounds,
and requires all selected tickers and warm-up history from that exact pipeline;
it never substitutes raw data. A shortcut initially selects the output's full
date range, so adjust the backtest start to allow the strategy's warm-up bars.
Backtest manifests record pipeline settings, processed revisions, and raw-parent
revisions. Keep the raw parent files for processed backtests: preflight verifies
their checksums and contracts as well as the processed output. Processed charts
can still be viewed after their raw parents are deleted. Transformed prices have
a distinct price-basis label in saved results.

### Saved Market data workspace

Market data automatically remembers its provider, ticker search and selection,
series/timeframe, revision, date window, timezone, selected chart logic instances,
logic settings, original-bar page, and expanded sections. Returning from another page
or reopening the dashboard restores those controls. Calculation parameter edits are
saved when committed in their inputs, even before **Apply**; results continue
to use the last applied calculation settings. Removals also remain pending
until Apply. Display settings are saved separately per instance and update
the existing chart immediately, including while calculation edits are pending.

Each data directory has a separate workspace in `.local/market_workspaces/`.
The latest applied result bundle is saved as compressed JSON and restored without
recomputation only when the source contents, selection, applied parameters, and
logic versions still match. Changed data requires **Apply**. Missing logic modules
retain their saved configurations with a warning; remove or restore them before
applying. Unavailable dataset selections fall back to available options. Sidebar data and
runs directories are remembered too. These project-local files are ignored by Git.

Chart zoom, pan, and drag mode are remembered in the same browser for the most
recent 32 chart identities. Reopening the same data at the same dashboard address
restores that view; a changed dataset starts a fresh view. Browser storage must
be available for chart-view persistence.

### Dynamic chart logic

The **Chart logic** panel supports any number of independent analysis instances.
Choose a logic, click **Add logic**, configure it, and click the shared **Apply**
button. Add the same logic again to compare parameter sets. **Remove** changes
only the draft; Apply removes its output. Apply with no selected logic clears all
analysis overlays. Original candles, volume, and navigation remain available.

Apply validates all configurations before calculating. Calculations use original
selected bars, never grouped display candles or the zoom window. Each logic is
cached independently by its version, settings, and verified source contents.
Successful results remain available if another calculation fails; the failed
logic's old output is cleared and its error is shown. Chart navigation changes
only bounded display samples. Results include each instance's applied settings.
Finding navigation and viewport loading refresh only the chart section, reusing
the verified snapshot and prepared layers instead of rebuilding the entire page.

After Apply, **Chart display and filters** appears directly above the chart.
Every logic has a **Show on chart** toggle; plugins can provide their own display
settings in the same panel. Display changes never call `calculate`, alter the
complete saved result, or enter its cache key. Hiding, filtering, or showing a
result again takes effect immediately and preserves zoom. Invalid display
settings show an error and retain the last valid display. Calculation drafts
remain independent and still require Apply.

Trend's λ control is a searchable multiselect populated from the actual accepted
results, not a JSON editor or a guessed grid. Options stay available when other
filters hide the matching lines. **Show all λ values** bypasses the selection;
turn it off to select individual suggestions. An empty selection deliberately
hides λ-derived lines and produces an explanatory message. Redundancy and slope
filters are also display settings. Open-price bands have separate upper, lower,
and opening-reference visibility toggles.

Saved configurations from the earlier combined-settings interface are split
into calculation and display settings automatically. Older result snapshots
may contain only filtered overlays, so they require one Apply to rebuild the
complete archive. Subsequent display changes and restored complete results do
not rerun analysis.

**Open price bands** defaults to 5% above and below the session open. Its reference
can instead be the first selected bar's open or each bar's own open. For intraday
session mode, regular hours follow the XNYS exchange calendar, including holidays,
DST and early closes. Extended hours use the explicit US-equity convention of
04:00–20:00 New York time (17:00 close on early-close days). The display timezone
never changes the reference. A session needs its actual opening bar in the
selected data or the available earlier history from the same source; missing
opens are reported and skipped, never replaced with the first visible bar. Daily bars use the
provider's daily open; extended-hours opens cannot be inferred from daily bars.
Weekly/monthly bars support first-bar and each-bar references. Session outputs
do not connect overnight.

### Private research analyses

Reusable analysis infrastructure, indicators, and trend analysis now live under
`research.analysis`. Import trend analysis from `research.analysis.trend`.
Import indicators from `research.analysis.indicators`.

Write ordinary decorated Python functions in the ignored root package
`secrete_analysis/analyses/`. Typed settings become dashboard controls; scripts
can use aligned indicator/trend helpers and return custom text, tables, points,
lines, bands, and navigation events. Add multiple instances through **Chart logic**
and click **Apply**. Layer display changes reuse calculated results. Above the
chart, use the searchable findings, **Previous**/**Next**, or select a signal
table row to jump to the corresponding bars.

The local `ma_rsi_example.py` demonstrates configurable MA crossovers, RSI, and
optional causal trend confirmation. A generic copy is tracked under
`examples/secrete_analysis/` for fresh checkouts. **Reload analyses** discovers
scripts and imports edits; script/helper fingerprints invalidate stale results
while preserving configured instances. Private Python runs with application
permissions and should come from trusted local authors.

The **Analysis** page runs these same scripts across multiple local tickers,
without charts. It shows overall metrics, ranked ticker summaries, and individual
text/table results. Counts are summed automatically; other metrics and ratios
can be configured in **Summary settings** without changing scripts. Runs are temporary
by default. Click **Save analysis** after a run finishes to keep it
under `<runs directory>/analysis/`. Reopen it through **Saved analysis run**,
or use **Delete analysis** to remove the selected saved run. Market data continues
to save only its latest workspace snapshot. Concurrency is bounded, and stopping a batch preserves completed results.

See [Writing research analyses](docs/analysis_scripts.md) for the toolkit,
result builders, history/alignment rules, and running the same scripts from
research without the dashboard.

### Technical indicators

Market data includes these independently configurable chart logics:

| Logic | Calculation settings | Outputs |
| --- | --- | --- |
| Moving average | SMA/EMA/WMA, period, price source | Average |
| RSI | Period, price source | RSI |
| MACD | Fast, slow and signal periods, price source | MACD, signal, histogram |
| ATR | Period, normalized percentage option | ATR or normalized ATR |
| Bollinger bands | Period, upper/lower deviations, price source | Middle, upper and lower bands |
| ADX | Period | ADX, +DI, −DI |
| Stochastic | Lookback, %K smoothing, %D period | Slow %K and %D |
| OBV | Price source | Cumulative signed volume |

Use **Add logic** and **Apply** as above. Multiple instances can compare, for
example, RSI(7) with RSI(14), or SMA(20) with EMA(50). Indicators appear in
panels below price and volume. Moving-average instances share one **Moving averages**
panel and vertical scale, so different periods or types can be compared for
crossovers. Each retains its own settings, color, visibility and table; hover
labels identify the average type, period and price source. For a 20-day/60-day
comparison, use daily bars with periods 20 and 60. Other indicators, including
Bollinger bands, retain separate panels.
Horizontal pan and zoom stay synchronized; each panel has its own vertical
scale. Colors, line widths, individual series visibility, and supported reference
levels update through **Chart display and filters** without recalculation.
MACD also has separate positive/negative histogram colors.

Calculations use completed original bars and available earlier bars from the
same selected data source/revision for initialization. Earlier bars do not
appear in the selected chart or result tables. Missing earlier history is not
downloaded automatically: values remain null until enough bars are available.
The result metadata records the actual history start, initialization count,
parameters, units, warmup, and algorithm/backend versions. Recursive indicators
depend on their starting history; adding earlier history can change their
values. They do not reset at trend analysis's moving-window boundaries.

Full-resolution tables are retained in saved results and shown in pages of up
to 500 rows, initially on the latest page. Chart transport uses bounded samples;
these samples never replace the calculation or table values. Large datasets
with many indicators still require time and memory to prepare and save tables.

Research uses the same standalone `research.analysis.indicators` package, with no dashboard
or network dependency. Install the normal project requirements, which pin
`TA-Lib==0.8.2`, then pass one ordered Polars series:

```python
from research.analysis.indicators import moving_average, rsi, macd, atr

# bars: Polars DataFrame with unique, increasing Date/Datetime timestamps.
# Supply earlier history before calculating, then slice the desired research range.
rsi_result = rsi(bars, period=14, price_col="close")
ema_result = moving_average(bars, kind="ema", period=50)
macd_result = macd(bars, fast=12, slow=26, signal=9)
atr_result = atr(bars, period=14, normalized=True)

values = rsi_result.frame  # source_index, timestamp, value; warmup stays null
settings_and_history = rsi_result.metadata
features = bars.join(
    values.select("timestamp", "value").rename({"value": "rsi_14"}),
    on="timestamp", how="left",
)
```

The package also exports `bollinger_bands`, `adx`, `stochastic`, and `obv`.
Results preserve input row alignment. Research callers provide completed bars
and a consistent price basis; optional instrument/timeframe identity columns
must each contain one non-null value. Required numeric inputs must be finite
and non-null. No forward filling, resampling, or future data is used.

Conventions are explicit: RSI/ATR/ADX use Wilder smoothing; EMA uses an SMA
seed; Bollinger bands use an SMA center and population standard deviation;
stochastic uses simple smoothing; OBV starts with the first bar's volume.
MACD follows TA-Lib's aligned EMA seeds and common signal warmup, so its initial
values need not equal subtraction of two independently seeded EMA calls.
Flat-price RSI is zero. Normalized ATR is `100 * ATR / close`, with zero for a
zero close after warmup. Full per-output warmup and initialization details are
available in `IndicatorResult.metadata` and the public function docstrings.

### Adding a chart logic module

Create one Python file under `src/research/analysis/builtins/`. A concrete
`BaseChartLogic` subclass is automatically discovered, including subpackages.
No registry entry, package export, dashboard branch, or renderer edit is needed.
New modules appear on the next dashboard rerun; restart after editing an already
imported module. Discovery imports trusted local Python code. Broken imports and
duplicate name/version pairs are reported without hiding unrelated logics.

For example, save this as `src/research/analysis/builtins/close_offset.py`:

```python
from research.analysis import BaseChartLogic, LogicResult


class CloseOffset(BaseChartLogic):
    name = "close_offset"
    version = "1"
    label = "Close offset"
    description = "A percentage offset from each original closing price."
    required_columns = ("timestamp", "close")
    config_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "percent": {"type": "number", "default": 5.0,
                        "title": "Offset (%)", "minimum": -100},
        },
    }

    def calculate(self, context, config, progress=None):
        prices = context.frame["close"].to_list()
        return LogicResult(overlays=[{
            "type": "line", "id": "offset", "name": "Close offset",
            "x": list(range(len(prices))),
            "y": [price * (1 + config["percent"] / 100) for price in prices],
            "line": {"color": "#2563eb", "width": 2},
        }])
```

The JSON Schema generates the controls and validates settings. Common numbers,
booleans, choices, strings, optional fields and nested objects use normal inputs;
arrays and maps use JSON editors. Optional schema `x-ui-groups` entries
(`title`, `fields`, `expanded`) organize settings into collapsible sections;
unlisted fields remain available. Override `validate(config)` for constraints
between fields. `supported_timeframes` can restrict inputs. `calculate` receives
`LogicContext` with original `frame`, `timeframe`, source `identity`, `cutoff`,
session timezone and optional supporting data. Use `context.completed_frame`
and `input_policy = "completed"` for closed-bar analyses. Set `uses_cutoff = True`
only if the calculation depends on exact cutoff time beyond bar completion.

Calculation and display settings have separate contracts. `config_schema`
defines settings passed to `calculate`; `display_schema` defines the live
chart controls. Their top-level field names must be distinct. Override `display_schema_for(result)` to populate choices from
the complete result. For suggested array selections, provide `items.enum` for
nonempty choices or `x-ui-options` for a list that may be empty; the dashboard
renders a searchable multiselect. `validate_display(config, result)` handles
cross-field constraints, and `present(result, config)` returns the filtered or
styled view. The framework supplies a defensive copy of the calculated result;
`present` must not call `calculate`, read new source data, or rerun analysis.
No central registry or dashboard branch is needed for these controls.

For example, add a display-only color selector and presentation method inside
your logic class:

```python
display_schema = {
    "type": "object", "additionalProperties": False,
    "properties": {"color": {"type": "string", "default": "blue",
                             "enum": ["blue", "green", "orange"]}},
}

def present(self, result, config):
    for overlay in result.overlays:
        overlay.setdefault("line", {})["color"] = config["color"]
    return result
```

Return the **complete, unfiltered** calculation as a `LogicResult` with
JSON-compatible `overlays`, optional `tables`
(`{"name": "Results", "rows": [{...}]}`), `messages` (strings), and optional
`data` (a JSON object containing any additional archive needed by `present`).
Supported overlay primitives are:

| Type | Coordinates | Purpose |
| --- | --- | --- |
| `line` | ordered `x`, matching `y` | Curves or horizontal segments |
| `markers` | ordered `x`, matching `y` | Events or isolated levels |
| `bar` | ordered `x`, matching `y` | Histograms |
| `band` | ordered `x`, `lower`, `upper` | Shaded envelopes |
| `log_line` | `start_index`, `end_index`, `slope`, `intercept` | Exact exponential fitted curves |

Coordinates refer to original selected bar indices, starting at zero. Each
primitive requires a locally unique `id`; the framework namespaces IDs by logic
instance. Optional `name`, `line`, `marker`, `fillcolor`, and `hovertemplate`
control appearance. Styles use scalar values per primitive, not per-point arrays.
Use null `y` values to disconnect line segments (both band
boundaries must share nulls). The renderer clips and bounds array samples for
transport without changing cached calculations. Keep descriptor counts bounded
by combining related series with gaps rather than creating one trace per bar.
`log_line` evaluates `exp(intercept + slope * (x - start_index))`.

To place primitives in a synchronized panel below volume, add the same
`"panel": {"id": "momentum", "title": "Momentum", "range": [0, 100]}`
declaration to each of them. The optional `range` fixes that panel's vertical
scale; omit it to autoscale from visible values. Panel IDs are local to each
logic instance by default. Set `"scope": "logic"` inside the panel declaration
to share that panel across instances of the same logic name and version.
All primitives sharing one ID must declare identical panel
settings. Panels appear in first-use order. Primitives without `panel` remain
on the price chart. Hiding every primitive in a panel removes that panel while
preserving horizontal navigation.

Bump `version` whenever calculation or parameter semantics change, and
`result_version` when the saved archive representation changes. All saved
results are declarative JSON, not pickled Python objects. New kinds of plots or
new input-source capabilities require a framework extension; new calculations
using this contract require only their module.

### Causal historical trend analysis

On **Market data**, choose **Trend analysis** under **Chart logic**, click
**Add logic**, adjust its settings, then click **Apply**. This describes the original close
prices in the selected ticker/provider/revision/date range; it does not forecast
returns. The chart's zoom and pan only change the view. They never refit a trend
or analyze display-grouped candles. Every eligible endpoint in the selected
history is analyzed using only observations at or before that endpoint. Discovery,
volatility normalization, scoring, and tail diagnosis all use that same prefix.
Results therefore have different end dates, rather than all ending on the last
selected bar. Broad and local trends may disagree at any given endpoint.

The chart shows every accepted interval except those matched by the redundancy
rule, including overlapping long and short trends when they remain distinct.
**Redundancy tolerance (%)** defaults to **5%**. For each pair of same-direction
intervals, the comparison resolution is
`max(minimum bars, percentage / 100 * shorter length)`. **Minimum redundancy
resolution (bars)** defaults to **5** and is saved with the other settings;
set it to **0** to use only the percentage. The minimum applies to both rules.
Length counts source bars including both endpoints. Boundary distances
are compared inclusively against that resolution without rounding. **Boundary
matching rule** offers **and** (the default), which requires
both start and end differences to be within the resolution, or **or**, which requires only one. OR can suppress different trend scales that
share a start or end, even at 0%. With both the percentage and minimum set to
zero, AND only matches identical boundaries and OR matches either exact boundary.

Selection is greedy in chronological order of the ending bar. Within the same
ending bar, ties favor higher clarity, then longer intervals, then the interval
ID. A candidate matching any already retained interval becomes redundant and
references the earliest-ending match, using the same tie-breakers. With unchanged
settings and earlier records, appending later endpoints cannot replace earlier
selections, even when a later trend has higher clarity. Trends become available
after their ending bar closes; chart timestamps label bar starts.
Only retained representatives participate, so matches never chain through a
redundant interval. This display choice does not refit, merge, or rewrite any
analysis. The table retains every accepted
interval with **On chart** or **Redundant** status and a reference to its
representative. Rejected candidates are **Uncertain** in the diagnostics table,
with the reason they failed acceptance.
Enable **Show redundant trends** in **Chart display and filters** to include
redundant intervals immediately, without rerunning analysis. It is off by default. The
table keeps their **Redundant** classification and representative reference,
while **On chart** reflects their current visibility. Uncertain candidates are
not plotted by this option.
**Show all λ values** is enabled by default. Turn it off and choose one or more
**Selected λ values** from the searchable suggestions to show only intervals
discovered at any selected penalty immediately. An empty selection hides λ-derived lines; enable
**Include trends without λ** to include tail-derived intervals. Filtering applies
after redundancy selection and keeps line colors. All results remain
in the table, with **On chart** reflecting visibility and **Filtered out** marking
hidden non-redundant intervals.
**Slope filter** defaults to **all**. Choose **positive** (`slope > 0`),
**negative** (`slope < 0`), or **range** with inclusive minimum and
maximum bounds. Bounds use signed fitted **log-price change per source bar**, shown
in the table and hover; this differs from the unsigned **Normalized slope** score.
For example, `0.001` corresponds to approximately `+0.10005%` per bar. Custom bounds
are retained when switching modes; zero matches only All slopes or a range containing
zero. A trend must pass both the λ and slope filters and the redundancy visibility
setting. Display changes take effect immediately and preserve chart navigation.
Invalid ranges show an error and retain the last valid display; they do not
change calculation settings or discard the complete results.

Intervals carry all numerical **lambda (λ)** penalties that discovered their
boundaries, together with their penalty-grid levels. A single boundary can be
discovered at several penalties; it is still one interval. Each line's color is
determined by its smallest λ, and hover shows all its λ values alongside the
interval, direction, clarity, rate, robustness, and tail diagnosis. Colors stay
consistent when zooming or panning. A candidate discovered only by tail diagnosis
is labeled as such and uses a separate neutral style, because it has no
segmentation λ. Every endpoint uses the same fixed numerical λ grid, built once
from configuration before historical analysis. **Starting λ** (stored as
`penalty_multiplier` for compatibility) is the first value, **Penalty grid step**
sets the multiplier between values to `2 ** step`, and **Number of λ levels**
sets the grid size (1–256). Defaults are starting λ **1**, step **1**, and **20**
levels: 1, 2, 4, ..., 524288. For example, starting λ 2.1, step 3, and 4 levels
produce 2.1, 16.8, 134.4, 1075.2 at every endpoint. No price data, endpoint bar count, or future observations
determine the grid, and it is not shortened when multiple levels choose the
same interval. The configured largest λ does not guarantee a no-split solution;
increase the level count if you want to test larger penalties. Volatility
normalization still uses each endpoint's own available data.
Adding observations can change which archived intervals represent nearby
results on the chart, but cannot change an earlier interval's metrics or λ
provenance when its inputs and configuration are unchanged.

Trend settings now belong to each selected logic instance and are saved in the
Market data workspace, including unapplied drafts. The old
`.local/trend_config.json` remains a compatibility input during the first
workspace migration; applying chart logic does not overwrite it. Existing trend
selections, draft settings, λ/slope filters, and compatible analysis snapshots
migrate automatically. An incompatible snapshot requires Apply. Adding multiple
trend instances allows different configurations on the same chart.

**Window advance and overlap (bars)** (stored as `max_history`) sets H, with a
default of 2,000 and a maximum of 100,000. Analysis windows hold up to **2H**
observations and advance by **H** after reaching their end. For H = 2,000, the
windows cover bars 1–4,000, 2,001–6,000, 4,001–8,000, and so on. At each shift,
segmentation is rebuilt for the overlapping H bars, then extended incrementally
for each new endpoint. Previously published endpoints are not recalculated or
replaced in the archive. The initial window grows from the minimum segment
length; after a shift, context grows from H+1 to 2H bars, including the current
bar. H is no longer a strict per-endpoint lookback limit. Segmentation,
volatility, scoring, and tail diagnosis use the same active window prefix,
never observations after the endpoint. Dropping context at a window shift can
change the discovered trends. Windows are anchored to the first selected input
bar, so changing the selected history's start can also change later results.

At least eight observations are required for a segment. Every eligible endpoint
in the selected history is analyzed; progress counts new endpoints, excluding
overlap rebuilding. Larger settings increase computation and memory use, and
window shifts can pause progress while their context is rebuilt. Version 5
results use these window semantics; saved version 4 trend settings are staged
for the new version and require Apply to recalculate their results.
Bars with potentially unfinished periods are
excluded: intraday bars wait their nominal duration; calendar bars wait until
their labelled period ends in America/New_York. This conservative policy can
delay inclusion of shortened session bars or an already closed daily session.
The current stored values are not guaranteed to be point-in-time vendor data;
later vendor revisions or adjustments can change a fresh historical analysis.

The independent package accepts an ordered single-series Polars dataframe with
timezone-aware timestamps and positive finite prices:

```python
from research.analysis.trend import TrendConfig, analyze, analyze_current

result = analyze(
    frame,
    timestamp_col="timestamp",
    price_col="close",
    config=TrendConfig(clarity_threshold=0.70, penalty_multiplier=1.2),
)
print(result.to_frame())

# Optional: only structures ending at the latest selected observation.
current = analyze_current(frame)
```

The package has no dashboard, fetching, storage, or market-calendar dependency.
Callers supply completed observations. An optional `as_of` truncates input at a
timestamp before any estimation. Input rows must have unique increasing
timestamps; the package does not fill gaps or alter the dataframe. Source-row
indices remain relative to the supplied frame, including analysis-window
offsets. `analyze` / `analyze_history` return the historical archive;
`analyze_current` returns one endpoint's snapshot using the same window boundaries
as historical analysis of that input prefix. `replay` yields independent
current snapshots for requested review dates. Historical interval IDs include
their start and end indices; they are stable within the same source frame.

The estimator proposes variable-length intervals using penalized piecewise
log-price regression at multiple resolutions. It combines ordinary regression,
Theil–Sen slope, chronological ordering, and directional efficiency with default
weights 0.25/0.20/0.35/0.20. Separate magnitude, direction-agreement, and boundary
robustness gates control acceptance. Tail diagnostics distinguish recent slope
changes, flattening, level changes, countertrends, and young abrupt counter-moves.
All candidates that pass acceptance remain in the analysis archive; redundancy
is classified only for display. The older `dedup_ratio` setting is accepted for
saved-settings compatibility but no longer removes intervals. Fitted curves use a robust
log-price slope and median residual intercept, and end at their own observed
endpoint. Future prices are never used to find an earlier interval's boundaries.

Clarity is an engineering score, not a probability or a significance test.
Unclear does not mean sideways. No detected tail change does not confirm future
continuation. Estimated starts describe the structure visible at the analysis
endpoint; they are not historical detection dates. Thresholds are starting
settings for visual development, not validated market constants. Use historical
prefixes, inspect charts before revealing overlays, tune one parameter family
at a time, and reserve later untouched periods for frozen-configuration review.
No labeled training dataset is required.

Open the local URL printed by Streamlit. The sidebar selects **Market data**,
**Analysis**, **Fetch**, **Process data**, **Backtest**, **Saved runs**, or
**Delete data**, and the data/results roots. Dataset
charts have candlesticks, volume, drag/scroll zoom, an overview navigator and reset
controls. Charts open on the latest 50 bars (or all bars for smaller datasets);
use **All history**, **Latest 50**, or the overview's period sliders to navigate.
Candle hover includes OHLC and volume. The horizontal axis advances by actual
stored bars, compressing nights, weekends and other empty periods. Zooming and
panning fit the price and volume scales to visible bars; candle widths follow
the zoom level.
Real timestamps remain on the labels and in hover text. Recorded omissions
appear at compressed boundaries with their timestamps in hover text.
The chart uses the installed Plotly bundle without a CDN. Close views show original
bars. Wide views use explicitly labelled **display-only summaries**: first open,
maximum high, minimum low, last close, and summed volume. Hover identifies the
source period and bar count. Intraday groups respect America/New_York trading
dates and missing intervals; wider daily/calendar-week summaries retain stored
extended-hours data. No bars are filled, and stored data, exports and backtests
are unchanged. Intraday display timezones are selectable; daily timestamps remain
session-date labels. Recorded omissions and unknown historical provenance remain
visible, with full records available through **Show omission details**.

The browser receives at most 2,000 candles including nearby buffers, plus a small
full-history overview. Moving to earlier data replaces the old window and evicts
its distant tail. Navigation is debounced, stale replies are ignored, and the
previous chart stays visible while a new window loads. The server still reads
and verifies the selected stored data; it retains prepared snapshots in a per-session
LRU cache capped at four entries and 64 MiB. Deletion, replacement and file-integrity
checks are never bypassed by this cache. **Show original bars** loads a paginated
table (500 rows per page). CSV creation is deferred until download and reads the
verified original data again.

The same workflow is available from an interactive Python session:

```python
from datetime import UTC, datetime
from research import Research
from research.strategies import MovingAverageCross, Momentum, WeightedStrategy
from research.backtesting import ExecutionSettings

research = Research("data", "runs")
start = datetime(2024, 3, 1, tzinfo=UTC)
end = datetime(2025, 1, 1, tzinfo=UTC)
strategies = [MovingAverageCross(10, 30), Momentum(20)]
strategies.append(WeightedStrategy(strategies, [0.5, 0.5]))
report = research.preflight(["AAPL", "MSFT"], strategies=strategies,
                            start=start, end=end, timeframe="1d")
print(report.to_frame())  # Empty when no blocking issues were found.
run = research.backtest(["AAPL", "MSFT"], strategies=strategies,
                        start=start, end=end, timeframe="1d",
                        settings=ExecutionSettings(commission_bps=5, slippage_bps=5))
print(run.comparison)
restored = research.load_run(run.run_id)
```

### Fetch, store and backtest the same interval

The Yahoo workflow supports `1m`, `2m`, `5m`, `15m`, `1h`, `90m` and `1d`.
Use the same interval when fetching and backtesting: AAPL `1m`, `15m`, `1h`
and `1d` are separate local series. `60m` is normalized to `1h` everywhere,
including CLI arguments; it cannot bypass overlap protection.

The dashboard uses provider dropdowns, with no free-text provider names.
**Fetch** lists registered providers and offers their declared intervals supported
by the dashboard. **Market data** and **Backtest** list providers present in the
local catalog, so stored data remains usable even if its adapter is no longer
registered. Each view or backtest uses one provider; histories are never combined
across providers. Yahoo’s missing-OHLC option appears only for Yahoo. New adapters
should declare `supported_timeframes` and be registered before launching the dashboard.

In the dashboard, select **Fetch**, choose a provider, bar size and UTC bounds,
and choose **Existing data**. **Fail on overlap** preserves the default behavior.
**Skip covered requests; refresh overlapping ranges** skips fully covered requests
without a vendor call and refreshes the combined range for partial overlaps.
**Fetch missing ranges; merge locally** requests only uncovered portions and
combines them with existing bars without changing the existing values. Planned
vendor ranges appear before each download. The report shows exact ranges and
statuses, newly downloaded rows, total saved rows, returned coverage, and which
datasets were replaced. Old revisions remain available by ID; any processed
derivatives of replaced inputs need rebuilding. Then select **Backtest**, choose
the same bar size, tickers and
strategies, and set the test bounds. Intraday forms include UTC time inputs;
the end is exclusive. The backtest page shows which tickers have local data
at that interval, and preflight checks their actual coverage before execution.
The CLI follows the same flow:

```bash
# Set these to recent UTC timestamps covering complete sessions for your test.
quant-research fetch --tickers AAPL MSFT --timeframe 15m \
  --start "$FETCH_START" --end "$END"
quant-research backtest --tickers AAPL MSFT --timeframe 15m \
  --start "$TEST_START" --end "$END" --strategies strategies.example.json
```

Choose `FETCH_START` earlier than `TEST_START` to include strategy warm-up.
Lookbacks count bars, not days: 20 preceding observations span different
amounts of time at `1m` and `1d`. Calendar generation walks backward through
trading sessions until enough warm-up bars are available. Local data must
contain those bars as well. Yahoo's retention and availability restrictions
still apply; provider failures are reported per ticker.

Intraday bars start at the session open and advance by the selected duration.
The final bar is clipped to the actual session close, including early closes.
For a 09:30–13:00 session, hourly labels are 09:30, 10:30, 11:30 and 12:30;
the last bar closes at 13:00. Off-grid provider timestamps cause preflight to
fail. No interval is silently substituted, resampled, filled, or relabeled.
An interval end or `as_of` cutoff inside an expected bar blocks the test.

`30m` raw data can be backtested through the same API, but our Yahoo adapter
intentionally rejects fetching it because yfinance resamples that interval.
Supply genuine raw `30m` bars from a compatible provider instead. Weekly and
monthly backtesting, lunch-break calendars, automatic session resets and
mandatory session-end liquidation are outside this implementation. Positions
may continue into the next session under the existing execution rules.

### Data checks and simulation rules

Before any strategy is evaluated, every ticker must have every required bar,
including warm-up, on the selected exchange schedule. Preflight checks actual
timestamps, holidays, DST, early closes, session alignment, completed bars,
file checksums and canonical values. Missing, unexpected, corrupt or unfinished
bars abort the entire request with ticker-specific details. Bounds in the
catalog alone do not establish completeness. Backtests never fetch or fabricate
missing data. Supported inputs are raw `1m`, `2m`, `5m`, `15m`, `30m`, `1h`,
`90m` and `1d` bars for USD instruments, using the explicitly selected calendar (default `XNYS`, U.S.
regular equity sessions). The calendar is a user-selected assumption, not
automatic exchange discovery; choose the appropriate calendar for the symbol.
Intraday calendars with lunch breaks are rejected.

A skipped bar blocks a test when its absence affects the test or warm-up window.
Omissions outside that window do not block it. Legacy datasets have `unknown`
fetch provenance, which remains visible; they can pass only after their actual
bars pass coverage checks. A quality report records what ingestion observed,
and cannot prove the provider supplied every real trade or correct prices.
Replacements receive fresh reports; processing/compaction preserve lineage and
combined reports. Old recorded omissions do not override a bar now present.

Allocation strategies give each strategy/ticker pair its own cash account.
Stop/target strategies support the capital allocation modes described below.
Allocation strategies remain long-only; stop/target strategies can opt into
short entries as described below. The dashboard enables fractional
shares by default; API execution settings retain whole shares unless requested.
At each completed bar, the strategy returns a target allocation in `[0, 1]`.
That target is rebalanced at the next available scheduled bar's open using
that open's price plus configured adverse slippage. Fees and price gaps are
included in affordability calculations; buys are reduced or rejected when
needed. Rejections are recorded. Zero-volume bars cannot execute orders.
The exact cash ledger uses Decimal arithmetic, fees round upward to eight
decimal places. Long-only cash/holdings/equity stay nonnegative; short losses can
exceed reserved collateral. Float columns
are for plotting; `*_exact` columns preserve ledger values.

Strategies receive isolated history containing only completed preceding bars.
Warm-up observations can produce the first test trade, but never contain test
fills. Each strategy/ticker has fresh state. Future-row perturbation tests,
manual fee/slippage examples and ledger invariants verify these rules. Custom
Python code can still access outside information; this API is not a sandbox
against a strategy that deliberately reads future data elsewhere. Arbitrary
preprocessed histories are excluded because valid timestamps alone cannot
establish that their transformations were causal.

This first engine reports **price returns**, with open positions valued at the
last included close. It does not credit dividends, reconstruct corporate
actions, model settlement, exchange order queues, market impact, or constrain
fills to a fraction of positive reported volume. It assumes positive-volume
bars can execute the requested affordable quantity at the slipped open.
Yahoo's vendor-provided historical split treatment remains in its raw prices;
these results are not a reconstruction of a broker account across corporate
actions. There is no live order execution. These assumptions are also saved
in result metadata.

### Shared portfolios with stop and target exits

Strategies implementing `research.strategies.BracketStrategy` run through the
same preflight, Backtest page, and saved-run pipeline. **Capital allocation**
selects one of three modes:

- **Distribute at each entry:** one shared cash account. Each new entry receives
  available cash divided by all eligible tickers not currently held, including
  those without a signal. Simultaneous entries use one allocation snapshot.
- **Fixed percentage of initial capital per entry:** set **Initial capital per entry (%)**
  to 5, 7, 10 (the default), or another percentage. Each signal gets that percentage
  of the original capital, including fees and slippage. Skip the signal if
  available cash cannot cover the full budget. Later signals may use replenished
  cash released by exits; skipped signals are not queued. Simultaneous signals are handled
  in ascending ticker order. This mode requires stop/target strategies and runs
  sequentially (`allocation_mode="fixed_tenth", entry_percentage=7` in Python;
  `--entry-percentage 7` in the CLI).
- **Split once at start:** divide total initial capital equally among the usable
  tickers. Each ticker retains its own cash and profits throughout the run.
  Independent ticker simulations can run in parallel using **Parallel workers**.
  A late-starting ticker's allocation remains idle until its warm-up completes.
  The report includes a combined summary per strategy and individual ticker details.

All modes include entry costs in their budgets and leave existing
holdings unchanged. **Allow fractional shares** is enabled by default in the
dashboard: entries need a budget of at least $5, with quantities rounded down
to eight decimal places. This is a minimum entry budget, not a $5 cash reserve.
Complete exits can be smaller than $5. Exact quantity and cash strings are saved
alongside display values. Python callers enable this with
`ExecutionSettings(fractional_shares=True)`.

**Allow missing history at start** lets a ticker join once it has enough real
warm-up bars. Only a missing leading prefix is tolerated: gaps after the first
available bar, including trailing gaps, still fail preflight. Tickers with no
usable execution bars are excluded with a recorded reason. In the shared mode,
a ticker joins the allocation pool at its effective start. Both settings are
recorded with saved runs. The Python options are
`allocation_mode="initial_split"`, `max_workers=4`, and
`skip_initial_missing_bars=True`; defaults retain shared per-entry execution and
strict history checks. Parallel workers require importable strategy classes;
local private registrations are loaded in each spawned worker.

**Skip problematic tickers** excludes tickers that fail local-data validation
(such as missing bars or corrupt files) and continues with the valid tickers.
Excluded tickers receive no capital; the summary and saved run include their
error codes and reasons. If no tickers remain, the run stops. This option defaults
to off and does not suppress strategy execution errors. Python callers use
`skip_problematic_tickers=True`.

For `1h` data, **Trading period → Hourly bar starts 09:00–15:00 (New York)**
accepts clock-aligned bars from 09:00 through 15:00, ending at 16:00, without shifting their
timestamps or changing their prices. Bars outside 09:00 inclusive to 16:00 exclusive
are excluded. Python callers use
`trading_period="09:00-15:00"`. This mode uses XNYS trading dates and New York
daylight-saving time. On early-close days it requires only bars starting before
the official close, excludes later bars from execution, and records warnings in
the report and saved run. Missing bars before the close and missing ordinary
sessions still fail validation. The default remains the regular-session grid.

The Backtest page automatically saves its configuration locally per data directory,
including strategy selections and parameters, JSON drafts, combination weights,
capital, fees, and validation options. These settings survive page navigation and
application restarts without needing to run a backtest.

Bracket strategies may provide `prepare_entry_signals(history)` to compute a
causal boolean series once per ticker. Row i must depend only on rows through i;
future rows must never change earlier decisions. The engine enters using row i-1
at row i's open. Strategies without this optional method keep the existing
completed-history `entry_signal` interface.

The private SMA/RSI strategy has an **Enable short selling** option, off by default
(`enable_shorts=True` in its configuration). A short signal requires the current
SMA to be below either of the preceding two SMA values and RSI to cross strictly
from above 50 to below 50. The existing long signal is unchanged. The short opens
at the next bar's open, with target `entry_open / win_percentage` and stop
`entry_open / stop_percentage`. A low strictly below the target wins; a high
strictly above the stop loses. Crossing both thresholds is an inconclusive stop.
Only one position per ticker can be open at a time, in either direction.

Short entries use the selected allocation mode and reserve the raw entry
notional plus entry fees from available cash. Sale proceeds stay locked until
the position is bought back, so they cannot fund additional entries. Sell fills
receive downward slippage and buybacks upward slippage. Equity includes reserved
collateral, locked proceeds, and the current buyback liability. A forced buyback
honors losses even when they exceed collateral; negative cash blocks new entries
until funds recover. Borrow availability, borrow fees, and maintenance-margin
calls are not modeled. Short-capable bracket strategies implement
`short_entry_signal(history)` and may also provide
`prepare_short_entry_signals(history)` under the same causality rules.

Entry decisions use completed bars and execute at the following open. Fixed
stop and target multipliers use that entry bar's raw open. Exit checks include
the entry bar and require strict high/low crossings; touching a threshold does
not exit. Crossing both thresholds records an inconclusive stop and closes at
the modeled stop price. Exits use threshold prices with configured adverse
slippage, including across gaps; this is a simplified fill assumption. Their
exact intrabar time is unknown, so proceeds become available at bar close.
Zero-volume bars cannot execute orders.

The report includes wins, losses, inconclusive stops, completed-trade details,
open positions, and realized/unrealized/total P&L after costs. Unresolved positions
are valued at their final close. Details record bar positions within the loaded
history (including warm-up), timestamps, allocations, quantities, thresholds,
prices, costs, and long/short direction. These tables survive saving and reopening. Stop/target
strategies cannot be included in weighted allocation combinations.

### Strategies, results and reproducibility

Built-ins are moving-average crossover, positive momentum, and weighted
combinations of strategy target allocations. Combining strategies produces
one target and one account; it does not sum independent equity curves.
Implement `research.strategies.Strategy` with `name`, `version`, JSON `config`,
positive `lookback`, and `target_weight(history)`. Pass instances directly or
register factories in `StrategyRegistry` and provide it to `Research`.
Version custom code whenever its behavior changes; external custom strategy
source is not automatically archived or fingerprinted.

### Private strategies

Personal implementations live in the root-level `private_strategies/` package.
The entire directory is Git-ignored and is outside the installable `src/`
packages. It is local to your checkout: a fresh clone will not contain your
private files. Back them up separately. Ignore rules do not remove files that
were previously tracked or protect files from local access.

```text
private_strategies/
├── __init__.py
├── register.py       # One explicit registration hook
└── example.py        # Local starter example; replace with your own code
```

The local starter registers **Private example (momentum)** in the dashboard.
Its implementation subclasses the existing momentum strategy to demonstrate
loading; it is not a new trading model. To add your own strategy, implement
the normal `Strategy` interface and register it in `register.py`, for example:

```python
from .my_strategy import MyStrategy

def register_strategies(registry):
    registry.register(
        "my_strategy", "1", lambda config: MyStrategy(**config),
        label="My Strategy",
        description="My personal strategy.",
        default_config={"window": 20},
        parameters={
            "window": {"type": "integer", "minimum": 1, "label": "History bars"},
        },
    )
```

The instance's `name` and `version` must match the registration. Use relative
imports between private files, such as `from .my_strategy import MyStrategy`.
Only implementations registered by this hook are offered; unrelated Python
files are not automatically imported. The loader imports the trusted local
package under an internal namespace without modifying Python's search path.

Restart the dashboard after editing private Python files. Modules are imported
once per process; fresh registries prevent duplicate registration on page reruns.
Missing `private_strategies/` is allowed and leaves built-ins available. An
existing but broken package, missing registration hook, invalid metadata, or
duplicate name/version raises a visible error instead of silently hiding the
problem. The loader searches relative to the installed source checkout, not
the current working directory or the selected data directory.

On **Backtest**, select your strategy from **Strategies**. Controls are generated
from its parameter descriptions:

| Parameter type | Control |
|---|---|
| `integer`, `number` | Numeric input; optional `minimum`, `maximum`, and `step` |
| `boolean` | Checkbox |
| `string` | Text input |
| Any scalar type with `choices` | Dropdown |
| `json`, missing descriptions, or partially described configuration | Complete JSON configuration editor |

Every described parameter requires a value in `default_config`. Optional `label`
and `help` fields control presentation. Registration validates metadata/defaults;
loading validates declared types, choices and bounds before calling the factory.
The strategy constructor remains responsible for relationships between fields,
such as `fast < slow`. Existing registrations without presentation metadata
remain valid and get a JSON editor. Numeric values too large for exact browser
integer inputs also use JSON. A JSON-editing checkbox is available for structured
controls; the two editing modes retain their own values.

Select multiple strategies and enable **Also test a weighted combination** to
combine any registered strategies, including private ones. The displayed weights
are relative, normalized to sum to one, and must include a positive weight.
Each selected strategy is tested separately as well as in the combined account.

The same discovery is used by `Research` when loading specifications and by
`quant-research backtest --strategies your-specs.json`. Private imports are lazy:
viewing market data or reopening saved results does not require the private code.
To select a different private package location explicitly in Python:

```python
from research import Research
from research.strategies import load_registry

registry = load_registry(project_dir="/path/to/project")
research = Research(strategies=registry)
```

Saved runs include strategy names, versions and parameter values, but do not
copy private source. Keep sensitive values out of shared manifests/screenshots.
Bump a private strategy's version whenever its behavior changes. Private loading
does not change the engine's execution rules or add session-end liquidation.

Each successful request creates an immutable directory under `runs/` with a
manifest and Parquet equity, trade and order tables. The manifest records
strategy configurations, costs, source revision IDs/checksums, quality,
calendar/version, research code fingerprint and return assumptions. The
entire run is published after all simulations succeed; failed simulations
produce no partial visible run. Loading verifies artifact checksums.
`compare_runs([id1, id2])` rejects different data revisions, periods, warm-up
lengths, calendars, package versions, account settings or engine fingerprints. Preserve original dataset revisions
and the code/environment to reproduce a run; checksums detect accidental
changes, not adversarial rewriting of both a file and its checksum.

```text
src/
├── data_pipeline/         # Providers, batch acquisition, quality and storage
│   └── batch_fetch.py    # Per-ticker fetch logic and reports
├── research/
│   ├── api.py            # Research application interface
│   ├── datasets.py       # All-ticker checks and pinned revision snapshots
│   ├── instruments.py    # Explicit exchange calendars and bar intervals
│   ├── timeframes.py     # Shared interval durations and workflow choices
│   ├── strategies/      # Versioned strategies and weighted combinations
│   ├── backtesting/     # Causal execution, cash accounting and metrics
│   ├── analysis/        # Script toolkit, registry, execution and plain results
│   │   ├── core/       # Contracts, registry and per-context execution
│   │   ├── scripting/  # Decorator, discovery, context and result builders
│   │   ├── batch/      # Multi-ticker execution, aggregation and saved runs
│   │   ├── indicators/ # Shared numerical indicators
│   │   ├── trend/      # Historical trend analysis and selection
│   │   └── builtins/   # Discoverable indicator/trend/open-price analyses
│   ├── runs.py           # Atomic saved runs and compatible comparisons
│   └── cli.py            # quant-research command
└── dashboard/
    ├── app.py            # Streamlit entry and routing
    ├── pages/            # Market data, analysis, fetch, processing and backtests
    ├── market_data/      # Analysis controls, display, results and navigation
    ├── batch_analysis/   # Batch job state and text result presentation
    ├── charts/           # Series, bounded cache, rendering and browser assets
    ├── components/       # Shared schema and strategy controls
    └── state/            # Workspace persistence and migrations
```

## Development setup

The committed `requirements.txt` file contains runtime dependencies. Install
test-only tooling separately:

```bash
./venv/bin/pip install -r requirements.txt -r requirements-dev.txt
./venv/bin/pip install -e .
```

The editable install makes `data_pipeline` importable from the virtual
environment while continuing to use the source files in this repository.

Run the normal, offline test suite with:

```bash
./venv/bin/python -m pytest
```

To also test chart gestures, scaling and resize in an installed Chrome/Chromium
browser (synthetic data only; no vendor requests):

```bash
./venv/bin/python -m pytest tests/test_dashboard_chart_browser.py \
  tests/test_dashboard_logic_overlay_browser.py tests/test_dashboard_indicator_panels.py \
  tests/test_dashboard_chart_streamlit.py --run-browser
```

Tests use per-test temporary directories and deterministic fixtures under
`tests/fixtures/`; they must not read from or write to the repository's
`data/` directory. Private-strategy discovery is redirected to temporary test
locations so tests never execute your personal implementations. Live vendor checks are marked `network` and skipped unless
explicitly enabled:

```bash
./venv/bin/python -m pytest --run-network -m network tests/test_yahoo_live.py
# Requires MASSIVE_API_KEY and access to the requested historical data:
./venv/bin/python -m pytest --run-network -m network tests/test_massive_live.py
```

## Current data contract

The first supported dataset is canonical OHLCV. A `DataRequest` has
timezone-aware UTC bounds, with an inclusive `start` and exclusive `end`.
Intraday timestamps identify the start of a bar and must be UTC. Daily
and longer bars use exchange session dates represented at UTC midnight; these
are date labels, **not actual exchange opening times**. Validated frames
use ordered `timestamp`, `symbol`, `open`, `high`, `low`, `close`, and
`volume` columns. Each frame represents one timeframe, and its rows are
ordered by symbol and timestamp.

## Fetch from Yahoo

From the repository root (date-only CLI bounds mean UTC midnight):

```bash
./venv/bin/python -m data_pipeline --data-dir data ingest \
  --symbol AAPL --provider yahoo --timeframe 1d \
  --start 2024-01-02 --end 2024-01-10

./venv/bin/python -m data_pipeline --data-dir data list
./venv/bin/python -m data_pipeline --data-dir data read \
  --provider yahoo --symbol AAPL --timeframe 1d
```

Ingestion prints the raw dataset ID and row count. Repeating the write raises
an overlap error. The exclusive end means January 10 is not included above.
After reinstalling editable mode, `./venv/bin/data-pipeline` is also available.

```python
from datetime import UTC, datetime
from data_pipeline import DataPipeline, DataQuery, DataRequest

pipeline = DataPipeline("data")
request = DataRequest(
    symbol="MSFT", start=datetime(2024, 1, 2, tzinfo=UTC),
    end=datetime(2024, 1, 10, tzinfo=UTC), timeframe="1d", provider="yahoo",
)
result = pipeline.ingest(request)
bars = pipeline.read_dataset(result.raw.dataset_id)
filtered = pipeline.read(DataQuery(provider="yahoo", symbol="MSFT", timeframe="1d"))
lazy = pipeline.scan(DataQuery(provider="yahoo", symbol="MSFT", timeframe="1d"),
                     columns=["timestamp", "close"])
```

Yahoo requires internet access and may return errors for unavailable symbols,
retention-limited intraday history, rate limits, or outages. Download response
timeout defaults to 10 seconds; metadata/cookie calls use yfinance's timeouts.
Supported intervals: `1m`, `2m`, `5m`, `15m`, `60m` (canonical `1h`), `90m`,
`1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`. The adapter rejects `30m` because
yfinance internally resamples that interval. It does not retry smaller windows
or switch vendors silently. Use recent dates when requesting intraday data.

Price policy is explicitly `unadjusted`: yfinance auto-adjust, back-adjust and
repair are disabled. This preserves Yahoo's supplied OHLC, but does not undo
historical split treatment already present in Yahoo data. `Adj Close` is not
stored. Raw means canonicalized vendor OHLCV, not the original HTTP response.
Source volume is checked on Yahoo's chart response before yfinance processes
it: missing volume is rejected with UTC timestamps instead of being silently
filled with zero. Genuine zero volume remains valid. With
`YFinanceProvider(skip_missing_ohlc=True)`, wholly missing OHLC bars can still
be skipped with a warning, but missing volume alone is never skipped or filled.
This guard uses per-request history hooks in the pinned yfinance version;
raw-response regression tests must pass when upgrading that dependency.
Yahoo symbols are canonicalized to uppercase, so `aapl` and `AAPL` share one
storage identity and cannot bypass overlap protection. Other providers retain
case-sensitive instrument identifiers. Local symbol queries apply Yahoo's
case normalization even when the provider filter is omitted.

## Fetch from Massive

The default registry includes `massive`; select it in the dashboard's **Fetch**
provider menu or pass `provider="massive"` to the pipeline. Credentials are
checked only when fetching, so local browsing and provider discovery work
without an API key. Set the key in the environment before launching Python or
Streamlit; keep real credentials out of source files and version control:

```bash
export MASSIVE_API_KEY="your-api-key"
```

```python
from datetime import UTC, datetime
from data_pipeline import DataPipeline, DataQuery

pipeline = DataPipeline("data")
report = pipeline.fetch_many(
    ["AAPL", "MSFT"], provider="massive", timeframe="15m",
    start=datetime(2024, 1, 2, tzinfo=UTC),
    end=datetime(2024, 1, 6, tzinfo=UTC),
    massive_request_interval_seconds=13,
)
print(report.to_frame())
bars = pipeline.read(DataQuery(provider="massive", symbol="AAPL", timeframe="15m"))
```

Choose dates covered by your subscription. Use exact, case-sensitive vendor
symbols, including the historical ticker when requesting covered delisted
stocks. This adapter does not build historical stock universes or add delisting
accounting to the simulator. Do not pass Yahoo's `skip_missing_ohlc` option.

The dashboard offers **Minimum seconds between API requests** for Massive only.
The free plan permits 5 requests per minute; about 13 seconds helps stay below
that limit when no other downloads share the key. The interval applies to every
HTTP request, including chunks, pagination, retries and later tickers in the
same batch. The first request starts immediately. An explicit
`massive_request_interval_seconds` applies only to that batch and preserves the
registered provider's other settings. Omit it to retain the registered interval;
the default is `0`, which disables deliberate pacing but still honors retry
waits. Separate batches and processes are paced independently.

`MassiveProvider(api_key=None, timeout=10, max_retries=2, request_interval_seconds=0)` also accepts an
explicit key for a custom registry. Its native intervals are `1m`, `2m`, `5m`,
`15m`, `30m`, `1h`, `90m` and `1d`; `60m` requests normalize to `1h`.
Requests explicitly use unadjusted prices, download every page, and split long
intraday ranges into 30-calendar-day chunks below the 50,000 base-bar limit.
Connection failures, HTTP 429 and selected temporary server errors receive
bounded retries. A 429 without a usable `Retry-After` waits 60 seconds before
retrying; a server-requested delay above 60 seconds fails the current fetch.
HTTP 401 indicates invalid
credentials; HTTP 403 can indicate subscription/history restrictions. A failed
download saves nothing for that ticker, while later tickers are still attempted.

Vendor OHLCV values, including fractional volume, are retained without filling
gaps or resampling. Missing or invalid numeric values fail validation. Intraday
timestamps and extended-hours bars are preserved; native hourly and 90-minute
grids are not moved to the regular-session open. Daily timestamps are converted
from their Eastern session date to the canonical UTC-midnight date label;
prices are unchanged. Missing-bar provenance remains `unknown`, since the
aggregate response does not explain every absent bar.

**Storage and charting support all these intervals; backtesting requires a
compatible schedule.** Preflight rejects extended-hours or misaligned bars in
its required window. Native hourly bars therefore do not match our 09:30-anchored
session grid, and a calendar-day download is not automatically ready for a
regular-session backtest. Massive daily backtests use completed prior-day bars
for next-session-open execution, with the same coverage and warm-up checks as
other providers. This assumes the prior-day historical values are available
before the next open; historical bars may include later vendor corrections.
The recorded decision timestamp retains the engine's regular-session-close
convention. No session filtering, alternative bar grid, or simulation changes
are implied by adding the provider. See [Massive's aggregate API](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
for vendor coverage and aggregate semantics.

## Structure

```text
src/data_pipeline/
├── api.py                 # DataPipeline orchestrates the complete workflow
├── batch_fetch.py         # Independent ticker fetches and structured outcomes
├── cli.py, __main__.py     # Command-line entry points
├── models.py              # DataRequest and DataQuery
├── exceptions.py          # Specific validation/provider/storage errors
├── schemas/ohlcv.py        # Canonical frame contract and validation
├── providers/             # Common interface, registry, Yahoo and Massive adapters
├── processing/            # Shared execution, contracts, validation and registry
│   └── processors/        # Shared base.py, plus scaling.py, resampling.py and time_window.py
└── storage/               # Immutable files, DuckDB catalog, locking, metadata

data/                      # Git-ignored local storage
├── raw/                   # Canonical vendor bars
├── processed/             # Versioned processor outputs
├── metadata/              # catalog.duckdb and store.lock
├── staging/               # In-progress writes
└── quarantine/            # Recoverable abandoned writes

tests/                     # Offline unit/integration/CLI tests and opt-in live test
```

Files are grouped by dataset/provider/symbol/timeframe and the first bar's year;
catalog bounds handle batches that cross years. Each batch is one immutable
Zstandard-compressed Parquet file with row-group statistics. Other packages
should use the API rather than construct paths. The original `base_fetcher.py`
and `yahoo_fetcher.py` remain compatibility imports.

## Processing fetched or stored raw data

```python
from data_pipeline.processing import Pipeline, ScalePrices

processors = Pipeline([ScalePrices(factor=2)])
# For a new request: pipeline.ingest(request, processors=processors)
processed = pipeline.process(result.raw.dataset_id, processors=processors)
print(processed.parent_ids, processed.pipeline_id)
```

`ScalePrices` demonstrates the interface; it multiplies OHLC prices and does
not implement corporate-action adjustment. Processors own their transformation
validation; the pipeline no longer imposes a subset-key rule on every step.
All outputs must still satisfy their declared contract, including schema,
duplicate-key and timestamp-label checks. Empty output remains an error.
Version changes are the processor author's responsibility. Name, version,
config and order determine a SHA-256 pipeline fingerprint; code is not hashed.

An empty processor list stores only raw data. Processing existing raw data
with an empty list returns the single raw input's metadata without creating a
duplicate. Multiple inputs with an empty processor list are rejected.
Raw data commits before processing; a processor failure reports the retained
raw ID so processing can be retried. Repeating the same set of raw revisions
and processor fingerprint raises an error, regardless of input-ID order.
Changed versions/config create
separate outputs. Queries must select one processor fingerprint when multiple
processed variants exist.

The CLI loads an ordered JSON list from a file:

```json
[{"name":"scale_prices","version":"1","config":{"factor":2}}]
```

```bash
./venv/bin/python -m data_pipeline process RAW_ID --processors processors.json
```

Python callers can pass processor instances directly or register custom
factories with `ProcessorRegistry`. The default loader, CLI, dashboard, and
processed-data backtests automatically discover processors; the bundled
processors are `scale_prices`, `resample_ohlcv`, and `filter_time_window`, all
version `1`.

### Implementing a processor

Processors live in `processing/processors/`: `base.py` contains `BaseProcessor`,
`scaling.py` contains `ScalePrices`, and `resampling.py` contains `ResampleOHLCV`
and its session helpers. Those classes are also exposed by
`data_pipeline.processing` and `data_pipeline.processing.processors`.
`time_window.py` contains `TimeWindowFilter`, imported directly from its module
and discovered automatically without package exports or dashboard changes.
Shared execution and metadata modules such as `pipeline.py`, `contracts.py`
and `registry.py` remain directly under `processing/`.

A concrete `BaseProcessor` subclass defined in any Python module under
`src/data_pipeline/processing/processors/` is automatically registered. Give it
unique `name` and `version` identifiers. No registry list, package export, or
dashboard branch needs updating. New modules appear on a dashboard rerun;
restart after editing modules already imported into the Python process. Broken
imports or duplicate identities are reported without hiding unrelated processors.
Only install trusted local Python processors: discovery imports their modules.

Configuration is inferred from typed dataclass fields or constructor parameters.
Defaults, `Literal`/enum choices, optional values, dates/times, nested dataclasses,
and lists/tuples are supported. Put display titles, descriptions, and constraints
in `field(metadata={"schema": {...}})` or a class-level `config_schema` override.
The dashboard renders numbers, text, choices, booleans, and nested objects;
arrays, maps, and complex unions have editable JSON controls. The same schema
validates dashboard and programmatic configuration before construction. A
processor with no settings displays “This processor doesn’t have any configurable
variables.” A custom `from_config(values)` classmethod can construct application
specific types. For example, a configurable processor can declare:

```python
from dataclasses import dataclass, field
from typing import ClassVar
from data_pipeline.processing import BaseProcessor

@dataclass(frozen=True)
class MyProcessor(BaseProcessor):
    name: ClassVar[str] = "my_processor"
    version: ClassVar[str] = "1"
    label: ClassVar[str] = "My processor"
    factor: float = field(default=1.0, metadata={
        "schema": {"title": "Factor", "exclusiveMinimum": 0}
    })
    # Implement the three hooks below to make this concrete and discoverable.
```

Increase the version when transformation semantics change. For backtesting,
explicitly declare `causal = True` only if each output bar uses information
available by that bar's completion. Declare `price_basis = "transformed"` when
changing price units; otherwise the default preserves the input price basis.
Timeframe-changing processors must also implement
`validate_backtest_calendar(bars, *, input_contract, output_contract)` to verify
output availability against the supplied expected bars (timestamp, open, close).
The base implementation accepts unchanged contracts. This validation lives in
the processor, so adding one does not require editing research logic. Processors
without a causality declaration remain usable for processing and charts.

Inherit `BaseProcessor` and implement the three internal hooks. Do not override
the public `transform(frame, input_contract)` wrapper: it resolves the output
contract first, then checks the shared contracts and calls input validation,
computation, and output validation in order. It returns `ProcessingResult`, not
a bare DataFrame. `@final` documents this
convention for type checkers; it is not runtime enforcement or a security sandbox.
Hooks get isolated frames, and validators raise an exception on failure.

```python
from data_pipeline.processing import BaseProcessor
from data_pipeline.processing.validators import SubsetKeys

class KeepPositiveVolume(BaseProcessor):
    name, version, config = "keep_positive_volume", "1", {}

    def _validate_input(self, frame, input_contract):
        pass  # Shared OHLCV validation is already performed by the wrapper.

    def _transform(self, frame, input_contract, output_contract):
        return frame.filter(frame["volume"] > 0)

    def _validate_output(self, original, output, input_contract, output_contract):
        SubsetKeys()(original, output)
        if not output.equals(original.filter(original["volume"] > 0)):
            raise ValueError("Unexpected filtering result")
```

`PreserveKeys` and `PreserveVolume` are also reusable checks. `ScalePrices`
uses those validators and verifies the configured price multiplication.

The default `output_contract(input_contract)` preserves the input contract.
Override it when changing timeframe or timestamp semantics. The **processor's
shared wrapper** resolves it once before computation and passes the contracts
to its hooks as invocation-local arguments, not mutable processor state.
The pipeline only calls `transform(frame, input_contract)`: it neither resolves
output contracts nor calls the internal hooks. It independently checks that the
returned object is a `ProcessingResult` with a valid `DataContract` and that the
output frame satisfies that contract, even for structural processors that bypass
the base class. Such custom implementations remain responsible for declaring
their output honestly and validating the input-to-output relationship.

`DataContract` is frozen and describes timeframe, OHLCV schema version and
timestamp convention. There is no separate context container. Both processors
and `Pipeline.run()` return a `ProcessingResult` with `.frame` and `.contract`:

```python
from data_pipeline.processing import DataContract, Pipeline, ScalePrices

contract = DataContract("1h")
result = ScalePrices(2).transform(frame, contract)
scaled_frame, output_contract = result.frame, result.contract

recipe = Pipeline([ScalePrices(2)])
result = recipe.run(frame, contract)
frame_only = recipe.transform(frame, contract=contract)
```

Every processing call now requires an explicit input contract, including the
frame-returning `Pipeline.transform()` convenience method and an empty pipeline.
An empty pipeline returns validated, canonically typed data with the same
contract; it no longer accepts arbitrary non-OHLCV frames. Bar spacing is never
used to infer timeframe. `DataPipeline.process()` supplies the input contract
from saved raw metadata automatically, so its Python/CLI usage is unchanged.

Migration: direct processor callers must supply the input contract and use
`.frame` on the returned result when they only need a DataFrame. Custom processors
should adopt the hooks above; structural implementations need `name`, `version`,
`config`, and `transform(frame, input_contract) -> ProcessingResult`, but do not
need a public `output_contract()` method. Identity/configuration must remain
unchanged during execution. Settings affecting output belong in config, not
hidden mutable state. Processor-provided checks do not replace independent tests.

### Hourly-to-daily resampling

Supply an explicit schedule of date labels and aware opening/closing instants.
The implementation does **not** assume exchange hours, fetch a calendar, or infer
holidays. Schedules must include every intended session, including sessions
with no observed bars. Below is a synthetic three-hour session, not a claim
about an exchange's actual hours:

```python
from datetime import UTC, date, datetime
from data_pipeline.processing import ResampleOHLCV, TradingSession

resampler = ResampleOHLCV(sessions=[
    TradingSession(
        label=date(2024, 1, 2),
        open=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        close=datetime(2024, 1, 2, 17, 30, tzinfo=UTC),
    ),
])

# Supply your real saved input IDs. They may split a session across files.
daily = pipeline.process([first_raw_id, second_raw_id], processors=[resampler])
bars = pipeline.read_dataset(daily.dataset_id)
assert daily.request.timeframe == "1d"
```

Input IDs must be distinct, active raw batches sharing provider, symbol,
dataset and timeframe. The pipeline loads them together in timestamp order;
storage rechecks every parent is still active when publishing the output.
The CLI accepts the same multi-batch operation:

```bash
./venv/bin/python -m data_pipeline process RAW_ID_1 RAW_ID_2 --processors processors.json
```

Example `processors.json` for the synthetic schedule above:

```json
[{"name":"resample_ohlcv","version":"1","config":{
  "target_timeframe":"1d",
  "incomplete":"raise",
  "sessions":[{"label":"2024-01-02","open":"2024-01-02T14:30:00Z","close":"2024-01-02T17:30:00Z"}]
}}]
```

Resampling currently accepts `1h` bars anchored at session open, one continuous
`[open, close)` window per date label. The last hourly bar may be shorter than
an hour. Correct schedules can express DST offsets, overnight sessions and
early closes. Intraday breaks, differently anchored vendor bars, other target
intervals, and partial daily output are not implemented.

`incomplete="raise"` is the default. `incomplete="drop"` explicitly omits
incomplete symbol/session groups, including wholly missing sessions; if every
group is dropped, processing fails rather than saving an empty dataset.
Off-grid and out-of-session input always raise errors. Completeness means the
expected hourly timestamps are present, not independent verification of every
underlying trade or the vendor's last-bar duration.

Open/close are the first/last prices, high/low the extrema, and volume the sum
within each group. Output timestamps use the **supplied session date** at
midnight UTC, not the UTC date of session open. Schedule and completeness
policy are fingerprinted: different schedules are distinct processing recipes.
Select their pipeline IDs explicitly; automatic merging of such variants is
not performed.

Processed metadata now describes the output's timeframe and actual timestamp
bounds (exclusive end = last timestamp + 1 millisecond), not the original
fetch request. Original request metadata remains accessible through `parent_ids`.
All contributing raw IDs are retained, and replacing/compacting any input
invalidates the derived revision without deleting history. Existing catalog
entries and files are not rewritten; their original metadata remains readable.
The low-level `write_processed()` accepts `output_contract` for callers supplying
already-processed frames; those callers are responsible for transformation
validation. Omitting it declares that the parent's contract is unchanged.

## Storage correctness and explicit changes

Ordinary writes reject duplicate bars within a frame and overlapping actual
first/last timestamp coverage within the same provider/symbol/timeframe/layer/
processor identity. An overlap anywhere in that inclusive coverage is rejected,
even if exact row keys differ. Non-overlapping batches can be appended. No
implicit merge, upsert, deduplication or overwrite is performed. The explicit
`fetch_many(..., overlap_policy="refresh_overlap")` policy can refresh and extend
an overlapping group as one new revision after validating that every old
timestamp is retained. The new file is staged and verified before the catalog
atomically activates it and retires its inputs; the old files remain as history.
Fully covered requests are skipped first. The separate
`overlap_policy="fetch_missing"` policy fetches uncovered ranges, then locally
merges them with overlapping or adjacent raw datasets while preserving existing
values. It uses the same atomic publication and historical revision protections.
Neither policy changes ordinary write protection or the exact-range replacement
command below.

To refresh an existing entire raw batch, inspect it and then repeat its exact
ID as confirmation:

```bash
./venv/bin/python -m data_pipeline inspect RAW_ID
./venv/bin/python -m data_pipeline replace RAW_ID --confirm RAW_ID
```

Replacement fetches the original request and requires the same first/last
bar bounds. It creates a new revision, retains the old file, and marks old
processed derivatives inactive. Historical data remains readable by ID, and
`list --include-history` reveals it. Partial replacement is intentionally
rejected. Python callers can use `store.replace_raw(id, frame, confirm=id)`.

`compact ID1 ID2 --confirm ID1 ID2` combines active raw batches with the same
identity into one file. It retains old revisions and invalidates their derived
outputs; rerun processing on the compacted raw ID. Compaction reduces files
read by active queries, but retaining history uses additional disk space.

### Delete local data and reclaim storage

For multiple tickers, use **Delete data → Deletion mode → Batch tickers**.
Select a layer, provider, and tickers, then preview and confirm. This removes all
stored dates, timeframes, pipelines, and historical revisions for that selection.
The batch report lists each operation's outcome. A failure stops the remaining
operations; completed deletions are not rolled back. Pending physical cleanup
uses the same recovery controls as single-series deletion.

Open **Delete data** in the dashboard. Select raw or processed, provider, ticker,
timeframe and a UTC start/end; processed data also requires a specific pipeline.
Preview the affected revisions, review the row counts, then check the confirmation
box and click **Delete permanently**. The interval is `[start, end)`: the start
is included and the end is excluded. For daily data, timestamps are UTC session
date labels as described above.

The Python API provides the same preview and confirmation flow:

```python
from datetime import UTC, datetime
from data_pipeline import DataPipeline, DataQuery

pipeline = DataPipeline("data")
plan = pipeline.plan_delete(DataQuery(
    layer="raw", provider="yahoo", symbol="AAPL", timeframe="1h",
    start=datetime(2025, 1, 1, tzinfo=UTC),
    end=datetime(2025, 2, 1, tzinfo=UTC),
))
print(plan.to_frame())
# Execute only after reviewing the selection above.
report = pipeline.delete(plan, confirm=plan.operation_id)
print(report.to_dict())
```

Deletion covers matching rows in **all cataloged revisions**, including inactive
history. It physically removes the affected original Parquet files and any known
copies of those revisions in staging/quarantine. Partial files are rewritten as
separate before/after batches, with the surviving values and precision unchanged.
No original-file backups are retained. Counts include stored rows across revisions,
so several revisions of a bar count several times. Reported bytes are file sizes,
not a measurement of filesystem free space; small partial deletions can increase
storage because each surviving file has its own overhead. Rewriting also requires
temporary free space. Deleting entire batches avoids rewriting their contents.

Raw and processed layers are independent: deleting either leaves the other's
files, active flags and contents unchanged. Other processed pipeline variants are
also untouched. Rewritten batches receive new IDs; their old IDs are recorded in
catalog tombstones but can no longer be read. Processed provenance and saved runs retain
their original source IDs, so deleting those sources prevents reproducing those
runs from this store. To regenerate a whole processed output from the same raw
parents, first delete that whole output: partial deletion retains the existing
same-parent/pipeline duplicate protection.

Previews are tied to a data root and its selected revisions. Changes to that
selection require a fresh preview. After the catalog transaction commits, physical
cleanup must finish before an operation reports `completed`. If cleanup fails,
the dashboard displays a pending operation with a retry action; Python callers
can inspect `pipeline.list_deletions(pending_only=True)` and call
`pipeline.recover()`. Recovery also removes unfinished deletion rewrites from
before a catalog commit; those preparations leave the original data unchanged.
New deletions are blocked until pending cleanup finishes.
The catalog automatically migrates from version 1 to
version 2 to store deletion journals and tombstones.

This deletes files owned by the store and identified by the catalog/journals; it
does not erase external backups, saved-run results, or unrelated orphan files
already in quarantine. It performs normal filesystem deletion, not secure wiping.

Writers stage, validate the Parquet round trip, flush the file, publish with an
exclusive hard link, and commit the DuckDB transaction. The catalog is the
visibility boundary. A POSIX file lock serializes readers of the catalog and
writers across processes. Reads finish loading under that lock, so a concurrent
deletion cannot remove their input midway. `scan()` still returns a `LazyFrame`,
but loads its filtered/projected rows into memory before returning; the snapshot
remains usable after deletion. Select narrow ranges and columns for large stores.
This implementation targets local Linux/macOS filesystems with
hard-link/flock/fsync support; network shares and Windows are
not supported. Do not modify the catalog or files outside the package while it
is running. Back up the entire data root, including catalog and history.

Ordinary failures roll back changes. An abrupt process termination can leave
uncataloged files from ordinary writes; `recover` moves them into quarantine.
It first cleans up deletion preparations and finishes committed deletions, so
deleted originals are removed rather than quarantined.
`audit` verifies all surviving active and historical data. Missing/corrupt cataloged files
raise errors; recovery does not fabricate their contents.
If data files remain but the catalog is missing, all store operations stop with
an integrity error: restore the catalog from a consistent backup. `recover`
requires a valid catalog to distinguish abandoned writes from committed data.
It does not rebuild a lost catalog or silently initialize missing catalog tables.

```bash
./venv/bin/python -m data_pipeline audit
./venv/bin/python -m data_pipeline recover
```

Reads verify SHA-256 and schema. Single-dataset reads/audits also validate row
counts, bounds and OHLCV values. Range queries prune catalog entries and
push filters/projections into Parquet before taking an in-memory snapshot;
checksum verification still reads each selected file once.
`coverage` reports actual bounds, not proof of complete
exchange-session coverage. Querying mixed providers/timeframes/pipelines is
rejected rather than returning ambiguous duplicate-looking bars.

## Adding another provider

Implement `BaseDataProvider.fetch(DataRequest) -> pl.DataFrame` and register
an instance or zero-argument factory:

```python
from data_pipeline.providers import ProviderRegistry, YFinanceProvider

providers = ProviderRegistry({"yahoo": YFinanceProvider})
# providers.register("bloomberg", YourBloombergProvider)
pipeline = DataPipeline("data", providers=providers)
```

The storage, processors and query API remain unchanged. Bloomberg itself is not
implemented and would require the vendor SDK/credentials and its own adapter.
The data package supports OHLCV ingestion, transformations and explicit-session
hourly-to-daily resampling. Research preflight adds selected exchange calendars;
it does not discover an instrument's exchange or currency automatically.

Implementation references: [Yahoo download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html),
[DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency),
[DuckDB transactions](https://duckdb.org/docs/current/sql/statements/transactions),
[Polars Parquet scans](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html).
