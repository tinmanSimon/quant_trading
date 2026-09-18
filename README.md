# Quant Trading

An OHLCV ingestion package with interchangeable providers, local raw/processed
Parquet storage, ordered processors, revision history, and a Python/CLI query API.
Yahoo Finance is the included vendor adapter.

The `data_pipeline` package also handles batch fetching and per-ticker outcomes.
The `research` package adds strict local-data preflight,
independent-account backtests, versioned strategies and saved comparisons.
A local Streamlit dashboard provides dataset navigation and interactive charts.

## Research quick start

Activate the environment once per terminal, then use the short commands:

```bash
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
dataset revision; missing volume is never silently changed to zero. Existing
overlapping data raises an error instead of being overwritten. Fetch requests
still obey Yahoo's availability/retention limits.

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
example historical daily ranges; choose recent ranges for Yahoo hourly data.
`python -m research` also works in place of `quant-research`.

Launch the dashboard from the repository root:

```bash
python -m streamlit run src/dashboard/app.py
```

Open the local URL printed by Streamlit. The sidebar selects **Market data**,
**Fetch**, **Backtest**, or **Saved runs**, and the data/results roots. Dataset
charts have candlesticks, volume, drag/scroll zoom, a range slider and reset
controls. They show original bars without resampling or gap filling. Intraday
display timezones are selectable; daily timestamps remain session-date labels.
Recorded omissions and unknown historical provenance are displayed explicitly.
Large ranges can be slow to render; narrow the visible date range as needed.

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

### Data checks and simulation rules

Before any strategy is evaluated, every ticker must have every required bar,
including warm-up, on the selected exchange schedule. Preflight checks actual
timestamps, holidays, DST, early closes, session alignment, completed bars,
file checksums and canonical values. Missing, unexpected, corrupt or unfinished
bars abort the entire request with ticker-specific details. Bounds in the
catalog alone do not establish completeness. Backtests never fetch or fabricate
missing data. The initial supported inputs are raw `1h` and `1d` bars for USD
instruments, using the explicitly selected calendar (default `XNYS`, U.S.
regular equity sessions). The calendar is a user-selected assumption, not
automatic exchange discovery; choose the appropriate calendar for the symbol.
Hourly calendars with lunch breaks are rejected.

A skipped bar blocks a test when its absence affects the test or warm-up window.
Omissions outside that window do not block it. Legacy datasets have `unknown`
fetch provenance, which remains visible; they can pass only after their actual
bars pass coverage checks. A quality report records what ingestion observed,
and cannot prove the provider supplied every real trade or correct prices.
Replacements receive fresh reports; processing/compaction preserve lineage and
combined reports. Old recorded omissions do not override a bar now present.

Each strategy/ticker pair starts with its own cash account. There is no shared
capital portfolio, leverage, short selling or fractional-share trading.
At each completed bar, the strategy returns a target allocation in `[0, 1]`.
That target is rebalanced at the next available scheduled bar's open using
that open's price plus configured adverse slippage. Fees and price gaps are
included in affordability calculations; buys are reduced or rejected when
needed. Rejections are recorded. Zero-volume bars cannot execute orders.
The exact cash ledger uses Decimal arithmetic, fees round upward to eight
decimal places, and cash/holdings/equity must stay nonnegative. Float columns
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
│   ├── strategies/      # Versioned strategies and weighted combinations
│   ├── backtesting/     # Causal execution, cash accounting and metrics
│   ├── runs.py           # Atomic saved runs and compatible comparisons
│   └── cli.py            # quant-research command
└── dashboard/            # Streamlit navigation and Plotly charts
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

Tests use per-test temporary directories and deterministic fixtures under
`tests/fixtures/`; they must not read from or write to the repository's
`data/` directory. Private-strategy discovery is redirected to temporary test
locations so tests never execute your personal implementations. Live vendor checks are marked `network` and skipped unless
explicitly enabled:

```bash
./venv/bin/python -m pytest --run-network -m network tests/test_yahoo_live.py
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

## Structure

```text
src/data_pipeline/
├── api.py                 # DataPipeline orchestrates the complete workflow
├── batch_fetch.py         # Independent ticker fetches and structured outcomes
├── cli.py, __main__.py     # Command-line entry points
├── models.py              # DataRequest and DataQuery
├── exceptions.py          # Specific validation/provider/storage errors
├── schemas/ohlcv.py        # Canonical frame contract and validation
├── providers/             # Common fetch interface, registry, Yahoo adapter
├── processing/            # Shared execution, contracts, validation and registry
│   └── processors/        # Shared base.py, plus scaling.py and resampling.py
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
factories with `ProcessorRegistry`. The CLI's built-in registry contains
`scale_prices` and `resample_ohlcv`, both version `1`.

### Implementing a processor

Processors live in `processing/processors/`: `base.py` contains `BaseProcessor`,
`scaling.py` contains `ScalePrices`, and `resampling.py` contains `ResampleOHLCV`
and its session helpers. Both `data_pipeline.processing` and
`data_pipeline.processing.processors` expose these classes for public imports.
Shared execution and metadata modules such as `pipeline.py`, `contracts.py`
and `registry.py` remain directly under `processing/`.

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
implicit merge, upsert, deduplication or overwrite is performed.

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
