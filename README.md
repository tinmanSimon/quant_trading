# Quant Trading

An OHLCV ingestion package with interchangeable providers, local raw/processed
Parquet storage, ordered processors, revision history, and a Python/CLI query API.
Yahoo Finance is the included vendor adapter.

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
`data/` directory. Live vendor checks are marked `network` and skipped unless
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
Yahoo symbols are canonicalized to uppercase, so `aapl` and `AAPL` share one
storage identity and cannot bypass overlap protection. Other providers retain
case-sensitive instrument identifiers. Local symbol queries apply Yahoo's
case normalization even when the provider filter is omitted.

## Structure

```text
src/data_pipeline/
├── api.py                 # DataPipeline orchestrates the complete workflow
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

Writers stage, validate the Parquet round trip, flush the file, publish with an
exclusive hard link, and commit the DuckDB transaction. The catalog is the
visibility boundary. A POSIX file lock serializes readers of the catalog and
writers across processes. File contents are immutable, so lazy query snapshots
remain valid across replacement. This implementation targets local Linux/macOS
filesystems with hard-link/flock/fsync support; network shares and Windows are
not supported. Do not modify the catalog or files outside the package while it
is running. Back up the entire data root, including catalog and history.

Ordinary failures roll back changes. An abrupt process termination can leave
uncataloged files; `recover` moves them into quarantine without deleting them.
`audit` verifies all active and historical data. Missing/corrupt cataloged files
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
counts, bounds and OHLCV values. Lazy range queries prune catalog entries and
push filters/projections into Parquet; checksum verification still reads each
selected file once. `coverage` reports actual bounds, not proof of complete
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
This version supports OHLCV ingestion, OHLCV transformations and explicit-session
hourly-to-daily resampling. Additional column schemas and automatic exchange
calendar integration are not implemented.

Implementation references: [Yahoo download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html),
[DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency),
[DuckDB transactions](https://duckdb.org/docs/current/sql/statements/transactions),
[Polars Parquet scans](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html).
