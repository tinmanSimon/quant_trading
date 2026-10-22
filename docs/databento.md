# Databento historical data

Databento is available from **Fetch → Provider → databento**. Downloads use the
official Python SDK and the same validated OHLCV storage as the other providers.
No API request is made just by opening the page. Native `1m` and `1h` bars are
supported; prices remain unadjusted and extended-hours bars remain in raw data.

## Setup

Install the project requirements into your virtual environment, then set the
key in the environment of the dashboard process:

```bash
venv/bin/python -m pip install -r requirements.txt
export DATABENTO_API_KEY="your-api-key"
venv/bin/streamlit run src/dashboard/app.py
```

Do not put the key in ticker input, saved
recipes, source code, or request reports. The application reads the environment;
it does not automatically load a `.env` file.

## Dashboard workflow

1. Select **databento**, click **Load available datasets**, and choose a dataset.
   The list is a catalog, not a guarantee of your account's entitlements. Choose
   the venue coverage and history appropriate to your analysis. No dataset is
   selected automatically.
2. Choose `1m` or `1h`, paste tickers separated by spaces, commas, or newlines,
   and set the UTC date/time bounds. The end is exclusive.
3. Choose an existing-data policy. **Fetch missing ranges; merge locally** is
   the default and avoids purchasing ranges already covered locally. It does
   not audit or repair holes inside a stored range. Refreshing overlap buys
   the combined overlapping range again.
4. Click **Estimate download**. This uses metadata and symbology APIs, not a
   billable market-data download. Review the estimated dollar cost, warnings,
   request groups, and ticker errors. The estimate is before account credits.
   A zero maximum estimated cost in the UI disables the estimate cap. The cap
   is not a guaranteed billing limit imposed by Databento.
5. Click **Fetch and save** to authorize the reviewed plan. The app groups
   matching ranges into requests of up to 2,000 symbols, submits remote batch
   jobs, downloads their files, and saves each valid ticker separately. Changing
   the inputs requires a new estimate before submission.
6. Leave **Automatically check and continue this download** enabled while the
   page is open, or use **Check progress / resume**. Returning to **Saved
   Databento downloads** after a restart resumes the saved job IDs. The remote
   job can finish while the dashboard is closed; local downloading/importing
   resumes when the page is open again.

File transfers reuse a single manifest and download up to four files at once,
reusing HTTP connections. The progress bar shows bytes transferred and files
verified. Complete cached files are checksum-verified and skipped; interrupted
files resume from their saved byte offset. The status table distinguishes
Databento's server preparation from local downloading, importing, and saving.

Status checks and downloads reuse existing jobs. If a submission times out
before its job ID is known, the app marks it for reconciliation instead of
blindly submitting another paid request. Find the job in Databento's Download
center and use **Reconnect an existing Databento job**. The app checks that its
dataset, symbols, schema, and dates match the planned request.

Reports distinguish saved, merged, refreshed, skipped, empty, and failed tickers.
A ticker is published only after all its requested groups have been decoded.
Invalid ticker data does not replace existing data. Successful tickers can
still be saved when another ticker fails. Dataset-condition warnings are broad
vendor notices; an `available` day does not prove every minute is present.

## Local storage and resume files

Feeds use distinct provider identities such as `databento-xnas.itch` and
`databento-equs.mini`. Market data, processing, and backtesting use these same
identities. Data from different feeds is never merged as one series.

Raw Parquet data follows the existing data-root layout. Plans and downloaded
files are under `<data_dir>/metadata/databento/`; they contain no API keys.
Keep this directory to resume downloads and imports. It retains downloaded DBN
and staging files, so disk usage exceeds the final canonical Parquet size.

Failed ticker imports can also be retried entirely offline from their decoded
cache, without an API key or another purchase:

```python
from data_pipeline import DataPipeline

pipeline = DataPipeline("data")
plan = pipeline.recover_databento(
    saved_plan_id,
    drop_invalid_outside_regular=True,
)
```

This explicit option removes null/nonfinite OHLC bars only when their entire
interval falls outside XNYS regular sessions, including holidays and early
closes. Valid extended-hours bars remain. Invalid regular-session prices and
other malformed data still fail validation. Omitted timestamps are recorded in
dataset quality metadata. Successful tickers and original cached files remain
unchanged; the saved plan's failed outcomes are replaced with recovery results.
Missing caches or changed local coverage stop recovery without fetching data.
Calling recovery again resumes interrupted publication or leaves already saved
tickers alone. Ordinary **Check progress / resume** retains its existing behavior.

## Make regular-session 30-minute bars

Download `1m` raw data, then open **Process data**:

1. Select the Databento feed, `1m` source bars, and the tickers/batches to process.
2. Add **Regular hours: 1m to 30m**.
3. Choose the first and last session dates, spanning the input range, and click
   **Generate session schedule**. The fixed XNYS calendar supplies regular hours.
4. Preview, then **Process and save**.

Only regular-session minutes are aggregated. Normal days produce 13 bars:
09:30–10:00, 10:00–10:30, ..., 15:30–16:00 New York time. A 13:00 early close
produces seven bars, ending with 12:30–13:00. Timestamps label the start of each
bar and are stored in UTC. DST, holidays, and early closes follow the exchange
calendar; edited schedules must still match actual regular hours.

Pre-market and after-hours bars are excluded and reported. Each output bar uses
the first open, maximum high, minimum low, last close, and summed volume. Missing
regular-session minutes cause an error by default; selecting `drop` omits the
entire affected 30-minute bar with a warning. It never synthesizes prices, and
dropped intervals can still appear as missing bars in backtesting.

Choose **Incomplete intervals → aggregate_available** to retain bars containing
some missing minutes. This uses the first available open, maximum observed high,
minimum observed low, last available close, and summed observed volume. Warnings
report the affected interval count and missing minutes in preview and saved batch
results. The processor does not infer that gaps are trading halts, and never fills
prices or volume. Wholly empty intervals remain absent. If the opening minute is
missing, the bar's open is the first later observed price, not a verified price at
the interval start. The `raise` default and existing `drop` behavior are unchanged.

For other schedules or `1h` output, the existing **Resample intraday (1m to 30m / 1h)**
processor remains available. Native downloaded hourly bars are not shifted or
relabeled to 09:30.

Raw data is retained, and processed datasets record their source revisions and
the exact session schedule. Select the processed layer in analysis/backtesting.

The same processor is available in Python for existing raw datasets:

```python
from datetime import date
from data_pipeline.processing import ResampleRegularHours30m

processor = ResampleRegularHours30m.for_dates(date(2024, 1, 2), date(2024, 1, 31))
processed = pipeline.process(raw_dataset_id, processors=[processor])
```

## Python usage

```python
from datetime import UTC, datetime
from data_pipeline import DataPipeline

pipeline = DataPipeline("data")
plan = pipeline.estimate_databento(
    ["AAPL", "MSFT"],
    dataset="XNAS.ITCH",  # Choose the intended feed; this is an example.
    timeframe="1m",
    start=datetime(2025, 1, 2, tzinfo=UTC),
    end=datetime(2025, 1, 4, tzinfo=UTC),
    overlap_policy="fetch_missing",
    max_estimated_cost=5.0,
)
print(plan["estimated_cost"], plan["warnings"])

# Execute only after reviewing the estimate. This authorizes all its groups.
state = pipeline.advance_databento(plan["id"], submit=True)
# Call again later to advance a stage, respecting vendor rate limits.
state = pipeline.advance_databento(plan["id"])
report = pipeline.databento_report(plan["id"])
print(report.to_frame())
```

`pipeline.databento_plans()` lists persisted plans and
`pipeline.databento_plan(plan_id)` reads one. These reads make no vendor calls.
`advance_databento` does not wait for remote jobs in a polling loop. Repeated
calls advance through submission, status checks, downloading, decoding, and
local publication. A single download/decode stage can take time on a large job.

For a small direct download, register a dataset-specific `DatabentoProvider`
or resolve its namespaced name through the default registry. Direct `ingest`
and ordinary `fetch_many` calls do not provide the cost-review/resume workflow;
use the Databento plan methods for bulk downloads.

## References and verification

- [Batch requests](https://databento.com/docs/api-historical/batch/batch-submit-job?lang=python)
- [Cost estimates](https://databento.com/docs/api-historical/metadata/metadata-get-cost?lang=python)
- [Credits and billing](https://databento.com/docs/faqs/usage-pricing-and-data-credits)
- [OHLCV conventions](https://databento.com/docs/schemas/ohlcv)

The integration tests use fake SDK responses, temporary storage, and local DBN
fixtures. They do not download market data or consume account credits.
