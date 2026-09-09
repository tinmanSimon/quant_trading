# Quant Trading

This repository contains the foundations of a local market-data ingestion and
storage pipeline for quantitative trading research.

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
`data/` directory. Future live vendor checks will be explicitly marked
`network` and run only when requested:

```bash
./venv/bin/python -m pytest -m network
```

## Current data contract

The first supported dataset is canonical OHLCV. A `DataRequest` has
timezone-aware UTC bounds, with an inclusive `start` and exclusive `end`.
OHLCV timestamps identify the start of a bar and must be UTC. Validated frames
use ordered `timestamp`, `symbol`, `open`, `high`, `low`, `close`, and
`volume` columns. Each frame represents one timeframe, and its rows are
ordered by symbol and timestamp.
