# Quant Trading

This repository contains the foundations of a local market-data ingestion and
storage pipeline for quantitative trading research.

## Development setup

The committed `requirements.txt` file contains runtime dependencies. Install
test-only tooling separately:

```bash
./venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

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
