from datetime import UTC, datetime
from pathlib import Path

from data_pipeline import DataPipeline, DataRequest


def main():
    project_root = Path(__file__).resolve().parents[1]
    pipeline = DataPipeline(project_root / "data")

    request = DataRequest(
        symbol="AAPL",
        provider="yahoo",
        timeframe="1d",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 2, 1, tzinfo=UTC),
    )

    result = pipeline.ingest(request)

    print(f"Saved {result.raw.row_count} rows")
    print(f"Dataset ID: {result.raw.dataset_id}")
    print(f"File: {pipeline.store.data_dir / result.raw.relative_path}")


if __name__ == "__main__":
    main()
