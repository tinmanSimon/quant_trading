from datetime import UTC, datetime
from pathlib import Path

from data_pipeline import DataPipeline, DataQuery


def main():
    project_root = Path(__file__).resolve().parents[1]
    pipeline = DataPipeline(project_root / "data")

    query = DataQuery(
        layer="raw",
        provider="yahoo",
        symbol="AAPL",
        timeframe="1d",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 2, 1, tzinfo=UTC),
    )

    frame = pipeline.read(query)

    print(f"Loaded {frame.height} rows")
    print(frame)


if __name__ == "__main__":
    main()
