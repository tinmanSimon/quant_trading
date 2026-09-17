"""Path construction keeps vendor identifiers out of filesystem syntax."""

from pathlib import Path
from urllib.parse import quote

from ..models import DataRequest


def relative_path(request: DataRequest, layer: str, dataset_id: str, year: int) -> Path:
    parts = [layer]
    for name in ("dataset", "provider", "symbol", "timeframe"):
        parts.append(f"{name}={quote(getattr(request, name), safe='')}")
    # Year is the first bar's year. Catalog bounds handle batches crossing years.
    return Path(*parts) / f"year={year}" / f"{dataset_id}.parquet"


def contained_path(root: Path, relative: str | Path) -> Path:
    from ..exceptions import DataIntegrityError

    relative = Path(relative)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise DataIntegrityError(f"Path escapes data root: {relative}")
    return resolved
