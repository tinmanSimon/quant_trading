"""Data-only public indicator results and calculation version identity."""

from dataclasses import dataclass
from typing import Any

import polars as pl
import talib


BACKEND_VERSION = talib.__version__
ALGORITHM_VERSION = f"1.0.0-talib-{BACKEND_VERSION}"


@dataclass(frozen=True)
class IndicatorResult:
    """Aligned values and JSON-friendly calculation provenance.

    ``source_index`` is the zero-based row position in the supplied input. The
    timestamp retains its input Date/Datetime type and timezone. Value columns
    are Float64, with nulls during initialization. Metadata includes theoretical
    warmup lengths even when too few observations were supplied to finish them.
    """

    frame: pl.DataFrame
    metadata: dict[str, Any]
