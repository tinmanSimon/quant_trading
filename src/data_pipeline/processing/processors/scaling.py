"""Price-scaling processors that preserve the canonical OHLCV schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import polars as pl

from ...exceptions import ProcessingError
from .base import BaseProcessor
from ..validators import PreserveKeys, PreserveVolume


@dataclass(frozen=True, slots=True)
class ScalePrices(BaseProcessor):
    """Multiply OHLC prices by a positive finite factor, leaving volume intact."""

    factor: float
    name: ClassVar[str] = "scale_prices"
    version: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        try:
            if type(self.factor) not in (int, float):
                raise TypeError("factor must be a number, excluding booleans.")
            factor = float(self.factor)
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError("factor must be positive and finite.")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProcessingError(
                "Processor 'scale_prices' version '1' requires a positive finite factor."
            ) from exc
        object.__setattr__(self, "factor", factor)

    @property
    def config(self) -> dict[str, float]:
        return {"factor": self.factor}

    def _validate_input(self, frame, input_contract) -> None:
        # The shared wrapper validates OHLCV; any supported timeframe is valid.
        pass

    def _transform(self, frame: pl.DataFrame, input_contract, output_contract) -> pl.DataFrame:
        return frame.with_columns(
            *[pl.col(column) * self.factor for column in ("open", "high", "low", "close")]
        )

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract) -> None:
        PreserveKeys()(input_frame, output_frame)
        PreserveVolume()(input_frame, output_frame)
        for column in ("open", "high", "low", "close"):
            if not (input_frame[column] * self.factor).equals(output_frame[column]):
                raise ValueError(f"Scaled {column} does not match the configured factor.")
