"""Deterministic split-and-cash-dividend adjustment of raw one-minute bars."""

from dataclasses import dataclass, replace
import json
from typing import ClassVar
import warnings

import polars as pl

from ....corporate_actions.models import ActionSnapshot, SUPPORTED_TYPES
from ....exceptions import ProcessingError
from ....schemas import CANONICAL_OHLCV_COLUMNS
from ...contracts import DataContract
from ..base import BaseProcessor
from .factors import attach_factors, event_factors
from .validation import relevant_events, validate_symbol


class PriceAdjustmentWarning(UserWarning):
    """A reported corporate action is outside the supported adjustment scope."""


@dataclass(frozen=True, slots=True)
class AdjustCorporateActions(BaseProcessor):
    """Backward-adjust US-equity 1m OHLCV to one pinned reference date.

    Supported actions are splits and regular/special USD cash dividends. Other
    actions remain unadjusted and are reported. The result is retrospective
    analytical data, without a causal backtest eligibility guarantee.
    """

    snapshot: ActionSnapshot

    name: ClassVar[str] = "adjust_corporate_actions"
    version: ClassVar[str] = "1"
    label: ClassVar[str] = "Adjust prices: splits and cash dividends"
    description: ClassVar[str] = (
        "Backward-adjust raw US-equity minute prices and split share volumes using a saved "
        "corporate-action snapshot. Other events are reported, not adjusted."
    )
    causal: ClassVar[bool] = False
    price_basis: ClassVar[str] = "transformed"
    config_schema: ClassVar[dict] = {"properties": {"snapshot": {
        "title": "Corporate-action snapshot", "format": "corporate_action_snapshot",
        "description": "Verified event snapshot with complete coverage and source price/volume basis for every ticker.",
    }}}

    def __post_init__(self):
        if not isinstance(self.snapshot, ActionSnapshot):
            raise ProcessingError("AdjustCorporateActions requires a validated ActionSnapshot.")
        if self.snapshot.methodology_version != "1":
            raise ProcessingError("Unsupported corporate-action methodology version; expected '1'.")

    @property
    def config(self):
        # Snapshot serialization owns backwards-compatible omission of optional
        # evidence fields, preserving saved pipeline identities when reloaded.
        return {"snapshot": self.snapshot.to_dict()}

    def output_contract(self, input_contract: DataContract) -> DataContract:
        if input_contract.timeframe != "1m":
            raise ProcessingError("Price adjustment requires a 1m input contract.")
        if input_contract.adjustment_json:
            raise ProcessingError("Input already has adjustment metadata; double adjustment is prohibited.")
        if input_contract.source_basis == "vendor_adjusted":
            raise ProcessingError("Vendor-adjusted source data cannot be adjusted again.")
        metadata = {
            "basis": "split_and_cash_dividend_adjusted", "reference_date": self.snapshot.reference_date.isoformat(),
            "snapshot_checksum": self.snapshot.checksum, "source": self.snapshot.source,
            "methodology": self.snapshot.methodology_version,
            "unsupported_events": "reported_not_adjusted",
        }
        return replace(input_contract, adjustment_json=json.dumps(metadata, sort_keys=True, separators=(",", ":")))

    def validate_backtest_calendar(self, bars, *, input_contract, output_contract):
        """Adjustment changes units, preserving bar labels and session timing.

        This validates timing only. The processor remains retrospective and
        research must separately opt into its noncausal price adjustments.
        """
        if self.output_contract(input_contract) != output_contract:
            raise ProcessingError("Price adjustment must preserve bar timing and its pinned adjustment metadata.")

    def _validate_input(self, frame, input_contract):
        if frame.select((pl.col("timestamp").dt.epoch("ms") % 60_000 != 0).any()).item():
            raise ProcessingError("Price adjustment requires bars on the whole-minute grid.")
        for (symbol,), group in frame.partition_by("symbol", as_dict=True).items():
            events = validate_symbol(self.snapshot, symbol, group["timestamp"][0],
                                     group["timestamp"][-1], input_contract.source_basis)
            event_factors(events)

    def _transform(self, frame, input_contract, output_contract):
        outputs = []
        for (symbol,), group in frame.partition_by("symbol", as_dict=True).items():
            events = relevant_events(self.snapshot, symbol, group["timestamp"][0])
            unsupported = [event for event in events if event.event_type not in SUPPORTED_TYPES]
            if unsupported:
                warnings.warn(
                    f"{symbol}: {len(unsupported)} other corporate actions reported but not adjusted: "
                    + ", ".join(f"{event.event_type} ({event.event_id})" for event in unsupported),
                    PriceAdjustmentWarning, stacklevel=2,
                )
            factored = attach_factors(group, event_factors(events))
            outputs.append(factored.with_columns(
                *[(pl.col(column) * pl.col("_price_factor")).alias(column)
                  for column in ("open", "high", "low", "close")],
                (pl.col("volume") * pl.col("_volume_factor")).alias("volume"),
            ).select(CANONICAL_OHLCV_COLUMNS))
        return pl.concat(outputs).sort("symbol", "timestamp")

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        if not input_frame.select("symbol", "timestamp").equals(output_frame.select("symbol", "timestamp")):
            raise ProcessingError("Price adjustment changed input bar keys.")
        # Independently form expected factors with timestamp predicates rather
        # than trusting the as-of schedule join used by the transformation.
        for (symbol,), group in input_frame.partition_by("symbol", as_dict=True).items():
            events = event_factors(relevant_events(self.snapshot, symbol, group["timestamp"][0]))
            price, volume = pl.lit(1.), pl.lit(1.)
            for stamp, price_factor, volume_factor in events:
                price = price * pl.when(pl.col("timestamp") < stamp).then(price_factor).otherwise(1.)
                volume = volume * pl.when(pl.col("timestamp") < stamp).then(volume_factor).otherwise(1.)
            actual = output_frame.filter(pl.col("symbol") == symbol)
            expected_factors = group.select(price.alias("price"), volume.alias("volume"))
            for column in ("open", "high", "low", "close", "volume"):
                expected = group[column] * expected_factors["volume" if column == "volume" else "price"]
                if not ((actual[column] - expected).abs() <= expected.abs() * 1e-12 + 1e-12).all():
                    raise ProcessingError(f"Adjusted {column} does not match the expected event factors.")
