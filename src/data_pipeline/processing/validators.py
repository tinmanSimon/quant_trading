"""Reusable transformation-specific checks; processors choose their own rules."""

import polars as pl


class SubsetKeys:
    def __call__(self, original: pl.DataFrame, output: pl.DataFrame) -> None:
        if output.select("symbol", "timestamp").join(
            original.select("symbol", "timestamp"), on=["symbol", "timestamp"], how="anti"
        ).height:
            raise ValueError("Processor introduced new (symbol, timestamp) keys.")


class PreserveKeys:
    def __call__(self, original: pl.DataFrame, output: pl.DataFrame) -> None:
        SubsetKeys()(original, output)
        if original.height != output.height:
            raise ValueError("Processor must preserve every input bar key.")


class PreserveVolume:
    def __call__(self, original: pl.DataFrame, output: pl.DataFrame) -> None:
        if not original.select("symbol", "timestamp", "volume").equals(
            output.select("symbol", "timestamp", "volume")
        ):
            raise ValueError("Processor must preserve volume for every input bar.")
